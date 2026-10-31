"""Retrieval failure or reasoning failure? Offline split of every line of finished arms (research log section 8.76).

For each line L, X(L) are the identifiers in L that the repository defines in another file (`eval.proxy_metrics`). The prompt
of an arm is rebuilt with the same code paths as the evaluation (CPU only), cropped the way the generator crops it, and the line
is placed in one of four classes together with the arm's logged result:

  self_contained  X(L) is empty: nothing to retrieve, so a wrong answer is a model failure (knowledge or reasoning);
  right           X(L) is not empty and the line is exact;
  retrieval       X(L) is not empty, the line is wrong, and some name of X(L) does not appear in the visible prompt;
  reasoning       X(L) is not empty, the line is wrong, and every name of X(L) appears in the visible prompt.

It is lexical (a name can be visible without its definition being useful), so it is a screen, not a proof. Unlike
`proxy_metrics` it is run over all datapoints, after the E4 results are final; nothing is tuned on it.

Example:
    python -m eval.failure_split --base ~/lca_remote/E4/codellama-7b/medium_context/arms \
        --arms e4_F_N1753:rag:1753 e4_pd_N1753:path_distance:1753 e4_F_N12000:rag:12000 e4_pd_N12000:path_distance:12000 \
        --out rag_results/reports/E4/failure_split.jsonl
"""
import argparse
import json
import os
from collections import Counter

from eval.proxy_metrics import definitions, identifiers, needed_names, repo_definitions, visible_prompt


def classify(needed: set, seen_identifiers: set, correct: bool) -> str:
    if not needed:
        return "right_self_contained" if correct else "wrong_self_contained"
    if correct:
        return "right"
    return "reasoning" if needed <= seen_identifiers else "retrieval"


def build_rows(arm: str, kind: str, cap, datapoints, composer_args, encode, decode, max_seq_len, rag_factory, results):
    from eval.compare_arms import static_provider_factory

    rows = []
    for dp_idx, dp in enumerate(datapoints):
        path = next(iter(dp.get_completion()))
        defined = repo_definitions(dp.get_context(), exclude_path=path)
        provider = rag_factory(dp) if kind == "rag" else static_provider_factory(kind, composer_args, cap, encode.tokenizer)(dp)
        for line_class, line_nums in dp.completion_lines.items():
            for line_num in line_nums:
                line = dp.get_line(line_num)
                needed = needed_names(line, dp.get_prefix(line_num), defined)
                result = results.get((dp_idx, line_num, line))
                if result is None:
                    continue
                prompt = provider(dp, line_num)[0]
                seen = visible_prompt(prompt, encode, decode, max_seq_len)
                seen_ids = identifiers(seen)
                correct = result["prediction_line"].strip() == result["ground_truth"].strip()
                rows.append({
                    "arm": arm, "dp": dp_idx, "repo": result["repo"], "line_num": line_num, "line_class": line_class,
                    "n_needed": len(needed), "n_visible": len(needed & seen_ids),
                    "n_defined_visible": len(needed & definitions(seen)), "correct": correct,
                    "cls": classify(needed, seen_ids, correct),
                    "rebuilt_tokens": len(encode(seen)), "logged_tokens": result["metrics"]["input_tokens"],
                })
    return rows


def summarize(rows: list) -> str:
    out = ["| arm | lines | right | wrong, self-contained | wrong, retrieval miss | wrong, reasoning (names visible) |",
           "|---|---:|---:|---:|---:|---:|"]
    for arm in dict.fromkeys(r["arm"] for r in rows):
        c = Counter(r["cls"] for r in rows if r["arm"] == arm)
        n = sum(c.values())
        right = c["right"] + c["right_self_contained"]
        out.append(f"| {arm} | {n} | {right} ({right / n:.1%}) | {c['wrong_self_contained']} ({c['wrong_self_contained'] / n:.1%}) | "
                   f"{c['retrieval']} ({c['retrieval'] / n:.1%}) | {c['reasoning']} ({c['reasoning'] / n:.1%}) |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", required=True, help="name:kind:cap, kind = rag | composer name, cap = tokens or 0")
    parser.add_argument("--base", required=True, help="directory holding the arms' generation_results.jsonl")
    parser.add_argument("--dataset", default="medium")
    parser.add_argument("--rag", nargs="*", default=["prefix_cap=15900"], help="overrides for the rag settings (design F is rag=F)")
    parser.add_argument("--limit", type=int, help="only the first N datapoints (smoke test)")
    parser.add_argument("--out", help="write per-line rows as JSON lines to this file")
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir
    from transformers import AutoTokenizer

    from eval.preprocessors import load_datapoints
    from rag.provider import make_rag_provider_factory

    with initialize_config_dir(config_dir=os.path.join(os.path.dirname(__file__), "config"), version_base=None):
        cfg = compose(config_name="compare_arms", overrides=[f"dataset={args.dataset}", "params=codellama7b", "rag=F"])
    budget = cfg.params.inference_params["seq_max_len"] - 100
    tokenizer = AutoTokenizer.from_pretrained(cfg.params.preprocess_params["tokenizer"])

    def encode(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    encode.tokenizer = tokenizer
    decode = tokenizer.decode
    data = load_datapoints(cfg.dataset)
    if args.limit:
        data = data[:args.limit]
    composer_args = dict(cfg.composers_config)

    rows = []
    for spec in args.arms:
        arm, kind, cap = spec.split(":")
        cap = int(cap) or None
        rag_settings = dict(cfg.rag)
        for item in args.rag:
            key, value = item.split("=", 1)
            rag_settings[key] = json.loads(value)
        if cap:
            rag_settings["max_context_tokens"] = cap
        rag_factory = make_rag_provider_factory(
            lambda text: len(encode(text)), budget, extension=cfg.composers_config.extension,
            lang_sep=cfg.composers_config.lang_sep_symbol, meta_sep=cfg.composers_config.meta_info_sep_symbol, **rag_settings)
        results = {}
        with open(os.path.join(os.path.expanduser(args.base), arm, "generation_results.jsonl"), encoding="utf8") as f:
            for line in f:
                r = json.loads(line)
                results[(r["datapoint_idx"], r["line_num"], r["ground_truth"])] = r
        arm_rows = build_rows(arm, kind, cap, data, composer_args, encode, decode, budget, rag_factory, results)
        same = sum(r["rebuilt_tokens"] + 1 == r["logged_tokens"] for r in arm_rows)  # the generator adds the BOS token
        print(f"{arm}: {len(arm_rows)} lines, rebuilt prompt (+ BOS) has the logged token count for {same} ({same / len(arm_rows):.1%})", flush=True)
        rows += arm_rows
    print(summarize(rows))
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
