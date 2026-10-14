"""Offline retrieval-quality proxy (research log section 8.3). CPU only; touches development datapoints only.

For a target line L of the completion file, X(L) are the identifiers that occur in L, are defined in some other
file of the repository snapshot, and do not occur in the prefix before L (so the model cannot copy them from the file
itself). The proxy asks how many of them are visible to the model in the prompt of a given arm:

  appearance recall = share of X(L) that occurs as an identifier anywhere in the visible prompt;
  definition recall = share of X(L) whose definition (`def x`, `class x`, `x = ...`) occurs in the visible prompt.

It is lexical and says nothing about whether the model uses the context; it is a screen, not a result.

Example (arms are name:kind, kind in zero | rag | a composer name; add `--results` to join EM):
    python -m eval.proxy_metrics --arms r0_zero:zero r0_path_distance:path_distance r0_v1:rag --results
"""
import argparse
import importlib
import json
import os
from collections import defaultdict
from typing import Callable, Iterable, Optional

import numpy as np

# The helpers live in `rag.code_names` (the retrieval package must not depend on `eval`); re-exported here.
from rag.code_names import ASSIGNMENT, DEFINITION, IDENTIFIER, KEYWORDS, definitions, identifiers  # noqa: F401


def repo_definitions(files: dict[str, str], exclude_path: Optional[str] = None) -> set[str]:
    """Names defined in the Python files of a snapshot, excluding the completion file itself."""
    names: set[str] = set()
    for path, code in files.items():
        if path.endswith(".py") and path != exclude_path:
            names |= definitions(code)
    return names


def needed_names(line: str, prefix: str, defined: set[str], strict: bool = False) -> set[str]:
    """X(L): names used in the target line that the repo defines in another file.

    Loose (default, log section 8.13): minus the names the completion file's own prefix defines. A name that is merely
    imported or used in the prefix is still needed, because its signature and behaviour are only in the other file.
    Strict (the first definition): minus every identifier that occurs anywhere in the prefix.
    """
    names = identifiers(line) & defined
    return names - (identifiers(prefix) if strict else definitions(prefix))


def targeted_lines(datapoint) -> dict[str, list[int]]:
    """The datapoint's target lines with a non-empty loose X(L), grouped by category (same keys as completion_lines).

    Uses only the repository snapshot and the file prefix, never any model output.
    """
    path = next(iter(datapoint.get_completion()))
    defined = repo_definitions(datapoint.get_context(), exclude_path=path)
    return {line_class: [n for n in line_nums
                         if needed_names(datapoint.get_line(n), datapoint.get_prefix(n), defined)]
            for line_class, line_nums in datapoint.completion_lines.items()}


def recalls(needed: set[str], visible_prompt: str) -> tuple[float, float]:
    """(appearance recall, definition recall) of `needed` in `visible_prompt`; requires a non-empty `needed`."""
    return (len(needed & identifiers(visible_prompt)) / len(needed),
            len(needed & definitions(visible_prompt)) / len(needed))


def visible_prompt(prompt: str, encode: Callable[[str], list[int]], decode: Callable[[list[int]], str],
                   max_seq_len: int) -> str:
    """The part of `prompt` the generator keeps: same character crop, then the last `max_seq_len` tokens."""
    if len(prompt) > max_seq_len * 6:
        prompt = prompt[-max_seq_len * 6:]
    ids = encode(prompt)
    return decode(ids[-max_seq_len:]) if len(ids) > max_seq_len else prompt


def hit_miss_difference(rows: list[dict], n_boot: int = 10_000, seed: int = 0):
    """EM(hit) - EM(miss) over lines with a non-empty X(L), averaged over arms, bootstrapped over datapoints.

    `hit` means every needed name appears in the visible prompt. Each row needs: arm, dp, hit, em.
    Returns (difference, low, high, n_hit, n_miss), or None if either group is empty for every arm.
    """
    by_arm_dp: dict = defaultdict(lambda: defaultdict(lambda: [[], []]))  # arm -> dp -> [miss ems, hit ems]
    for r in rows:
        by_arm_dp[r["arm"]][r["dp"]][int(r["hit"])].append(r["em"])
    arms = sorted(by_arm_dp)
    dps = sorted({dp for a in arms for dp in by_arm_dp[a]})

    def estimate(sample: Iterable[int]) -> Optional[float]:
        diffs = []
        for arm in arms:
            miss, hit = [], []
            for dp in sample:
                if dp in by_arm_dp[arm]:
                    miss += by_arm_dp[arm][dp][0]
                    hit += by_arm_dp[arm][dp][1]
            if miss and hit:
                diffs.append(np.mean(hit) - np.mean(miss))
        return float(np.mean(diffs)) if diffs else None

    point = estimate(dps)
    if point is None:
        return None
    rng = np.random.default_rng(seed)
    boots = [b for b in (estimate(rng.choice(dps, size=len(dps))) for _ in range(n_boot)) if b is not None]
    low, high = np.percentile(boots, [2.5, 97.5])
    n_hit = sum(1 for r in rows if r["hit"])
    return point, float(low), float(high), n_hit, len(rows) - n_hit


# ---------------------------------------------------------------------------------------------------------------
# Reconstructing what each arm shows the model (same code paths as the evaluation), development datapoints only.

def prompts_for_arm(kind: str, datapoint, composer_args: dict, rag_factory: Optional[Callable] = None,
                    wanted: Optional[set[int]] = None):
    """Yield (line_class, k, line_num, prompt) in the order the generator visits the lines.

    `k` counts every line of the category, so it stays aligned with the generator's order; `wanted`, if given,
    limits which line numbers get a prompt built."""
    from composers.composer_registry import COMPOSERS

    static_context = None
    if kind not in ("zero", "rag"):
        spec = COMPOSERS[kind]
        composer = getattr(importlib.import_module(spec["module"]), spec["name"])(**composer_args)
        static_context = composer.context_composer(datapoint)
    provider = rag_factory(datapoint) if kind == "rag" else None
    for line_class, line_nums in datapoint.completion_lines.items():
        for k, line_num in enumerate(line_nums):
            if wanted is not None and line_num not in wanted:
                continue
            prefix = datapoint.get_prefix(line_num)
            if kind == "zero":
                prompt = prefix
            elif kind == "rag":
                prompt = provider(datapoint, line_num)[0]
            else:
                prompt = "\n".join([static_context, prefix])
            yield line_class, k, line_num, prompt


def proxy_rows(kind: str, arm: str, datapoints: list, composer_args: dict, encode, decode, max_seq_len: int,
               rag_factory: Optional[Callable] = None, strict: bool = False) -> list[dict]:
    rows = []
    for dp_idx, dp in enumerate(datapoints):
        path = next(iter(dp.get_completion()))
        defined = repo_definitions(dp.get_context(), exclude_path=path)
        needed_by_line = {n: needed_names(dp.get_line(n), dp.get_prefix(n), defined, strict)
                          for nums in dp.completion_lines.values() for n in nums}
        wanted = {n for n, names in needed_by_line.items() if names}
        for line_class, k, line_num, prompt in prompts_for_arm(kind, dp, composer_args, rag_factory, wanted):
            needed = needed_by_line[line_num]
            seen = visible_prompt(prompt, encode, decode, max_seq_len)
            app, dfn = recalls(needed, seen)
            rows.append({"arm": arm, "dp": dp_idx, "line_class": line_class, "k": k, "n_needed": len(needed),
                         "app": app, "def": dfn, "hit": app == 1.0})
    return rows


def summarize(rows: list[dict]) -> str:
    out = ["| arm | lines with X(L) | mean appearance recall | mean definition recall | lines with all names visible |",
           "|---|---:|---:|---:|---:|"]
    for arm in sorted({r["arm"] for r in rows}):
        sub = [r for r in rows if r["arm"] == arm]
        out.append(f"| {arm} | {len(sub)} | {np.mean([r['app'] for r in sub]):.3f} | "
                   f"{np.mean([r['def'] for r in sub]):.3f} | {np.mean([r['hit'] for r in sub]):.3f} |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", required=True, help="name:kind pairs, kind = zero | rag | composer name")
    parser.add_argument("--rag", nargs="*", default=[], help="overrides for the rag settings, e.g. prefix_cap=2000")
    parser.add_argument("--results", action="store_true", help="join EM from the arms' generation_results.jsonl")
    parser.add_argument("--base", default="~/lca_artifacts/python/starcoderbase-1b/small_context/arms")
    parser.add_argument("--dataset", default="small", help="small | medium | large | huge (development datapoints only)")
    parser.add_argument("--dev-size", type=int, default=20)
    parser.add_argument("--dev-extra", type=int, default=28)
    parser.add_argument("--out", help="write per-line rows as JSON lines to this file")
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir
    from transformers import AutoTokenizer

    from eval.analyze_arms import arm_lines
    from eval.line_generators import dev_indices
    from eval.preprocessors import load_datapoints
    from rag.provider import make_rag_provider_factory

    with initialize_config_dir(config_dir=os.path.join(os.path.dirname(__file__), "config"), version_base=None):
        cfg = compose(config_name="compare_arms", overrides=[f"dataset={args.dataset}"])
    budget = cfg.params.inference_params["seq_max_len"] - 100
    tokenizer = AutoTokenizer.from_pretrained(cfg.params.preprocess_params["tokenizer"])
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    decode = lambda ids: tokenizer.decode(ids)
    rag_settings = dict(cfg.rag)
    for item in args.rag:
        key, value = item.split("=", 1)
        rag_settings[key] = json.loads(value.lower() if value.lower() in ("true", "false") else value)
    rag_factory = make_rag_provider_factory(
        lambda text: len(encode(text)), budget, extension=cfg.composers_config.extension,
        lang_sep=cfg.composers_config.lang_sep_symbol, meta_sep=cfg.composers_config.meta_info_sep_symbol,
        **rag_settings)

    data = load_datapoints(cfg.dataset)
    dev = [data[i] for i in dev_indices(len(data), args.dev_size, args.dev_extra)]  # development datapoints only
    composer_args = dict(cfg.composers_config)

    rows = []
    for spec in args.arms:
        arm, kind = spec.split(":")
        arm_rows = proxy_rows(kind, arm, dev, composer_args, encode, decode, budget, rag_factory)
        if args.results:
            em = {key: value["em"] for key, value in arm_lines(os.path.expanduser(args.base), arm).items()}
            for r in arm_rows:
                r["em"] = em[(r["dp"], r["line_class"], r["k"])]
        rows += arm_rows
    print(summarize(rows))
    if args.results:
        result = hit_miss_difference(rows)
        print("\nEM(all needed names visible) - EM(not), averaged over arms, bootstrap over datapoints:")
        if result:
            diff, low, high, n_hit, n_miss = result
            print(f"  {diff:+.3f} [{low:+.3f}, {high:+.3f}]  (line-arm pairs: {n_hit} hit, {n_miss} miss)")
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
