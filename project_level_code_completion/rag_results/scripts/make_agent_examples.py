"""Concrete examples of what the agent did to RAG F's lines (RAG_EXPERIMENT_LOG section 8.92).

Random examples (fixed seed, not picked by hand) from each kind of change between the RAG F arm and the agent arm at N=1,753,
shown with the code before the line, the truth, both predictions, the names the line needs from other files that each prompt did
not contain, and what the planner searched, added and displaced. Prompts are rebuilt as in eval/failure_split.py (CPU only; the
planner is replayed from its logged actions).

Run from a directory with the tree-sitter grammar clone, with the project root on PYTHONPATH:
    python rag_results/scripts/make_agent_examples.py [--base DIR] [--per-kind 3] [--seed 0] [--out FILE]
"""
import argparse
import gzip
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("e4_F_N1753", "e5_agent_full")

# kind -> (title, test on (RAG F split class, agent split class, RAG F correct, agent correct), how many to show: None = --per-kind)
KINDS = [
    ("broke_names_visible", "Broke: RAG F right, agent wrong, the needed names were visible in both prompts",
     lambda r, a: r["cls"] == "right" and a["cls"] == "reasoning", None),
    ("broke_self_contained", "Broke: RAG F right, agent wrong, the line needs nothing from other files",
     lambda r, a: r["cls"] == "right_self_contained" and a["cls"] == "wrong_self_contained", None),
    ("broke_name_lost", "Broke: RAG F right, agent wrong, a needed name disappeared from the prompt (eviction)",
     lambda r, a: r["cls"] == "right" and a["cls"] == "retrieval", None),
    ("fixed", "Fixed: RAG F wrong, agent right",
     lambda r, a: not r["correct"] and a["correct"], None),
    ("miss_filled", "RAG F retrieval miss that the agent filled (every needed name now visible), all of them",
     lambda r, a: r["cls"] == "retrieval" and a["cls"] != "retrieval", 99),
    ("miss_stays", "RAG F retrieval miss that stays a miss under the agent",
     lambda r, a: r["cls"] == "retrieval" and a["cls"] == "retrieval", None),
]


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf8") as f:
        return [json.loads(line) for line in f]


def describe_action(action):
    if action is None:
        return "(invalid reply)"
    kind = action.get("action")
    if kind == "search_code":
        return f'search_code "{action.get("query", "")}"'
    if kind == "find_definition":
        return f'find_definition "{action.get("name", action.get("symbol", ""))}"'
    if kind == "read_file":
        return f'read_file {action.get("path")} lines {action.get("start_line")}-{action.get("end_line")}'
    if kind == "submit_context":
        items = [(f'chunk #{x["chunk"]}' if "chunk" in x else f'{x.get("path")} {x.get("start_line")}-{x.get("end_line")}') if isinstance(x, dict) else str(x)
                 for x in action.get("add", [])]
        return "submit_context add=[" + ", ".join(items) + "]"
    return json.dumps(action)[:100]


def clip(text, n=110):
    text = text.rstrip("\n")
    return text if len(text) <= n else text[:n - 3] + "..."


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=os.path.join(HERE, "..", "agent", "E5_full", "arms"))
    parser.add_argument("--per-kind", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(HERE, "..", "reports", "E5", "agent_examples.md"))
    args = parser.parse_args()
    base = os.path.expanduser(args.base)

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    from agent.llm import ReplayModel
    from agent.provider import make_agent_provider_factory
    from agent.tools import Limits
    from eval.preprocessors import load_datapoints
    from eval.proxy_metrics import definitions, identifiers, needed_names, repo_definitions, visible_prompt
    from rag.provider import make_rag_provider_factory

    splits = {arm: {(r["dp"], r["line_num"]): r for r in read_jsonl(os.path.join(base, arm, "failure_split.jsonl.gz"))} for arm in ARMS}
    results = {arm: {(r["datapoint_idx"], r["line_num"]): r for r in read_jsonl(os.path.join(base, arm, "generation_results.jsonl.gz"))}
               for arm in ARMS}
    keys = sorted(splits[ARMS[0]])
    rng = random.Random(args.seed)
    chosen = []
    for name, title, test, limit in KINDS:
        pool = [k for k in keys if test(splits[ARMS[0]][k], splits[ARMS[1]][k])]
        picked = sorted(rng.sample(pool, min(limit or args.per_kind, len(pool))))
        chosen.append((name, title, len(pool), picked))

    with initialize_config_dir(config_dir=os.path.join(HERE, "..", "..", "eval", "config"), version_base=None):
        cfg = compose(config_name="compare_arms", overrides=["dataset=medium", "params=codellama7b", "rag=F", "agent=append"])
    budget = cfg.params.inference_params["seq_max_len"] - 100
    tokenizer = AutoTokenizer.from_pretrained(cfg.params.preprocess_params["tokenizer"])

    def encode(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def count(text):
        return len(encode(text))

    rag_settings = dict(cfg.rag)
    rag_settings.update(prefix_cap=15900, max_context_tokens=1753)
    common = dict(extension=cfg.composers_config.extension, lang_sep=cfg.composers_config.lang_sep_symbol,
                  meta_sep=cfg.composers_config.meta_info_sep_symbol)
    rag_factory = make_rag_provider_factory(count, budget, **common, **rag_settings)
    replay = ReplayModel()
    agent_factory = make_agent_provider_factory(
        count, budget, replay, rag_options=dict(rag_settings), limits=Limits(**OmegaConf.to_container(cfg.agent.limits)),
        planner_prefix_tokens=cfg.agent.planner_prefix_tokens, max_listed_files=cfg.agent.max_listed_files, **common)
    data = load_datapoints(cfg.dataset)

    out = ["# What the agent did to RAG F's lines: random examples (N=1,753, CodeLlama-7B)\n",
           "Each kind below is a random draw (seed %d) from all lines of that kind; the number in brackets is how many lines there are. "
           "'Needed names' are the names the target line uses that the repository defines in another file (the lexical screen of the failure "
           "split); 'missing' = not present in the prompt the model saw. A name being present does not mean its definition was useful.\n" % args.seed]
    for name, title, total, picked in chosen:
        out.append(f"\n## {title} [{total} lines]\n")
        for dp_idx, line_num in picked:
            dp = data[dp_idx]
            path = next(iter(dp.get_completion()))
            line = dp.get_line(line_num)
            prefix = dp.get_prefix(line_num)
            needed = needed_names(line, prefix, repo_definitions(dp.get_context(), exclude_path=path))
            row = {arm: results[arm][(dp_idx, line_num)] for arm in ARMS}
            rag_prompt, _, rag_info = rag_factory(dp)(dp, line_num)
            replay.load(row[ARMS[1]]["retrieval"]["actions"])
            agent_prompt, _, agent_info = agent_factory(dp)(dp, line_num)
            missing = {}
            for label, prompt in (("RAG F", rag_prompt), ("agent", agent_prompt)):
                seen = identifiers(visible_prompt(prompt, encode, tokenizer.decode, budget))
                missing[label] = sorted(needed - seen)
            rag_chunks = [(p, tuple(r)) for p, r in zip(rag_info["paths"], rag_info["lines"])]
            agent_chunks = [(p, tuple(r)) for p, r in zip(agent_info["paths"], agent_info["lines"])]
            added = [c for c in agent_chunks if c not in rag_chunks]
            evicted = [c for c in rag_chunks if c not in agent_chunks]
            out.append(f"### {row[ARMS[0]]['repo']}, `{path}`, line {line_num} ({row[ARMS[0]]['line_class']}; datapoint {dp_idx})\n")
            out.append("Code before the line (last 6 lines of the prefix):\n```python\n" + "\n".join(clip(l) for l in prefix.rstrip("\n").split("\n")[-6:]) + "\n```")
            out.append("```text\n"
                       f"truth      : {clip(line.strip())}\n"
                       f"RAG F      : {clip(row[ARMS[0]]['prediction_line'].strip())}   [{'right' if splits[ARMS[0]][(dp_idx, line_num)]['correct'] else 'wrong'}]\n"
                       f"agent      : {clip(row[ARMS[1]]['prediction_line'].strip())}   [{'right' if splits[ARMS[1]][(dp_idx, line_num)]['correct'] else 'wrong'}]\n"
                       "```")
            out.append(f"- needed names from other files: {sorted(needed) if needed else 'none'}; missing in RAG F's prompt: {missing['RAG F'] or 'none'}; "
                       f"missing in the agent's prompt: {missing['agent'] or 'none'}")
            out.append("- planner: " + " -> ".join(describe_action(a) for a in row[ARMS[1]]["retrieval"]["actions"]))
            out.append("- added to the context: " + (", ".join(f"{p} {r[0]}-{r[1]}" for p, r in added) or "nothing")
                       + "; displaced from RAG F's context: " + (", ".join(f"{p} {r[0]}-{r[1]}" for p, r in evicted) or "nothing") + "\n")
    text = "\n".join(out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf8") as f:
        f.write(text + "\n")
    print(text[:3000])


if __name__ == "__main__":
    main()
