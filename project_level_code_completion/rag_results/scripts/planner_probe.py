"""Dev-set checks for the E5 planner agent (RAG_EXPERIMENT_LOG sections 8.80-8.82). Development datapoints only.

    probe     run one planner candidate over some development lines (no CodeLlama, no accuracy) and report how reliably
              it ends in a valid submit_context:
                  python planner_probe.py probe --planner NAME --dtype 8bit --out DIR
    choose    apply the pre-registered selection rule to two probe summaries:
                  python planner_probe.py choose DIR_A DIR_B
              rule (fixed before any run): the candidate with the higher share of lines that end in a valid
              submit_context wins; if the shares differ by less than 2 points, the one with the lower mean planner time.
              Accuracy is not used.
    check     compare the agent arm with the RAG F arm on the same development lines (needs both arms' results):
                  python planner_probe.py check RAG_ARM_DIR AGENT_ARM_DIR
Run from ~/ts_test with PYTHONPATH=$PROJECT (tree-sitter chunking needs the grammar in the working directory).
"""
import argparse
import json
import os
import sys
import time

DEV_DATAPOINTS = 4  # dev_indices(n, 4) = 0, 56, 112, 168 on the medium set: all inside the 28+28 development set


def probe(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    from agent.llm import LocalChatModel
    from agent.provider import make_agent_provider_factory
    from agent.tools import Limits
    from eval.line_generators import dev_indices
    from eval.preprocessors import load_datapoints

    with initialize_config_dir(config_dir=os.path.join(os.environ["PROJECT"], "eval", "config"), version_base=None):
        cfg = compose(config_name="compare_arms", overrides=[
            "arm=agent", "rag=F", "rag.prefix_cap=15900", f"context_cap={args.cap}", "params=codellama7b",
            "dataset=medium"])
    tokenizer = AutoTokenizer.from_pretrained(cfg.params.preprocess_params["tokenizer"])
    budget = cfg.params.inference_params["seq_max_len"] - 100
    rag_options = OmegaConf.to_container(cfg.rag)
    rag_options["max_context_tokens"] = args.cap
    planner = LocalChatModel(args.planner, max_new_tokens=cfg.agent.planner_max_new_tokens, dtype=args.dtype)
    factory = make_agent_provider_factory(
        lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"]), budget, planner,
        rag_options=rag_options, limits=Limits(**OmegaConf.to_container(cfg.agent.limits)),
        planner_prefix_tokens=cfg.agent.planner_prefix_tokens, max_listed_files=cfg.agent.max_listed_files,
        extension=cfg.composers_config.extension, lang_sep=cfg.composers_config.lang_sep_symbol,
        meta_sep=cfg.composers_config.meta_info_sep_symbol)

    data = load_datapoints(cfg.dataset)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    started = time.time()
    for index in dev_indices(len(data), DEV_DATAPOINTS):
        datapoint = data[index]
        provider = factory(datapoint)
        lines = sorted(n for nums in datapoint.completion_lines.values() for n in nums)
        step = max(1, len(lines) // args.lines)
        for line_num in lines[::step][:args.lines]:
            _, _, info = provider(datapoint, line_num)
            rows.append({"datapoint_idx": index, "line_num": line_num, **info})
            print(f"dp {index} line {line_num}: {info['planner_status']} steps={info['steps']} tools={info['tool_calls']} "
                  f"changed={info['changed']}", flush=True)
    with open(os.path.join(args.out, "probe_rows.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    n = len(rows)

    def share(pred):
        return round(sum(1 for r in rows if pred(r)) / n, 4)

    def mean(key):
        return round(sum(r[key] for r in rows) / n, 2)

    summary = {
        "planner": args.planner, "dtype": args.dtype, "lines": n, "datapoints": DEV_DATAPOINTS, "cap": args.cap,
        "submitted_share": share(lambda r: r["planner_status"] == "submitted"),
        "invalid_replies_share": share(lambda r: r["planner_status"] == "invalid_replies"),
        "no_submit_share": share(lambda r: r["planner_status"] == "no_submit"),
        "used_a_tool_share": share(lambda r: r["tool_calls"] > 0), "changed_share": share(lambda r: r["changed"]),
        "fallback_share": share(lambda r: r["fallback"]), "mean_steps": mean("steps"), "mean_tool_calls": mean("tool_calls"),
        "mean_planner_input_tokens": mean("planner_input_tokens"), "mean_planner_output_tokens": mean("planner_output_tokens"),
        "mean_planner_ms": mean("planner_ms"), "wall_seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(args.out, "probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def choose(args):
    a, b = (json.load(open(os.path.join(d, "probe_summary.json"))) for d in (args.dir_a, args.dir_b))
    gap = a["submitted_share"] - b["submitted_share"]
    if abs(gap) >= 0.02:
        winner = a if gap > 0 else b
        why = f"higher share of valid submit_context ({a['submitted_share']:.3f} vs {b['submitted_share']:.3f})"
    else:
        winner = a if a["mean_planner_ms"] <= b["mean_planner_ms"] else b
        why = (f"valid-submit shares within 2 points ({a['submitted_share']:.3f} vs {b['submitted_share']:.3f}); "
               f"lower mean planner time ({a['mean_planner_ms']} vs {b['mean_planner_ms']} ms)")
    print(f"CHOSEN {winner['planner']} because {why}")
    return winner["planner"]


def load(path):
    rows = {}
    with open(os.path.join(path, "generation_results.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            rows[(row["datapoint_idx"], row["line_num"])] = row
    return rows


def check(args):
    rag, agent = load(args.rag_dir), load(args.agent_dir)
    keys = sorted(set(rag) & set(agent))
    if set(rag) != set(agent):
        print(f"WARNING: arms cover different lines (rag {len(rag)}, agent {len(agent)}, common {len(keys)})")

    def right(row):
        return row["prediction_line"].strip() == row["ground_truth"].strip()

    changed = [k for k in keys if agent[k]["retrieval"]["changed"]]
    unchanged = [k for k in keys if not agent[k]["retrieval"]["changed"]]
    same = sum(1 for k in unchanged if rag[k]["prediction_line"] == agent[k]["prediction_line"])
    both = sum(1 for k in changed if right(rag[k]) and right(agent[k]))
    only_agent = sum(1 for k in changed if right(agent[k]) and not right(rag[k]))
    only_rag = sum(1 for k in changed if right(rag[k]) and not right(agent[k]))
    summary = {
        "lines": len(keys), "changed": len(changed), "fallback": sum(1 for k in keys if agent[k]["retrieval"]["fallback"]),
        "unchanged_lines_with_identical_prediction": f"{same} of {len(unchanged)}",
        "rag_right": sum(right(rag[k]) for k in keys), "agent_right": sum(right(agent[k]) for k in keys),
        "changed_lines_both_right": both, "changed_lines_only_agent_right": only_agent,
        "changed_lines_only_rag_right": only_rag,
        "agent_total_tokens_per_line": round(sum(agent[k]["metrics"]["total_tokens"] for k in keys) / len(keys), 1),
        "rag_total_tokens_per_line": round(sum(rag[k]["metrics"]["input_tokens"] + rag[k]["metrics"]["output_tokens"]
                                               for k in keys) / len(keys), 1),
    }
    print(json.dumps(summary, indent=2))
    if same != len(unchanged):
        print("FAIL: some lines the planner did not change got a different prediction than RAG F; "
              "the prompts should be identical, so check the determinism of the two runs before trusting any comparison")
        sys.exit(1)
    print("OK: every unchanged line has the same prediction as RAG F")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--planner", required=True)
    p.add_argument("--dtype", default="8bit")
    p.add_argument("--out", required=True)
    p.add_argument("--lines", type=int, default=12, help="lines per datapoint")
    p.add_argument("--cap", type=int, default=1753)
    p.set_defaults(func=probe)
    c = sub.add_parser("choose")
    c.add_argument("dir_a")
    c.add_argument("dir_b")
    c.set_defaults(func=choose)
    k = sub.add_parser("check")
    k.add_argument("rag_dir")
    k.add_argument("agent_dir")
    k.set_defaults(func=check)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
