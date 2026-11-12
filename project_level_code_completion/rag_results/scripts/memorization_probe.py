"""Memorization probe for the planner (RAG_EXPERIMENT_LOG section 8.93, run before any full agent arm).

The benchmark's datapoints are public GitHub repositories, so a planner may have seen them in training and could "recall" the next
line without any repository context. This asks the planner, with NO repository at all, to write the next line from the file
prefix alone, on a fixed sample of lines, and reports exact match overall and split by the datapoint's commit date around a
cutoff date (the planner's training cutoff, to be read from its model card and given with --cutoff). A much higher exact match on
datapoints committed before the cutoff than after it is a warning sign (not proof: repositories also change in difficulty).
CodeLlama-7B's zero-context exact match on the same lines (E4 arm e4_zero) is printed as a reference for the writer.

    python rag_results/scripts/memorization_probe.py --cutoff 2024-03-01 [--planner Qwen/Qwen2.5-Coder-32B-Instruct --dtype 4bit]
        [--dev] [--lines-per-datapoint 10] [--out probe.jsonl]
Needs a GPU for the planner (run it where the planner runs). Commit dates come from rag_results/reports/E4/medium_commit_dates.json
(joined by repository and commit hash; datapoints without a date are left out of the split but kept in the overall figure).
"""
import argparse
import gzip
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM = ("You complete Python code. You are shown the beginning of a file. Reply with only the next line of code, exactly as it "
          "would appear in the file, with no explanation and no code fences.")


def pick_lines(lines, k):
    """k evenly spaced lines (all of them if there are at most k)."""
    if len(lines) <= k:
        return list(lines)
    return [lines[i * len(lines) // k] for i in range(k)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planner", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--dtype", default="4bit")
    parser.add_argument("--cutoff", required=True, help="training cutoff of the planner, YYYY-MM-DD (from its model card)")
    parser.add_argument("--dev", action="store_true", help="only the development split of 8.93 (24 datapoints)")
    parser.add_argument("--lines-per-datapoint", type=int, default=10)
    parser.add_argument("--prefix-tokens", type=int, default=1500)
    parser.add_argument("--arms", default=os.path.join(HERE, "..", "agent", "E5_full", "arms"))
    parser.add_argument("--dates", default=os.path.join(HERE, "..", "reports", "E4", "medium_commit_dates.json"))
    parser.add_argument("--out", help="write one JSON row per line to this file")
    args = parser.parse_args()

    from datasets import load_dataset
    from hydra import compose, initialize_config_dir
    from transformers import AutoTokenizer

    from agent.llm import LocalChatModel
    from eval.line_generators import dev_indices
    from eval.preprocessors import load_datapoints
    from rag.builder import crop_prefix

    with initialize_config_dir(config_dir=os.path.join(HERE, "..", "..", "eval", "config"), version_base=None):
        cfg = compose(config_name="compare_arms", overrides=["dataset=medium", "params=codellama7b"])
    tokenizer = AutoTokenizer.from_pretrained(cfg.params.preprocess_params["tokenizer"])

    def count(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    data = load_datapoints(cfg.dataset)
    hashes = [row["commit_hash"] for row in load_dataset(split="test", **cfg.dataset)]  # same order as `data`
    dates = {(d["repo"], d["sha"]): d.get("commit_date") for d in json.load(open(args.dates, encoding="utf8"))}
    indices = dev_indices(len(data), 8, 16) if args.dev else range(len(data))

    zero = {}
    with gzip.open(os.path.join(args.arms, "e4_zero", "generation_results.jsonl.gz"), "rt", encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            zero[(r["datapoint_idx"], r["line_num"])] = r["prediction_line"].strip() == r["ground_truth"].strip()

    planner = LocalChatModel(args.planner, max_new_tokens=100, dtype=args.dtype, device=cfg.params.eval_params["device"])
    rows = []
    for dp_idx in indices:
        dp = data[dp_idx]
        path = next(iter(dp.get_completion()))
        date = dates.get((dp.repo_name, hashes[dp_idx]))
        for line_num in pick_lines([n for lines in dp.completion_lines.values() for n in lines], args.lines_per_datapoint):
            prefix = crop_prefix(dp.get_prefix(line_num), args.prefix_tokens, count)
            reply = planner([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": f"File: {path}\n\n{prefix}"}])
            guess = reply.text.strip("\n").split("\n")[0]
            truth = dp.get_line(line_num)
            rows.append({"dp": dp_idx, "line_num": line_num, "truth": truth.strip(), "guess": guess.strip(),
                         "correct": guess.strip() == truth.strip(), "commit_date": date,
                         "before_cutoff": (date[:10] < args.cutoff) if date else None,
                         "codellama_zero_context_correct": zero.get((dp_idx, line_num))})
        print(f"datapoint {dp_idx}: {len(rows)} lines so far, EM {statistics.fmean(r['correct'] for r in rows):.3f}", flush=True)

    def em(group):
        return f"{statistics.fmean(r['correct'] for r in group):.3f} on {len(group)} lines" if group else "n/a"

    before = [r for r in rows if r["before_cutoff"] is True]
    after = [r for r in rows if r["before_cutoff"] is False]
    print(f"planner {args.planner} ({args.dtype}), no repository, cutoff {args.cutoff}")
    print(f"  overall: {em(rows)}")
    print(f"  commit before the cutoff: {em(before)}; after the cutoff: {em(after)}; no date: {em([r for r in rows if r['before_cutoff'] is None])}")
    ref = [r for r in rows if r["codellama_zero_context_correct"] is not None]
    print(f"  CodeLlama-7B zero-context (E4) on the same lines: {statistics.fmean(r['codellama_zero_context_correct'] for r in ref):.3f}")
    if args.out:
        with open(args.out, "w", encoding="utf8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
