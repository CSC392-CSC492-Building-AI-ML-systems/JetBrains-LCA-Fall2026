"""Random samples of wrong lines for the week 8 failure taxonomy (RAG_EXPERIMENT_LOG 8.106), medium set, 224 datapoints, CodeLlama-7B, N = 1,753.
Groups (seed 0, 25 lines each): wrong lines of no context, path_distance, RAG F and the 7B planner agent; and lines the agent broke (RAG F right,
agent wrong, context changed). Each item shows the ground truth and the four predictions, so labels can be compared across arms.
Reads the arms in experiments/results/agent/E5_full/arms. Writes experiments/reports/final/failure_samples_week8.txt and .json (sample ids with line keys).
Usage: python experiments/scripts/report_analysis/sample_failures_week8.py"""

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from analyze_e5_full import load, right  # noqa: E402

BASE = os.path.join(HERE, "..", "..", "results", "agent", "E5_full", "arms")
ARMS = {"zero": "e4_zero", "pd": "e4_pd_N1753", "F": "e4_F_N1753", "agent": "e5_agent_full"}
N = 25


def main():
    data = {k: load(BASE, v) for k, v in ARMS.items()}
    keys = sorted(data["F"])
    groups = {f"wrong:{k}": [x for x in keys if not right(data[k][x])] for k in ARMS}
    groups["broke:agent"] = [
        x
        for x in keys
        if right(data["F"][x]) and not right(data["agent"][x]) and data["agent"][x]["retrieval"].get("changed")
    ]
    rng = random.Random(0)
    text, index = [], []
    for name, pool in groups.items():
        text.append(f"\n######## {name}: {len(pool)} lines in the group, random sample of {N} (seed 0)")
        for i, x in enumerate(rng.sample(pool, N)):
            sid = f"{name}#{i + 1}"
            row = data["F"][x]
            index.append({"id": sid, "dp": x[0], "line_num": x[1]})
            text.append(f"[{sid} {row['repo']} L{x[1]} {row['line_class']}]")
            text.append("  GT   : " + row["ground_truth"].strip()[:220])
            for k in ARMS:
                mark = "ok " if right(data[k][x]) else "   "
                text.append(f"  {k:<5}{mark}: " + data[k][x]["prediction_line"].strip()[:220])
    out = os.path.join(HERE, "..", "..", "reports", "final")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "failure_samples_week8.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(text) + "\n")
    with open(os.path.join(out, "failure_samples_week8.json"), "w", encoding="utf-8") as f:
        json.dump({"groups": {k: len(v) for k, v in groups.items()}, "items": index}, f, indent=1)
    print({k: len(v) for k, v in groups.items()}, len(index), "items")


if __name__ == "__main__":
    main()
