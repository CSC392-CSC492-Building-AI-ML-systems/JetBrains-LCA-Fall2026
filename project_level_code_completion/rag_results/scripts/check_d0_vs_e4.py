"""Reproduction check for the dev arms (RAG_EXPERIMENT_LOG 8.99 prediction): D0 (RAG F on the RTX PRO 6000 Blackwell box, 24 dev datapoints,
1,372 lines) against E4's RAG F arm (e4_F_N1753, RTX 5090, all 224 datapoints) on the same lines. Same prompts, same greedy decoding; the
differences are the GPU numerics of 8.96. Writes rag_results/reports/E5v3/d0_vs_e4.md.
Usage: python rag_results/scripts/check_d0_vs_e4.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from analyze_e5_full import dp_means, em, load  # noqa: E402
from eval.analyze_arms import paired_bootstrap  # noqa: E402


def main():
    d0 = load(os.path.join(HERE, "..", "agent", "E5v3_dev", "arms"), "e5v3dev_d0")
    e4 = load(os.path.join(HERE, "..", "agent", "E5_full", "arms"), "e4_F_N1753")
    # The dev split renumbers datapoints, so each dev datapoint is matched to the E4 datapoint with the same repo and the same set of
    # (line number, ground truth) pairs; the match must be unique. E4 rows are then re-keyed by the dev arm's keys.
    def signature(rows, dp):
        r = [v for k, v in rows.items() if k[0] == dp]
        return (r[0]["repo"], tuple(sorted((v["line_num"], v["ground_truth"]) for v in r)))
    e4_sigs = {}
    for dp in sorted({k[0] for k in e4}):
        e4_sigs.setdefault(signature(e4, dp), []).append(dp)
    keys = sorted(d0)
    mapping = {}
    for dp in sorted({k[0] for k in keys}):
        found = e4_sigs.get(signature(d0, dp), [])
        assert len(found) == 1, f"dev datapoint {dp} matches {len(found)} E4 datapoints"
        mapping[dp] = found[0]
    e4 = {k: e4[(mapping[k[0]], k[1])] for k in keys}
    a = dp_means(e4, keys, em)
    b = dp_means(d0, keys, em)
    diff_pred = [k for k in keys if d0[k]["prediction_line"].strip() != e4[k]["prediction_line"].strip()]
    diff_em = [k for k in keys if em(d0[k]) != em(e4[k])]
    only_e4 = sum(em(e4[k]) > em(d0[k]) for k in keys)
    only_d0 = sum(em(d0[k]) > em(e4[k]) for k in keys)
    same_in = np.mean([d0[k]["metrics"]["input_tokens"] == e4[k]["metrics"]["input_tokens"] for k in keys])
    d, lo, hi, _ = paired_bootstrap(a, b)
    out = [
        "# D0 (Blackwell box) against E4 RAG F (RTX 5090) on the 24 dev datapoints\n",
        f"Lines: {len(keys)}, datapoints: {len(a)}. Writer input tokens identical on {100 * same_in:.1f} % of lines.\n",
        "| arm | EM datapoint mean | EM pooled |", "|---|---:|---:|",
        f"| E4 RAG F (5090) | {np.mean(list(a.values())):.4f} | {np.mean([em(e4[k]) for k in keys]):.4f} |",
        f"| D0 RAG F (Blackwell) | {np.mean(list(b.values())):.4f} | {np.mean([em(d0[k]) for k in keys]):.4f} |\n",
        f"D0 minus E4: {100 * d:+.2f} points [{100 * lo:+.2f}, {100 * hi:+.2f}] (paired bootstrap over datapoints).",
        f"Predictions that differ: {len(diff_pred)} of {len(keys)} lines ({100 * len(diff_pred) / len(keys):.1f} %); EM differs on {len(diff_em)} "
        f"(only E4 right {only_e4}, only D0 right {only_d0}).",
    ]
    path = os.path.join(HERE, "..", "reports", "E5v3", "d0_vs_e4.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
