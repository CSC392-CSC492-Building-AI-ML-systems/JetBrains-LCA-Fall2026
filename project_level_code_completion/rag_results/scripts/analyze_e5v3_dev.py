"""Analysis of the stronger agent's development arms (RAG_EXPERIMENT_LOG 8.93, 8.99): D0 RAG F, D1 7B+old tools, D2 32B+old tools,
D3 32B+v3, D4 7B+v3 tools, D5 32B+v3 with one tool call, all on the 24 development datapoints of medium (1,372 lines), same machine
for every arm. Applies the pre-registered decision rule of 8.93 to the arm named by --candidate (default d3):
  full run only if the datapoint-mean EM difference to D0 is >= +0.5 points, its 95 % interval lower bound is >= -1.0, and among the lines
  where the planner changed the context, the lines only RAG F gets right are at most 2x the lines only the agent gets right.
Reads the arms from rag_results/agent/E5v3_dev/arms (plain or gzipped per-line files). Writes rag_results/reports/E5v3/dev_analysis.md.
Usage: python rag_results/scripts/analyze_e5v3_dev.py [--candidate d3]"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from analyze_e5_full import dp_means, em, es, load, right, seconds  # noqa: E402
from eval.analyze_arms import paired_bootstrap  # noqa: E402

NAMES = {"d0": "D0 RAG F", "d1": "D1 7B + old tools", "d2": "D2 32B + old tools", "d3": "D3 32B + v3",
         "d4": "D4 7B + v3 tools", "d5": "D5 32B + v3, 1 tool call"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=os.path.join(HERE, "..", "agent", "E5v3_dev", "arms"))
    parser.add_argument("--out", default=os.path.join(HERE, "..", "reports", "E5v3", "dev_analysis.md"))
    parser.add_argument("--candidate", default="d3")
    args = parser.parse_args()
    base = os.path.expanduser(args.base)
    data = {}
    for key in NAMES:
        folder = f"e5v3dev_{key}"
        if os.path.exists(os.path.join(base, folder, "generation_results.jsonl")) or os.path.exists(os.path.join(base, folder, "generation_results.jsonl.gz")):
            data[key] = load(base, folder)
    keys = sorted(data["d0"])
    for k, rows in data.items():
        assert sorted(rows) == keys, f"{k} covers different lines than d0"
    ref_rows = data["d0"]
    ref = dp_means(ref_rows, keys, em)
    out, w = [], None
    out = []
    w = out.append
    w(f"# Stronger agent, development arms ({len(keys)} lines, {len(ref)} datapoints of medium; CodeLlama-7B writer, N=1,753, one machine)\n")
    w("Primary metric: EM = mean over datapoints (as in the E4/E5 tables); pooled-over-lines EM is shown next to it because two datapoints hold 42 % of the lines. "
      "Intervals: paired bootstrap over datapoints (10,000 draws, seed 0, same resample for both arms).\n")
    w("## 1. Headline\n")
    w("| arm | EM (datapoint mean) | ES | EM pooled | vs D0 (datapoint mean) | changed lines | fallback | writer input tokens/line | s/line (context+generation) |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, rows in data.items():
        dpm = dp_means(rows, keys, em)
        es_dp = dp_means(rows, keys, es)
        changed = sum(bool(rows[k]["retrieval"].get("changed")) for k in keys)
        fallback = sum(bool(rows[k]["retrieval"].get("fallback")) for k in keys)
        t = np.mean([seconds(rows[k]) for k in keys])
        if key == "d0":
            diff = "reference"
        else:
            d, lo, hi, n = paired_bootstrap(ref, dpm)
            diff = f"{100 * d:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]{' *' if lo > 0 or hi < 0 else ''}"
        w(f"| {NAMES[key]} | {np.mean(list(dpm.values())):.4f} | {np.mean(list(es_dp.values())):.2f} | "
          f"{np.mean([em(rows[k]) for k in keys]):.4f} | {diff} | {changed} | {fallback} | "
          f"{np.mean([rows[k]['metrics']['input_tokens'] for k in keys]):.0f} | {t:.1f} |")
    w("\n`*` = the 95 % interval excludes 0. Seconds per line of D2 to D5 were measured with two arms sharing the card.\n")

    w("## 2. Changed lines: who gets them right (against D0)\n")
    w("| arm | changed | both right | both wrong | only D0 right (broke) | only the arm right (fixed) | broke : fixed |")
    w("|---|---:|---:|---:|---:|---:|---|")
    ratios = {}
    for key, rows in data.items():
        if key == "d0":
            continue
        ch = [k for k in keys if rows[k]["retrieval"].get("changed")]
        both = sum(right(rows[k]) and right(ref_rows[k]) for k in ch)
        neither = sum(not right(rows[k]) and not right(ref_rows[k]) for k in ch)
        broke = sum(right(ref_rows[k]) and not right(rows[k]) for k in ch)
        fixed = sum(right(rows[k]) and not right(ref_rows[k]) for k in ch)
        ratios[key] = (broke, fixed)
        w(f"| {NAMES[key]} | {len(ch)} | {both} | {neither} | {broke} | {fixed} | {broke}:{fixed} |")
    w("")
    w("## 3. Lines with an unchanged context: does the writer reproduce D0 (same machine, same prompt)?\n")
    w("| arm | unchanged lines | prediction differs from D0 | of which EM differs |")
    w("|---|---:|---:|---:|")
    for key, rows in data.items():
        if key == "d0":
            continue
        un = [k for k in keys if not rows[k]["retrieval"].get("changed")]
        diff_pred = [k for k in un if rows[k]["original_prediction"] != ref_rows[k]["original_prediction"]]
        diff_em = [k for k in diff_pred if right(rows[k]) != right(ref_rows[k])]
        w(f"| {NAMES[key]} | {len(un)} | {len(diff_pred)} | {len(diff_em)} |")
    w("")
    w("## 4. What the planners did\n")
    w("| arm | lines | planner calls/line | tool calls/line | submitted at once | invalid-reply lines (fallback) | gate none / passed / rejected | planner tokens/line (in+out) | tokens of the writer prompt/line |")
    w("|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for key, rows in data.items():
        if key == "d0":
            continue
        R = [rows[k]["retrieval"] for k in keys]
        gate = Counter(r.get("gate") for r in R)
        w(f"| {NAMES[key]} | {len(R)} | {np.mean([r['steps'] for r in R]):.2f} | {np.mean([r['tool_calls'] for r in R]):.2f} | "
          f"{sum(r['tool_calls'] == 0 for r in R)} | {sum(r['planner_status'] != 'submitted' for r in R)} ({sum(bool(r['fallback']) for r in R)}) | "
          f"{gate.get('none', 0)} / {gate.get('passed', 0)} / {gate.get('rejected', 0)} (off {gate.get('off', 0)}) | "
          f"{np.mean([r['planner_input_tokens'] + r['planner_output_tokens'] for r in R]):.0f} | "
          f"{np.mean([rows[k]['metrics']['input_tokens'] for k in keys]):.0f} |")
    w("")
    cand = args.candidate
    if cand in data and cand != "d0":
        d, lo, hi, n = paired_bootstrap(ref, dp_means(data[cand], keys, em))
        broke, fixed = ratios[cand]
        ok_point, ok_low = 100 * d >= 0.5, 100 * lo >= -1.0
        ok_ratio = broke <= 2 * fixed
        w(f"## 5. Pre-registered decision rule (8.93) for {NAMES[cand]}\n")
        w(f"- point estimate {100 * d:+.2f} points (needs >= +0.50): **{'met' if ok_point else 'NOT met'}**")
        w(f"- lower bound {100 * lo:+.2f} (needs >= -1.00): **{'met' if ok_low else 'NOT met'}**")
        w(f"- broke : fixed among changed lines = {broke} : {fixed} (needs broke <= 2 x fixed): **{'met' if ok_ratio else 'NOT met'}**"
          f"{' (nothing changed enough to judge)' if broke + fixed < 5 else ''}")
        w(f"- **{'GO: run the full arm' if ok_point and ok_low and ok_ratio else 'NO-GO: do not run the full arm; report the dev table'}**\n")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
