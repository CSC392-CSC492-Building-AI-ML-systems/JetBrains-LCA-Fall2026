"""Final experiment table for the week 8 deliverable (accuracy, tokens, latency, compute across all approaches).
Part A (all 224 medium datapoints, 8,676 lines, CodeLlama-7B writer): E4 arms from figures/E4/e4_results_table.csv, the 7B agent arm (E5) from
figures/E5/e5_results_table.csv. Part B (24 development datapoints, 1,372 lines, one RTX PRO 6000 box): the stronger-agent arms recomputed from
rag_results/agent/E5v3_dev/arms. Writes rag_results/reports/final/final_experiment_table.{md,csv}.
Compute: GPU hours x $0.69 per hour (the price used in E4). E4 arms: wall hours of the run. Agent arm: summed per-line work hours (the shards ran
on several machines, so no single wall time exists); part B has no compute column because two arms shared one GPU.
Usage: python rag_results/scripts/make_final_table.py"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from analyze_e5_full import dp_means, em, load, seconds  # noqa: E402
from eval.analyze_arms import paired_bootstrap  # noqa: E402

PRICE = 0.69
RR = os.path.join(HERE, "..")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    e4 = read_csv(os.path.join(RR, "figures", "E4", "e4_results_table.csv"))
    e5 = read_csv(os.path.join(RR, "figures", "E5", "e5_results_table.csv"))
    rows_a = []  # approach, N, EM, tokens, p50, p95, gpu_h, usd
    for r in e4:
        hours = float(r["wall_hours"])
        rows_a.append(
            [
                r["approach"],
                r["budget_N"],
                f"{float(r['em_all']):.3f} [{float(r['em_ci_low']):.3f}, {float(r['em_ci_high']):.3f}]",
                f"{int(float(r['input_tokens_per_line'])):,}",
                r["line_p50_s"],
                r["line_p95_s"],
                f"{hours:.1f}",
                f"{hours * PRICE:.2f}",
            ]
        )
    agent = next(r for r in e5 if r["approach"].startswith("agent"))
    hours = float(agent["line_gpu_hours"])
    rows_a.append(
        [
            "7B planner agent (E5, add-only)",
            "1,753",
            f"{float(agent['em_all']):.3f} [{float(agent['em_ci_low']):.3f}, {float(agent['em_ci_high']):.3f}]",
            f"{int(float(agent['total_tokens_per_line'])):,}",
            agent["line_p50_s"],
            agent["line_p95_s"],
            f"{hours:.1f}",
            f"{hours * PRICE:.2f}",
        ]
    )

    base = os.path.join(RR, "agent", "E5v3_dev", "arms")
    names = [
        ("d0", "RAG F (same card)"),
        ("d1", "7B planner, E5 tools"),
        ("d2", "32B planner, E5 tools"),
        ("d3", "32B planner, v3, with gate"),
        ("d5", "32B planner, v3, one tool call"),
        ("d6", "32B planner, v3, no gate"),
    ]
    data = {k: load(base, f"e5v3dev_{k}") for k, _ in names}
    keys = sorted(data["d0"])
    ref = dp_means(data["d0"], keys, em)
    rows_b = []
    for k, label in names:
        rows = data[k]
        dpm = dp_means(rows, keys, em)
        if k == "d0":
            diff = "reference"
        else:
            d, lo, hi, _ = paired_bootstrap(ref, dpm)
            diff = f"{100 * d:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"
        planner = np.mean(
            [
                rows[x]["retrieval"].get("planner_input_tokens", 0)
                + rows[x]["retrieval"].get("planner_output_tokens", 0)
                for x in keys
            ]
        )
        writer = np.mean(
            [rows[x]["metrics"]["input_tokens"] + rows[x]["metrics"].get("output_tokens", 0) for x in keys]
        )
        secs = [seconds(rows[x]) for x in keys]
        rows_b.append(
            [
                label,
                f"{np.mean(list(dpm.values())):.4f}",
                diff,
                f"{planner + writer:,.0f}",
                f"{np.median(secs):.1f}",
                f"{np.mean(secs):.1f}",
                f"{sum(bool(rows[x]['retrieval'].get('changed')) for x in keys)}",
            ]
        )

    out = [
        "# Final experiment table (week 8): accuracy, tokens, latency and compute across all approaches\n",
        "## A. All 224 medium datapoints (8,676 lines), CodeLlama-7B writer, greedy, 100 new tokens max\n",
        "| approach | N | EM (95% CI) | tokens/line (planner + writer) | p50 s/line | p95 s/line | GPU hours | USD at $0.69/h |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    out += ["| " + " | ".join(r) + " |" for r in rows_a]
    out += [
        "\nTokens: the writer's prompt for the static arms (their only model); planner + writer for the agent arm. Seconds: context building (planner "
        "included) plus generation. GPU hours: wall time of the run for the E4 arms (includes model loading); summed per-line work for the agent arm "
        "(the real bill was about $22 because shards ran on several machines and two were resumed). EM 95% intervals are per arm and ignore pairing; "
        "the paired tests are in the E4 and E5 reports.\n",
        f"## B. Stronger agent, 24 development datapoints ({len(keys)} lines), one RTX PRO 6000, N = 1,753\n",
        "| arm | EM (mean over datapoints) | vs RAG F, points (95% CI) | tokens/line (planner + writer) | p50 s/line | mean s/line | lines changed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    out += ["| " + " | ".join(r) + " |" for r in rows_b]
    out += [
        "\nSeconds in part B are with another arm sharing the GPU for most of the run, so read them as an upper bound and compare them only with each "
        "other. Part B tokens include the writer's output (about 16 tokens per line), as the E5 total does; E5_report_section.md quotes planner plus "
        "writer prompt only, so its numbers are lower by that amount (under 0.1 %). The 7B planner with the v3 tools was stopped after 174 lines "
        "(log 8.101) and has no row.\n"
    ]
    md_path = os.path.join(RR, "reports", "final", "final_experiment_table.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    with open(md_path.replace(".md", ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["part", "approach", "N", "EM", "tokens_per_line", "p50_s", "p95_s_or_mean_s", "gpu_hours", "usd"])
        for r in rows_a:
            w.writerow(["A"] + r)
        for r in rows_b:
            w.writerow(["B", r[0], "1,753", r[1], r[3], r[4], r[5], "", ""])
    print("\n".join(out))


if __name__ == "__main__":
    main()
