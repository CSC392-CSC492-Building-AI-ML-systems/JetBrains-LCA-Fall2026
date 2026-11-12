"""E5 figures and results table: the agent against the three E4 arms at N=1,753 (RAG_EXPERIMENT_LOG sections 8.87-8.89).

Reads the arm folders in rag_results/agent/E5_full/arms (summary.json, per_datapoint_scores.json, generation_results.jsonl[.gz]
of e4_zero, e4_pd_N1753, e4_F_N1753 and e5_agent_full) and writes PNG + PDF figures and a table (csv, markdown) to
rag_results/figures/E5/. Same style and colors as make_e4_figures.py: path_distance = blue, RAG F = orange, agent = aqua,
no context = gray.

    python rag_results/scripts/make_e5_figures.py [--base DIR] [--out DIR]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_e5_full as an  # noqa: E402  (also puts the project root on sys.path)
from make_e4_figures import AQUA, BLUE, GRAY, GRID, INK, INK2, MUTED, ORANGE, SURFACE, bootstrap_ci, plt, save  # noqa: E402

ORDER = [("no context", "e4_zero", GRAY), ("path_distance", "e4_pd_N1753", BLUE), ("RAG F", "e4_F_N1753", ORANGE),
         ("agent (v2)", "e5_agent_full", AQUA)]
USD_PER_HOUR = 0.69  # the E4 price, applied to every arm the same way (sum of per-line seconds), see the note in the table


def arm_stats(base, folder):
    rows = an.load(base, folder)
    keys = sorted(rows)
    scores = json.load(open(os.path.join(base, folder, "per_datapoint_scores.json"), encoding="utf8"))
    em = np.array((scores.get("em_list") or scores["em_zero_list"])["all"])
    secs = np.array([an.seconds(rows[k]) for k in keys])
    total_tokens = np.mean([rows[k]["metrics"].get("total_tokens", rows[k]["metrics"]["input_tokens"] + rows[k]["metrics"]["output_tokens"])
                            for k in keys])
    return {
        "rows": rows, "keys": keys, "em": em, "es": np.mean(list(an.dp_means(rows, keys, an.es).values())),
        "wrong": sum(not an.right(rows[k]) for k in keys), "input_tokens": np.mean([rows[k]["metrics"]["input_tokens"] for k in keys]),
        "total_tokens": total_tokens, "p50": np.percentile(secs, 50), "p95": np.percentile(secs, 95), "mean_s": secs.mean(),
        "line_hours": secs.sum() / 3600, "truncated": np.mean([bool(rows[k]["metrics"]["truncated"]) for k in keys]),
    }


def fig_em(arms, out):
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    y = np.arange(len(ORDER))[::-1]
    for yi, (label, folder, color) in zip(y, ORDER):
        a = arms[folder]
        lo, hi = bootstrap_ci(a["em"])
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=3, alpha=0.5, solid_capstyle="round")
        ax.scatter([a["em"].mean()], [yi], s=70, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"{a['em'].mean():.3f}", (hi, yi), textcoords="offset points", xytext=(8, -3.5), fontsize=8.5, color=INK2)
    ref = an.dp_means(arms["e4_F_N1753"]["rows"], arms["e4_F_N1753"]["keys"], an.em)
    oth = an.dp_means(arms["e5_agent_full"]["rows"], arms["e5_agent_full"]["keys"], an.em)
    d, lo, hi, _ = an.paired_bootstrap(ref, oth)
    ax.set_yticks(y, [label for label, _, _ in ORDER])
    ax.set_xlim(0.34, 0.60)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("exact match (all lines)")
    ax.set_title("The agent does not beat plain retrieval", loc="left", fontsize=11, color=INK)
    fig.text(0.0, -0.05, f"CodeLlama-7B, N = 1,753 tokens, 224 medium datapoints, 8,676 lines per arm; bars = 95% bootstrap CI over datapoints.\n"
             f"Paired: agent minus RAG F = {100 * d:+.2f} points [{100 * lo:+.2f}, {100 * hi:+.2f}]; the unpaired bars overlap because they ignore pairing.",
             fontsize=8, color=MUTED, ha="left", va="top")
    save(fig, out, "fig_e5_em")


def fig_by_class(arms, out):
    rag, agent = arms["e4_F_N1753"], arms["e5_agent_full"]
    classes = ["inproject", "infile", "common", "commited", "random", "non_informative"]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    y = np.arange(len(classes))[::-1]
    ax.axvline(0, color=INK2, linewidth=1)
    for yi, cls in zip(y, classes):
        r_ = an.dp_means(rag["rows"], rag["keys"], an.em, cls)
        a_ = an.dp_means(agent["rows"], agent["keys"], an.em, cls)
        d, lo, hi, n = an.paired_bootstrap(r_, a_)
        ax.plot([100 * lo, 100 * hi], [yi, yi], color=AQUA, linewidth=3, alpha=0.5, solid_capstyle="round")
        ax.scatter([100 * d], [yi], s=60, color=AQUA, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"{100 * d:+.1f}", (100 * hi, yi), textcoords="offset points", xytext=(8, -3.5), fontsize=8.5, color=INK2)
    ax.set_yticks(y, classes)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("exact match, agent minus RAG F (points)")
    ax.set_title("No line class improves; cross-file lines lose most", loc="left", fontsize=11, color=INK)
    fig.text(0.0, -0.05, "Per class: mean over the datapoints that have lines of the class; bars = paired 95% bootstrap CI over datapoints.\n"
             "inproject lines need names defined in other project files, the case an agent that can search and read files should help.",
             fontsize=8, color=MUTED, ha="left", va="top")
    save(fig, out, "fig_e5_by_class")


def fig_cost(arms, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    rag = arms["e4_F_N1753"]
    for ax, key, title, fmt in ((axes[0], "total_tokens", "tokens per line (planner + writer)", "{:,.0f}"),
                                (axes[1], "p50", "median seconds per line", "{:.2f}")):
        y = np.arange(len(ORDER))[::-1]
        for yi, (label, folder, color) in zip(y, ORDER):
            v = arms[folder][key]
            ax.barh(yi, v, color=color, height=0.62, zorder=2)
            suffix = f"  ({v / rag[key]:.1f}x RAG F)" if folder == "e5_agent_full" else ""
            ax.annotate(fmt.format(v) + suffix, (v, yi), textcoords="offset points", xytext=(5, -3.5), fontsize=8.5, color=INK2)
        ax.set_yticks(y, [l for l, _, _ in ORDER] if ax is axes[0] else [])
        ax.set_xlim(0, max(arms[f][key] for _, f, _ in ORDER) * 1.55)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.suptitle("What the agent costs", x=0.01, y=1.04, ha="left", fontsize=11, color=INK)
    fig.text(0.0, -0.03, "Seconds = context building (planner included) + generation, as in the E4 latency figure. The agent arm ran on several rented GPUs of\n"
             "different speed, so its times are an order of magnitude, not a precise figure.", fontsize=8, color=MUTED, ha="left", va="top")
    save(fig, out, "fig_e5_cost")


def table(arms, out):
    rows = []
    for label, folder, _ in ORDER:
        a = arms[folder]
        lo, hi = bootstrap_ci(a["em"])
        rows.append({
            "approach": label, "em_all": round(a["em"].mean(), 4), "em_ci_low": round(lo, 4), "em_ci_high": round(hi, 4),
            "es_all": round(a["es"], 2), "lines": len(a["keys"]), "wrong_lines": a["wrong"],
            "writer_input_tokens_per_line": round(a["input_tokens"]), "total_tokens_per_line": round(a["total_tokens"]),
            "line_p50_s": round(a["p50"], 2), "line_p95_s": round(a["p95"], 2), "line_mean_s": round(a["mean_s"], 2),
            "truncated_share": round(a["truncated"], 4), "line_gpu_hours": round(a["line_hours"], 2),
            "line_cost_usd_at_0.69": round(a["line_hours"] * USD_PER_HOUR, 2),
        })
    with open(os.path.join(out, "e5_results_table.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out, "e5_results_table.md"), "w") as f:
        f.write("| approach | EM (95% CI) | ES | wrong lines (of 8,676) | writer input tokens/line | total tokens/line | p50 s/line | p95 s/line | truncated | line work (GPU h) |\n"
                "|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['approach']} | {r['em_all']:.3f} [{r['em_ci_low']:.3f}, {r['em_ci_high']:.3f}] | {r['es_all']:.1f} | {r['wrong_lines']:,} | "
                    f"{r['writer_input_tokens_per_line']:,} | {r['total_tokens_per_line']:,} | {r['line_p50_s']:.2f} | {r['line_p95_s']:.2f} | "
                    f"{r['truncated_share'] * 100:.1f}% | {r['line_gpu_hours']:.2f} |\n")
        f.write("\nN = 1,753 tokens, CodeLlama-7B, 224 medium datapoints. Total tokens = planner + writer input + writer output (non-agent arms: writer only). "
                "Line work = sum of per-line context + generation seconds, the same definition for every arm (E4's table prices wall time of a run at $0.69/h, "
                "which also counts model loading; the money actually spent on the agent arm is on the rental bill, see log 8.88).\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=os.path.join(HERE, "..", "agent", "E5_full", "arms"))
    parser.add_argument("--out", default=os.path.join(HERE, "..", "figures", "E5"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arms = {folder: arm_stats(args.base, folder) for _, folder, _ in ORDER}
    fig_em(arms, args.out)
    fig_by_class(arms, args.out)
    fig_cost(arms, args.out)
    for r in table(arms, args.out):
        print(r["approach"], r["em_all"], r["wrong_lines"], r["total_tokens_per_line"], r["line_p50_s"], r["line_gpu_hours"])


if __name__ == "__main__":
    main()
