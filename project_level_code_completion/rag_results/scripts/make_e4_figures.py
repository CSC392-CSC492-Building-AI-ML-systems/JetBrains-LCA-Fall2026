"""E4 figures and results table from the synced arms (RAG_EXPERIMENT_LOG section 8.78).

Reads ~/lca_remote/E4 (summary.json, per_datapoint_scores.json, generation_results.jsonl of the nine arms and failure_split.jsonl)
and writes PNG + PDF figures, a results table (csv, markdown, latex) to rag_results/figures/E4/. Colors follow the dataviz palette
(categorical slots 1-4, validated; light mode): path_distance = blue, RAG F = orange.

    python make_e4_figures.py [--arms DIR] [--split FILE] [--out DIR]
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#898781"
SIZES = [("256", "N256"), ("1,753", "N1753"), ("12,000", "N12000"), ("16K window", "16k")]
CATEGORIES = ["infile", "inproject", "commited", "common", "non_informative", "random"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})


def load_arm(base, arm):
    folder = os.path.join(base, arm)
    summary = json.load(open(os.path.join(folder, "summary.json")))
    scores = json.load(open(os.path.join(folder, "per_datapoint_scores.json")))
    em = scores.get("em_list") or scores["em_zero_list"]
    es = scores.get("es_list") or scores["es_zero_list"]
    end_to_end = []
    with open(os.path.join(folder, "generation_results.jsonl"), encoding="utf8") as f:
        for line in f:
            m = json.loads(line)["metrics"]
            end_to_end.append((m["context_ms"] + m["gen_ms"]) / 1000)
    score = summary["scores"].get("em") or summary["scores"]["em_zero"]
    return {
        "arm": arm, "em": np.array(em["all"]), "es_all": float(np.mean(es["all"])),
        "per_category": {c: score[c] for c in CATEGORIES}, "cost": summary["cost"],
        "p50": float(np.percentile(end_to_end, 50)), "p95": float(np.percentile(end_to_end, 95)),
    }


def bootstrap_ci(values, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_accuracy(arms, out):
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(SIZES))
    for approach, prefix, color in (("path_distance", "e4_pd_", BLUE), ("RAG F", "e4_F_", ORANGE)):
        vals = np.array([arms[prefix + key]["em"].mean() for _, key in SIZES])
        cis = np.array([bootstrap_ci(arms[prefix + key]["em"]) for _, key in SIZES])
        ax.fill_between(x, cis[:, 0], cis[:, 1], color=color, alpha=0.15, linewidth=0)
        ax.plot(x, vals, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=approach)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 9 if approach == "RAG F" else -15),
                        ha="center", fontsize=8.5, color=INK2)
    zero = arms["e4_zero"]["em"].mean()
    ax.axhline(zero, color=GRAY, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(len(SIZES) - 0.95, zero - 0.004, f"no repository context: {zero:.3f}", ha="right", va="top", fontsize=8.5, color=INK2)
    ax.set_xticks(x, [label for label, _ in SIZES])
    ax.set_xlabel("repository-context budget N (tokens)")
    ax.set_ylabel("exact match (all lines)")
    ax.set_ylim(0.34, 0.58)
    ax.set_xlim(-0.25, len(SIZES) - 0.75)
    ax.legend(loc="upper left")
    ax.set_title("More context helps, retrieval uses it better", loc="left", fontsize=11, color=INK)
    fig.text(0.0, -0.04, "CodeLlama-7B, 224 medium datapoints, 8,676 lines per arm; shaded band = 95% bootstrap CI over datapoints.\n"
             "The x-axis is the paper's ladder, evenly spaced.", fontsize=8, color=MUTED, ha="left", va="top")
    save(fig, out, "fig_em_vs_context")


def fig_cost(arms, out):
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    # label offsets (points) per budget, chosen so that no label sits on a marker or a line
    offsets = {"path_distance": [(-9, -4, "right"), (9, -14, "left"), (0, -16, "center"), (0, 9, "center")],
               "RAG F": [(0, 10, "center"), (9, 4, "left"), (0, -16, "center"), (0, 9, "center")]}
    for approach, prefix, color in (("path_distance", "e4_pd_", BLUE), ("RAG F", "e4_F_", ORANGE)):
        xs = [arms[prefix + key]["cost"]["arm_cost_usd"] for _, key in SIZES]
        ys = [arms[prefix + key]["em"].mean() for _, key in SIZES]
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.6)
        ax.scatter(xs, ys, s=60, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=approach)
        for (label, _), xv, yv, (dx, dy, ha) in zip(SIZES, xs, ys, offsets[approach]):
            short = {"16K window": "16K"}.get(label, label)
            ax.annotate(short, (xv, yv), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8.5, color=INK2)
    zx, zy = arms["e4_zero"]["cost"]["arm_cost_usd"], arms["e4_zero"]["em"].mean()
    ax.scatter([zx], [zy], s=60, color=GRAY, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="no context")
    ax.annotate("none", (zx, zy), textcoords="offset points", xytext=(9, -4), ha="left", fontsize=8.5, color=INK2)
    ax.set_xlabel("GPU cost of the arm (USD, RTX 5090 at $0.69/h, 8,676 lines)")
    ax.set_ylabel("exact match (all lines)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("RAG reaches the plateau at half the cost", loc="left", fontsize=11, color=INK)
    fig.text(0.0, -0.04, "Point labels: budget N in tokens. Cost = wall time of the arm times the hourly price.", fontsize=8, color=MUTED, ha="left", va="top")
    save(fig, out, "fig_cost_vs_em")


def fig_latency(arms, out):
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    groups = [("no context", None, "e4_zero")] + [(label.replace(" window", ""), key, None) for label, key in SIZES]
    x = np.arange(len(groups))
    width = 0.36
    for offset, approach, prefix, color in ((-width / 2, "path_distance", "e4_pd_", BLUE), (width / 2, "RAG F", "e4_F_", ORANGE)):
        for xi, (label, key, fixed) in zip(x, groups):
            if fixed:
                continue
            a = arms[prefix + key]
            ax.bar(xi + offset, a["p50"], width=width - 0.04, color=color, label=approach if xi == 1 else None, zorder=2)
            ax.plot([xi + offset] * 2, [a["p50"], a["p95"]], color=INK2, linewidth=1.2, zorder=3)
            ax.plot([xi + offset - 0.05, xi + offset + 0.05], [a["p95"]] * 2, color=INK2, linewidth=1.2, zorder=3)
    z = arms["e4_zero"]
    ax.bar(0, z["p50"], width=width * 1.6, color=GRAY, label="no context", zorder=2)
    ax.plot([0, 0], [z["p50"], z["p95"]], color=INK2, linewidth=1.2, zorder=3)
    ax.plot([-0.05, 0.05], [z["p95"]] * 2, color=INK2, linewidth=1.2, zorder=3)
    ax.set_xticks(x, [g[0] for g in groups])
    ax.set_xlabel("repository-context budget N (tokens)")
    ax.set_ylabel("seconds per line (context building + generation)")
    ax.set_title("Latency follows prompt length; retrieval adds little", loc="left", fontsize=11, color=INK)
    ax.legend(loc="upper left")
    fig.text(0.0, -0.04, "Bar = median (p50); whisker top = 95th percentile. Greedy decoding, batch size 1, up to 100 new tokens.", fontsize=8, color=MUTED, ha="left", va="top")
    save(fig, out, "fig_latency")


def fig_failures(split_path, out):
    order = [("e4_pd_N1753", "path_distance 1,753"), ("e4_F_N1753", "RAG F 1,753"), ("e4_pd_N12000", "path_distance 12,000"),
             ("e4_F_N12000", "RAG F 12,000"), ("e4_pd_16k", "path_distance 16K"), ("e4_F_16k", "RAG F 16K")]
    classes = [("right", ("right", "right_self_contained"), BLUE, "#ffffff"),
               ("wrong, needs nothing from other files", ("wrong_self_contained",), ORANGE, "#ffffff"),
               ("wrong, needed names visible", ("reasoning",), AQUA, "#ffffff"),
               ("wrong, needed name missing", ("retrieval",), YELLOW, "#0b0b0b")]
    counts = {arm: {} for arm, _ in order}
    with open(split_path) as f:
        for line in f:
            r = json.loads(line)
            if r["arm"] in counts:
                counts[r["arm"]][r["cls"]] = counts[r["arm"]].get(r["cls"], 0) + 1
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    y = np.arange(len(order))[::-1]
    for yi, (arm, label) in zip(y, order):
        total = sum(counts[arm].values())
        left = 0
        for name, keys, color, text_color in classes:
            share = sum(counts[arm].get(k, 0) for k in keys) / total
            ax.barh(yi, share, left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.62, label=name if yi == y[0] else None)
            if name.startswith("wrong, needed name missing"):
                ax.text(1.012, yi, f"{share * 100:.1f}%", ha="left", va="center", fontsize=8.5, color=INK2)  # too thin to label inside
            else:
                ax.text(left + share / 2, yi, f"{share * 100:.0f}%", ha="center", va="center", fontsize=8.5, color=text_color)
            left += share
    ax.text(1.012, y[0] + 0.52, "needed name\nmissing", ha="left", va="bottom", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [label for _, label in order])
    ax.set_xlim(0, 1.13)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("share of the 8,676 lines")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=8.5)
    ax.set_title("Most wrong lines are not retrieval failures", loc="left", fontsize=11, color=INK)
    save(fig, out, "fig_failure_split")


def results_table(arms, out):
    rows = []
    for name, approach, budget in [("e4_zero", "no context", "-")] + [
            (prefix + key, approach, label) for label, key in SIZES
            for approach, prefix in (("path_distance", "e4_pd_"), ("RAG F", "e4_F_"))]:
        a = arms[name]
        lo, hi = bootstrap_ci(a["em"])
        c = a["cost"]
        rows.append({
            "arm": name, "approach": approach, "budget_N": budget, "em_all": round(a["em"].mean(), 4),
            "em_ci_low": round(lo, 4), "em_ci_high": round(hi, 4), "es_all": round(a["es_all"], 2),
            **{f"em_{k}": round(a["per_category"][k], 4) for k in CATEGORIES},
            "input_tokens_per_line": round(c["input_tokens_mean"]), "gen_p50_s": round(c["gen_ms_p50"] / 1000, 2),
            "gen_p95_s": round(c["gen_ms_p95"] / 1000, 2), "line_p50_s": round(a["p50"], 2), "line_p95_s": round(a["p95"], 2),
            "context_ms_mean": round(c["context_ms_mean"], 1), "truncated_share": round(c["share_truncated"], 4),
            "wall_hours": round(c["wall_seconds_per_line"] * 8676 / 3600, 2), "arm_cost_usd": round(c["arm_cost_usd"], 2),
        })
    with open(os.path.join(out, "e4_results_table.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cols = ["approach", "budget_N", "em_all", "es_all", "input_tokens_per_line", "line_p50_s", "line_p95_s", "truncated_share", "arm_cost_usd"]
    with open(os.path.join(out, "e4_results_table.md"), "w") as f:
        f.write("| approach | N | EM (95% CI) | ES | input tokens/line | p50 s/line | p95 s/line | truncated | cost USD |\n|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['approach']} | {r['budget_N']} | {r['em_all']:.3f} [{r['em_ci_low']:.3f}, {r['em_ci_high']:.3f}] | {r['es_all']:.1f} | "
                    f"{r['input_tokens_per_line']:,} | {r['line_p50_s']:.2f} | {r['line_p95_s']:.2f} | {r['truncated_share'] * 100:.1f}% | {r['arm_cost_usd']:.2f} |\n")
    with open(os.path.join(out, "e4_results_table.tex"), "w") as f:
        f.write("\\begin{tabular}{llrrrrrr}\n\\toprule\nApproach & $N$ & EM [95\\% CI] & ES & Tokens/line & p50 s & p95 s & Cost \\$ \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['approach'].replace('_', chr(92) + '_')} & {r['budget_N']} & {r['em_all']:.3f} [{r['em_ci_low']:.3f}, {r['em_ci_high']:.3f}] & "
                    f"{r['es_all']:.1f} & {r['input_tokens_per_line']:,} & {r['line_p50_s']:.2f} & {r['line_p95_s']:.2f} & {r['arm_cost_usd']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", default=os.path.expanduser("~/lca_remote/E4/codellama-7b/medium_context/arms"))
    parser.add_argument("--split", default=os.path.expanduser("~/lca_remote/E4/failure_split.jsonl"))
    parser.add_argument("--out", default=os.path.join(here, "..", "figures", "E4"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = ["e4_zero"] + [p + key for _, key in SIZES for p in ("e4_pd_", "e4_F_")]
    arms = {n: load_arm(args.arms, n) for n in names}
    fig_accuracy(arms, args.out)
    fig_cost(arms, args.out)
    fig_latency(arms, args.out)
    fig_failures(args.split, args.out)
    for r in results_table(arms, args.out):
        print(r["arm"], r["em_all"], r["arm_cost_usd"], r["line_p50_s"], r["line_p95_s"])


if __name__ == "__main__":
    main()
