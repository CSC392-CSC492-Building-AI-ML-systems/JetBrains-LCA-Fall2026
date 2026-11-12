"""Analysis of the full E5 agent arm against E4's arms (RAG_EXPERIMENT_LOG sections 8.87 and 8.89).

The analyses were fixed in 8.87 before any shard ran: headline paired bootstrap over datapoints (the E4 procedure, same
resamples for both arms), changed / fallback shares, changed-lines-only paired counts, the headline without fallback lines,
the headline over the 216 datapoints not used for design, results by line class, wrong lines, tokens, time and cost.

Usage (run from project_level_code_completion, with the repository's python):
    python rag_results/scripts/analyze_e5_full.py --base DIR [--out report.md]
DIR holds the arm folders e4_zero, e4_pd_N1753, e4_F_N1753 and e5_agent_full (each with generation_results.jsonl).
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from thefuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.analyze_arms import paired_bootstrap  # noqa: E402

DESIGN_DATAPOINTS = [i * 224 // 8 for i in range(8)]  # the 8 development datapoints of 8.84 / 8.86 (original indices)
ARMS = {"zero-context": "e4_zero", "path_distance": "e4_pd_N1753", "RAG F": "e4_F_N1753", "agent (v2)": "e5_agent_full"}


def load(base, arm):
    rows = {}
    with open(os.path.join(base, arm, "generation_results.jsonl"), encoding="utf8") as f:
        for raw in f:
            row = json.loads(raw)
            rows[(row["datapoint_idx"], row["line_num"])] = row
    return rows


def right(row):
    return row["prediction_line"].strip() == row["ground_truth"].strip()


def dp_means(rows, keys, fn, category=None):
    per = defaultdict(list)
    for k in keys:
        if category is None or rows[k]["line_class"] == category:
            per[k[0]].append(fn(rows[k]))
    return {dp: float(np.mean(v)) for dp, v in per.items()}


def em(row):
    return float(right(row))


def es(row):
    return float(fuzz.ratio(row["prediction_line"], row["ground_truth"]))


def delta(ref, other, label):
    r = paired_bootstrap(ref, other)
    d, lo, hi, n = r
    star = " *" if lo > 0 or hi < 0 else ""
    return f"{label}: {100 * d:+.2f} pts [{100 * lo:+.2f}, {100 * hi:+.2f}]{star} over {n} datapoints"


def seconds(row):
    return (row["metrics"]["context_ms"] + row["metrics"]["gen_ms"]) / 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    base = os.path.expanduser(args.base)
    data = {name: load(base, folder) for name, folder in ARMS.items()}
    keys = sorted(data["agent (v2)"])
    for name, rows in data.items():
        assert sorted(rows) == keys, f"{name} covers different lines than the agent arm"
    rag, agent = data["RAG F"], data["agent (v2)"]
    out = []
    w = out.append

    w(f"# E5 full arm: agent (v2) against E4 arms at N=1,753 ({len(keys)} lines, {len({k[0] for k in keys})} datapoints)\n")

    w("## 1. Headline (EM = mean over datapoints, as in the E4 table; ES likewise; wrong lines = lines not exactly right)\n")
    w("| arm | EM | ES | EM pooled over lines | wrong lines | writer input tokens/line | p50 s/line | p95 s/line | truncated |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, rows in data.items():
        em_dp = dp_means(rows, keys, em)
        es_dp = dp_means(rows, keys, es)
        t = np.array([seconds(rows[k]) for k in keys])
        w(f"| {name} | {np.mean(list(em_dp.values())):.4f} | {np.mean(list(es_dp.values())):.2f} | "
          f"{np.mean([em(rows[k]) for k in keys]):.4f} | {sum(not right(rows[k]) for k in keys)} | "
          f"{np.mean([rows[k]['metrics']['input_tokens'] for k in keys]):.0f} | {np.percentile(t, 50):.2f} | {np.percentile(t, 95):.2f} | "
          f"{100 * np.mean([bool(rows[k]['metrics']['truncated']) for k in keys]):.1f}% |")
    w("")
    ref = dp_means(rag, keys, em)
    for name in ("agent (v2)", "path_distance", "zero-context"):
        w("- " + delta(ref, dp_means(data[name], keys, em), f"EM {name} minus RAG F"))
    w("- " + delta(dp_means(data["path_distance"], keys, em), dp_means(agent, keys, em), "EM agent minus path_distance"))
    d_es, lo_es, hi_es, n_es = paired_bootstrap(dp_means(rag, keys, es), dp_means(agent, keys, es))  # ES is already on a 0-100 scale
    w(f"- ES agent minus RAG F: {d_es:+.2f} ES points [{lo_es:+.2f}, {hi_es:+.2f}]{' *' if lo_es > 0 or hi_es < 0 else ''} over {n_es} datapoints")
    pooled = (np.mean([em(agent[k]) for k in keys]) - np.mean([em(rag[k]) for k in keys])) * 100
    w(f"- EM pooled over lines, agent minus RAG F: {pooled:+.2f} pts")
    per_dp_a, per_dp_r = dp_means(agent, keys, em), ref
    better = sum(per_dp_a[d] > per_dp_r[d] for d in per_dp_r)
    worse = sum(per_dp_a[d] < per_dp_r[d] for d in per_dp_r)
    w(f"- datapoints where the agent is better / worse / tied: {better} / {worse} / {len(per_dp_r) - better - worse}\n")

    w("## 2. What the planner did\n")
    changed = [k for k in keys if agent[k]["retrieval"]["changed"]]
    fallback = [k for k in keys if agent[k]["retrieval"]["fallback"]]
    w(f"- changed the context on {len(changed)} of {len(keys)} lines ({100 * len(changed) / len(keys):.1f}%); fallback to RAG's prompt on {len(fallback)} ({100 * len(fallback) / len(keys):.2f}%)")
    w("- planner status: " + ", ".join(f"{k} {v}" for k, v in Counter(agent[k]["retrieval"]["planner_status"] for k in keys).most_common()))
    w(f"- steps per line {np.mean([agent[k]['retrieval']['steps'] for k in keys]):.2f}, tool calls per line {np.mean([agent[k]['retrieval']['tool_calls'] for k in keys]):.2f}")
    tok_a = np.mean([agent[k]["metrics"]["total_tokens"] for k in keys])
    tok_r = np.mean([rag[k]["metrics"]["input_tokens"] + rag[k]["metrics"]["output_tokens"] for k in keys])
    w(f"- total tokens per line (planner + writer): agent {tok_a:.0f} vs RAG F {tok_r:.0f} = {tok_a / tok_r:.1f}x")
    t_a, t_r = np.array([seconds(agent[k]) for k in keys]), np.array([seconds(rag[k]) for k in keys])
    w(f"- seconds per line (context + generation, planner time is inside context): agent mean {t_a.mean():.2f}, RAG F mean {t_r.mean():.2f}")
    w("")

    w("## 3. Lines the planner changed (paired counts; unchanged lines carry the same prompt as RAG F)\n")
    both = sum(right(agent[k]) and right(rag[k]) for k in changed)
    only_a = sum(right(agent[k]) and not right(rag[k]) for k in changed)
    only_r = sum(right(rag[k]) and not right(agent[k]) for k in changed)
    neither = len(changed) - both - only_a - only_r
    w(f"- changed lines: {len(changed)}; both right {both}, only agent right {only_a}, only RAG F right {only_r}, neither {neither}")
    w(f"- EM on changed lines: agent {(both + only_a) / len(changed):.4f}, RAG F {(both + only_r) / len(changed):.4f}; net change {only_a - only_r:+d} lines ({100 * (only_a - only_r) / len(keys):+.2f} pts of all lines)")
    same_pred = sum(agent[k]["prediction_line"] == rag[k]["prediction_line"] for k in changed)
    w(f"- of the changed lines, {same_pred} still got the same prediction as RAG F")
    unchanged = [k for k in keys if k not in set(changed)]
    diff_unchanged = [k for k in unchanged if agent[k]["prediction_line"] != rag[k]["prediction_line"]]
    w(f"- unchanged lines with a different prediction than RAG F: {len(diff_unchanged)} of {len(unchanged)}")
    for k in diff_unchanged:
        w(f"  - datapoint {k[0]} line {k[1]} ({agent[k]['line_class']}): agent `{agent[k]['prediction_line'].strip()[:90]}` | RAG F `{rag[k]['prediction_line'].strip()[:90]}` | truth `{agent[k]['ground_truth'].strip()[:90]}`; "
          f"writer input tokens {agent[k]['metrics']['input_tokens']} vs {rag[k]['metrics']['input_tokens']}")
    w("")

    w("## 4. Robustness (fixed in 8.87)\n")
    fb = set(fallback)
    keep = [k for k in keys if k not in fb]
    w("- without fallback lines (dropped from both arms, " + f"{len(fb)} lines): " + delta(dp_means(rag, keep, em), dp_means(agent, keep, em), "EM agent minus RAG F"))
    held = [k for k in keys if k[0] not in set(DESIGN_DATAPOINTS)]
    w(f"- over the 216 datapoints not used for design (the 8 of 8.84/8.86 left out; {len(held)} lines): " + delta(dp_means(rag, held, em), dp_means(agent, held, em), "EM agent minus RAG F"))
    w("")

    w("## 5. By line class (EM, mean over the datapoints that have the class; agent minus RAG F, 95% CI)\n")
    w("| class | datapoints | lines | RAG F | agent | difference |")
    w("|---|---:|---:|---:|---:|---|")
    for cls in sorted({agent[k]["line_class"] for k in keys}):
        r_ = dp_means(rag, keys, em, cls)
        a_ = dp_means(agent, keys, em, cls)
        n_lines = sum(agent[k]["line_class"] == cls for k in keys)
        d, lo, hi, n = paired_bootstrap(r_, a_)
        star = " *" if lo > 0 or hi < 0 else ""
        w(f"| {cls} | {n} | {n_lines} | {np.mean(list(r_.values())):.3f} | {np.mean(list(a_.values())):.3f} | {100 * d:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]{star} |")
    w("\n`*` = the 95% interval excludes 0 (resampling datapoints, 10,000 draws, seed 0, same resample for both arms).")

    text = "\n".join(out)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
