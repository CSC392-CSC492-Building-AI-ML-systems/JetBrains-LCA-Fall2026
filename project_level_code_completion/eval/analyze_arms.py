"""Compare arms from their `generation_results.jsonl` with paired bootstrap confidence intervals.

Metrics follow the benchmark: a datapoint's score is the mean over its lines; "all" is the mean over datapoints, and a
category is the mean over the datapoints that have lines of that category. Uncertainty comes from resampling
datapoints with replacement (the same resample for both arms of a comparison), because lines inside one datapoint are
correlated.

Example:
    python -m eval.analyze_arms --base ~/lca_artifacts/python/starcoderbase-1b/small_context/arms \
        --ref dev_path_distance --arms dev_zero dev_rag --out report.md
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Optional

import numpy as np
from thefuzz import fuzz

METRICS = ('em', 'es')


def load_lines(path: str, zero_context: bool) -> dict[tuple, dict]:
    """Rows of one pass keyed by (datapoint_idx, line_class, k-th line of that class in the datapoint)."""
    rows, seen = {}, defaultdict(int)
    with open(path) as f:
        for raw in f:
            row = json.loads(raw)
            if bool(row['zero_context']) != zero_context:
                continue
            base = (row['datapoint_idx'], row['line_class'])
            rows[base + (seen[base],)] = {
                'em': float(row['prediction_line'].strip() == row['ground_truth'].strip()),
                'es': float(fuzz.ratio(row['prediction_line'], row['ground_truth'])),
            }
            seen[base] += 1
    return rows


def arm_lines(base: str, arm: str) -> dict[tuple, dict]:
    """Lines of the arm's own pass: the context pass if it has one, otherwise (zero-context arm) the zero pass."""
    path = os.path.join(base, arm, 'generation_results.jsonl')
    return load_lines(path, zero_context=False) or load_lines(path, zero_context=True)


def datapoint_means(lines: dict[tuple, dict], metric: str, category: Optional[str]) -> dict[int, float]:
    """Per-datapoint mean of `metric` over the lines of `category` (all lines if None)."""
    per_dp = defaultdict(list)
    for (dp, cls, _), values in lines.items():
        if category is None or cls == category:
            per_dp[dp].append(values[metric])
    return {dp: float(np.mean(v)) for dp, v in per_dp.items()}


def paired_bootstrap(ref: dict[int, float], other: dict[int, float], n_boot: int = 10_000, seed: int = 0):
    """Mean difference (other - ref) over datapoints present in both, with a 95% bootstrap CI."""
    ids = sorted(set(ref) & set(other))
    if not ids:
        return None
    diff = np.array([other[i] - ref[i] for i in ids])
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high), len(ids)


def report(base: str, ref: str, arms: list[str], n_boot: int = 10_000) -> str:
    data = {name: arm_lines(base, name) for name in [ref] + arms}
    categories = sorted({cls for lines in data.values() for (_, cls, _) in lines})
    n_lines = {name: len(lines) for name, lines in data.items()}
    out = [f"Reference arm: `{ref}`. Lines per arm: {n_lines}. {n_boot} bootstrap resamples over datapoints.\n"]
    for metric in METRICS:
        out.append(f"### {metric.upper()}: mean per arm, and difference to `{ref}` [95% CI]\n")
        out.append("| category (datapoints) | " + f"{ref} | " + " | ".join(f"{a} | Δ vs ref" for a in arms) + " |")
        out.append("|---|---:|" + "---:|---:|" * len(arms))
        for category in [None] + categories:
            ref_means = datapoint_means(data[ref], metric, category)
            cells = [f"{np.mean(list(ref_means.values())):.3f}" if ref_means else "n/a"]
            for arm in arms:
                means = datapoint_means(data[arm], metric, category)
                result = paired_bootstrap(ref_means, means, n_boot)
                if not means or result is None:
                    cells += ["n/a", "n/a"]
                    continue
                delta, low, high, _ = result
                sig = " *" if low > 0 or high < 0 else ""
                cells += [f"{np.mean(list(means.values())):.3f}", f"{delta:+.3f} [{low:+.3f}, {high:+.3f}]{sig}"]
            n_dp = len(ref_means)
            out.append(f"| {category or 'all'} ({n_dp}) | " + " | ".join(cells) + " |")
        out.append("")
    out.append("`*` = the 95% interval excludes 0. Categories with few datapoints have wide intervals.")
    return "\n".join(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True, help='directory containing the arm directories')
    parser.add_argument('--ref', required=True, help='reference arm directory name')
    parser.add_argument('--arms', nargs='+', required=True, help='arm directory names to compare with the reference')
    parser.add_argument('--boot', type=int, default=10_000)
    parser.add_argument('--out', help='also write the report to this file')
    args = parser.parse_args()
    text = report(os.path.expanduser(args.base), args.ref, args.arms, args.boot)
    print(text)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(text + "\n")
