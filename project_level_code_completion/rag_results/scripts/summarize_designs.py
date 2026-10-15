"""Compact comparison of designs against a reference arm on T_dev (small + medium), with the promotion rule applied.

usage: summarize_designs.py <log> <ref_arm> <arm> [<arm> ...]
"""
import json
import os
import sys

import numpy as np

from eval.analyze_arms import arm_lines, datapoint_means, paired_bootstrap

home = os.path.expanduser("~")
S = f"{home}/lca_artifacts/python/starcoderbase-1b/small_context/arms"
M = f"{home}/lca_artifacts/python/starcoderbase-1b/medium_context/arms"
log_path, ref, arms = sys.argv[1], sys.argv[2], sys.argv[3:]

log = open(os.path.expanduser(log_path), errors="replace").read()
print("log: runs with exit=0:", log.count("exit=0"), "| non-zero exits:", sum(1 for l in log.splitlines()
      if "=====" in l and "exit=" in l and "exit=0" not in l), "| tracebacks:", log.count("Traceback"))


def stats(arm, bases, metric, category=None):
    ref_m = datapoint_means(arm_lines(bases, ref), metric, category)
    new_m = datapoint_means(arm_lines(bases, arm), metric, category)
    result = paired_bootstrap(ref_m, new_m)
    return (np.mean(list(new_m.values())), *result[:3]) if result else (float("nan"),) * 4


print(f"\nreference: {ref}; EM on T lines; delta = arm - reference, [95% CI over datapoints]; * = CI excludes 0\n")
print(f"{'arm':14s}{'T-EM':>7s}  {'pooled delta':>26s}  {'medium delta':>24s}  {'inproject delta':>26s}  {'ES delta':>20s}  rule")
for arm in arms:
    em, d, lo, hi = stats(arm, [S, M], "em")
    _, dm, lom, him = stats(arm, [M], "em")
    _, di, loi, hii = stats(arm, [S, M], "em", "inproject")
    _, de, loe, hie = stats(arm, [S, M], "es")
    star = lambda lo_, hi_: "*" if lo_ > 0 or hi_ < 0 else " "
    promoted = d > 0 and lo > -0.02
    print(f"{arm:14s}{em:7.3f}  {d:+7.3f} [{lo:+.3f},{hi:+.3f}]{star(lo, hi)}  "
          f"{dm:+7.3f} [{lom:+.3f},{him:+.3f}]{star(lom, him)}  {di:+7.3f} [{loi:+.3f},{hii:+.3f}]{star(loi, hii)}  "
          f"{de:+6.2f} [{loe:+.1f},{hie:+.1f}]{star(loe, hie)}  {'PROMOTE' if promoted else '-'}")
