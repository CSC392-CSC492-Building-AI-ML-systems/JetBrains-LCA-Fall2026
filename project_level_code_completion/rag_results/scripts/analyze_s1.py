"""S1: every static composer vs path_distance on the development data (overall EM, benchmark style)."""
import os

import numpy as np

from eval.analyze_arms import arm_lines, datapoint_means, paired_bootstrap

home = os.path.expanduser("~")
base = f"{home}/lca_artifacts/python/starcoderbase-1b"
SETS = {
    "MEDIUM dev (56 datapoints, 2,013 lines)": (f"{base}/medium_context/arms", "g_path_distance",
        ["s1_zero", "s1_naive", "s1_alphabetical", "s1_file_length", "s1_half_memory", "s1_half_memory_path_distance",
         "s1_function_class_mask_half", "s1_function_class_mask_one", "s1_imports_first", "g_v1", "g_F"]),
    "SMALL dev (48 datapoints, 1,573 lines)": (f"{base}/small_context/arms", "r0_path_distance",
        ["r0_zero", "s1_naive", "s1_alphabetical", "s1_file_length", "s1_half_memory", "s1_half_memory_path_distance",
         "s1_function_class_mask_half", "s1_function_class_mask_one", "r0_imports_first", "r0_v1", "g_F"]),
}
for title, (directory, ref, arms) in SETS.items():
    print(f"\n=================== {title}: reference {ref} ===================")
    ref_all = datapoint_means(arm_lines(directory, ref), "em", None)
    ref_es = datapoint_means(arm_lines(directory, ref), "es", None)
    print(f"{'arm':30s}{'EM':>7s}{'  Δ vs path_distance [95% CI]':>34s}{'ES':>8s}{'inproject':>11s}{'infile':>8s}{'commited':>10s}")
    rows = [(ref, "em")] + [(a, "em") for a in arms]
    for arm in [ref] + arms:
        lines = arm_lines(directory, arm)
        m = datapoint_means(lines, "em", None)
        es = np.mean(list(datapoint_means(lines, "es", None).values()))
        cats = {c: datapoint_means(lines, "em", c) for c in ("inproject", "infile", "commited")}
        cat_vals = {c: (np.mean(list(v.values())) if v else float("nan")) for c, v in cats.items()}
        if arm == ref:
            delta = "(reference)"
        else:
            d, lo, hi, _ = paired_bootstrap(ref_all, m)
            delta = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if lo > 0 or hi < 0 else ''}"
        print(f"{arm:30s}{np.mean(list(m.values())):7.3f}{delta:>34s}{es:8.2f}{cat_vals['inproject']:11.3f}"
              f"{cat_vals['infile']:8.3f}{cat_vals['commited']:10.3f}")
