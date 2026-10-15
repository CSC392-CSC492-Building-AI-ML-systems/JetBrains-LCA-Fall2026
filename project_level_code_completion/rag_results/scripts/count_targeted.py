"""Count targeted (T) lines on development datapoints only. usage: count_targeted.py <dataset> <dev_size> <dev_extra>"""
import os
import sys
from collections import Counter

from hydra import compose, initialize_config_dir

from eval.line_generators import dev_indices
from eval.preprocessors import load_datapoints
from eval.proxy_metrics import targeted_lines

dataset, dev_size, dev_extra = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with initialize_config_dir(config_dir=os.path.join(os.environ["PROJECT"], "eval", "config"), version_base=None):
    cfg = compose(config_name="compare_arms", overrides=[f"dataset={dataset}"])
data = load_datapoints(cfg.dataset)
idx = dev_indices(len(data), dev_size, dev_extra)
print(f"{dataset}: {len(data)} datapoints; dev = {len(idx)} (the other {len(data) - len(idx)} are not inspected)")

total, targeted, dps = Counter(), Counter(), 0
per_dp = []
for i in idx:  # development datapoints only
    dp = data[i]
    t = targeted_lines(dp)
    n_t = sum(len(v) for v in t.values())
    dps += n_t > 0
    per_dp.append(n_t)
    for cls, nums in dp.completion_lines.items():
        total[cls] += len(nums)
        targeted[cls] += len(t[cls])
print(f"{'category':16s}{'lines':>8s}{'targeted':>10s}")
for cls in sorted(total):
    print(f"{cls:16s}{total[cls]:8d}{targeted[cls]:10d}")
print(f"{'ALL':16s}{sum(total.values()):8d}{sum(targeted.values()):10d}")
print(f"datapoints with at least one targeted line: {dps} of {len(idx)}; per-datapoint max {max(per_dp)}, "
      f"median {sorted(per_dp)[len(per_dp) // 2]}")
