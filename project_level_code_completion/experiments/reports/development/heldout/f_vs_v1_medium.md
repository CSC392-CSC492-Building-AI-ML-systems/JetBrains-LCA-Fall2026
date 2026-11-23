Reference arm: `h_v1`. Lines per arm: {'h_v1': 6663, 'h_F': 6663}. 10000 bootstrap resamples over datapoints.

### EM: mean per arm, and difference to `h_v1` [95% CI]

| category (datapoints) | h_v1 | h_F | Δ vs ref |
|---|---:|---:|---:|
| all (168) | 0.414 | 0.421 | +0.008 [+0.002, +0.014] * |
| commited (63) | 0.400 | 0.400 | -0.000 [-0.007, +0.005] |
| common (95) | 0.374 | 0.400 | +0.026 [-0.004, +0.061] |
| infile (143) | 0.345 | 0.345 | -0.000 [-0.019, +0.014] |
| inproject (114) | 0.440 | 0.436 | -0.004 [-0.025, +0.011] |
| non_informative (168) | 0.350 | 0.367 | +0.017 [+0.002, +0.033] * |
| random (168) | 0.504 | 0.509 | +0.005 [-0.005, +0.015] |

### ES: mean per arm, and difference to `h_v1` [95% CI]

| category (datapoints) | h_v1 | h_F | Δ vs ref |
|---|---:|---:|---:|
| all (168) | 72.222 | 72.869 | +0.647 [+0.176, +1.154] * |
| commited (63) | 70.403 | 71.261 | +0.858 [-0.438, +2.417] |
| common (95) | 70.532 | 72.432 | +1.901 [-0.340, +4.814] |
| infile (143) | 69.256 | 70.003 | +0.747 [-0.238, +1.852] |
| inproject (114) | 73.105 | 73.244 | +0.140 [-1.618, +1.547] |
| non_informative (168) | 62.621 | 63.826 | +1.205 [+0.105, +2.422] * |
| random (168) | 76.964 | 77.243 | +0.279 [-0.417, +0.972] |

`*` = the 95% interval excludes 0. Categories with few datapoints have wide intervals.
