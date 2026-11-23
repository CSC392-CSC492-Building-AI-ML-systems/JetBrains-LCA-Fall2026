Reference arm: `g_v1`. Lines per arm: {'g_v1': 2013, 'g_F': 2013}. 10000 bootstrap resamples over datapoints.

### EM: mean per arm, and difference to `g_v1` [95% CI]

| category (datapoints) | g_v1 | g_F | Δ vs ref |
|---|---:|---:|---:|
| all (56) | 0.475 | 0.485 | +0.010 [-0.000, +0.020] |
| commited (26) | 0.358 | 0.409 | +0.051 [-0.004, +0.141] |
| common (32) | 0.316 | 0.329 | +0.013 [-0.048, +0.092] |
| infile (40) | 0.453 | 0.475 | +0.021 [-0.006, +0.055] |
| inproject (36) | 0.465 | 0.466 | +0.000 [-0.009, +0.009] |
| non_informative (56) | 0.375 | 0.391 | +0.016 [-0.008, +0.043] |
| random (56) | 0.596 | 0.605 | +0.009 [-0.010, +0.028] |

### ES: mean per arm, and difference to `g_v1` [95% CI]

| category (datapoints) | g_v1 | g_F | Δ vs ref |
|---|---:|---:|---:|
| all (56) | 74.592 | 74.979 | +0.387 [-0.315, +1.096] |
| commited (26) | 73.414 | 74.224 | +0.810 [-2.424, +4.076] |
| common (32) | 64.457 | 65.940 | +1.483 [-4.277, +8.699] |
| infile (40) | 75.141 | 76.355 | +1.214 [+0.342, +2.274] * |
| inproject (36) | 75.609 | 75.068 | -0.541 [-2.419, +1.339] |
| non_informative (56) | 64.206 | 65.456 | +1.250 [-0.315, +3.173] |
| random (56) | 80.411 | 81.400 | +0.989 [+0.200, +1.871] * |

`*` = the 95% interval excludes 0. Categories with few datapoints have wide intervals.
