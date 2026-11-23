Reference arm: `h_v1`. Lines per arm: {'h_v1': 2614, 'h_F': 2614}. 10000 bootstrap resamples over datapoints.

### EM: mean per arm, and difference to `h_v1` [95% CI]

| category (datapoints) | h_v1 | h_F | Δ vs ref |
|---|---:|---:|---:|
| all (162) | 0.395 | 0.398 | +0.003 [-0.016, +0.022] |
| commited (27) | 0.342 | 0.332 | -0.011 [-0.033, +0.003] |
| common (42) | 0.433 | 0.440 | +0.006 [-0.019, +0.032] |
| infile (57) | 0.404 | 0.447 | +0.043 [-0.000, +0.099] |
| inproject (120) | 0.420 | 0.417 | -0.004 [-0.024, +0.010] |
| non_informative (68) | 0.297 | 0.349 | +0.053 [-0.001, +0.119] |
| random (35) | 0.531 | 0.502 | -0.029 [-0.086, +0.000] |

### ES: mean per arm, and difference to `h_v1` [95% CI]

| category (datapoints) | h_v1 | h_F | Δ vs ref |
|---|---:|---:|---:|
| all (162) | 71.553 | 71.423 | -0.130 [-1.380, +0.929] |
| commited (27) | 72.192 | 71.417 | -0.775 [-2.079, +0.389] |
| common (42) | 73.815 | 72.496 | -1.319 [-5.262, +1.254] |
| infile (57) | 66.994 | 69.874 | +2.880 [+0.289, +6.601] * |
| inproject (120) | 72.339 | 72.030 | -0.309 [-2.189, +1.266] |
| non_informative (68) | 67.252 | 68.504 | +1.251 [-1.131, +4.129] |
| random (35) | 80.895 | 78.276 | -2.619 [-9.533, +2.276] |

`*` = the 95% interval excludes 0. Categories with few datapoints have wide intervals.
