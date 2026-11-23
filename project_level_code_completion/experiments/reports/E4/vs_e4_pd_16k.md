Reference arm: `e4_pd_16k`. Lines per arm: {'e4_pd_16k': 8676, 'e4_F_16k': 8676}. 10000 bootstrap resamples over datapoints.

### EM: mean per arm, and difference to `e4_pd_16k` [95% CI]

| category (datapoints) | e4_pd_16k | e4_F_16k | Δ vs ref |
|---|---:|---:|---:|
| all (224) | 0.492 | 0.537 | +0.045 [+0.027, +0.064] * |
| commited (89) | 0.420 | 0.435 | +0.015 [-0.042, +0.070] |
| common (127) | 0.439 | 0.512 | +0.074 [+0.030, +0.120] * |
| infile (183) | 0.429 | 0.486 | +0.057 [+0.034, +0.082] * |
| inproject (150) | 0.484 | 0.535 | +0.051 [+0.023, +0.083] * |
| non_informative (224) | 0.447 | 0.465 | +0.018 [-0.007, +0.044] |
| random (224) | 0.579 | 0.629 | +0.051 [+0.030, +0.073] * |

### ES: mean per arm, and difference to `e4_pd_16k` [95% CI]

| category (datapoints) | e4_pd_16k | e4_F_16k | Δ vs ref |
|---|---:|---:|---:|
| all (224) | 79.107 | 80.843 | +1.736 [+0.970, +2.592] * |
| commited (89) | 75.253 | 75.461 | +0.208 [-2.721, +2.844] |
| common (127) | 77.797 | 80.770 | +2.973 [+1.293, +4.843] * |
| infile (183) | 77.799 | 79.873 | +2.074 [+1.018, +3.211] * |
| inproject (150) | 79.453 | 81.718 | +2.265 [+1.001, +3.631] * |
| non_informative (224) | 72.733 | 72.947 | +0.214 [-1.157, +1.632] |
| random (224) | 82.472 | 84.535 | +2.062 [+1.131, +3.102] * |

`*` = the 95% interval excludes 0. Categories with few datapoints have wide intervals.
