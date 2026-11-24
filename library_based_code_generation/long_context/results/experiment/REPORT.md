# API-list context experiment

Matched examples: 150/150; full split: True.

| Arm | ChrF | API Recall |
| --- | ---: | ---: |
| plain | 0.453374 | 0.359943 |
| bm25_20 | 0.455836 | 0.397102 |
| direct_8k | 0.460572 | 0.439834 |
| full_list | 0.457716 | 0.446413 |
| bm25_8k | 0.457057 | 0.470067 |
| random_8k | 0.456570 | 0.444789 |
| bm25_20_repeat | 0.456040 | 0.393204 |

| Contrast | Primary | API Recall difference [95% CI] |
| --- | --- | --- |
| full_list_minus_direct_8k | True | +0.006579 [-0.009957, +0.023538] |
| bm25_8k_minus_direct_8k | True | +0.030232 [+0.000765, +0.061612] |
| bm25_8k_minus_random_8k | True | +0.025278 [+0.000264, +0.049166] |
| bm25_20_repeat_minus_bm25_20 | False | -0.003898 [-0.018878, +0.009837] |
| full_list_minus_bm25_20 | False | +0.049311 [+0.016322, +0.082575] |
| bm25_20_minus_plain | False | +0.037160 [+0.004570, +0.065161] |

Bootstrap unit: repository. Intervals are descriptive, unadjusted for multiplicity.
A null result is not equivalence. Read extra-call diagnostics, repeat variation and context strata in comparison.json.
Calls outside the inventory are not automatically hallucinations. No execution correctness is measured.
