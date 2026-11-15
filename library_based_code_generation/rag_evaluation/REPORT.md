# LBCG saved RAG evaluation

Common scored examples: 150/150. Complete: True.

Legacy provenance: row order is assumed; generation dataset and prompt identity cannot be verified.

| Arm | ChrF | API Recall |
| --- | ---: | ---: |
| plain | 0.455679 | 0.367891 |
| bm25 | 0.452718 | 0.385235 |
| dense | 0.453783 | 0.374898 |
| hybrid | 0.451920 | 0.385701 |
| hybrid_normalized | 0.450321 | 0.382718 |

| Paired comparison | Metric | Difference | 95% CI | Wins/losses/ties |
| --- | --- | ---: | --- | --- |
| plain_minus_bm25 | ChrF | +0.002960 | [-0.003676, +0.009947] | 77/73/0 |
| plain_minus_bm25 | API_recall | -0.017344 | [-0.041533, +0.006294] | 41/55/54 |
| bm25_minus_plain | ChrF | -0.002960 | [-0.009940, +0.003677] | 73/77/0 |
| bm25_minus_plain | API_recall | +0.017344 | [-0.006254, +0.041540] | 55/41/54 |
| dense_minus_plain | ChrF | -0.001896 | [-0.010516, +0.005737] | 81/69/0 |
| dense_minus_plain | API_recall | +0.007007 | [-0.015592, +0.028900] | 54/38/58 |
| dense_minus_bm25 | ChrF | +0.001064 | [-0.008104, +0.009580] | 78/72/0 |
| dense_minus_bm25 | API_recall | -0.010337 | [-0.033658, +0.012517] | 52/52/46 |
| hybrid_minus_plain | ChrF | -0.003759 | [-0.012367, +0.003873] | 85/65/0 |
| hybrid_minus_plain | API_recall | +0.017810 | [-0.007212, +0.042465] | 56/42/52 |
| hybrid_minus_bm25 | ChrF | -0.000799 | [-0.009507, +0.007030] | 84/66/0 |
| hybrid_minus_bm25 | API_recall | +0.000466 | [-0.018429, +0.019110] | 49/40/61 |
| hybrid_normalized_minus_plain | ChrF | -0.005358 | [-0.014708, +0.003135] | 82/68/0 |
| hybrid_normalized_minus_plain | API_recall | +0.014827 | [-0.009092, +0.039171] | 52/39/59 |
| hybrid_normalized_minus_bm25 | ChrF | -0.002398 | [-0.010420, +0.005396] | 76/74/0 |
| hybrid_normalized_minus_bm25 | API_recall | -0.002517 | [-0.022260, +0.017100] | 47/45/58 |

Bootstrap unit: individual example; library clustering not accounted for. Intervals are exploratory and unadjusted for multiple comparisons.

Syntax validity is a diagnostic, not execution correctness. Review cases are selected for manual inspection, not classified causes.

Provider tokens, latency, truncation and cost are unavailable in the legacy prediction artifacts.
