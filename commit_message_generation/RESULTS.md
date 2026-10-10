# CMG baseline results

Completed on 2026-10-08: CodeT5 on CPU, **163 commits per run**, using
seeds `2687987020`, `42`, and `123`. All three runs and all eight metrics completed.
The original model, preprocessing, generation settings, and evaluator were reused.

## Scores

These are the **average of the three separately evaluated runs**.
Standard deviation shows how much the scores varied between runs.

| Metric | Paper mean | Our mean | Our standard deviation |
| --- | ---: | ---: | ---: |
| BLEU | 0.355 | 0.359595 | 0.068959 |
| chrF | 11.862 | 11.855617 | 0.885535 |
| ROUGE-1 | 13.615 | 13.355448 | 0.696354 |
| ROUGE-2 | 2.633 | 2.310806 | 0.166366 |
| ROUGE-L | 11.439 | 11.016432 | 0.403295 |
| BERTScore F1 | 0.845 | 0.844534 | 0.001008 |
| Normalized BERTScore F1 | 0.083 | 0.078854 | 0.005974 |

B-Norm: **2.004443**, standard deviation **0.159173**.
BLEU, chrF, B-Norm, and ROUGE use the evaluator's 0–100 scale.
Standard deviations use the population formula (divide by three).

Published scores come from [Table 14 of the Long Code Arena paper](https://arxiv.org/html/2406.11612v1).
The paper averages three runs on GPU. Only `2687987020` is supplied in the
original example; `42` and `123` are our choices.

## Files to share

- [aggregate.json](replication_results/aggregate.json): averages, standard deviations, and individual run scores.
- [predictions.jsonl](replication_results/predictions.jsonl): all 489 predictions, each labelled with its seed.

Verified that each run covers the same 163 unique commits in the same order,
uses the same configuration apart from its seed, and has valid prediction checksums.
The original model and metric source files are unchanged.
