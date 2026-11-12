# 8K direct-context results

8K direct context achieved a mean CompScore of **0.721544** on all 216 module summarization tasks from 43 repositories. It led the 2K prefix by **2.32 points** on a 0–100 scale, with a paired 95% interval of **+0.93 to +4.28**.

The runner supplied the first 8,000 GPT-2 code tokens from each task's benchmark-provided context. It used the original documentation prompt, the pinned GPT-4.1 mini snapshot, and a 2,000-token output cap. Across the full set, 204 tasks received more code than at 2K, and 174 reached the 8K cap.

| Measure | Result |
| --- | ---: |
| Predictions / tasks | 216 / 216 |
| Mean CompScore | 0.721544 |
| Diagnostic ROUGE-L F1 | 0.112306 |
| Mean code tokens in final prompt | 7,203 |
| API input / output tokens | 868,581 / 226,458 |
| Generation wall time | 30.4 min |
| Generation API cost | US$0.71 |
| GPU scoring time, including judge loading | 15.8 min |

All responses ended normally and passed the source-offset audit. CompScore used 432 judge evaluations, two per prediction, with the pinned Mistral judge and the shared 6,000-token judge context. This judge context can omit evidence used by an 8K generator. The earlier 2K arms used CUDA 12.8, while the new arms used CUDA 12.6 with the same judge revisions and precision.

The extra context helped on task 49, where the prompt included `SurvfuncRight` and the summary explained it. On task 199, the prefix still missed `MLPClassifier` and `MLPRegressor`. The [four-way comparison](../docs/RAG_FOUR_WAY_COMPARISON.md) reports the paired differences and the repository bootstrap procedure.

[Full configuration](full.json) · [Smoke run](smoke/) · [Per-task scores](results/long-context-8k-full-ff1b80725edebf82/compscore_results.json) · [Scoring environment](results/long-context-8k-full-ff1b80725edebf82/compscore_environment.json)
