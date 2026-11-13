# Agentic code search results

Agentic code search achieved a mean CompScore of **0.693622** on all 216 module summarization tasks from 43 repositories. It scored **2.79 points below 8K direct context** on a 0–100 scale, with a paired 95% interval of **−5.04 to −1.04**. After excluding the 12 development tasks, the difference was **−2.85 points** across 204 tasks, with a 95% interval of **−5.05 to −1.24**.

The agent used read-only search and read tools on the benchmark-provided code. Each task allowed six tool calls, 1,500 GPT-2 code tokens per response, and 8,000 code tokens across responses. A separate writing request used the original documentation prompt, the pinned GPT-4.1 mini snapshot, and a 2,000-token output cap.

| Measure | Result |
| --- | ---: |
| Predictions / tasks | 216 / 216 |
| Mean CompScore | 0.693622 |
| Diagnostic ROUGE-L F1 | 0.118665 |
| Mean tool calls per task | 4.79 |
| Mean code tokens returned by tools | 4,333 |
| Mean code tokens in final prompt | 3,015 |
| API input / output tokens across all turns | 2,362,623 / 271,801 |
| API request time across recorded responses | 44.7 min |
| Confirmed generation API charges | US$1.38 |
| GPU scoring time, including judge loading | 15.6 min |

Search returned zero code on five tasks, and task 73 hit the final output cap. The evaluation includes those outputs. Two interrupted requests were retried, and their original reservations remain unresolved. CompScore used the same pinned judge and 6,000-token judge context as 8K direct context.

On task 199, search found both public MLP classes beyond the 8K prefix. The summary covered both classes and also discussed `BernoulliRBM`, broadening the scope. On task 4, the retrieved code contained `session.call` guidance, but the summary recommended the deprecated `session.delayed_job` method. The [four-way comparison](../docs/RAG_FOUR_WAY_COMPARISON.md) gives the full cost and quality comparison, development task IDs, and bootstrap procedure.

[Full configuration](full.json) · [Smoke run](smoke/) · [Development runs](development/) · [Per-task scores](results/agentic-8k-full-fbc31a6c0d568440/compscore_results.json) · [Scoring environment](results/agentic-8k-full-fbc31a6c0d568440/compscore_environment.json)
