# Module summarization RAG full benchmark

The run covered all 216 tasks from 43 repositories. Each task produced one prefix baseline summary and one BM25 summary using the same prompt, code budget, and GPT-4.1 mini model. The GPT-3.5 generation endpoint returned HTTP 404, so both conditions used the replacement model. The API returned snapshot `gpt-4.1-mini-2025-04-14` for all 432 predictions.

| Metric | Prefix baseline | BM25 RAG |
| --- | ---: | ---: |
| Mean official CompScore | 0.698315 | 0.690909 |
| Mean diagnostic ROUGE-L F1 | 0.113822 | 0.114475 |
| Mean code tokens supplied to generator | 1948.0 | 1771.8 |

The paired CompScore difference was -0.007406 (-0.741 points on a 0–100 scale). BM25 scored higher on 99 tasks, lower on 110, and tied on 7. The 95% percentile interval was [-0.027561, +0.012429], using 10,000 bootstrap draws that resampled whole repositories and retained all paired tasks within each sampled repository.

BM25 used the documentation intent and filename to select code from the supplied relevant context. Each strategy had a maximum of 2,000 GPT-2 code tokens. BM25 changed the context on 203 tasks and retrieved source beyond the prefix on 186. The source-offset audit passed for all 432 contexts.

Generation took 58.8 minutes and used US$0.78 in API tokens. The 864 judge evaluations took 28.4 minutes on the Vast.ai RTX A6000, including model loading. GPU setup and rental charges are separate.

CompScore used the benchmark implementation, with the same 6,000-token source context for both strategies and both comparison orders against the reference. The Mistral judge revision was `63a8b081895390a26e140280378bc85ec8bce07a`. The Llama 2 tokenizer revision was `f5db02db724555f92da89c216ac04704f23d4590`. The results archive contains the generation manifest, every summary, retrieval traces, per-task scores, and the GPU environment.
