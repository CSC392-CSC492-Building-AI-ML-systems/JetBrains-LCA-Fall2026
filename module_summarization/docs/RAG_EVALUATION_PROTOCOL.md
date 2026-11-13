# Module summarization evaluation protocol

Frozen on November 10, 2026 from the completed paired run. Keep the archived run as the reference for the 2K prefix and 2K BM25 arms. The 8K direct-context and agent settings below define the initial smoke tests. Record any changes before launching a full arm.

## Shared inputs and generation

| Setting | Frozen value |
| --- | --- |
| Dataset | `JetBrains-Research/lca-module-summarization`, test split, 216 tasks across 43 repositories |
| Task IDs and order | Zero-based original row IDs 0–215, in the exact `original_ids` order in the [run manifest](../benchmark_results/paired-full-gpt41-mini-15055eac3076fca7/manifest.json) |
| Dataset SHA-256 | `94603885b57d8cf3886266b36925a321aec1341b8352f19fbc920cbd63887189` |
| Shared code pool | Each task's supplied `relevant_code_context`. Keep `target_text` out of generation and tool inputs. |
| Generator | Saved requests used `gpt-4.1-mini` and all responses reported `gpt-4.1-mini-2025-04-14`. Request that snapshot for new arms and verify the returned model on the smoke run. |
| Generation prompt | Reuse [`build_prompt`](../rag_pipeline/rag_replication.py) with the original intent and filename, the selected code, and the same documentation-only instruction. The original system message is `You are a helpful assistant.` Agent planning adds tool instructions. Keep the final documentation request and output cap consistent and record any extra messages. |
| Decoding | Temperature 0, top-p 1, one completion, no streaming, frequency and presence penalties 0, up to 2,000 completion tokens for the final summary |
| Code-budget tokenizer | GPT-2 tokenizer JSON, SHA-256 `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`. Log provider-reported prompt and completion tokens separately. |

The saved dataset and tokenizer matched these hashes. All 432 prompt and prediction digests matched their archived artifacts when checked against the current prompt builder. The archived generation implementation fingerprint is in the run manifest.

## Arms and budgets

| Arm | Code selection and limits | Status |
| --- | --- | --- |
| 2K prefix | First 2,000 code tokens | Completed on all 216 tasks |
| 2K BM25 | Up to 2,000 code tokens, 400-token chunks, 50-token overlap, BM25 k1 1.5 and b 0.75. Query is the intent plus filename stem. The prefix is used when there are no matching terms. | Completed on all 216 tasks |
| 8K direct context | First 8,000 code tokens from the same supplied context | Planned. Verify the 8K limit on five tasks with enough code before expanding. |
| Agentic search | Read-only search and code reads over the same supplied context for the main comparison. Proposed smoke limits are six tool calls per task, up to 1,500 code tokens per response, and up to 8,000 code tokens across tool responses. Log all requests, returned evidence, and the stop reason. | Planned. Verify limits, traces, and costs on the same five tasks. |

For the repository-snapshot pilot, use a pinned snapshot and the same eligible source files for BM25, direct context, and agentic search. Exclude the reference documentation and copies from tool and retrieval inputs. Report pilot results separately because the source pool differs from the four-way comparison above.

## Judging and reporting

- Reuse the benchmark CompScore implementation and judge each prediction against the same reference in both orders. Supply the same first 6,000 judge-tokenized code tokens from the original context to every method. The archived evaluator used bfloat16 and FlashAttention 2.
- Use `mistralai/Mistral-7B-Instruct-v0.2` at revision `63a8b081895390a26e140280378bc85ec8bce07a` and `meta-llama/Llama-2-7b-chat-hf` as the judge-context tokenizer at revision `f5db02db724555f92da89c216ac04704f23d4590`. The [GPU environment](../benchmark_results/paired-full-gpt41-mini-15055eac3076fca7/compscore_environment.json) records dependency versions and scorer source hashes.
- Report mean CompScore, paired differences with a repository-resampled 95% interval, paired wins/losses/ties, and completed tasks out of 216. Use 10,000 repository bootstrap draws with seed 20261110. Report ROUGE-L as a diagnostic.
- Record total provider input/output tokens, API cost, generation time, and judging time. For the agent, include planning and tool turns in usage and latency. Log missing rows, truncation, and fallbacks and reconcile them before claiming full coverage. The [error review](RAG_PAIRED_ERROR_REVIEW.md) documents the development cases used to inspect the two existing arms.

| Saved method | Predictions | CompScore | Diagnostic ROUGE-L F1 | API prompt tokens | API completion tokens | API cost (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K prefix | 216 | 0.698315 | 0.113822 | 272,342 | 175,829 | 0.3903 |
| 2K BM25 | 216 | 0.690909 | 0.114475 | 241,687 | 181,970 | 0.3878 |

[Saved configuration](../benchmark_results/full-gpt41-mini.json) · [Per-task CompScore](../benchmark_results/paired-full-gpt41-mini-15055eac3076fca7/compscore_results.json) · [Full benchmark report](RAG_FULL_RUN_RESULTS.md)
