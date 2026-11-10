# Agentic evaluation: all 50 Python test issues

Completed on 2026-11-09 using `gpt-4.1-2025-04-14` and
`settings.example.json`. Only the agentic arm was run in this evaluation.

| Metric | Result |
|---|---:|
| Evaluated issues | 50 / 50 |
| MAP, full ranking | 0.615779 |
| MRR | 0.878144 |
| Hit@1 | 82% (41 / 50) |
| Hit@5 | 94% (47 / 50) |
| Hit@10 | 96% (48 / 50) |
| Full BM25 fallback rate | 2% (1 / 50) |
| Preparation/pipeline failures | 0 |
| Cumulative prompt tokens | 1,846,503 |
| Cumulative completion tokens | 21,869 |
| Cumulative total tokens | 1,868,372 |
| Model requests | 279 |
| Tool calls | 229 (4.58 per issue) |
| Cumulative model-request latency | 388.798 seconds |
| Mean model-request latency per issue | 7.776 seconds |
| Summed per-issue processing time | 560.534 seconds |

All model requests reported usage and completed on their first API attempt.
The summed processing time includes snapshot preparation, retrieval, packing,
agent requests and result construction; it is distinct from model-request latency.

## Configuration and provenance

- Dataset: `JetBrains-Research/lca-bug-localization`, `py/test`, indices 0–49.
- Local Arrow dataset fingerprint: `5f719511f18b8a99`.
- Initial serialized source budget: 4,096 tokens per issue; actual mean 4,010.28.
- Tool-call budget: six; search returns at most five snippets; reads at most
  120 lines; each tool result has a 1,500-token cap.
- Chunking: window 40, stride 20; BM25 k1=1.2, b=0.75.
- Model context allowance: 128,000 tokens; output allowance: 4,096 tokens.
- Model temperature 0, top_p 1, seed 17; strict JSON-schema responses.
- Shared Python candidate filter, base-SHA Git blobs, ranking validation and
  BM25 completion policy; no fix commit, patch or gold labels supplied to the model.
- Run signature: `35a54bf9e37ac8d26fb45632bae5cf09193b1f3171a1e9a2f7db6b01c0f5ab74`.

The exact model settings, implementation hashes, dataset identity, input hashes
and tool limits are retained in the run metadata.

## Outcomes and interpretation

Seven rankings passed without repair; 42 valid partial rankings were completed
using the shared BM25 tail. These 42 completions are not full BM25 fallbacks.
One issue, index 7, exhausted its tool budget and the final ranking response hit
the output-length limit. Its full BM25 fallback is included in all headline
scores. It was not retried or omitted. There were no preparation failures.

Twenty-four investigations stopped by submitting a ranking; 26 exhausted their
tool budget and proceeded to the final ranking request. The process returned
exit code 1 because it explicitly flags any full fallback; all 50 saved
rankings were present and the full-evaluation scorer passed successfully.

Metrics use the unchanged `baseline_eval.evaluate_tfidf.score_example` and
`aggregate` functions. Twenty gold files across 19 issues were outside the
candidate set; they remain misses and count in AP denominators. Do not interpret
these results as evidence of improvement over RAG or direct context without
running the matched comparison arms.

## Saved artifacts

- [Human-readable score report](outputs/test50/agent_scoring/comparison.md).
- [Aggregate metrics and resource totals](outputs/test50/agent_scoring/comparison.json).
- [Per-issue metrics](outputs/test50/agent_scoring/per_example_metrics.json).
- [Run metadata](outputs/test50/agent/35a54bf9e37ac8d26fb45632bae5cf09193b1f3171a1e9a2f7db6b01c0f5ab74/run_metadata.json).
- [Run summary](outputs/test50/agent/35a54bf9e37ac8d26fb45632bae5cf09193b1f3171a1e9a2f7db6b01c0f5ab74/summary.json).
- Rankings and full traces: `outputs/test50/agent/35a54bf9e37ac8d26fb45632bae5cf09193b1f3171a1e9a2f7db6b01c0f5ab74/examples/000.json` through `049.json`.

Generated outputs are retained locally and ignored by Git.

## Commands used

The local test Arrow file was
`C:/Users/Admin/.cache/huggingface/datasets/JetBrains-Research___lca-bug-localization/py/0.0.0/f2bb26a1c3320ccc1d7b02d2630077ff327a80f6/lca-bug-localization-test.arrow`.

```powershell
$testData = 'C:/Users/Admin/.cache/huggingface/datasets/JetBrains-Research___lca-bug-localization/py/0.0.0/f2bb26a1c3320ccc1d7b02d2630077ff327a80f6/lca-bug-localization-test.arrow'
.\.venv\Scripts\python.exe -m bug_localization.agentic.run --config bug_localization/agentic/settings.example.json --split test --start 0 --limit 50 --repo-token-budget 4096 --tool-calls 6 --window 40 --stride 20 --dataset-file $testData --output-dir bug_localization/agentic/outputs/test50/agent
.\.venv\Scripts\python.exe -m bug_localization.agentic.evaluate bug_localization/agentic/outputs/test50/agent --dataset-file $testData --output-dir bug_localization/agentic/outputs/test50/agent_scoring
```
