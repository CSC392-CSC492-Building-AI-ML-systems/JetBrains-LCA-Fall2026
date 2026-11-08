# Completed 50-example evaluation

Run date: November 8, 2026. Dataset: `JetBrains-Research/lca-bug-localization`,
`py` / `test`, all indices 0-49. Dataset fingerprint: `b63444d4ca764913`.
All five conditions completed: 250 scored pipeline rankings on the same 50 issues.
There were 248 usable model responses and two explicit BM25 fallbacks, with zero
request failures, zero preparation/pipeline failures and no remaining examples.
One separate access-check request was made before these runs and is excluded from scores.

## Common 50-example pipeline comparison

| Mode | Source budget* | MAP | MRR | Hit@1 | Hit@5 | Hit@10 | Scored | Validation fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 8192 | 0.623651 | 0.877333 | 0.780000 | 1.000000 | 1.000000 | 50/50 | 0 |
| rag | 8192 | 0.637463 | 0.867627 | 0.780000 | 0.980000 | 0.980000 | 50/50 | 0 |
| direct | 65536 | 0.613600 | 0.853000 | 0.760000 | 1.000000 | 1.000000 | 50/50 | 1 |
| rag | 65536 | 0.651043 | 0.882576 | 0.800000 | 0.980000 | 0.980000 | 50/50 | 0 |
| issue_only | 0 | 0.604859 | 0.848872 | 0.740000 | 0.980000 | 0.980000 | 50/50 | 1 |

RAG at the larger configured source budget has the highest MAP in this run.
These are pipeline scores: all modes use the shared BM25 completion tail, including
`issue_only`. A usable partial response is marked `repaired` and completed with
that tail. Two invalid responses use the complete BM25 fallback. The issue-only
model receives zero source tokens; its final ranking still benefits from BM25.

## Common model-success subset

This sensitivity comparison excludes example 8 and example 37 from **every**
condition, leaving the same 48 issues in each row. Validation-fallback counts
shown beside these subset scores are run-wide, not subset counts.

| Mode | Source budget* | MAP | MRR | Hit@1 | Hit@5 | Hit@10 | Scored | Validation fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 8192 | 0.611442 | 0.872222 | 0.770833 | 1.000000 | 1.000000 | 48/50 | 0 |
| rag | 8192 | 0.622358 | 0.862112 | 0.770833 | 0.979167 | 0.979167 | 48/50 | 0 |
| direct | 65536 | 0.612778 | 0.863542 | 0.770833 | 1.000000 | 1.000000 | 48/50 | 1 |
| rag | 65536 | 0.636503 | 0.877683 | 0.791667 | 0.979167 | 0.979167 | 48/50 | 0 |
| issue_only | 0 | 0.607625 | 0.861806 | 0.750000 | 1.000000 | 1.000000 | 48/50 | 1 |

## Settings, token accounting and failures

The model was `gpt-4.1-2025-04-14`, temperature 0, top_p 1, seed 17, with
`json_schema` responses, a configured 128,000-token context window, a 4,096-token
response reservation and the existing 5% context safety reservation. The shared
candidate filters, chunking, BM25 parameters, ranking validator and evaluator were
fixed across conditions. Preparation verified all 250 requests fit their local
context bounds and all 50 snapshots, candidate sets and BM25 tails matched.

**\*The source budgets use the documented conservative UTF-8 byte-count fallback,
not exact model-token counts.** Installed tiktoken 0.8.0 did not recognize this
pinned model name. The 8,192 and 65,536 settings are upper-bound estimates,
including JSON source labels and escaping; do not interpret this as an experiment
with exactly 65,536 model source tokens. Provider-reported total prompt usage ranged
from 311 to 32,585 tokens per request. The 250 evaluation requests used
2,741,468 reported prompt tokens and 81,992 completion tokens.
These usage totals exclude the separate access check. No transport retries occurred.

- `direct`, budget 65,536, example 8: output reached 4,096 completion tokens;
  `finish_reason=length`, recorded `fallback_validation` and the full BM25 ranking.
- `issue_only`, example 37: `refusal_or_incomplete_response`,
  `finish_reason=length`, 4096 completion tokens; recorded `fallback_validation` and the full BM25 ranking.

The unchanged `baseline_eval.evaluate_tfidf.score_example/aggregate` computes all
scores. All raw gold files remain in AP denominators, including 20 ground-truth
paths outside the candidate sets; those paths count as misses.

## Artifacts and reproduction

- [Full comparison](outputs/evaluation50/comparison/comparison.md)
- [Machine-readable comparison](outputs/evaluation50/comparison/comparison.json)
- [Per-example metrics](outputs/evaluation50/comparison/per_example_metrics.json)
- [Token usage, model outcomes and failure details](outputs/evaluation50/comparison/diagnostics.json)
- [Exact model settings](outputs/evaluation50/comparison/model_settings.json)
- [250-prompt preparation verification](outputs/preparation/verification.json)

Per-example prompts, responses, repairs, rankings, provenance and manifests are in
`outputs/evaluation50/<condition>/<signature>/`. The comparison JSON contains the
exact summary paths. The cached input Arrow file used for all five conditions was
`C:/Users/Admin/.cache/huggingface/datasets/JetBrains-Research___lca-bug-localization/py/0.0.0/434d4845687a0f04/lca-bug-localization-test.arrow`.

Use [evaluate_all.ps1](evaluate_all.ps1) with this dataset via `-DatasetFile` and the
same configuration to reproduce/resume; see [README.md](README.md) for exact commands.
Resume reuses usable saved responses and retries the two fallback examples, so a
future resumed comparison may differ. All 19 experiment tests passed during setup;
the launcher also passed full 250-prompt preparation and missing-key preflight checks.
