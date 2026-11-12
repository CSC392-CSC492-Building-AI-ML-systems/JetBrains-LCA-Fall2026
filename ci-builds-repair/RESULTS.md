# CI builds repair results

**Keep Qwen without retrieval as the default for now.** Larger BM25 context
hurt the local Mistral run; lean retrieval was effectively tied with no
retrieval on the six-case Qwen comparison. These are patch-level results,
not the paper's CI Pass@1. No GitHub Actions evaluator was run.

## Main experiments

| Experiment | Cases and seeds | No retrieval | With retrieval |
| --- | --- | ---: | ---: |
| [Mistral 7B, 4-bit: original BM25 comparison](rag/results/REPORT.md) | 68 cases × 3 seeds | 0.195233 edit F1 | 0.117880 edit F1 |
| [Qwen2.5-Coder 7B, 4-bit: lean BM25](baseline/results/QWEN_LEAN_RAG.md) | 6 cases outside the development cohort × 3 seeds | 0.250119 edit F1 | 0.249960 edit F1 |

- **Mistral:** 123/204 applicable answers without retrieval vs. 68/204
  with it. Extra context increased marker failures and output truncation.
- **Qwen:** original all-block markers held fixed. Lean retrieval added at
  most three deduplicated snippets and 768 tokens. It won seven pairs, tied
  nine, and lost two. The mean delta was −0.000159; the direction is sensitive
  to individual cases. These are six CI cases, not 18 independent cases.

## Reproduction and format development

- The [verified non-RAG Mistral run](baseline/results/REPORT.md) checked all
  204 saved predictions. The [16,000-token rerun](baseline/results/RUN_16K.md)
  kept the same scores: it reused compatible answers and generated the three
  formerly over-budget cases, which all reached the output cap. The separate
  partial run `2a159171f4b13a8d7ceb` is not included.
- The [reference oracle](baseline/run_reference_oracle.py) reconstructed
  68/68 reference repairs. This is an integrity check using the actual
  reference diff, not a model result; F1 was undefined for three
  whitespace-only repairs.
- Sparse output made format validation easier in the
  [constrained synthetic trials](baseline/results/RAW_EDIT_SMOKE.md), but on
  real CI examples [Qwen's original format](baseline/results/QWEN_CI_DEV.md)
  beat constrained raw edits (0.2681 vs. 0.1270 edit F1). Eight valid
  constrained answers returned no edits. The intermediate
  [JSON](baseline/results/CONSTRAINED_EDIT_SMOKE.md) and
  [plain-text marker](baseline/results/SPARSE_MARKER_DEV.md) results are
  exploratory, not alternative headline scores.

## What the local checks establish

| Case | Check | Result |
| --- | --- | --- |
| [SkyPilot 37](baseline/results/QWEN_CI_DEV.md#case-37-local-replay-of-the-affected-file-formatter) | Affected-file YAPF 0.32.0, failed and reference controls | Original Qwen format passed 3/3 seeds; constrained raw edits passed 1/3. |
| [W&B 119](baseline/results/QWEN_LEAN_RAG.md#local-checks-not-ci-pass1) | Affected-file Black 23.7.0, failed and reference controls | No retrieval and lean retrieval each passed 0/3 seeds. |
| [Starlette 144](baseline/results/QWEN_LEAN_RAG.md#local-checks-not-ci-pass1) | Replayed the logged literal error and compared the resulting Python AST | Both arms corrected the literal. The full file AST matched the reference on 1/3 plain and 3/3 retrieval seeds. No pytest or app code ran. |

These are targeted checks; repeated seeds on one file do not count as
independent fixes, and no full workflow was evaluated.

## Comparison with Talha and Long Code Arena

[Talha's long-context run](https://github.com/CSC392-CSC492-Building-AI-ML-systems/JetBrains-LCA-Fall2026/blob/c0593f9f4be4db5f28026fd59abf2a6a95f8560e/bug_localization/long_context/RESULTS.md)
ranked files on 50 Python issues. At the larger matched source budget, GPT-4.1
with RAG improved MAP from 0.6136 to 0.6510 and Hit@1 from 76% to 80%, with a
shared BM25 completion tail. Ranking files is not generating a valid patch.

[Long Code Arena](https://arxiv.org/abs/2406.11612v1) reports GPT-4 bug
localization MAP of 0.39 (Table 20) and Mistral-7B CI Pass@1 of 6.5% (Table 7).
Talha's Python-only ranking setup and newer model are not a controlled
comparison to Table 20. Our quantized Mistral, reconstructed prompt, current
68-case dataset, and offline metrics do not replicate the paper's 77-case
workflow evaluation. Edit F1 is not numerically comparable to CI Pass@1.

## Artifacts and next step

The [machine-readable results](baseline/results/published_metrics.json) retain
run identities, verified Qwen case scores, local checks, and Mistral
verification summaries. The [original Mistral aggregate and per-case data](rag/results/)
are tracked separately. Raw prompts, responses, full patches, caches, and
weights remain local under ignored `out/` and `.venv/` directories.

Expand local failed/reference checks and include deterministic formatters
as baselines for formatting cases. Test any rule for when to retrieve on
new cases rather than scaling the current RAG setting by default.
