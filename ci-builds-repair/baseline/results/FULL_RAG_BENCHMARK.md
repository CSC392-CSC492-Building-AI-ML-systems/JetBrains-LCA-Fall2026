# Full paired lean-RAG benchmark

Evaluated on 2026-11-19. Model: `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`. Generation run `372e8aa90832060ccc54`.

**68 cases × 3 seeds × 2 arms**; all raw answers audited.
Context: 32768 tokens; output cap: 8192; temperature: 0.2.
Both arms use the same oracle blocks, decoding settings, and improved composition.
RAG adds up to three BM25 snippets and 768 actual tokens from the failed snapshot.

| Arm | Valid responses | Mean edit F1 | Exact reference |
| --- | --- | --- | --- |
| plain | 184/204 | 0.281071 | 21 |
| lean_rag | 183/204 | 0.229798 | 21 |

**These edit scores are not CI Pass@1.**

The local evaluator recognizes 7 of 68 workflows. 61 remain explicitly unsupported.

| Arm | Seed | Supported cases resolved | Passing | Supported-case CI Pass@1 |
| --- | --- | --- | --- | --- |
| plain | 0 | 7/7 | 1 | 14.3% |
| plain | 1 | 7/7 | 1 | 14.3% |
| plain | 2 | 7/7 | 2 | 28.6% |
| lean_rag | 0 | 7/7 | 2 | 28.6% |
| lean_rag | 1 | 7/7 | 2 | 28.6% |
| lean_rag | 2 | 7/7 | 2 | 28.6% |

Full-dataset CI Pass@1 remains unavailable unless every selected workflow resolves. The machine-readable export retains all unsupported and unresolved cases, full-dataset bounds, failed/reference controls, and the supported-subset denominators.

Local Linux containers differ from hosted CI. This is not an exact replication of the original paper evaluation. No GitHub Actions evaluator was used.

[Portable metrics, per-case outcomes, and audit provenance](full_rag_metrics.json)

[Run configuration and reproduction instructions](../local_ci/README.md)

## Model, seeds, and decoding details

All arms in this full comparison use **Qwen2.5-Coder-7B-Instruct, four-bit**:
`mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`, revision
`019cc73c45c770444708a6dd8690c66243cc5c80`, through the local MLX runtime.
Configured seeds are **0, 1, 2**; the effective RNG seed is **seed + dataset ID**.
Each arm makes one prediction per case/seed; scores average separate Pass@1
runs and never select the best of three answers. Context is 32,768 tokens,
output cap 8,192, temperature 0.2, with 256 tokens of input/output headroom.
Model hashes, package versions, failing commits and source hashes are pinned
in the portable generation manifest. Both arms use the same improved
marker-recovery and guarded whitespace composition. Earlier Mistral and Qwen
pilot runs use different budgets and are listed separately in the registry.

For the full experiment registry, quantified failure taxonomy, case-level
explanations, causal limitations, and comparison with the paper, see
[models and seeds](../../RESULTS.md#models-seeds-and-experiment-registry),
[failure analysis](../../RESULTS.md#failure-analysis-across-the-full-paired-qwen-experiments),
and [paper comparison](../../RESULTS.md#comparison-with-the-paper-similarities-and-differences).
