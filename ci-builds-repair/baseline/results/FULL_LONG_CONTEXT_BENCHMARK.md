# Full paired long-context benchmark

Evaluated on 2026-11-19. Model: `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`. Generation run `e7f55dfd54fce4c2b4fa`.

**68 cases × 3 seeds × 2 arms**; all raw answers audited.
Context: 32768 tokens; output cap: 8192; temperature: 0.2.
Both arms use the same oracle blocks, decoding settings, and improved composition.
Long context adds failed-snapshot affected files, direct Python imports, and root configuration. Oversized targets use bounded windows; omissions are recorded. The identical verified standard-baseline answers are reused with source checksums.

| Arm | Valid responses | Mean edit F1 | Exact reference |
| --- | --- | --- | --- |
| plain | 184/204 | 0.281071 | 21 |
| long_context | 146/204 | 0.191581 | 7 |

**These edit scores are not CI Pass@1.**

The local evaluator recognizes 7 of 68 workflows. 61 remain explicitly unsupported.

| Arm | Seed | Supported cases resolved | Passing | Supported-case CI Pass@1 |
| --- | --- | --- | --- | --- |
| plain | 0 | 7/7 | 1 | 14.3% |
| plain | 1 | 7/7 | 1 | 14.3% |
| plain | 2 | 7/7 | 2 | 28.6% |
| long_context | 0 | 7/7 | 0 | 0.0% |
| long_context | 1 | 7/7 | 0 | 0.0% |
| long_context | 2 | 7/7 | 0 | 0.0% |

## Comparison with lean RAG

Same model, cases, seeds, generation budgets, and composition. The standard-baseline answers are shared, not generated again.

| Approach | Mean edit F1 | Supported-case CI Pass@1, mean across seeds |
| --- | --- | --- |
| plain | 0.281071 | 19.0% |
| long_context | 0.191581 | 0.0% |
| lean_rag | 0.229798 | 28.6% |

Full-dataset CI Pass@1 remains unavailable unless every selected workflow resolves. The machine-readable export retains all unsupported and unresolved cases, full-dataset bounds, failed/reference controls, and the supported-subset denominators.

Local Linux containers differ from hosted CI. This is not an exact replication of the original paper evaluation. No GitHub Actions evaluator was used.

[Portable metrics, per-case outcomes, and audit provenance](full_long_context_metrics.json)

[Run configuration and reproduction instructions](../local_ci/README.md)

## Failure analysis and next steps

Long context produced 43 marker-format failures and 15 output-limit failures,
compared with 15 and 5 for standard, and 17 and 4 for lean RAG. It changed
45 previously valid standard answers into invalid answers, while recovering
seven previously invalid answers. These are paired case-seed observations,
not independent case counts. Exact reference matches fell from 21/204 to 7/204.
The measured losses include output-format, syntax, and repair errors; they
cannot be attributed to context length alone from this experiment.

On the seven supported workflows, long context has 11 invalid attempts and
10 failed workflows, with zero passes. Case 37's seed-0 patch introduces an
unexpected indent. All three case-142 patches fail linting. Case 144's three
answers omit the required replacement markers. All failed/reference controls
validate, including the complete Python 3.8–3.12 matrices.

The median long-context input is 23,802 tokens. Eleven cases use partial files;
42 omit one or more candidate files. Recorded model-generation durations sum
to 6.11 hours for long context, versus 1.25 hours for the earlier standard
answers and 1.32 hours for lean RAG. Standard answers were reused in this run.
These are observed durations, not a controlled hardware performance benchmark.

Keep long context experimental. Retain standard as the default for the broader
cached dataset based on its stronger edit scores; lean RAG remains the better
CI result on the limited seven-case subset. Next, prioritize reliable output
markers and syntax validation, then evaluate a smaller, targeted file-context
policy on unseen cases. Expand supported workflow coverage before claiming
full-dataset CI Pass@1 or comparing it numerically with the paper.

The initial CI stage stopped because execution reuse required a complete
full-dataset score. A separate `evaluation-code-v2` snapshot fixes that reuse
check: independently audited supported executions can be reused while
unsupported cases retain their denominators. Original snapshots and logs are
preserved. Generation and workflow audits passed after recovery; no model
answers were regenerated or changed.

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
