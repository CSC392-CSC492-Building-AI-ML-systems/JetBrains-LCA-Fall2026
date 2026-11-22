# CI builds repair results

**Keep Qwen without retrieval as the default for now; keep long context
experimental.** The completed paired comparisons use all 68 cached cases and
seeds 0, 1, and 2 with the same checkpoint, output budget, and composition.
Standard has the strongest full-dataset edit scores; lean RAG has the best
local CI result on the seven supported workflows. No GitHub Actions evaluator
was run.

| Approach | Valid answers | Mean edit F1, all 68 cases | CI Pass@1, supported seven cases, mean across seeds |
| --- | --- | --- | --- |
| Standard | 184/204 | 0.281071 | 19.0% (4/21 case-seed attempts) |
| Lean RAG | 183/204 | 0.229798 | 28.6% (6/21) |
| Long context | 146/204 | 0.191581 | 0.0% (0/21) |

All seven failed/reference controls validate. The other 61 workflows remain
unsupported: **full-dataset CI Pass@1 is unavailable**, and the seven-case
subset is not comparable to the paper's original score. Repeated seeds are
not independent cases. See the [full lean-RAG report](baseline/results/FULL_RAG_BENCHMARK.md)
and [long-context comparison](baseline/results/FULL_LONG_CONTEXT_BENCHMARK.md)
for audits, settings, per-seed rates, and portable metrics.

## Earlier experiments

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

Those original checks cover individual files; repeated seeds on one file do
not count as independent fixes.

## Full local workflow validation

The [execution-based validation](baseline/results/LOCAL_CI_VALIDATION.md) now
covers five cases using 42 saved predictions. All five failed/reference pairs
validate, including Starlette's complete Python 3.8–3.12 matrix with linting,
builds, tests, and coverage. It uses local Linux containers and no GitHub Actions.

Mistral's no-retrieval baseline passes 0/15 selected predictions; original BM25
passes 3/15, all on SkyPilot case 37. On that shared formatting case, Qwen's
original format passes 3/3 and constrained raw output passes 1/3. On Starlette
case 144, both Qwen no-retrieval and lean retrieval pass 0/3: the saved answers
lose a required blank line and fail formatting despite the earlier AST credit.
These deliberately selected validation cases do not establish a full dataset
or paper score. The report includes portable metrics and reproduction commands.

The subsequent [repair pipeline improvement](baseline/results/REPAIR_PIPELINE_IMPROVEMENT.md)
re-evaluates the same answers with unambiguous marker recovery and guarded
boundary-whitespace preservation. Mistral's no-retrieval arm improves from
0/15 to 3/15; Qwen on case 144 improves from 0/3 to 1/3 without retrieval and
0/3 to 3/3 with lean retrieval. CI checks remain unchanged. An additional fresh
Qwen typing case scores 0/3 under both policies, so semantic errors remain.
These are development-subset improvements, not a match to the paper's score.

## Comparison with Long Code Arena

[Long Code Arena](https://arxiv.org/abs/2406.11612v1) reports GPT-4 bug
localization MAP of 0.39 (Table 20) and Mistral-7B CI Pass@1 of 6.5% (Table 7).
File ranking is distinct from patch generation. Our quantized Mistral, reconstructed prompt, current
68-case dataset, and offline metrics do not replicate the paper's 77-case
workflow evaluation. Edit F1 is not numerically comparable to CI Pass@1.

## Artifacts and next step

For the full 68-case paired Qwen lean-retrieval benchmark, see its
[run status and results](baseline/results/FULL_RAG_BENCHMARK.md). It records
fresh matched predictions, three seeds, independent audits, and explicit local
CI coverage limits.

The [long-context comparison](baseline/results/FULL_LONG_CONTEXT_BENCHMARK.md)
uses the same model and seeds, adding affected files and direct imports from
the failed snapshot within the native 32,768-token limit. Its paired standard
answers are shared with the completed lean-RAG run. The report distinguishes
input coverage, offline edit scores, and supported-workflow CI Pass@1.

The [machine-readable results](baseline/results/published_metrics.json) retain
run identities, verified Qwen case scores, local checks, and Mistral
verification summaries. The [original Mistral aggregate and per-case data](rag/results/)
are tracked separately. Raw prompts, responses, full patches, caches, and
weights remain local under ignored `out/` and `.venv/` directories.

Expand local failed/reference checks and include deterministic formatters
as baselines for formatting cases. Test any rule for when to retrieve on
new cases rather than scaling the current RAG setting by default.


## Models, seeds, and experiment registry

Model abbreviations below identify exact local checkpoints: **M** is
`mlx-community/Mistral-7B-Instruct-v0.2-4bit`; **Q** is
`mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`. Both are 7B instruction models
with four-bit weights, run locally through MLX on an Apple M3 Pro with 18 GB
memory. They are different model families; Qwen results are not Mistral
replications. Qwen revision is `019cc73c45c770444708a6dd8690c66243cc5c80`;
individual manifests pin model-file hashes and runtime packages.

Every generation experiment in this table uses seeds **0, 1, 2**, temperature
**0.2**, and effective RNG seed **configured seed + dataset ID**. Each
case/seed produces one answer per arm, without selecting the best answer from
three seeds. A mean across seeds is not Pass@3. Repeated seeds share the same
case and are not independent projects.

| Experiment / generation run | Model | Cases | Seeds | Total context / output cap | Interpretation |
| --- | --- | ---: | --- | --- | --- |
| Original standard vs BM25, `2d9349bb16247814246b` | M | 68 | 0, 1, 2 | 8,192 / 2,048 | 408 paired records including input-budget skips; original composition |
| Expanded-input standard, `88a9149b9da105e57f04` | M | 68 | 0, 1, 2 | 16,000 / 2,048 | 201 compatible old answers reused; three new answers reached cap |
| Original vs constrained output, `ac114173bcf206333b0a` | Q | 6: 98, 45, 143, 85, 37, 126 | 0, 1, 2 | 8,192 / 2,048 | Development experiment without retrieval |
| Lean-RAG pilot, `bf64963b600ed42de523` | Q | 6: 60, 119, 144, 41, 57, 39 | 0, 1, 2 | 8,192 / 2,048 | Different pilot cohort; original composition |
| Full standard vs lean RAG, `372e8aa90832060ccc54` | Q | 68 | 0, 1, 2 | 32,768 / 8,192 | 204 fresh answers per arm; improved composition |
| Full standard vs long context, `e7f55dfd54fce4c2b4fa` | Q | 68 | 0, 1, 2 | 32,768 / 8,192 | 204 new long-context answers; same 204 verified standard controls reused |
| Additional typing diagnostic, `67df8c1eb2aa1c0b77bf` | Q | 1: 142 | 0, 1, 2 | 16,000 / 8,192 | Fresh standard answers; 0/3 local CI passes |
| Larger-output diagnostic, `2ea11b2c7fb3fd05bed8` | M | 1: 33 | 0, 1, 2 | 32,768 / 16,384 | Two incomplete stop responses, one capped response; 0/3 local CI passes |

Local workflow validation and composition experiments replay existing answers;
they introduce neither a new model nor new random seeds. Their Mistral inputs
come from the original M run, and Qwen inputs from the development/pilot and
explicit diagnostic runs above. Boundary preservation and unambiguous marker
recovery change patch assembly only. They do not change generated answers,
CI commands, or choose a repair using reference success. Historical original
scores and improved-composition scores must remain separate.

## Failure analysis across the full paired Qwen experiments

The following counts cover 204 answers per arm (68 cases times three seeds).
They come from the audited per-answer records. Input budgets fit every case
in these full Qwen experiments; their failures are not input-budget skips.

| Observation | Standard | Lean RAG | Long context |
| --- | ---: | ---: | ---: |
| Valid replacement-format answers | 184 | 183 | 146 |
| Marker-format failures | 15 | 17 | 43 |
| Output-token limit failures | 5 | 4 | 15 |
| Valid but unchanged repairs | 37 | 55 | 24 |
| Nonempty patches with detected Python syntax regression | 15 | 16 | 34 |
| Exact reference matches | 21 | 21 | 7 |
| Supported-workflow passes / 21 case-seed attempts | 4 | 6 | 0 |
| Supported-workflow failures after an applicable answer | 11 | 12 | 10 |
| Invalid answers on supported workflows | 6 | 3 | 11 |

The first three rows partition output validity: a capped answer is classified
as a token-limit failure even if markers are also incomplete. No-op and syntax
rows describe valid answers and are not additional disjoint failure totals.
Syntax checks use Python 3.13 AST parsing, not the historical CI interpreters;
they cannot certify imports, types, formatting, tests, or runtime behavior.
A valid no-op leaves the known failing snapshot unchanged. Exact-match failure
alone does not establish a wrong repair: a different valid patch can pass CI.

### Output protocol and truncation

Some answers omit file/block markers, return fenced code alone, or fail to
return every requested block. For example, long-context case 144 returns a
short fenced replacement without the required markers. A plausible code
fragment is unusable under this benchmark's all-block output contract.
Unambiguous marker recovery can repair certain wrong line labels, but cannot
infer absent blocks or rescue an output that hit its token cap. Increasing
input context did not remove this serialization problem: long context turns
45 valid standard answers into invalid answers and recovers seven previously
invalid answers. Its 15 output-limit failures versus standard's five occur
with the same 8,192-token cap. These are measured protocol outcomes, not proof
that the intended semantic repair was always wrong.

### Syntax, formatting, and incomplete semantic repairs

Long context has 34 nonempty syntax regressions versus 15 for standard.
Case 37 seed 0 introduces an unexpected indent, which fails the actual YAPF
step; the other two long-context seeds are invalid. Case 142's three
long-context patches reach lint checks but fail formatting. Correct output
markers therefore do not imply a usable Python file, and AST validity does
not imply a formatter pass. The earlier case-144 pilot also demonstrated why
an AST-equal or correct-literal patch can still fail the workflow when blank
lines are lost. Guarded boundary preservation addresses that assembly issue,
but cannot fix an incorrect assertion or other substantive model mistake.

Typing and project-level failures remain harder than superficial edit overlap
suggests. The separate case-142 standard diagnostic still scores 0/3 after
composition fixes. The larger-output Mistral case-33 diagnostic does not
produce a complete accepted set of replacements, even at a 16,384-token cap.
More generation budget alone is therefore insufficient in those measured
cases. Early lint failure can also hide later semantic failures: an execution
that stops at formatting establishes a formatting failure, not that tests
would otherwise pass.

### Retrieval effects and the role of context

Lean RAG adds up to three distinct snippets and 768 tokens from the failed
snapshot, excluding the oracle edit spans. Long context instead supplies
complete affected files, direct imports, and configuration, with deterministic
windows when targets are too large. Median inputs are 346.5 tokens for
standard, 923 for lean RAG, and 23,802 for long context. Eleven long-context
cases have partial files and 42 omit candidate files; these categories overlap.
The experiment is bounded file context, not a complete repository prompt.

The longer input is associated with more invalid outputs, syntax regressions,
and lower edit F1. A plausible explanation is that redundant or unrelated code
competes with the log and output contract, encouraging whole-file or fenced
responses instead of the requested blocks. BM25 is lexical: similar names do
not guarantee evidence of the causal failure. Direct imports likewise need
not reveal dynamic configuration or distant dependencies. These explanations
are hypotheses consistent with the observations, not isolated causal results:
length, evidence contents, and representation all change together. No attention
analysis, relevance annotation, or controlled length-only ablation was run.
Four-bit quantization and model capacity could also matter, but no unquantized
control exists to measure their individual effects.

RAG's higher subset CI rate does not mean it improves every problem. Standard
passes case 37 on all seeds and case 144 on seed 2. Lean RAG passes case 36
and case 144 on all seeds, but not case 37. Neither arm passes cases 33, 34,
142, or 143. Thus its six versus four successes reflect gains and regressions
on a tiny, deliberately supported subset. Its lower full-dataset edit F1 and
55 no-ops versus standard's 37 show why the subset result should not justify
making retrieval universal. Recorded generation durations total about 6.11
hours for long context versus 1.25 for standard and 1.32 for lean RAG; these
are observed durations from different runs, not a controlled speed benchmark.

### Infrastructure, unsupported cases, and evaluator failures

The seven supported workflows have valid failed/reference controls: original
sources reproduce the recorded failure, and reference repairs pass all required
jobs, including the Starlette Python 3.8-3.12 matrices. There are no unresolved
supported attempts in the final comparison. The other 61 cases are explicitly
unsupported, not failed model repairs or successes. They remain in full-dataset
denominators, so full-dataset CI Pass@1 is unavailable. Linux amd64 containers,
Python patch versions, historical package availability, and hosted-runner
differences remain environmental limitations even with valid controls.

The first long-context CI stage stopped because its reuse gate incorrectly
required a complete full-dataset score from the source evaluation. The v2
snapshot permits independently audited resolved executions to be reused while
retaining unsupported denominators. This was an evaluator orchestration error,
not a model failure. Original logs/snapshots were retained; the final run was
re-audited without regenerating answers or changing CI success criteria.

## Comparison with the paper: similarities and differences

Source: the supplied *Long Code Arena* paper, section 2.2 (PDF pages 3-4)
and appendix B.5, Tables 5 and 7 (PDF pages 24-25),
[paper](https://arxiv.org/abs/2406.11612v1). The attached PDF is the reference
for these numbers, rather than later dataset versions or other tasks' results.

| Dimension | Paper's CI-repair experiment | These experiments |
| --- | --- | --- |
| Dataset | 77 items: 35 formatting, 14 local/typing, 28 dependency-sensitive | 68 cached items; equivalence to the original 77 has not been established |
| Inputs | Oracle files/hunks, failed code, seven-line first-error log excerpt (or final seven lines) | Same broad oracle/log setup, reconstructed prompt; added stable block IDs and explicit output contract |
| Models | Table 7 includes Mistral-7B, Mixtral, Llama-2, GPT-3.5 and GPT-4 | Local four-bit Mistral checkpoint for earlier runs; four-bit Qwen checkpoint for the full RAG/long-context comparison |
| Sampling | CI section does not specify our seed/temperature/cap schedule | Explicit seeds 0/1/2, effective seed plus ID, temperature 0.2; budgets listed above |
| Additional evidence | Reported oracle baseline edits all blocks in one request | Standard keeps this approach; RAG and long-context evidence are extensions, not paper replications |
| Primary correctness | Generated patch succeeds on the GitHub Actions workflow | Original supported workflow checks reconstructed in local Linux containers; coverage only 7/68 |
| Reported metrics | CI Pass@1: successfully fixed builds | Conditional supported-case CI Pass@1 plus explicitly separate edit F1, exact match, syntax and protocol diagnostics |

Table 7 reports Mistral-7B **6.5%**, GPT-3.5 **16.9%**, and GPT-4 **15.6%**
CI Pass@1. These are not edit F1 values. Our conditional seven-case values
(19.0% standard, 28.6% lean RAG, 0% long context) cannot show that we beat or
match those results: the models, precision, dataset, coverage, environment,
prompt details, decoding, and composition differ. Qwen is not a model evaluated
in that table. The earlier Mistral run matches a model family, not necessarily
the paper's exact weights, precision, or inference procedure.

Our failures resemble the paper's task categories: formatting, local/typing,
and dependencies across files can all prevent a build from passing. However,
the paper's difficulty labels classify the *original repair tasks*. Our marker,
truncation, syntax, and orchestration counts classify *model or pipeline
outcomes*. The cited CI section does not supply corresponding model-failure
counts, so we cannot claim the paper observed the same dominant failure mechanism.
It also does not establish that retrieval or full-file context improves CI
repair; retrieval gains in other benchmark tasks are not CI-repair evidence.

The shared lesson is that the complete workflow is the correctness test.
A syntactically plausible fragment or a high reference-edit score is insufficient.
Our measured protocol fragility and context-associated regressions explain part
of our local performance; they do not explain the numerical gap to the paper
without an aligned dataset/model/environment experiment.

## Next experiments justified by this evidence

Keep standard as the default for broad offline performance and retain lean RAG
as a conditional candidate based on the limited CI subset. Do not promote the
current long-context policy. First test output-protocol reliability and
syntax/format validation as explicitly separate variants without changing
historical results. Then compare targeted evidence budgets on unseen cases,
keeping model, oracle blocks, decoding, and output processing fixed. Finally,
expand workflow coverage with verified failed/reference controls before making
a full-dataset or paper-comparability claim. Inspect failure stages as well as
aggregate success; a gain on one formatting case is not evidence of improved
project-level reasoning.
