# Local CI workflow evaluator

`../run_local_ci.py` evaluates **saved predictions**, without generating model
answers, pushing commits, loading credentials, or submitting GitHub Actions.
Project commands execute only inside disposable Linux containers. The host
reads Git objects and prepares archives; it never imports benchmark projects.

This is a local reproduction of the workflow success criterion, not an exact
reproduction of the paper's hosted environments or its original 77-case set.

For measured output-processing improvements, see the
[repair pipeline results](../results/REPAIR_PIPELINE_IMPROVEMENT.md).

## Full paired RAG run

The [full RAG run](../results/FULL_RAG_BENCHMARK.md) compares both arms on all
68 cached cases with the same three seeds and improved composition. Reproduce
generation with:

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/run_qwen_lean_rag.py --all-cases --seeds 0 1 2 --context-window 32768 --max-tokens 8192 --composition recover-markers
```

Add `--prepare-only` to validate every input budget without loading model
weights. Use `--audit-run PATH` to reconstruct queries, validate retrieved
source text and token budgets, and independently recompute all saved scores
without model inference. The runner pins the checkpoint, packages, source
files, failing revisions, retrieval settings, and dataset checksum.

`finish_rag_benchmark.py` can follow the generation process and automatically
run the audits, local workflow evaluation, and final exports. Its live
`benchmark_status.json` distinguishes generation, audits, workflow execution,
completion with limited CI coverage, and failures. Raw answers are retained
under ignored `out/`; final portable results are written into `baseline/results/`.

The workflow evaluator accepts `--all-cases`. Unrecognized workflows remain
explicitly unsupported under the same arm and denominator. Multiple validated
control sets can be supplied using `--reuse-evaluation` followed by repeated
`--additional-reuse-evaluation` arguments. Reuse never changes a repair or drops
a required check.

## Long-context run

The [long-context experiment](../results/FULL_LONG_CONTEXT_BENCHMARK.md) uses
the same Qwen checkpoint, 68 cases, seeds 0/1/2, temperature 0.2, 32,768 context
limit, 8,192 output cap, and improved composition as the full lean-RAG experiment.
It adds full affected files, directly imported Python project files, and root
configuration from the failed commit. Affected files that exceed their input
allocation receive merged windows around the oracle locations. The original
oracle prompt is never truncated. Direct imports are read with Python's AST;
project code is never imported. This is bounded file context, not full-repository
context, and uses no BM25 or embedding retrieval.

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/run_qwen_long_context.py --prepare-only
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/run_qwen_long_context.py
```

The first command checks source provenance, budget allocation, and all 68 inputs
without loading model weights. The second generates 204 new long-context answers
and pairs them with the 204 identical audited standard-baseline answers. Baseline
reuse requires matching checkpoint hashes, runtime packages, input prompts,
decoding settings, composition, failed commits, and dataset checksum. Original
answers remain immutable; copied controls record source answer/manifest checksums.

Evidence records full/partial files, omitted candidates, exact line spans, and
token counts. `--audit-run PATH` independently reconstructs deterministic packing,
checks evidence bytes against the failed commit, audits the original baseline
with its frozen code, checks copied controls, and recomputes every score.

Use `finish_rag_benchmark.py --run PATH --generation-pid PID --evaluation-date YYYY-MM-DD
--reuse-evaluation PRIOR_LOCAL_CI_PATH` to follow generation, freeze evaluation
code, audit both stages, and publish `FULL_LONG_CONTEXT_BENCHMARK.md` plus
`full_long_context_metrics.json`. Reuse the completed paired lean-RAG local
evaluation so all seven validated failed/reference controls are preserved;
different patched trees still execute all original workflow checks.

An audited source evaluation may include unsupported workflows if every
supported failed/reference control validates and all supported executions
resolve. This authorizes reuse of the verified execution evidence; unsupported
cases still retain their denominators and full-dataset CI Pass@1 stays unavailable.
After an evaluator fix, supply `--evaluation-code-name evaluation-code-v2` to
freeze a new snapshot while retaining the original code and logs. Completed
model answers are audited and reused without additional inference.

## Improved repair composition

The original policy remains the default for replay of previous results. Use
`--composition recover-markers` for complete, unambiguous file/block-ID recovery
and boundary blank-line preservation. `--composition preserve-boundaries`
isolates the whitespace change. These policies use old source and model output
only. They do not change CI commands, skip checks, choose repairs using reference
code, or salvage length-capped responses. Python whitespace changes require
identical full-file ASTs.

Evaluate the combined policy on the original validation cohort:

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/run_local_ci.py --ids 33 34 36 37 144 --composition recover-markers --reuse-evaluation ci-builds-repair/ci-builds-repair-benchmark/out/local-ci/9d2833dd754e877a0e06
```

Reuse audits the source evaluation against its own original prediction runs.
New runs may use a different model, but the dataset, workflow semantics, and
worker must match. A verdict is reused only for a byte-identical patched
tree at the same failed commit. Copied evidence and its source checksums are
recorded; changed trees execute the full workflow. Reused dependency images
originate from failed controls, and setup commands still execute.

Generate fresh standard answers with no retrieval:

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/generate_local_baseline.py --model qwen --ids 142 --seeds 0 1 2
```

The generator also accepts `--model mistral` and `--all-cases`. For large oracle
prompts, explicitly budget for input and output, for example
`--context-window 32768 --max-tokens 16384`. Larger budgets are a separate
generation variant, not a claim about the paper's decoding setup. Raw answers,
model/source hashes, prompts, package versions, effective seeds, and budget
failures are saved. Existing predictions are verified and resumed, not replaced.

Audit generation without loading the model:

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/generate_local_baseline.py --audit-run PATH_TO_GENERATION_RUN
```

Evaluate a new run with `--runs standard-baseline/RUN_ID`; supply the same
`--runs` when auditing its workflow evaluation. Generation and composition
remain separate, so raw answers can be compared under both policies.

## Requirements and execution

Use the existing benchmark Python environment, its cached dataset and source
repositories, the ignored saved model runs, and Docker Desktop with a running
Linux engine. Internet access is needed for public images and historical
dependencies. No OpenAI API key or GitHub token is needed.

From the repository root, inspect all 68 cached workflows without Docker:

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/run_local_ci.py --inventory-only
```

Run the complete formatting workflows on cases 33, 34, 36, and 37:

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/run_local_ci.py
```

Add the complete five-version Starlette workflow:

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/run_local_ci.py --ids 33 34 36 37 144
```

The default saved runs contain the original Mistral baseline/RAG answers and
the Qwen development/lean-retrieval answers. Each run contributes only cases
that it actually generated; cross-model comparisons must use shared case IDs.
Use `--runs` with paths relative to the benchmark's `out/` directory to select
different saved runs. Missing predictions or changed checksums are errors.

Results go under ignored `ci-builds-repair-benchmark/out/local-ci/<signature>/`.
The directory contains the manifest, per-configuration logs and installed
package versions, failed/reference outcomes, prediction outcomes, and summary.
The inventory outside this directory covers all 68 dataset records.
Exact evaluator sources are archived with their manifest checksums. Existing
evaluation directories are never overwritten: audit an existing run or choose
a new `--output-dir` for another execution.

Independently audit a saved evaluation without Docker or model execution:

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/run_local_ci.py --audit-run PATH_TO_EVALUATION_DIRECTORY
```

The audit reconstructs model edits from raw, checksummed answers; verifies
target text against the exact failed Git revision; checks every required job,
matrix entry, command, Python version, recorded exit code and original-failure
fingerprint; and writes `audited_summary.json`. It can apply stronger control
validation to an earlier exploratory evaluation without relabeling its source
manifest or rerunning predictions. The audit records its own source hashes.

## Supported scope

The compiler currently recognizes four SkyPilot formatting cases and three
Starlette test cases. Recognition means a workflow can be compiled; each case
still needs its own successful failed/reference validation. All other cases
are explicitly unsupported until their workflow semantics have been reviewed.
Unknown actions, conditions, defaults, expressions, services, matrix features,
operating systems, missing checks, and unreviewed job sets are rejected.

For each supported case, the evaluator runs these in fresh source trees:

1. The original failed revision.
2. The reference patch applied to that same failed revision.
3. Each distinct valid model patch, only after both controls validate.

The original failed control must fail at a recorded failing step. Formatting
controls additionally require identical recorded diff hunks; the Starlette
test control must reproduce its recorded TypeError. The reference must pass
the entire selected workflow. Every required matrix configuration is executed;
no step is dropped to rescue a reference or model result.

Exact byte-identical patched trees share an execution result within a case,
including reference-identical repairs and no-op predictions. This is recorded
as an explicit execution reference. Seeds remain separate predictions; this
deduplication does not estimate independence or measure test flakiness.

To reduce repeated dependency downloads, the unmodified control can supply a
prepared container image for later variants. The source tree is replaced and
**all original setup and check commands run again**. Model-produced environments
are never used to prepare subsequent variants. Matrix jobs run at most two at
a time. Job containers and the package service/network are removed afterward;
downloaded and prepared images remain available locally.

## Scoring and limits

For each model arm and seed, local CI Pass@1 is passing single predictions
divided by selected cases, provided every selected case is resolved. Malformed
answers count as failed predictions when controls are valid. Setup failures,
timeouts, unsupported workflows, and invalid environment controls remain
unresolved. Incomplete groups receive lower/upper bounds and a null headline
score. The evaluator never silently removes them from the denominator.
Across-seed results are the mean and population standard deviation of the
individual Pass@1 values, not an any-seed success rate.

Container images use `linux/amd64` and the workflow's Python minor versions,
with exact image identities and actual patch versions recorded in execution
evidence. Debian replaces Ubuntu, and the original hosted Python patch
versions may differ. The historical package-index service is pinned by digest;
its address is translated from localhost to an isolated container-network name.
No host directories containing source caches or credentials are mounted.
Only the temporary evaluator input directory is mounted read-only. Containers
have resource limits, dropped Linux capabilities, and no Docker socket.
Source files and an initial Git index are recreated from the failed revision;
the original commit history and hosted checkout metadata are not reconstructed.

These differences, incomplete dataset coverage, and unavailable original model
snapshots prevent an exact paper reproduction claim. A successful affected-file
formatter check alone is still not full workflow success.
