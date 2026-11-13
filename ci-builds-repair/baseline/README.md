# Local baseline and experimental runs

For the headline results and comparison with the paper, start with the
[results summary](../RESULTS.md). This guide documents how to reproduce or
audit the baseline and the subsequent pilots. The
[published metrics](results/published_metrics.json) can be read without
downloading models or accessing the ignored run cache.

The historical manifests retain exact source checksums from before the
documentation cleanup. To audit all published Mistral and Qwen runs from
their original source in a temporary directory, use the local environment:

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/audit_historical_runs.py
```

The wrapper needs the recorded Git revision and ignored data caches. It does
not load the model, regenerate predictions, or change the saved run outputs.
The direct commands below refer to the original source version when used
against those historical artifacts.

## Verified non-RAG Mistral baseline

The completed non-RAG Mistral run contains **204 evaluation cases: 68 records
with seeds 0, 1, and 2**. It used no retrieved context. See the standalone
[results and paper-comparability assessment](results/REPORT.md).

These are the baseline predictions already generated in run
`2d9349bb16247814246b`, not another model run. The verification below reads the
saved raw answers, verifies checksums and unaugmented prompts, validates the
replacements against each answer, and independently recomputes reference-edit
and syntax scores from target blobs at the exact failed commits. It neither
runs retrieval nor executes project code, loads credentials, or calls GitHub
Actions. Missing or changed predictions, dataset files, implementation sources,
or scores cause an error rather than silently reducing coverage.

From the repository root:

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/verify_completed_run.py \
  --run-dir ci-builds-repair/ci-builds-repair-benchmark/out/rag-local/2d9349bb16247814246b \
  --dataset-file ci-builds-repair/ci-builds-repair-benchmark/.venv/dataset-cache/JetBrains-Research___lca-ci-builds-repair/default/0.0.0/ebf12dad7a97c3c0cdc38705403d8bc8ddde47dc/lca-ci-builds-repair-test.arrow
```

Small verified summaries are stored in `results/metrics.json` and
`results/per_example_metrics.csv`. Original prompts, responses, and patches
remain in the ignored source run directory. Verification does not establish
the model's correctness or the paper's CI Pass@1 score.

For **new generation**, the existing local runner supports baseline-only mode:

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/rag/run_benchmark.py --methods baseline --seeds 0 1 2
```

This generates a separate content-addressed run; it does not reuse the
combined run's prediction files. Its settings have the same limitations as
the verified experiment. Do not describe it as exact paper reproduction.

Run the verifier's checks without loading the model:

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python -m unittest discover -s ci-builds-repair/baseline -p 'test_*.py' -v
```

## Paired edit-format pilot

`run_edit_format_pilot.py` compares legacy all-block markers with sparse JSON
block replacements, each with and without the **same saved BM25 context**.
The two legacy arms reuse checksummed model answers and re-score them; the two
sparse arms generate new answers. The prompt keeps the old source and log
excerpt; only the output instruction and parser change. Omitted blocks stay
unchanged, an empty edit object is a valid no-op, and malformed or truncated
answers get no edit credit. The parser accepts a single enclosing JSON fence
and restores the original final line ending if a nonempty replacement omits it.

The six examples are selected by hashing record IDs with a fixed salt, without
consulting prior outcomes. All three prior seeds, model files, runtime packages,
temperature, and context/output limits are held fixed. This is an exploratory
pilot on the existing test data, not a held-out evaluation. It intentionally
does not change retrieval or run any builds.

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/run_edit_format_pilot.py
```

Add `--prepare-only` to verify inputs and create the manifest without loading the
model. The runner resumes from checksummed predictions under
`out/edit-format-pilot/<signature>/` in the benchmark directory. It updates
`metrics.json` after each case. Compare all four arms only after
`complete: true`; interim contrasts use only cases for which all four arms have
scores. Use nonempty patch rate alongside response validity, since valid
responses can be no-ops. F1 and Python syntax checks are still offline proxies.

## Sparse-marker development check

The JSON pilot accepted none of its 36 new responses. The follow-up keeps the
same six development examples, seeds, Mistral checkpoint, failing code and log
excerpts, and compares the saved no-retrieval control to **optional plain-text
blocks**. The model returns complete marker pairs only for changed blocks;
omitted blocks stay unchanged. Exact labels, balanced markers, and one
replacement per block are required. A single outer Markdown fence is accepted,
but text outside the marker pairs is not applied. Empty replacements can delete
a block; `[no edits]` is a valid no-op. A length-capped answer is not salvaged.

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/run_sparse_marker_dev.py
```

The resumable run records 18 fresh answers and 18 re-scored controls under
`out/sparse-marker-dev/<signature>/`. This development set is not held out.
Advance to a fresh retrieval comparison only if the sparse format increases
the number of nonempty patches without a detected syntax regression, matches
or improves overall edit F1, and produces at least one nonempty multi-block
case without a detected regression. Non-Python edits are not validated by
the Python syntax check. No build, project code, or GitHub Actions runs here.

The [completed development check](results/SPARSE_MARKER_DEV.md) failed this
gate; a parser-only diagnostic did not rescue syntax-preserving edits.

## Enforced short-ID format smoke test

`run_constrained_edit_smoke.py` tests six handwritten, non-benchmark fixtures
with the same local Mistral checkpoint and seeds 0–2. Each fixture is generated
with and without the token mask, using the same prompt, temperature, and
192-token output limit. The prompt asks for a flat JSON object keyed by short
IDs such as `B0`. Omitted keys leave blocks unchanged, an empty string deletes
a block, and an empty object represents no edits. The mask uses
[LLGuidance](https://github.com/guidance-ai/llguidance) to allow only continuations
of an object with known ordered keys and string values. The decoder stops at
the first complete object. Both arms use the same post-generation validator;
length-capped answers are not salvaged.

Install the pinned Apple Silicon wheel in a separate ignored folder, then run:

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python -m pip --isolated install --only-binary=:all: --no-deps --require-hashes --target ci-builds-repair/ci-builds-repair-benchmark/.venv/constrained-deps -r ci-builds-repair/baseline/requirements-constrained-macos.txt
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/run_constrained_edit_smoke.py
```

The 36 responses, model identity, and per-fixture scores are stored under
`out/constrained-edit-smoke/<signature>/`. The smoke gate requires all 18
constrained answers to pass format and Python parsing, at least 12 to match
the expected Python AST, and at least one AST match on each multi-block
fixture. AST comparison ignores formatting but does not establish behavioral
equivalence. No generated source is executed. Passing this gate justifies a
new no-retrieval development check, not a retrieval rollout or a CI pass claim.

The [completed synthetic smoke test](results/CONSTRAINED_EDIT_SMOKE.md)
eliminated malformed envelopes, but failed the repair-quality gate.

## Raw-code edit smoke test

The follow-up adds a multiline replacement fixture to the same six synthetic
examples. It replaces JSON strings with an action (`KEEP`, `DELETE`, or
`REPLACE`) keyed by a short block ID. Replacement code is raw text terminated
by an `END` line. `DONE` ends the answer. Omitted blocks stay unchanged.
The original common indentation is stripped when displaying an input block
and restored when applying a replacement. Both arms share the exact prompt
and scorer; one uses the same token-mask engine as the JSON smoke test.

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/run_raw_edit_smoke.py
```

It uses the same pinned grammar dependency, model, seeds, and output limit.
Raw answers and metrics are saved under `out/raw-edit-smoke/<signature>/`.
The gate requires all 21 constrained answers to be valid and parse as Python,
at least 14 expected-AST matches, and an expected-AST match on each multi-block
fixture, the multiline fixture, and the no-op fixture. This is a smoke test
of known-good structure, not a measure of CI repair success. No source executes.

The [completed raw-code smoke test and model comparison](results/RAW_EDIT_SMOKE.md)
record the Mistral result and the separate Qwen Coder trial. With the pinned
checkpoint cached under the benchmark environment's `.venv/qwen-coder-model`,
run the matched comparison using:

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/compare_raw_edit_models.py
```

## Qwen-only CI repair development run

The synthetic comparison qualified Qwen Coder for a separate development
experiment on the same six real cases used in the earlier pilots. Both arms
generate fresh answers with the same Qwen checkpoint, seeds 0–2, log excerpts,
oracle blocks, temperature, and input/output budgets. The control requests
the original all-block markers. The candidate uses constrained raw edits and
base-indentation anchoring. Neither arm sees retrieved context or gold
replacement code. The Mistral run and its artifacts are unchanged.

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/run_qwen_ci_dev.py
```

This is a 36-answer, resumable, offline scoring run. Output is saved under
`out/qwen-ci-dev/<signature>/`; add `--audit-only` to verify and re-score a
completed run without the GPU. A valid no-op is not a nonempty patch. The
gate requires more nonempty patches without detected Python syntax
regressions, at least the control's overall edit-token F1, and at least one
nonempty multi-block case without a detected regression. Non-Python edits
are not validated by the Python syntax check. No builds or GitHub Actions run.

The [completed Qwen development report](results/QWEN_CI_DEV.md) documents the
failed constrained-format gate and the independently replayed YAPF check for
case 37. `verify_qwen_yapf_case.py` uses the two packages pinned in
`requirements-local-format-check.txt`, installed with `--target` into the
benchmark environment's `.venv/local-format-check` directory. It passes
source as data to the formatter under a compatible Python interpreter
(default `/usr/bin/python3` on this Mac).

## Matched lean-retrieval experiment

`run_qwen_lean_rag.py` keeps Qwen and the original all-block markers fixed in
both arms. It selects six IDs by a fixed hash, excluding the development
cohort and the previously inspected comparison examples. These are new to
Qwen prompt development, not a new dataset. Both arms use the same seeds,
base prompt, editable blocks, output cap, and offline evaluator.

The retrieval query uses only failure log text: the short error window plus
filename, exception, and test-name cues from the tail of each failed step.
BM25 searches 20-line windows from the cached failing commit. The RAG arm
packs up to three snippets, at most one per file, and no duplicate text or
overlap with editable blocks. It adds at most 768 tokens as counted by Qwen's
chat tokenizer. With no fitting evidence, the RAG arm reuses the identical
plain answer and records that no context was added.

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/baseline/run_qwen_lean_rag.py
```

The resumable run saves predictions, packed evidence, and paired scores under
`out/qwen-lean-rag/<signature>/`. Use `--audit-only` after completion to check
the saved answers and evidence against the failed source and re-score them
without the GPU. No project tests, workflows, or GitHub Actions run here.

For the affected-file Black check in W&B case 119, install `black==23.7.0`,
`click==8.1.7`, `mypy-extensions==1.0.0`, `packaging==23.1`,
`pathspec==0.11.2`, `platformdirs==3.10.0`, and `tomli==2.0.1` in the
benchmark environment's ignored `.venv/local-black-check` target. Then run
`verify_lean_rag_black_case.py`. With `--controls-only`, it checks the exact
failing and reference texts first. It uses the failed commit's Black style
and treats patches as text, never executing them. This is not the full
pre-commit job or CI Pass@1.

The [completed lean-retrieval report](results/QWEN_LEAN_RAG.md) includes the
paired results and the Starlette literal/AST check. Re-run the latter with
`verify_lean_rag_starlette_case.py`; it does not execute the app or test.
