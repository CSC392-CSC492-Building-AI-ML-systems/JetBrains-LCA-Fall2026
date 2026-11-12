# Local CI repair RAG benchmark

This experiment adapts the bug-localization retrieval pipeline:
failed-revision snapshots → deterministic overlapping chunks → BM25 retrieval
→ token-budgeted model context → saved predictions → separate evaluation.
See [the pipeline review](PIPELINE_REVIEW.md) for the adaptations and verification.

The completed 68-record, three-seed experiment is documented in the
[results report](results/REPORT.md) and [interpretation](results/ANALYSIS.md).
All 408 evaluation cases completed; RAG's mean edit-token F1 was 0.1179 versus
0.1952 for baseline. These are offline proxy scores, not build pass rates.

Both conditions use Mistral-7B-Instruct-v0.2 in its installed 4-bit MLX
conversion, temperature 0.2, 2,048 output tokens, an 8,192-token context limit,
and seeds 0/1/2. Each record uses `seed + dataset_id`. The baseline receives
the paper-style oracle edit blocks and a seven-line error-log excerpt. RAG
receives the same edit blocks and extra BM25 evidence from the failing snapshot.
This is **oracle-localized RAG**, not an end-to-end fault-localization system.
Gold replacement code never enters generation or retrieval. Both methods have
the same target locations so the comparison isolates extra repository context.

## Run

From the repository root, use the existing ignored benchmark environment:

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python ci-builds-repair/rag/run_benchmark.py
```

The default runs all records from the cached test Arrow file, both conditions,
and three seeds. No GitHub token is needed. The runner never loads `.env` files,
pushes branches, edits workflows, or invokes the GitHub Actions evaluator. Public
benchmark forks are cloned as bare Git object stores. Project code is not run.

Use `--ids 53` for a smoke experiment, `--prepare-only` to download and validate
source revisions without model inference, `--dataset-file path/to/test.arrow`
to select another fixed dataset artifact, and `--evaluate-only path/to/run`
to rebuild aggregate metrics from saved records. `--seeds`, `--temperature`,
`--max-tokens`, `--context-window`, `--model-path`, and `--output-dir` are
explicit experiment settings. Identical settings and source hashes reuse saved
predictions; changed settings produce a new content-addressed run directory.

The supplied [requirements](requirements.txt) describe the installed Mac runtime.
MLX requires Apple Silicon and GPU access. Tests and offline metric functions
do not load the model:

```bash
cd ci-builds-repair/rag
../ci-builds-repair-benchmark/.venv/bin/python -m unittest test_rag -v
```

## Retrieval policy

- Read text blobs from the exact failed SHA, including tests; never read the
  successful SHA. Symlinks, binary/non-UTF-8 blobs, and blobs over 512 KiB are
  excluded with counts in snapshot metadata.
- Index Python and text/configuration files. Use 40-line windows, stride 20,
  and preserved physical source lines.
- Query only the workflow name and seven log lines around the first
  case-insensitive `error`, or the final seven lines if absent.
- Use identifier/snake/camel tokenization with Okapi BM25 (`k1=1.2`, `b=0.75`)
  and stable ties. Rank at most 30 positive chunks. Path identifiers are indexed
  too, since logs often mention a module/file path.
- Reserve all 2,048 output tokens and 256 safety tokens. Count the actual model
  chat template while greedily packing complete chunks. Skip chunks overlapping
  already-supplied oracle edit blocks. Never truncate a source chunk.
- Skip inference when even the baseline prompt exceeds the input budget. Save
  this as an unapplied repair and retain it in metric denominators. Malformed
  or truncated responses also become unapplied repairs without extra attempts.

## Alternative evaluation

The local evaluator replaces remote CI execution with transparent **proxies**:

| Metric | Meaning |
| --- | --- |
| Exact reference repair | All edited files equal the files reconstructed from the dataset patch. This misses valid alternative repairs. |
| Edit-token precision/recall/F1 | Multiset overlap of added/removed tokens, including path and edit direction; unchanged context earns no credit. |
| Response application rate | The answer provides valid replacements for every target block. This is format/application validity, not repair correctness. |
| Nonempty patch rate | The model actually changes at least one target file. |
| Syntax regression | Python 3.13 `ast.parse` fails after an edit on a file that parsed before. No imports or tests run. |
| Retrieval file recall | Fraction of reference-changed files represented before and after evidence packing, evaluated after prediction. |

These metrics **are not CI Pass@1**, semantic validation, or paper-score
reproduction. Old dependencies, platform-specific failures, formatting rules,
and behavior across multiple files can only be fully checked by actual builds
or tests. Unchanged files trivially preserve syntax, so reports also expose
syntax preservation among nonempty edits and its denominator.

## Artifacts and provenance

Large raw artifacts stay in the ignored benchmark `out/rag-local/<signature>/`:
`run_metadata.json`, per-record retrieval evidence, per-method/seed prompts,
raw responses, patches, metrics, and explicit failure records. The manifest
records the Arrow checksum, model configuration checksum, package versions,
settings, selected IDs, and implementation hashes. `metrics.json` reports
coverage and per-seed results plus means and population standard deviations.
Small completed metrics and a human-readable report are copied into `results/`.

The current cached split has 68 records; its identity is recorded, rather than
assuming it matches a historical version. The original publication does not
specify CI repair seeds or decoding settings; the settings above define this
experiment. Retrieval hyperparameters are fixed before generation and are
not selected using reference repairs.
