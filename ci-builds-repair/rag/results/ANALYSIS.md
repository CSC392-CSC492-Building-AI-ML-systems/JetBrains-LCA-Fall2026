# Interpretation of the completed experiment

All 68 cached test records were evaluated for baseline and RAG at seeds 0, 1,
and 2: **408 evaluation cases**, with **zero pipeline/source failures**.
Six cases exceeded the fixed input budget, leaving 402 actual model calls.
No GitHub Actions evaluator, workflow, project test suite, or project code was
executed. The evaluator used reference-edit comparison and AST parsing only.

| Outcome across three seeds | Baseline (204 cases) | BM25 RAG (204 cases) |
| --- | ---: | ---: |
| Valid replacement response | 123 (60.29%) | 68 (33.33%) |
| Invalid/missing block markers | 65 | 77 |
| Output token limit reached | 13 | 56 |
| Input prompt over budget | 3 | 3 |
| Exact reference target-text matches | 0 | 0 |
| Mean edit-token F1 ± population SD across seeds | 0.1952 ± 0.0025 | 0.1179 ± 0.0091 |

RAG **did not improve reference-edit agreement in this experiment**: the mean
F1 difference was -0.0774. It helped on some individual records, but more
answers violated the output contract or reached the output limit. Failures
remain in the primary denominator and receive zero edit credit. Responses
were not repaired with additional LLM calls or selected using gold answers.

The saved generation/packing steps total about 1.22 hours for baseline and
3.17 hours for RAG, excluding snapshot preparation and model loading. This
configuration therefore added substantial inference time while reducing
average agreement. This is evidence about this particular model, prompt,
retrieval configuration, dataset, and proxy metric; it is not a general claim
that retrieval harms repair or a statistical significance test.

Syntax preservation is deliberately separate from repair quality. Unchanged
fallbacks trivially preserve syntax, and even syntactically valid edits can
break imports, formatting, tests, or behavior. The report exposes preservation
among nonempty edits and its denominator rather than claiming a build pass.
No prediction exactly reproduced all reference target-file text; valid
alternative repairs can nevertheless differ from the reference.

One dataset record (45) adds a file; all six of its responses were unapplied.
There are no deleted-file reference cases in this cached split. Reference
metrics compare target text, not filesystem existence or environment behavior.

## Audit files

- `REPORT.md`: per-seed comparison, aggregate metrics, and evaluation limits.
- `per_example_metrics.csv`: all 408 result rows with IDs, errors, retrieval
  coverage, syntax checks, timing, and prediction checksums.
- `metrics.json` and `diagnostics.json`: aggregate values and failure counts.
- `run_metadata.json`: generation settings, dataset checksum, source hashes.
- `runtime_identity.json`: model weights/tokenizer checksums, model Hub revision,
  actual runtime versions, base Git commit, and imported BM25 dependency hash.

The exporter verified every recorded generation-source hash against the files
used in this checkout. Raw prediction checksums were validated before report
generation. Prompts, responses, retrieved code, and patches remain under the
ignored `ci-builds-repair-benchmark/out/rag-local/2d9349bb16247814246b/` directory.

The reference paper's CI pass percentage cannot be inferred from these results.
This experiment uses a 4-bit conversion, reconstructed prompts, a different
cached test set size, declared seeds/decoding settings, and offline proxies.
