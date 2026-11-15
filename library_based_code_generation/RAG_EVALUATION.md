# Saved RAG evaluation

This evaluator adopts module summarization's paired analysis, bug localization's
matched-cohort reporting, and CI repair's saved-artifact coverage checks. CI repair
currently exposes a saved-job baseline evaluator, not a RAG-specific evaluator.
LBCG keeps its own official ChrF and API Recall implementations; it does not use a
summary judge or CI pass rate as a substitute for code-generation metrics.

Run from `library_based_code_generation/` in the Python environment used for the
original RAG run:

```powershell
python -m src.evaluation.rag_analysis
```

This loads the test dataset and scores the existing five GPT-4.1 result directories:
plain, official BM25, dense, hybrid, and hybrid with normalized tokens. It never
constructs a model client, requests embeddings, or generates predictions. Loading
the dataset may access Hugging Face. No OpenAI credentials are needed.

To score entirely from local data:

```powershell
python -m src.evaluation.rag_analysis --dataset-file path/to/test.arrow
# JSONL test data is also accepted:
python -m src.evaluation.rag_analysis --dataset-file path/to/test.jsonl
```

Use `--revision SHA` when the original dataset revision is known. The legacy
numbered predictions lack generation-time dataset, prompt, snapshot and response
hashes. Their row mapping therefore remains an explicit assumption even when all
150 prediction files exist. Model/split metadata and row ranges are checked;
duplicate/unexpected indices are rejected. The new manifest fingerprints the
evaluation dataset, scorer source and prediction bytes for future audits, but
cannot retroactively establish generation provenance.

Outputs under `rag_evaluation/`:

Published paths are relative to `library_based_code_generation/`; external dataset
locations retain only `external/<filename>`. To migrate older reports without
rescoring, run `python -m src.evaluation.rag_analysis --sanitize-paths-only`.

- `REPORT.md` and `comparison.csv`: scores on the same completed examples.
- `metrics.json`: coverage, missing/empty files, paired differences, 95% bootstrap
  intervals, wins/losses/ties, syntax-valid and zero-API-recall counts, and differences
  from the old aggregate scores when the full cohort is present.
- `per_example_metrics.json`: unchanged ChrF/API Recall for every scored prediction.
- `review_cases.json`: largest API Recall gains/losses against plain and official
  BM25, with instructions and paths for manual review. These are review selections,
  not inferred failure causes.
- `evaluation_manifest.json` and `prediction_hashes.json`: evaluation provenance.

All arms are compared to plain and official BM25 on their common scored cohort.
Missing and empty predictions are reported and excluded from that common cohort;
partial results are never marked complete. Invalid Python is still scored by the
official metrics; syntax validity is a separate diagnostic. ChrF/API Recall do not
establish execution correctness. Older aggregates are retained without overwriting
their metadata. Usage, costs, latency and generation truncation cannot be recovered
from the existing raw `.py` files and are explicitly marked unavailable.

The default 10,000-draw bootstrap (seed 42) resamples individual examples. If your
dataset has a library/repository identifier, pass `--group-field FIELD` to resample
whole libraries with all their paired tasks, as module summarization does. The
field must be a nonempty string on every row. Sample-level intervals do not account
for shared libraries. Intervals are exploratory, without multiple-test correction.

Tests require only the standard library:

```powershell
python -m unittest discover -s tests -p test_rag_analysis.py -v
```
