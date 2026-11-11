# Module Summarization: first RAG implementation

The experiment asks whether selecting relevant code helps the existing model write better module documentation than simply showing it the beginning of the supplied code. It compares two conditions with the same model, original prompt, temperature 0, maximum 2,000 output tokens, and at most 2,000 GPT-2 tokens of code:

| Condition | Code given to the model |
| --- | --- |
| Prefix baseline | First 2,000 tokens of the benchmark's supplied code context |
| BM25 RAG | Chunks ranked against the documentation intent and document filename, packed within the same budget |

The October 22 pilot used `gpt-3.5-turbo-0125`. On November 10, 2026, the generation endpoint returned HTTP 404 for that model. The full run uses `gpt-4.1-mini` for both conditions. Retrieval and API generation run without a local GPU. CompScore uses a local Mistral judge on a CUDA machine.

## Scope and design

Retrieval searches each task's **supplied `relevant_code_context`**, not a cloned repository. The benchmark has already selected relevant source material. This tests selection within that material; it does not demonstrate finding a module across thousands of arbitrary repository files.

BM25 uses 400-token windows with 50-token overlap, normalizes identifiers such as snake_case and camelCase, and removes repeated source spans when assembling the context. Queries use only intent and filename. The reference documentation (`target_text`) is used only during evaluation. Source offsets, hashes, token counts, fallbacks, prompts' hashes, returned model, usage, and predictions are saved for inspection. Generic queries with no matching terms fall back to the prefix and are explicitly logged.

This first version uses token windows rather than syntax-aware functions/classes; chunks may cut across definitions. Keyword retrieval can miss semantic matches, and its identifier extraction primarily handles ASCII names. The two conditions share a maximum context allowance, but may use different actual token counts. Neither condition receives the reference documentation during generation.

The smoke configuration chooses five tasks from five repositories. The pilot chooses 20 tasks from 20 repositories, using a fixed seed and balancing documentation type, source length, and generic versus specific intent where possible. These are development samples, not a held-out test after tuning. The full configuration selects all 216 tasks from 43 repositories. Original dataset row IDs are retained for scoring.

## Run from a repository checkout

Run these commands from the repository's `module_summarization/` directory with a Python 3.10+ environment. Set `RAG_DATA` to the benchmark test data (Arrow or JSONL) and `RAG_TOKENIZER` to the GPT-2 tokenizer JSON. Keep API credentials in a private env file outside Git.

```bash
python -m pip install -r rag_pipeline/rag_requirements.txt

# Offline retrieval
python -m rag_pipeline.rag_replication \
  --mode retrieve --config rag_pipeline/rag_configs/full-gpt41-mini.json \
  --dataset "$RAG_DATA" --tokenizer "$RAG_TOKENIZER" \
  --output-root rag_runs --cache-dir rag_runs/index_cache

# Paid generation
python -m rag_pipeline.rag_replication \
  --mode generate --config rag_pipeline/rag_configs/full-gpt41-mini.json \
  --dataset "$RAG_DATA" --tokenizer "$RAG_TOKENIZER" \
  --env-file "$RAG_ENV_FILE" \
  --output-root rag_runs --cache-dir rag_runs/index_cache \
  --ledger rag_runs/spending.sqlite
```

The directory reorganization changed the implementation fingerprint. The original completed run is under `benchmark_results/`. Use that run directory for analysis or scoring of its saved predictions.

All paid experiments must share **the same `rag_runs/spending.sqlite` ledger**. The $4 cap covers requests made through this runner, not unrelated account activity. Before each request it reserves a conservative input/output cost bound, then settles using returned usage and the documented model price. Do not delete the ledger or change its path to restart a run. Timeouts keep their reservation and stop execution; reconcile them manually instead of blindly retrying. The runner has no automatic retries. An implementation/configuration change makes a new run identity and can produce new charges.

Use the printed run directory for analysis:

```bash
python -m rag_pipeline.rag_analyze \
  --run-dir "rag_runs/PASTE_RUN_DIRECTORY_NAME" --dataset "$RAG_DATA"
python -m rag_pipeline.rag_compscore \
  --run-dir "rag_runs/PASTE_RUN_DIRECTORY_NAME" --dataset "$RAG_DATA"
```

The second command only prepares an official-scoring plan. It does not download a judge or run a GPU. See `RAG_GPU_GUIDE.md` before adding `--execute`.

## Evaluate and interpret

The local analyzer checks source provenance and reports whitespace-token ROUGE-L F1 as a diagnostic. Lexical overlap alone does not measure factual correctness or establish an improvement. Official CompScore uses the repository's existing Mistral judge, reference documentation, original 6,000-token judge context, and probability-based metric. The judge receives the same source context for both generation conditions. The full GPU run is recorded in [RAG_FULL_RUN_RESULTS.md](RAG_FULL_RUN_RESULTS.md).

Next, review paired outputs for missing functionality, unsupported claims, irrelevant code, and output truncation. Include at least a generic query, a long module, a short module, and a no-match fallback. Before claiming a benefit, run official scoring and compare paired differences; report variability and results by repository, source length, documentation type, and length × documentation type. Record where retrieval loses as well as wins. Repeated source contexts and repository overlap mean individual examples are not independent, so use repository-level resampling for uncertainty in a larger evaluation. Do not tune on all tasks and then present the same tasks as unseen evaluation.

## Files and verification

- `bm25_retrieval/rag_retrieval.py`: BM25, chunking, tokenizer, cache and source spans.
- `rag_pipeline/rag_replication.py`: selection, generation, reproducibility and API access check.
- `rag_pipeline/rag_budget.py`: shared persistent spending guard.
- `rag_pipeline/rag_analyze.py`: source audit and diagnostic scores.
- `rag_pipeline/rag_compscore.py`: preparation and optional original metric execution.
- `rag_pipeline/rag_configs/`: smoke, pilot and full configurations.
- `rag_pipeline/tests/test_rag.py`: offline retrieval, baseline parity and budget tests.

Run tests with `python -m unittest discover -s rag_pipeline/tests -v` from `module_summarization/`, using the lightweight environment. Measured experiment results and timings are in `RAG_FIRST_RUN_RESULTS.md`.

Sources: [benchmark dataset](https://huggingface.co/datasets/JetBrains-Research/lca-module-summarization), [model and pricing](https://developers.openai.com/api/docs/models/gpt-3.5-turbo). The original baseline and metric source files in this repository determine the comparison behavior.
