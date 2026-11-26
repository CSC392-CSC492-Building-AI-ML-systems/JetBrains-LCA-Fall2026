# Bug Localization RAG Pipeline

This directory implements the standalone two-stage **BM25 + LLM RAG** pipeline for the Python (`py`) bug-localization benchmark. All commands run from the **repository root**.

For detailed specifications — snapshot/candidate semantics, exact lexical tokenization rules, BM25 formula, prompt-budget packing, JSON validation/repair, cache integrity, and metric definitions — see **[METHODS.md](METHODS.md)**.

> **Note:** For the controlled comparison against direct long-context and issue-only baselines on `gpt-4.1-2025-04-14`, see [`../long_context/`](../long_context/README.md). This standalone `rag/` pipeline exposes only files represented in the packed chunks to the LLM (defaulting to `gpt-4` with an 8,192-token window), whereas `long_context` supplies the full repository candidate list.

---

## Architecture (Steps 1–5)

| Step | Module(s) | Purpose | Default Output Directory |
|---|---|---|---|
| **1. Prepare** | [`prepare_repositories.py`](prepare_repositories.py) | Read `.py` Git blobs at `base_sha` (excluding `test/` & `tests/`) and chunk into 40-line windows (stride 20) | `bug_localization/rag/outputs/prepared_snapshots/` |
| **2. Index** | [`retrieval_tokenizer.py`](retrieval_tokenizer.py), [`bm25.py`](bm25.py), [`bm25_index.py`](bm25_index.py) | Tokenize code identifiers (snake_case + camelCase) and build a content-addressed Okapi BM25 chunk index (`k1=1.2`, `b=0.75`) | `bug_localization/rag/outputs/bm25_indexes/` |
| **3. Retrieve** | [`retrieval.py`](retrieval.py), [`run_retrieval.py`](run_retrieval.py), [`evaluate_retrieval.py`](evaluate_retrieval.py) | Query BM25 with `issue_title + "\n\n" + issue_body`, score files by max chunk score, keep top-30 chunks, and evaluate BM25-only ranking | `bug_localization/rag/outputs/bm25_retrieval/` |
| **4. Rerank** | [`ranking_prompt.py`](ranking_prompt.py), [`model.py`](model.py), [`file_ranking.py`](file_ranking.py), [`run_ranking.py`](run_ranking.py), [`evaluate_ranking.py`](evaluate_ranking.py) | Pack top chunks into the model budget ([`prompts/file_ranking_v1.txt`](prompts/file_ranking_v1.txt)), request JSON file ranking, validate/repair, and append BM25 tail | `bug_localization/rag/outputs/llm_ranking/` |
| **5. Benchmark** | [`run_benchmark.py`](run_benchmark.py), [`evaluate_benchmark.py`](evaluate_benchmark.py) | End-to-end sequential, resumable 50-issue `py/test` runner with optional TF-IDF / BM25 / RAG comparison CSV | `bug_localization/outputs/rag_python50/` |

---

## Setup

Use Python 3.11 with [`../requirements-rag.txt`](../requirements-rag.txt) (pins `openai==1.60.1`, `datasets==2.15.0`, `pyarrow>=14,<21`, `GitPython`, `tiktoken`, etc.):

```powershell
# Only needed if .venv does not already exist:
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r bug_localization/requirements-rag.txt
.\.venv\Scripts\python.exe -m pip check
```

For live LLM inference (Steps 4–5), set `OPENAI_API_KEY` in your process environment (`.env` files are not loaded and credentials are never serialized):

```powershell
$keyInput = Read-Host 'OPENAI_API_KEY' -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $keyInput).Password
Remove-Variable keyInput
```

---

## Step-by-Step Usage

### Step 1: Prepare repository snapshots

```powershell
# Smoke run on example 0 (py/test):
.\.venv\Scripts\python.exe bug_localization/rag/prepare_repositories.py --limit 1

# Subset or full split:
.\.venv\Scripts\python.exe bug_localization/rag/prepare_repositories.py --split test --start 0 --limit 5 --window 40 --stride 20
.\.venv\Scripts\python.exe bug_localization/rag/prepare_repositories.py --split test --all
```

Key flags: `--split {test,train,dev}`, `--start`, `--limit`, `--all`, `--window` (40), `--stride` (20), `--cache-dir`, `--dataset-cache-dir`.

### Step 2: Build or query a BM25 chunk index

Pass the snapshot directory printed by Step 1:

```powershell
.\.venv\Scripts\python.exe -m bug_localization.rag.bm25_index bug_localization/rag/outputs/prepared_snapshots/<snapshot_sha256> --query "binary search" --top-k 3 --k1 1.2 --b 0.75
```

### Step 3: Run BM25 retrieval and evaluation

```powershell
# Small subset with evaluation:
.\.venv\Scripts\python.exe -m bug_localization.rag.run_retrieval --split test --start 0 --limit 3 --max-chunks 30 --window 40 --stride 20 --k1 1.2 --b 0.75 --evaluate

# Full 50-example Python test split:
.\.venv\Scripts\python.exe -m bug_localization.rag.run_retrieval --split test --all --max-chunks 30 --evaluate
```

To evaluate a saved retrieval run separately:

```powershell
$outputRoot = "bug_localization/rag/outputs/bm25_retrieval"
$runName = (Get-Content "$outputRoot/latest_run.json" | ConvertFrom-Json).run_directory
.\.venv\Scripts\python.exe -m bug_localization.rag.evaluate_retrieval "$outputRoot/$runName/summary.json"
```

### Step 4: LLM reranking over saved retrieval predictions

[`ranking_config.json`](ranking_config.json) defaults to `gpt-4` (8,192 context window, 2,048 max output tokens, `temperature=0`, `response_format="text"`). Any field can be overridden via CLI flags (`--model`, `--context-window`, `--max-output-tokens`, `--temperature`, `--top-p`, `--seed`, `--response-format`, `--tokenizer`, `--safety-margin`, `--timeout-seconds`, `--max-retries`).

```powershell
$retrieval = "bug_localization/rag/outputs/bm25_retrieval"
$config = "bug_localization/rag/ranking_config.json"

# 1. Dry run (builds and saves prompts; no API key or model call):
.\.venv\Scripts\python.exe -m bug_localization.rag.run_ranking $retrieval --config $config --dry-run --limit 1 --output-dir bug_localization/rag/outputs/llm_prompts

# 2. Live single-example run (requires OPENAI_API_KEY):
.\.venv\Scripts\python.exe -m bug_localization.rag.run_ranking $retrieval --config $config --start 0 --limit 1 --output-dir bug_localization/rag/outputs/llm_ranking

# 3. Evaluate an existing non-dry run (optionally comparing against TF-IDF):
.\.venv\Scripts\python.exe -m bug_localization.rag.evaluate_ranking bug_localization/rag/outputs/llm_ranking --tfidf-summary bug_localization/outputs/tfidf_python50
```

### Step 5: End-to-end 50-issue Python test benchmark

[`run_benchmark.py`](run_benchmark.py) chains Step 3 retrieval and Step 4 reranking sequentially over `py/test` using [`benchmark_config.json`](benchmark_config.json):

```powershell
# Dry run (1 example, no model call):
.\.venv\Scripts\python.exe -m bug_localization.rag.run_benchmark --dry-run --limit 1

# Paid smoke run (2 examples) + evaluation:
.\.venv\Scripts\python.exe -m bug_localization.rag.run_benchmark --limit 2

# Paid full 50-example run (resumable; preserves completed successes and fallbacks):
.\.venv\Scripts\python.exe -m bug_localization.rag.run_benchmark --all --resume

# Optional: explicitly retry completed BM25 fallbacks (archives prior attempts under attempts/):
.\.venv\Scripts\python.exe -m bug_localization.rag.run_benchmark --all --resume --retry-fallbacks

# Evaluate-only mode on a saved run (no inference or repository access):
.\.venv\Scripts\python.exe -m bug_localization.rag.run_benchmark --evaluate-only bug_localization/outputs/rag_python50/<run_signature>
```

**Offline options:** Steps 3–5 accept `--dataset-file <path-to-arrow>` (pass the same file to evaluation), `--dataset-cache-dir`, `--repository-cache-dir` (read-only local clones, no fetch), and `--tokenizer bytes` (offline UTF-8 byte counter).

**Generating the matched TF-IDF baseline for comparison:**

```powershell
.\.venv\Scripts\python.exe -m pip install -c bug_localization/requirements.txt nltk scikit-learn
.\.venv\Scripts\python.exe -m bug_localization.baseline_eval.run_all_tfidf
```

Supply the resulting run directory via `--tfidf-summary` (defaults to `bug_localization/outputs/tfidf_python50`). Comparison enforces matching dataset fingerprint, split, base commits, candidate universes, and gold labels.

---

## Local Verification & Tests

Run the full RAG unit test suite (90 tests covering preparation, tokenization, BM25 scoring/caching, retrieval, prompt packing, validation/repair, retries, resume invalidation, and evaluation):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s bug_localization/rag/tests -v
```

*(The two Git fixture tests skip automatically if `GitPython` is not installed.)*

### Recorded Smoke Checks
- **Step 1–2 snapshot check (`py/test` index 0, `thealgorithms/python`):** matched all 195 TF-IDF candidate paths, generated 533 chunks (0 excluded, 3,426 vocabulary terms, mean length 106.45 tokens). Diagnostic query `"binary search"` returned `searches/binary_search.py` lines 1–40, 21–60, and 41–80 as the top 3 matches.
- **Step 3 offline BM25 smoke check (`py/test` indices 0–2):** 3/3 succeeded with 0 failures — macro candidate recall 0.6667, mean 13.67 distinct retrieved files (out of 30 chunks max), BM25 full-ranking MAP/MRR 0.3763, Hit@1/5/10 0.3333. *(Smoke subset only; not a full-benchmark estimate.)*
