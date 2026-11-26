# 🏟️ Long Code Arena — Bug Localization

This directory contains the [Long Code Arena](https://huggingface.co/spaces/JetBrains-Research/long-code-arena) **Bug Localization** benchmark and our Fall 2026 extensions comparing retrieval, long-context, and agentic approaches.

**Task:** given a GitHub issue description (`issue_title`, `issue_body`) and the repository snapshot at `base_sha` (the commit where the bug is reproducible), identify and rank the project files that need to be modified to fix the bug.

---

## 📁 Directory Overview

| Path | Description | Key Docs |
|---|---|---|
| [`baseline_eval/`](baseline_eval/) | Python `py/test` TF-IDF + NLTK baseline replication and shared ranking evaluator (`evaluate_tfidf.py`) | [run_all_tfidf.py](baseline_eval/run_all_tfidf.py), [evaluate_tfidf.py](baseline_eval/evaluate_tfidf.py) |
| [`rag/`](rag/) | Standalone two-stage BM25 chunk retrieval + LLM reranking pipeline (Steps 1–5) | [README](rag/README.md), [METHODS](rag/METHODS.md) |
| [`long_context/`](long_context/) | Controlled 5-condition comparison of `issue_only`, `direct` whole-file packing, and `rag` chunk packing | [README](long_context/README.md), [METHODS](long_context/METHODS.md), [RESULTS](long_context/RESULTS.md) |
| [`agentic/`](agentic/) | Multi-turn bounded JSON action-loop prototype (`search_code`, `read_file`, `submit_ranking`) | [README](agentic/README.md), [RESULTS](agentic/RESULTS.md), [SMOKE_RESULTS](agentic/SMOKE_RESULTS.md) |
| [`src/`](src/) | Original JetBrains LCA data-collection scripts, exploratory notebooks, and chat-based baselines | [data/preprocessing](src/data/preprocessing/README.md), [data/hf](src/data/hf/README.md) |
| [`configs/`](configs/) | Hydra configurations and prompt templates for the original LCA chat baselines | [run.yaml](configs/run.yaml), [eval.yaml](configs/eval.yaml) |
| [`tests/`](tests/) | Original issue-link parsing tests (additional unit suites live inside `rag/tests/`, `long_context/tests/`, and `agentic/tests/`) | — |

---

## 🧭 Approaches Implemented

1. **Original LCA Baselines (`src/baselines/`, `baseline_eval/`)**
   - **TF-IDF + NLTK (`baseline_eval/`)**: Replicates the archived LCA TF-IDF baseline (`977d200d`) on the 50-issue `py/test` split. Reads `.py` blobs directly from Git at `base_sha` (strict UTF-8, excluding paths containing `test/` or `tests/` case-insensitively) and ranks all candidate files by cosine similarity. Its scorer ([`evaluate_tfidf.py`](baseline_eval/evaluate_tfidf.py)) is reused across all newer experiments.
   - **Original Chat Baselines (`src/baselines/`)**: One-shot LLM set-prediction with `issue_only`, `filepath` (BM25-ordered paths), and `filepath_imports` (tree-sitter imports) context composers.
2. **Standalone BM25 + LLM RAG ([`rag/`](rag/README.md))**
   - Splits candidate `.py` files into 40-line windows (stride 20), indexes them with a code-aware ASCII identifier tokenizer and local Okapi BM25 (`k1=1.2`, `b=0.75`), aggregates file scores by **max chunk score**, packs top chunks into the model's context budget, and prompts an LLM to rerank the retrieved candidate files (with remaining candidates appended in BM25 order).
3. **Controlled Source-Context Study ([`long_context/`](long_context/README.md))**
   - Compares **`issue_only`** (0 source tokens), **`direct`** (lexicographic whole-file packing at 8,192 and 65,536 source budgets), and **`rag`** (BM25 chunk packing at 8,192 and 65,536 source budgets) under identical controls: the full issue, the complete lexicographic candidate path list, `gpt-4.1-2025-04-14`, strict JSON schema output, and a shared BM25 completion tail.
4. **Agentic Multi-Turn Investigation ([`agentic/`](agentic/README.md))**
   - Starts from the same issue, full candidate path list, and initial BM25 snippets (4,096-token budget), then runs a bounded JSON action loop (up to 6 tool calls) allowing `search_code(query)`, `read_file(path, start_line, end_line)`, or early `submit_ranking(paths)`. Valid partial rankings receive the same shared BM25 completion tail.

---

## 📊 Summary Results (`py/test`, 50 Issues, `gpt-4.1-2025-04-14`)

All rows use the shared `.py` non-test candidate filter, full-ranking evaluator ([`evaluate_tfidf.py`](baseline_eval/evaluate_tfidf.py)), and shared BM25 completion tail. Twenty ground-truth files across 19 issues fall outside the candidate filter and remain misses in AP denominators.

| Approach / Condition | Source Budget\* | MAP (Full) | MRR | Hit@1 | Hit@5 | Hit@10 | Full BM25 Fallbacks | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `issue_only` | 0 | 0.604859 | 0.848872 | 74.0% | 98.0% | 98.0% | 1 / 50 | [long_context/RESULTS.md](long_context/RESULTS.md) |
| `direct` (small) | 8,192 | 0.623651 | 0.877333 | 78.0% | **100.0%** | **100.0%** | 0 / 50 | [long_context/RESULTS.md](long_context/RESULTS.md) |
| `direct` (large) | 65,536 | 0.613600 | 0.853000 | 76.0% | **100.0%** | **100.0%** | 1 / 50 | [long_context/RESULTS.md](long_context/RESULTS.md) |
| `rag` (small) | 8,192 | 0.637463 | 0.867627 | 78.0% | 98.0% | 98.0% | 0 / 50 | [long_context/RESULTS.md](long_context/RESULTS.md) |
| **`rag` (large)** | **65,536** | **0.651043** | **0.882576** | 80.0% | 98.0% | 98.0% | **0 / 50** | [long_context/RESULTS.md](long_context/RESULTS.md) |
| **`agentic` (≤6 tools)** | **4,096** | 0.615779 | 0.878144 | **82.0%** | 94.0% | 96.0% | 1 / 50 | [agentic/RESULTS.md](agentic/RESULTS.md) |

> **Notes:**
> - \*Configured source caps used the conservative UTF-8 byte-count fallback because `tiktoken 0.8.0` did not recognize `gpt-4.1-2025-04-14`.
> - The five `long_context` arms were evaluated together in a single controlled run (Nov 8, 2026; 2.82M total tokens across 250 requests). The `agentic` arm was evaluated separately on the same 50 issues (Nov 9, 2026; 1.87M total tokens across 279 requests and 229 tool calls); see [`agentic/RESULTS.md`](agentic/RESULTS.md) for cross-run caveats.

---

## 💾 Setup & Local Verification

Run commands from the **repository root** with Python 3.11:

```powershell
# Minimal environment for RAG, long-context, and agentic pipelines:
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r bug_localization/requirements-rag.txt

# Optional: add TF-IDF baseline dependencies:
.\.venv\Scripts\python.exe -m pip install -c bug_localization/requirements.txt nltk scikit-learn

# Optional: full original LCA baseline dependencies:
.\.venv\Scripts\python.exe -m pip install -r bug_localization/requirements.txt
```

Run the offline unit test suites (no API key or network required):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s bug_localization/rag/tests -v
.\.venv\Scripts\python.exe -m unittest discover -s bug_localization/long_context/tests -v
.\.venv\Scripts\python.exe -m unittest discover -s bug_localization/agentic/tests -v
```

---

## 🤗 Dataset

All data is hosted on [HuggingFace 🤗 (`JetBrains-Research/lca-bug-localization`)](https://huggingface.co/datasets/JetBrains-Research/lca-bug-localization):

```python
from datasets import load_dataset

# Configurations: "py", "java", "kt", "mixed"
# Splits: "dev" (all), "test" (manually curated), "train" (dev minus test)
dataset = load_dataset("JetBrains-Research/lca-bug-localization", "py", split="test")
```

- **Configurations:** `py` (only `.py` files in diff), `java` (only `.java`), `kt` (only `.kt`), `mixed` (at least one `.py`/`.java`/`.kt` plus possibly other extensions).
- **Splits:** `dev` (all collected data), `test` (manually selected data; [labeling artifacts](https://docs.google.com/spreadsheets/d/1cEyFHjse-iUYQlUO7GO5KpqkvJ3wu6vheou4W61TMOg/edit?usp=sharing)), `train` (all collected data not in `test`).
- **Original LCA Baselines:** Embedding-based ([TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [GTE](https://huggingface.co/thenlper/gte-large), [CodeT5](https://huggingface.co/Salesforce/codet5p-110m-embedding), [BM25](https://github.com/dorianbrown/rank_bm25)) and chat-based ([GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5-turbo), [GPT-4](https://platform.openai.com/docs/models/gpt-4), plus DeepSeek/Claude/Llama/Qwen configs under [`configs/backbone/`](configs/backbone/)).
