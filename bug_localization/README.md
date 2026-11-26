# 🏟️ Long Code Arena — Bug Localization

Bug localization benchmark from [Long Code Arena](https://huggingface.co/spaces/JetBrains-Research/long-code-arena) (`JetBrains-Research/lca-bug-localization`), extended with retrieval, long-context, and agentic approaches on the 50-issue Python test split (`py/test`).

**Task:** given a GitHub issue (`issue_title`, `issue_body`) and the repository snapshot at `base_sha`, rank the files that must be modified to fix the bug.

---

## 🧭 Approaches

| Approach | Folder | How It Works | Docs |
|---|---|---|---|
| **TF-IDF + NLTK Baseline** | [`baseline_eval/`](baseline_eval/) | Replicates the archived LCA TF-IDF cosine-similarity ranker (`977d200d`) over `.py` Git blobs at `base_sha` (excluding `test/` & `tests/`). Its evaluator ([`evaluate_tfidf.py`](baseline_eval/evaluate_tfidf.py)) scores all approaches below. | [run_all_tfidf.py](baseline_eval/run_all_tfidf.py) |
| **Original Chat Baselines** | [`src/baselines/`](src/baselines/) | Upstream LCA one-shot LLM set-prediction (`issue_only`, BM25-sorted `filepath`, tree-sitter `filepath_imports`) configured via [`configs/`](configs/). | [configs/run.yaml](configs/run.yaml) |
| **Standalone BM25 + LLM RAG** | [`rag/`](rag/) | 40-line sliding windows (stride 20) → code-aware BM25 (`k1=1.2`, `b=0.75`, max-chunk file scoring) → packs top chunks for an LLM (`gpt-4`, 8k default) to rerank retrieved files, appending remaining candidates in BM25 order. | [README](rag/README.md), [METHODS](rag/METHODS.md) |
| **Controlled Long-Context Study** | [`long_context/`](long_context/) | Compares **`issue_only`** (0 source tokens), **`direct`** (lexicographic whole-file packing), and **`rag`** (all positive BM25 chunks) at 8,192 and 65,536 source budgets under identical `gpt-4.1-2025-04-14` controls, full candidate path lists, and a shared BM25 completion tail. | [README](long_context/README.md), [METHODS](long_context/METHODS.md), [RESULTS](long_context/RESULTS.md) |
| **Agentic Multi-Turn Loop** | [`agentic/`](agentic/) | Starts from the same issue, full candidate list, and initial BM25 snippets (4,096 budget), then runs a bounded JSON action loop (≤6 tool calls) with `search_code`, `read_file`, and `submit_ranking`, plus the same BM25 tail. | [README](agentic/README.md), [RESULTS](agentic/RESULTS.md), [SMOKE_RESULTS](agentic/SMOKE_RESULTS.md) |

---

## 📊 Results & Comparison (`py/test`, 50 Issues, `gpt-4.1-2025-04-14`)

All rows use the same `.py` non-test candidate filter, full-ranking evaluator, and BM25 completion tail. Twenty ground-truth files across 19 issues fall outside the candidate filter and remain misses in AP denominators.

| Condition | Source Budget\* | MAP (Full) | MRR | Hit@1 | Hit@5 | Hit@10 | Fallbacks | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `issue_only` | 0 | 0.6049 | 0.8489 | 74% | 98% | 98% | 1 / 50 | [long_context/RESULTS.md](long_context/RESULTS.md) |
| `direct` (small) | 8,192 | 0.6237 | 0.8773 | 78% | **100%** | **100%** | 0 / 50 | [long_context/RESULTS.md](long_context/RESULTS.md) |
| `direct` (large) | 65,536 | 0.6136 | 0.8530 | 76% | **100%** | **100%** | 1 / 50 | [long_context/RESULTS.md](long_context/RESULTS.md) |
| `rag` (small) | 8,192 | 0.6375 | 0.8676 | 78% | 98% | 98% | 0 / 50 | [long_context/RESULTS.md](long_context/RESULTS.md) |
| **`rag` (large)** | **65,536** | **0.6510** | **0.8826** | 80% | 98% | 98% | **0 / 50** | [long_context/RESULTS.md](long_context/RESULTS.md) |
| **`agentic` (≤6 tools)** | **4,096** | 0.6158 | 0.8781 | **82%** | 94% | 96% | 1 / 50 | [agentic/RESULTS.md](agentic/RESULTS.md) |

### Key Takeaways
- **RAG (65k) is the strongest overall ranker:** highest MAP (`0.6510`) and MRR (`0.8826`) with zero fallbacks. Scaling the budget from 8k → 65k improved RAG (+0.0136 MAP, +2 pp Hit@1) because BM25 orders extra context by relevance.
- **Direct long-context degrades at larger budgets:** `direct` dropped from `0.6237` (8k) to `0.6136` (65k) MAP (flat on the 48-issue model-success subset) because lexicographic whole-file packing fills extra space with alphabetically early, irrelevant files.
- **Agentic wins Hit@1 (82%) but trails on MAP (`0.6158`) at ~3.3× token cost:** interactive search/read pinpoints the primary culprit well (41/50 Hit@1), but focusing on a few files with a smaller 4k initial payload hurts multi-file recall and Hit@5/10 while using **1.87M tokens** (37.4k/issue, 279 requests) vs. **2.82M tokens across all five one-shot arms combined** (~11.3k/request).
- **Strong zero-code floor:** `issue_only` reaches `0.6049` MAP and `98%` Hit@5 because the model still sees all repository file paths and omitted candidates receive the BM25 tail.
- **Caveats & other baselines:** \*Source caps used the UTF-8 byte fallback (`tiktoken 0.8.0` did not recognize `gpt-4.1-2025-04-14`). The `agentic` arm ran separately from the 5-arm `long_context` study (see [`agentic/RESULTS.md`](agentic/RESULTS.md) for matched-arm notes). Full 50-issue TF-IDF and standalone `gpt-4` RAG outputs are local-only (`outputs/` is gitignored).

---

## 💾 Setup, Tests & Dataset

Run from the **repository root** with Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r bug_localization/requirements-rag.txt

# Optional: TF-IDF baseline or full original LCA dependencies
.\.venv\Scripts\python.exe -m pip install -c bug_localization/requirements.txt nltk scikit-learn
.\.venv\Scripts\python.exe -m pip install -r bug_localization/requirements.txt

# Run all 115 offline unit tests (no API key required)
.\.venv\Scripts\python.exe -m unittest discover -s bug_localization/rag/tests -v
.\.venv\Scripts\python.exe -m unittest discover -s bug_localization/long_context/tests -v
.\.venv\Scripts\python.exe -m unittest discover -s bug_localization/agentic/tests -v
```

Dataset: [HuggingFace 🤗 (`JetBrains-Research/lca-bug-localization`)](https://huggingface.co/datasets/JetBrains-Research/lca-bug-localization) — configs `py`, `java`, `kt`, `mixed`; splits `dev`, `train`, `test` ([labeling artifacts](https://docs.google.com/spreadsheets/d/1cEyFHjse-iUYQlUO7GO5KpqkvJ3wu6vheou4W61TMOg/edit?usp=sharing)).
