# Bug Localization RAG — Methods & Specifications

Technical design specifications for the standalone RAG pipeline in [`bug_localization/rag/`](README.md).

---

## 1. Repository Snapshot & Chunking Policy (Step 1)

- **Buggy commit (`base_sha`)**: Data collection saves each PR's `base.sha` as `base_sha` and `head.sha` as `head_sha` ([`prepare_data_for_analysis.py`](../src/data/preprocessing/prepare_data_for_analysis.py)). Preparation reads committed Git blobs directly at `base_sha` via [`run_all_tfidf.py`](../baseline_eval/run_all_tfidf.py) (`get_repository` and `read_content`) without checking out or executing repository code. `head_sha`, diffs, and fix commits are never read.
- **Candidate filter (Python-only)**: Matches the baseline TF-IDF runner — case-sensitive `.py` blobs, excluding paths containing `test/` or `tests/` case-insensitively (a substring rule: `contest/a.py` is excluded while `test_root.py` is kept). Strict UTF-8 decoding is required; decode errors fail preparation explicitly. Repository-relative `/` separators are enforced, and empty/whitespace-only candidate files remain in `candidate_paths` in sorted order. Java, Kotlin, and mixed policies are not implemented here.
- **Sliding-window chunker (`chunk_files(files, window=40, stride=20)`)**:
  - 1-based inclusive line numbers (`1–40`, `21–60`, `41–80`, `61–100` for a 100-line file); stops once a window reaches EOF.
  - Requires integer `window > 0` and `0 < stride <= window`.
  - Whitespace-only windows are skipped; identical chunks across different files are retained.
  - Preserves exact indentation, source characters, and CR/LF line endings (a trailing newline does not create an extra empty line).
  - Sequential `chunk_id`s starting at 0, ordered by `(path, start_line)`.
- **Label isolation**: `prepare_snapshot(repository, repository_identity, base_sha, ...)` accepts only code provenance; issue text, labels, and diffs never enter snapshot preparation or cache keys.
- **Cache layout (`outputs/prepared_snapshots/<sha256>/`)**:
  - `chunks.jsonl`: one chunk record per line (`chunk_id`, `path`, `start_line`, `end_line`, `text`).
  - `metadata.json`: provenance, `format_version`, `candidate_policy`, chunking settings, complete `candidate_paths`, counts, and `chunks_sha256`.
  - Keyed by repository identity, full resolved `base_sha`, window/stride, and policy/format versions. Writes use unique temporary directories and atomic replace with `metadata.json` written last.

---

## 2. Tokenization & BM25 Chunk Indexing (Step 2)

### Exact Lexical Tokenization Policy ([`retrieval_tokenizer.py`](retrieval_tokenizer.py))
Both chunk text and bug-report queries pass through the same `tokenize(text)` function:

1. Extract ASCII identifiers matching `[A-Za-z_][A-Za-z0-9_]*`. Bare numbers are ignored; digits inside identifiers are retained. Punctuation and non-ASCII characters delimit matches.
2. Before lowercasing, split on underscores and camelCase boundaries (`(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])`, e.g., `HTTPServer` → `HTTP`, `Server`). Discard empty components.
3. Emit the lowercase whole identifier, followed by lowercase components that differ from the whole (so `ordinary` is emitted once, while `foo_foo` emits `foo_foo, foo, foo` and `_name` emits `_name, name`).
4. Filter out tokens shorter than 2 characters and lowercase Python hard keywords (`keyword.kwlist`, including `true`, `false`, `none`; soft keywords are kept). No stemming or natural-language stopword removal is applied.

| Input | Emitted Tokens |
|---|---|
| `parse_expression` | `parse_expression, parse, expression` |
| `loadUserProfile` | `loaduserprofile, load, user, profile` |
| `HTTPServer` | `httpserver, http, server` |
| `parseXMLFile` | `parsexmlfile, parse, xml, file` |
| `ordinary ordinary` | `ordinary, ordinary` |
| `if x: return None` | *(none)* |

Comments, docstrings, and string literals are tokenized with the same rules; file paths are kept as metadata and do not enter the chunk index.

### BM25 Scoring ([`bm25.py`](bm25.py), [`bm25_index.py`](bm25_index.py))
Each **nonempty tokenized chunk** is one BM25 document (`k1=1.2`, `b=0.75`; requires finite `k1 > 0` and `0 <= b <= 1`). With $N$ indexed chunks, term frequency $\text{tf}$, document frequency $\text{df}$, chunk length $\text{dl}$, and mean length $\text{avgdl}$:

```text
IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score(chunk, query) = sum over distinct query tokens t (in sorted order) of:
    IDF(t) * tf(t, chunk) * (k1 + 1)
    / (tf(t, chunk) + k1 * (1 - b + b * dl / avgdl))
```

- Distinct query terms are summed in sorted order so floating-point accumulation is deterministic across Python hash seeds.
- Only positive-scoring chunks are returned, sorted descending by score with exact ties broken by original `chunk_id`.
- Chunks with zero remaining tokens are excluded from BM25 (`excluded_count`) but retained in `snapshot.chunks` and `index.chunks_by_id`.
- **Cache layout (`outputs/bm25_indexes/<sha256>/`)**: stores `index.json` (`chunk_ids`, `doc_len`, `postings`) and `metadata.json` (identity, stats, `index_sha256`). Reloading validates step-1 cache integrity plus posting/length invariants without re-tokenizing source.

---

## 3. Query Construction & BM25 File Aggregation (Step 3)

- **Column projection**: Prediction reads only `repo_owner`, `repo_name`, `base_sha`, `issue_url`, `issue_title`, and `issue_body`. Arrow datasets are projected to these columns **before** row materialization so `changed_files`, `head_sha`, and diffs are never loaded during prediction.
- **Query (`make_query`)**: Joins nonblank `issue_title` and `issue_body` with `\n\n`, preserving stack traces, code blocks, indentation, and Markdown verbatim. Null/blank fields contribute nothing.
- **File ranking (`retrieve(index, query, max_chunks=30)`)**:
  - Scores **every** indexed chunk before applying `max_chunks`.
  - Aggregates chunk scores to files via **maximum chunk score** so overlapping windows never double-count credit.
  - Every candidate file appears once in `ranking` (descending score, ties lexicographic by path); unmatched or empty files receive `0.0`.
  - `retrieved_chunks` holds up to `max_chunks` highest-scoring positive chunks (descending score, ties by `chunk_id`). Changing `max_chunks` cannot alter the full BM25 file ranking.

---

## 4. Prompt Packing, Model Client & Ranking Validation (Steps 4–5)

### Model Interface ([`model.py`](model.py))
- Uses OpenAI Chat Completions (`sdk_max_retries=0`) with explicit bounded transport retries on connection errors, 408/409/429, and 5xx responses (exponential backoff `1, 2, 4, ...`s capped at 8s, up to `max_retries`).
- Uses `max_tokens` for legacy `gpt-4` / `gpt-4-0613` / `gpt-4-0314` and `max_completion_tokens` for newer models.
- Never serializes exception bodies, headers, or credentials; non-retryable errors fall back to BM25 immediately. No automatic response-format switching or LLM repair retries.

### Prompt & Token Accounting ([`ranking_prompt.py`](ranking_prompt.py), [`prompts/file_ranking_v1.txt`](prompts/file_ranking_v1.txt))
- **Prompt structure**: System message defines the JSON ranking task and marks all issue/source text as untrusted data. User message is a compact JSON object containing `issue_title`, `issue_body`, `allowed_candidate_paths` (only distinct files represented in the packed chunks, ordered by BM25), and `retrieved_code` chunks.
- **Token counting (`TokenCounter`)**: Uses `tiktoken.encoding_for_model(model_name)` when available; otherwise falls back to **1 token per UTF-8 byte** (a conservative upper bound for byte-BPE models, also selectable via `--tokenizer bytes`). Adds `16 tokens/message + 32/request` plus serialized `response_format` JSON, and reserves `max_output_tokens + max(safety_margin, ceil(context_window * 0.05))`.
- **Issue truncation**: If the fixed prompt + issue (with zero evidence) exceeds `input_budget`, deterministically trims suffixes from `issue_body` first, then `issue_title`, in blocks of $\lceil\text{remaining}/8\rceil$ characters, recounting after each step.
- **Whole-chunk packing**: Iterates retrieved chunks by descending BM25 score (`chunk_id` tie-break), rebuilding and recounting the full request including the updated `allowed_candidate_paths`. Chunks that fit are included whole; oversized chunks are skipped and later chunks are still tried. Empty retrieval or zero fitting chunks records `fallback_no_evidence` without calling the API.

### Validation, Completion & Resume ([`file_ranking.py`](file_ranking.py), [`run_ranking.py`](run_ranking.py), [`run_benchmark.py`](run_benchmark.py))
- **Strict JSON validation (`validate_response`)**: Requires a JSON object with exact key `ranked_files: list` (rejects duplicate JSON keys, `NaN`/`Infinity`, Markdown fences, and extra fields). Paths are normalized (`\` → `/`) and must exactly match `allowed_candidate_paths`.
- **Repairs vs. full fallback**: Invalid entries and duplicates are removed; valid partial rankings append omitted `allowed_candidate_paths` in BM25 order (`status="repaired"`). Refusals, truncated outputs (`finish_reason != "stop"`), malformed JSON, or all-invalid lists fall back completely to the BM25 ranking (`bm25_fallback=true`).
- **Complete universe permutation (`complete_ranking`)**: Validated LLM candidates lead the final ranking, followed by all remaining step-1 candidates in BM25 order. Final `score` values are descending ordinal integers satisfying the evaluator's descending-score contract.
- **Persistence & resume**: Run directories are keyed by a canonical SHA-256 manifest covering dataset fingerprint, selected indices, retrieval/model/prompt settings, tokenizer identity, and source file hashes. Artifacts are written atomically with `payload_sha256` checksums. `run_benchmark.py` preserves both valid model successes and completed fallbacks on `--resume` (use `--retry-fallbacks` to archive and retry fallbacks).

---

## 5. Evaluation Metrics & Coverage ([`evaluate_retrieval.py`](evaluate_retrieval.py), [`evaluate_ranking.py`](evaluate_ranking.py), [`evaluate_benchmark.py`](evaluate_benchmark.py))

Labels are loaded **only** during evaluation, after prediction artifacts exist, and written solely under `evaluation/`.

- **Full-ranking metrics**: Computed via unchanged [`evaluate_tfidf.py`](../baseline_eval/evaluate_tfidf.py) (`score_example` and `aggregate`): `MAP_full_ranking`, `MRR`, `Hit@1/5/10`, single-file `Recall@1`, multi-file `Recall@2` / `Precision@2`, and macro `F1@k` (`k=1` for single-file, `k=2` for multi-file). All raw `changed_files` stay in denominators; outside-candidate gold paths count as misses.
- **Retrieval coverage (before & after packing)**: Reports macro candidate recall, at-least-one retrieved, and all-retrieved against both **eligible** gold files (excluding zero-eligible examples from eligible macro means) and **all raw** gold files.
- **Fallbacks & failures**: Primary headline metrics include all completed BM25 fallbacks (`bm25_fallback=true`). Unrecoverable preparation/pipeline failures have no ranking and are disclosed in `pipeline_failures` and coverage counts.
- **Matched baseline comparison**: Comparing TF-IDF, BM25, and RAG in `comparison.csv` validates identical dataset identity/fingerprint, split, raw-label policy, base SHA, candidate universe, and gold labels, restricting the three-way table to common matched examples.
