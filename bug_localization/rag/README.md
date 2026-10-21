# Bug localization RAG

Repository preparation, tokenization, BM25 indexing, tests, and generated caches live in this directory. Commands below run from the repository root.

## Repository preparation for RAG (step 1)

From the repository root, with the existing `datasets` and `GitPython`
dependencies installed, prepare one Python test example:

```shell
python bug_localization/rag/prepare_repositories.py --limit 1
```

Use `--start 10 --limit 3` for a subset, or `--all` for the full split.
`--split` accepts `test` (default), `train`, and `dev`; `--window 40 --stride 20`
sets the line windows. `--cache-dir` overrides the prepared-output directory;
`--dataset-cache-dir` controls the Hugging Face dataset cache. The CLI reports
repository, resolved commit, candidate/chunk counts, and cache location for
each example. It continues after individual failures and returns a nonzero
exit status if any fail. This component currently matches the local **Python**
TF-IDF runner; Java, Kotlin, and mixed candidate policies are not implemented.

Snapshot and candidate semantics:

- The dataset supplies `base_sha` directly. Data collection saves the pull
  request's `base.sha` as `base_sha` and `head.sha` as `head_sha` in
  [`prepare_data_for_analysis.py`](../src/data/preprocessing/prepare_data_for_analysis.py).
  [`HFDataSource`](../src/baselines/data_sources/hf_data_source.py) and the local
  TF-IDF runner load `base_sha` as the buggy snapshot. Preparation follows that
  convention, without deriving a parent commit or reading `head_sha` or diffs.
- The unchanged [`run_all_tfidf.py`](../baseline_eval/run_all_tfidf.py)
  `get_repository` and `read_content` utilities resolve/obtain repository history
  under `bug_localization/repos/<owner>__<name>` and read committed blobs without
  checkout. Missing history may trigger a GitHub clone/fetch; a missing old
  commit may require the benchmark's archived repository in that directory.
- Candidates are exactly that runner's case-sensitive `.py` blobs, excluding
  paths containing `test/` or `tests/`, case-insensitively. This is a substring
  rule: it also excludes `contest/a.py`, while including `test_root.py`.
  Decoding is strict UTF-8; errors fail preparation rather than silently
  dropping files. No further filters are added. This preserves the local
  runner's portability changes relative to the older checkout/default-encoding
  loader; it does not claim equivalence to the original paper environment.
- The in-memory source representation is a path-to-full-text mapping from the
  baseline reader. Paths use repository-relative `/` separators. All candidate
  paths, including empty and whitespace-only files, are saved in sorted order.

`chunk_files(files, window=40, stride=20)` returns records with `chunk_id`,
`path`, `start_line`, `end_line`, and `text`. Line numbers are **1-based and
inclusive**; 100 lines yield 1–40, 21–60, 41–80, and 61–100. The loop stops when
a window reaches EOF. Window and stride must be integers satisfying
`window > 0` and `0 < stride <= window`. Whitespace-only windows are skipped;
identical chunks from different files are retained. Paths and line ranges
determine chunk order, with sequential IDs starting at zero. Text preserves
indentation, source characters, and CR/LF line endings, including final
newlines; a terminal newline does not create a phantom source line.

Adriana's [`rag/chunker.py`](../../project_level_code_completion/rag/chunker.py)
provides the same window/stride loop convention, but uses zero-based positions,
token counts, optional parsing, and cross-file deduplication. Preparation uses
an independent small chunker to meet this task's line and evidence requirements.
No tokenization, BM25, prompting, LLM calls, or rankings are performed.

`prepare_snapshot(repository, repository_identity, base_sha, ...)` accepts only
code provenance and returns a `PreparedSnapshot` exposing `metadata`, `chunks`,
`candidate_paths`, `cache_path`, and `reused`. Its metadata contains repository
identity, full resolved `base_sha`, candidate paths/counts, and chunking settings.
The CLI passes only `repo_owner`, `repo_name`, and `base_sha` to repository loading.
Issue text, labels, changed-file lists, and fix diffs are never saved or used.

Caches live under `bug_localization/rag/outputs/prepared_snapshots/<sha256>/`:

- `metadata.json`: provenance, format version, candidate policy, chunk settings,
  complete candidate paths, file/chunk counts, and the chunk file checksum.
- `chunks.jsonl`: one chunk record per line.

The cache key includes repository identity, **full resolved commit SHA**, window,
stride, format version, and relevant filtering/decoding settings. Metadata,
checksum, counts, record fields, paths, ordering, IDs, and line ranges are
validated before reuse. Invalid/incomplete caches are rebuilt. Unique temporary
files are replaced atomically, with metadata written last as the completion
marker. If the reader or preparation semantics change, update the policy/format
version to invalidate previous caches.

Run focused checks without adding a test dependency:

```shell
python -m unittest discover -s bug_localization/rag/tests -p test_prepare_repositories.py -v
```

The suite covers windows and content preservation, empty candidates, duplicates,
determinism, cache invalidation/corruption/interruption, CLI subsets/failures, and
a local Git fixture checking buggy revision, baseline candidate parity, decoding
failure, and preservation of dirty working files. Git fixture tests skip when
GitPython is unavailable.

Validation on the locally cached Python test example 0 (`thealgorithms/python`,
`dbfc220264e514cbc94320b6e4769acfaea85fad`) matched all **195** TF-IDF candidate
paths, verified every one of **533** chunks against the recorded source lines,
and reused the cache on a second call. That check read the cached dataset Arrow
file and existing repository directly; fresh network dataset loading and cloning,
and the remaining benchmark examples, were not exercised.

## Tokenization and BM25 chunk indexing (step 2)

This step consumes an **existing prepared snapshot directory**. It needs only
Python's standard library and the local code-completion BM25 module; no dataset,
GitPython, model, or network access is needed. It does not run the benchmark or
aggregate chunks into file scores. Code completion, preparation, and TF-IDF
behavior are unchanged.

From the repository root, build the locally available example's index:

```shell
python bug_localization/rag/bm25_index.py bug_localization/rag/outputs/prepared_snapshots/a1434f887950d7db316650c45529b716d30a7e4d81bbeece52b6c6e2ce4145f6
```

Build or reload it and run a diagnostic query:

```shell
python bug_localization/rag/bm25_index.py bug_localization/rag/outputs/prepared_snapshots/a1434f887950d7db316650c45529b716d30a7e4d81bbeece52b6c6e2ce4145f6 --query "binary search" --top-k 3 --k1 1.2 --b 0.75
```

For another snapshot, substitute the directory printed by step 1. The module
entry point `python -m bug_localization.rag.bm25_index` accepts the same
arguments. `--cache-dir` overrides `bug_localization/rag/outputs/bm25_indexes/`.
The CLI prints repository/SHA, indexed and excluded counts, vocabulary size,
average token length, BM25 settings, cache status/location, and optional matches
with scores, original IDs, paths, and inclusive line ranges. Invalid prepared
input fails clearly with a nonzero exit code; regenerate it using step 1.

### Exact tokenization policy

[`retrieval_tokenizer.py`](retrieval_tokenizer.py) provides the
single `tokenize(text)` function used for both chunk text and queries:

1. Find ASCII word/identifier matches using `[A-Za-z_][A-Za-z0-9_]*`. Bare
   numbers are ignored; digits within identifiers are retained. Other
   punctuation and non-ASCII characters separate matches. Underscores stay in
   the whole identifier and also delimit its components.
2. Before lowercasing, split at lowercase-letter/digit to uppercase-letter
   transitions and before an uppercase letter followed by lowercase letters
   within an acronym (e.g. `HTTPServer` becomes `HTTP`, `Server`). Split on
   underscores too; discard empty components.
3. Emit the lowercase whole identifier, then lowercase components that differ
   from that whole. This avoids emitting `ordinary` twice. Keep repeated source
   occurrences and repeated components: `foo_foo` emits `foo_foo, foo, foo`.
4. Remove tokens shorter than two characters and Python hard keywords, after
   lowercasing. The stop set is the lowercase runtime `keyword.kwlist`, including
   `true`, `false`, and `none`; soft keywords are retained. No stemming or
   natural-language stopword removal is performed.

Examples:

| Input | Tokens |
| --- | --- |
| `parse_expression` | `parse_expression, parse, expression` |
| `loadUserProfile` | `loaduserprofile, load, user, profile` |
| `HTTPServer` | `httpserver, http, server` |
| `parseXMLFile` | `parsexmlfile, parse, xml, file` |
| `ordinary ordinary` | `ordinary, ordinary` |
| `if x: return None` | none |

Comments, docstrings, and string literals are scanned with the same rules,
without parsing or executing code. Only chunk **text** enters the index;
file paths remain source metadata. This ASCII lexical policy is an initial
limitation for non-English prose and Unicode identifiers.

The extraction/splitting expressions come from Adriana's
[`rag/bm25.py`](../../project_level_code_completion/rag/bm25.py). That tokenizer is
unchanged; the new task-specific function filters after lowercasing and retains
a component such as `name` in `_name` even when there is only one component.

### Scoring and Python API

[`bm25_index.py`](bm25_index.py) reuses Adriana's `BM25Index`
constructor, inverted postings, and scorer directly. A BM25 document is one
**nonempty tokenized chunk**. Defaults are `k1=1.2`, `b=0.75`; require finite
`k1 > 0` and `0 <= b <= 1`. With `N` indexed chunks, term frequency `tf`,
document frequency `df`, chunk token length `dl`, and mean length `avgdl`:

```text
IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score(chunk, query) = sum over distinct query tokens t of:
    IDF(t) * tf(t, chunk) * (k1 + 1)
    / (tf(t, chunk) + k1 * (1 - b + b * dl / avgdl))
```

This is the requested non-negative IDF, with natural logarithms. Code term
frequencies include repeated terms, and every distinct query term contributes
once. Terms are summed in sorted order for reproducible floating-point results
across Python hash seeds. Only positive matches are returned, descending by
score, with exact ties ordered by original chunk ID. Empty, fully filtered,
or unmatched queries return `[]`; oversized limits return all matches. A zero
limit returns `[]`; negative or non-integer limits raise `ValueError`.

```python
from bug_localization.rag.bm25_index import (
    load_prepared_snapshot, build_or_load_index,
)

snapshot = load_prepared_snapshot("path/to/prepared_snapshots/<key>")
index = build_or_load_index(snapshot, k1=1.2, b=0.75)
results = index.search("parser fails to loadUserProfile", max_results=5)
# Each result: chunk_id, score, path, start_line, end_line, text.
# All source evidence, including excluded chunks:
chunk = index.chunks_by_id[0]
```

`index.bm25` exposes `n_docs`, `doc_len`, `avg_len`, and term `postings` containing
`(dense_document_index, term_frequency)` pairs. `index.document_frequency` gives
each term's posting count; `index.chunk_ids` maps dense indexes to original
step-1 IDs. `index.excluded_count` counts chunks with no remaining tokens.
Excluded chunks stay available in `snapshot.chunks` and `index.chunks_by_id`.
Candidate paths, source text, IDs, and provenance are never rewritten. A snapshot
with no indexable chunks has zero vocabulary and average length, and searches
return `[]` safely.

### Persistence and validation

Each index lives in `bug_localization/rag/outputs/bm25_indexes/<sha256>/` (ignored
by Git). The content-addressed identity includes a fingerprint of **all prepared
metadata**, a fingerprint of the actual chunk records, tokenizer version and
effective settings (including regexes and keyword list), scoring/implementation
version, persistence format version, `k1`, and `b`. Changing those inputs creates
a separate cache. Bump the tokenizer or scoring version when changing semantics
that are not represented by settings.

- `index.json`: JSON object with `chunk_ids`, `doc_len`, and `postings` mapping
  terms to arrays of `[dense_document_index, term_frequency]` pairs. These are
  sufficient statistics to restore the index without re-tokenizing source.
- `metadata.json`: complete identity, indexed/excluded/vocabulary counts, average
  chunk length, and SHA-256 checksum of `index.json`. Document frequencies are
  reconstructed from posting-list lengths.

Loading first validates step 1's original cache using its existing validator.
Index reuse then validates identity, schema, checksum, counts, average length,
ID membership/order, positive term frequencies, posting order, and token totals
against chunk lengths. Invalid or incomplete **indexes** are rebuilt from the
prepared snapshot. Unique temporary files are replaced atomically, with
metadata written last as the completion marker; missing or mismatched files
cannot pass validation. Reloading reads/hashes source evidence for validation,
but performs no source tokenization. JSON and source records are currently held
in memory; larger-scale storage is outside this step.

### Focused validation

```shell
python -m unittest discover -s bug_localization/rag/tests -p test_bm25_index.py -v
python -m unittest discover -s bug_localization/rag/tests -p test_prepare_repositories.py -v
```

The step-2 suite checks lexical examples, shared query/code tokenization,
repetitions, empty and unknown queries, limits, empty/filtered snapshots,
metadata/ID preservation after exclusions, deterministic ties and cross-process
hash seeds, hand-computed TF/DF/length-normalized scores, reload without source
tokenization, invalidation, corrupt caches, interrupted writes, and CLI output.

The local prepared `thealgorithms__python` snapshot at
`dbfc220264e514cbc94320b6e4769acfaea85fad` indexed **533** chunks, excluded **0**,
and produced **3,426** terms with mean length **106.448405**. The diagnostic query
`binary search` returned `searches/binary_search.py` lines **1-40**, **21-60**,
and **41-80**, scoring **10.71538223**, **10.47295327**, and **10.41151613**.
Reloading returned identical matches/scores. This exercised one cached snapshot,
without new dataset downloads, repository cloning, LLM integration, file
reranking, or full-split retrieval/evaluation.

Validation passed all **18** step-2 tests with the default Python interpreter.
The **16** step-1 regression tests also passed using the existing sibling
`lca-baselines/.venv` environment with GitPython installed (the default
interpreter skips its two Git fixtures). No new dependencies were installed.
