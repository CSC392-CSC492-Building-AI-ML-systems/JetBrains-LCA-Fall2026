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

## Bug-report retrieval and BM25 file ranking (step 3)

[`retrieval.py`](retrieval.py) implements label-free query construction and
retrieval. [`run_retrieval.py`](run_retrieval.py) loads benchmark examples,
matches snapshots/indexes, and saves resumable predictions.
[`evaluate_retrieval.py`](evaluate_retrieval.py) reads labels only after those
predictions exist, reusing the original TF-IDF evaluator's scoring functions.
All generated artifacts stay under `rag/outputs/` by default. No prompts,
LLM calls, generation, title weighting, query rewriting, embeddings, overlap
merging, or token-budget packing are performed.

### Query and snapshot matching

The verified dataset fields are `issue_title` and `issue_body`. The query is
the nonblank title followed by a blank line (`\n\n`) and the nonblank body.
Missing, null, or whitespace-only fields contribute nothing. The original
contents are otherwise preserved, including stack traces, identifiers,
Markdown/code blocks, indentation, CR/LF, and links. Both fields must be strings
or null; invalid types produce explicit failures. The step-2 tokenizer processes
this query with exactly the same rules as chunk text.

Prediction input is explicitly limited to `repo_owner`, `repo_name`, `base_sha`,
`issue_url`, `issue_title`, and `issue_body`. Arrow datasets are projected to
these columns **before reading any example rows**. Dataset labels, `head_sha`,
diffs, fix commit messages, and fixed code are not prediction inputs. An example
is identified by its split, zero-based dataset index, and issue URL, within a
run carrying the dataset fingerprint.

A full SHA first looks up step 1's exact content-addressed snapshot and validates
its provenance, candidate policy, chunking, and checksum. A validated snapshot
already records the resolved full SHA, so this path needs no Git access. A cache
miss or abbreviated SHA goes through the existing step-1 Git resolution and
preparation API; repository handles are closed afterward. Repository identity,
resolved SHA, chunk settings, and step-2 index metadata must agree before
retrieval. Matching examples reuse the same prepared/index cache identities.
The search always operates on one matching repository revision.

### Evidence and complete file rankings

`retrieve(index, query, max_chunks=30)` returns:

- `retrieved_chunks`: up to the requested number of positive matches, each with
  `rank`, original `chunk_id`, `score`, `path`, inclusive `start_line`/`end_line`,
  and exact `text`. Order is descending score, then original chunk ID.
- `ranking`: every candidate once, as `{rank, file, score}` records compatible
  with the existing file-ranking evaluator. The score is the **maximum** chunk
  score for that file, computed across **all indexed chunks**, before limiting
  the evidence list. Files without matches or indexable chunks, including empty
  candidates, receive zero. Order is descending score, then lexicographic path.
- `candidate_count` and `distinct_retrieved_files`: the complete candidate
  universe size and the distinct file count within the selected evidence.

Multiple chunks from one file are allowed. Overlaps never accumulate extra
credit because aggregation uses maximum rather than sum. The chunk limit cannot
change the complete file ranking. Empty, fully filtered, and unmatched queries
produce no evidence and a complete zero-score ranking in path order. A zero
chunk limit also preserves full-ranking scores. Negative/non-integer limits
are rejected.

### Run and evaluate

With `datasets` and `GitPython` available, run a small subset from the repository
root:

```shell
python -m bug_localization.rag.run_retrieval --split test --start 0 --limit 3 --max-chunks 30 --window 40 --stride 20 --k1 1.2 --b 0.75 --evaluate
```

The Python test split has **50** examples. To explicitly run the full experiment:

```shell
python -m bug_localization.rag.run_retrieval --split test --all --max-chunks 30 --window 40 --stride 20 --k1 1.2 --b 0.75 --evaluate
```

`--split` also accepts `train` and `dev`. `--all` selects all examples from
`--start`; otherwise `--limit` defaults to one. The preparation candidate policy
is still Python-only. Default cache/output locations are:

- `--prepared-cache-dir`: `bug_localization/rag/outputs/prepared_snapshots`
- `--index-cache-dir`: `bug_localization/rag/outputs/bm25_indexes`
- `--output-dir`: `bug_localization/rag/outputs/bm25_retrieval`

`--output-dir` is the **parent** of content-addressed run folders. The CLI prints
the actual run location; `latest_run.json` records its folder name. Resume is
on by default (`--resume`); use `--no-resume` to recompute predictions. Failed
examples are retried. Missing repositories follow step 1's existing clone/fetch
behavior unless `--repository-cache-dir` points to existing clones; that option
only reads their Git histories, with no checkout or fetch. `--dataset-cache-dir`
controls dataset loading. `--dataset-file` instead reads an existing Arrow file
offline; it must contain the selected split in the original example order.

Evaluate an existing run separately, using PowerShell to resolve the latest
output directory:

```powershell
$outputRoot = "bug_localization/rag/outputs/bm25_retrieval"
$runName = (Get-Content "$outputRoot/latest_run.json" | ConvertFrom-Json).run_directory
$runDir = "$outputRoot/$runName"
python -m bug_localization.rag.evaluate_retrieval "$runDir/summary.json"
```

If retrieval used `--dataset-file` or `--dataset-cache-dir`, pass the same option
to evaluation. Evaluation checks the dataset fingerprint and allowed example
inputs against the saved predictions. The original standalone evaluator can
also consume the separately saved labeled rankings for the Python test split:

```powershell
python bug_localization/baseline_eval/evaluate_tfidf.py "$runDir/evaluation/summary.json" --allow-partial
```

Omit `--allow-partial` for a complete 50-example run. Its console heading remains
the original TF-IDF heading; the saved summary identifies the BM25 baseline.

### Persistence, resume, and label separation

The run folder is keyed by a canonical-JSON SHA-256 of the manifest, including
the dataset fingerprint, split/selected indices, tokenizer/scoring versions,
chunking and BM25 settings, retrieval/query/aggregation policies, and hashes of
the implementation and reused scoring/reader source files.

- `run_metadata.json`: complete run manifest.
- `examples/000.json`, etc.: label-free predictions. Successful records contain
  the example index/URL, repository and resolved buggy SHA, query text, requested
  maximum chunks, snapshot metadata fingerprint/cache path, BM25 identity/cache
  key/path, tokenizer/BM25 settings, evidence chunks, complete file ranking,
  candidate count, distinct retrieved file count, timing, and identity signature.
- `summary.json`: selected indices, dataset/selected totals, successful and
  failed counts, processed/remaining counts, reused count, successful result
  paths, and explicit failure records. Failed example files have `status=failed`
  and an error, rather than a successful empty retrieval.

Prediction files carry `payload_sha256` over all other fields. Unique temporary
files are atomically replaced after serialization. Resume requires a successful
record with a valid checksum, exact current input/config/snapshot/index identity,
complete candidate coverage, valid score/order/count fields, and evidence that
still matches prepared source. Corrupt, incomplete, stale, or failed outputs are
recomputed. Query changes invalidate per-example identity even for callers
without a dataset fingerprint. Relevant code/settings changes select a new run.

Labels are saved **only under `evaluation/`**, after prediction:

- `metrics.json`: file-ranking metrics, candidate-retrieval macro metrics,
  denominators/coverage, prediction/evaluation failure counts and details.
- `per_example_metrics.json`: original ranking metrics plus candidate recall,
  outside-candidate gold paths, retrieved chunk count, and distinct file count.
- `examples/000.json`, etc., and `summary.json`: labeled ranking records in the
  original evaluator's expected format. Prediction artifacts stay unchanged.

### Metrics and coverage

The original `evaluate_tfidf.score_example` and `aggregate` functions provide
MAP, MRR, single-file Recall@1, multiple-file Recall@2/Precision@2, F1 at the
established cutoff, and Hit@1/5/10. Their definitions and saved ranking order
are unchanged. **All raw gold paths** stay in those metric denominators;
outside-candidate gold paths remain missed predictions.

Candidate-file recall instead divides retrieved **eligible gold files** by
gold files within the complete candidate universe. Repeated evidence chunks
from a file count once. Gold matching follows the existing evaluator's exact
path equality, with no additional slash/case normalization or test-file filter.
Outside-candidate paths are listed separately per example and counted globally.
Macro candidate recall, the fraction retrieving at least one eligible file,
and the fraction retrieving all eligible files use evaluated examples with at
least one eligible gold file. Zero-eligible examples receive null recall/hit
values and are explicitly counted outside those applicable denominators.

Both prediction and evaluation failures are reported alongside metrics, with
coverage over the requested examples. Metrics are computed on valid successes;
failures are never silently treated as successful empty retrievals or dropped
from the reported counts. The CLI returns nonzero for either failure type.
The evidence limit is a number of **chunks**, while recall and distinct-file
counts operate on **files**.

### Step-3 checks and local smoke results

```shell
python -m unittest discover -s bug_localization/rag/tests -v
```

The step-3 tests cover query preservation/null fields, label/fix access guards,
Arrow projection before row loading, exact snapshot/revision/index matching,
multiple chunks per file, maximum aggregation, ranking independence from limits,
all-candidate coverage, empty/unknown/filtered queries, deterministic ties,
candidate-recall arithmetic and macro denominators, zero-eligible cases, failure
reporting/retry, output corruption, and resume invalidation for settings,
tokenizer changes, query changes, and changed prepared evidence. The evaluator
adapter is checked against the original standalone evaluator.

The available cached Python test dataset contains 50 examples. A local smoke
run over indices **0-2**, using existing Git histories and no network, completed
**3/3** predictions with **0** prediction or evaluation failures:

| Index | Repository | Retrieved chunks (limit 30) | Distinct files | Candidate files |
| --- | --- | ---: | ---: | ---: |
| 0 | `thealgorithms/python` | 30 | 23 | 195 |
| 1 | `keras-team/keras` | 30 | 9 | 553 |
| 2 | `serverless/serverless` | 12 | 9 | 10 |

Candidate recall macro was **0.666667**, at-least-one and all-eligible fractions
were both **0.666667**, and mean distinct retrieved files was **13.666667**.
BM25 full-ranking MAP/MRR were **0.376263**, Recall@1/F1 were **0.333333**, and
Hit@1/5/10 were each **0.333333**. All three cases had one raw gold file and no
gold paths outside the candidate universe; multiple-file metrics were therefore
not applicable. These are smoke-subset results, not full-benchmark estimates.

The exact local offline command (PowerShell) uses the existing sibling virtual
environment and repository cache:

```powershell
$python = "C:/Users/Admin/Desktop/Projects/lca-baselines/.venv/Scripts/python.exe"
$arrow = "C:/Users/Admin/.cache/huggingface/datasets/JetBrains-Research___lca-bug-localization/py/0.0.0/f2bb26a1c3320ccc1d7b02d2630077ff327a80f6/lca-bug-localization-test.arrow"
$repos = "C:/Users/Admin/Desktop/Projects/lca-baselines/bug_localization/repos"
& $python -m bug_localization.rag.run_retrieval --dataset-file $arrow --repository-cache-dir $repos --limit 3 --max-chunks 30 --evaluate
```

All **59** RAG tests passed using the existing environment with GitPython
(**25** step-3 checks plus **34** step-1/2 checks). All **72** retrieved smoke
chunks were checked against the original prepared text, paths, IDs, and line
ranges, and each full ranking covered its complete candidate universe. A second
smoke invocation resumed all three predictions with identical results/metrics.

The full 50-example experiment was not run. This version retains step 1's
Python-only candidates and step 2's ASCII lexical tokenizer. Step 3 leaves
evidence unpacked; step 4 below adds the LLM budget and ranking stage.

## Complete Python test benchmark (step 5)

Run from the repository root with `python -m bug_localization.rag.run_benchmark`.
The runner requires the **50-row Python test split**, even when selecting a
smoke subset. It retains original dataset indices and derives filesystem-safe
example IDs from label-free issue/repository inputs. Dataset fingerprint,
selected inputs, source hashes, candidate policy, unchanged retrieval settings,
tokenizers, model settings, prompt identity, and evaluation policy determine
the run signature. Full mode requires indices 0 through 49.

Step 5 uses [`benchmark_config.json`](benchmark_config.json), defaulting to
**gpt-4**, with an 8,192-token context window, text output containing the prompted
JSON ranking, and no seed parameter. These defaults match step 4's
`ranking_config.json`. Model settings remain
configurable through JSON or CLI overrides. A model-only override does not
automatically change context limits or response-format support.

For each example, the runner invokes step 3's existing `run_examples` with
the **original index**, reusing snapshot and BM25 index caches. It passes the
validated label-free retrieval artifact to step 4's `pack_prompt` and `predict`.
No retrieval setting, prompt, scoring rule, or baseline implementation changes.
Model requests are sequential, at most one logical request per example plus
bounded transport retries. Results are saved immediately and atomically.
Prediction reads only the established input-column allowlist; evaluation
reads labels afterwards and retains unreachable labels as misses.

Outputs live in `bug_localization/outputs/rag_python50/<sha256>/`:

- `run_metadata.json`: deterministic configuration, dataset and code identity.
- `examples/<index>_<id>.json`: step-4 result schema, exact packed messages,
  submitted messages, raw response, validation/repairs, full BM25 and final
  rankings, model usage/timing, retrieval provenance, and retry history.
- `retrieval/<signature>/`: step-3 artifacts and metadata, referenced by absolute
  summary path. Prepared snapshots and indexes remain in their shared caches.
- `summary.json`: updated after every example; processed, remaining, failed,
  successful model rankings, completed fallbacks, dry prompts and reuse counts.
- `metrics.json`, `comparison.csv`, and `evaluation/`: generated after real
  predictions or by evaluate-only mode. Dry runs produce no evaluation files.
- `attempts/`: previous completed fallbacks archived when explicitly retried.

`successful` retains step 4's completed-artifact convention; use
`successful_model_rankings`, `completed_bm25_fallbacks`, and `dry_run_prompts`
to distinguish outcomes. Dry runs, including empty-evidence examples, always
have status `dry_run` and cannot be evaluated as RAG predictions.

Resume is on by default. Matching, checksum-validated successes **and all
completed fallbacks** are reused, preserving paid responses. Failed, corrupt,
or incomplete records are retried. `--retry-fallbacks` explicitly retries
fallbacks and archives their previous responses; `--no-resume` reruns everything.
An abrupt interruption after the provider accepts a request but before its
response is persisted can require another request; exactly-once billing is
not guaranteed. A failed preparation/retrieval is an execution failure, never
an empty successful prediction. Failures do not stop later examples.

Evaluation reuses the unchanged TF-IDF scorer and aggregates for MAP, MRR,
Hit@1/5/10, single-file Recall@1, multiple-file Recall@2/Precision@2 and macro
F1 (k=1 for single-file, k=2 otherwise; not verified equivalent to paper F1).
Metrics include **all completed BM25 fallbacks**. Each report records selected,
evaluated and missing indices; only 50 valid predictions are labelled complete.
Metric groups state their denominators. Retrieval coverage before/after packing
is reported against both eligible candidate gold files and all raw gold files.
Counts, available usage, transport retries and latency are recorded. Failed
attempts may have unreported token usage. Explicit retries' older attempts are
archived; aggregate metrics describe the current saved prediction per issue.

TF-IDF comparison defaults to `bug_localization/outputs/tfidf_python50`, or
accepts an explicit `--tfidf-summary` (summary, run folder or latest-run parent).
The evaluator verifies dataset identity/fingerprint, split, raw-label policy,
issue identity, buggy commit, candidate universe and gold labels. Unavailable
or incompatible baselines are reported with an empty TF-IDF metric row.
Compatible partial baselines use **only common examples for all three CSV
rows**, with exact indices and denominators. Main BM25/RAG metrics still cover
all completed RAG predictions. No remembered scores are inserted.

### Commands (paid commands are provided for manual execution only)

Use this checkout's root `.venv` (Python 3.11). The RAG dependency subset is
[`../requirements-rag.txt`](../requirements-rag.txt); it reuses the pins in
`requirements.txt`, including `openai==1.60.1`, without installing unrelated
baseline backends. It also bounds PyArrow below 21 for `datasets==2.15.0`'s
`PyExtensionType` compatibility. Conda base's Python 3.13 cannot install the
project's pinned NumPy 1.26.4. The sibling `lca-baselines/.venv` is a separate
checkout's environment and is not required for this setup.

From **`bug_localization`**, install or repair this environment with:

```powershell
& ..\.venv\Scripts\python.exe -m pip install -r .\requirements-rag.txt
& ..\.venv\Scripts\python.exe -m pip check
```

Enter the key through a masked prompt, then launch from the same terminal.
The `-m bug_localization...` module command requires the repository root on
Python's import path, so this block temporarily moves there and returns to
`bug_localization` afterwards. Activation is not needed, even with Conda base
active. The following block launches paid inference only when you execute it:

```powershell
$keyInput = Read-Host 'OPENAI_API_KEY' -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $keyInput).Password
Remove-Variable keyInput
Write-Host ('OPENAI_API_KEY available: ' + [bool]$env:OPENAI_API_KEY)

Push-Location ..
try {
    & .\.venv\Scripts\python.exe -m bug_localization.rag.run_benchmark --all --model gpt-4 --resume --prepared-cache-dir bug_localization/rag/outputs/step5_caches/prepared --index-cache-dir bug_localization/rag/outputs/step5_caches/indexes
}
finally {
    Pop-Location
}
```

The explicit cache paths use the writable cache directories verified locally;
they do not change retrieval or ranking settings. Matching completed results
are reused, final saved rankings are evaluated, and the absolute run directory
is printed. Model access requires a live request and is not established by
an import/CLI check. The environment key is read without loading a `.env` file.

Use an environment with the existing `datasets`, `GitPython` and OpenAI
dependencies. `tiktoken` is optional; `--tokenizer bytes` explicitly selects
the existing conservative byte counter. Credentials come only from
`OPENAI_API_KEY`; they are never written to artifacts.

```powershell
# Dry run: retrieve and save one prompt; no model calls.
python -m bug_localization.rag.run_benchmark --dry-run --limit 1

# Paid smoke run: two original examples, then evaluate saved rankings.
python -m bug_localization.rag.run_benchmark --limit 2

# Paid full run: all 50 examples, then evaluate.
python -m bug_localization.rag.run_benchmark --all

# Resume that full configuration, preserving completed fallbacks too.
python -m bug_localization.rag.run_benchmark --all --resume

# Optional: explicitly retry completed fallbacks in the same run.
python -m bug_localization.rag.run_benchmark --all --resume --retry-fallbacks

# Evaluate a saved non-dry run; no inference or repository access.
python -m bug_localization.rag.run_benchmark --evaluate-only bug_localization/outputs/rag_python50/<run_id>
```

Preserve the same configuration, dataset loading mode and subset when resuming.
Both startup and completion print the absolute run directory. `--dataset-file`
loads an existing Arrow file for offline work; its available fingerprint can
differ from `load_dataset` and will then explicitly prevent baseline comparison.
`--repository-cache-dir` reads existing clones without fetching; prepared/index
cache locations can be overridden without changing experiment settings.

Verification: mocked integration tests cover label isolation, complete rankings,
50-example evaluation, partial/common-example comparison, failures, fallback
preservation/retry archives, model/prompt isolation, corruption recovery,
evaluate-only and atomic write interruption. A real-data offline example-0 dry
run retrieved 30 chunks from 23 files in a 195-file candidate universe. No live
model calls or paid full benchmark were launched. The local default cache had
write-permission problems, so verification used fresh `step5_caches` directories.
The read-only clone required a process-scoped Git safe-directory setting.

Exact local offline verification command (PowerShell, repository root):

```powershell
$py = 'C:/Users/Admin/Desktop/Projects/lca-baselines/.venv/Scripts/python.exe'
$data = 'C:/Users/Admin/.cache/huggingface/datasets/JetBrains-Research___lca-bug-localization/py/0.0.0/f2bb26a1c3320ccc1d7b02d2630077ff327a80f6/lca-bug-localization-test.arrow'
$repos = 'C:/Users/Admin/Desktop/Projects/lca-baselines/bug_localization/repos'
# Scope this setting to the current PowerShell process; no global Git changes.
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = "$repos/thealgorithms__python"
& $py -m bug_localization.rag.run_benchmark --dry-run --limit 1 --tokenizer bytes --dataset-file $data --repository-cache-dir $repos --prepared-cache-dir bug_localization/rag/outputs/step5_caches/prepared --index-cache-dir bug_localization/rag/outputs/step5_caches/indexes
```

The inspected offline Arrow fingerprint was `f3b750dd56376f44`; the available
sibling TF-IDF run (`lca-baselines/bug_localization/outputs/tfidf_python50/ca1cf6b4dcc4`)
records `61031751de66717d`. These cannot be compared under the strict fingerprint
policy. For comparison, use the same `load_dataset` loading mode/data revision
as the baseline, and supply its actual `--tfidf-summary` location.

## LLM integration and RAG file ranking (step 4)

[`run_ranking.py`](run_ranking.py) consumes **saved step-3 predictions**. It does
not load a dataset, clone repositories, retrieve more code, rewrite queries, or
run agents. Each example gets at most one logical ranking request, with bounded
transport retries. TF-IDF, BM25 retrieval, and their output files are unchanged.
Only files represented in the packed chunks can be reordered by the LLM; every
other step-1 candidate remains in its original BM25 order in the tail.

### Commands from the repository root

Dry runs need only the standard library. Real requests need the existing
`openai==1.60.1` dependency; `tiktoken~=0.8.0` enables model-specific tokenization.
Evaluation also needs the existing `datasets` dependency. Select your existing
Python environment with these dependencies. Credentials come exclusively from
`OPENAI_API_KEY` in that process's environment; do not put a key in the JSON
configuration or command line. No API call was made during implementation.

These PowerShell commands use the already saved local step-3 example 0:

```powershell
$retrieval = "bug_localization/rag/outputs/bm25_retrieval/23e8068409387329b18c997a2becc322c5c248241ea1796cc687699f826efb8e/summary.json"
$config = "bug_localization/rag/ranking_config.json"

# 1. Build and save prompts, without a client, credentials, or model calls.
python -m bug_localization.rag.run_ranking $retrieval --config $config --dry-run --limit 1 --tokenizer bytes --output-dir bug_localization/rag/outputs/llm_prompts

# 2. Explicitly request one model example (requires OPENAI_API_KEY).
python -m bug_localization.rag.run_ranking $retrieval --config $config --start 0 --limit 1 --output-dir bug_localization/rag/outputs/llm_ranking
```

For another run, replace `$retrieval` with its `summary.json`. A run directory or
parent containing `latest_run.json` is also accepted. `--start` is the minimum
original dataset index; `--limit` limits the number of selected step-3 examples
(default **1**). Failed/unavailable step-3 examples count toward that limit and
are recorded as pipeline failures because no valid BM25 fallback exists.

For the **50-example Python test split**, first ensure all 50 step-3 predictions
and the existing TF-IDF baseline are available. These commands explicitly request
the full retrieval and model experiment; they are instructions, not an automatic
run. If step 3 is already complete, use that run instead of regenerating it.

```powershell
# Preparation only if you do not already have a complete step-3 run:
python -m bug_localization.rag.run_retrieval --split test --limit 50 --max-chunks 30 --output-dir bug_localization/rag/outputs/bm25_python50

# 3. Rank and evaluate all 50 against the existing TF-IDF and BM25 predictions.
python -m bug_localization.rag.run_ranking bug_localization/rag/outputs/bm25_python50 --config bug_localization/rag/ranking_config.json --limit 50 --resume --output-dir bug_localization/rag/outputs/llm_python50 --evaluate --tfidf-summary bug_localization/outputs/tfidf_python50

# Evaluate an already saved model run without making requests:
python -m bug_localization.rag.evaluate_ranking bug_localization/rag/outputs/llm_python50 --tfidf-summary bug_localization/outputs/tfidf_python50
```

`--tfidf-summary` accepts the original TF-IDF summary, its run directory, or its
latest-run parent. If it lives elsewhere, supply its actual path. The unchanged
baseline runner is `python bug_localization/baseline_eval/run_all_tfidf.py`.
Without `--tfidf-summary`, the report explicitly includes BM25/RAG only. A
supplied TF-IDF run must match every evaluated example's index, issue URL,
resolved snapshot SHA, candidate universe, and labels, as well as the dataset
fingerprint and split; mismatches fail comparison rather than silently changing
the cohort. Check `coverage`, `total_examples`, `pipeline_failures`, and
`remaining_examples` before interpreting any run as a full 50-example result.

`--dataset-file path/to/test.arrow` and `--dataset-cache-dir` are evaluation-only
options. The dataset is opened only after all predictions have been saved.
`--dry-run --evaluate` is rejected because prompts are not model predictions.

### Model configuration and reuse decisions

[`ranking_config.json`](ranking_config.json) defaults to **gpt-4**, temperature
**0**, and an **8,192** token context window. Seed is omitted for this legacy
model. GPT-4 requests use `max_tokens` for the output budget.
CLI options override JSON fields: `--backend openai`, `--model`,
`--context-window`, `--max-output-tokens`, `--temperature`, `--top-p`, `--seed`,
`--timeout-seconds`, `--max-retries`, `--response-format`, `--tokenizer`, and
`--safety-margin`. Generation fields can be `null` in JSON to omit unsupported
parameters. Deterministic settings do **not** guarantee identical responses.

[`model.py`](model.py) exposes `ModelClient.generate(messages) -> ModelResponse`,
including raw content, available usage, latency, request/completion IDs, returned
model, fingerprint, refusal, and finish reason. It uses the same OpenAI Python
SDK/environment convention as the existing bug-localization backbone and
[`library_based_code_generation` client](../../library_based_code_generation/src/models/openai_model.py).
Those wrappers are coupled to their own context builders and do not expose all
the required metadata. The bug-localization wrapper's nested decorators also
mix transport and semantic retries, so they are not reused. SDK retries are
disabled; this adapter retries only connections, 408/409/429, and 5xx errors,
up to `max_retries` additional attempts, with delays of 1, 2, 4, ... seconds
capped at 8. Every attempt and available request ID is recorded. Other request
errors return a BM25 fallback immediately. Exception bodies, headers, and
credentials are never persisted.

The default uses text output through Chat Completions, with the prompt requiring
a JSON ranking object. [GPT-4](https://developers.openai.com/api/docs/models/gpt-4)
has an 8,192 token context window and does not support structured outputs.
Text mode omits the API `response_format` parameter. Other models can explicitly
select `--response-format json_schema` or `json_object` when supported;
there is no automatic format switch or LLM repair loop.
Refusals, truncated completions, malformed JSON, and unusable lists fall back to
the original BM25 ranking. Unsupported model settings fail explicitly.

### Prompt and token policy

[`prompts/file_ranking_v1.txt`](prompts/file_ranking_v1.txt) is the dedicated
versioned template. The system message specifies the task and treats all issue
and source contents as untrusted evidence. The user message is a JSON record
with the issue title/body, explicit allowed paths, and complete chunks carrying
their original IDs, paths, inclusive line ranges, and exact source text. No
labels, fix diffs, fixed code, or evaluation feedback enter prediction.

[`ranking_prompt.py`](ranking_prompt.py) uses `tiktoken.encoding_for_model` for
the selected model when available, without the older tokenizer wrapper's
hardcoded GPT-4 encoding and heavyweight transformer import. When unavailable,
it counts **one token per UTF-8 byte**, a conservative upper bound for byte-BPE
text; `--tokenizer bytes` explicitly selects this offline mode. Both count the
response-format JSON and add **16 tokens per message + 32 per request** for
framing. Text token counts are exact only for the selected encoding, while
server framing/schema overhead remains an estimate. Additionally reserve
`max(safety_margin, ceil(context_window * 0.05))` tokens, plus the full configured
output budget, before packing. Counts, tokenizer version/encoding or fallback
reason, reservations, and effective input budget are saved.

If the issue with empty evidence exceeds that input budget, deterministically
remove suffixes from the **body first, then title**, in blocks of one eighth of
the remaining characters rounded up (at least one). Recount after each removal;
retain exact character prefixes and record original/kept character counts. This
policy can sacrifice evidence coverage for very long issues; it does not tune
truncation using labels. If even fixed prompt overhead cannot fit, skip the call.

Consider chunks by descending BM25 score, then original ID. Trial packing
rebuilds and counts the whole prompt **including its new candidate list**.
Include a complete chunk if it fits; otherwise skip it and continue. Candidate
paths are exactly the distinct included-chunk paths in original BM25 file order.
Recount the final prompt before submission. Empty retrieval/no fitting evidence
produces `fallback_no_evidence` without a model call. Every included/excluded ID
and reason is saved. Oversized overhead is marked `fits_budget=false` and is
never submitted. No source chunk is truncated or expanded.

### Validation, complete rankings, and resume

[`file_ranking.py`](file_ranking.py) accepts only a JSON object with exactly
`ranked_files: list`. Duplicate JSON keys, non-JSON numeric constants, Markdown
fences, and extra fields are rejected. Paths follow step 1's validator: only
backslashes become `/`; preserve case and whitespace, reject absolute paths and
dot segments, and require exact membership in allowed paths. No fuzzy matching.
For a usable partial list, keep valid paths in returned order, remove invalid
entries/duplicates, and append omitted allowed paths in BM25 order. Record all
problems and repairs. Empty/all-invalid lists fall back to the original BM25
ranking. There is no response-validation retry.

Validated candidates lead the complete ranking, followed by all other step-1
candidates in BM25 order, including zero-score/empty files exactly once. RAG
`score` fields are descending **ordinal scores**, solely to satisfy the existing
evaluator's ordering contract; they are not relevance probabilities or BM25
scores. `bm25_ranking` separately preserves the full original scores and order.

Outputs live under `rag/outputs/llm_ranking/<run-sha256>/` by default:

- `run_metadata.json`: model/backend and exact request parameters, prompt hash
  and version, tokenizer identity, implementation/package hashes, dataset
  provenance, selected indices, and retrieval input hashes.
- `examples/NNN.json`: checksum, retrieval identity and self-contained label-free
  retrieval input, full snapshot metadata, model settings, exact packed and
  submitted messages, token counts/truncation/chunk decisions, raw response and
  metadata, parsed/validated order, repairs, BM25 ranking, and final ranking.
- `summary.json`: processed/remaining counts, artifact paths, status counts,
  failures, reuse counts, and selected-example coverage. Its `successful` count
  means completed pipeline artifacts, including fallbacks; model successes are
  separate statuses.
- `evaluation/metrics.json` and `per_example_metrics.json`: labels and metrics
  remain separate from model inputs; evaluator-compatible complete rankings are
  also saved under `evaluation/bm25/` and `evaluation/rag/`.

Atomic writes and payload checksums guard partial artifacts. A `running` record
with the prepared prompt is saved before requesting the model; interruption can
leave request state unknown and resuming it may repeat a billable request.
Content-addressed run identity covers retrieval contents, prompt/configuration,
model settings, tokenizer, and relevant implementation/package hashes. Resume
only reuses matching `success`, `repaired`, `fallback_no_evidence`, or `dry_run`
artifacts. Request/validation failures and incomplete/corrupt records are retried
on a subsequent run. Dry and live runs have different identities.
`--no-resume` explicitly regenerates results. CLI request/validation or pipeline
failures return nonzero, while still saving every available BM25 fallback.

### Evaluation and focused verification

[`evaluate_ranking.py`](evaluate_ranking.py) calls the unchanged
`evaluate_tfidf.score_example` and `aggregate` functions: full-ranking MAP, MRR,
Hit@1/5/10, the existing single/multiple-file recall and precision, and macro
F1 at the existing cutoffs. Outside-candidate gold remains in the ranking-metric
denominators. Recall before/after packing uses distinct eligible gold candidate
files; zero eligible gold yields null and is excluded from recall macro means.
It also reports model successes, repaired responses, request failures, unusable
responses, all BM25 fallbacks, available token totals/reporting counts, retries,
and request latency (including retries/backoff). The primary pipeline metrics
include recorded fallbacks. No successful-model-only subset is reported.
Unrecoverable pipeline failures have no prediction and are reported in coverage.

```shell
python -m unittest bug_localization.rag.tests.test_file_ranking -v
python -m unittest discover -s bug_localization/rag/tests -v
```

Step-4 tests use synthetic evidence and mock clients; they cover prompt budget,
candidate-list overhead, skipping large chunks, deterministic issue truncation,
empty evidence, JSON/path repairs, full-universe order, bounded transport retries,
raw metadata and credential isolation, request/parse fallbacks, CLI dry runs,
resume invalidation, label isolation, unchanged three-way metrics, and recall
loss after packing. Validation used one existing cached retrieval example for
dry-run packing only. The default interpreter lacks GitPython, so the two
existing Git snapshot fixtures skip. Real provider calls and the full model
experiment have not been run; prompts were not tuned against test labels.
