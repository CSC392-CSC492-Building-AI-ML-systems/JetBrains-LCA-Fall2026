# 🏟️ Long Code Arena Baselines
## Bug localization

This directory contains the code for the Bug localization benchmark. The challenge is: 
given an issue with the bug description and the repository code in the state where the issue is reproducible, identify the files within the project that need to be modified to address the reported bug.

We provide scripts for [data collection and processing](./src/data), [data exploratory analysis](./src/notebooks), as well as several [baselines implementations](./src/baselines) for solving the task with [the calculation of evaluation metrics](./src/notebooks).

## 💾 Install dependencies
We provide dependencies for the pip dependency manager, so please run the following command to install all the required packages:
```shell
pip install -r requirements.txt
```

## 🤗 Dataset

All the data is stored in [HuggingFace 🤗](JetBrains-Research/lca-bug-localization). It contains:

* A **dataset** with the bug localization data (issue description, SHA of the repo in initial state, and SHA of the repo after fixing the issue).
You can access the data using the [datasets](https://huggingface.co/docs/datasets/en/index) library:
    ```python3
    from datasets import load_dataset
    
    # Select a configuration from ["py", "java", "kt", "mixed"]
    configuration = "py"
    # Select a split from ["dev", "train", "test"]
    split = "dev"
    # Load data
    dataset = load_dataset("JetBrains-Research/lca-bug-localization", configuration, split=split)
    ```
    ...where the labels are:\
    `dev` — all the collected data;\
    `test` — manually selected data ([labeling artifacts](https://docs.google.com/spreadsheets/d/1cEyFHjse-iUYQlUO7GO5KpqkvJ3wu6vheou4W61TMOg/edit?usp=sharing));\
    `train` — all the collected data that is not in `test`;

    ...and configurations are:\
    `py` — only `.py` files in diff;\
    `java` — only `.java` files in diff;\
    `kt` — only `.kt` files in diff;\
    `mixed` — at least one `.py`, `.java`, or `.kt` file and maybe files with another extensions in diff.


* **Archived repos** (from which we can extract repository content at different stages and get diffs that contains bugs fixes).\

## ⚙️ Baselines

* Embedding-based:
  * [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer);
  * [GTE](https://huggingface.co/thenlper/gte-large);
  * [CodeT5](https://huggingface.co/Salesforce/codet5p-110m-embedding);
  * [BM25]().
  
* Chat-based:
  * [GPT3.5](https://platform.openai.com/docs/models/gpt-3-5-turbo);
  * [GPT4](https://platform.openai.com/docs/models/gpt-4).

## Repository preparation for RAG (step 1)

From the repository root, with the existing `datasets` and `GitPython`
dependencies installed, prepare one Python test example:

```shell
python bug_localization/baseline_eval/prepare_repositories.py --limit 1
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
  [`prepare_data_for_analysis.py`](src/data/preprocessing/prepare_data_for_analysis.py).
  [`HFDataSource`](src/baselines/data_sources/hf_data_source.py) and the local
  TF-IDF runner load `base_sha` as the buggy snapshot. Preparation follows that
  convention, without deriving a parent commit or reading `head_sha` or diffs.
- The unchanged [`run_all_tfidf.py`](baseline_eval/run_all_tfidf.py)
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

Adriana's [`rag/chunker.py`](../project_level_code_completion/rag/chunker.py)
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

Caches live under `bug_localization/outputs/prepared_snapshots/<sha256>/`:

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
python -m unittest discover -s bug_localization/tests -p test_prepare_repositories.py -v
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
