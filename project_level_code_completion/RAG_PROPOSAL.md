# Proposal: Retrieval-Augmented Context for Project-Level Code Completion

Long Code Arena (LCA) – project-level code completion baseline
Author: Adriana Gaydov · Status: draft v2 for review

## 1. Summary

The current baseline builds one static context string per completion file by reordering or filtering
repository files (path distance, imports first, etc.). We propose replacing this with **per-line
retrieval**: for every target line, retrieve the code chunks from the repository snapshot that are most
relevant to the code written so far, and fill the model's token budget with them. We implement a BM25
retriever first, compare it with the existing baselines on one model family under an identical token
budget, and use the results to decide whether dense or hybrid retrieval is worth adding.

## 2. How the baseline works today

| Stage | Where | Behaviour |
|---|---|---|
| Data | `eval/preprocessors.py`, `data_classes/` | Each datapoint holds a repo snapshot (`context_dict`), one completion file, and `completion_lines` (target line numbers grouped by category). `DatapointCommitDataset.get_context()` keeps all `.py` files plus non-`.py` files that are not hidden, not licenses, and under 16,000 characters. |
| Composition | `composers/*.py` | A composer orders/filters whole files into one string, **once per datapoint**. It never sees which line will be predicted. |
| Truncation | `eval/preprocessors.py`, `model_hub/model_inference.py` | Context is cropped to the last `5*seq_max_len` characters, then (perplexity stage only) to `context_max` tokens. |
| Composer choice | `eval/eval_pipeline.py` | Perplexity on the completion file at three context sizes; the lowest wins and is used for generation. |
| Generation | `eval/line_generators.py` | Prompt = `context + "\n" + prefix(line_num)`, left-cropped to `seq_max_len - 100` tokens (7,900 with the default `seq_max_len=8000`). Greedy decoding of one line; exact match (EM) and edit similarity (ES), each also run with zero context. Note that `context_max` is **not** used here. |

Dataset facts (`small_context`, from the HF datasets-server): 144 datapoints, about 37.6 MB. Line counts
by category, summed over all datapoints:

| Category | Lines | Datapoints with ≥1 line |
|---|---:|---:|
| infile | 1,430 | 136 |
| commited | 1,426 | 99 |
| random | 703 | 144 |
| non_informative | 532 | 144 |
| common | 500 | 88 |
| inproject | 95 | 11 |
| **Total** | **4,686** | |

## 3. Limitations motivating this work

1. **Static context.** Every line in a file gets the same context, regardless of what that line needs.
2. **File-level granularity.** A large file is included (or cropped away) as a whole even when only one
   function in it is relevant.
3. **No relevance signal.** Ordering is by path distance, import graph, or length. No composer scores
   similarity to the code being written.
4. **Selection/metric mismatch.** Composers are chosen by whole-file perplexity, but the reported
   metric is line-level EM/ES.

## 4. Proposed design

### 4.1 Pipeline

```
repo snapshot ──► chunker ──► index (once per datapoint)
                                   │
prefix before target line ─► query ─┴─► ranked chunks ─► budgeted prompt ─► model ─► line
```

### 4.2 Token budget (applies to every arm)

- **Total input cap B** = `seq_max_len - 100` tokens (7,900 by default), measured with the model's
  own tokenizer. This is the existing generation limit, so all arms see the same maximum input length.
- **Prefix.** The completion-file prefix before the target line, left-cropped to at most **P = 4,000
  tokens**. Its actual length is `p ≤ P`.
- **Retrieved context.** Fills the remaining `B - p` tokens, including separators.
- **Static arms** keep their existing behaviour (full composed context, then prefix, left-cropped to
  B). They already fill B, so the arms differ in content rather than in length.
- The static arms' prefix can grow up to B, while RAG caps it at P; the cap P is itself ablated (§5).

### 4.3 Components and defaults

**Chunker.** Uses the existing tree-sitter parser (`tree_sitter_parser/`).
- Each function or method is one chunk (decorators included; methods get their `class X:` line
  prepended). Remaining module-level code (imports, constants) is one "module header" chunk per file.
- Any chunk over 512 tokens, any file that fails to parse, and any non-`.py` file is split into line
  windows of 40 lines with stride 20.
- Non-`.py` files are indexed only if `get_context()` already keeps them (§2). An ablation drops them.
- Exact duplicate chunk texts are dropped (the first path wins). Chunk token counts are computed once
  at index time.

**Index.** Built once per datapoint. The completion file is excluded, because its own prefix is
already in the prompt. Retrieval is in-memory; no vector database is needed at this scale.

**Query.** The last **N = 10** non-empty full lines before the target line. The target line itself is
never part of the query (`get_prefix(line_num)` returns lines `[:line_num]` only). If the prefix is
empty, the query is empty and the prompt falls back to the `path_distance` file ordering.

**Retriever 1 – BM25 (stage 1).** Hand-rolled Okapi BM25 (about 30 lines), so there is no new
dependency. Parameters `k1 = 1.2`, `b = 0.75`. Tokenization: regex identifiers
(`[A-Za-z_][A-Za-z0-9_]*`), lower-cased, with each identifier kept whole and also split on `_` and
camelCase boundaries; Python keywords are dropped; IDF handles the remaining common tokens.

**Retriever 2 – dense (stage 2, only if BM25 shows retrieval helps).** Candidate embedding models are
`Salesforce/codet5p-110m-embedding` or `jinaai/jina-embeddings-v2-base-code` (final choice after a
quick check that it loads under the pinned `transformers`). Cosine similarity over an in-memory matrix;
chunk embeddings cached per repo snapshot.

**Retriever 3 – hybrid (stage 3, optional).** Reciprocal-rank fusion of BM25 and dense (constant 60).

**Selection.** Walk chunks in descending score, skipping zero-score chunks. A chunk that does not fit
the remaining budget is skipped (chunks are never truncated) and the next smaller one is tried. Stop
when the budget is filled or after 30 chunks.

**Prompt assembly.** Selected chunks are emitted in **ascending** relevance, so the best chunk sits
nearest the prefix. Each chunk uses the existing `path + METASEP + content` layout joined with
`LANGSEP`, matching the baseline format. The completion-file path and prefix follow.

### 4.4 Integration with the existing pipeline

- New package `rag/` (chunker, retrievers, prompt builder). Existing composers are untouched.
- The composer interface cannot see `line_num`, so RAG hooks into `LineGeneratorBase._get_context`
  behind a flag (`context_mode: static | rag`, default `static`).
- RAG is a **generation-only** arm: it skips the composer and perplexity sweep. Its results are not
  eligible for perplexity-based selection and do not enter `completion_results.json`.
- The existing pipeline picks one composer by perplexity, so it cannot guarantee that
  `path_distance` or `imports_first` is evaluated. A small comparison driver
  (`eval/compare_arms.py`) therefore runs each arm explicitly: it preprocesses the chosen composer
  (or skips preprocessing for RAG), calls `evaluate_generation` with the same seed and budget, and
  writes to a **separate directory per arm** (`<out>/<model>/<dataset>/<arm>/`). Separate directories
  matter because `generation_results.jsonl` is opened in append mode.
- `evaluate_generation` always re-runs the zero-context pass. The driver will add a `skip_zero` option
  so that pass runs once per model instead of once per arm.
- Logged per line, for analysis only: retrieved paths, scores, and token counts.
- Retrieval parameters go in a Hydra config group.

### 4.5 Parameters to ablate

| Parameter | Default | Values |
|---|---|---|
| Chunk granularity | function + windows | windows only; function only |
| Query lines N | 10 | 3, 10, 30 |
| Chunk order | best-last | best-first |
| Prefix cap P | 4,000 | 2,000, 4,000, 6,000 |
| Non-`.py` files | indexed | excluded |
| Retriever | BM25 | dense, hybrid (stages 2–3) |

## 5. Evaluation plan

- **Models.** `starcoderbase-1b` for iteration and all ablations; `starcoderbase-7b` on the RTX 5090
  (32 GB) for the headline comparison only. Both are in `MODEL_REGISTRY`.
- **Data.** `small_context` first (144 datapoints, 4,686 lines). `medium_context` (224 datapoints) if
  time allows, mainly for more lines in the rare categories.
- **Fixed settings.** Seed 42, `seq_max_len` 8,000, greedy decoding, the same token budget B for every
  arm.
- **Arms.**
  - A. Zero context (existing)
  - B. `path_distance` composer (existing)
  - C. `imports_first` composer (existing)
  - D. RAG with BM25 and the defaults in §4.3 (new)
  - E. Ablations of D from §4.5 (1B only)
- **Metrics.** EM and ES overall and per line category, plus per-datapoint win/loss against zero
  context (the pipeline's existing `EM_difference`).
- **Statistics.** Differences between arms are reported as paired differences with a **bootstrap 95%
  CI resampled over datapoints** (144 clusters, not 4,686 independent lines, because lines within a
  file are correlated).
- **Power caveat.** `inproject` has only 95 lines across 11 datapoints in `small_context`, so
  `inproject` results will be wide and are reported as descriptive, not as the headline claim. The
  `medium_context` per-category counts will be tallied before we decide whether to run it.
- **Compute.** About 4,686 generation calls per arm. Wall-clock time on the 1B model will be measured
  in a pilot (20 datapoints) before the full runs. If 7B proves too slow, the 7B comparison is limited
  to arms A, B, D, with a sampled subset of datapoints if necessary, and that is stated in the results.

### Success criteria

Defined before running, to be calibrated against the pilot's noise level:
- **Success:** D beats B on overall EM, with the bootstrap 95% CI of the paired difference excluding 0,
  on the 1B model and with the same direction on 7B.
- **Partial:** D beats B on one or more specific categories (reported with their CIs) but not overall.
- **Negative:** D is not distinguishable from, or is worse than, B. This is reported as a negative
  result together with the diagnostics below.

Diagnostics for any outcome: retrieval latency, retrieved-token share, and, for each line, whether a
retrieved chunk contained the identifier that the ground-truth line uses (analysis only, not a filter).

## 6. Expected outcomes (hypotheses, not results)

- RAG helps most on lines that depend on code defined elsewhere in the repository. By the dataset's
  naming these should be `inproject`, and plausibly `commited` and `common`; the exact definitions of
  those categories still need to be confirmed against the dataset card before we rely on this.
- `infile`, `random` and `non_informative` lines change little, since the completion file's own prefix
  carries most of the signal.
- Gains are larger when the repository is much longer than B, where file-level composers waste tokens,
  and shrink for small repositories that already fit.
- Dense or hybrid retrieval may beat BM25 on paraphrase-like matches, but BM25 is expected to be
  competitive on code because identifiers repeat literally.
- Negative results are plausible: chunks that look similar but are not the needed definition can
  distract the model and lower EM on some lines. Per-datapoint win/loss is there to expose this.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Leakage: retrieving the target line from the completion file | Completion file excluded from the index; unit test asserts no chunk comes from it |
| Per-line retrieval slows generation | Index once per datapoint; BM25 over thousands of chunks is milliseconds; latency logged |
| Prefix starved by large retrieved context | Fixed budget split (§4.2); P is ablated |
| Unfair comparison with perplexity-selected composers | All arms run through the explicit driver at the same budget B on line-level metrics |
| Small, clustered data gives noisy comparisons | Bootstrap CI over datapoints; `medium_context` as a follow-up; `inproject` reported descriptively |
| Local environment mismatch | The project pins Python 3.11–3.12, `torch==2.14.0`, `transformers==5.17.0`; the machine's default Python is 3.14. Create a 3.11/3.12 environment and confirm the pins run on the RTX 5090 before milestone 3 |
| 7B runs too slow | Run ablations on 1B only; pilot timing first; reduce 7B to arms A, B, D |

## 8. Implementation plan and milestones

Each milestone ends with a review checkpoint and a small, separate commit.

1. **Proposal** (this document).
2. **Environment check and chunker.** Python 3.11/3.12 environment; confirm the pinned stack runs on
   the 5090; chunker with unit tests on a toy repo.
3. **BM25 retriever and prompt builder**, with unit tests (leakage test, budget test).
4. **Per-line integration and driver** behind `context_mode=rag`; pilot on `starcoderbase-1b` with
   about 20 datapoints; measure timing and noise.
5. **Experiments:** arms A–E on `small_context`, then the 7B comparison; results tables and plots.
6. **Write-up:** fill in §9; optionally add dense/hybrid retrieval.

## 9. Results

_To be filled after the experiments in milestone 5._
