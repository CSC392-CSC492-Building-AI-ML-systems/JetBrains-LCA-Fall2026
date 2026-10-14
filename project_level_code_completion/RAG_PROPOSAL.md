# Retrieval-Augmented Context for Project-Level Code Completion

Long Code Arena (LCA) – project-level code completion baseline
Author: Adriana Gaydov · Status: **v3, implemented and evaluated** (v2 was the pre-implementation proposal)

This document describes the design, how it was evaluated, and what was found. The complete lab notebook, with every
setting, prediction and result in the order it happened, is [RAG_EXPERIMENT_LOG.md](RAG_EXPERIMENT_LOG.md); result
files are in [rag_results/](rag_results/). Section 7 lists what changed relative to the v2 proposal.

## 1. Summary

The benchmark's baseline builds one static context string per completion file by ordering or filtering whole
repository files (path distance, imports first, etc.). We replaced it with **per-line retrieval**: for every target
line, BM25 retrieves the most relevant code chunks of the repository snapshot using the last 10 lines before it, and
fills the same 7,900-token budget with them. Each retrieved chunk is followed by the next 20 lines of its file.

**Result (held-out data, evaluated once, `starcoderbase-1b`):** on `medium_context` the design beats the strongest static
composer we evaluated (`path_distance`; `imports_first` was lower; the other seven registered composers were **not**
run, see §8) by **+6.4 exact-match points** (0.357 → 0.421, 95% CI [+4.1, +8.9], 168 datapoints,
6,663 lines), in all six line categories. On the lines that use names defined in other files the gain is +7.1 points,
and a 7B model shows the same gain (+7.3). On `small_context` there is no difference (−0.1 [−0.6, +0.4]), because
those repositories mostly fit in the budget anyway. A plain BM25 version without any tuning already gives +5.6; the
design search added +0.8 on top (CI [+0.2, +1.4]).

## 2. How the baseline works

| Stage | Where | Behaviour |
|---|---|---|
| Data | `eval/preprocessors.py`, `data_classes/` | Each datapoint holds a repo snapshot (`context_dict`), one completion file, and `completion_lines` (target line numbers grouped by category). `get_context()` keeps all `.py` files plus non-`.py` files that are not hidden, not licenses, and under 16,000 characters. |
| Composition | `composers/*.py` | A composer orders or filters whole files into one string, **once per datapoint**; it never sees which line will be predicted. |
| Truncation | `eval/preprocessors.py` | Context is cropped to the last `5*seq_max_len` characters. |
| Composer choice | `eval/eval_pipeline.py` | Perplexity on the completion file at three context sizes; the lowest wins and is used for generation. |
| Generation | `eval/line_generators.py` | Prompt = `context + "\n" + prefix(line_num)`, left-cropped to `seq_max_len - 100` tokens (7,900 by default). Greedy decoding of one line; exact match (EM) and edit similarity (ES). |

Datasets (HF revision `1cdf62335c3c30b3b887498fdf2127e39e01b785`): `small_context` has 144 datapoints and 4,686
target lines; `medium_context` has 224 datapoints and 8,676 lines. The sets are split by the size of the `.py` files of the
repository snapshot (small < 48K characters, medium 48K–192K, large 192K–768K, huge > 768K; about 3 characters per token),
with at most 5 datapoints per repository. The snapshot is the repository **before** the commit that added the completion
file, so the completion file itself is never in it. Target lines have one of six categories, assigned in this order of
priority (benchmark paper, Appendix C): **commited** (uses a function or class declared in files created in the same commit,
which are therefore *not* in the snapshot), **inproject** (declared in the snapshot), **infile** (declared in the completion
file), **common** (very common names such as `main` or `get`), **non_informative** (too short or long, `print`, `import`,
declarations, comments), and **random** (the rest).

### 2.1 What the benchmark paper reports (Bogomolov et al., *Long Code Arena*, 2024)

The paper's baselines use CodeLlama-7B (16K context) and DeepSeek-Coder 1.3B / 6.7B, and compare several context composers.
Its main finding for this task is that the path-distance composer works best. For CodeLlama-7B at 16K, exact match with
only the file prefix ("file-level") vs path distance: `small_context` 0.35 → 0.37 (+5%), **`medium_context` 0.39 → 0.49 (+26%)**,
large 0.35 → 0.47, huge 0.39 → 0.45; on medium the `inproject` lines go from 0.32 to 0.49 (+53%). On the medium set, the
paper's perplexity ranking of composers (Table 10, 12,000 context tokens) puts path distance first (1.607), then half-memory +
path distance (1.636), only-declarations + path distance (1.710), imports-first (1.732), file-length (1.742), naive (1.761),
against 1.849 for file-level.

Our setting differs (StarCoderBase 1B/7B and an 8K window instead of CodeLlama-7B and 16K), so the numbers are not directly
comparable, but the pattern is the same: static context helps a lot on `medium_context` (ours: 0.300 → 0.357, +19%; `inproject`
+39%) and barely on `small_context` (ours: 0.230 → 0.234). The paper does not evaluate retrieval for this task; retrieval
(BM25) appears only as a baseline for its library-based generation task.

## 3. Motivation

1. **Static context.** Every line in a file gets the same context, whatever that line needs.
2. **File-level granularity.** A large file is included or cropped away as a whole even if one function matters.
3. **No relevance signal.** Ordering is by path distance, import graph or length; nothing measures similarity to the
   code being written.
4. **Selection/metric mismatch.** Composers are chosen by whole-file perplexity, but the metric is line-level EM/ES.

**A measurement that shaped the evaluation.** Only a minority of lines depend on code defined in other files. Counting,
for each target line, identifiers that the repository defines in another file and that the file's own prefix does not
define (called *targeted* or `T` lines): 6.0% of `small_context` lines but 38.3% of `medium_context` lines. Overall EM
on small repositories therefore cannot move much, however good the retrieval, and medium repositories are where it can
be tested. (Log §8.13.)

## 4. Design as built

### 4.1 Pipeline

```
repo snapshot ──► chunker ──► BM25 index (once per datapoint)
                                   │
prefix before target line ─► query ─┴─► ranked chunks ─► + 20 following lines ─► budgeted prompt ─► model ─► line
```

### 4.2 Token budget (identical for every arm)

- Total input cap **B = `seq_max_len − 100` = 7,900 tokens**, counted with the model's own tokenizer; the same cap the
  generator already applies, so all arms see the same maximum length.
- The completion-file prefix is kept in full (cropped on the left only if it exceeds B). Retrieved context fills what
  remains. After assembly the exact total is verified with the tokenizer and the lowest-ranked chunk is dropped until it
  fits, because per-chunk token costs are not exactly additive.
- Static arms keep their behaviour (composed context, newline, prefix, left-cropped to B).

### 4.3 Components (`rag/`)

- **Chunker (`chunker.py`).** tree-sitter: each function or method is one chunk (decorators included; methods get their
  `class X:` line prepended); class headers and module-level code are separate chunks. Chunks over 512 tokens, files that
  fail to parse, and non-Python files are cut into 40-line windows with stride 20. Exact duplicates are dropped. Chunks
  record the symbol they define.
- **Index.** Built once per datapoint, over every file of the snapshot **except the completion file** (its own prefix is
  already in the prompt; the snapshot never contains an older copy of it, checked on 104 development datapoints).
- **Query.** The last 10 non-empty lines before the target line; the target line itself is never included. An empty
  prefix falls back to the `path_distance` file order.
- **BM25 (`bm25.py`).** Hand-rolled Okapi BM25 with an inverted index, `k1 = 1.2`, `b = 0.75`; identifiers are lower-cased,
  kept whole and also split on `_` and camelCase; Python keywords dropped.
- **Selection and assembly (`builder.py`).** Chunks are taken in descending score until 30 chunks or the budget is
  reached; a chunk that does not fit is skipped, never truncated. **Each chunk is followed by the next 20 lines of its
  file** (`continuation_lines=20`). Chunks are emitted in ascending relevance so the best one is nearest the prefix, in
  the baseline's `path + METASEP + content` layout.
- **Optional components, all off in the final design:** path-proximity and import priors, definition-name lookup, a context
  cap, best-first ordering, fine-grained windows, and dense (`dense.py`, cached embeddings) and hybrid retrieval with
  reciprocal-rank fusion.

### 4.4 Integration

- `eval/line_generators.py` gained an optional per-line context hook (`context_provider`) and options to run only the
  zero-context or only the context pass, select datapoints, and restrict to given lines. Defaults reproduce the original
  behaviour (pinned by tests).
- `eval/compare_arms.py` runs **one arm per invocation** (`zero`, `rag`, or any registered composer) with the same seed
  and token budget into its own output directory, refusing to append to existing results, and records the git commit and
  a dirty-tree flag. Static composers are fed through the same per-line path (verified identical to the original
  preprocessed path: 0 differing predictions out of 1,573).
- Analysis (`eval/analyze_arms.py`, `eval/make_targets.py`): benchmark-style means (per-datapoint mean, then mean over
  datapoints) and a paired bootstrap over datapoints (10,000 resamples, seed 0, 95% CI), overall and per category.
- Final configuration: `arm=rag rag.prefix_cap=7900 rag.continuation_lines=20`, all else default.

## 5. Evaluation as run

- **Model:** `starcoderbase-1b` for all arms and development; `starcoderbase-7b` for a confirmation on the targeted lines
  of `medium_context`. RTX 5090; greedy decoding, seed 42.
- **Pre-registration and splits.** Hypotheses, predictions and decision rules were written in the log before the
  corresponding runs. Data was split into **development** (48 of 144 `small_context` and 56 of 224 `medium_context`
  datapoints) and **held-out** (96 and 168 datapoints, never run or inspected until the end). Design choices were made on
  development data only, with a cap on the number of runs; the final design was frozen, committed (`566331c`, clean
  tree), and evaluated once on the held-out data with no change afterwards.
- **Arms (held-out):** zero context; `path_distance` and `imports_first` (the benchmark's composers); **v1** (plain BM25,
  untuned); **F** (final design).
- **Metrics:** EM and ES, overall and per category; uncertainty from the paired bootstrap over datapoints, because lines in
  one file are correlated.
- **Primary comparisons:** P1, `medium_context` all lines, F − `path_distance`; P2, targeted lines of small + medium;
  P3, `small_context` all lines.

## 6. Results

### 6.1 Held-out (EM; Δ vs `path_distance`, 95% CI; `*` excludes 0)

| | zero | imports_first | path_distance | v1 (BM25) | **F (final)** |
|---|---:|---:|---:|---:|---:|
| **P1 `medium_context`, all lines (168 dp)** | 0.300 | 0.312 | 0.357 | 0.414 (+0.056 \*) | **0.421 (+0.064 [+0.041, +0.089] \*)** |
| **P2 targeted lines, small + medium (162 dp)** | 0.263 | 0.269 | 0.327 | 0.395 (+0.067 \*) | **0.398 (+0.071 [+0.033, +0.109] \*)** |
| **P3 `small_context`, all lines (96 dp)** | 0.230 | 0.231 | 0.234 | 0.234 (+0.000) | 0.233 (−0.001 [−0.006, +0.004]) |
| 7B, targeted lines of `medium_context` (145 dp) | 0.285 | | 0.360 | | **0.433 (+0.073 [+0.038, +0.109] \*)** |

- F improves **every category** on `medium_context` (common +10.9, random +7.0, infile +6.5, inproject +6.5, commited +4.1,
  non_informative +4.1 points; all intervals exclude 0). ES on medium: 70.4 → 72.9 (+2.5 \*).
- F − v1: +0.8 [+0.2, +1.4] \* on medium all lines; +0.3 (not significant) on targeted lines.
- Static context helps on medium (`path_distance` − zero = +5.7 \*) and not on small (+0.4); `imports_first` is worse than
  `path_distance` on medium (−4.6 \*).

### 6.2 What the design search tried (development data; Δ EM vs v1 on 866 targeted lines, 57 datapoints)

| change | Δ vs v1 |
|---|---|
| **continuation lines: 20 (final)** / 40 / 10 | **+1.9 \*** / +2.3 (+0.4 vs 20, not adopted) / +1.0 |
| context cap 1,000 / 2,000 / 4,000 tokens | −3.1 \* / −1.0 / −0.1 |
| path-proximity prior (0.25 / 0.5 / 1.0) | −0.3 / −0.5 / −0.4 |
| import boost (0.5 / 1.0) | −0.7 / −4.4 \* |
| definition-name lookup (with / without imports) | −0.6 / −0.3 |
| query length 3 / 20 / 30 lines | −1.8 \* / −0.1 / 0.0 |
| dense retrieval (gte-modernbert) / hybrid / hybrid with unixcoder | 0.0 / −0.9 / −0.4 |
| fine windows + continuation | +1.5 (not adopted: not simpler, CI includes 0) |
| best-first ordering | +1.0 alone, but −1.1 on top of continuation: dropped |

### 6.3 Hypotheses from v2 and what happened

| v2 hypothesis | outcome |
|---|---|
| Gains are larger when the repository is much longer than B and shrink for small repositories | **Supported** (medium +6.4, small −0.1) |
| RAG helps most on lines that depend on code defined elsewhere (`inproject`) | **Partly**: `inproject` gains +6.5, but every category gains on medium, including the ones expected to change little |
| `infile`, `random`, `non_informative` change little | **Supported on small, not on medium** (+6.5, +7.0, +4.1) |
| Dense or hybrid retrieval may beat BM25 | **Not supported** for the models tried |
| Negative results are plausible | Not observed on medium; null on small as explained above |
| Success: beats `path_distance` overall on 1B and same direction on 7B | **Met on `medium_context`**; 7B was run on targeted lines only |

### 6.4 Verification of the implementation (after the held-out run)

| check | result |
|---|---|
| BM25 against a brute-force reference formula | equal within 1e-9 on 200 random corpora |
| Prompts rebuilt for 400 random held-out lines | never contain the completion file; **4 exceed the 7,900-token budget** when the prefix alone fills it, so no chunk fits (3.3% of medium and 4.7% of small held-out lines are in this regime; zero context, `path_distance` and F score alike there; excluding them changes the result by 0.2 points) |
| Robustness on `medium_context` held-out | F better on 76 datapoints, worse on 33, tied on 59; repository-clustered 95% CI [+3.6, +9.7]; leave-one-repository-out +5.4 … +6.7 |
| Controls (development data, targeted lines) | F 0.378; random chunks 0.233; proximity-only chunks 0.244; stale query 0.297: **relevance of the retrieved chunks, not extra text, drives the gain** |
| Where the gain comes from | in 21% of lines F's context contains the target line verbatim; those lines carry about 75% of F's advantage (+2.3 points where neither context does) |

By the pre-registered rule ("strong only if every check passes") the implementation does **not** qualify, because of the
budget edge case, which has no effect on any result; otherwise it passes. Details: log §8.30–§8.32.

## 7. Changes relative to the v2 proposal

| v2 said | what was done | why |
|---|---|---|
| Prefix capped at 4,000 tokens | Prefix kept in full | The cap was an unfair handicap against the static arms (development data, log §6) |
| `context_mode: static \| rag` flag in `_get_context` | `context_provider` hook plus the `compare_arms` driver | Cleaner to run arms separately with identical settings |
| Evaluate on all of `small_context` | Development / held-out split; `medium_context` added | A single `test` split invites tuning on reported data; small repositories cannot show an effect (§3) |
| Tune ablations one by one | Pre-registered batches, promotion rule and run caps | Limit optimism from trying many designs |
| Dense retrieval from `codet5p-110m-embedding` or `jina-embeddings-v2-base-code` | `gte-modernbert-base` and `unixcoder-base` | The two named models fail to load under the pinned `transformers` 5.17 |
| `skip_zero` option | `run_zero` / `run_context` options | Same purpose |
| Hydra config group for retrieval | `rag:` block in `compare_arms.yaml` | Simpler |

## 8. Limitations

- **Only two of the benchmark's static composers were evaluated** (`path_distance`, `imports_first`), plus zero context.
  The registry also contains `naive`, `alphabetical`, `file_length`, `half_memory`, `half_memory_path_distance`,
  `function_class_mask_half` and `function_class_mask_one`, and the benchmark's own pipeline chooses among all of them by
  perplexity. `path_distance` was used as the reference because it is the natural strong baseline and was the better of the
  two we ran, not because it was shown to be the best of all. Whether any other composer is stronger than `path_distance`
  on `medium_context` is untested here; the claim above is therefore about `path_distance` specifically.
- Only `starcoderbase-1b` (and `starcoderbase-7b` on targeted lines); the benchmark's other baseline models were not run.
- **Much of the gain depends on near-copies.** About 75% of F's advantage over `path_distance` comes from lines whose text
  appears verbatim elsewhere in the repository (§6.4); where it does not, the advantage is about +2.3 points.
- **No room for retrieval in 3–5% of lines:** when the prefix itself fills the 7,900-token budget no chunk can be added
  (§6.4); there F equals the static baselines.
- One model family (StarCoder base; 1B for all arms, 7B on targeted lines only), Python only, greedy decoding of a single
  line; `large_context` and `huge_context` were not evaluated.
- The design search compared about 22 variants on 104 development datapoints, so the incremental value of the final design
  over plain BM25 is the least certain finding (it shrank from +1.0 on development to +0.8 on held-out; the untuned design
  keeps about two thirds of its development advantage).
- The targeted-line definition is lexical; the 82 targeted lines of `small_context` held-out (17 datapoints) are too few
  to support a claim.
- The benchmark's generator keeps stopping-criterion state across the lines of a datapoint, so a prediction can depend
  slightly on which lines were generated before it (about 1% of lines). It affects all arms alike when the line set is
  the same; it was left unchanged. (Log §8.24.)
- The chunker falls back to plain line windows when the tree-sitter grammar is not present in the working directory. All
  reported runs used the grammar. The fallback was silent when the experiments were run; after the held-out run it was
  changed to emit a clear warning once (log §8.28), with no effect on any result.

## 9. Reproduction

```bash
cd project_level_code_completion
poetry install                                           # Python 3.11, pinned torch/transformers
git clone --branch v0.21.0 https://github.com/tree-sitter/tree-sitter-python   # ABI-14 grammar; run from this folder
poetry run python -m unittest discover -s tests -t .     # 154 tests
poetry run python -m eval.compare_arms arm=path_distance dataset=medium split=heldout dev_size=28 dev_extra=28 arm_name=h_path_distance
poetry run python -m eval.compare_arms arm=rag rag=F dataset=medium split=heldout dev_size=28 dev_extra=28 arm_name=h_F
#   rag=F is the final design (same as rag.prefix_cap=7900 rag.continuation_lines=20); rag=v1 is plain BM25
poetry run python -m eval.analyze_arms --base data/code_generation/artifacts/python/starcoderbase-1b/medium_context/arms --ref h_path_distance --arms h_F
```
`starcoderbase-1b` is gated on HuggingFace (accept the licence and log in).

## 10. Possible next steps

`large_context` and `huge_context`; the 7B model on all lines; other model families; iterative retrieve-then-generate
using the model's draft line as a second query; a learned retriever.
