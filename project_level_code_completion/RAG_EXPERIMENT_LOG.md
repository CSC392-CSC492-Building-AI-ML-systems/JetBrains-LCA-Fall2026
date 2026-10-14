# RAG experiment log (lab notebook)

Companion to [RAG_PROPOSAL.md](RAG_PROPOSAL.md). Conventions:

- Entries are appended in the order things happened; an earlier entry is never rewritten. Corrections are added as
  dated amendments.
- Hypotheses and decision rules are written **before** the corresponding results exist, and say so.
- Numbers come only from files under `~/lca_artifacts/` (WSL) produced by `eval.compare_arms`; the path of each
  result is recorded.
- Every code change gets a row in the change register (§4) with its reason and tests.

## 0. Provenance

| Item | Value |
|---|---|
| Date started | 2026-10-12 (entries below dated as they occur) |
| Code base | `JetBrains-LCA-Fall2026`, branch `main`; HEAD at start of this log: `c31287e` (milestone-4 code was not yet committed, see §4) |
| Dataset | `JetBrains-Research/lca-project-level-code-completion`, config `small_context`, split `test` (144 datapoints, 4,686 target lines), HF revision `1cdf62335c3c30b3b887498fdf2127e39e01b785` |
| Model | `bigcode/starcoderbase-1b` (the `starcoderbase-7b` checkpoint is also cached, not yet used) |
| Software | Python 3.11.16 (WSL2 Ubuntu), torch 2.14.0+cu130, transformers 5.17.0, datasets 5.0.1, tree-sitter 0.20.4 with `tree-sitter-python` tag `v0.21.0` (grammar ABI 14) |
| Hardware | NVIDIA GeForce RTX 5090 (32 GB), driver 616.56 |
| Decoding | greedy, `max_new_tokens=100`, stop at newline, seed 42 (existing benchmark settings) |
| Token budget | `seq_max_len` 8000 − 100 = **7,900** tokens of input for every arm |
| Run location | WSL, from `~/ts_test` (contains the grammar clone so tree-sitter chunking is active), `PYTHONPATH` = project dir |

## 1. Research question and hypotheses (from the proposal, §6)

**RQ.** Does per-line BM25 retrieval of repository chunks give a better prompt than the benchmark's static,
file-level composers at the same token budget?

- **H-main.** RAG improves exact match (EM) over `path_distance`, mainly on lines that depend on code defined
  elsewhere in the repository.
- **H-null.** There is no EM difference, or RAG is worse (the proposal lists distraction by similar-looking chunks
  as a plausible mechanism).

Primary metric: EM over all lines, computed as the benchmark does it (mean over datapoints of that datapoint's
line-level EM). Secondary: edit similarity (ES) and per-category EM.

## 2. E1: pilot (retrospective entry)

*Written after the fact: the run was done first and documented here afterwards. Only the proposal's hypotheses
(§1) pre-date it; everything under "post-hoc" does not.*

**Setup.** `eval.compare_arms`, 20 evenly spaced datapoints (`limit_datapoints=20`, indices `i*144//20`),
709 target lines, arms `zero`, `path_distance`, `rag` (BM25, defaults: query = last 10 non-empty lines,
prefix cap 4,000 tokens, ≤30 chunks, non-Python files indexed). Commands (from `~/ts_test`):

```
python -m eval.compare_arms arm=<arm> limit_datapoints=20 arm_name=pilot_<arm> artifacts_dir=~/lca_artifacts overwrite=true
```

Wall time about 2m40s per arm. Outputs: `~/lca_artifacts/python/starcoderbase-1b/small_context/arms/pilot_<arm>/`.

**Results (EM, benchmark-style means; one run per arm, no confidence intervals).**

| EM | zero | path_distance | rag |
|---|---:|---:|---:|
| all | 0.239 | 0.266 | 0.252 |
| infile (208 lines) | 0.238 | 0.279 | 0.207 |
| commited (234) | 0.193 | 0.176 | 0.164 |
| common (44) | 0.078 | 0.149 | 0.135 |
| inproject (43) | 0.142 | 0.186 | 0.171 |
| non_informative (81) | 0.170 | 0.177 | 0.188 |
| random (99) | 0.275 | 0.315 | 0.315 |
| ES all | 61.2 | 64.3 | 63.0 |
| line-level EM (709 lines) | 0.224 | 0.226 | 0.210 |

Line-level discordant pairs: RAG right & path_distance wrong = 7, path_distance right & RAG wrong = 18;
RAG right & zero wrong = 15, zero right & RAG wrong = 25.

**Descriptive observations.** RAG is below `path_distance` overall and on `infile`, and at line level is not
above zero context. 58 of 709 lines (8%) retrieved no chunk at all (no query term matched), so they were run with
no repository context. 46% of retrieved chunks came from non-Python files (`.md`, `.yaml`, `.js`, …).

**Post-hoc hypotheses (generated after seeing the pilot; untested, and not evidence until tested):**

- **P1 (design confound).** RAG caps the prefix at 4,000 tokens, while the zero and static arms let the prefix use
  up to the whole 7,900-token budget. Long completion files (some exceed 1,000 lines) therefore give RAG less of the
  file itself, which would hurt `infile` lines. This is a flaw in my design of the comparison, not a property of
  retrieval.
- **P2.** Non-Python chunks are noise for a Python benchmark.
- **P3.** Lines with an empty retrieval behave like zero context, so RAG loses the baseline's "always some context".
- **P4.** The 10-line query and the best-last ordering may be poor choices.

**Threats to validity of E1.** Only 20 datapoints (≈1.6-point standard error at line level, and lines within a
datapoint are correlated); a single seed (greedy decoding is deterministic, but a different datapoint sample would
differ); RAG configuration chosen a priori with no tuning, whereas `path_distance` is a long-standing baseline.

**Amendment (2026-10-13).** The discordant-pair counts above treat lines as independent. With datapoint-clustered
bootstrap CIs (§6.3) the overall RAG − `path_distance` difference is −1.4 points [−3.8, +0.6], i.e. not distinguishable
from 0. Do not cite the 7-vs-18 counts as evidence that RAG is worse.

## 3. Evaluation design change: separate development and held-out data (decided 2026-10-13)

**Problem.** The dataset has a single `test` split. If I tune RAG on the pilot datapoints and then report numbers
from the same datapoints, the reported gain is optimistically biased.

**Decision.**

- **dev** = the 20 pilot datapoints (indices `i*144//20`, 709 lines). Used for diagnosis and for choosing among
  ablations.
- **held-out** = the other 124 datapoints. Used **once** for the final comparison, with the configuration frozen
  beforehand (§5.3). No parameter is changed after seeing held-out results; if RAG loses there, that is reported as
  the result.

## 4. Change register

| ID | Files | Change | Reason | Tests | Status |
|---|---|---|---|---|---|
| C-M1 | `rag/chunker.py`, `tests/test_chunker.py`, `README.md` | Function/method/header chunker with window fallback; README pins grammar tag `v0.21.0` | Proposal §4.3; locked tree-sitter 0.20.4 rejects ABI-15 grammar | 14 | committed (`df7a086`, `664f51b`) |
| C-M3 | `rag/bm25.py`, `rag/builder.py`, `tests/test_bm25.py`, `tests/test_builder.py` | BM25 and budgeted context builder; exact final token check; cached chunk cost; binary-search prefix crop | Proposal §4.2–4.3; real-data smoke test found a 3-token budget overrun and a 1.8 s/line prefix-crop hotspot | 22 | committed (`c31287e`) |
| C-M4 | `eval/line_generators.py`, `eval/preprocessors.py`, `eval/compare_arms.py`, `eval/config/compare_arms.yaml`, `rag/provider.py`, `tests/test_generation_hook.py` | Per-line context hook; `evaluate_generation` flags (`run_zero`, `run_context`, `limit`, `input_data`, provider factory); per-arm driver | Compare arms under identical seed/budget; preprocessed JSON has no snapshot, so RAG needs the dataset loader | 12 | written and used for E1; **uncommitted at time of writing** |
| C-1 | `eval/line_generators.py`, `tests/test_generation_hook.py` | Each saved generation row also records `datapoint_idx` and `repo`; new `subset_indices`; `select_subset(..., heldout_of=N)` and `evaluate_generation(heldout_of=N)` evaluate every datapoint except the N evenly spaced ones | Needed for paired, datapoint-clustered statistics and for the dev/held-out split (§3) | 5 new | written 2026-10-13, uncommitted |
| C-2 | `eval/compare_arms.py`, `eval/config/compare_arms.yaml` | `split=all\|dev\|heldout` with `dev_size` (default 20); `validate` rejects an unknown split and combining `split` with `limit_datapoints`; `split` and `dev_size` recorded in `summary.json` | Implements §3 | 2 new | written 2026-10-13, uncommitted |
| C-3 | `eval/analyze_arms.py`, `tests/test_analyze_arms.py` | Benchmark-style per-datapoint means and a paired bootstrap over datapoints (10,000 resamples, seed 0, 95% CI) of EM/ES differences between arms, overall and per category | Report uncertainty instead of single-run differences | 12 new | written 2026-10-13, uncommitted |

### 4.1 Verification notes for C-1..C-3

- Full suite after the changes: **65 tests pass** (48 before). They run on CPU with the GPU hidden
  (`CUDA_VISIBLE_DEVICES=""`).
- Split: the test with 144 items confirms dev (20) and held-out (124) are disjoint and cover every index; dev
  indices equal the E1 pilot's `i*144//20`, so **E1's datapoints are exactly the dev set**.
- Analysis: tests pin the benchmark definition (mean of datapoint means, categories averaged only over datapoints
  that have them), the zero-arm fallback, and bootstrap behaviour on consistent, null and noisy differences, plus
  determinism.
- Existing behaviour is unchanged by default (`datapoint_idx` is only written when the evaluator sets it; the
  static output shape is pinned by an existing test).
- Not verifiable without a run: end-to-end behaviour of `split=dev/heldout` on the real model (done in E2).

## 5. Pre-registration for E2 (written 2026-10-13, before any E2 run)

### 5.1 Reproduction check R

Re-run `zero`, `path_distance`, `rag` (defaults) with `split=dev` using the final code. Decoding is greedy, so the
dev results should equal E1's. **Pass criterion:** identical per-line predictions and identical EM to E1. If not,
record the discrepancy and the likely cause (e.g. GPU nondeterminism) before interpreting anything else.

### 5.2 Dev ablations (arm names `dev_*`, 1B model, dev set only)

| ID | Change vs RAG defaults | Predicted effect (stated now) | Rationale |
|---|---|---|---|
| A1 | `rag.prefix_cap=7000` (practically uncapped) | `infile` EM rises by at least 2 points and returns to roughly the zero-context level (0.238) or above; overall EM +1 point or more | P1 |
| A2 | `rag.include_non_py=false` | Overall EM changes by less than ±1 point; the share of lines with no retrieved chunk goes up | P2 (weak prediction; non-Python chunks may be harmless) |
| A3 | A1 + A2 | No better than the better of A1 and A2 by more than noise | additivity check |

Not yet implemented, hence not run in E2: best-first ordering and window-only chunking (proposal §4.5). They are
deferred to a later experiment so that this one stays interpretable.

**Amendment (2026-10-13, before any E2 run).** A1 uses `rag.prefix_cap=7900`, not 7000. A cap of 7000 still reserves
at least ~900 tokens for chunks, whereas in the static arms the prefix may occupy the whole 7,900-token budget (the
context is left-cropped first). `prefix_cap` equal to the budget reproduces that: the prefix is only cropped if it
exceeds the budget, and chunks get whatever space remains (none for very long prefixes, where RAG then degrades to
zero context, as the static arms effectively do). The predicted effects above are unchanged.

### 5.3 Decision rule

1. A1 tests whether my prefix cap was a flaw. Independent of its outcome, the **held-out comparison will use
   uncapped prefixes**, because the static arms have no cap and equal access to the file being completed is the fair
   comparison. A1's result is reported either way.
2. A2 is adopted only if dev overall EM improves by at least 1 point over the A1 configuration **and** no category
   drops by more than 2 points. Otherwise non-Python files stay indexed (the benchmark's own context includes them).
3. Query length, ordering, chunk granularity and the retriever are **not** tuned on dev in this round.
4. After the decision, the configuration is frozen (recorded here before running held-out) and evaluated once on the
   held-out 124 datapoints for arms `zero`, `path_distance`, `imports_first`, `rag` (frozen config) and `rag`
   (E1 defaults, for reference).
5. **Primary held-out comparison:** frozen RAG vs `path_distance`, overall EM, paired bootstrap over datapoints,
   95% CI. RAG "improves" only if the CI excludes 0 in its favour. Per-category and ES results are secondary and
   reported with their CIs; the `inproject` category has few lines (95 across 11 datapoints in the full set) and is
   descriptive only.

## 6. Results of E2 (run 2026-10-13, 17:25–17:39, all six arms exit 0, ~2–3 min each)

Code state: git HEAD `c31287e` plus uncommitted C-M4, C-1, C-2, C-3 (§4). Arms and outputs:
`~/lca_artifacts/python/starcoderbase-1b/small_context/arms/dev_{zero,path_distance,rag,rag_A1,rag_A2,rag_A3}/`.
Statistical reports: `~/lca_artifacts/reports/e2_vs_path_distance.md` and `e2_ablations_vs_rag.md`
(paired bootstrap over the 20 dev datapoints, 10,000 resamples, seed 0).

### 6.1 Reproduction check R: PASS

Per-line predictions of `dev_zero`, `dev_path_distance`, `dev_rag` are **identical to E1's** (0 differing
predictions out of 709 for each arm, same ground-truth order). Greedy decoding is deterministic on this setup, and the
new code (C-1..C-3) did not change behaviour.

### 6.2 EM on the dev set (20 datapoints, 709 lines; mean over datapoints, benchmark style)

| EM | zero | path_distance | rag (E1 defaults) | A1: prefix cap 7900 | A2: Python only | A3: A1 + A2 |
|---|---:|---:|---:|---:|---:|---:|
| all | 0.239 | 0.266 | 0.252 | 0.263 | 0.252 | 0.262 |
| infile | 0.238 | 0.279 | 0.207 | 0.227 | 0.217 | 0.238 |
| commited | 0.193 | 0.176 | 0.164 | 0.174 | 0.169 | 0.179 |
| common | 0.078 | 0.149 | 0.135 | 0.149 | 0.078 | 0.078 |
| inproject (3 datapoints) | 0.142 | 0.186 | 0.171 | 0.171 | 0.158 | 0.158 |
| non_informative | 0.170 | 0.178 | 0.188 | 0.188 | 0.178 | 0.178 |
| random | 0.275 | 0.315 | 0.315 | 0.315 | 0.315 | 0.315 |
| ES all | 60.5 | 64.2 | 63.4 | 63.6 | 63.4 | 63.5 |

(`analyze_arms.py` and E1's `summary.json` agree to rounding.)

### 6.3 Paired differences (Δ in EM, 95% bootstrap CI over datapoints; `*` = interval excludes 0)

| Comparison | all lines | infile |
|---|---|---|
| rag (defaults) − path_distance | −0.014 [−0.038, +0.006] | −0.073 [−0.187, +0.001] |
| **A1 − path_distance** | **−0.003 [−0.016, +0.011]** | −0.052 [−0.158, +0.009] |
| A3 − path_distance | −0.005 [−0.020, +0.009] | −0.041 [−0.150, +0.021] |
| zero − path_distance | −0.027 [−0.119, +0.031] | −0.041 [−0.150, +0.021] |
| A1 − rag (defaults) | +0.011 [+0.000, +0.029] | +0.021 [+0.000, +0.059] |
| A2 − rag (defaults) | −0.000 [−0.014, +0.013] | +0.011 [+0.000, +0.024] |
| A3 − rag (defaults) | +0.009 [−0.010, +0.031] | **+0.031 [+0.006, +0.073] \*** |

In ES, `zero` is significantly below `path_distance` (−3.7 [−8.8, −0.4]); no RAG variant differs significantly from
`path_distance` in overall ES (A1: −0.6 [−1.7, +0.5]).

### 6.4 Retrieval diagnostics (709 dev lines)

| arm | lines with no chunk retrieved | mean chunks | Python share of chunks | `infile` lines with no chunk |
|---|---:|---:|---:|---:|
| rag (defaults) | 58 | 13.5 | 54% | 21 / 208 |
| A1 | 101 | 13.0 | 55% | 31 / 208 |
| A2 | 322 | 10.0 | 100% | 80 / 208 |
| A3 | 335 | 9.8 | 100% | 88 / 208 |

Follow-up check on A2: in **6 of 20 dev datapoints (283 of the 709 lines)**, no line retrieved anything once
non-Python files were excluded (`fedml`, `data-diff`, `label-sleuth`, `ibm-block-csi-driver`, `langtest`,
`dongtai`). Their snapshots contain no usable Python other than the (excluded) completion file, so under A2 those
lines get zero context. This explains A2's large empty-retrieval count and why A3's `infile` and `common` EM equal the
zero-context values (0.238, 0.078): for those datapoints A3 *is* the zero-context arm.

### 6.5 Predictions scored (from §5.2, written before the runs)

| Prediction | Outcome |
|---|---|
| A1: `infile` EM +≥2 points | **Met numerically** (+2.1), but the CI [0.000, +0.059] touches 0 |
| A1: `infile` returns to ~zero-context level (0.238) or above | **Not met**: 0.227, 1.1 points short (and still −5.2 vs path_distance) |
| A1: overall EM +≥1 point | **Met numerically** (+1.1), CI lower bound 0.000 |
| A2: overall EM within ±1 point; empty-retrieval lines go up | **Met** (−0.0; 58 → 322, far more than I expected) |
| A3: no better than the better of A1/A2 by more than noise | **Met** (0.262 vs 0.263) |

### 6.6 Interpretation (what the dev data does and does not support)

1. **P1 (prefix cap) was a real confound, but only a partial explanation.** Removing the cap recovers part of the gap
   to `path_distance` (overall −1.4 → −0.3 points) but not the `infile` category (still about −5 points,
   not significant on 20 datapoints).
2. **On dev, no RAG variant differs from `path_distance` in a statistically detectable way**, and none is better.
   The best variant (A1) is −0.3 points [−1.6, +1.1]. This is *consistent with* H-null and gives no support to
   H-main; with 20 datapoints it cannot rule out differences of ±1–2 points either way.
3. **Retraction.** E1 reported discordant line pairs (RAG right/path_distance wrong 7 vs 18) as suggestive of RAG being
   worse. That count treats 709 lines as independent; they are clustered in 20 datapoints, and the clustered CI for the
   overall difference (−1.4 [−3.8, +0.6]) includes 0. The line-count comparison should not be cited.
4. **Excluding non-Python chunks is not a free win here**: for 30% of dev datapoints it removes all context. P2 is not
   supported as stated.
5. **Cautions.** Dev has 20 datapoints (3 with `inproject`), so per-category results, especially `common`
   (14 datapoints) and `inproject`, are descriptive only. Several comparisons are shown without multiple-comparison
   correction, so a single starred interval (A3 vs rag, `infile`) should not be over-read.
6. **Static context adds only modest EM over zero on dev**: `path_distance − zero` = +2.7 points in EM [−3.1, +11.9]
   (not significant), +3.7 in ES (significant). The benchmark's own best context therefore leaves little room, and a
   small RAG effect would be hard to detect with this sample.

## 7. Decisions and next steps

### 7.1 Decisions applied from the pre-registered rule (§5.3)

1. Held-out RAG uses an **uncapped prefix** (`rag.prefix_cap=7900`), decided in advance for fairness; A1 confirms it
   helps a little.
2. A2/A3 are **rejected**: A3 is not ≥1 point above A1 (−0.1), and `common` drops 7.1 points (0.149 → 0.078), over the
   2-point limit. Non-Python files stay indexed.
3. Query length, ordering, chunk granularity and the retriever were **not** tuned.

### 7.2 Frozen configuration for the held-out evaluation (recorded 2026-10-13, before any held-out run)

`arm=rag`, BM25 (`k1=1.2`, `b=0.75`), `rag.query_lines=10`, `rag.max_chunks=30`, `rag.prefix_cap=7900`,
`rag.include_non_py=true`, best-chunk-last ordering, token budget 7,900, model `starcoderbase-1b`, seed 42.
Code: the working tree as of E2 (C-M4, C-1..C-3 included); it must be committed before the held-out run so the
result is tied to a hash.

### 7.3 Held-out protocol (not yet run; awaiting approval)

Run with `split=heldout` (124 datapoints, the complement of dev): `zero`, `path_distance`, `imports_first`,
`rag` (frozen), and `rag` with E1 defaults (reference only). Estimated 15–17 minutes per arm (dev takes ≈ 2.5 minutes
for 20 datapoints), about 80 minutes in total. Analysis: `eval.analyze_arms` with `path_distance` as the reference.
Primary result: frozen RAG − `path_distance`, overall EM, paired bootstrap over datapoints, as in §5.3 item 5. If the
CI includes 0, the conclusion is "no detectable difference", not "RAG is as good".

**Amendment (2026-10-13, after the code commit `6a380e8`).** The held-out run in §7.3 has **not** been executed. The
research program in §8 supersedes it if approved: the held-out set is still untouched, so it may be redefined and used
once, at the end, for a stronger design.

### 7.4 Open items (not part of the held-out run)

- **Mechanism check (cheap, CPU only):** for each dev line, does the context actually contain the identifiers the
  ground-truth line needs, for `path_distance` vs RAG? This tests whether retrieval *finds* the right code or the
  model fails to use it. It was in the proposal's diagnostics and has not been done.
- **Untested ablations from the proposal:** best-first ordering, window-only vs function chunks, query length (3/10/30),
  dense and hybrid retrieval. Run only on dev, in a later experiment, and only after the held-out result is recorded.
- **Not yet done:** `imports_first` arm on dev, 7B model, `medium_context`.
- **Commits:** done on 2026-10-13: C-M4, C-1..C-3 and this log are in `168f962`..`6a380e8`.

## 8. Research program: design the strongest retrieval I can (pre-registered 2026-10-13, before any run in this program)

**Goal.** Find the retrieval design that maximises EM at the fixed 7,900-token budget, then measure it once on
untouched data. Honest outcome space: it may still not beat `path_distance`; that would be reported as such.

### 8.1 Why E2 leaves room for a better design (hypotheses about *why* BM25-on-last-10-lines is weak)

- **D1, dilution.** The prompt is filled to the budget with up to 30 chunks (mean 13.5). Most are probably irrelevant
  and may distract a 1B model; the static composer has the same flaw but orders by a strong prior (path proximity).
- **D2, no proximity prior.** `path_distance` encodes "files near the target matter". BM25 ignores it.
- **D3, lexical query.** The last 10 lines retrieve code that *looks similar*; the line often needs the *definition*
  of a name it uses (an import, a class, a helper). Similarity and definition lookup are different relations.
- **D4, chunk boundary.** A matched chunk shows the code that resembles the query; code that *follows* a match
  (what was written next elsewhere) is often more useful for completion (the idea behind RepoCoder-style retrieval).
- **D5, single signal.** Dense retrieval and fusion may match what BM25 misses (paraphrases, renamed identifiers).

### 8.2 Protocol changes (all decided before looking at any held-out result; held-out has never been run)

1. **Larger dev set.** 20 datapoints cannot separate designs a point apart. Redefine
   `dev` = the old 20 datapoints (indices `i*144//20`) **plus 28 more**, evenly spaced among the other 124
   (the j-th extra is `rest[j*124//28]`): **48 dev datapoints, 96 held-out**. The old dev set must stay inside dev,
   because E1/E2 already informed a design decision (uncapped prefix); an independent `i*144//48` split would have put
   13 of those seen datapoints into held-out. The 96 held-out datapoints have never been run, inspected or measured.
   E1/E2 numbers stay valid for the old 20; anything compared across sets is re-run on the new dev set.
   *Needs a small code change (C-4: a `dev_extra` option in the split logic, with a test that old dev ⊂ new dev and
   that dev and held-out are disjoint and complete).*
2. **Two-stage evaluation to afford many designs.** *No stage touches the held-out datapoints in any form*, including
   offline measurements of their snapshots.
   - *Stage A (CPU, minutes, dev datapoints only, no model output and no EM):* an offline
     **retrieval-quality proxy** (§8.3), used only to discard clearly weak designs.
   - *Stage B (GPU, ~5–7 min per design on 48 datapoints):* EM on dev for the designs that survive.
3. **Proxy validity is tested, not assumed.** Before relying on it, check on the six E2 arms (plus `zero`) whether proxy
   ordering matches EM ordering. If the rank agreement is poor (Spearman < 0.5), the proxy is demoted to a diagnostic
   and Stage B decides alone.
4. **Promotion rule (greedy forward selection).** A change is adopted into the running best design when its dev
   EM improves on the current best (point estimate > 0) **and** its paired 95% CI lower bound is above −1 point
   (no clear harm). Ties go to the simpler design. At most **25 Stage-B runs** in total (≈ 2.5 GPU hours) to bound
   dev overfitting; every run is logged, including failures.
5. **Selection bias is expected and stated.** The best of ~20 designs on 48 datapoints is optimistically biased; the
   held-out evaluation is the only unbiased number.
6. **Held-out is used once**, with the final design frozen and its commit hash recorded here first (§8.6).

### 8.3 Retrieval-quality proxy (Stage A)

For each target line `L` in the completion file:
`X(L)` = identifiers (non-keyword, length ≥ 2) that occur in `L`, are **defined in some other snapshot file**
(`def`/`class`/top-level assignment), and do **not** occur in the prefix (so the model cannot copy them from the file
itself). Over lines with `X(L) ≠ ∅`:

- **appearance recall** = share of `X(L)` that occurs anywhere in the prompt context;
- **definition recall** = share of `X(L)` whose definition (`def x`, `class x`, `x =`) occurs in the context.

Limits, declared now: lexical (a name appearing is not the same as the model understanding it); only covers lines
with cross-file names; says nothing about whether the model uses the context. It is a screen, not a result.
Coverage (number of lines with `X(L) ≠ ∅`, per category) is reported with every proxy number.

### 8.4 Candidate designs, in the order they will be tried (each gets its own log entry with settings, a prediction, and results)

| Step | Design change | Targets | Prediction (EM on dev, relative to the current best) |
|---|---|---|---|
| R0 | Re-baseline on the new dev set: `zero`, `path_distance`, `imports_first`, `rag` (E2-A1 config) | reference | defines the numbers to beat |
| R1 | Cap retrieved context: `max_context_tokens` ∈ {1k, 2k, 4k} instead of filling the budget (the rest of the budget goes to prefix) | D1 | a smaller cap helps by ≥ 1 point if dilution matters; no effect if not |
| R2 | Path-proximity prior fused into the score: `score = bm25_norm + λ·proximity`, λ ∈ {0.25, 0.5, 1} | D2 | closes part of the gap to `path_distance` on `common`/`inproject` |
| R3 | Import-aware boost for files the completion file imports | D3 | helps `inproject`, `commited` |
| R4 | Symbol-definition retrieval: resolve identifiers used in the prefix to their definitions via a tree-sitter definition index, place definitions (signatures first) in the context | D3 | raises definition recall sharply; EM gain on lines using cross-file names |
| R5 | Continuation chunks: after a window match, include the following lines | D4 | helps lines that continue a pattern |
| R6 | Query variants: last 3 / 10 / 30 lines; identifier-only query | D3 | 3 lines is too noisy, 30 too diffuse; 10 near a plateau |
| R7 | Dense retrieval (code-embedding model, cosine, per-repo cache) and hybrid reciprocal-rank fusion with BM25 | D5 | dense alone ≈ BM25; hybrid ≥ either by 0.5–1.5 points |
| R8 | Ordering best-first vs best-last; near-duplicate removal (MMR) | D1 | ordering effect small |
| R9 | Combine the adopted changes, check for interaction on dev | | gains roughly additive, at most |

Candidate embedding models for R7 (availability and licence to be verified before use): a small code-embedding model
that loads under the pinned `transformers`, e.g. `Salesforce/codet5p-110m-embedding` or
`jinaai/jina-embeddings-v2-base-code`. Decision on which is made by a loading/speed smoke test, logged.

### 8.5 Settings recorded for every Stage-B run

Arm name, git hash (and whether the tree is dirty), full Hydra config (`config.json` is written by the driver), model,
dataset revision, split and `dev_size`, seed, token budget, wall time, and the output directory. Every table in this
log states which of these differ from the reference run.

### 8.6 Final evaluation (after the design is frozen)

1. Freeze: record the design's configuration and the commit hash here.
2. Run once on the 96 held-out datapoints, `starcoderbase-1b`: `zero`, `path_distance`, `imports_first`,
   `rag` at E2-A1 settings (the "v1" design), and the final design.
3. Run `zero`, `path_distance` and the final design on `starcoderbase-7b` (same held-out datapoints) to check that the
   direction holds on a larger model (proposal §5, success criterion 1).
4. Primary result: final design − `path_distance`, overall EM, paired bootstrap over datapoints. Report
   secondary categories and ES with their CIs; no re-tuning afterwards.

### 8.7 Engineering plan (each a small commit, tested, listed in the change register when done)

New configuration options in `rag/builder.py` / `rag/provider.py` (default values reproduce today's behaviour, pinned
by tests): `max_context_tokens`, proximity prior, import boost, definition index, continuation lines, query mode,
ordering, MMR. New `rag/dense.py` for embeddings. New `eval/proxy_metrics.py` for §8.3 (CPU only).

### 8.8 Approvals needed before the first run

1. Redefining dev as 48 datapoints (held-out 96): safe because held-out has never been run; changes §3.
2. Downloading one or two embedding models from HuggingFace for R7 and running them on the GPU.
3. The compute budget in §8.2 item 4 (≤ 25 dev runs, about 2.5 GPU hours) plus the final runs (~1.5 GPU hours).

**Approved by the user on 2026-10-13: all three items**, with the instruction to add any further safeguards that make sense.

### 8.9 Decisions on additional safeguards (2026-10-13, before any run in this program)

- **Larger dev set (64 datapoints): not adopted.** It would leave 80 held-out datapoints. The held-out comparison is the
  only unbiased result and its precision matters more than a modestly less noisy dev set (standard error scales with
  1/√n: 80 vs 96 held-out datapoints costs about 9% more width on the final confidence interval, while dev would
  improve by about 15%). Dev stays at 48.
- **Restricted combination search: adopted.** After R1–R8 each have been tried once and adopted or rejected by the
  promotion rule, only **one** combined run (R9) plus **at most two** pairwise interaction checks are allowed. No
  open-ended re-tuning. The 25-run cap stays.
- **Provenance recorded automatically.** Every run's `summary.json` now includes the git commit, a dirty-tree flag, the
  wall-clock time, `split`, `dev_size` and `dev_extra` (change C-4 below).
- **Reproduction check before trusting any new run.** R0 re-runs `path_distance` and `rag` on the new dev set; the 20
  datapoints shared with E2 must reproduce E2's per-line predictions exactly (checked on the overlapping datapoints).

### 8.10 Change register additions

| ID | Files | Change | Reason | Tests | Status |
|---|---|---|---|---|---|
| C-4 | `eval/line_generators.py`, `eval/compare_arms.py`, `eval/config/compare_arms.yaml`, `tests/test_analyze_arms.py`, `tests/test_generation_hook.py` | `dev_indices` / `heldout_indices` (dev = `dev_size` evenly spaced + `dev_extra` more among the rest); `evaluate_generation(select=...)`; driver uses them (`dev_extra: 28` default, giving 48 dev / 96 held-out on `small_context`); `summary.json` records git commit, dirty flag and wall time | §8.2 item 1 and §8.5 | 7 new (suite: 72 tests pass) | written 2026-10-13, uncommitted |

### 8.11 R0: re-baseline on the new dev set (pre-registered 2026-10-13, before the run)

Settings: `split=dev` (48 datapoints; `dev_size=20`, `dev_extra=28`), `starcoderbase-1b`, budget 7,900, seed 42,
greedy decoding, from `~/ts_test`. Arms: `zero`; `path_distance`; `imports_first`; `rag` with `rag.prefix_cap=7900`
(the E2-A1 configuration, called **v1** from here on). Output names `r0_<arm>`. The code is the working tree with
uncommitted C-4 (recorded as dirty in `summary.json`).

Predictions (stated before the run):

1. `path_distance − zero` in overall EM lies between +1 and +4 points (E2 on 20 datapoints gave +2.7).
2. `v1 − path_distance` lies within ±2 points of 0 (E2: −0.3).
3. `imports_first − path_distance` lies within ±2 points of 0 (no prior information).
4. The 20 datapoints that are also in E2 reproduce E2's per-line predictions exactly for `zero`, `path_distance`, `v1`.

R0 decides nothing by itself; it fixes the reference numbers for R1 onwards.
