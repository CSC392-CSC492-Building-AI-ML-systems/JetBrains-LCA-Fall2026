# Completed API-list context evaluation

BM25-selected context at an 8K-token budget achieved the highest API Recall,
**0.470067**, on all **150 tasks from 62 repositories**. It exceeded direct-8K by
**3.02 percentage points** and token-matched random context by **2.53 points**.
Both repository-bootstrap intervals narrowly exclude zero, but are descriptive
and unadjusted for multiple comparisons. These are promising reference-API-usage
results, not evidence of execution correctness.

Supplying the full list improved recall over direct-8K by only **0.66 points**,
with an interval spanning zero. The experiment does not establish a benefit from
the additional names, nor establish equivalence. Direct-8K had the highest ChrF.

## Complete matched comparison

Seven configurations completed on the same 150 test tasks: **1,050 logical
predictions from 972 real API responses**. The 78 direct-8K/full-list cases with
identical requests shared one response each. The top-20 repeat always used an
independent real request. No simulated predictions or historical alias-run
predictions were imported.

| Arm | Mean evidence tokens | ChrF | API Recall | Scored | Output-limit cases* |
| --- | ---: | ---: | ---: | ---: | ---: |
| plain | 0.0 | 0.453374 | 0.359943 | 150/150 | 3 |
| bm25_20 | 117.3 | 0.455836 | 0.397102 | 150/150 | 1 |
| direct_8k | 5,985.7 | 0.460572 | 0.439834 | 150/150 | 6 |
| full_list | 11,209.5 | 0.457716 | 0.446413 | 150/150 | 7 |
| bm25_8k | 5,985.8 | 0.457057 | 0.470067 | 150/150 | 4 |
| random_8k | 5,985.3 | 0.456570 | 0.444789 | 150/150 | 7 |
| bm25_20_repeat | 117.3 | 0.456040 | 0.393204 | 150/150 | 3 |

*Shared responses can appear in two rows. Across
unique requests, 947 ended normally and 25 reached the output limit. All remain
in scores. Direct-8K's ChrF advantage over BM25-8K is not established by their
saved paired ChrF interval.

## Primary paired comparisons

Differences below are API Recall **percentage points**, not relative percentages.
The three primary contrasts were specified before generation; the proposed
smallest meaningful gain was +2 points. Intervals use 10,000 seeded bootstrap
draws over whole repositories, preserving their paired tasks.

| Primary contrast | Difference [95% interval], points | Wins / losses / ties |
| --- | --- | --- |
| Full list minus direct-8K | +0.66 [-1.00, +2.35] | 28 / 16 / 106 |
| BM25-8K minus direct-8K | +3.02 [+0.08, +6.16] | 59 / 42 / 49 |
| BM25-8K minus random-8K | +2.53 [+0.03, +4.92] | 59 / 39 / 52 |

BM25-8K's point estimates exceed the +2-point target, but the lower bounds are
near zero: these intervals do not establish that the true gains exceed +2 points.
No multiplicity-adjusted hypothesis tests or rejection claims were made. An
interval spanning zero is not proof of no effect.

The independent top-20 repeat differed by **-0.39 points [-1.89, +0.98]**, with
26 wins, 29 losses and 95 ties. This demonstrates within-configuration variation;
one repeated arm does not define a universal noise threshold. Numerical ties use
an absolute tolerance of 1e-8, which is separate from model variability.

Exploratory comparisons were full-list versus top-20, **+4.93 [+1.63, +8.26]**,
and top-20 versus plain, **+3.72 [+0.46, +6.52]**. All six contrasts, including
ChrF intervals, are retained in the [comparison JSON](results/experiment/comparison.json).

## More evidence versus ordering

The [pre-generation audit](audit/summary.json) found 100% reference-name coverage
in the full inventory but only 9.95% macro reference-name retrieval recall for
official BM25@20. This is hint coverage, not generated API Recall: the model can
use names not present in its hints. Full-list contexts ranged from 582 to 49,180
tokens, with a median of 7,585. All lists fit within the 65,536-token allowance;
this is an API-list study, not a test of 100K+ context windows.

| Exploratory stratum | Tasks | Full-list minus direct-8K, points | BM25-8K minus direct-8K, points |
| --- | ---: | ---: | ---: |
| Full list fits within 8K | 78 | +0.00 | +4.72 |
| Direct-8K omits names | 72 | +1.37 | +1.18 |

The zero direct/full difference in the 78 full-fit tasks is by construction:
identical prompts shared their responses. In that stratum BM25 can alter order,
and its independently generated responses also introduce generation variability.
The larger BM25 gain in full-fit tasks cautions against attributing the overall
gain solely to retrieving omitted APIs. Truncated tasks can change both selection
and order. These subgroup means are exploratory and have no separate intervals.

Random context used seed 42 and was matched to each task's achieved BM25-8K
evidence count. The maximum shortfall was 10 tokens and the mean was 0.53 tokens,
within the frozen max(8 tokens, 2% of target) tolerance. Thus the random comparison
controls approximate evidence volume, rather than simply comparing different caps.

## Extra-call diagnostics

API Recall does not penalize extra calls. These macro fractions use unique bare
called names. Zero-denominator cases are excluded; reporting counts appear in
parentheses. They are diagnostics, not semantic precision or hallucination rates.

| Arm | Extra names among used library names | Calls outside the library inventory |
| --- | ---: | ---: |
| plain | 64.29% (149) | 50.27% (149) |
| bm25_20 | 62.83% (150) | 45.52% (150) |
| direct_8k | 63.16% (145) | 35.54% (145) |
| full_list | 62.61% (144) | 34.78% (144) |
| bm25_8k | 61.49% (147) | 34.34% (148) |
| random_8k | 61.95% (145) | 36.20% (145) |
| bm25_20_repeat | 63.07% (150) | 46.23% (150) |

BM25-8K's higher recall is not accompanied by a larger mean extra-library-name
fraction than direct-8K in this diagnostic. That does not prove better precision
or correctness: valid alternative programs may use extra APIs, and unlisted names
include builtins, local functions and third-party libraries. Bare-name collisions
cannot distinguish defining classes. The [review cases](results/experiment/review_cases.json)
select ten tasks for manual inspection; they are not completed semantic error
annotations.

## Usage, outcomes and accounting

| Measure | Verified result |
| --- | ---: |
| Successful unique API responses | 972/972 |
| Physical responses reaching output limit | 25 |
| Provider input tokens | 4,302,508 |
| Cached input tokens, included above | 377,856 |
| Provider output tokens | 933,537 |
| Sum of request latencies | 96.1 min |
| Median request latency | 5.52 s |
| Uncached list-price accounting | US$16.073312 |

Every saved request succeeded and reported the frozen snapshot. No examples
remain. Length-limited responses were retained and explicitly counted; no BM25
fallback replaced generated code.

The ledger uses $2/M input and $8/M output and conservatively charges cached
input at the uncached rate. **US$16.073312 is list-price accounting, not a verified
invoice**; actual billed cost may be lower. Summed request latency excludes
preparation, scoring, file I/O and pauses; end-to-end wall time was not measured.
No local GPU or model judge was needed.

## Frozen setup and limitations

- Dataset: `JetBrains-Research/lca-library-based-code-generation`, test indices
  0-149; revision `dc460b7e403f63e58ec7128b80eaf2c9c95344f8`.
- Dataset-content SHA-256:
  `96421d0f1e2775fb7fd6d426a85f7c3396eb7e0028f87991d591a754fc568ded`.
- Model: `gpt-4.1-2025-04-14`; temperature 0, top-p 1, frequency/presence penalties
  0; maximum 2,048 output tokens.
- Same instruction template and hint wording across evidence arms. The supplied
  API-name inventory was preserved, including duplicates, private names and test
  helpers. No source code, signatures or documentation were added.
- Context allowance: 128,000; safety reserve: 1,024. Evidence counted with
  `o200k_base` / tiktoken 0.14.0; final full prompts were independently token-checked.
  Message framing remains an estimate; provider usage is logged separately.
- Unchanged benchmark ChrF/API Recall and tree-sitter call extraction. Bootstrap:
  62 repository groups, 10,000 draws, seed 42. Repository sizes are unequal.
- One model, one dataset, one random-context seed and one repeated arm. Aggregate
  audit findings were seen before the experiment; this is not an untouched holdout.
- No code-execution/test-pass metric, formal hypothesis rejection or semantic
  equivalence claim. Historical unpinned alias results are not merged with this run.

## Artifacts and reproduction

- [Saved score report](results/experiment/REPORT.md),
  [comparison JSON](results/experiment/comparison.json),
  [comparison CSV](results/experiment/comparison.csv).
- [Per-task scores](results/experiment/per_example_scores.json),
  [diagnostics](results/experiment/diagnostics.json),
  [review cases](results/experiment/review_cases.json).
- [Frozen manifest](results/experiment/manifest.json),
  [preparation and audit](results/experiment/preparation.json),
  [artifact verification](results/experiment/verification.json),
  [accounting summary](results/experiment/ledger_summary.json).
- [Pre-generation audit](audit/summary.json), [protocol](EXPERIMENT_PLAN.md),
  [runner instructions](README.md). Exact prompts and real API responses remain
  under the completed run's `arms/` and `requests/` directories.

The archive was renamed to `experiment`; its manifest still identifies original
run `c3e1b77487ae49285a84`. Do not rerun generation merely to rebuild this report:
generation expects its original hash directory. From
`library_based_code_generation/`, use:

```bash
# Verify saved artifacts and regenerate diagnostics; standard library only.
python long_context/verify_results.py

# Recompute official scores from saved real responses; no generation calls.
./.venv/bin/python -m src.evaluation.long_context --evaluate-only \
  --run-dir long_context/results/experiment
```

Publication verification checked all 1,050 context artifacts and 972 response
checksums, request/context mappings, source hashes, saved budget bounds, complete
coverage, aggregate means and paired mean differences. It did not rerun the
official scorer or bootstrap intervals. Provider records are saved evidence, not
cryptographically authenticated responses. Published links are relative; private
credentials and the SQLite spending ledger are excluded.
