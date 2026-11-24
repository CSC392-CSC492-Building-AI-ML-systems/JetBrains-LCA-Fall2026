# API-list context experiment

The [completed evaluation](RESULTS.md) covers all 150 tasks and seven
configurations. Its archive is `results/experiment/`; use the saved-run
evaluate-only command below after renaming, rather than starting generation.

This runner uses real GPT-4.1 API responses. Preparation only builds real benchmark
prompts and computes the request count/cost; it produces no simulated responses or
benchmark scores. See [the protocol](EXPERIMENT_PLAN.md) and [audit](audit/summary.json).

Run from `library_based_code_generation/` in the WSL environment used for evaluation:

```bash
source .venv/bin/activate
python -m pip install openai==1.60.1 rank-bm25==0.2.2 tiktoken
python -m src.evaluation.long_context
```

The preparation command defaults to all 150 tasks at the audited dataset revision.
It prints the real unique request count, uncached input tokens and an estimate
allowing all 2,048 output tokens per request. It also records a more conservative
byte-based reservation bound used for spending control. Review this estimate
before choosing a cap. The default price settings are $2/M input and $8/M output.

## Arms and controls

| Arm | Evidence |
| --- | --- |
| plain | No API-name evidence |
| bm25_20 | Official BM25 top-20 |
| direct_8k | Supplied API-list order, maximum 8,192 evidence tokens |
| full_list | Entire supplied API list, required to fit within 65,536 evidence tokens |
| bm25_8k | Official BM25 order, maximum 8,192 evidence tokens |
| random_8k | Seeded shuffled candidate list, packed to BM25-8K's achieved token count |
| bm25_20_repeat | Independent real request with the same prompt/settings as bm25_20 |

All evidence arms keep the same hint wording. All use the unfiltered candidate
inventory, preserving duplicates.
BM25 preserves the official identifier tokenizer, query tokenizer and reverse
NumPy argsort ties. Names are packed whole, skipping those that do not fit. The
random match must be within max(8 tokens, 2% of target), otherwise preparation
stops. Its seed and achieved mismatch are saved. Reference names are used only
for audit/scoring, never randomization, retrieval or packing.

The fixed model is `gpt-4.1-2025-04-14`, temperature 0, top-p 1, up to 2,048 output
tokens. Context allowance is configured to 128,000 with a 1,024-token safety reserve.
The full user prompt is counted with model-specific tiktoken plus estimated framing;
provider usage is saved separately. Changing the tokenizer to `--tokenizer bytes`
defines a different experiment. Full-list overflow is an error, not silent truncation.

The audited dataset has 62 repositories; all source lists are under 50K tokens.
This is an API-list study, not a 100K+ context-window evaluation. The runner verifies
the dataset against the saved audit before preparing remote-Hub data. Local Arrow
or JSONL data can be supplied with `--dataset-file`; repository group identities
must be established separately for a different dataset.

## Real generation

Once you select a total cap and set `OPENAI_API_KEY` in the shell, use:

```bash
# Replace CAP_USD with your chosen dollar amount.
python -m src.evaluation.long_context --generate --spend-limit-usd CAP_USD
```

For a real two-task smoke run, append `--limit 2` to both preparation and generation
commands. It uses its own run identity and costs extra; all smoke/full generation
must use the same `.cache/long_context_spending.sqlite` ledger. A changed 
source/package/configuration makes a different run identity.
Finish installing dependencies before final preparation and make sure to keep them fixed.

Byte-identical requests for the same task share one immutable response across
ordinary arms. The repeat has a separate request key and always requests an
independent response. All responses must report the frozen snapshot. No historical
alias-run predictions are imported. Length-limited replies are scored and flagged;
empty/refused/filtered replies are recorded as failures and stop generation.

## Resume and spending

Rerun the identical command to reuse verified successes. Keep the same dataset,
settings, source, package versions, output directory and ledger. The ledger reserves
conservative cost transactionally before each request and settles from returned
usage using uncached list prices. It includes all runs sharing the ledger, so the
cap is cumulative, not an additional allowance on each invocation. Cache discounts
can make these accounted costs higher than the invoice. A changed ledger identity
is rejected for an existing generation run. Do not delete or switch ledgers to
reset spending. Single-runner operation is intended.

Reaching the cap before submission leaves existing results reusable. Errors and
interrupted/in-flight requests retain reservations; missing-usage responses retain
their reservations too. Failed/incomplete requests are never retried automatically:
inspect the request and provider billing, then reconcile manually before retrying.
This avoids duplicating an unknown billable request. Stop and review if recorded
usage exceeds its reservation. The cap covers this runner, not unrelated account use.

## Scoring and artifacts

```bash
python -m src.evaluation.long_context --evaluate-only
# Re-score a particular saved run without changing its generation identity:
python -m src.evaluation.long_context --evaluate-only --run-dir long_context/results/RUN_ID
```

No API clients are constructed during preparation or evaluation. The scorer uses
the unchanged benchmark ChrF/API Recall implementations and parser. Results live
under `long_context/results/<identity>/`:

- `manifest.json`: pinned data, source hashes, package versions, tokenizer, settings.
- `preparation.json`: audit, exact unique request estimates, deduplication counts.
- `arms/<arm>/<index>.json`: exact prompts, selected names, packing decisions, request IDs.
- `requests/<id>.json`: real raw responses, hashes, returned model, finish reason,
  usage, latency, errors and known cost. Shared responses are counted once in totals.
- `comparison.json`, `comparison.csv`, `REPORT.md`: common-cohort metrics, coverage,
  three primary contrasts and exploratory historical-control/repeat comparisons.
- `per_example_scores.json`, `review_cases.json`: per-task scores, extra/unlisted
  call diagnostics and a fixed-seed sample for manual review.

The three primary contrasts are full-list minus direct-8K, BM25-8K minus direct-8K,
and BM25-8K minus random-8K. The meaningful API Recall target is +0.02. Bootstrap
intervals use 10,000 draws over repository groups; report group counts and sparse
coverage. Intervals are descriptive and unadjusted for multiple comparisons; this
runner makes no p-value/rejection claims. A null comparison is not equivalence.
Full-fit versus truncated strata are reported separately. Numerical ties use 1e-8;
that is not a model-noise threshold. The independent repeat estimates variation.

Extra-library-call fraction uses called names matching the candidate inventory;
unlisted-call fraction includes builtins, local and third-party functions. Neither
proves incorrectness. Zero denominators are null with explicit reporting counts.
No code execution or semantic correctness is claimed. Published paths are portable;
private credentials and SQLite spending ledgers are ignored by Git.
