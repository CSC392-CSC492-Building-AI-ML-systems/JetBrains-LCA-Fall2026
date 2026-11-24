# Audit-gated API-context experiment

Status: implemented protocol, awaiting the real-request cost review and spending
cap. No paid generation is authorized by this document. The earlier five-arm
prototype has been replaced by the seven-configuration runner described below.

## Unpaid feasibility gate

Run `python retrieval_recall.py` from LBCG's directory. It pins the resolved test
revision, hashes the full data, reproduces official BM25 query/index tokenization
and reverse-argsort tie handling, and records per-task distinct reference coverage
in the full corpus and top-20. It also reports full-list token sizes and candidate
grouping fields. Reference data is evaluation-only; it must never enter ordinary
context construction, filtering, ordering or model prompts. This aggregate audit
is development information, so the eventual test is not an untouched holdout.

Full-list coverage is the ceiling for reference API names supplied through hints,
not an upper bound on generated API Recall: the model can know unlisted APIs.
Assess missing-reference names, bare-name collisions, duplicated names and empty
references explicitly. Preserve the official baseline's original candidate policy.
Any filtered/deduplicated corpus is a separate matched experiment applied to every
arm, rather than a silent change to official BM25.

If additional hints rarely expose a missing reference API, report that constraint
and reduce scope; do not assume a larger context will repair evidence that does
not exist. If 8K/64K produce identical prompts, collapse them and call the study
"full API list versus top-20", not a 100K-window experiment. Freeze any alternative
budgets from context-size statistics, without optimizing against generation scores.

## Frozen design following the audit

The November 19 audit on all 150 tasks found 100% full-list reference-name
coverage and 9.9482% macro official BM25@20 coverage. All tasks have reference
names beyond top-20. Full-list context sizes range from 582 to 49,180 tokens
(median 7,585); 78 tasks fit at 8K and all fit at 64K. These are reference-name
retrieval scores, not generated-code API Recall. See `audit/summary.json`.
The audit also verified `repo_full_name` against owner/name using the dataset's
`owner__name` convention (case-insensitive): 62 repository groups, with group
sizes retained in the summary. Use this repository identifier for clustered
bootstrap; do not use owner alone (59 owners can own multiple repositories).

Given that evidence, prefer six configurations: plain, official BM25@20,
direct-8K, full-list-direct, BM25-8K, and random-subset-8K. Add an independent
BM25@20 repeat. Freeze three primary contrasts: full-list minus direct-8K
(additional evidence), BM25-8K minus direct-8K (selection/order), and BM25-8K
minus random-8K (relevance versus approximately matched volume). Use +0.02
API Recall as the proposed meaningful effect. All other contrasts, including
full-list versus top-20, are exploratory. The alternative designs below are
contingencies, not extra arms to add automatically.

Match random evidence to the achieved per-task BM25-8K evidence count with a
predefined whole-name tolerance, recording mismatch. Freeze one random seed
before inference. For the 78 full-fit tasks, full-list/direct-8K are potentially
the same prompt; audit exact request hashes and share responses if identical.
For the remaining 72, the comparison can measure increased evidence. The
independent repeat bypasses reuse deliberately. Maximum logical cases are
7 * 150 = 1,050, but exact-prompt reuse reduces paid requests. This is an
API-list study, not an evaluation of 100K+ contexts.

Use one frozen snapshot, dataset order, instruction template, 2,048 output tokens,
temperature 0 and top-p 1. Retain the same hint wording for all evidence arms,
including full lists. Pin tokenizer/version; reserve the complete instruction,
chat overhead, output and safety tokens before evidence. Log actual token counts.

Random-8K shuffles the full candidate inventory using seed 42 combined with a
label-free task digest. It greedily packs whole names to the achieved BM25-8K
token count, skipping oversized names. Maximum shortfall is max(8 tokens, 2% of
target); preparation stops if this cannot be met. Record achieved differences.
Do not assume dataset order is semantically good. Full-fit tasks isolate ordering
when sets match; truncated tasks confound selection and order. Report both strata.

An oracle containing reference APIs plus distractors is optional diagnostic work
only. It must use separate output paths and be excluded from main scores and
selection decisions. It is not a deployable method.

## Inference and statistical commitments

The three primary contrasts are the audited-design contrasts named above.
Treat the remaining contrasts and subgroup findings as exploratory. Smallest meaningful
API Recall improvement: +0.02 absolute. This is a scientific target, not a claim
that 150 tasks can detect it. Report interval widths; an interval containing zero
and meaningful improvements means inconclusive, not equivalence. Use a planned
multiple-comparison procedure if adding hypothesis tests. The current runner reports
descriptive, unadjusted intervals and makes no p-value/rejection claims.

Repeat BM25@20 as an independently requested arm on all selected tasks to measure
within-configuration variation. Normal runs deduplicate byte-identical requests
by dataset task, exact messages, frozen model and all decoding settings, caching
one immutable response and recording reuse. The repeat deliberately bypasses
cross-arm deduplication and has a separate replicate identity. Never deduplicate
by selected-name set alone: order and wording change requests.

API Recall ties use an absolute tolerance of 1e-8 for numerical comparison.
Separately report task changes of at least 0.02 and the repeated-arm variation;
do not present floating-point tolerance as a model-noise threshold. Freeze a ChrF
numerical tolerance of 1e-8 as well. Temperature 0 is not deterministic.

Validate library identifiers using actual schema and task mappings before clustered
bootstrap. Group hashes or bare field names are not evidence of valid identities.
If reliable, resample whole libraries with all paired tasks, report group counts
and sizes, and use 10,000 seeded draws. If unavailable, explicitly label sample-level
intervals and their independence limitation. Few groups limit precision.

## Metrics and failure review

Keep official ChrF/API Recall, complete/common coverage, paired intervals and
wins/losses/ties. Add unique generated-call diagnostics using the benchmark parser:
among observed calls matching candidate-library names, fraction absent from the
reference; count/ratio of calls not in the candidate-library inventory; and total
unique called names. Report zero-denominator cases, macro denominators and the
full-list reference coverage alongside these metrics.

Calls outside that inventory include builtins, external libraries, local functions
and missing inventory entries. They are unlisted calls, not automatically invented
or incorrect APIs. Bare-name collisions also limit these diagnostics. Inspect a
fixed sample manually; syntax checks and ChrF do not establish execution correctness.
Separate evidence-missing cases, evidence-present-but-unused cases and extra-call
cases as diagnostic screens rather than proved semantic failures.

## Snapshot, cost and artifacts

Use `gpt-4.1-2025-04-14` if available at preflight and record the returned snapshot.
Older alias runs without returned-model metadata remain historical. Rerun plain
and official BM25 on the frozen snapshot for primary comparisons; neither label
them compatible by assumption nor silently import their responses.

At the checked standard rates ($2/M input and $8/M output), calculate exact planned
unique-request input tokens plus conservative maximum output before generation:
`2 * input_tokens / 1e6 + 8 * (unique_requests * 2048) / 1e6` USD. Include independent
repeat and smoke calls; do not assume prompt-cache discounts. Archive the price
source/date and require a concrete total spending cap selected by the experiment
owner before paid calls. Reserve worst-case cost persistently before each call,
settle from usage afterwards, and retain unresolved reservations on timeouts.
The runner implements these reservations in a persistent SQLite ledger. An existing
generation run is bound to its original ledger identity. Errors and unknown billing
states require manual reconciliation; no automatic generation retries occur.

OpenAI currently lists shutdown on January 14, 2027. Verify access and availability
again before committing funds. Source: https://developers.openai.com/api/docs/deprecations
Pricing: https://developers.openai.com/api/docs/models/gpt-4.1 (checked November 19, 2026).

Save portable-path manifests, source/input/prompt hashes, package versions, exact
submitted messages, chosen APIs/order, raw responses, returned model, finish reason,
usage, latency, cost, truncation/failures, response hashes and reuse records. Stop
on budget exhaustion and resume verified successes without regenerating. Do not
overwrite historical results or conceal incomplete coverage. Archive the completed
protocol, predictions, per-task metrics, comparison and manual-review findings.

## Before full inference

Run full preparation with the final source/packages, review actual unique requests
and the output-cap estimate, and select an explicit cumulative spending cap. Set
the API credential in the execution environment. Preparation produces prompts and
estimates only; all generation and benchmark outputs must come from real API
responses. No simulated responses are used.
