# Local full-workflow validation

Validated on 2026-11-18 using **42 saved Mistral and Qwen predictions**, without
new inference or GitHub Actions. All five selected cases have valid controls:
the failed revision reproduces the recorded failure, and its reference patch
passes the complete selected workflow. An independent audit verified the raw
answers, applied edits, commands, matrix entries, execution evidence, and scores.

The evaluator executed **55 job configurations across 23 distinct workflow
replays**. Identical patched source trees share explicitly recorded executions;
seeds remain separate predictions. These are five deliberately selected
validation cases, not a representative sample or the paper's 77-case evaluation.

## Controls and execution scope

| Dataset IDs | Complete workflow | Failed control | Reference control |
| --- | --- | --- | --- |
| 33, 34, 36, 37: SkyPilot | Repository-wide YAPF, Black, and both isort checks; Python 3.8 | Recorded YAPF diff hunks reproduced for every case | All four checks pass for every case |
| 144: Starlette | Lint/type checks, package and documentation builds, full tests, and 100% coverage; all five Python versions | Recorded `TypeError: unhashable type: 'dict'` reproduced on Python 3.8 through 3.12 | Entire workflow passes on all five versions |

Actual Starlette Python versions were 3.8.20, 3.9.25, 3.10.22, 3.11.18,
and 3.12.16. Execution used Linux amd64 containers, Debian instead of Ubuntu,
and the dataset's dated package-index service pinned by image digest. These
differences from hosted CI are retained in the execution evidence. Source files
and a Git index are recreated from the failed commit; hosted checkout metadata
and the original Git history are not reconstructed.

## Model results

Each cell below counts successful **single saved predictions**, not cases where
any of several attempts passed. Every seed was 0, 1, or 2, as in the original
saved experiments. The Qwen experiments cover different case subsets, so their
overall percentages must not be compared directly with the five-case Mistral
percentages.

| Model and saved arm | Evaluated cases | Passing predictions | Mean local CI Pass@1 across seeds |
| --- | --- | --- | --- |
| Mistral 7B, 4-bit, no retrieval | 33, 34, 36, 37, 144 | 0/15 | 0% |
| Mistral 7B, 4-bit, original BM25 | 33, 34, 36, 37, 144 | 3/15 | 20% |
| Qwen2.5-Coder 7B, 4-bit, original output format | 37 | 3/3 | 100% |
| Qwen2.5-Coder 7B, 4-bit, constrained raw output | 37 | 1/3 | 33.3% |
| Qwen2.5-Coder 7B, 4-bit, no retrieval | 144 | 0/3 | 0% |
| Qwen2.5-Coder 7B, 4-bit, lean BM25 | 144 | 0/3 | 0% |

All three Mistral BM25 successes come from **case 37**. On that shared case,
the Mistral baseline passes 0/3, Mistral BM25 passes 3/3, Qwen's original format
passes 3/3, and Qwen's constrained raw format passes 1/3. This establishes an
execution-based success for the saved BM25 repair on one case; it does not
establish a general retrieval improvement. Byte-identical repairs are not
independent demonstrations of robustness.

## Why the earlier Starlette proxy scores were optimistic

Both Qwen arms fail Starlette's **Ruff formatting check before tests execute**.
The raw responses omit the blank line separating the nested `send` function
from the following comment. The lean-retrieval response corrects the literal
and preserves the reference AST, but its missing blank line still fails CI.
Inspection confirmed that this blank line was already absent in the raw answer.
The no-retrieval seeds 0 and 1 also add an unsupported `send.call_count` assertion.

Thus the prior literal/AST checks correctly detected the local logical repair,
but did not establish workflow success. The evaluator retains the original
outputs and parsing behavior; no formatting or other edits were added to rescue
these predictions. Mistral's Starlette repairs also fail the full workflow.

## Verification, artifacts, and next step

All **45 baseline tests and 5 retrieval tests pass**. The independent execution
audit passes. A negative end-to-end audit rejected deliberately altered workflow
commands, even when both the job evidence and workflow summary were changed.

- [Portable verified metrics and execution summaries](local_ci_metrics.json)
- [All 68 workflows and their current support status](local_ci_inventory.json)
- [Evaluator setup, execution, audit commands, and scoring rules](../local_ci/README.md)

The original raw evidence and exact evaluator source snapshots remain under
`ci-builds-repair-benchmark/out/local-ci/6f6774e258b497b5fd71/`. The published
metrics include checksums for the execution evidence. No project source,
prompts, responses, credentials, or model weights are added to these exports.

The compiler currently recognizes seven workflows; **five have been validated**
in this run, and the other 61 dataset cases are explicitly unsupported. This is
not a full 68-case score or an exact paper replication. Unsupported workflows
and failed environment controls never disappear from a selected denominator.

Next, expand reviewed workflow coverage and validate each new failed/reference
pair before evaluating more model answers. Test whitespace preservation or a
formatter finishing step as a separately named repair variant, with the current
results retained as the unchanged baseline. Keep model, retrieval, and output
processing changes separate so any gain has an identifiable cause.
