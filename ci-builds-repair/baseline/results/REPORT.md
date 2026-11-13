# Non-RAG Mistral baseline: completed, not paper-score comparable

Verified on 2026-11-05. Source run: `2d9349bb16247814246b`.
**204/204 saved evaluation cases verified; zero retrieved chunks in prompts.**
This audit re-scored the already generated baseline predictions. No fresh model
generation, project execution, GitHub push, or GitHub Actions evaluation occurred.

## Results

| Seed | Evaluated | Valid applied response | Exact reference repair | Edit-token F1 |
| --- | --- | --- | --- | --- |
| 0 | 68 | 39/68 | 0/68 | 0.192789 |
| 1 | 68 | 41/68 | 0/68 | 0.194282 |
| 2 | 68 | 43/68 | 0/68 | 0.198628 |

Across-seed mean edit-token F1: **0.195233**, population standard deviation
**0.002477**. Valid applied responses: **123/204**. Of the other 81 cases,
65 failed marker validation, 13 reached the output-token limit, and three were
over the input budget. Those cases remain in denominators as unchanged repairs.
There were 201 model calls and three budget skips.

Applied-response counts measure output-format validity, not successful builds.
Zero exact reference repairs does not imply zero working repairs: alternative
correct fixes need not reproduce the human patch. Neither F1 nor exact reference
agreement can be compared numerically to the paper's Pass@1.

## What matches the published baseline

- Mistral-7B-Instruct-v0.2 model family; a single answer per record/seed.
- Oracle edit locations from the reference diff, with only failing-side code
  shown to the model. Gold replacement code is reserved for scoring.
- All edit blocks in one request; instruction beginning
  `Fix CI in order for tests to pass. Relevant logs:`.
- Log excerpt centered on the first case-insensitive `error`, or the final
  seven lines when no error occurs.
- No BM25 or other retrieved repository evidence in the model prompt.

## Why paper comparability is NOT established

The attached paper, section 2.2 and appendix B.5/Table 7, reports **0.065
Mistral CI Pass@1 on a described 77-example dataset**. The local run differs:

| Dimension | Published description / metadata | Completed local run |
| --- | --- | --- |
| Evaluation | Actual workflow/build success, CI Pass@1 | Reference-edit F1/exact agreement and Python 3.13 AST checks; no tests or builds |
| Dataset | Paper: 77 examples, difficulty counts 35/14/28 | Current cached revision: 68 examples |
| Model weights | Mistral-7B-Instruct-v0.2; equivalence of a 4-bit conversion is not documented | `mlx-community/Mistral-7B-Instruct-v0.2-4bit` |
| Input context | Public result metadata says `context_size: 16000` | 8192 total configured tokens, reserving 2048 output + 256 safety |
| Markers | Paper: `file.py#L12` | Reconstructed format adds `#B{index}` and explicit return-format instructions |
| Logs | Paper refers to failed-step log lines | Existing helper also concatenates step-name strings, which can alter the excerpt |
| Decoding and seeds | CI-specific settings are not specified in the paper | Chosen temperature 0.2, output cap 2048, effective seed = configured seed + record ID |

The original prompt-assembly implementation remains the subject of an
[unanswered upstream issue](https://github.com/JetBrains-Research/lca-baselines/issues/12).
Thus even matching the visible prompt description would not demonstrate exact
implementation parity. The public
[Mistral result metadata](https://huggingface.co/datasets/JetBrains-Research/lca-results/blob/main/ci_builds_repair/results/mistral_7b.json)
confirms the model link and 16000 context size, but its `Pass/golden: 8/120`
field also differs from the paper's described denominator; it is not a verified
list of the paper's 77 examples.

Historical-data check: the public dataset revision
[`ad2efca1728e002c3aa2255655a496a4e89b32b1`](https://huggingface.co/datasets/JetBrains-Research/lca-ci-builds-repair/tree/ad2efca1728e002c3aa2255655a496a4e89b32b1/data/python)
(2024-06-19) contains **78 unique examples**, difficulty counts **35/14/29**.
The downloaded snapshot is cached locally under `.venv/paper-dataset`.
It cannot simply be relabeled as the paper's 77-item split, and no example was
arbitrarily removed to force the denominator. Historical IDs also differ from
the current split, so comparisons must join repository/commit identities, not
assume matching numeric IDs.

## Required for a defensible paper comparison

1. Resolve the original 77-example identities and use a pinned dataset revision.
2. Use the original checkpoint without an unvalidated quantization change;
   document the reconstructed prompt, context budget, and unspecified decoding
   choices explicitly.
3. Execute the relevant original checks in isolated local environments, since
   GitHub Actions remains disabled by request. For every evaluable record,
   first verify that the unmodified failed snapshot fails and the gold repair
   passes under the same environment. Only then score a model repair as a fix.
4. Report local execution coverage and environment differences separately.
   A validated local subset is not the full paper benchmark.

At verification time the Docker client could not connect to a running daemon.
No source code was run on the host as a substitute. Therefore there is currently
**no measured local or remote CI Pass@1** for these predictions.

## Audit artifacts

`metrics.json` embeds the original manifest and all 204 recomputed case metrics;
`per_example_metrics.csv` is the baseline-only compact table. The verifier checks
the original dataset checksum, source implementation hashes, prediction payload
checksums, identities, raw-response replacements, baseline-only prompts, exact
failed source revisions, and independently recomputed scores. It does not verify
the original inference execution by rerunning the model.
