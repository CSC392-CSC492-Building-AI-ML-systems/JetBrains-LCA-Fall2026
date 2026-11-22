# Repair pipeline improvement

The improved output-composition policy increases actual local workflow
success on the existing validation cases. It uses the **same saved model
answers**, original failed snapshots, workflows, and seeds. Original results
are retained; no model answer, reference repair, CI command, or success
condition was changed to produce these comparisons.

## Paired results

Every entry counts single predictions, with seeds 0, 1, and 2. These are
deliberately selected development cases, not estimates for the full dataset
or direct comparisons with the paper's 77-case evaluation.

| Saved model arm | Case IDs | Original | Boundary whitespace | Markers + boundaries |
| --- | --- | --- | --- | --- |
| Mistral, no retrieval | 33, 34, 36, 37, 144 | 0/15 | 0/15 | **3/15** |
| Mistral, original BM25 | 33, 34, 36, 37, 144 | 3/15 | 3/15 | 3/15 |
| Qwen, original output format | 37 | 3/3 | 3/3 | 3/3 |
| Qwen, constrained raw format | 37 | 1/3 | 1/3 | 1/3 |
| Qwen, no retrieval | 144 | 0/3 | **1/3** | **1/3** |
| Qwen, lean BM25 | 144 | 0/3 | **3/3** | **3/3** |

Seven previously failing predictions pass and none regress. This pooled count
includes different arms and case subsets and must not be presented as one
model's benchmark accuracy. Identical repairs share recorded execution
evidence; repeated seeds are not independent examples.

## Why the standard Mistral baseline improves

All three new no-retrieval successes are on **SkyPilot case 36**. Raw answers
contain a complete replacement for the correct file and block ID, but use
different line numbers and an `E0` end-marker alias. The original parser rejects
them. The new parser maps the unambiguous file/block-ID pair to the original
oracle location; generated line numbers never determine the edit location.
The resulting source passes repository-wide YAPF, Black, and both isort checks.

All blocks must be present exactly once. Unknown paths or IDs, nested/repeated
envelopes, incomplete answers, and prose outside envelopes are rejected.
Length-capped and input-skipped answers remain failures. There is no candidate
search, reference-code matching, or CI-guided selection among repairs.

## Why Qwen improves

The policy preserves original blank lines at replacement boundaries. Python
transformations require the complete file's AST to remain identical to the
model's proposed patch. This prevents accidental changes to blank lines inside
multiline strings. It does not repair code or indentation, or modify non-Python
boundary whitespace.

This resolves the missing blank line in case 144. All three lean-retrieval
answers and no-retrieval seed 2 become **byte-identical to the validated
reference-patched tree**, which passed the entire five-version Starlette
workflow, including builds, tests, and 100% coverage. No-retrieval seeds 0 and 1
retain their incorrect `send.call_count` assertion and still fail. Formatting
success does not hide those failures.

## Scope checks and limits

The parser policy was checked against all **408 cached Mistral answers**.
For the 204 no-retrieval answers, accepted responses increase from 123 to 126;
for the 204 BM25 answers, from 68 to 72. These are response-validity counts,
not full-dataset CI scores. The policy accepts only old source and model output;
reference replacement text is excluded from its API.

An additional case, **Starlette 142**, was selected before generating its first
Qwen answers. Three fresh no-retrieval answers used the same checkpoint,
temperature 0.2, and effective seeds `seed + dataset ID`. Failed/reference
controls validate across Python 3.8–3.12. Both original and improved composition
score **0/3**: the model adds unused imports and misses the typing correction.
This negative check shows that composition does not solve missing reasoning.

A separate no-retrieval Mistral diagnostic on case 33 expanded the context
to 32,768 tokens and the output limit to 16,384, processing its 11,510-token
prompt instead of skipping it. Seeds 0 and 2 stopped after 6,066 and 9,221
output tokens but omitted required blocks. Seed 1 hit the 16,384-token limit.
All three remain invalid under both parsers, so this diagnostic scores **0/3**.
It is not substituted into the original paired comparison. Increasing the
budget alone does not resolve Mistral's output-protocol failures.

The policy was developed on inspected examples. Six cases now have validated
workflow controls; most of the 68-case cache still needs reviewed workflow
support. Debian containers, four-bit checkpoints, reconstructed prompting, and
missing original 77-case membership prevent exact paper reproduction. A 20%
development-subset result does not establish agreement with, or superiority
to, the paper's Mistral score.

The current [upstream dataset](https://huggingface.co/datasets/JetBrains-Research/lca-ci-builds-repair)
also lists 68 rows, and the [upstream prompt-assembly request](https://github.com/JetBrains-Research/lca-baselines/issues/12)
remains open. Exact reproduction still requires resolving those differences;
adjusting a development-set percentage cannot establish it.

## Verification and reproduction

- [Paired metrics, manifests, source checksums, and additional-case outcomes](repair_pipeline_metrics.json)
- [All cached Mistral parsing/composition checks](repair_preflight_metrics.json)
- [Original strict-protocol workflow results](LOCAL_CI_VALIDATION.md)
- [Setup, improved-pipeline commands, and audit instructions](../local_ci/README.md)

The boundary-only comparison is run `9d2833dd754e877a0e06`; the combined policy
is `f1041aaa99fc5a78c929`. They required 20 and 1 additional job executions
respectively. Reused verdicts require identical patched source trees and
audited evidence. Raw outputs and exact source snapshots remain in ignored
`out/` directories.

All **60 baseline tests and 5 retrieval tests pass**. Independent execution and
generation audits pass. A negative audit rejects substituted prediction
identities. The original policy remains available with `--composition original`;
improved runs are explicitly labeled with their policy.
