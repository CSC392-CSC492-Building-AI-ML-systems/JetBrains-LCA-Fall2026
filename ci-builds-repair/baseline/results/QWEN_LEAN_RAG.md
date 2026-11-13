# Qwen lean-retrieval comparison

Run `bf64963b600ed42de523` completed. The offline audit verified 36 saved
predictions and their prompts, checked retrieved snippets against cached
failed revisions, and re-scored all 18 pairs. Both arms used Qwen and the
original output markers, with three seeds on six cases outside the earlier
Qwen development cohort. All six cases received three snippets; the median
added context was 569 tokens.

| Arm | Valid responses | Nonempty patches without detected Python syntax regression | Mean edit F1 | Exact reference text |
| --- | ---: | ---: | ---: | ---: |
| No retrieval | 15/18 | 13/18 | 0.250119 | 0/18 |
| Lean retrieval | 14/18 | 14/18 | 0.249960 | 0/18 |

Retrieval won seven pairs, tied nine, and lost two. The paired mean change
was **−0.000159**: more wins did not produce a larger mean because the losses
were larger. These are six independent cases, not 18 independent projects.

| Case | Project | Mean F1, no retrieval | Mean F1, lean retrieval |
| --- | --- | ---: | ---: |
| 60 | HACS | 0.0000 | 0.0490 |
| 119 | W&B | 0.3768 | 0.3333 |
| 144 | Starlette | 0.9278 | 1.0000 |
| 41 | Beets | 0.0000 | 0.0000 |
| 57 | HACS | 0.0593 | 0.0680 |
| 39 | Beets | 0.1368 | 0.0494 |

Leaving out Starlette shifts the mean delta to −0.0146; leaving out Beets
case 39 shifts it to +0.0173. The direction is sensitive to individual cases.

## Local checks, not CI Pass@1

- **W&B case 119:** The pinned Black 23.7.0 check on the affected file
  reproduced the recorded change, and the reference fix passed. Neither arm
  passed any seed (0/3 each). Offline edit credit did not imply a formatter fix.
- **Starlette case 144:** Evaluating only the failing argument literal
  reproduced the logged `TypeError: unhashable type: 'dict'`. The reference
  literal and all six model literals produced the correct mapping. All three
  RAG patches matched the reference file's Python AST; only one plain patch
  did. The other two plain patches added a `send.call_count` assertion against
  a local async function. The literal check did not execute that assertion,
  Starlette, or pytest. AST equality ignores formatting; it is not a full
  test result.

## Decision

Do not enable retrieval by default on the strength of this pilot. Keep
Qwen without retrieval as the control. Before expanding generation, add
local fail/reference controls for more reproducible checks, and include the
original formatter itself as a deterministic baseline on formatter cases.
If testing selective retrieval, declare its routing rule in advance and use
different cases rather than selecting the winning cases from this run.
The paper-style Mistral results are unchanged. No workflows or GitHub
Actions were run.
