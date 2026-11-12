# Sparse plain-text edits: development decision

Run `410fbcb28c8172538bfa` completed on six development examples (98, 45,
143, 85, 37, 126), seeds 0–2, with no retrieval. The 18 saved controls and
18 fresh answers used the same 4-bit Mistral checkpoint, logs, source blocks,
generation settings, and offline evaluator. The audit verified all 36
prediction checksums, matched the saved controls and prompts, and recomputed
each score from the exact failing revision.

| No-retrieval format | Nonempty patches | Nonempty without detected Python syntax regression | Mean edit F1 | Output-cap failures |
| --- | ---: | ---: | ---: | ---: |
| Original markers | 6/18 | 0/18 | 0.098621 | 9/18 |
| Sparse plain-text markers | 0/18 | 0/18 | 0.000000 | 3/18 |

The sparse format shortened responses (median 431.5 versus 1,349 output
tokens) but did not pass the development gate: no gain in nonempty patches
without detected syntax regressions, lower overall F1, and no applicable
multi-block repair. Fourteen answers had text outside the accepted block
format, one used an unknown marker, and three hit the output cap.

## Parser-only diagnostic

Re-read the same responses, removing only non-marker lines outside blocks.
Keep the entire body inside each block and require exact known labels,
balanced nonduplicate pairs, and a nontruncated answer. This post-hoc check
accepted **one** answer (example 85, seed 2), but that patch introduced a
Python syntax regression. Overall edit F1 under this diagnostic was 0.030146,
still below the control. Eight uncapped answers contained no full marker line;
the other uncapped rejections contained unknown or duplicate/nested markers.
Removing surrounding prose is therefore insufficient for this cohort. These
diagnostic numbers do not replace the fixed run scores.

## Decision

Do not advance this format to the retrieval comparison. First verify a
shorter, mechanically enforceable edit protocol on small non-benchmark
examples, then repeat the no-retrieval development gate. Freeze the protocol
before measuring retrieval on different examples. Neither edit F1 nor
Python parsing measures build success; no project tests, builds, or GitHub
Actions were run.
