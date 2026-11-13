# Qwen-only development decision and local formatter check

Run `ac114173bcf206333b0a` is complete. All 36 responses were verified
against their checksums and independently re-scored on the exact failed
snapshots. Both arms used the same Qwen checkpoint, six development records,
three seeds, and no retrieval.

| No-retrieval arm | Valid responses | Nonempty patches | Nonempty without detected Python syntax regression | Mean edit F1 | Exact reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original all-block markers | 15/18 | 12/18 | 12/18 | 0.268143 | 3/18 |
| Constrained raw edits | 16/18 | 8/18 | 7/18 | 0.126990 | 1/18 |

The constrained format failed the gate. It had slightly fewer length caps
(2 versus 3), but eight valid answers were only `DONE`, versus three valid
no-ops for the control. Two raw no-ops were on trials where the control
matched the reference exactly. Raw edits won three paired F1 comparisons,
lost seven, and tied eight. All exact matches came from one example, so
they are not evidence of broad repair coverage.

## Case 37: local replay of the affected-file formatter

For the SkyPilot formatting failure, the runner applied the saved answers in
memory and checked `sky/resources.py` with YAPF 0.32.0, TOML 0.10.2, and the
failed commit's `[tool.yapf]` style. The unmodified file reproduced the exact
diff hunk in the recorded failed job; the reference-fixed file produced no diff.

| Input | Passes affected-file YAPF check |
| --- | ---: |
| Original failed file | No |
| Reference-fixed file | Yes |
| Original-marker Qwen answers | 3/3 |
| Constrained raw Qwen answers | 1/3 |

All four passing model patches yielded the same file as the reference. The
two raw failures were unchanged files. The replay used macOS Python 3.9.6,
not the original Ubuntu Python 3.8 job. It ran the pinned formatter on one
file, not the recursive YAPF step or the entire workflow. No project or
generated code was executed, and no GitHub Actions ran. The local check is
stored as `case_37_yapf.json` in the development run directory.

**Decision:** Do not promote constrained raw edits to retrieval. The original
Qwen output format is the better control on these development records. A
new lean-retrieval experiment, if pursued, should hold that output format and
model fixed in both arms and use a different set of records. Expand local
fail/reference controls for other reproducible checks; do not extrapolate
this one-file check or edit F1 to the paper's CI Pass@1. The Mistral
replication track is unchanged.
