# E2: Chunking strategy (draft report section)

## Question

Does the way the repository is cut into retrievable chunks change completion accuracy?

## Setup

One variable changes at a time against the E1 base, "v1" (tree-sitter function, class and module chunks; BM25; 10-line query; 30 chunks). Everything else is fixed: CodeLlama-7B, 16K window, medium set, all 224 datapoints and all 8,676 lines, seed 42. We state predictions before running and read results with a paired bootstrap over datapoints (10,000 resamples, 95% interval); an arm "changes performance" only if its interval excludes zero.

Arms:
- **fine10, fine20, fine40:** fixed windows of 10, 20 or 40 lines with a stride of half the window, instead of structure-aware chunks.
- **whole files:** each file is one chunk.
- **F:** v1 plus 20 continuation lines after each retrieved chunk (the design used in the later experiments).

## Result (EM over all lines, difference vs v1)

| arm | difference vs v1 [95% CI] |
|---|---|
| fine10 (EM 0.524) | -0.006 [-0.012, +0.001] |
| fine20 (EM 0.535) | +0.006 [-0.000, +0.011] |
| fine40 | +0.009 [+0.003, +0.015] |
| whole files | +0.002 [-0.010, +0.012] |
| F, continuation lines (EM 0.537) | +0.007 [+0.003, +0.013] |

v1 itself scores 0.530.

- Chunk shape matters little: every variant is within about one EM point of v1. Only fine40 and F are statistically distinguishable from v1, and both gains are under one point.
- Where fixed windows differ from structure chunks, it is in the infile and inproject categories (fine10 is lower there, -1.8 and -1.3).
- The retrieval itself carries the gain; the way it is chunked is a second-order choice. Later experiments (including E4) use the structure-aware chunker with continuation lines (F).

## Predictions

Of 13 pre-registered predictions across E2 and E3, 11 were met. Two were missed and both were over-confident priors taken from earlier 1B-model development runs: the gap between `path_distance` and no context (+10.5 observed against [+0.5, +4] predicted), and the whole-file arm, which we expected to lose 1–8 points and which tied v1.

## Limitations

- One model and one dataset size.
- F was tuned on development data from the smaller model; the later E4 comparison reuses it, so its held-out gain on the 1B model (+0.8) is the unbiased figure; the all-data +0.7 here on CodeLlama-7B is consistent with it.
- Training-data overlap is possible (see the E4 section).
