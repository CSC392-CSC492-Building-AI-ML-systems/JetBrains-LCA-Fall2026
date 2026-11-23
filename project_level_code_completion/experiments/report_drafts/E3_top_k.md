# E3: Number of retrieved chunks, top-k (draft report section)

## Question

How many retrieved chunks does the prompt need? Too few may miss the relevant code; too many may crowd the file being completed.

## Setup

Same base and protocol as E2: the E1 design "v1" (30 chunks), CodeLlama-7B, 16K window, medium set, all 224 datapoints and 8,676 lines, seed 42. Only the chunk cap `max_chunks` changes: 1, 3, 5, 10, 30 (the base) and 60. Paired bootstrap over datapoints (10,000 resamples, 95% interval).

## Result (EM over all lines)

| chunks (k) | EM | difference vs k=30 [95% CI] |
|---:|---:|---|
| 1 | 0.489 | -0.041 [-0.052, -0.030] |
| 3 | 0.510 | -0.020 [-0.027, -0.013] |
| 5 | 0.520 | -0.010 [-0.016, -0.004] |
| 10 | 0.524 | -0.006 [-0.010, -0.001] |
| 30 (base) | 0.530 | reference |
| 60 | 0.530 | +0.000 [-0.003, +0.004] |

For reference, `path_distance` scores 0.492 and no context 0.387.

- Accuracy rises steeply from 1 to 5 chunks, flattens by 10 and saturates by 30. Doubling to 60 changes nothing in any category.
- A single retrieved chunk (EM 0.489) already roughly matches the static baseline (0.492); three chunks (0.510) beat it.
- Beyond about 30 chunks, the 16K token budget, not the cap, limits what fits, so more chunks cannot add information.

## Predictions

All the k-sweep predictions (k=1, 3, 5, 10, 60) were met; the missed predictions for the experiment pair are described in the E2 section.

## Limitations

- One model and one dataset size; the best k may differ with a smaller window.
- Chunks are ranked by BM25 only; a better ranker could make small k more competitive.
- Training-data overlap is possible (see the E4 section).
