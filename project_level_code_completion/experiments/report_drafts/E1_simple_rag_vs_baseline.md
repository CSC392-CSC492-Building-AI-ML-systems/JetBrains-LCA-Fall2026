# E1: Simple RAG versus the benchmark baseline (draft report section)

## Question

Does a simple retrieval-augmented prompt beat the strongest static context composer from Long Code Arena, `path_distance`, on project-level line completion?

## Setup

- Model: CodeLlama-7B with the 16K window, as in the paper's line-completion tables. Our no-context and `path_distance` numbers reproduce the paper's Table 9 on all four dataset sizes, within 0.01 EM (for medium: 0.387 against 0.39 and 0.492 against 0.49).
- Data: the medium set, all 224 datapoints and all 8,676 target lines. Greedy decoding, at most 100 new tokens, seed 42.
- Metric: exact match (EM) as the benchmark defines it (mean of datapoint means), plus edit similarity (ES), overall and for the six line categories. Uncertainty is a paired bootstrap over datapoints (10,000 resamples, 95% interval).
- RAG "v1", the simple design: the repository is split into function, class and module chunks with tree-sitter; chunks are ranked with BM25 using the last 10 non-empty lines before the cursor as the query; up to 30 chunks are placed in the prompt ahead of the file prefix, within the 16K budget.

## Result

| approach | EM (all lines) | difference vs v1 [95% CI] |
|---|---:|---|
| no context | 0.387 | -0.143 [-0.173, -0.114] |
| `path_distance` | 0.492 | -0.038 [-0.058, -0.020] |
| RAG v1 | 0.530 | reference |

Simple RAG beats `path_distance` by +3.8 EM points [+2.0, +5.8] and beats no context by +14.3 [+11.4, +17.3].

By category (v1 minus `path_distance`): common +6.6, infile +5.5, inproject +4.8 and random +4.0 are significant; commited +1.8 and non_informative -0.3 are not.

## How we got here

A first pilot used StarCoder-1B on 20 datapoints and showed RAG slightly below `path_distance` (EM 0.252 against 0.266). That difference was not significant once uncertainty was computed over datapoints (-1.4 points [-3.8, +0.6]), and we retracted an early line-count argument that treated lines as independent. The pilot also exposed a flaw in our own design: RAG capped the file prefix at 4,000 tokens while the static arms did not, so the comparison was not fair. We removed the cap, and the final comparison uses the paper's model, window and all data.

## Limitations

- One model and one dataset size here; the RAG gain also holds on the large (+9.2) and huge (+8.8) sets with the tuned design, but not on small, where the repository fits the window.
- Training-data overlap is possible (see the E4 section) and could raise every arm, including the baselines.
