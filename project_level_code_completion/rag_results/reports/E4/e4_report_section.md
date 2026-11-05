# E4: Retrieval versus larger direct context (draft report section)

## Question

Does retrieving the relevant parts of a repository beat simply giving the model more of it? We compare our RAG design (tree-sitter chunking, BM25 retrieval, token-budgeted prompts) with the strongest static composer from Long Code Arena, `path_distance`, on CodeLlama-7B at the paper's context budgets: 256, 1,753 and 12,000 tokens, plus the full 16K window (15,900 token budget). A no-context arm gives the floor.

## Setup

- Data: the medium project-level completion set, all 224 datapoints and all 8,676 target lines. Nothing was subsampled.
- Protocol: as in the paper. Greedy decoding, at most 100 new tokens, seed 42, exact match (EM) reported overall and by line category.
- Budget: for `path_distance`, the last N tokens of the composed repository context; for RAG, at most N tokens of retrieved chunks. The prefix of the file being completed is kept in both.
- Statistics: paired bootstrap over datapoints (10,000 resamples, 95% interval, seed 0).
- Logged per line: input and output tokens, context-building time, generation time, truncation. Cost is GPU wall time at $0.69/hour on one RTX 5090.

## Results

| approach | N | EM (95% CI) | tokens/line | p50 s/line | cost |
|---|---|---|---|---|---|
| no context | - | 0.387 [0.365, 0.409] | 2,951 | 0.56 | $1.36 |
| path_distance | 256 | 0.395 [0.372, 0.418] | 3,207 | 0.57 | $1.36 |
| RAG F | 256 | 0.408 [0.385, 0.431] | 3,221 | 0.59 | $1.48 |
| path_distance | 1,753 | 0.415 [0.391, 0.439] | 4,695 | 0.69 | $1.60 |
| RAG F | 1,753 | 0.525 [0.492, 0.558] | 4,705 | 0.72 | $1.70 |
| path_distance | 12,000 | 0.488 [0.458, 0.519] | 14,247 | 1.73 | $3.36 |
| RAG F | 12,000 | 0.537 [0.503, 0.570] | 13,996 | 1.76 | $3.39 |
| path_distance | 16K | 0.492 [0.461, 0.524] | 15,900 | 1.91 | $3.67 |
| RAG F | 16K | 0.537 [0.503, 0.571] | 15,088 | 1.92 | $3.63 |

(Figures: `fig_em_vs_context`, `fig_cost_vs_em`, `fig_latency`, `fig_failure_split` in `rag_results/figures/E4/`.)

1. **RAG beats `path_distance` at every budget.** Paired differences in EM are +0.013 (256), +0.110 (1,753), +0.049 (12,000) and +0.045 (16K); all four intervals exclude zero. The per-arm intervals in the table overlap in places because they ignore pairing; the paired tests are the ones to quote.
2. **More context stops helping.** `path_distance` reaches about 0.49 and RAG about 0.537. Going from 12,000 tokens to the full 16K window changes RAG by 0.000 (interval -0.002 to +0.003).
3. **RAG is cheaper for the same accuracy.** RAG at 1,753 tokens (EM 0.525, $1.70) is more accurate than `path_distance` at 12,000 tokens (0.488, $3.36) and at 16K (0.492, $3.67), at about half the cost.
4. **Latency tracks prompt length, not retrieval.** Median time per line rises from 0.6 s at about 3K tokens to 1.9 s at 16K. Retrieval adds only 0.02-0.03 s at the median.

## Why: where the failures are

We classified each line by whether the identifiers it needs from other repository files were visible in the prompt (a lexical screen, not proof).

- About 62% of lines need nothing from other files, so no retrieval method can help them.
- Retrieval misses are rare: 0.5-2.0% of lines for RAG against 2.0-5.9% for `path_distance`.
- Most remaining errors are plausible code that is not what the file does next. A hand-read sample of RAG failures at 12,000 tokens gave: wrong statement 54%, near miss 21%, wrong name 9%, free text 8%, equivalent code 7%, data lines 1%.
- Retrieval confidence is informative: the median top BM25 score is 51.7 on lines the model got right and 31.5 on lines it got wrong.
- At 12,000 tokens `path_distance` truncates the context on 24.6% of lines, but this does not explain its gap to RAG.

## Limitations

- One model (CodeLlama-7B) and one dataset size (medium). Other models and sizes are not tested here.
- Training-data overlap is possible. The benchmark repositories are public, and 87% of the benchmark commits (152 of the 175 whose dates we could fetch) are dated before February 2023, earlier than the stated data cutoffs of DeepSeek-Coder and StarCoder. CodeLlama's and DeepSeek-Coder's training data are not released, so we cannot audit them directly. Overlap could raise every arm, including the no-context floor, so it affects absolute numbers more than the comparison between arms.
- The failure classes use a lexical screen and a hand-read sample, not a full manual label of every failure.
