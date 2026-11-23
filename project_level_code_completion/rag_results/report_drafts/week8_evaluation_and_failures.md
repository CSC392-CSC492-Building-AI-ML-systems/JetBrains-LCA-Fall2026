# Evaluation across approaches: accuracy, cost, latency and failures (week 8 section)

## Final experiment table

All rows: CodeLlama-7B writer, greedy decoding, at most 100 new tokens, exact match (EM) as the main metric, all 224 medium datapoints
(8,676 lines). "Tokens per line" is what the writer reads for the static arms and everything the planner and writer read and write for the agent (that row also counts the writer's output, about 18 tokens, so RAG F would be 4,723 on the same basis); seconds include building the context (the planner for the
agent); compute is GPU hours times $0.69 per hour, the rental price we used throughout. Intervals are per arm; the paired tests are in the E4 and E5 sections.

| approach | context budget N | EM (95% CI) | tokens/line | median s/line | p95 s/line | GPU hours | USD |
|---|---|---|---:|---:|---:|---:|---:|
| no context | - | 0.387 [0.365, 0.409] | 2,951 | 0.56 | 1.26 | 2.0 | 1.36 |
| path_distance (best static composer) | 256 | 0.395 [0.372, 0.418] | 3,207 | 0.57 | 1.28 | 2.0 | 1.36 |
| RAG F | 256 | 0.408 [0.385, 0.431] | 3,221 | 0.59 | 1.39 | 2.1 | 1.48 |
| path_distance | 1,753 | 0.415 [0.391, 0.439] | 4,695 | 0.69 | 1.41 | 2.3 | 1.60 |
| **RAG F** | **1,753** | **0.525 [0.492, 0.558]** | 4,705 | 0.72 | 1.53 | 2.5 | 1.70 |
| path_distance | 12,000 | 0.488 [0.458, 0.519] | 14,247 | 1.73 | 2.30 | 4.9 | 3.36 |
| RAG F | 12,000 | 0.537 [0.503, 0.570] | 13,996 | 1.76 | 2.37 | 4.9 | 3.39 |
| path_distance (full 16K window) | 15,900 | 0.492 [0.461, 0.524] | 15,900 | 1.91 | 2.39 | 5.3 | 3.67 |
| RAG F (full 16K window) | 15,900 | 0.537 [0.503, 0.571] | 15,088 | 1.92 | 2.46 | 5.3 | 3.63 |
| 7B planner agent | 1,753 | 0.513 [0.482, 0.544] | 32,201 | 11.34 | 22.16 | 30.2 | 20.85 |

The stronger (32B) agent was tested on a 24-datapoint development set only (1,372 lines, one machine); it is in the second table.

| arm (24 dev datapoints) | EM | vs RAG F, points (95% CI) | tokens/line | mean s/line | lines changed |
|---|---:|---:|---:|---:|---:|
| RAG F | 0.5037 | reference | 5,063 | 1.2 | none |
| 7B planner, E5 tools | 0.4865 | -1.72 [-3.90, -0.33] | 34,481 | 21.6 | 416 |
| 32B planner, E5 tools | 0.5036 | -0.01 [-0.58, +0.45] | 21,535 | 11.0 | 173 |
| 32B planner, v3, with gate | 0.5037 | +0.00 | 13,985 | 8.0 | 3 |
| 32B planner, v3, one tool call | 0.5037 | +0.00 | 10,672 | 3.8 | 3 |
| 32B planner, v3, no gate | 0.5037 | +0.00 | 13,627 | 5.2 | 44 |

Seconds in the second table were measured with another run sharing the GPU, so they are upper bounds, comparable only with each other.

**Reading the table.**
- **Best accuracy per cost: RAG F at 1,753 tokens.** It beats the best static composer at every budget and beats `path_distance` with 12,000 or the full 16K tokens of context
  (0.525 against 0.488 and 0.492) at about half the cost ($1.70 against $3.36 and $3.67).
- **More context stops helping:** RAG F gains 0.011 (paired, [0.007, 0.016]) from 1,753 to 12,000 tokens and nothing from 12,000 to 16K, while tokens triple and time per line rises from 0.72 to 1.9 s.
- **The agent costs far more and does not help:** 6.8 times the tokens and 16 times the time per line of RAG F for 1.2 points less accuracy; the stronger planner removes that loss but
  adds no gain (development set).

## Why there is little room: where the failures are

Of RAG F's lines at N = 1,753, 2.0 % (171 of 8,676) are retrieval misses, wrong lines that need a name from another file that is not in the prompt. On the development set
the share is 1.5 % (21 of 1,372 lines), and the agents touched few of them (7B: 11, 32B with the E5 tools: 8, stronger tools: 2 or 0).

We also read 25 random wrong lines for each of four arms, plus 25 lines the agent broke, and labeled what is wrong in the prediction (one reader, so rough: about +-10 points per share).

| group (25 lines each) | wrong name | different statement | near miss | equivalent alternative | free text | data / fragment |
|---|---:|---:|---:|---:|---:|---:|
| no context | 8% | 40% | 36% | 12% | 4% | 0% |
| path_distance, N = 1,753 | 0% | 60% | 20% | 8% | 8% | 4% |
| RAG F, N = 1,753 | 4% | 52% | 36% | 4% | 0% | 4% |
| 7B planner agent | 20% | 28% | 44% | 4% | 4% | 0% |
| lines the agent broke (RAG F was right) | 24% | 40% | 16% | 4% | 8% | 8% |

- **The typical failure is plausible code that is not what the author wrote next (about 45% of wrong lines) or a small deviation from it (about a third):** a different statement, a missing wrapper such as
  `return`, a different literal or default, a wrong word in a message. Better retrieval does not change this much: the mix looks alike for every arm.
- **Wrong names, the failure retrieval can fix, are about 1 line in 10.** The agent's sample has more of them (5 of 25 against 1 for RAG F), a hint (not a result at this sample size) that the added code does not supply missing names.
- **Where the agent breaks a line RAG F got right**, it mostly swaps in a different statement (10 of 25) or a wrong name (6 of 25), rarely a near miss (4 of 25).
- **About 13% of the wrong lines are not errors a developer would fix under exact match** (equivalent alternatives, comment wording, fragments); near misses would earn partial credit under edit similarity.

Limits: one labeler and no second rater; 25 lines per group; a random sample of all wrong lines. The retrieval-miss counts come from a lexical screen (is a needed name in the prompt?), not from checking definitions.

Files: `rag_results/reports/final/final_experiment_table.md` and `.csv`, `failure_taxonomy_week8.md`, `failure_samples_week8.txt` (every labeled line with the truth and all predictions),
`rag_results/reports/E5v3/dev_headroom.md`; scripts `make_final_table.py`, `sample_failures_week8.py`, `tabulate_failure_labels.py`, `check_headroom_dev.py`.
