# Final experiment table (week 8): accuracy, tokens, latency and compute across all approaches

## A. All 224 medium datapoints (8,676 lines), CodeLlama-7B writer, greedy, 100 new tokens max

| approach | N | EM (95% CI) | tokens/line (planner + writer) | p50 s/line | p95 s/line | GPU hours | USD at $0.69/h |
|---|---|---|---:|---:|---:|---:|---:|
| no context | - | 0.387 [0.365, 0.409] | 2,951 | 0.56 | 1.26 | 2.0 | 1.36 |
| path_distance | 256 | 0.395 [0.372, 0.418] | 3,207 | 0.57 | 1.28 | 2.0 | 1.36 |
| RAG F | 256 | 0.408 [0.385, 0.431] | 3,221 | 0.59 | 1.39 | 2.1 | 1.48 |
| path_distance | 1,753 | 0.415 [0.391, 0.439] | 4,695 | 0.69 | 1.41 | 2.3 | 1.60 |
| RAG F | 1,753 | 0.525 [0.492, 0.558] | 4,705 | 0.72 | 1.53 | 2.5 | 1.70 |
| path_distance | 12,000 | 0.488 [0.458, 0.519] | 14,247 | 1.73 | 2.3 | 4.9 | 3.36 |
| RAG F | 12,000 | 0.537 [0.503, 0.570] | 13,996 | 1.76 | 2.37 | 4.9 | 3.39 |
| path_distance | 16K window | 0.492 [0.461, 0.524] | 15,900 | 1.91 | 2.39 | 5.3 | 3.67 |
| RAG F | 16K window | 0.537 [0.503, 0.571] | 15,088 | 1.92 | 2.46 | 5.3 | 3.63 |
| 7B planner agent (E5, add-only) | 1,753 | 0.513 [0.482, 0.544] | 32,201 | 11.34 | 22.16 | 30.2 | 20.85 |

Tokens: the writer's prompt for the static arms (their only model); planner + writer for the agent arm, which also counts the writer's output (about 18 tokens per line; on that basis RAG F at N = 1,753 is 4,723, the figure used in E5_report_section.md). Seconds: context building (planner included) plus generation. GPU hours: wall time of the run for the E4 arms (includes model loading); summed per-line work for the agent arm (the real bill was about $22 because shards ran on several machines and two were resumed). EM 95% intervals are per arm and ignore pairing; the paired tests are in the E4 and E5 reports.

## B. Stronger agent, 24 development datapoints (1372 lines), one RTX PRO 6000, N = 1,753

| arm | EM (mean over datapoints) | vs RAG F, points (95% CI) | tokens/line (planner + writer) | p50 s/line | mean s/line | lines changed |
|---|---:|---:|---:|---:|---:|---:|
| RAG F (same card) | 0.5037 | reference | 5,063 | 1.1 | 1.2 | 0 |
| 7B planner, E5 tools | 0.4865 | -1.72 [-3.90, -0.33] | 34,481 | 19.8 | 21.6 | 416 |
| 32B planner, E5 tools | 0.5036 | -0.01 [-0.58, +0.45] | 21,535 | 7.6 | 11.0 | 173 |
| 32B planner, v3, with gate | 0.5037 | +0.00 [+0.00, +0.00] | 13,985 | 4.8 | 8.0 | 3 |
| 32B planner, v3, one tool call | 0.5037 | +0.00 [+0.00, +0.00] | 10,672 | 3.1 | 3.8 | 3 |
| 32B planner, v3, no gate | 0.5037 | +0.00 [+0.00, +0.00] | 13,627 | 3.1 | 5.2 | 44 |

Seconds in part B are with another arm sharing the GPU for most of the run, so read them as an upper bound and compare them only with each other. Part B tokens include the writer's output (about 16 tokens per line), as the E5 total does; E5_report_section.md quotes planner plus writer prompt only, so its numbers are lower by that amount (under 0.1 %). The 7B planner with the v3 tools was stopped after 174 lines (log 8.101) and has no row.

