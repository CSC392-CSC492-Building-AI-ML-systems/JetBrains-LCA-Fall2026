| approach | EM (95% CI) | ES | wrong lines (of 8,676) | writer input tokens/line | total tokens/line | p50 s/line | p95 s/line | truncated | line work (GPU h) |
|---|---|---|---|---|---|---|---|---|---|
| no context | 0.387 [0.365, 0.409] | 74.2 | 5,432 | 2,951 | 2,968 | 0.56 | 1.26 | 0.4% | 1.52 |
| path_distance | 0.415 [0.391, 0.439] | 75.7 | 5,232 | 4,695 | 4,712 | 0.69 | 1.41 | 0.8% | 1.84 |
| RAG F | 0.525 [0.492, 0.558] | 80.3 | 4,153 | 4,705 | 4,723 | 0.72 | 1.53 | 0.4% | 1.94 |
| agent (v2) | 0.513 [0.482, 0.544] | 80.0 | 4,255 | 4,609 | 32,201 | 11.34 | 22.16 | 0.4% | 30.22 |

N = 1,753 tokens, CodeLlama-7B, 224 medium datapoints. Total tokens = planner + writer input + writer output (non-agent arms: writer only). Line work = sum of per-line context + generation seconds, the same definition for every arm (E4's table prices wall time of a run at $0.69/h, which also counts model loading; the money actually spent on the agent arm is on the rental bill, see log 8.88).
