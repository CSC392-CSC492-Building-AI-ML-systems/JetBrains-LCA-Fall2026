# E5 full arm: agent (v2) against E4 arms at N=1,753 (8676 lines, 224 datapoints)

## 1. Headline (EM = mean over datapoints, as in the E4 table; ES likewise; wrong lines = lines not exactly right)

| arm | EM | ES | EM pooled over lines | wrong lines | writer input tokens/line | p50 s/line | p95 s/line | truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero-context | 0.3870 | 74.18 | 0.3739 | 5432 | 2951 | 0.56 | 1.26 | 0.4% |
| path_distance | 0.4147 | 75.68 | 0.3970 | 5232 | 4695 | 0.69 | 1.41 | 0.8% |
| RAG F | 0.5252 | 80.35 | 0.5213 | 4153 | 4705 | 0.72 | 1.53 | 0.4% |
| agent (v2) | 0.5128 | 79.98 | 0.5096 | 4255 | 4609 | 11.34 | 22.16 | 0.4% |

- EM agent (v2) minus RAG F: -1.24 pts [-1.81, -0.72] * over 224 datapoints
- EM path_distance minus RAG F: -11.04 pts [-13.72, -8.46] * over 224 datapoints
- EM zero-context minus RAG F: -13.82 pts [-16.86, -10.97] * over 224 datapoints
- EM agent minus path_distance: +9.81 pts [+7.43, +12.29] * over 224 datapoints
- ES agent minus RAG F: -0.37 ES points [-0.63, -0.13] * over 224 datapoints
- EM pooled over lines, agent minus RAG F: -1.18 pts
- datapoints where the agent is better / worse / tied: 16 / 48 / 160

## 2. What the planner did

- changed the context on 2718 of 8676 lines (31.3%); fallback to RAG's prompt on 19 (0.22%)
- planner status: submitted 8657, invalid_replies 19
- steps per line 4.48, tool calls per line 3.48
- total tokens per line (planner + writer): agent 32201 vs RAG F 4723 = 6.8x
- seconds per line (context + generation, planner time is inside context): agent mean 12.54, RAG F mean 0.80

## 3. Lines the planner changed (paired counts; unchanged lines carry the same prompt as RAG F)

- changed lines: 2718; both right 1377, only agent right 34, only RAG F right 136, neither 1171
- EM on changed lines: agent 0.5191, RAG F 0.5567; net change -102 lines (-1.18 pts of all lines)
- of the changed lines, 2259 still got the same prediction as RAG F
- unchanged lines with a different prediction than RAG F: 1 of 5958
  - datapoint 75 line 213 (non_informative): agent `print(f"Unexpected opcode: {opcode}", file=sys.stderr)` | RAG F `` | truth `print(f"Unexpected websocket opcode: {opcode}", file=sys.stderr)`; writer input tokens 4945 vs 4945

## 4. Robustness (fixed in 8.87)

- without fallback lines (dropped from both arms, 19 lines): EM agent minus RAG F: -1.24 pts [-1.81, -0.72] * over 224 datapoints
- over the 216 datapoints not used for design (the 8 of 8.84/8.86 left out; 8134 lines): EM agent minus RAG F: -1.23 pts [-1.82, -0.70] * over 216 datapoints

## 5. By line class (EM, mean over the datapoints that have the class; agent minus RAG F, 95% CI)

| class | datapoints | lines | RAG F | agent | difference |
|---|---:|---:|---:|---:|---|
| commited | 89 | 1495 | 0.465 | 0.463 | -0.19 [-0.76, +0.34] |
| common | 127 | 779 | 0.493 | 0.489 | -0.46 [-0.99, -0.05] * |
| infile | 183 | 2224 | 0.473 | 0.466 | -0.72 [-1.29, -0.18] * |
| inproject | 150 | 2236 | 0.521 | 0.500 | -2.16 [-4.23, -0.56] * |
| non_informative | 224 | 858 | 0.455 | 0.440 | -1.53 [-2.81, -0.41] * |
| random | 224 | 1084 | 0.617 | 0.607 | -1.01 [-1.76, -0.30] * |

`*` = the 95% interval excludes 0 (resampling datapoints, 10,000 draws, seed 0, same resample for both arms).
