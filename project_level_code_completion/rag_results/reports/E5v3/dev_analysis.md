# Stronger agent, development arms (1372 lines, 24 datapoints of medium; CodeLlama-7B writer, N=1,753, one machine)

Primary metric: EM = mean over datapoints (as in the E4/E5 tables); pooled-over-lines EM is shown next to it because two datapoints hold 42 % of the lines. Intervals: paired bootstrap over datapoints (10,000 draws, seed 0, same resample for both arms).

## 1. Headline

| arm | EM (datapoint mean) | ES | EM pooled | vs D0 (datapoint mean) | changed lines | fallback | writer input tokens/line | s/line (context+generation) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 RAG F | 0.5037 | 80.73 | 0.4286 | reference | 0 | 2 | 5047 | 1.2 |
| D1 7B + old tools | 0.4865 | 80.27 | 0.4206 | -1.72 [-3.90, -0.33] * | 416 | 7 | 4972 | 21.6 |
| D2 32B + old tools | 0.5036 | 80.75 | 0.4293 | -0.01 [-0.58, +0.45] | 173 | 0 | 5012 | 11.0 |
| D3 32B + v3 | 0.5037 | 80.73 | 0.4286 | +0.00 [+0.00, +0.00] | 3 | 1 | 5046 | 8.0 |
| D5 32B + v3, 1 tool call | 0.5037 | 80.74 | 0.4286 | +0.00 [+0.00, +0.00] | 3 | 0 | 5046 | 3.8 |
| D6 32B + v3, no gate | 0.5037 | 80.80 | 0.4286 | +0.00 [+0.00, +0.00] | 44 | 1 | 5037 | 5.2 |

`*` = the 95 % interval excludes 0. Seconds per line of D2 to D5 were measured with two arms sharing the card.

## 2. Changed lines: who gets them right (against D0)

| arm | changed | both right | both wrong | only D0 right (broke) | only the arm right (fixed) | broke : fixed |
|---|---:|---:|---:|---:|---:|---|
| D1 7B + old tools | 416 | 166 | 237 | 12 | 1 | 12:1 |
| D2 32B + old tools | 173 | 72 | 98 | 1 | 2 | 1:2 |
| D3 32B + v3 | 3 | 0 | 3 | 0 | 0 | 0:0 |
| D5 32B + v3, 1 tool call | 3 | 0 | 3 | 0 | 0 | 0:0 |
| D6 32B + v3, no gate | 44 | 15 | 29 | 0 | 0 | 0:0 |

## 3. Lines with an unchanged context: does the writer reproduce D0 (same machine, same prompt)?

| arm | unchanged lines | prediction differs from D0 | of which EM differs |
|---|---:|---:|---:|
| D1 7B + old tools | 956 | 0 | 0 |
| D2 32B + old tools | 1199 | 0 | 0 |
| D3 32B + v3 | 1369 | 0 | 0 |
| D5 32B + v3, 1 tool call | 1369 | 0 | 0 |
| D6 32B + v3, no gate | 1328 | 0 | 0 |

## 4. What the planners did

| arm | lines | planner calls/line | tool calls/line | submitted at once | invalid-reply lines (fallback) | gate none / passed / rejected | planner tokens/line (in+out) | tokens of the writer prompt/line |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| D1 7B + old tools | 1372 | 4.94 | 3.93 | 3 | 7 (7) | 0 / 0 / 0 (off 1372) | 29493 | 4972 |
| D2 32B + old tools | 1372 | 3.64 | 2.64 | 189 | 0 (0) | 0 / 0 / 0 (off 1372) | 16507 | 5012 |
| D3 32B + v3 | 1372 | 1.86 | 0.78 | 893 | 1 (1) | 1362 / 3 / 7 (off 0) | 8923 | 5046 |
| D5 32B + v3, 1 tool call | 1372 | 1.35 | 0.35 | 893 | 0 (0) | 1287 / 4 / 81 (off 0) | 5610 | 5046 |
| D6 32B + v3, no gate | 1372 | 1.78 | 0.78 | 893 | 1 (1) | 0 / 0 / 0 (off 1372) | 8573 | 5037 |

## 5. Pre-registered decision rule (8.93) for D3 32B + v3

- point estimate +0.00 points (needs >= +0.50): **NOT met**
- lower bound +0.00 (needs >= -1.00): **met**
- broke : fixed among changed lines = 0 : 0 (needs broke <= 2 x fixed): **met** (nothing changed enough to judge)
- **NO-GO: do not run the full arm; report the dev table**

