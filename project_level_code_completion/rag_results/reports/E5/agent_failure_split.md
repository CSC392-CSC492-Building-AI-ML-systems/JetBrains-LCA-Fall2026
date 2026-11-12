# Failure split of the agent next to RAG F at N=1,753 (8676 lines)

Rebuild check for the agent arm (planner replayed from the logged actions, no GPU): the replayed final context equals the logged one on 8673 of 8676 lines; the rebuilt prompt has the logged token count on 8643 of 8676.

## 1. Classes (as in the E4 failure split)

| arm | lines | right | wrong, self-contained | wrong, retrieval miss | wrong, reasoning (names visible) |
|---|---:|---:|---:|---:|---:|
| RAG F | 8676 | 4523 (52.1%) | 2755 (31.8%) | 171 (2.0%) | 1227 (14.1%) |
| agent (v2) | 8676 | 4421 (51.0%) | 2797 (32.2%) | 194 (2.2%) | 1264 (14.6%) |

## 2. What the planner's edits did to the visibility of the names a line needs from other files

Lines that need at least one name from another file: 3303. Of them the planner changed the context on 1197.

- more needed names visible than under RAG F: **7** lines; fewer visible (an evicted chunk held a name): **35** lines; same number: 3261
- only lines whose context was changed can differ; the counts above are over all lines that need names.

| | lines | right under RAG F | right under agent | only agent right | only RAG F right |
|---|---:|---:|---:|---:|---:|
| names gained | 7 | 0 | 0 | 0 | 0 |
| names lost | 35 | 24 | 5 | 1 | 20 |

## 3. Retrieval misses of RAG F and what happened to them

- RAG F retrieval misses: 171; under the agent a name set that is now fully visible: 7; of those the line is right: 1; lines right under the agent in all: 1
- new retrieval misses under the agent (RAG F did not have one): 30
- net change in retrieval misses: +23

## 4. Changed lines by class pair (RAG F class -> agent class), lines whose context the planner changed

| RAG F | agent | lines |
|---|---|---:|
| right_self_contained | right_self_contained | 729 |
| wrong_self_contained | wrong_self_contained | 728 |
| right | right | 648 |
| reasoning | reasoning | 373 |
| right | reasoning | 63 |
| retrieval | retrieval | 54 |
| right_self_contained | wrong_self_contained | 53 |
| reasoning | right | 22 |
| right | retrieval | 20 |
| wrong_self_contained | right_self_contained | 11 |
| reasoning | retrieval | 10 |
| retrieval | reasoning | 6 |
| retrieval | right | 1 |
