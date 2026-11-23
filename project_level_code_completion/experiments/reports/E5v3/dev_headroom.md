# Room for a context editor on the 24 dev datapoints (1372 lines)

Classes of RAG F's lines at N = 1,753 from the E4 lexical failure split, matched to the dev lines (`check_headroom_dev.py`).

| class of the RAG F line | lines | share of the 1,372 |
|---|---:|---:|
| wrong_self_contained | 613 | 44.7% |
| right_self_contained | 449 | 32.7% |
| reasoning | 154 | 11.2% |
| right | 135 | 9.8% |
| retrieval | 21 | 1.5% |

Wrong lines of RAG F on the dev set: 788 (57.4%); **retrieval misses (a needed name is not in the prompt): 21 (1.5%)**. A context editor can only add a missing name on those lines; on all 8,676 lines the share is 2.0 % (171 lines).

## Lines whose context each arm changed, by the class of the RAG F line

| arm | changed lines | RAG F right | RAG F wrong: retrieval miss | RAG F wrong: names visible (reasoning) | RAG F wrong: self-contained |
|---|---:|---:|---:|---:|---:|
| 7B planner, E5 tools | 416 | 177 | 11 | 49 | 179 |
| 32B planner, E5 tools | 173 | 73 | 8 | 45 | 47 |
| 32B planner, v3 with gate | 3 | 0 | 0 | 0 | 3 |
| 32B planner, v3 no gate | 44 | 15 | 2 | 9 | 18 |

Reading: only 21 of the 1372 dev lines (1.5%) are retrieval misses, the case a planner that adds a missing name can fix. The planners that edit touch few of them (7B 11, 32B with the E5 tools 8, v3 without the gate 2, v3 with the gate 0 of the 21), and most lines they change are lines where RAG F was already right or the line needs nothing from other files (7B 86 %, 32B with the E5 tools 69 %, v3 without the gate 75 %). Class names are those of `experiments/reports/E4/failure_split.md`; 'right' lines have all needed names visible.
