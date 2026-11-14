# Can a planner agent improve retrieved context? (report section for E5)

## Question and approach

Our retrieval recipe (RAG F, experiments E1 to E4) builds the context for each line the same way every time: BM25 over code chunks, a
fixed token budget, the file prefix kept whole. An agent could instead look around the repository, search for code, find definitions, read
files, and then decide what the code model sees. We asked whether that beats the fixed recipe, holding everything else fixed: the same
writer (CodeLlama-7B), the same budget (N = 1,753 tokens of repository context), the same data and the same exact-match metric.

The agent is a separate instruction model, the *planner*. It sees only the file prefix and RAG F's context, never the line to predict or
anything after it. It may call search and file tools, then submits an edited context. If it changes nothing, or its answer is invalid, the
writer gets exactly RAG F's prompt. This rule is enforced in code and tested, so every difference from RAG F comes from a changed context.

## Experiment 1: a 7B planner on all of the medium set

Qwen2.5-Coder-7B-Instruct (8-bit), able to search, find definitions and read files, and only allowed to add code at the lowest priority.
We ran it on all 224 datapoints (8,676 lines). Everything was written down before the run: the expected outcome, the tests and the analyses.

| approach | EM (95% CI) | tokens per line (planner + writer) | median seconds per line |
|---|---|---:|---:|
| no context | 0.387 [0.365, 0.409] | 2,968 | 0.56 |
| path_distance (best static composer) | 0.415 [0.391, 0.439] | 4,712 | 0.69 |
| RAG F | 0.525 [0.492, 0.558] | 4,723 | 0.72 |
| 7B planner agent | 0.513 [0.482, 0.544] | 32,201 | 11.3 |

The agent is **1.24 points worse than RAG F** (95% interval -1.81 to -0.72, paired bootstrap over datapoints) while using 6.8 times the tokens
and about 16 times the time. It is 9.8 points better than the best static composer, but all of that gain is already in RAG F. When the planner
changed the context (31% of lines) it broke 136 lines that RAG F got right and fixed 34. No line class improved; the largest loss (-2.2 points)
was on lines that need names defined in other project files, the case an agent was meant to help.

Why there is little to gain: only 2% of lines (171 of 8,676) are a *retrieval miss*, meaning a name the line needs from another file is not in
RAG F's prompt. The agent made the missing name visible on 7 of those 171, and its evictions of default chunks removed a needed name on 35 lines.
Most of the lines it broke did not lose a name, so the added code seems to distract a writer that already had a good context. That reading is
not tested (no ablation with random added chunks).

## Experiment 2: a stronger planner (32B), tested on a development set

A mentor-approved larger planner (Qwen2.5-Coder-32B-Instruct, 4-bit) was combined with every improvement we could justify from experiment 1:
hybrid search (BM25 plus dense embeddings), tools for usages, file outlines and imports, a mode that can swap chunks as well as add them, a
prompt that asks for a stated hypothesis (which name is missing), and a do-no-harm gate that keeps an edit only if the missing name it names
is in the added code and not already visible. Before any result we fixed the plan, the predictions and a rule: run the full 224-datapoint
arm only if the development gain is at least +0.5 points, its interval's lower bound at least -1.0, and broken lines at most twice fixed lines.
Two revisions of the design were allowed and both were used (a reminder to name the missing symbol; removing the gate).

Development set: 24 datapoints (1,372 lines), the same machine for every arm.

| arm | EM | vs RAG F | lines changed | broke : fixed | tokens per line (planner + writer prompt) | vs RAG F |
|---|---:|---:|---:|---|---:|---:|
| RAG F | 0.5037 | reference | none | none | 5,047 | 1.0x |
| 7B planner, experiment 1 tools | 0.4865 | -1.72 [-3.90, -0.33] | 416 (30%) | 12 : 1 | 34,465 | 6.8x |
| 32B planner, experiment 1 tools | 0.5036 | -0.01 [-0.58, +0.45] | 173 (13%) | 1 : 2 | 21,519 | 4.3x |
| 32B planner, new tools, gate | 0.5037 | +0.00 | 3 (0.2%) | none | 13,969 | 2.8x |
| 32B planner, new tools, one tool call | 0.5037 | +0.00 | 3 (0.2%) | none | 10,656 | 2.1x |
| 32B planner, new tools, no gate | 0.5037 | +0.00 | 44 (3.2%) | 0 : 0 | 13,610 | 2.7x |

Tokens are the mean per line of planner input plus output plus the writer's input prompt (the writer's output, about 20 tokens, is left out);
planner tokens are 29,493, 16,507, 8,923, 5,610 and 8,573 per line in the five agent rows. The new tools cost fewer tokens than the old ones
because the planner submits at once on about two lines in three, but they spend them on almost no edits.

The 7B agent reproduces its loss (-1.72 here, -1.24 on all 224). A larger planner removes the harm (12 : 1 becomes 1 : 2) but adds no gain.
With the new tools and prompt, the 32B planner submits the unchanged context on about two lines in three and almost never edits; without the
gate it edits 44 lines and none of those edits changes what the writer produces. The decision rule says no, so we did **not** run the stronger
agent on all 224 datapoints. In tokens the stronger agents cost 2.1 to 4.3 times RAG F per line and the 7B agent 6.8 times (the same ratio as on
all 224 datapoints); in time they took 4 to 22 seconds per line against 1.2 for RAG F, with two arms sharing the GPU, so read the seconds as an
order of magnitude.

## Conclusion and limits

On this benchmark, with this writer and budget, a planner agent does not improve on a tuned retrieval recipe: the small one hurts, the larger
one is neutral, and both cost many times more. We attribute this to how little room there is (retrieval misses are 2% of lines), not to a
proof that agents cannot help.

Limits to state with the result:
- The stronger agent was evaluated on 24 development datapoints, which resolve about +-1 point; it was not run on the full set, by our own rule.
- The 7B planner with the new tools was started and stopped after 174 lines for cost (64 seconds per line, one line changed); it has no result.
- One writer (CodeLlama-7B) and one budget were used. A trained planner, other writers, or letting the writer ask for more context were not tested.
- The development machine's GPU gives slightly different greedy predictions from the one used for experiment 1 (2.9% of lines, +0.29 points
  [+0.01, +0.69] for RAG F on the same lines), so all development arms, RAG F included, ran on the same card.
- Timings are an order of magnitude: experiment 1 ran on several rented GPUs of different speed.

Where to look: `rag_results/report_drafts/E5_agent.md` and `E5_design_choices.md` (details), `rag_results/reports/E5/` and
`rag_results/reports/E5v3/` (full analyses), `RAG_EXPERIMENT_LOG.md` sections 8.87 to 8.105 (plans, predictions, every run). All analyses
rerun from a clean clone with `analyze_e5_full.py`, `analyze_e5v3_dev.py` and `check_d0_vs_e4.py`.
