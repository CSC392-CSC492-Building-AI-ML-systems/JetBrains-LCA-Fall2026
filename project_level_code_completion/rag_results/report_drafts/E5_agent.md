# E5: Can a planner agent improve the retrieved context? (draft report section)

## Question

RAG design F (E1 to E4) builds the context with one fixed recipe. Would a model that can look around the repository (search code,
find definitions, read files) and then decide what the writer sees do better? We test the simplest version that is fair to
the benchmark: a small local instruction model, the planner, edits the context that RAG F would have used; CodeLlama-7B, the
same model as in every other arm, writes the line.

## Setup

- **Agent.** Qwen2.5-Coder-7B-Instruct (8-bit), chosen on development data by a rule written down before the run (valid answers on
  1.0 of the probe lines against 0.0 for CodeLlama-Instruct). It sees only the file prefix and the RAG F context, and can call
  `search_code`, `find_definition`, `read_file`, then `submit_context`. Greedy decoding, at most a fixed number of steps and
  tool calls.
- **Two versions.** v1 could add, drop and reorder context. v2 (the one run in full) can only add, at the lowest priority:
  added code goes last, at most 2 additions, and at most 2 of the lowest-ranked default chunks are evicted if there is no room.
- **Fairness, enforced in code and tested.** The planner never sees the line to predict or anything after it. If it changes
  nothing or its reply is invalid, the writer gets exactly RAG F's prompt, so those lines are the same as the RAG F arm. The token
  budget N = 1,753 for repository context and the whole-prefix rule are the same as in the E4 arms.
- **Data and protocol.** Medium project-level completion set, all 224 datapoints, all 8,676 target lines; greedy decoding, at most
  100 new tokens, seed 42, exact match (EM) as the main metric; paired bootstrap over datapoints (10,000 resamples, 95% interval,
  seed 0). The comparison arms are the E4 arms at N = 1,753 (nothing was re-run for them).
- **How it was run.** Six shards over rented RTX 5090 GPUs, merged afterwards; two shards ran out of GPU memory and were continued
  with a resume option that keeps finished datapoints (log 8.88). Expected outputs, tests and analyses were fixed before the run
  (log 8.87) and nothing was changed after seeing the result.

## Results

| approach | EM (95% CI) | ES | wrong lines (of 8,676) | writer input tokens/line | total tokens/line | p50 s/line | p95 s/line |
|---|---|---|---|---|---|---|---|
| no context | 0.387 [0.365, 0.409] | 74.2 | 5,432 | 2,951 | 2,968 | 0.56 | 1.26 |
| path_distance | 0.415 [0.391, 0.439] | 75.7 | 5,232 | 4,695 | 4,712 | 0.69 | 1.41 |
| RAG F | 0.525 [0.492, 0.558] | 80.3 | 4,153 | 4,705 | 4,723 | 0.72 | 1.53 |
| agent (v2) | 0.513 [0.482, 0.544] | 80.0 | 4,255 | 4,609 | 32,201 | 11.34 | 22.16 |

N = 1,753 tokens, CodeLlama-7B. Total tokens = planner plus writer. Seconds = context building (planner included) plus generation,
the E4 definition. Agent times mix GPUs of different speed (the useful line work was about 30 GPU hours, against 1.9 for RAG F):
read them as an order of magnitude. The same table as csv, and the full set of fixed analyses, are in `rag_results/figures/E5/` and
`rag_results/reports/E5/e5_full_analysis.md`. (Figures: `fig_e5_em`, `fig_e5_by_class`, `fig_e5_cost`.)

1. **The agent is slightly worse than plain retrieval.** Agent minus RAG F: -1.24 EM points [-1.81, -0.72] (paired, 224 datapoints);
   -1.18 points pooled over lines; ES -0.37 [-0.63, -0.13]. It is worse on 48 datapoints, better on 16, tied on 160.
2. **It is far better than the best static composer.** Agent minus `path_distance`: +9.81 points [+7.43, +12.29]. All of that gain
   is already in RAG F.
3. **It is expensive.** 6.8x the tokens and about 16x the time per line of RAG F (p50 11.3 s against 0.72 s).
4. **The result does not depend on a few lines or on the development data.** Without the 19 lines where the planner failed and RAG's
   prompt was used: -1.24. Over the 216 datapoints not used for design: -1.23 [-1.82, -0.70].
5. **Where it loses.** The planner changed the context on 31.3% of the lines. On those lines RAG F is right on 1,513 and the agent on 1,411:
   136 lines only RAG F gets right, 34 only the agent. Unchanged lines are the same as RAG F (5,957 of 5,958; the one difference is a
   line where both arms are wrong). By line class (agent minus RAG F, points): inproject -2.2 [-4.2, -0.6], non_informative -1.5,
   random -1.0, infile -0.7, common -0.5, commited -0.2 (interval includes 0). No class improves, and the biggest loss is on lines that
   need names from other project files, the case an agent that searches and reads files was meant to help.

## Why it loses: where the planner's edits land (failure split)

We ran the E4 lexical failure split on the agent arm (the planner is replayed from its logged actions, no GPU; the rebuilt context equals
the logged one on 8,673 of 8,676 lines). A wrong line is a *retrieval miss* if a name it needs from another file is not in the prompt.

| arm | right | wrong, needs nothing from other files | wrong, retrieval miss | wrong, names visible |
|---|---|---|---|---|
| RAG F | 4,523 (52.1%) | 2,755 (31.8%) | 171 (2.0%) | 1,227 (14.1%) |
| agent (v2) | 4,421 (51.0%) | 2,797 (32.2%) | 194 (2.2%) | 1,264 (14.6%) |

- **The agent hardly ever supplies a missing name.** Of RAG F's 171 retrieval misses it makes all needed names visible on 7 (one of them right). Even a perfect planner could add only about 2 points this way, because only 2% of lines are retrieval misses.
- **Its evictions remove names.** 35 lines lose a needed name that RAG F showed (RAG F right on 24 of them, the agent on 5); retrieval misses rise from 171 to 194.
- **Most broken lines did not lose a name.** Of the 136 lines only RAG F gets right (among those the planner changed), 63 had their names visible in both prompts, 53 need nothing from other files, and only 20 became retrieval misses. The 34 lines only the agent gets right split 22 / 11 / 1 the same way.
- **Reading, not a test:** extra code placed in the context perturbs a writer that already had a good context. The screen is lexical, and we did not run an ablation (for example adding random chunks) that would test this cause.

## What the edits look like (examples and numbers)

`rag_results/reports/E5/agent_examples.md` has random examples of each kind of change (fixed seed, 3 per kind, all 7 filled misses): code before
the line, the truth, both predictions, the needed names missing from each prompt, the planner's searches, and what it added and displaced.

- **An addition is in practice a swap.** With the context budget fixed at N = 1,753 tokens the prompt is almost always full: on 2,704 of the 2,718
  lines the planner changed, at least one of RAG F's chunks is gone (727 lines lose one, 1,844 lose two, 133 lose three to six). The agent
  swaps the lowest-ranked RAG F chunks for what the planner picked, often a larger module-level or function chunk (median 17 lines, mean 73,
  against median 25 and mean 34 for RAG F's chunks).
- **Typical break.** The planner searches for a word from the prefix (sometimes a generic one, such as `os` or `logger`), adds the chunk it finds,
  and the writer then produces a plausible line that differs from the truth, for example a `channel.send(...)` line where the file does `elif find['type']==2:`.
- **The filled misses are weak wins.** Of RAG F's 171 retrieval misses the agent filled 7 and got 1 right; three of the seven are real cross-file names
  (`create_task`, `ConversationMessage`, `FederationDescendant`) and the rest are generic words. The lexical screen counts any identifier defined anywhere in the
  repository as a needed name, so some "misses" are false alarms.
- **Not tested:** giving the planner extra budget so that it adds without displacing. That changes the context size, which E4 already measured.

## What this does and does not show

- It shows that this setup, a 7B planner with a small tool budget that may only add context, does not improve on a well-tuned retrieval
  recipe on this benchmark, and that it costs several times more. Development runs said the same on 8 datapoints (-0.9 points for v2), and
  the first version, which could also drop context, was much worse (-11.4 points on 4 datapoints).
- It does not show that agents cannot help: a stronger or trained planner, other writers, other budgets and giving the writer a way to ask
  for more context were not tested. We also did not test why the changes hurt; one reading, not tested here, is that BM25 on the last lines
  already puts the likeliest chunk first and extra code mostly adds distraction for a model that was given a good context.
- Statistical power is that of 224 datapoints: differences under about 0.5 points are not distinguishable from 0 here.

## Reproduce

`python rag_results/scripts/analyze_e5_full.py` and `python rag_results/scripts/make_e5_figures.py` from `project_level_code_completion/`
(they read the arms kept in `rag_results/agent/E5_full/arms`). Layout of the raw results: `rag_results/agent/README.md`.
