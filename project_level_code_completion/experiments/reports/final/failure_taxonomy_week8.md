# Failure taxonomy (week 8): hand-read samples of wrong lines, medium set, CodeLlama-7B, N = 1,753

One reader (the assistant), subjective labels, 25 random wrong lines per group (seed 0, `sample_failures_week8.py`; every line is in `failure_samples_week8.txt` with the truth and all four predictions). With 25 lines a share has roughly +-10 points of sampling noise, so only large differences mean anything. Definitions in `tabulate_failure_labels.py`.

| group | lines in group | A wrong name | B different statement | C near miss | D equivalent alternative | E free text | F data / fragment |
|---|---:|---:|---:|---:|---:|---:|---:|
| no context, wrong lines | 5,432 | 2 (8%) | 10 (40%) | 9 (36%) | 3 (12%) | 1 (4%) | 0 (0%) |
| path_distance (N=1,753), wrong lines | 5,232 | 0 (0%) | 15 (60%) | 5 (20%) | 2 (8%) | 2 (8%) | 1 (4%) |
| RAG F (N=1,753), wrong lines | 4,153 | 1 (4%) | 13 (52%) | 9 (36%) | 1 (4%) | 0 (0%) | 1 (4%) |
| 7B planner agent, wrong lines | 4,255 | 5 (20%) | 7 (28%) | 11 (44%) | 1 (4%) | 1 (4%) | 0 (0%) |
| agent breaks a line RAG F got right (136 lines) | 136 | 6 (24%) | 10 (40%) | 4 (16%) | 1 (4%) | 2 (8%) | 2 (8%) |

All four wrong-line groups pooled (100 lines): A 8%, B 45%, C 34%, D 7%, E 4%, F 2%.

## What the samples show

- **Most failures are plausible code that is not what the author wrote next (B) or a small deviation from it (C).** Together they are 79 of 100 wrong lines across the four arms. Adding retrieval or an agent does not change this much: the shares of B and C stay in the same range (72 to 88 % together, per arm) for every arm, within the sampling noise.
- **Wrong names (A) are the failure retrieval can fix**, and they are a minority (about 1 in 10 of wrong lines, a little higher for the agent's sample). The agent's wrong lines contain a few more wrong names than RAG F's (5 against 1 of 25), in line with the E5 finding that its added code does not supply missing names; at this sample size that difference is a hint, not a result.
- **When the agent breaks a line RAG F got right**, the agent's replacement is mostly a different statement (10 of 25) or a wrong name (6 of 25): the line is not nudged slightly, it goes somewhere else. Only 4 of 25 are near misses. This fits the view that the added chunks steer the writer away from the right statement.
- **A share of the wrong lines are not developer errors under exact match:** equivalent alternatives (D), comment or docstring wording (E) and fragments (F) make up about 13 % of the wrong lines of the four arms; near misses (C) would earn partial credit under edit similarity.
- **The arms often fail in the same way:** on many of the sampled lines two or more arms give the identical wrong prediction (see the samples file), consistent with failures being mainly about the writer model and the open-ended next-step choice.

Limits: one labeler, no inter-rater check, 25 lines per group, random sample of all wrong lines (no stratification by line class). The retrieval-miss analysis (names missing from the prompt) is the lexical screen of E4 and E5, not this table.
