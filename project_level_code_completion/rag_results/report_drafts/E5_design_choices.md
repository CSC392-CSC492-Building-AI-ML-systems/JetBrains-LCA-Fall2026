# E5 agent: design choices, why we made them, and what to keep in mind

Short companion to `E5_agent.md` (results). Everything here is documented step by step in `RAG_EXPERIMENT_LOG.md`, sections 8.80 to 8.91.

## The question

Our retrieval design (RAG F) builds the context for each line with one fixed recipe. Does a model that can look around the repository
and decide what to show the writer do better? E5 tests the simplest fair version of that.

## Design choices and the reason for each

| choice | why |
|---|---|
| **Built on top of RAG F**, the planner edits the context RAG F would have built | The honest test is "can an agent improve our best static context?", not "can an agent beat nothing". It also lets us attribute any difference to the edits. |
| **Writer = CodeLlama-7B (base), the same as every other arm** | Only the context may differ between arms, so a score difference can only come from the context. |
| **Planner = Qwen2.5-Coder-7B-Instruct, 8-bit, run locally** | The team works with 7B models only (no large hosted model). Two 7B models fit on one 32 GB GPU only if the planner is quantized to 8-bit. The planner was chosen on development data by a rule written down before the test: it gave a valid final answer on 1.0 of the probe lines, CodeLlama-7B-Instruct on 0.0. |
| **Tools: search_code, find_definition, read_file, submit_context** | The cheap, deterministic operations a code agent would plausibly use; they target the cross-file names that our failure analysis (E4) says matter. Tools are deterministic, so a run can be replayed exactly offline. |
| **The planner sees only the file prefix and RAG's context, never the line to predict** | No leakage of the answer. Tested in code. |
| **If the planner changes nothing or fails, the writer gets RAG F's exact prompt (byte for byte)** | Keeps the comparison clean: those lines are the RAG F arm. 5,957 of 5,958 unchanged lines have the same prediction as RAG F (the one difference is a line both get wrong; likely a greedy-decoding tie-break across machines, not tested). |
| **Add-only edits in the version we ran in full (v2)** | The first version could also drop and reorder context and lost 11.4 points on 4 development datapoints. We redesigned it before the full run so it can only add code at the lowest priority (at most 2 additions; at most 2 of the lowest-ranked default chunks may be evicted if there is no room). v2 lost 0.9 points on 8 development datapoints. |
| **A decision rule written before the full run** | "If v2 is within 2 points of RAG F on the development data, run the full arm." It was met, so we ran all 224 datapoints. No design or setting was changed after seeing the full result. |
| **Same protocol as the benchmark and E4** | All 224 medium datapoints, all 8,676 lines, context budget N = 1,753 tokens, greedy decoding, at most 100 new tokens, seed 42, exact match; paired bootstrap over datapoints (10,000 resamples, 95% interval). Predictions and the list of analyses were fixed before any shard ran. |
| **Run in 6 shards on rented GPUs, merged afterwards** | One run would take about a day; shards make a crash cost hours, not a day, and let several machines work at once. Merging is tested to be equivalent to one run. |

**Note on "add-only".** With the context budget fixed (as in every arm) the prompt is almost always full, so in practice an addition swaps out the
one or two lowest-ranked RAG F chunks: on 2,704 of the 2,718 changed lines at least one RAG F chunk is no longer in the prompt. "Add-only" describes
what the planner may request; "swap" describes what happens. Giving the agent extra budget would add without displacing, but it would change the
context size, a variable E4 already measured, so it is not part of the fair comparison at N = 1,753.

## Result in one paragraph

Agent EM 0.513 against 0.525 for RAG F: -1.24 points, 95% CI [-1.81, -0.72] (a small, real loss; under 0.5 counts as a tie by our rule),
and +9.8 over `path_distance`. It uses 6.8 times the tokens and about 16 times the time per line. When the planner changes the
context (31% of lines) it breaks 136 lines RAG F got right and fixes 34. No line class improves; the largest loss, -2.2 points, is on
lines that need names from other project files.

## What to keep in mind (limitations and honest caveats)

- **It is one agent setup, not "agents".** A 7B planner, 8-bit, add-only edits, a small tool budget, a writer that cannot ask for more context.
  A stronger or trained planner, other writers or other budgets were not tested. The claim is "this setup does not help here".
- **Two models against one.** The agent arm uses two 7B models; the baselines use one. This gives the agent extra capability, so it does not weaken the result, but the cost comparison is against a one-model pipeline.
- **Why it loses is described, not tested.** A failure split (offline, the planner replayed from its logs) shows that the agent supplied a missing name on only 7 of RAG F's 171 retrieval misses, that its evictions removed a needed name on 35 lines (retrieval misses rise from 171 to 194), and that most of the 136 lines it breaks (116) did not lose a name: the added code seems to perturb the writer. Only 2% of lines are retrieval misses, so the headroom for supplying names is about 2 points. This is a lexical screen and no ablation (for example adding random chunks) was run, so the cause is a reading, not a finding.
- **Design used 8 of the 224 datapoints.** The result is the same on the other 216 (-1.23 [-1.82, -0.70]).
- **Timings are an order of magnitude.** The agent arm ran on several rented RTX 5090 machines of different speed, and two of the six shards were continued after out-of-memory crashes (a resume option keeps finished datapoints and redoes the unfinished one; each saved row is checked before reuse). Token counts are exact.
- **Cost.** About $22 of rented GPU time in total for the full agent arm including the crashes and a replaced machine; about 30 GPU hours of useful line work.
- **Statistical power.** 224 datapoints; differences under about 0.5 points cannot be told from zero here. Lines inside a datapoint are correlated, so the bootstrap resamples datapoints.

## Where everything is

`rag_results/agent/README.md` (map and commands), `report_drafts/E5_agent.md` (results), `rag_results/figures/E5/` (3 figures and a table),
`rag_results/reports/E5/` (full analysis), code in `agent/`. The whole analysis reruns from a clean clone.

## Questions for the mentor

1. How should we frame an honest "does not beat RAG, costs 7x" result in the report?
2. Is one more agent experiment (a stronger planner, or letting the writer ask for context) worth it, or should the remaining time go to the write-up?
