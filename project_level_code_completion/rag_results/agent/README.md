# Agent results (E5)

Everything produced by the E5 experiment: a planner agent that edits the retrieved context of our RAG design F before
CodeLlama-7B writes the line. Design, decisions and every step are in `../../RAG_EXPERIMENT_LOG.md` (sections 8.80 to 8.89).
The code is in `../../agent/`; the report text is `../report_drafts/E5_agent.md`.

**Result in one line.** At N = 1,753 tokens (CodeLlama-7B, all 224 medium datapoints, 8,676 lines) the agent scores EM 0.513
against 0.525 for RAG F (-1.24 points, 95% CI [-1.81, -0.72]) and 0.415 for `path_distance`, at 6.8x the tokens and about 16x
the time per line of RAG F.

## Layout

| folder | what it holds |
|---|---|
| `E5_dev_v1/` | development run of the first agent (can drop and reorder context; 4 datapoints, 166 lines; -11.4 points). Log 8.80-8.84. Also the planner-selection probe (`lca_artifacts/e5_dev/`). |
| `E5_dev_v2/` | development run of the second agent (`agent=append`: add-only, lowest priority; 8 datapoints, 542 lines; -0.9 points). Log 8.85-8.86. |
| `E5_full/arms/` | **the comparison**: `e5_agent_full` (all six shards merged) and the three E4 arms it is compared with (`e4_zero`, `e4_pd_N1753`, `e4_F_N1753`). Each has `summary.json`, `per_datapoint_scores.json`, `config.json` and `generation_results.jsonl.gz` (one row per line: prediction, ground truth, class, metrics, and for the agent what the planner did). |
| `E5_full/shards/` | the six shards the agent arm was run in (`c6_s0` ... `c6_s5`, 6 shards over rented GPUs), the inputs of the merge. `c6_s3` also has `generation_results.jsonl.unfinished`: the lines of the datapoint that was running when the shard crashed (set aside by `resume=true`, redone). `crashed_before_resume/c6_s2_on_E5c/` is what shard 2 had saved when it crashed; the finished shard 2 is `c6_s2`. |
| `E5_full/logs/` | trimmed run logs per rented machine (E5a, E5c, E5d; the repeated 8-bit warning was removed, a note on line 1 says how many lines). |

Shard to machine: s0, s2 (finished), s3, s4 on E5a; s1 on E5d; s5 on E5c; s2 first ran on E5c and crashed (out of memory). A first
machine for shards 1 and 4 (E5b) expired and was rerun. All machines were RTX 5090; anchor hashes were identical on all
(log 8.45, 8.87, 8.88). The raw runs of earlier experiments on rented machines are in `../rented/`.

## Reproduce

All from `project_level_code_completion/` with the project's python (the analysis needs numpy and thefuzz; the figures matplotlib).

```
python rag_results/scripts/analyze_e5_full.py     # -> rag_results/reports/E5/e5_full_analysis.md (reads E5_full/arms, runs from a clean clone)
python rag_results/scripts/make_e5_figures.py     # -> rag_results/figures/E5/ (3 figures + table)
python rag_results/scripts/analyze_agent_split.py # -> rag_results/reports/E5/agent_failure_split.md (agent vs RAG F, uses the failure_split.jsonl.gz files in E5_full/arms)
# the failure split itself (CPU, about 7 minutes; replays the planner from its logged actions, run from a directory with the tree-sitter grammar clone):
python -m eval.failure_split --base rag_results/agent/E5_full/arms --arms e5_agent_full:agent:1753 --agent append --out agent_split.jsonl
# rebuild the merged arm from the shards (what `e5_agent_full` is):
python rag_results/scripts/merge_shards.py OUT_DIR rag_results/agent/E5_full/shards/c6_s{0,1,2,3,4,5}
python rag_results/scripts/planner_probe.py check <unzipped e4_F_N1753 folder> OUT_DIR   # unchanged lines must match RAG F (5,957 of 5,958 do)
```

To rerun the generation itself: `rag_results/scripts/rental/run_E5_full.sh <shard count> <shards>` (and `run_E5_resume.sh` to finish
an interrupted shard), on a 32 GB GPU, with the Hugging Face login for CodeLlama and the Qwen planner.

## Not in the repository

The raw 9 to 11 GB machine logs (only trimmed copies are kept) and the planner and CodeLlama weights. The E4 arms other than the
three used here are in `../rented/E4/` (summaries and per-datapoint scores only).
