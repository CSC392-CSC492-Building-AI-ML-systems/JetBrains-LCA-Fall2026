# Scripts: what each one is for

Everything we ran or computed lives here, in five purpose folders plus `rental/`. The experiment log
([`docs/RAG_EXPERIMENT_LOG.md`](../../docs/RAG_EXPERIMENT_LOG.md)) cites scripts by file name, so every script keeps its name; section
numbers (§) below are log sections. The shell scripts that run experiments expect our WSL set-up (log §10: poetry environment,
`~/ts_test` holding the tree-sitter grammar clone, artifacts under `~/lca_artifacts`); the Python analysis scripts need only the
repository.

| folder | purpose |
|---|---|
| [`report_analysis/`](report_analysis/) | **rebuild the report's tables, figures and checks from the stored results (no GPU)** |
| [`agent/`](agent/) | tools for the agent experiment: planner probes, merging shards |
| [`paper_protocol/`](paper_protocol/) | the runs that reproduce the paper's protocol on all data (E1 to E3 on CodeLlama-7B) |
| [`development/`](development/) | the development phase on StarCoder-1B: pilot, re-baseline, design search, held-out check, verification controls |
| [`rental/`](rental/) | scripts that run on rented GPU boxes (vast.ai): set-up, E4, E5 and the stronger-agent runs |
| [`tools/`](tools/) | environment checks, test runner, smoke tests, result export |
| [`common.sh`](common.sh) | shell helpers shared by the experiment scripts (GPU wait, paths) |

## 1. Rebuild the numbers without a GPU

The per-line results of every arm used in the report are in [`../results/`](../results/) (the agent folder has its own
[README](../results/agent/README.md)). From `project_level_code_completion` with `PYTHONPATH=.`:

| step | command | writes |
|---|---|---|
| tests (288, CPU only) | `bash experiments/scripts/tools/run_tests.sh with-grammar` | |
| E4 table and figures | `python experiments/scripts/report_analysis/make_e4_figures.py` | `experiments/figures/E4/` |
| E5 analysis | `python experiments/scripts/report_analysis/analyze_e5_full.py` | `experiments/reports/E5/` |
| E5 figures and table | `python experiments/scripts/report_analysis/make_e5_figures.py` | `experiments/figures/E5/` |
| agent failure split, change statistics, examples | `analyze_agent_split.py`, `agent_change_stats.py`, `make_agent_examples.py` (same folder) | `experiments/reports/E5/` |
| stronger-agent dev table | `python experiments/scripts/report_analysis/analyze_e5v3_dev.py` | `experiments/reports/E5v3/dev_analysis.md` |
| D0 against E4 check | `python experiments/scripts/report_analysis/check_d0_vs_e4.py` | `experiments/reports/E5v3/d0_vs_e4.md` |
| dev headroom | `python experiments/scripts/report_analysis/check_headroom_dev.py` | `experiments/reports/E5v3/dev_headroom.md` |
| week 8 final table | `python experiments/scripts/report_analysis/make_final_table.py` | `experiments/reports/final/` |
| failure samples, then taxonomy | `sample_failures_week8.py`, then `tabulate_failure_labels.py` (same folder) | `experiments/reports/final/` |

All of these reproduced byte for byte after the restructuring of 2026-11-22 (log §8.109); only the figure PDFs differ by an
embedded creation time.

## 2. Experiments, by phase

### `development/` (StarCoder-1B development phase, mostly §2 to §8.35)

| experiment | log § | scripts |
|---|---|---|
| E1 pilot (20 datapoints) | §2 | `run_pilot.sh`, `analyze_pilot.py` |
| E2 reproduction check, ablations on dev | §5-§7 | `run_e2.sh`, `analyze_e2.py`, `run_analysis.sh` |
| R0 / R0-T re-baseline, targeted lines | §8.11-§8.16 | `run_r0.sh`, `run_r0t.sh`, `analyze_r0.py`, `run_r0_analysis.sh`, `run_r0t_analysis.sh`, `run_proxy_r0.sh`, `count_targeted.py`, `run_count_targeted.sh` |
| V1, V2 provider path and target filter | §8.14, §8.24 | `run_v1.sh`, `run_v2.sh`, `check_v1.py`, `diag_v2.py` |
| B1-B3 design search | §8.15-§8.21 | `run_b1.sh`, `run_b2.sh`, `run_b2_detail.sh`, `run_b3.sh`, `summarize_designs.py` |
| guardrail runs | §8.22-§8.23 | `run_guardrail.sh`, `run_guardrail_analysis.sh` |
| held-out evaluation | §8.25-§8.26 | `run_heldout.sh`, `analyze_heldout.sh`, `status.sh` (progress of that run) |
| clean-up checks (`rag=F` preset, grammar warning) | §8.28 | `run_cleanup_check.sh`, `demo_warning.py`, `check_repro.py` |
| verification controls V-A to V-F | §8.30-§8.32 | `verify_cpu.py`, `run_verify_cpu.sh`, `run_ve.sh`, `run_vf.sh`, `noroom_impact.py`, `run_noroom.sh`, `diag_budget.py`, `run_diag_budget.sh`, `coverage_check.py`, `run_coverage.sh` |
| S1 all static composers | §8.34-§8.35 | `run_s1.sh`, `analyze_s1.py`, `check_fcm.py` |
| pre-held-out audit, early sanity checks | | `audit.py`, `run_audit.sh`, `check_a2.py`, `check_driver.py`, `run_driver_check.sh` |

### `paper_protocol/` (CodeLlama-7B and DeepSeek on all data: E1 to E3, §8.36-§8.71)

| step | log § | scripts |
|---|---|---|
| P0 timing pilot, S3a, S4 | §8.36-§8.37 | `run_p0.sh`, `run_evidence.sh`, `run_evidence_resume.sh` (superseded by v2), `analyze_s3a.sh` |
| the paper-protocol queue on all data | §8.38-§8.39 | `run_evidence_v2.sh`, `switch_chain.sh`, `wait_marker2.sh` |
| E2/E3 chunking and top-k | §8.40-§8.41 | `run_e23.sh`, `switch_e23.sh`, `run_local_tail.sh` |
| dry run of the repository's own pipeline (Table 10) | §8.71 | `dryrun_t10.py` |

### `rental/` (rented machines, §8.45 onward)

`setup_box.sh` and `box_lib.sh` prepare a fresh box and hold the shared helpers; `run_A.sh`, `run_B.sh`, `run_C.sh`, `run_tail.sh`
ran the E2/E3 and Table 9 arms (§8.45-§8.71); `run_E4.sh` the long-context sweep (§8.72-§8.73); `run_E5_dev.sh`,
`run_E5_dev_v2.sh`, `run_E5_full.sh`, `run_E5_resume.sh` the agent (§8.83-§8.89); `run_E5v3_speed.sh`, `run_E5v3_dev.sh` the
stronger agent (§8.94-§8.104). `requirements_freeze.txt` pins the box's Python packages.

### `agent/`

| script | purpose | log § |
|---|---|---|
| `planner_probe.py` | dev-set checks of the planner (probe, choose, check) | §8.80-§8.82 |
| `memorization_probe.py` | does the 32B planner memorize the benchmark? | §8.93 |
| `merge_shards.py` | merge the shards of one arm into the folder an unsharded run would have made | §8.87 |

### `report_analysis/`

`make_e4_figures.py` (E4 table and figures, §8.78), `analyze_e5_full.py`, `analyze_agent_split.py`, `agent_change_stats.py`,
`make_agent_examples.py`, `make_e5_figures.py` (E5, §8.87-§8.92), `analyze_e5v3_dev.py`, `check_d0_vs_e4.py`,
`check_headroom_dev.py` (stronger agent, §8.99-§8.106), `make_final_table.py`, `sample_failures_week8.py`,
`tabulate_failure_labels.py` (week 8, §8.106), `sample_wrong_lines.py` (E4 hand-read sample, §8.77).

### `tools/`

| script | purpose |
|---|---|
| `run_tests.sh` | runs the unit tests (`with-grammar` from `~/ts_test`, `no-grammar` from an empty folder) |
| `run_py.sh` | CPU-only Python runner with the project on `PYTHONPATH` |
| `env_check.sh`, `verify_env.py`, `run_check.sh` | GPU, cached models, library versions, tree-sitter grammar ABI |
| `export_results.sh` | copy the small result files from `~/lca_artifacts` into `experiments/` |
| `smoke_composers.py`, `smoke_real.py`, `run_smoke.sh`, `smoke_dense.py`, `smoke_dense_real.py`, `run_dense_smoke.sh`, `run_dense_real.sh` | smoke tests: every composer runs, real data loads, a code embedder loads and ranks sensibly |

Removed during the week 8 clean-up (they only waited for or printed the state of a long-gone background job, and nothing cited them):
`wait_marker.sh`, `wait_s1.sh`, `status_ev.sh`, `status_s1.sh`. They are in the git history.
