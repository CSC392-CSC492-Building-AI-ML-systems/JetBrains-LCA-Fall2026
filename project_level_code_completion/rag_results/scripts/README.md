# Scripts: what each one is for

Everything we ran or computed for the RAG and agent work lives in this folder. The experiment log
([`RAG_EXPERIMENT_LOG.md`](../../RAG_EXPERIMENT_LOG.md)) cites these scripts by file name, so they keep their names and
places. This page groups them by purpose; section numbers (§) are log sections. `PROJECT` is the project folder, and the shell
scripts expect the WSL set-up described in log §10 (poetry environment, `~/ts_test` with the tree-sitter grammar clone,
artifacts under `~/lca_artifacts`).

## 1. If you only want to reproduce the final numbers

The per-line result files of every arm used in the report are in [`../arms`](../arms), [`../agent`](../agent/README.md) and
[`../rented`](../rented), so these steps need no GPU:

| step | command (from `project_level_code_completion`, `PYTHONPATH=.`) | writes |
|---|---|---|
| tests | `bash rag_results/scripts/run_tests.sh with-grammar` | 288 unit tests, CPU only |
| E4 table and figures | `python rag_results/scripts/make_e4_figures.py` | `rag_results/figures/E4/` |
| E5 analysis | `python rag_results/scripts/analyze_e5_full.py` | `rag_results/reports/E5/` |
| E5 figures and table | `python rag_results/scripts/make_e5_figures.py` | `rag_results/figures/E5/` |
| stronger-agent dev table | `python rag_results/scripts/analyze_e5v3_dev.py` | `rag_results/reports/E5v3/dev_analysis.md` |
| D0 against E4 check | `python rag_results/scripts/check_d0_vs_e4.py` | `rag_results/reports/E5v3/d0_vs_e4.md` |
| dev headroom | `python rag_results/scripts/check_headroom_dev.py` | `rag_results/reports/E5v3/dev_headroom.md` |
| week 8 final table | `python rag_results/scripts/make_final_table.py` | `rag_results/reports/final/` |
| failure samples, taxonomy | `python rag_results/scripts/sample_failures_week8.py`, then `tabulate_failure_labels.py` | `rag_results/reports/final/` |

## 2. Running experiments

| experiment | log § | scripts |
|---|---|---|
| E1 pilot (StarCoder-1B, 20 datapoints) | §2 | `run_pilot.sh`, `analyze_pilot.py` |
| E2 reproduction check, ablations on dev | §5-§7 | `run_e2.sh`, `analyze_e2.py` |
| R0 / R0-T re-baseline, targeted lines | §8.11-§8.16 | `run_r0.sh`, `run_r0t.sh`, `analyze_r0.py`, `run_r0_analysis.sh`, `run_r0t_analysis.sh`, `run_proxy_r0.sh`, `count_targeted.py`, `run_count_targeted.sh` |
| V1, V2 provider path and target filter | §8.14, §8.24 | `run_v1.sh`, `run_v2.sh`, `check_v1.py`, `diag_v2.py` |
| B1-B3 design search | §8.15-§8.21 | `run_b1.sh`, `run_b2.sh`, `run_b2_detail.sh`, `run_b3.sh`, `summarize_designs.py` |
| guardrail runs | §8.22-§8.23 | `run_guardrail.sh`, `run_guardrail_analysis.sh` |
| held-out evaluation | §8.25-§8.26 | `run_heldout.sh`, `analyze_heldout.sh` |
| cleanup checks (`rag=F` preset, grammar warning) | §8.28 | `run_cleanup_check.sh`, `demo_warning.py` |
| verification controls V-A to V-F | §8.30-§8.32 | `verify_cpu.py`, `run_verify_cpu.sh`, `run_ve.sh`, `run_vf.sh`, `noroom_impact.py`, `run_noroom.sh`, `diag_budget.py`, `run_diag_budget.sh`, `coverage_check.py`, `run_coverage.sh` |
| S1 all static composers | §8.34-§8.35 | `run_s1.sh`, `analyze_s1.py`, `check_fcm.py` |
| P0 timing pilot, S3a, S4 | §8.36-§8.37 | `run_p0.sh`, `run_evidence.sh`, `run_evidence_resume.sh` (superseded by v2), `analyze_s3a.sh` |
| paper-protocol queue on all data | §8.38-§8.39 | `run_evidence_v2.sh`, `switch_chain.sh`, `wait_marker2.sh` |
| E2/E3 chunking and top-k, CodeLlama-7B | §8.40-§8.41 | `run_e23.sh`, `switch_e23.sh`, `run_local_tail.sh` |
| rented machines (vast.ai) | §8.45-§8.71 | `rental/run_A.sh`, `run_B.sh`, `run_C.sh`, `run_tail.sh`, `rental/setup_box.sh`, `rental/box_lib.sh` |
| E4 long context against retrieval | §8.72-§8.78 | `rental/run_E4.sh`, `make_e4_figures.py` (reports from `eval/compare_arms.py` and `eval/analyze_arms.py`) |
| E4 failure split and reading | §8.75-§8.77 | `eval/failure_split.py` (library), `sample_wrong_lines.py` |

## 3. The planner agent (E5)

| step | log § | scripts |
|---|---|---|
| dev checks of the planner (`planner_probe.py`), memorization probe of the 32B planner (`memorization_probe.py`) | §8.80-§8.82, §8.93 | `planner_probe.py`, `memorization_probe.py` |
| dev runs v1 and v2 | §8.83-§8.86 | `rental/run_E5_dev.sh`, `rental/run_E5_dev_v2.sh` |
| full run on all 224 datapoints, in shards | §8.87-§8.89 | `rental/run_E5_full.sh`, `rental/run_E5_resume.sh`, `merge_shards.py` |
| analysis of the full run | §8.87-§8.92 | `analyze_e5_full.py`, `analyze_agent_split.py`, `agent_change_stats.py`, `make_agent_examples.py`, `make_e5_figures.py` |
| stronger agent (v3): speed test, dev arms D0-D6 | §8.93-§8.104 | `rental/run_E5v3_speed.sh`, `rental/run_E5v3_dev.sh`, `analyze_e5v3_dev.py` |
| close-out checks and week 8 deliverables | §8.105-§8.106 | `check_d0_vs_e4.py`, `check_headroom_dev.py`, `make_final_table.py`, `sample_failures_week8.py`, `tabulate_failure_labels.py` |

## 4. Tooling and one-off checks

| script | purpose |
|---|---|
| `common.sh`, `run_py.sh` | shared shell helpers (GPU wait, paths) and a CPU-only Python runner |
| `run_tests.sh` | runs the unit tests (`with-grammar` from `~/ts_test`, `no-grammar` from an empty folder) |
| `env_check.sh`, `verify_env.py`, `run_check.sh` | GPU, cached models, library versions and tree-sitter grammar ABI |
| `export_results.sh` | copy the small result files from `~/lca_artifacts` into `rag_results/` |
| `status.sh` | progress of the held-out run (markers, tracebacks, processes) |
| `audit.py`, `run_audit.sh` | pre-held-out audit on development datapoints (leakage and snapshot checks) |
| `check_a2.py`, `check_repro.py`, `check_driver.py`, `run_driver_check.sh` | reproduction and driver sanity checks from the early phase |
| `smoke_composers.py`, `smoke_real.py`, `run_smoke.sh`, `smoke_dense.py`, `smoke_dense_real.py`, `run_dense_smoke.sh`, `run_dense_real.sh` | smoke tests: every composer runs, real data loads, a code embedder loads and ranks sensibly |
| `run_analysis.sh`, `run_cleanup_check.sh` | analysis wrappers for early dev arms |

Removed during the week 8 clean-up (they only waited for or printed the state of a long-gone background job, and nothing cited them):
`wait_marker.sh`, `wait_s1.sh`, `status_ev.sh`, `status_s1.sh`. They are in the git history.
