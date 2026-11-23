# experiments/: results, reports, figures and scripts of all our experiments

Everything that was measured, and everything that turns measurements into the numbers of the report, is here. The code that
runs a retrieval or agent arm is in [`rag/`](../rag/), [`agent/`](../agent/) and [`eval/`](../eval/); the story of every
step, with settings and predictions written before each run, is [`docs/EXPERIMENT_LOG.md`](../docs/EXPERIMENT_LOG.md).

## Folders

| folder | what is in it |
|---|---|
| [`report_drafts/`](report_drafts/) | **the findings in prose, one file per experiment**: start here (E1 to E5, and `week8_evaluation_and_failures.md` for the final table and failure taxonomy) |
| [`reports/`](reports/) | generated analyses (paired tests, failure splits, dev tables); `final/` has the final experiment table and the failure taxonomy, `E4/`, `E5/`, `E5v3/` the experiment analyses, `development/` the early dev-phase reports |
| [`figures/`](figures/) | the figures and results tables for the report (`E4/`, `E5/`; tables as `.md`, `.csv`, `.tex`) |
| [`results/`](results/) | the raw per-arm results (summary, per-datapoint scores, configuration; per-line predictions where small enough), see below |
| [`scripts/`](scripts/README.md) | every script, grouped by purpose; the README there has the commands that rebuild every table and figure |

## Where is what, by experiment

| experiment | question | report text | analysis files | raw results (`results/...`) |
|---|---|---|---|---|
| E1 | does simple retrieval beat the best static composer? | `report_drafts/E1_simple_rag_vs_baseline.md` | `reports/development/` (design phase) | `rented_gpu/C/` (`s3c_*`, `p9_small_*`, `p9_large_*`), `codellama_local_gpu/` (`s3a_*`, `p9_huge_*`); the design itself was developed on `starcoder_dev_and_heldout/` |
| E2 | which chunking works best? | `report_drafts/E2_chunking.md` | `reports/development/` | `rented_gpu/B/` (`e23_fine*`, `e23_file`, `e23_F`) |
| E3 | how many chunks (top-k)? | `report_drafts/E3_top_k.md` | `reports/development/` | `rented_gpu/A/` (`e23_k*`) |
| E4 | retrieval against a larger direct context (256 to 16K tokens) | `report_drafts/E4_retrieval_vs_larger_context.md` | `reports/E4/`, `figures/E4/` | `rented_gpu/E4/` |
| E5 | can a planner agent improve the retrieved context? | `report_drafts/E5_report_section.md` (short), `E5_agent.md`, `E5_design_choices.md` (details) | `reports/E5/`, `reports/E5v3/`, `figures/E5/` | `results/agent/` ([its README](results/agent/README.md) is the map) |
| week 8 | final table and failure taxonomy across all approaches | `report_drafts/week8_evaluation_and_failures.md` | `reports/final/` | built from the E4 and E5 results |

## The raw results

| folder under `results/` | what it holds |
|---|---|
| `starcoder_dev_and_heldout/` | StarCoder-1B and 7B arms of the early development phase and the held-out check (small files only: summary and configuration) |
| `codellama_local_gpu/` | CodeLlama-7B and DeepSeek-Coder arms that ran on the local GPU (timing pilot, medium dev, the HUGE set) |
| `rented_gpu/` | arms that ran on rented machines: `A`, `B`, `C` (E2/E3 ablations, the paper's small, large and medium baselines, DeepSeek) and `E4` (the long-context sweep) |
| `agent/` | the agent experiment E5: dev runs, the full 224-datapoint run, the stronger-agent dev arms, plus the E4 arms it is compared with (gzipped, so every analysis reruns from a clean clone) |

One arm folder looks like `<arm_name>/summary.json` (the numbers and the code version that produced them), `config.json` (every
setting) and `per_datapoint_scores.json`; the agent folders also keep the per-line predictions as `.jsonl.gz`.

## Rebuilding the numbers

No GPU is needed to rebuild the tables, figures and checks of the report; see the first table of
[`scripts/README.md`](scripts/README.md). Running a new arm needs a GPU; the commands for that are in the project
[`README.md`](../README.md) and in `scripts/paper_protocol/` and `scripts/rental/`.
