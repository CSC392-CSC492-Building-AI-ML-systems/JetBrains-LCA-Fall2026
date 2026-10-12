# Long Code Arena Experiment Dashboard

A web dashboard for the project's experiments. It reads the results this repo's task folders already produce, lets you set up new evaluation runs, compares the baseline, RAG, long-context and agent approaches, and lists failures for analysis. The layout follows the Figma prototype in the project report.

```
dashboard/
  backend/    FastAPI app that reads results from the repo
  frontend/   React + TypeScript (Vite) single-page app
```

| Screen | What it shows |
| --- | --- |
| **Configure** | Setup checklist and a tree of every YAML config in the repo. Click one to load its task and model into the form. Pick the task, strategy, model and the settings for that strategy, then click **Launch evaluation** or **Save draft** (saves to `configs/drafts/`). |
| **Runs** | All runs. The selected run shows its stages, score, every recorded metric and setting, where it was read from, the evidence the model was given (when recorded), and its console. Queued and running runs refresh every 2 s. |
| **Compare** | For one task and one model, the newest completed run of each approach, with the change from that model's baseline. Below it, a leaderboard of every model on the task by approach. |
| **Failures** | The failure taxonomy from the report, with counts. Filter by task, approach, category or run. Failed CI jobs link to their GitHub Actions run. |

## Quick start

```bash
# backend (from the repo root)
cd dashboard/backend
python -m venv .venv
.venv/Scripts/activate            # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload     # http://localhost:8000/docs

# frontend (second terminal)
cd dashboard/frontend
npm install
npm run dev                       # http://localhost:5173 (proxies /api to :8000)
```

To serve everything from one server, run `npm run build` in `frontend/`. FastAPI then serves `frontend/dist` at http://localhost:8000.

Tests: `cd dashboard/backend && pytest`.

## Where results come from

The backend reads results in two ways. Everything is re-read on each request, so new results show up without a restart.

### 1. Results the task folders already write (read-only)

Readers for these live in `backend/app/sources.py`.

| Task | Files read | What the dashboard shows |
| --- | --- | --- |
| Library-based code generation | `library_based_code_generation/results/[bm25/]<model>/metadata.json` and the generated `<n>.py` files | One run per model folder. Folders under `bm25/` add the top-20 BM25-ranked library names to the prompt, so they appear as **RAG**; the others are **Baseline**. Score is API Recall; ChrF is kept as a second metric. |
| CI builds repair | any `jobs_results.jsonl` under `ci-builds-repair/` (plus `jobs_ids`, `jobs_invalid`, `jobs_awaiting` next to it) | One run per output folder. Score is the share of finished jobs whose CI passed, with pass rate by difficulty. Failed and invalid jobs are listed on Failures as "Not yet analysed". The fix function comes from the branch names (`__diff__` means `fix_apply_diff`, the oracle). |
| Module summarization | `module_summarization/data/**/summary.json` from `baseline_replication.py`, and `compscore_results.json` from `compscore_replication_eval.py` | One run per prediction folder. Score is CompScore once it has been computed; ROUGE-L is shown only as a diagnostic. |

Bug localization, commit message generation and project-level code completion write their results to Hydra run folders, W&B, or gitignored output folders outside the repo, so nothing is read for them yet. To show those, either add a reader to `sources.py` or have the pipeline write the dashboard format below.

At the moment the repo contains 12 library-based code generation runs (6 models, with and without BM25) and the CI builds repair example output (`fix_apply_diff`, 150 of 194 finished jobs pass).

### 2. Dashboard runs (`results/`)

Runs launched from the dashboard, and any pipeline that wants full control over what is shown, use one folder per run:

```
results/<run_id>/
  run.json         required: metadata, metrics, stages, evidence
  config.yaml      the exact config the run used
  log.txt          console output, one line per event
  failures.jsonl   one analysed failure per line
```

```json
{
  "id": "2026-10-04-rag-bug-localization",
  "task": "bug_localization",
  "approach": "rag",
  "model": "Qwen2.5-Coder-7B-Instruct",
  "repo_version": "long-code-arena@main",
  "status": "completed",
  "created_at": "2026-10-04T14:00:00+00:00",
  "settings": { "chunking": "file-aware", "top_k": 8 },
  "metrics": { "score": 0.58, "recall_at_5": 0.66 },
  "runtime_s": 47, "cost_usd": 0.09,
  "tokens": { "input": 112800, "output": 1880 },
  "instances": 150,
  "observation": "Relevant files retrieved",
  "stages": [{ "name": "Retrieve context", "status": "completed" }],
  "evidence": [{ "path": "src/auth/session.py", "kind": "both", "score": 0.91 }]
}
```

- `task` is one of `library_based_code_generation`, `project_level_code_completion`, `bug_localization`, `ci_builds_repair`, `commit_message_generation`, `module_summarization`.
- `approach` is one of `baseline`, `rag`, `long_context`, `agent`.
- `metrics.score` is the task's primary metric, and it is the value that Compare ranks on. Other metrics are kept and shown with the run.
- `status` is one of `queued`, `running`, `completed`, `failed`. A stage `status` is one of `waiting`, `running`, `completed`, `failed`, `skipped`.
- `evidence[].kind` is `retrieved` (given to the model), `expected` (in the gold answer but not given), or `both`.

Only `task` and `approach` are required, so a pipeline can begin by writing only scores.

`failures.jsonl`:

```json
{"instance_id": "bl-0007", "category": "reasoning", "summary": "Retrieved the config file but still produced a wrong repair.", "retrieved": ["config/ci.yml"], "expected": ["config/ci.yml"]}
```

`category` is one of `missing_context`, `irrelevant_context`, `cross_file`, `reasoning`, `incorrect_code`, `context_limit`, `uncategorized`. A failure may also carry a `url`.

## Launching runs

**Launch evaluation** creates `results/<run_id>/` containing `run.json` (status `queued`) and `config.yaml`. To execute the run as well, set `LCA_RUN_COMMAND` to your evaluation entry point. The command runs from the repo root with these environment variables:

| Variable | Value |
| --- | --- |
| `LCA_RUN_DIR` | the run folder; write `run.json`, `failures.jsonl` and so on here |
| `LCA_RUN_CONFIG` | path to the run's `config.yaml` |
| `LCA_RUN_ID` | the run id |

The command's stdout and stderr stream into `log.txt`, which the Runs console displays. It may update `run.json` while it runs (stage status, evidence, metrics). If it exits without setting a final status, the run is marked `completed` or `failed` according to the exit code.

To try the flow without a model:

```bash
LCA_RUN_COMMAND="python dashboard/backend/scripts/simulate_run.py" uvicorn app.main:app
```

## Configuration

| Env var | Default |
| --- | --- |
| `LCA_REPO_ROOT` | the repo root (two folders above `backend/`) |
| `LCA_RESULTS_DIR` | `$LCA_REPO_ROOT/results` |
| `LCA_CONFIGS_DIR` | `$LCA_REPO_ROOT/configs` (where drafts are saved) |
| `LCA_RUN_COMMAND` | unset (launched runs stay queued) |
| `LCA_READ_REPO_RESULTS` | `1`; set to `0` to show only `results/` |
| `LCA_CORS_ORIGINS` | `http://localhost:5173` |
| `LCA_FRONTEND_DIST` | `dashboard/frontend/dist` |

## API

Interactive docs are at `/docs`. Main endpoints:

`GET /api/meta` · `GET /api/runs` · `POST /api/runs` · `GET /api/runs/{id}` · `GET /api/runs/{id}/log?offset=` · `GET /api/compare?task=&model=` · `GET /api/failures?task=&approach=&category=&run_id=` · `GET /api/configs` · `POST /api/configs/drafts` · `GET /api/project-tree`
