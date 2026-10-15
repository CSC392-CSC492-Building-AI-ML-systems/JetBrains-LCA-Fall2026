# 🏟️ Long Code Arena Baselines

## CI builds repair

This directory contains the evaluation harness for the
[🤗 CI builds repair dataset](https://huggingface.co/datasets/JetBrains-Research/lca-ci-builds-repair).
It runs a repair function against failing repository snapshots and evaluates
the resulting commits with GitHub Actions.

## Prerequisites

- Python 3.10
- Git
- Membership in the
  [LCA-CI-builds-repair organization](https://github.com/LCA-CI-builds-repair)
- A GitHub personal access token with permission to push to the benchmark
  repositories and read their Actions results

## 💾 Install dependencies

From the repository root, change into this directory:

```bash
cd ci-builds-repair/ci-builds-repair-benchmark
```

Poetry is the preferred dependency manager:

```bash
poetry install
```

Alternatively, install the exported dependencies with pip:

```bash
python -m pip install -r requirements.txt
```

## ⚙️ Configure

Create a local configuration file from the tracked template:

```bash
cp config_template.yaml config.yaml
```

Set the following values in `config.yaml`:

| Field | Description |
| --- | --- |
| `repos_folder` | Directory where benchmark repositories are cloned. |
| `out_folder` | Directory where evaluation results are written. |
| `data_cache_dir` | Directory used for the Hugging Face dataset cache. |
| `username_gh` | GitHub username used to push benchmark branches. |
| `test_username` | Optional name included in benchmark branch names; defaults to `username_gh`. |
| `language` | Dataset language. Currently only `Python` is supported. |

Store your GitHub token in a local `.env` file (ignored by Git):

```dotenv
TOKEN_GH=your_token_here
```

Keep `config.yaml`, `.env`, cloned repositories, cached data, and run outputs
local. The repository-level `.gitignore` excludes `config.yaml` and the
conventional local `repos/`, `data/`, and `out/` directories.

## 🚀 Run

Use [`run_benchmark.py`](run_benchmark.py) as the starting point. Uncomment the
operation you want to run, then execute it with your selected environment:

```bash
poetry run python run_benchmark.py
```

The main entry point is `CIFixBenchmark.eval_dataset(fix_repo_function)`. It:

1. Downloads the dataset.
2. Clones and prepares each failing repository snapshot.
3. Calls `fix_repo_function` to modify the local checkout.
4. Pushes the repair branch to GitHub and waits for Actions results.
5. Collects and summarizes the results.

Start with a small run while developing a repair function:

```python
CIBenchPython.eval_dataset(fix_repo_function, num_dp=1)
```

### ⚒️ Repair function

A repair function may accept the following arguments:

```python
def fix_repo_function(datapoint, repo_path, repo, out_folder):
    ...
```

- `datapoint`: Dataset record containing the failure metadata and logs.
- `repo_path`: Path to the local repository checkout.
- `repo`: `git.Repo` instance for the checkout.
- `out_folder`: Directory for benchmark output files.

Two reference implementations are available in
[`benhmark_functions.py`](benhmark_functions.py):

- `fix_none` leaves the checkout unchanged.
- `fix_apply_diff` applies the ground-truth repair diff.

Use `CIFixBenchmark.get_dataset()` to download or inspect the dataset without
running an evaluation.

## 📈 Outputs

The configured output directory contains:

- `jobs_ids_<model>.jsonl`: identifiers for jobs submitted to GitHub.
- `jobs_results_<model>.jsonl`: completed job results.
- `jobs_awaiting_<model>.jsonl`: jobs still waiting for a result.
- `jobs_invalid_<model>.jsonl`: jobs that could not be evaluated.

Example files are available in the [`examples`](examples/) directory.

To resume result collection from a previously created jobs file, call:

```python
CIBenchPython.eval_jobs(job_ids_file="path/to/jobs_ids_<model>.jsonl")
```

## 📩 Contact info

If you have any questions or requests concerning this dataset, please contact lca@jetbrains.com
