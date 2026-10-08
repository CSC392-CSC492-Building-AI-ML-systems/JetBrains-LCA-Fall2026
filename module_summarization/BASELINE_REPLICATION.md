# Module Summarization Baseline Replication

This folder contains a reproducible runner for the Long Code Arena Module Summarization baseline.

## Goal

Reproduce the fixed-context OpenAI baseline for:

- Dataset: `JetBrains-Research/lca-module-summarization`
- Split: `test`
- Number of examples: 216
- Strategy: `baseline_fixed_relevant_context`
- Model: `gpt-3.5-turbo-0125`
- Context budget: 2000 tokens

For each dataset sample, the runner uses `relevant_code_context`, trims it to the configured token budget, and asks the model to generate documentation for `docfile_name` about `intent`.

## API Key

Create `module_summarization/.env.local` or export the key in your shell:

```bash
OPENAI_API_KEY=your_key_here
```

Do not commit `.env.local`.

## Install

From `module_summarization/`, install dependencies with Poetry:

```bash
poetry install
```

If you are using an existing virtual environment instead, install the packages from `pyproject.toml`.

## Smoke Run

Run 5 examples first:

```bash
poetry run python baseline_replication.py --config replication_configs/config_openai_smoke.yaml
```

Outputs are written to:

```text
data/baseline_smoke/predictions-chatgpt3-2k/
```

Expected files:

- `0.txt` through `4.txt`
- `summary.json`

## Full Baseline Run

Run all 216 test examples:

```bash
poetry run python baseline_replication.py --config replication_configs/config_openai_full_2k.yaml
```

Outputs are written to:

```text
data/baseline_full/predictions-chatgpt3-2k/
```

Expected files:

- `0.txt` through `215.txt`
- `summary.json`

The runner defaults to `skip_existing: true`, so it can resume from existing prediction files.

## Summary File

Each run writes a `summary.json` with:

- dataset and split
- model name
- strategy name
- max context tokens
- number of dataset rows
- number of predictions
- missing prediction indices
- prediction length statistics
- target length statistics
- diagnostic ROUGE-L F1

ROUGE-L F1 is only a local diagnostic. It is not the official Module Summarization score.

## Official Evaluation

The official Long Code Arena metric for Module Summarization is CompScore. It compares generated documentation against the gold documentation with an LLM judge.

To evaluate the full baseline outputs without modifying the official `metrics.py` file:

```bash
poetry run python compscore_replication_eval.py --config replication_configs/config_compscore_full.yaml
```

This writes:

```text
data/baseline_full/compscore_results.json
```

The replication evaluator reuses the official scoring functions from `metrics.py` but does not edit that file. The default evaluation config uses `mistralai/Mistral-7B-Instruct-v0.2` on `cuda:0`, so it may require a GPU and enough local memory. If those resources are not available, keep the generated prediction files and run CompScore later in a suitable environment. Do not report ROUGE-L as the official benchmark score.

## Files Added for Replication

- `baseline_replication.py`: reproducible baseline generation runner
- `compscore_replication_eval.py`: targeted CompScore evaluation wrapper
- `replication_configs/config_openai_smoke.yaml`: 5-example smoke run
- `replication_configs/config_openai_full_2k.yaml`: full 216-example run
- `replication_configs/config_compscore_full.yaml`: targeted CompScore evaluation config
- `BASELINE_REPLICATION.md`: this guide

The original JetBrains baseline files, including `chatgpt.py` and `metrics.py`, are left unchanged.
