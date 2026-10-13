# 🏟️ Long Code Arena Baselines
## Library-based code generation

This directory contains the code for running baselines for Library-based code generation task in Long Code Arena benchmark.

We provide the implementation of baselines running inference via [OpenAI](https://platform.openai.com/docs/overview) and [Together.AI](https://www.together.ai/).
We evaluate multiple models in two settings: 
* Generating a program based on a plain instruction, without any repository-level information;
* Generating a program based on the instruction and top-20 method and class names from the library according to BM-25 with instruction as a reference.

# How-to

## 💾 Install dependencies

We provide dependencies for two Python dependencies managers: [pip](https://pip.pypa.io/en/stable/) and [Poetry](https://python-poetry.org/docs/). Poetry is preferred, `requirements.txt` is obtained by running `poetry export --with dev,eda --output requirements.txt`.

* If you prefer pip, run `pip install -r requirements.txt`
* If you prefer Poetry, run `poetry install`

## 🚀 Run

In order to evaluate models, run the [evaluation script](src/evaluation/evaluate.py).

* If you use Poetry, run: `poetry run python -m src.evaluation.evaluate`
* Otherwise, run: `python -m src.evaluation.evaluate`

The script will generate code with the available models and compute the metrics (`ChrF` and `API Recall`). 
If the code for the specific model is already available in the `results` directory, the evaluation script will just compute the metrics.

## Reproducible baseline replication

`baseline_replication.py` replicates the BM25 top-20 API-hint baseline with the benchmark's existing prompts, model adapters, code extraction, ChrF, and API Recall metrics. It pins the dataset to the current Hub revision and records source hashes and package versions for each fresh run.

Set `OPENAI_API_KEY` for OpenAI models and `TOGETHER_API_KEY` for Together models. Keys are read from the environment and are not stored in config files.

Run a five-example smoke test:

```bash
poetry run python baseline_replication.py --config replication_configs/config_bm25_smoke.yaml
```

Run the full test split with all six released models:

```bash
poetry run python baseline_replication.py --config replication_configs/config_bm25_full.yaml
```

The smoke config writes under `results/bm25/gpt-3.5-turbo-0125/`. The full config writes per-model predictions, run metadata, and summaries under `results/bm25/`. New predictions are nested in each model folder, preserving the existing numbered `.py` files. Fresh predictions resume automatically; use `--restart` to regenerate them. Use `--evaluate-only` to score completed saved predictions without model API calls.

For pip installation, use `python -m pip install -r requirements-replication.txt`. The results table and existing baseline-score provenance are documented in [RESULTS.md](RESULTS.md). The original [evaluation workflow](src/evaluation/evaluate.py) remains available.


## ⚙️ Customize

* To implement more metrics, refer to the available [examples](src/metrics). Currently, we run `ChrF` and `API Recall`.
* To add more models, implement new instances of [abstract model](src/models/example_generation_model.py). We provide connectors to all models available via OpenAI and Together.AI.
* To work with code, we provide [examples of parsing](src/context) with the [tree-sitter](https://github.com/tree-sitter/py-tree-sitter) library.
