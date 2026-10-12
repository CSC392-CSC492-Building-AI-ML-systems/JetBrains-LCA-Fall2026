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

`baseline_replication.py` replicates only the released BM25 top-20 API-hint baseline, using revision-pinned test data and the benchmark's existing prompts and metrics. It saves predictions under `replication_results/bm25/`, along with the dataset revision, run settings, source hashes, and per-example scores. Set the matching `OPENAI_API_KEY` or `TOGETHER_API_KEY` environment variable before generation.

Run a 5-example smoke run for one model:

```bash
poetry run python baseline_replication.py --model gpt-3.5-turbo-0125 --limit 5
```

Run all six released models on the full test set:

```bash
poetry run python baseline_replication.py
```

Repeat `--model` to select several models. Runs resume from existing numbered predictions; use `--restart` to regenerate them. To recompute metrics without model API calls, use the same settings with `--evaluate-only`. Partial runs are labeled in their summaries and should not be reported as full benchmark scores.

The existing `src/evaluation/evaluate.py` and `results/` workflow remains available for the original experiments.


## ⚙️ Customize

* To implement more metrics, refer to the available [examples](src/metrics). Currently, we run `ChrF` and `API Recall`.
* To add more models, implement new instances of [abstract model](src/models/example_generation_model.py). We provide connectors to all models available via OpenAI and Together.AI.
* To work with code, we provide [examples of parsing](src/context) with the [tree-sitter](https://github.com/tree-sitter/py-tree-sitter) library.
