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

## RAG comparison

Run `python -m src.evaluation.compare` from this directory to compare the same
generator on plain instructions, the unchanged official BM25 baseline, normalized
BM25 RAG, dense retrieval, and two hybrid variants. This command makes paid model
requests for uncached predictions; dense/hybrid arms also use OpenAI embeddings.

RAG retrieves only the supplied `project_defined_elements` API names. It does not
retrieve repository source, signatures, or documentation. Queries use the task
instruction; reference code and reference APIs are used only for scoring.
Normalized BM25 splits camelCase and snake_case identically in queries and names
and requires no embeddings. Dense and hybrid retrieval remain experimental arms.

Each RAG arm selects up to k unique names (default 20), then packs complete names
in relevance order into an 8,000-byte UTF-8 evidence budget, skipping names that
do not fit. This conservatively bounds evidence text tokens for byte-BPE models;
it is not a full prompt/context-window check. The instruction is preserved. Empty
candidates add no hints; score ties preserve candidate order. The most recent
retrieval trace is available as `model.last_retrieval` in memory.

Revised arms write under `rag_v2_*` paths, including tokenizer, hybrid RRF setting,
k, and evidence budget, to avoid reusing earlier RAG predictions. Metadata records
the retrieval source and settings. No new benchmark scores are claimed until
these arms have been evaluated. The official baseline remains unchanged.

Offline checks: `python -m unittest discover -s tests -v`.

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
