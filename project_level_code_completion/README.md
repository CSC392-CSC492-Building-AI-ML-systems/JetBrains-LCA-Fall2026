# 🏟️ Long Code Arena Baselines
## Project-level code completion

This directory contains the code for running baselines for the Project-level code completion task in the [Long Code Arena benchmark](https://huggingface.co/spaces/JetBrains-Research/long-code-arena).

We provide the implementation for the following baseline: a language model that is fed with differently composed context from a repository snapshot.

The evaluation steps are the following:
* Choose the context composer:
    * Run the next token prediction for different context composers;
    * Choose the best one based on the lowest perplexity on the completion file.
* Evaluate the code completion with the composer:
    * Run one-line code completion with zero project context;
    * Run one-line code completion with a project context composed by the chosen context composer.

# How-to

## 💾 Install dependencies

* Change the working directory: `cd project_level_code_completion`
* install all the dependencies: `poetry install`
* If you are going to use Flash Attention, run `poetry run pip install flash-attn --no-build-isolation`.
  * Refer to the [official documentation](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) for more details.
  * If you are facing `ImportError` read more here: [Issue](https://github.com/Dao-AILab/flash-attention/issues/981)
* Some of the composers use [Tree Sitter](https://tree-sitter.github.io/tree-sitter/). Run `git clone --branch v0.21.0 https://github.com/tree-sitter/tree-sitter-python` to clone the Tree-sitter for Python.
  * Use this tag: the locked `tree-sitter` 0.20.4 only loads grammars with ABI 13-14, and the default branch of `tree-sitter-python` is newer (ABI 15), which fails with `Incompatible Language version 15`.
  * The library will be compiled automatically in [`tree_sitter_parser/parser.py`](tree_sitter_parser/parser.py).
  * Refer to the [official documentation](https://github.com/tree-sitter/py-tree-sitter?tab=readme-ov-file#setup) for more details.

## ⚙️ Configure a baseline

We use [Hydra](https://hydra.cc/docs/intro/) for configuration. The main config used for running experiments is located in [`eval/config/config.yaml`](eval/config/config.yaml).

### Add your model
* Base class to build the model is [`ModelBuilderBase`](model_hub/model_classes.py).
* To evaluate your model, you need to add it to [registry](model_hub/model_registry.py).

### Add your context composer
* Base class for a composer is [`OneCompletionFileComposer`](composers/one_completion_file_composer.py).
* To evaluate your composer, you need to add it to `COMPOSERS` dictionary that is located in [composers/composer_registry.py](composers/composer_registry.py).

### Retrieval-augmented context (RAG)
A per-line retrieval approach (BM25 over repository chunks, each followed by the next 20 lines of its file) is in
[`rag/`](rag/); it improves exact match on `medium_context` by about 6 points over the best static composer. Design,
results and limitations: [`RAG_PROPOSAL.md`](RAG_PROPOSAL.md); every experiment step: [`RAG_EXPERIMENT_LOG.md`](RAG_EXPERIMENT_LOG.md);
result files: [`rag_results/`](rag_results/). The tree-sitter grammar clone above must be in the working directory,
otherwise a warning is printed and Python files are cut into plain windows. Run one arm at a time:
```
poetry run python -m eval.compare_arms arm=rag rag=F dataset=medium split=dev dev_size=28 dev_extra=28 arm_name=my_run
poetry run python -m eval.compare_arms arm=path_distance dataset=medium split=dev dev_size=28 dev_extra=28 arm_name=my_baseline
poetry run python -m eval.analyze_arms --base data/code_generation/artifacts/python/starcoderbase-1b/medium_context/arms --ref my_baseline --arms my_run
```
`rag=F` is the final design, `rag=v1` the untuned BM25 design; `split=dev|heldout` selects the development or the
held-out part of the dataset.

### Suported datasets:
* All configurations of [`JetBrains-Research/lca-project-level-code-completion`](https://huggingface.co/datasets/JetBrains-Research/lca-project-level-code-completion):
   * `small_context`
   * `medium_context`
   * `large_context`
   * `huge_context`

## 🚀 Run

The main running script is [`eval/eval_pipeline.py`](eval/eval_pipeline.py).

* To start evaluation with Poetry, run: `poetry run python -m eval.eval_pipeline params=codellama7b`.
You can also add command-line arguments using [Hydra's override feature](https://hydra.cc/docs/advanced/override_grammar/basic/).

### Hydra Config Main Parameters
* `params` – to choose a model to evaluate, possible values are filenames from [the directory](eval/config/params);
* `dataset` – to choose a dataset, possible values are filenames from [the directory](eval/config/dataset);
* `artifacts_dir` – where to put all the artifacts of evaluation:
    * the results are stored in `os.path.join(config.artifacts_dir, config.language, model_name, dataset_name)`;
* `wandb_project_name` – WandB project name for the step of choosing the composer;
* `wandb_project_name_generation` – WandB project name for the step of line generation.

### Examples
* [Starcoder Base 7B](https://huggingface.co/bigcode/starcoderbase-7b) on the `small_context` set:
  * Command:
    ```
    poetry run python -m eval.eval_pipeline dataset=small params=starcoderbase7b
    ```
* [CodeLlama 7B](https://huggingface.co/codellama/CodeLlama-7b-hf) in 4bit quantization with context window 8K on the `medium_context` dataset  
  * Command:
    ```
    poetry run python -m eval.eval_pipeline dataset=medium params=codellama7b_4bit params.inference_params.seq_max_len=8000
    ```

### Run selected context composers

By default, the pipeline evaluates every registered composer. To run only path
distance plus the automatic zero-repository-context baseline, use:

```bash
poetry run python -m eval.eval_pipeline dataset=small params=codellama7b 'composers=[path_distance]'
```

This runs four perplexity evaluations (zero context and three path-distance
context sizes), followed by generation with and without repository context
when `do_generation=True`. It retains the existing perplexity-based selection
of the generation configuration; it does not force path distance if zero
context scores better.

Select several strategies with `'composers=[naive,path_distance]'`. Omit the
argument or use `composers=null` for the full sweep. The selection does not
change artifact paths; use a separate `artifacts_dir` when retaining multiple
runs of the same model and dataset.
