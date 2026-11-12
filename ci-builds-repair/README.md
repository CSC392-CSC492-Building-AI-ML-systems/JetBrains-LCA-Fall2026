# CI builds repair

Start with the [results summary](RESULTS.md) for the completed Mistral and
Qwen experiments, the comparison with the paper, and the local
validation checks. [Baseline and pilot instructions](baseline/README.md)
explain how to reproduce or audit the offline runs.

## Local baseline

From the repository root, with the cached dataset, source snapshots, and local
Mistral MLX model:

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python \
  ci-builds-repair/rag/run_benchmark.py --methods baseline --seeds 0 1 2
```

Use `--ids 53` for a one-record trial or see the
[local benchmark guide](rag/README.md) for the baseline/RAG comparison.
Inference requires Apple Silicon with Metal access. No API keys or GitHub
Actions are used. Missing public repository snapshots may require network
access. The results are offline patch and syntax metrics, not CI Pass@1.

## Original evaluator

The upstream [benchmark harness](ci-builds-repair-benchmark/README.md) is
separate: it checks out the failing revision, runs a repair function, and
pushes to the benchmark forks to run GitHub Actions. It was not used for
the results reported on this branch.
