# 🏟️ Long Code Arena Baselines

## CI builds repair

This directory contains the evaluation harness for the 🤗
[CI builds repair dataset](https://huggingface.co/datasets/JetBrains-Research/lca-ci-builds-repair).

For each datapoint, the benchmark clones the failing repository, checks out the
recorded revision, invokes a user-provided repair function, and pushes the
result to GitHub so that the original CI workflow can evaluate the repair.

See the [benchmark README](ci-builds-repair-benchmark/README.md) for setup,
configuration, and usage instructions.
