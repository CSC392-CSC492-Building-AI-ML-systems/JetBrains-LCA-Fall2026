#!/usr/bin/env bash
# usage: run_tests.sh [with-grammar|no-grammar] [tail-lines]
# CPU only: the GPU is hidden so a running training job is never touched.
export CUDA_VISIBLE_DEVICES=""
P="${PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
if [ "$1" = "no-grammar" ]; then mkdir -p ~/empty_cwd && cd ~/empty_cwd; else cd ~/ts_test; fi
PYTHONPATH=$P PYTHONDONTWRITEBYTECODE=1 $PY -m unittest discover -s "$P/tests" -t "$P" 2>&1 | tail -${2:-40}
