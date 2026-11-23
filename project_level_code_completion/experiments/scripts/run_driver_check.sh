#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""
export PROJECT="${PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
mkdir -p ~/empty_cwd && cd ~/empty_cwd
PYTHONPATH=$PROJECT PYTHONDONTWRITEBYTECODE=1 $PY $(dirname "$0")/check_driver.py 2>&1 | tail -20
