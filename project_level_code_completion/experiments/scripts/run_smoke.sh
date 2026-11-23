#!/usr/bin/env bash
P="${PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
S=$(dirname "$0")
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
cd ~/ts_test
PYTHONPATH=$P PYTHONDONTWRITEBYTECODE=1 $PY $S/smoke_real.py "$@" 2>&1 | grep -v -i "warn" | tail -30
