#!/usr/bin/env bash
P=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
S=$(dirname "$0")
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
cd ~/ts_test
PYTHONPATH=$P PYTHONDONTWRITEBYTECODE=1 $PY $S/smoke_real.py "$@" 2>&1 | grep -v -i "warn" | tail -30
