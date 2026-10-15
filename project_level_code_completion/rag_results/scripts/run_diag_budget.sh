#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
export PYTHONPATH=$PROJECT PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd ~/ts_test
$PY $(dirname "$0")/diag_budget.py 2>&1 | grep -v -i -E "warn|Generating|Loading"
