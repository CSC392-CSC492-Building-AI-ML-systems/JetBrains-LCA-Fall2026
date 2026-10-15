#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
export PYTHONPATH=$PROJECT PYTHONDONTWRITEBYTECODE=1
echo "### from ~/ts_test (what the experiment scripts use)"; cd ~/ts_test; $PY $(dirname "$0")/audit.py 2>&1 | grep -v -i -E "warn|Generating"
echo; echo "### from ~/empty_cwd (no grammar): chunk kinds"
cd ~/empty_cwd; $PY $(dirname "$0")/audit.py 2>&1 | grep -E "chunk kinds|symbols"
