#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
export PYTHONPATH=$PROJECT PYTHONDONTWRITEBYTECODE=1
cd ~/empty_cwd
$PY $(dirname "$0")/count_targeted.py small 20 28 2>&1 | grep -v -i "warn" | tail -14
echo
$PY $(dirname "$0")/count_targeted.py medium 28 28 2>&1 | grep -v -i "warn" | tail -14
