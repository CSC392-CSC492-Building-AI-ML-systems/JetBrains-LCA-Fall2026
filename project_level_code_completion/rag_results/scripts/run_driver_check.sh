#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
mkdir -p ~/empty_cwd && cd ~/empty_cwd
PYTHONPATH=$PROJECT PYTHONDONTWRITEBYTECODE=1 $PY $(dirname "$0")/check_driver.py 2>&1 | tail -20
