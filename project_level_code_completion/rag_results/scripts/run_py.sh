#!/usr/bin/env bash
# usage: run_py.sh script.py [args]   (CPU only, project on PYTHONPATH)
export CUDA_VISIBLE_DEVICES=""
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
export PYTHONPATH=$PROJECT PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
mkdir -p ~/empty_cwd; cd ~/empty_cwd
SCRIPT=$1; shift
$PY $(dirname "$0")/$SCRIPT "$@" 2>&1 | grep -v -i "warn"
