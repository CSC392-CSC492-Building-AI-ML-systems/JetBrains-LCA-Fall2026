#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
BASE=$HOME/lca_artifacts/python/starcoderbase-1b/small_context/arms
mkdir -p $HOME/lca_artifacts/reports ~/empty_cwd
cd ~/empty_cwd
export PYTHONPATH=$PROJECT PYTHONDONTWRITEBYTECODE=1
$PY $(dirname "$0")/analyze_r0.py
echo; echo "=================== REPORT: R0 arms vs path_distance ==================="
$PY -m eval.analyze_arms --base $BASE --ref r0_path_distance --arms r0_zero r0_imports_first r0_v1 --out $HOME/lca_artifacts/reports/r0_vs_path_distance.md
