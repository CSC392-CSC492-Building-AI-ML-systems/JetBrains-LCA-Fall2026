#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
BASE=$HOME/lca_artifacts/python/starcoderbase-1b/small_context/arms
mkdir -p $HOME/lca_artifacts/reports
cd ~/empty_cwd
export PYTHONPATH=$PROJECT PYTHONDONTWRITEBYTECODE=1
$PY $(dirname "$0")/analyze_e2.py
echo; echo "=================== REPORT 1: vs path_distance ==================="
$PY -m eval.analyze_arms --base $BASE --ref dev_path_distance --arms dev_zero dev_rag dev_rag_A1 dev_rag_A3 --out $HOME/lca_artifacts/reports/e2_vs_path_distance.md
echo; echo "=================== REPORT 2: ablations vs RAG defaults ==================="
$PY -m eval.analyze_arms --base $BASE --ref dev_rag --arms dev_rag_A1 dev_rag_A2 dev_rag_A3 --out $HOME/lca_artifacts/reports/e2_ablations_vs_rag.md
