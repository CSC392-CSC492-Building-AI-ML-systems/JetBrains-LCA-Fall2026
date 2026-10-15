#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
export PYTHONPATH=$PROJECT PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
mkdir -p $HOME/lca_artifacts/reports
cd ~/ts_test
time $PY -m eval.proxy_metrics --arms r0_zero:zero r0_path_distance:path_distance r0_imports_first:imports_first r0_v1:rag \
    --rag prefix_cap=7900 --results --out $HOME/lca_artifacts/reports/proxy_r0_rows.jsonl 2>&1 | grep -v -i warn
