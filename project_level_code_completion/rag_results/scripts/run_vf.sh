#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
export PYTHONPATH=$PROJECT PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
M=$HOME/lca_artifacts/python/starcoderbase-1b/medium_context/arms
cd ~/ts_test
echo "### V-F on medium dev, all lines (development datapoints only): static and v1"
$PY -m eval.proxy_metrics --dataset medium --dev-size 28 --dev-extra 28 --base $M \
    --arms g_path_distance:path_distance g_v1:rag --rag prefix_cap=7900 2>&1 | grep -E "^\|"
echo; echo "### V-F: F (continuation lines 20)"
$PY -m eval.proxy_metrics --dataset medium --dev-size 28 --dev-extra 28 --base $M \
    --arms g_F:rag --rag prefix_cap=7900 continuation_lines=20 2>&1 | grep -E "^\|"
