#!/usr/bin/env bash
# Shared helpers for experiment scripts. Source this file.
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
ART=$HOME/lca_artifacts
export WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false

# Wait until the GPU has at least $1 MiB free (another job may be using it); give up after $2 seconds.
wait_gpu() {
  NEED=${1:-12000}; LIMIT=${2:-3600}; WAITED=0
  while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ $((TOTAL - USED)) -ge "$NEED" ]; then return 0; fi
    if [ "$WAITED" -ge "$LIMIT" ]; then echo "GPU stayed busy (used ${USED}/${TOTAL} MiB) for ${LIMIT}s: giving up" ; return 1; fi
    sleep 60; WAITED=$((WAITED + 60))
  done
}

# run_arm LOG NAME OVERRIDES...  (runs eval.compare_arms from ~/ts_test so tree-sitter chunking is active)
run_arm() {
  LOG=$1; NAME=$2; shift 2
  wait_gpu 12000 3600 || { echo "===== $NAME skipped: GPU busy =====" | tee -a $LOG; return 1; }
  echo "===== $NAME start $(date +%T) overrides: $* =====" | tee -a $LOG
  (cd ~/ts_test && PYTHONPATH=$PROJECT $PY -m eval.compare_arms arm_name=$NAME artifacts_dir=$ART overwrite=true "$@" >> $LOG 2>&1)
  echo "===== $NAME exit=$? end $(date +%T) =====" | tee -a $LOG
}
