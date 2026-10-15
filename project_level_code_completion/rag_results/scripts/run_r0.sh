#!/usr/bin/env bash
# R0: re-baseline on the 48-datapoint dev set (dev_size=20 + dev_extra=28), starcoderbase-1b.
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
ART=$HOME/lca_artifacts
LOG=$HOME/lca_r0.log
export WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd ~/ts_test || exit 1
: > $LOG
run() {  # name, then hydra overrides
  NAME=$1; shift
  echo "===== $NAME start $(date +%T) overrides: $* =====" | tee -a $LOG
  PYTHONPATH=$PROJECT $PY -m eval.compare_arms split=dev arm_name=$NAME artifacts_dir=$ART overwrite=true "$@" >> $LOG 2>&1
  echo "===== $NAME exit=$? end $(date +%T) =====" | tee -a $LOG
}
run r0_zero            arm=zero
run r0_path_distance   arm=path_distance
run r0_imports_first   arm=imports_first
run r0_v1              arm=rag rag.prefix_cap=7900
