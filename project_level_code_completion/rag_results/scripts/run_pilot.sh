#!/usr/bin/env bash
# Pilot: 20 evenly spaced datapoints of small_context on starcoderbase-1b, one arm per invocation.
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
ART=$HOME/lca_artifacts
LOG=$HOME/lca_pilot.log
export WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd ~/ts_test || exit 1
: > $LOG
for ARM in zero path_distance rag; do
  echo "===== arm=$ARM start $(date +%T) =====" | tee -a $LOG
  PYTHONPATH=$PROJECT $PY -m eval.compare_arms arm=$ARM limit_datapoints=20 arm_name=pilot_$ARM \
      artifacts_dir=$ART overwrite=true >> $LOG 2>&1
  echo "===== arm=$ARM exit=$? end $(date +%T) =====" | tee -a $LOG
done
