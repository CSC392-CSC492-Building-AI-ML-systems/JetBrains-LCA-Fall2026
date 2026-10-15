#!/usr/bin/env bash
# E2: reproduction check R + dev ablations A1-A3 on the dev split (20 datapoints), starcoderbase-1b.
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
ART=$HOME/lca_artifacts
LOG=$HOME/lca_e2.log
export WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd ~/ts_test || exit 1
: > $LOG
echo "git HEAD $(git -C $PROJECT rev-parse --short HEAD), dirty files: $(git -C $PROJECT status --short | wc -l)" | tee -a $LOG
run() {  # name, then hydra overrides
  NAME=$1; shift
  echo "===== $NAME start $(date +%T) overrides: $* =====" | tee -a $LOG
  PYTHONPATH=$PROJECT $PY -m eval.compare_arms split=dev arm_name=$NAME artifacts_dir=$ART overwrite=true "$@" >> $LOG 2>&1
  echo "===== $NAME exit=$? end $(date +%T) =====" | tee -a $LOG
}
run dev_zero            arm=zero
run dev_path_distance   arm=path_distance
run dev_rag             arm=rag
run dev_rag_A1          arm=rag rag.prefix_cap=7900
run dev_rag_A2          arm=rag rag.include_non_py=false
run dev_rag_A3          arm=rag rag.prefix_cap=7900 rag.include_non_py=false
