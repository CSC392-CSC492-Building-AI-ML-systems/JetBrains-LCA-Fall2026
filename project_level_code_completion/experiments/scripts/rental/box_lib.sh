#!/usr/bin/env bash
# Shared helpers for a rented machine (same behaviour as ../common.sh, but with paths for a fresh Ubuntu box).
# Expects: the repository cloned at $HOME/lca-repo, `poetry install` done, tree-sitter-python (tag v0.21.0) cloned in $HOME/ts_test.
export PROJECT=$HOME/lca-repo/project_level_code_completion
PY=$(cd $PROJECT && poetry env info -e)
ART=$HOME/lca_artifacts
LOG=${LOG:-$HOME/lca_box.log}
export WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false

run_arm() {  # NAME OVERRIDES...  (runs from ~/ts_test so tree-sitter chunking is active)
  NAME=$1; shift
  echo "===== $NAME start $(date +%T) overrides: $* =====" | tee -a $LOG
  (cd ~/ts_test && PYTHONPATH=$PROJECT $PY -m eval.compare_arms arm_name=$NAME artifacts_dir=$ART overwrite=true "$@" >> $LOG 2>&1)
  echo "===== $NAME exit=$? end $(date +%T) =====" | tee -a $LOG
}

summary_path() {  # NAME OVERRIDES...
  NAME=$1; shift; O="$*"
  case "$O" in *codellama7b*) M=codellama-7b;; *deepseekcoder1b*) M=deepseek-coder-1b;; *) M=starcoderbase-1b;; esac
  case "$O" in *dataset=medium*) D=medium_context;; *dataset=large*) D=large_context;; *dataset=huge*) D=huge_context;; *) D=small_context;; esac
  echo "$ART/python/$M/$D/arms/$NAME/summary.json"
}
run_once() {  # skips an arm whose summary.json exists
  NAME=$1; shift
  if [ -f "$(summary_path $NAME "$@")" ]; then echo "----- $NAME already done, skipped -----" | tee -a $LOG; return 0; fi
  run_arm $NAME "$@"
}
mark() { grep -q "$1" $LOG || echo "===== $1 $(date +%T) =====" | tee -a $LOG; }

CL="params=codellama7b"; DS1="params=deepseekcoder1b"
F16="arm=rag rag=F rag.prefix_cap=15900"
V1="arm=rag rag=v1 rag.prefix_cap=15900"
ALL="params=codellama7b dataset=medium split=all lines=all"
MED_HELD="dataset=medium split=heldout dev_size=28 dev_extra=28"
MED_DEV="dataset=medium split=dev dev_size=28 dev_extra=28"

three() {  # PREFIX, MODEL_PARAMS, DATA OVERRIDES... -> zero, path_distance, F16 on all lines
  P=$1; MP=$2; shift 2
  run_once ${P}_zero $MP "$@" lines=all arm=zero
  run_once ${P}_pd   $MP "$@" lines=all arm=path_distance static_mode=provider
  run_once ${P}_F16  $MP "$@" lines=all $F16
}

# Cross-machine consistency check: the same 5 datapoints on every machine; compare the printed hashes between machines.
anchor_check() {
  run_once anchor_zero $CL dataset=medium split=all limit_datapoints=5 lines=all arm=zero
  run_once anchor_v1   $CL dataset=medium split=all limit_datapoints=5 lines=all $V1
  for A in anchor_zero anchor_v1; do
    F=$ART/python/codellama-7b/medium_context/arms/$A/generation_results.jsonl
    H=$(python3 -c "import sys,json,hashlib; h=hashlib.md5(); [h.update((str(d['datapoint_idx'])+'|'+str(d['line_num'])+'|'+d['original_prediction']+'\n').encode()) for d in map(json.loads, open(sys.argv[1]))]; print(h.hexdigest())" $F)
    echo "ANCHOR $A predictions-hash $H on $(hostname) / $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" | tee -a $LOG
  done
}
