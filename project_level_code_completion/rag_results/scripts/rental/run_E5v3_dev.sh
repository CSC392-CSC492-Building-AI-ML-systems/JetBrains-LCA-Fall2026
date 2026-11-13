#!/usr/bin/env bash
# Stronger agent (v3), development arms on the 24 development datapoints of RAG_EXPERIMENT_LOG section 8.93 (split=dev
# dev_size=8 dev_extra=16, medium, 1,372 lines), CodeLlama-7B writer, N=1753. Arms (letters as in 8.93):
#   d0  RAG F                                 reference
#   d1  7B planner + old tools   (agent=append, Qwen2.5-Coder-7B-Instruct 8-bit)        E5's agent on the 24 datapoints
#   d2  32B planner + old tools  (agent=append, Qwen2.5-Coder-32B-Instruct 4-bit)       planner size alone
#   d3  32B planner + v3         (agent=v3)                                              the full stronger agent
#   d4  7B planner + v3 tools    (agent=v3, planner 7B 8-bit)                            tools alone
#   d5  32B planner + v3, one tool call (agent=v3 agent.limits.tool_calls=1)             does the loop help?
# Usage on one machine:  LCA_USD_PER_HOUR=<price> setsid nohup bash run_E5v3_dev.sh <arm> [<arm> ...] > /dev/null 2>&1 &
#   e.g. run_E5v3_dev.sh d0 d3     (the 32B arms need a GPU with >= 48 GB: the script refuses to start them on less)
# An interrupted arm is continued (resume=true, up to MAX_TRIES tries); finished arms are skipped.
source $(dirname "$0")/box_lib.sh
[ $# -ge 1 ] || { echo "give at least one arm: d0 d1 d2 d3 d4 d5"; exit 1; }
TAG=$(echo "$@" | tr ' ' '_')
LOG=$HOME/lca_box_E5v3dev_${TAG}.log
export LCA_USD_PER_HOUR=${LCA_USD_PER_HOUR:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAX_TRIES=${MAX_TRIES:-4}
P7=Qwen/Qwen2.5-Coder-7B-Instruct
P32=Qwen/Qwen2.5-Coder-32B-Instruct
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

# hashes must equal 8fe8932f... (zero) and 6941beca... (v1) on 5090 boxes (the RTX PRO 6000 differs on 11 of 316 line-runs, log 8.96).
# SKIP_ANCHOR=1 is for a second parallel stream on a machine whose first stream already ran the check.
[ -n "${SKIP_ANCHOR:-}" ] || anchor_check

(cd ~/ts_test && $PY -c "import bitsandbytes" > /dev/null 2>&1) || (cd $PROJECT && poetry run pip install bitsandbytes >> $LOG 2>&1)
(cd ~/ts_test && $PY -c "import bitsandbytes; print('bitsandbytes', bitsandbytes.__version__)" 2>&1 | tee -a $LOG) \
  || { echo "bitsandbytes unusable" | tee -a $LOG; exit 1; }

M="params=codellama7b dataset=medium split=dev dev_size=8 dev_extra=16 lines=all"
F="rag=F rag.prefix_cap=15900 context_cap=1753"

arm_overrides() {  # prints the overrides of an arm, or fails
  case $1 in
    d0) echo "arm=rag $F" ;;
    d1) echo "arm=agent agent=append $F agent.planner_model=$P7 agent.planner_dtype=8bit" ;;
    d2) echo "arm=agent agent=append $F agent.planner_model=$P32 agent.planner_dtype=4bit" ;;
    d3) echo "arm=agent agent=v3 $F" ;;
    d4) echo "arm=agent agent=v3 $F agent.planner_model=$P7 agent.planner_dtype=8bit" ;;
    d5) echo "arm=agent agent=v3 $F agent.limits.tool_calls=1" ;;
    *) return 1 ;;
  esac
}

for A in "$@"; do
  NAME=e5v3dev_$A
  OV=$(arm_overrides $A) || { echo "unknown arm $A" | tee -a $LOG; exit 1; }
  case $A in d2|d3|d5) [ "${GPU_MB:-0}" -ge 45000 ] || { echo "arm $A needs a GPU with >= 48 GB (this one has ${GPU_MB} MB)" | tee -a $LOG; exit 1; } ;; esac
  for TRY in $(seq 1 $MAX_TRIES); do
    if [ -f "$(summary_path $NAME $M)" ]; then break; fi
    echo "===== $NAME try $TRY start $(date +%T) overrides: $OV =====" | tee -a $LOG
    (cd ~/ts_test && PYTHONPATH=$PROJECT $PY -m eval.compare_arms arm_name=$NAME artifacts_dir=$ART resume=true $M $OV >> $LOG 2>&1)
    echo "===== $NAME try $TRY exit=$? end $(date +%T) =====" | tee -a $LOG
  done
  [ -f "$(summary_path $NAME $M)" ] || { echo "===== $NAME NOT finished after $MAX_TRIES tries =====" | tee -a $LOG; exit 1; }
done
mark "MACHINE E5v3dev ${TAG} complete"
