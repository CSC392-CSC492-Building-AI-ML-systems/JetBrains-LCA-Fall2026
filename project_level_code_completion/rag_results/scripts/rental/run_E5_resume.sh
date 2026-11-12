#!/usr/bin/env bash
# Finish E5 full-arm shards that were interrupted (log section 8.88): same arm, same settings as run_E5_full.sh, but with
# resume=true: the datapoints whose lines are all saved in the shard's arm folder are kept as they are, the rest are
# generated again. If a run dies again (CUDA out of memory), it is restarted up to MAX_TRIES times; each restart resumes.
#
# Usage:   LCA_USD_PER_HOUR=<price> setsid nohup bash run_E5_resume.sh <SHARD_COUNT> <shard> [<shard> ...] > /dev/null 2>&1 &
# Needs the interrupted shard's folder (e5_agent_c<COUNT>_s<shard>) in $HOME/lca_artifacts/python/codellama-7b/medium_context/arms.
source $(dirname "$0")/box_lib.sh
COUNT=${1:?usage: run_E5_resume.sh <SHARD_COUNT> <shard> [<shard> ...]}; shift
[ $# -ge 1 ] || { echo "give at least one shard index"; exit 1; }
TAG=$(echo "$@" | tr ' ' '_')
LOG=$HOME/lca_box_E5resume_c${COUNT}_s${TAG}.log
export LCA_USD_PER_HOUR=${LCA_USD_PER_HOUR:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # the 32 GB card runs close to full; less fragmentation
DTYPE=${PLANNER_DTYPE:-8bit}
PLANNER=Qwen/Qwen2.5-Coder-7B-Instruct
MAX_TRIES=${MAX_TRIES:-6}

anchor_check

if [ "$DTYPE" != "bfloat16" ]; then
  (cd ~/ts_test && $PY -c "import bitsandbytes" > /dev/null 2>&1) || (cd $PROJECT && poetry run pip install bitsandbytes >> $LOG 2>&1)
  (cd ~/ts_test && $PY -c "import bitsandbytes; print('bitsandbytes', bitsandbytes.__version__)" 2>&1 | tee -a $LOG) \
    || { echo "bitsandbytes unusable" | tee -a $LOG; exit 1; }
fi

M="params=codellama7b dataset=medium split=all lines=all"
F="rag=F rag.prefix_cap=15900 context_cap=1753"
for S in "$@"; do
  NAME=e5_agent_c${COUNT}_s${S}
  for TRY in $(seq 1 $MAX_TRIES); do
    if [ -f "$(summary_path $NAME $M)" ]; then break; fi
    echo "===== $NAME resume try $TRY start $(date +%T) =====" | tee -a $LOG
    (cd ~/ts_test && PYTHONPATH=$PROJECT $PY -m eval.compare_arms arm_name=$NAME artifacts_dir=$ART resume=true \
      $M arm=agent agent=append $F shard_index=$S shard_count=$COUNT \
      agent.planner_model=$PLANNER agent.planner_dtype=$DTYPE >> $LOG 2>&1)
    echo "===== $NAME resume try $TRY exit=$? end $(date +%T) =====" | tee -a $LOG
  done
  [ -f "$(summary_path $NAME $M)" ] || { echo "===== $NAME NOT finished after $MAX_TRIES tries =====" | tee -a $LOG; exit 1; }
done
mark "MACHINE E5resume c${COUNT} shards ${TAG} complete"
