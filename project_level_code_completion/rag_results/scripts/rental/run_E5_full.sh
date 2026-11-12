#!/usr/bin/env bash
# E5 full agent arm (log section 8.87): the v2 planner agent (`agent=append`) on ALL 224 medium datapoints, all lines, N=1753,
# split over several machines in small shards that are merged afterwards with rag_results/scripts/merge_shards.py.
# Everything except the planner is identical to the E4 arm `e4_F_N1753` (RAG F, CodeLlama-7B, prefix cap 15900, context cap 1753),
# which is the comparison arm; nothing is re-run for it.
#
# Usage on each machine:   LCA_USD_PER_HOUR=<price> setsid nohup bash run_E5_full.sh <SHARD_COUNT> <shard> [<shard> ...] > /dev/null 2>&1 &
# Example (6 shards over 3 machines, about 4 GPU hours per shard):
#   machine 1: run_E5_full.sh 6 0 3      machine 2: run_E5_full.sh 6 1 4      machine 3: run_E5_full.sh 6 2 5
# Every machine must use the same SHARD_COUNT. Finished shards are skipped when the script is run again; a shard that was
# interrupted starts over (no resume inside a shard), which is why shards are small. Needs a GPU with >= 32 GB.
source $(dirname "$0")/box_lib.sh
COUNT=${1:?usage: run_E5_full.sh <SHARD_COUNT> <shard> [<shard> ...]}; shift
[ $# -ge 1 ] || { echo "give at least one shard index"; exit 1; }
TAG=$(echo "$@" | tr ' ' '_')
LOG=$HOME/lca_box_E5full_c${COUNT}_s${TAG}.log
export LCA_USD_PER_HOUR=${LCA_USD_PER_HOUR:-}
DTYPE=${PLANNER_DTYPE:-8bit}
PLANNER=Qwen/Qwen2.5-Coder-7B-Instruct

anchor_check   # hashes must equal 8fe8932f... (zero) and 6941beca... (v1), as on every earlier machine

if [ "$DTYPE" != "bfloat16" ]; then
  (cd ~/ts_test && $PY -c "import bitsandbytes" > /dev/null 2>&1) || (cd $PROJECT && poetry run pip install bitsandbytes >> $LOG 2>&1)
  (cd ~/ts_test && $PY -c "import bitsandbytes; print('bitsandbytes', bitsandbytes.__version__)" 2>&1 | tee -a $LOG) \
    || { echo "bitsandbytes unusable" | tee -a $LOG; exit 1; }
fi

M="params=codellama7b dataset=medium split=all lines=all"
F="rag=F rag.prefix_cap=15900 context_cap=1753"
for S in "$@"; do
  run_once e5_agent_c${COUNT}_s${S} $M arm=agent agent=append $F shard_index=$S shard_count=$COUNT \
    agent.planner_model=$PLANNER agent.planner_dtype=$DTYPE
done
mark "MACHINE E5full c${COUNT} shards ${TAG} complete"
