#!/usr/bin/env bash
# E5 planner v2 development run on ONE rented machine (log section 8.85). Development datapoints only, about 1 GPU hour.
#   1. anchor check (same 5 datapoints as every machine)
#   2. RAG F versus the v2 planner (`agent=append`: it can only ADD code) on 8 development datapoints
#      (0, 28, 56, 84, 112, 140, 168, 196: the 4 used for v1 in section 8.84 plus 4 not seen before), N=1753,
#      planner Qwen/Qwen2.5-Coder-7B-Instruct (chosen by the pre-registered rule of section 8.83)
#   3. check: every line the planner did not change must get the identical prediction as RAG F
# Usage: LCA_USD_PER_HOUR=0.563 setsid nohup bash run_E5_dev_v2.sh > /dev/null 2>&1 &
# Re-run to resume: finished arms are skipped.
source $(dirname "$0")/box_lib.sh
LOG=$HOME/lca_box_E5dev_v2.log
export LCA_USD_PER_HOUR=${LCA_USD_PER_HOUR:-}
SCRIPTS=$PROJECT/rag_results/scripts
DTYPE=${PLANNER_DTYPE:-8bit}
PLANNER=Qwen/Qwen2.5-Coder-7B-Instruct

anchor_check   # hashes must equal 8fe8932f... (zero) and 6941beca... (v1), as on every earlier machine

if [ "$DTYPE" != "bfloat16" ]; then
  (cd ~/ts_test && $PY -c "import bitsandbytes" > /dev/null 2>&1) || (cd $PROJECT && poetry run pip install bitsandbytes >> $LOG 2>&1)
  (cd ~/ts_test && $PY -c "import bitsandbytes; print('bitsandbytes', bitsandbytes.__version__)" 2>&1 | tee -a $LOG) \
    || { echo "bitsandbytes unusable" | tee -a $LOG; exit 1; }
fi

M8="params=codellama7b dataset=medium split=dev dev_size=8 dev_extra=0 lines=all"
F="rag=F rag.prefix_cap=15900 context_cap=1753"
run_once e5dev2_rag   $M8 arm=rag $F
run_once e5dev2_agent $M8 arm=agent agent=append $F agent.planner_model=$PLANNER agent.planner_dtype=$DTYPE
ARMS=$ART/python/codellama-7b/medium_context/arms
(cd ~/ts_test && PYTHONPATH=$PROJECT $PY $SCRIPTS/planner_probe.py check $ARMS/e5dev2_rag $ARMS/e5dev2_agent 2>&1 | grep -v MatMul8bitLt | tee -a $LOG)
mark "MACHINE E5dev2 complete"
