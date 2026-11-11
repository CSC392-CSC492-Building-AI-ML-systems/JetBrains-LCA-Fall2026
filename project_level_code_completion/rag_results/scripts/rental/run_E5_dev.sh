#!/usr/bin/env bash
# E5 development prototype on ONE rented machine (log sections 8.80-8.82). Development datapoints only, about 1-1.5 GPU hours.
#   1. anchor check (same 5 datapoints as every machine) - proves the environment reproduces the earlier results
#   2. planner selection: two local candidates, planner only (no CodeLlama, no accuracy), pre-registered rule
#   3. end to end on 4 development datapoints (0, 56, 112, 168): RAG F at N=1753 versus the agent with the chosen planner;
#      every line the planner did not change must get the identical prediction as RAG F
# Usage: LCA_USD_PER_HOUR=0.69 setsid nohup bash run_E5_dev.sh > /dev/null 2>&1 &
# Re-run to resume: finished arms are skipped, finished probes are not repeated.
source $(dirname "$0")/box_lib.sh
LOG=$HOME/lca_box_E5dev.log
export LCA_USD_PER_HOUR=${LCA_USD_PER_HOUR:-}
SCRIPTS=$PROJECT/rag_results/scripts
OUT=$HOME/lca_artifacts/e5_dev
mkdir -p $OUT
DTYPE=${PLANNER_DTYPE:-8bit}   # 8bit leaves room next to CodeLlama on 32 GB; set PLANNER_DTYPE=bfloat16 only on a bigger card
CAND_A=Qwen/Qwen2.5-Coder-7B-Instruct
CAND_B=codellama/CodeLlama-7b-Instruct-hf

anchor_check   # hashes must equal 8fe8932f... (zero) and 6941beca... (v1), as on every earlier machine

if [ "$DTYPE" != "bfloat16" ]; then
  (cd ~/ts_test && $PY -c "import bitsandbytes" > /dev/null 2>&1) || (cd $PROJECT && poetry run pip install bitsandbytes >> $LOG 2>&1)
  (cd ~/ts_test && $PY -c "import bitsandbytes; print('bitsandbytes', bitsandbytes.__version__)" 2>&1 | tee -a $LOG) \
    || { echo "bitsandbytes unusable; rerun with PLANNER_DTYPE=bfloat16 (planner alone) or decide on another option" | tee -a $LOG; exit 1; }
fi

probe() {  # NAME MODEL
  if [ -f $OUT/$1/probe_summary.json ]; then echo "----- probe $1 already done -----" | tee -a $LOG; return; fi
  echo "===== probe $1 ($2, $DTYPE) start $(date +%T) =====" | tee -a $LOG
  (cd ~/ts_test && PYTHONPATH=$PROJECT $PY $SCRIPTS/planner_probe.py probe --planner $2 --dtype $DTYPE --out $OUT/$1 >> $LOG 2>&1)
  echo "===== probe $1 exit=$? end $(date +%T) =====" | tee -a $LOG
}
probe qwen $CAND_A
probe codellama_instruct $CAND_B

CHOICE=$(cd ~/ts_test && PYTHONPATH=$PROJECT $PY $SCRIPTS/planner_probe.py choose $OUT/qwen $OUT/codellama_instruct | tee -a $LOG | sed -n 's/^CHOSEN \([^ ]*\) because.*/\1/p')
echo "planner chosen: $CHOICE" | tee -a $LOG
[ -n "$CHOICE" ] || { echo "no planner chosen (a probe failed); see $LOG" | tee -a $LOG; exit 1; }

M4="params=codellama7b dataset=medium split=dev dev_size=4 dev_extra=0 lines=all"
F="rag=F rag.prefix_cap=15900 context_cap=1753"
run_once e5dev_rag   $M4 arm=rag $F
run_once e5dev_agent $M4 arm=agent $F agent.planner_model=$CHOICE agent.planner_dtype=$DTYPE
ARMS=$ART/python/codellama-7b/medium_context/arms
(cd ~/ts_test && PYTHONPATH=$PROJECT $PY $SCRIPTS/planner_probe.py check $ARMS/e5dev_rag $ARMS/e5dev_agent 2>&1 | tee -a $LOG)
mark "MACHINE E5dev complete"
