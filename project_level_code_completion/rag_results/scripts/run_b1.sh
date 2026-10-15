#!/usr/bin/env bash
source $(dirname "$0")/common.sh
LOG=$HOME/lca_b1.log; : > $LOG
R0T=$HOME/lca_r0t.log

# Wait for R0-T to finish (both v1 runs logged), then require every R0-T run to have exited 0.
while [ "$(grep -c 'r0t_v1 exit=' $R0T)" -lt 2 ]; do sleep 30; done
if grep -E '=====.*exit=[1-9]' $R0T > /dev/null || grep -c 'exit=0' $R0T | grep -qv '^8$'; then
  echo "R0-T did not finish cleanly (need 8 runs with exit=0); B1 not started" | tee -a $LOG; exit 1
fi

SMALL="split=dev dev_size=20 dev_extra=28 lines=targeted"
MEDIUM="dataset=medium split=dev dev_size=28 dev_extra=28 lines=targeted"
V1="arm=rag rag.prefix_cap=7900"
b1() {  # id, overrides...
  ID=$1; shift
  run_arm $LOG b1_$ID $SMALL  $V1 "$@"
  run_arm $LOG b1_$ID $MEDIUM $V1 "$@"
}
b1 v1repro                       # validation: final code reproduces r0t_v1 (not a design run)
b1 R4a  rag.definitions_first=true
b1 R4b  rag.definitions_first=true rag.definition_imports=false
b1 R2b  rag.proximity_weight=0.5
b1 R2a  rag.proximity_weight=0.25
b1 R2c  rag.proximity_weight=1.0
b1 R1b  rag.max_context_tokens=2000
b1 R1c  rag.max_context_tokens=4000
b1 R1a  rag.max_context_tokens=1000
b1 R3a  rag.import_weight=0.5
b1 R3b  rag.import_weight=1.0
b1 R8   rag.order=best_first
echo "===== B1 complete $(date +%T) =====" | tee -a $LOG
