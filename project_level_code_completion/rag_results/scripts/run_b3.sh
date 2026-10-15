#!/usr/bin/env bash
source $(dirname "$0")/common.sh
LOG=$HOME/lca_b3.log; : > $LOG
SMALL="split=dev dev_size=20 dev_extra=28 lines=targeted"
MEDIUM="dataset=medium split=dev dev_size=28 dev_extra=28 lines=targeted"
V1="arm=rag rag.prefix_cap=7900"
b3() {  # id, overrides...
  ID=$1; shift
  run_arm $LOG b3_$ID $SMALL  $V1 "$@"
  run_arm $LOG b3_$ID $MEDIUM $V1 "$@"
}
b3 R9a  rag.continuation_lines=20 rag.order=best_first
b3 R5c  rag.continuation_lines=40
b3 R5d  rag.continuation_lines=10
echo "===== B3 complete $(date +%T) =====" | tee -a $LOG
