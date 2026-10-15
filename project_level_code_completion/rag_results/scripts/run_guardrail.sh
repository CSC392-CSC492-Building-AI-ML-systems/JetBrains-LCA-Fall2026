#!/usr/bin/env bash
source $(dirname "$0")/common.sh
LOG=$HOME/lca_guardrail.log; : > $LOG
SMALL="split=dev dev_size=20 dev_extra=28 lines=all"
MEDIUM="dataset=medium split=dev dev_size=28 dev_extra=28 lines=all"
F="arm=rag rag.prefix_cap=7900 rag.continuation_lines=20"
run_arm $LOG g_F            $SMALL  $F
run_arm $LOG g_path_distance $MEDIUM arm=path_distance
run_arm $LOG g_v1            $MEDIUM arm=rag rag.prefix_cap=7900
run_arm $LOG g_F             $MEDIUM $F
echo "===== guardrail complete $(date +%T) =====" | tee -a $LOG
