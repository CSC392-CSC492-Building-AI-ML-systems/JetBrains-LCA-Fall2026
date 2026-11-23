#!/usr/bin/env bash
# Final held-out evaluation (RAG_EXPERIMENT_LOG.md section 8.25). usage: run_heldout.sh <expected commit hash>
# Refuses to start unless the code is committed (clean tree) at the expected hash.
source $(dirname "$0")/common.sh
EXPECTED=$1
GIT="git -C $PROJECT"
HEAD_HASH=$($GIT rev-parse --short HEAD)
DIRTY=$($GIT status --porcelain -- . | wc -l)
if [ -z "$EXPECTED" ] || [ "$HEAD_HASH" != "$EXPECTED" ] || [ "$DIRTY" != "0" ]; then
  echo "REFUSING TO RUN: HEAD=$HEAD_HASH expected=$EXPECTED uncommitted files in project=$DIRTY"; exit 1
fi
LOG=$HOME/lca_heldout.log; : > $LOG
echo "held-out evaluation at commit $HEAD_HASH, clean tree, $(date)" | tee -a $LOG

SMALL="split=heldout lines=all"
MEDIUM="dataset=medium split=heldout dev_size=28 dev_extra=28 lines=all"
F="rag.prefix_cap=7900 rag.continuation_lines=20"
V1="rag.prefix_cap=7900"
for SET in medium small; do          # medium first: it carries the primary comparison P1
  if [ $SET = small ]; then OPTS="$SMALL"; else OPTS="$MEDIUM"; fi
  run_arm $LOG h_path_distance $OPTS arm=path_distance
  run_arm $LOG h_F             $OPTS arm=rag $F
  run_arm $LOG h_v1            $OPTS arm=rag $V1
  run_arm $LOG h_zero          $OPTS arm=zero
  run_arm $LOG h_imports_first $OPTS arm=imports_first
done
# 7B on the targeted lines of medium held-out
M7="dataset=medium split=heldout dev_size=28 dev_extra=28 lines=targeted params=starcoderbase7b"
run_arm $LOG h7_path_distance $M7 arm=path_distance
run_arm $LOG h7_F             $M7 arm=rag $F
run_arm $LOG h7_zero          $M7 arm=zero
echo "===== held-out complete $(date +%T) =====" | tee -a $LOG
