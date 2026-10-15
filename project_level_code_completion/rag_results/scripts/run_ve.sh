#!/usr/bin/env bash
source $(dirname "$0")/common.sh
LOG=$HOME/lca_ve.log; : > $LOG
SMALL="split=dev dev_size=20 dev_extra=28 lines=targeted"
MEDIUM="dataset=medium split=dev dev_size=28 dev_extra=28 lines=targeted"
ve() {  # id, overrides...
  ID=$1; shift
  run_arm $LOG ve_$ID $SMALL  arm=rag rag=F "$@"
  run_arm $LOG ve_$ID $MEDIUM arm=rag rag=F "$@"
}
ve F_repro                                 # validation: the control code leaves F unchanged
ve C1_random     rag.retriever=random      # random chunks, same packing and continuation lines as F
ve C2_proximity  rag.retriever=proximity   # nearest files first, no lexical relevance
ve C3_stale      rag.query_mode=first      # BM25 with the FIRST 10 lines of the file instead of the last 10
echo "===== V-E complete $(date +%T) =====" | tee -a $LOG
