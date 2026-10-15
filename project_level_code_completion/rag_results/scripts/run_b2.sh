#!/usr/bin/env bash
source $(dirname "$0")/common.sh
LOG=$HOME/lca_b2.log; : > $LOG
SMALL="split=dev dev_size=20 dev_extra=28 lines=targeted"
MEDIUM="dataset=medium split=dev dev_size=28 dev_extra=28 lines=targeted"
V1="arm=rag rag.prefix_cap=7900"
b2() {  # id, overrides...
  ID=$1; shift
  run_arm $LOG b2_$ID $SMALL  $V1 "$@"
  run_arm $LOG b2_$ID $MEDIUM $V1 "$@"
}
GTE="rag.embedder=Alibaba-NLP/gte-modernbert-base"
b2 v1repro                                   # validation on the final code (not a design run)
b2 R7b  rag.retriever=hybrid $GTE
b2 R7a  rag.retriever=dense  $GTE
b2 R5a  rag.continuation_lines=20
b2 R5b  rag.chunking=fine rag.fine_window=10 rag.fine_stride=5 rag.continuation_lines=20
b2 R6a  rag.query_lines=3
b2 R6b  rag.query_lines=20
b2 R6c  rag.query_lines=30
b2 R7c  rag.retriever=hybrid rag.embedder=microsoft/unixcoder-base rag.pooling=mean
echo "===== B2 complete $(date +%T) =====" | tee -a $LOG
