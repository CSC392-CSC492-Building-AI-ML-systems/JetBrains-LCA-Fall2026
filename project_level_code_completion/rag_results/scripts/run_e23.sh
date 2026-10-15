#!/usr/bin/env bash
# E2/E3 controlled ablation (RAG_EXPERIMENT_LOG.md section 8.40): StarCoder-1B, medium_context, ALL 224 datapoints, ALL lines.
# Base = v1 (plain BM25, structure chunks, 30 chunks, 7,900-token budget). Each arm changes ONE variable. Skips finished arms.
# Afterwards: finishes the S4 step that this script pre-empted, then starts the v2 (paper protocol) chain.
source $(dirname "$0")/common.sh
LOG=$HOME/lca_evidence.log
D=$(dirname "$0")
echo "===== E2/E3 chain start $(date +%T) =====" | tee -a $LOG

summary_path() {  # NAME, OVERRIDES...
  NAME=$1; shift; O="$*"
  case "$O" in *codellama7b*) M=codellama-7b;; *deepseekcoder1b*) M=deepseek-coder-1b;; *) M=starcoderbase-1b;; esac
  case "$O" in *dataset=medium*) DS=medium_context;; *dataset=large*) DS=large_context;; *dataset=huge*) DS=huge_context;; *) DS=small_context;; esac
  echo "$ART/python/$M/$DS/arms/$NAME/summary.json"
}
run_once() {
  NAME=$1; shift
  if [ -f "$(summary_path $NAME "$@")" ]; then echo "----- $NAME already done, skipped -----" | tee -a $LOG; return 0; fi
  run_arm $LOG $NAME "$@"
}

ALL="dataset=medium split=all lines=all"
V1="arm=rag rag=v1"

# Baselines on exactly the same data
run_once e23_zero $ALL arm=zero
run_once e23_pd   $ALL arm=path_distance static_mode=provider
# Base
run_once e23_v1   $ALL $V1
# E3: how many chunks (top-k); base k = 30
for K in 10 5 3 1 60; do
  run_once e23_k$K $ALL $V1 rag.max_chunks=$K
done
# E2: chunking granularity; base = structure chunks (function / class header / module header; 40/20 windows elsewhere)
run_once e23_fine10 $ALL $V1 rag.chunking=fine rag.fine_window=10 rag.fine_stride=5
run_once e23_fine20 $ALL $V1 rag.chunking=fine rag.fine_window=20 rag.fine_stride=10
run_once e23_fine40 $ALL $V1 rag.chunking=fine rag.fine_window=40 rag.fine_stride=20
run_once e23_file   $ALL $V1 rag.chunking=fine rag.fine_window=100000 rag.fine_stride=100000
# Final design F (v1 + 20 continuation lines) on the same data, for the link to the earlier results
run_once e23_F    $ALL arm=rag rag=F
echo "===== E2/E3 complete $(date +%T) =====" | tee -a $LOG

# The S4 step that this chain pre-empted (DeepSeek-1.3B, 16K, medium held-out)
run_once s4_ds1b_F16 params=deepseekcoder1b dataset=medium split=heldout dev_size=28 dev_extra=28 lines=all arm=rag rag=F rag.prefix_cap=15900
echo "===== S4 complete $(date +%T) =====" | tee -a $LOG

setsid nohup bash $D/run_evidence_v2.sh > /dev/null 2>&1 < /dev/null &
