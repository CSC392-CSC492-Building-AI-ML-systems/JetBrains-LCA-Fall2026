#!/usr/bin/env bash
# Local tail (log section 8.48): the work not given to the rented machines A/B/C. Runs on the local GPU in WSL. Skips finished arms.
#   1. V3 completion: path_distance through the provider path on the same 5 datapoints as the rented V3 (preprocess path)
#   2. HUGE (paper Table 9): CodeLlama-7B/16K, zero, path_distance, F16 on all 296 datapoints, all lines
#   3. H-all (optional, E4-adjacent): the other static composers on StarCoder-1B at 8K, medium and small held-out
source $(dirname "$0")/common.sh
LOG=$HOME/lca_local_tail.log
echo "===== local tail start $(date +%T) =====" | tee -a $LOG

summary_path() {  # NAME, OVERRIDES...
  NAME=$1; shift; O="$*"
  case "$O" in *codellama7b*) M=codellama-7b;; *deepseekcoder1b*) M=deepseek-coder-1b;; *) M=starcoderbase-1b;; esac
  case "$O" in *dataset=medium*) D=medium_context;; *dataset=large*) D=large_context;; *dataset=huge*) D=huge_context;; *) D=small_context;; esac
  echo "$ART/python/$M/$D/arms/$NAME/summary.json"
}
run_once() {
  NAME=$1; shift
  if [ -f "$(summary_path $NAME "$@")" ]; then echo "----- $NAME already done, skipped -----" | tee -a $LOG; return 0; fi
  run_arm $LOG $NAME "$@"
}
mark() { grep -q "$1" $LOG || echo "===== $1 $(date +%T) =====" | tee -a $LOG; }

F16="arm=rag rag=F rag.prefix_cap=15900"
CL="params=codellama7b"
MED_HELD="dataset=medium split=heldout dev_size=28 dev_extra=28"

run_once v3_cl7b_pd_provider $CL dataset=medium limit_datapoints=5 split=all lines=all arm=path_distance static_mode=provider
mark "V3 provider done"

run_once p9_huge_cl7b_zero $CL dataset=huge split=all lines=all arm=zero
run_once p9_huge_cl7b_pd   $CL dataset=huge split=all lines=all arm=path_distance static_mode=provider
run_once p9_huge_cl7b_F16  $CL dataset=huge split=all lines=all $F16
mark "HUGE complete"

for C in naive alphabetical file_length half_memory half_memory_path_distance function_class_mask_half function_class_mask_one; do
  run_once hall_$C $MED_HELD lines=all arm=$C static_mode=provider
done
for C in naive alphabetical file_length half_memory half_memory_path_distance function_class_mask_half function_class_mask_one; do
  run_once hall_small_$C split=heldout lines=all arm=$C static_mode=provider
done
mark "H-all complete"
