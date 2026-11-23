#!/usr/bin/env bash
# Resumable version of run_evidence.sh: skips every arm that already has a summary.json. Appends to the same log.
source $(dirname "$0")/common.sh
LOG=$HOME/lca_evidence.log
echo "===== resume $(date +%T) =====" | tee -a $LOG

summary_path() {  # NAME, OVERRIDES...  -> expected summary.json of the arm
  NAME=$1; shift; O="$*"
  case "$O" in *codellama7b*) M=codellama-7b;; *deepseekcoder1b*) M=deepseek-coder-1b;; *) M=starcoderbase-1b;; esac
  case "$O" in *dataset=medium*) D=medium_context;; *) D=small_context;; esac
  echo "$ART/python/$M/$D/arms/$NAME/summary.json"
}
run_once() {  # NAME, OVERRIDES...
  NAME=$1; shift
  if [ -f "$(summary_path $NAME "$@")" ]; then echo "----- $NAME already done, skipped -----" | tee -a $LOG; return 0; fi
  run_arm $LOG $NAME "$@"
}

F16="arm=rag rag=F rag.prefix_cap=15900"
MED_DEV="dataset=medium split=dev dev_size=28 dev_extra=28 lines=all"
MED_HELD="dataset=medium split=heldout dev_size=28 dev_extra=28"
CL="params=codellama7b"; DS="params=deepseekcoder1b"

run_once s3a_cl7b_zero $CL $MED_DEV arm=zero
run_once s3a_cl7b_pd   $CL $MED_DEV arm=path_distance static_mode=provider
run_once s3a_cl7b_F16  $CL $MED_DEV $F16
grep -q "S3a complete" $LOG || echo "===== S3a complete $(date +%T) =====" | tee -a $LOG

run_once s4_ds1b_zero $DS $MED_HELD lines=all arm=zero
run_once s4_ds1b_pd   $DS $MED_HELD lines=all arm=path_distance static_mode=provider
run_once s4_ds1b_F16  $DS $MED_HELD lines=all $F16
grep -q "S4 complete" $LOG || echo "===== S4 complete $(date +%T) =====" | tee -a $LOG

run_once s3b_cl7b_zero $CL $MED_HELD lines=targeted arm=zero
run_once s3b_cl7b_pd   $CL $MED_HELD lines=targeted arm=path_distance static_mode=provider
run_once s3b_cl7b_F16  $CL $MED_HELD lines=targeted $F16
grep -q "S3b complete" $LOG || echo "===== S3b complete $(date +%T) =====" | tee -a $LOG

for C in naive alphabetical file_length half_memory half_memory_path_distance function_class_mask_half function_class_mask_one; do
  run_once hall_$C $MED_HELD lines=all arm=$C static_mode=provider
done
for C in naive alphabetical file_length half_memory half_memory_path_distance function_class_mask_half function_class_mask_one; do
  run_once hall_$C split=heldout lines=all arm=$C static_mode=provider
done
grep -q "H-all complete" $LOG || echo "===== H-all complete $(date +%T) =====" | tee -a $LOG
