#!/usr/bin/env bash
# Evidence chain (RAG_EXPERIMENT_LOG.md section 8.36): S3a -> S4 -> S3b -> H-all. One marker per step.
source $(dirname "$0")/common.sh
LOG=$HOME/lca_evidence.log; : > $LOG
F16="arm=rag rag=F rag.prefix_cap=15900"
MED_DEV="dataset=medium split=dev dev_size=28 dev_extra=28 lines=all"
MED_HELD="dataset=medium split=heldout dev_size=28 dev_extra=28"

# S3a: CodeLlama-7B, 16K, medium dev, all lines
CL="params=codellama7b"
run_arm $LOG s3a_cl7b_zero $CL $MED_DEV arm=zero
run_arm $LOG s3a_cl7b_pd   $CL $MED_DEV arm=path_distance static_mode=provider
run_arm $LOG s3a_cl7b_F16  $CL $MED_DEV $F16
echo "===== S3a complete $(date +%T) =====" | tee -a $LOG

# S4: DeepSeek-Coder-1.3B, 16K, medium held-out, all lines
DS="params=deepseekcoder1b"
run_arm $LOG s4_ds1b_zero $DS $MED_HELD lines=all arm=zero
run_arm $LOG s4_ds1b_pd   $DS $MED_HELD lines=all arm=path_distance static_mode=provider
run_arm $LOG s4_ds1b_F16  $DS $MED_HELD lines=all $F16
echo "===== S4 complete $(date +%T) =====" | tee -a $LOG

# S3b: CodeLlama-7B, 16K, medium held-out, targeted lines
run_arm $LOG s3b_cl7b_zero $CL $MED_HELD lines=targeted arm=zero
run_arm $LOG s3b_cl7b_pd   $CL $MED_HELD lines=targeted arm=path_distance static_mode=provider
run_arm $LOG s3b_cl7b_F16  $CL $MED_HELD lines=targeted $F16
echo "===== S3b complete $(date +%T) =====" | tee -a $LOG

# H-all: the seven composers not yet on held-out (1B, 8K), medium then small
for C in naive alphabetical file_length half_memory half_memory_path_distance function_class_mask_half function_class_mask_one; do
  run_arm $LOG hall_$C $MED_HELD lines=all arm=$C static_mode=provider
done
for C in naive alphabetical file_length half_memory half_memory_path_distance function_class_mask_half function_class_mask_one; do
  run_arm $LOG hall_$C split=heldout lines=all arm=$C static_mode=provider
done
echo "===== H-all complete $(date +%T) =====" | tee -a $LOG
