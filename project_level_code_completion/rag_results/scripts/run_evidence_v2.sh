#!/usr/bin/env bash
# Evidence chain v2 = exact reproduction of the paper's protocol on all data (log section 8.39). Skips finished arms.
source $(dirname "$0")/common.sh
LOG=$HOME/lca_evidence.log
echo "===== v2 chain start $(date +%T) =====" | tee -a $LOG

summary_path() {  # NAME, OVERRIDES...
  NAME=$1; shift; O="$*"
  case "$O" in *codellama7b*) M=codellama-7b;; *deepseekcoder1b*) M=deepseek-coder-1b;; *deepseekcoder7b*) M=deepseek-coder-7b;; *) M=starcoderbase-1b;; esac
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
CL="params=codellama7b"; DS1="params=deepseekcoder1b"
MED_HELD="dataset=medium split=heldout dev_size=28 dev_extra=28"
MED_DEV="dataset=medium split=dev dev_size=28 dev_extra=28"
ALL="split=all lines=all"

three() {  # PREFIX, MODEL_PARAMS, DATASET_AND_SPLIT_OVERRIDES...  -> zero, path_distance, F16 on all lines
  P=$1; MP=$2; shift 2
  run_once ${P}_zero $MP "$@" lines=all arm=zero
  run_once ${P}_pd   $MP "$@" lines=all arm=path_distance static_mode=provider
  run_once ${P}_F16  $MP "$@" lines=all $F16
}

# V3: original preprocessing path vs per-line provider path for the paper's model and window (5 datapoints)
run_once v3_cl7b_pd_preprocess $CL dataset=medium limit_datapoints=5 split=all lines=all arm=path_distance static_mode=preprocess
mark "V3 done"

# S3c: medium held-out, all lines (S3a already did the 56 dev datapoints) -> all 224 medium datapoints
three s3c_cl7b $CL $MED_HELD
mark "S3c complete"

# T10: the repository's own perplexity pipeline, unmodified, all composers, CodeLlama-7B/16K, medium (paper Table 10)
if [ ! -f $ART/t10/python/codellama-7b/medium_context/results/completion_results.json ]; then
  wait_gpu 12000 3600
  echo "===== t10 start $(date +%T) =====" | tee -a $LOG
  (cd ~/ts_test && PYTHONPATH=$PROJECT $PY -m eval.eval_pipeline dataset=medium params=codellama7b do_generation=False artifacts_dir=$ART/t10 >> $LOG 2>&1)
  echo "===== t10 exit=$? end $(date +%T) =====" | tee -a $LOG
fi
mark "T10 complete"

# Table 9, small: all 144 datapoints, all lines
three p9_small_cl7b $CL split=all
mark "SMALL complete"

# DeepSeek-1.3B on the 56 medium dev datapoints, so that S4 covers all 224
three s4dev_ds1b $DS1 $MED_DEV
mark "DS-dev complete"

# Table 9, large and huge: all datapoints, all lines
three p9_large_cl7b $CL dataset=large split=all
mark "LARGE complete"
three p9_huge_cl7b $CL dataset=huge split=all
mark "HUGE complete"

# H-all: remaining composers on held-out (StarCoder-1B, 8K)
for C in naive alphabetical file_length half_memory half_memory_path_distance function_class_mask_half function_class_mask_one; do
  run_once hall_$C $MED_HELD lines=all arm=$C static_mode=provider
done
for C in naive alphabetical file_length half_memory half_memory_path_distance function_class_mask_half function_class_mask_one; do
  run_once hall_$C split=heldout lines=all arm=$C static_mode=provider
done
mark "H-all complete"
