#!/usr/bin/env bash
# Machine C: the paper-protocol chain (log section 8.39), minus HUGE and H-all (see run_tail.sh).
source $(dirname "$0")/box_lib.sh
LOG=$HOME/lca_box_C.log
anchor_check
run_once v3_cl7b_pd_preprocess $CL dataset=medium limit_datapoints=5 split=all lines=all arm=path_distance static_mode=preprocess
mark "V3 done"
three s3c_cl7b $CL $MED_HELD
mark "S3c complete"
run_once s4_ds1b_F16 $DS1 $MED_HELD lines=all $F16
mark "S4 F16 done"
three p9_small_cl7b $CL split=all
mark "SMALL complete"
three p9_large_cl7b $CL dataset=large split=all
mark "LARGE complete"
three s4dev_ds1b $DS1 $MED_DEV
mark "DS-dev complete"
# T10: the repository's own perplexity pipeline (Stage 1), unmodified
if [ ! -f $ART/t10/python/codellama-7b/medium_context/results/completion_results.json ]; then
  echo "===== t10 start $(date +%T) =====" | tee -a $LOG
  (cd ~/ts_test && PYTHONPATH=$PROJECT $PY -m eval.eval_pipeline dataset=medium params=codellama7b do_generation=False artifacts_dir=$ART/t10 >> $LOG 2>&1)
  echo "===== t10 exit=$? end $(date +%T) =====" | tee -a $LOG
fi
mark "MACHINE C complete"
