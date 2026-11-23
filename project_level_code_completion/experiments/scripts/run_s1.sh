#!/usr/bin/env bash
source $(dirname "$0")/common.sh
LOG=$HOME/lca_s1.log; : > $LOG
SMALL="split=dev dev_size=20 dev_extra=28 lines=all"
MEDIUM="dataset=medium split=dev dev_size=28 dev_extra=28 lines=all"
COMPOSERS="naive alphabetical file_length half_memory half_memory_path_distance function_class_mask_half function_class_mask_one"
# medium dev first: it carries the selection
for C in $COMPOSERS; do run_arm $LOG s1_$C $MEDIUM arm=$C static_mode=provider; done
run_arm $LOG s1_imports_first $MEDIUM arm=imports_first static_mode=provider
run_arm $LOG s1_zero          $MEDIUM arm=zero
for C in $COMPOSERS; do run_arm $LOG s1_$C $SMALL arm=$C static_mode=provider; done
echo "===== S1 complete $(date +%T) =====" | tee -a $LOG
