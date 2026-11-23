#!/usr/bin/env bash
# Tail: HUGE (Table 9) and, last, H-all. Run on whichever machine (A, B or C) becomes free first, on a machine that is
# otherwise done (HUGE compares only its own three arms, so it may run on any machine).
source $(dirname "$0")/box_lib.sh
LOG=$HOME/lca_box_tail.log
three p9_huge_cl7b $CL dataset=huge split=all
mark "HUGE complete"
# H-all (E4-adjacent, optional): remaining composers on StarCoder-1B at 8K
for C in naive alphabetical file_length half_memory half_memory_path_distance function_class_mask_half function_class_mask_one; do
  run_once hall_$C $MED_HELD lines=all arm=$C static_mode=provider
done
for C in naive alphabetical file_length half_memory half_memory_path_distance function_class_mask_half function_class_mask_one; do
  run_once hall_small_$C split=heldout lines=all arm=$C static_mode=provider
done
mark "H-all complete"
