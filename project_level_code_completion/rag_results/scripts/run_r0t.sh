#!/usr/bin/env bash
source $(dirname "$0")/common.sh
LOG=$HOME/lca_r0t.log; : > $LOG
SMALL="split=dev dev_size=20 dev_extra=28 lines=targeted"
MEDIUM="dataset=medium split=dev dev_size=28 dev_extra=28 lines=targeted"
for SET in small medium; do
  if [ $SET = small ]; then OPTS="$SMALL"; else OPTS="$MEDIUM"; fi
  run_arm $LOG r0t_zero           $OPTS arm=zero
  run_arm $LOG r0t_path_distance  $OPTS arm=path_distance
  run_arm $LOG r0t_imports_first  $OPTS arm=imports_first
  run_arm $LOG r0t_v1             $OPTS arm=rag rag.prefix_cap=7900
done
