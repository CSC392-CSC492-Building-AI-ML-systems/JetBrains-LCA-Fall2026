#!/usr/bin/env bash
source $(dirname "$0")/common.sh
export PYTHONPATH=$PROJECT
echo "### real warning without the grammar (cwd has no tree-sitter-python clone)"
mkdir -p ~/empty_cwd; (cd ~/empty_cwd && CUDA_VISIBLE_DEVICES= $PY $(dirname "$0")/demo_warning.py 2>&1 | grep -v -i "^$")
echo; echo "### with the grammar (cwd = ~/ts_test)"
(cd ~/ts_test && CUDA_VISIBLE_DEVICES= $PY $(dirname "$0")/demo_warning.py 2>&1 | grep -v -i "^$")
echo; echo "### reproduction: F via the new preset (rag=F) vs the earlier explicit-override run b2_R5a, targeted dev lines"
LOG=$HOME/lca_cleanup.log; : > $LOG
run_arm $LOG cleanup_F split=dev dev_size=20 dev_extra=28 lines=targeted arm=rag rag=F
run_arm $LOG cleanup_F dataset=medium split=dev dev_size=28 dev_extra=28 lines=targeted arm=rag rag=F
grep -c Traceback $LOG
CUDA_VISIBLE_DEVICES= $PY $(dirname "$0")/check_repro.py b2_R5a cleanup_F 2>&1 | grep -v -i warn
