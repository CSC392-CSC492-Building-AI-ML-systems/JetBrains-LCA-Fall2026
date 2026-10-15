#!/usr/bin/env bash
source $(dirname "$0")/common.sh
wait_gpu 8000 1800 || exit 1
export PYTHONPATH=$PROJECT
cd ~/ts_test
$PY $(dirname "$0")/smoke_dense_real.py "$@" 2>&1 | grep -v -i -E "warn|Loading weights|^$" | tail -12
