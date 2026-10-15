#!/usr/bin/env bash
source $(dirname "$0")/common.sh
LOG=$HOME/lca_v1.log; : > $LOG
run_arm $LOG v1_path_distance_provider split=dev arm=path_distance static_mode=provider
