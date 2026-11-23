#!/usr/bin/env bash
# Machine A: E2/E3 part 1 (baselines, base v1 and the top-k sweep). CodeLlama-7B, 16K, medium, all 224 datapoints, all lines.
# Log section 8.40/8.41/8.44. Skips finished arms. Run: setsid nohup bash run_A.sh > /dev/null 2>&1 &
source $(dirname "$0")/box_lib.sh
LOG=$HOME/lca_box_A.log
anchor_check
run_once e23_zero $ALL arm=zero
run_once e23_pd   $ALL arm=path_distance static_mode=provider
run_once e23_v1   $ALL $V1
for K in 10 5 3 1 60; do run_once e23_k$K $ALL $V1 rag.max_chunks=$K; done
mark "MACHINE A complete"
