#!/usr/bin/env bash
# E4 on ONE rented machine (log section 8.72): retrieval against supplying more context directly, following the paper:
# same model, all 224 medium datapoints, all lines; only the approach and the repository-context size N change.
# N = 256, 1753, 12000 (the benchmark pipeline's own ladder) and the full 16K window (no cap). Each arm logs tokens, latency and cost.
# Usage: LCA_USD_PER_HOUR=0.65 setsid nohup bash run_E4.sh > /dev/null 2>&1 &
# Cheapest arms first, so partial results appear early. Skips finished arms (re-run the script to resume).
source $(dirname "$0")/box_lib.sh
LOG=$HOME/lca_box_E4.log
export LCA_USD_PER_HOUR=${LCA_USD_PER_HOUR:-}
anchor_check   # same 5 datapoints as before; the hashes must equal 8fe8932f... (zero) and 6941beca... (v1)

M="params=codellama7b dataset=medium split=all lines=all"
F="arm=rag rag=F rag.prefix_cap=15900"
PD="arm=path_distance static_mode=provider"

run_once e4_zero            $M arm=zero
for N in 256 1753 12000; do
  run_once e4_pd_N$N        $M $PD context_cap=$N
  run_once e4_F_N$N         $M $F context_cap=$N
done
run_once e4_pd_16k          $M $PD
run_once e4_F_16k           $M $F
mark "MACHINE E4 complete"
