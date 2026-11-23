#!/usr/bin/env bash
# P0: timing pilot. CodeLlama-7B, 16K window, first 5 medium datapoints (results are NOT used for any claim).
source $(dirname "$0")/common.sh
LOG=$HOME/lca_p0.log; : > $LOG
P="params=codellama7b dataset=medium limit_datapoints=5 split=all lines=all"
run_arm $LOG p0_cl7b_zero $P arm=zero
run_arm $LOG p0_cl7b_pd   $P arm=path_distance
run_arm $LOG p0_cl7b_F16  $P arm=rag rag=F rag.prefix_cap=15900
echo "===== P0 complete $(date +%T) =====" | tee -a $LOG
for a in p0_cl7b_zero p0_cl7b_pd p0_cl7b_F16; do
  python3 - <<EOF
import json,os
d=os.path.expanduser("~/lca_artifacts/python/codellama-7b/medium_context/arms/$a/summary.json")
try:
    s=json.load(open(d)); n=sum(s['line_counts'].values())
    print("$a", "lines", n, "wall", s['wall_seconds'], "s", round(s['wall_seconds']/max(n,1),2), "s/line", "budget", s['token_budget'])
except Exception as e:
    print("$a", "no summary:", e)
EOF
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
