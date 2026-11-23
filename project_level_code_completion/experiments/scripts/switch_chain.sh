#!/usr/bin/env bash
# Waits for S4 to complete, stops the old chain before it spends time on S3b, and starts the v2 chain.
D=$(dirname "$0")
until grep -q "S4 complete" ~/lca_evidence.log; do sleep 15; done
pkill -f run_evidence.sh
sleep 1
pkill -f "eval.compare_arms"
sleep 10
rm -rf ~/lca_artifacts/python/codellama-7b/medium_context/arms/s3b_cl7b_zero   # partial output of the superseded S3b, if any
echo "===== old chain stopped after S4; switching to v2 (S3c all lines) $(date +%T) =====" >> ~/lca_evidence.log
setsid nohup bash $D/run_evidence_v2.sh > /dev/null 2>&1 < /dev/null &
