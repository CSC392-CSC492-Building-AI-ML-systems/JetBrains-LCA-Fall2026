#!/usr/bin/env bash
# E2/E3 gets priority (user instruction 2026-10-14). Waits for the arm that is running (s4_ds1b_pd) to finish, stops the old
# chain before it starts s4_ds1b_F16, then runs run_e23.sh (which resumes S4 and the v2 chain afterwards).
D=$(dirname "$0")
pkill -f switch_chain.sh
until grep -q "s4_ds1b_pd exit" ~/lca_evidence.log; do sleep 10; done
pkill -f run_evidence.sh
sleep 1
pkill -f "eval.compare_arms"
sleep 10
rm -rf ~/lca_artifacts/python/deepseek-coder-1b/medium_context/arms/s4_ds1b_F16   # partial output, if any
echo "===== old chain stopped after s4_ds1b_pd; E2/E3 first (user priority) $(date +%T) =====" >> ~/lca_evidence.log
setsid nohup bash $D/run_e23.sh > /dev/null 2>&1 < /dev/null &
