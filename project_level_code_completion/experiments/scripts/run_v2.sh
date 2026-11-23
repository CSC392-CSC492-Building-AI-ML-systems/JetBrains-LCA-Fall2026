#!/usr/bin/env bash
source $(dirname "$0")/common.sh
LOG=$HOME/lca_v2.log; : > $LOG
run_arm $LOG v2_zero_all split=dev dev_size=20 dev_extra=28 lines=all arm=zero
export CUDA_VISIBLE_DEVICES=""
mkdir -p ~/empty_cwd; cd ~/empty_cwd
export PYTHONPATH=$PROJECT
$PY -m eval.make_targets --dataset small --split dev --dev-size 20 --dev-extra 28 --out $ART/reports/targets_small_dev.json 2>&1 | grep -v -i warn | tail -2
$PY - <<'EOF'
import json, os
from eval.analyze_arms import arm_lines
home = os.path.expanduser("~")
S = f"{home}/lca_artifacts/python/starcoderbase-1b/small_context/arms"
targets = json.load(open(f"{home}/lca_artifacts/reports/targets_small_dev.json"))
filtered = arm_lines(S, "v2_zero_all", [targets])           # zero-context pass of the all-lines run, T lines only
reference = arm_lines(S, "r0t_zero")                         # the earlier lines=targeted run
print("lines kept by the filter:", len(filtered), "| lines in r0t_zero:", len(reference))
# datapoint positions differ between the two runs (the targeted run skips datapoints without T lines), so compare
# the multiset of per-line outcomes and the mean EM
em_f = sorted(v["em"] for v in filtered.values()); em_r = sorted(v["em"] for v in reference.values())
print("same multiset of line-level EM outcomes:", em_f == em_r, "| mean EM", round(sum(em_f)/len(em_f), 4), round(sum(em_r)/len(em_r), 4))
es_f = sorted(round(v["es"], 6) for v in filtered.values()); es_r = sorted(round(v["es"], 6) for v in reference.values())
print("same multiset of line-level ES values:", es_f == es_r)
EOF
