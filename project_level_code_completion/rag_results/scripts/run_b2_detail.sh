#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
export PYTHONPATH=$PROJECT PYTHONDONTWRITEBYTECODE=1
S=$HOME/lca_artifacts/python/starcoderbase-1b/small_context/arms
M=$HOME/lca_artifacts/python/starcoderbase-1b/medium_context/arms
mkdir -p ~/empty_cwd; cd ~/empty_cwd
$PY -m eval.analyze_arms --base $S $M --ref r0t_v1 --arms b2_R5a b1_R8 --out $HOME/lca_artifacts/reports/b2_R5a_R8_vs_v1.md | sed -n '1,14p'
echo; echo "unique T-EM means and exact line-level agreement (v1 vs R5a):"
$PY - <<'EOF'
import json, os
home = os.path.expanduser("~")
def rows(ds, arm):
    return [r for r in map(json.loads, open(f"{home}/lca_artifacts/python/starcoderbase-1b/{ds}/arms/{arm}/generation_results.jsonl")) if not r["zero_context"]]
tot = same = a_only = b_only = 0
for ds in ("small_context", "medium_context"):
    a, b = rows(ds, "r0t_v1"), rows(ds, "b2_R5a")
    for x, y in zip(a, b):
        ex = x["prediction_line"].strip() == x["ground_truth"].strip()
        ey = y["prediction_line"].strip() == y["ground_truth"].strip()
        tot += 1; same += ex == ey; a_only += ex and not ey; b_only += ey and not ex
print(f"lines {tot}: same outcome {same}; v1 right only {a_only}; R5a right only {b_only}")
EOF
