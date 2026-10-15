#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
export PYTHONPATH=$PROJECT PYTHONDONTWRITEBYTECODE=1
S=$HOME/lca_artifacts/python/starcoderbase-1b/small_context/arms
M=$HOME/lca_artifacts/python/starcoderbase-1b/medium_context/arms
mkdir -p ~/empty_cwd $HOME/lca_artifacts/reports; cd ~/empty_cwd
echo "--- guardrail log ---"; grep -E "=====" $HOME/lca_guardrail.log | sed 's/overrides.*//'; echo "tracebacks: $(grep -c Traceback $HOME/lca_guardrail.log)"
for spec in "small_context dev ref r0_path_distance:r0_v1,g_F" "medium_context dev ref g_path_distance:g_v1,g_F"; do :; done
echo; echo "=========== SMALL dev, all lines: ref path_distance ==========="
$PY -m eval.analyze_arms --base $S --ref r0_path_distance --arms r0_v1 g_F --out $HOME/lca_artifacts/reports/guard_small_vs_pd.md | sed -n '1,13p'
echo; echo "=========== SMALL dev, all lines: ref v1 ==========="
$PY -m eval.analyze_arms --base $S --ref r0_v1 --arms g_F --out $HOME/lca_artifacts/reports/guard_small_vs_v1.md | sed -n '3,13p'
echo; echo "=========== MEDIUM dev, all lines: ref path_distance ==========="
$PY -m eval.analyze_arms --base $M --ref g_path_distance --arms g_v1 g_F --out $HOME/lca_artifacts/reports/guard_medium_vs_pd.md | sed -n '1,13p'
echo; echo "=========== MEDIUM dev, all lines: ref v1 ==========="
$PY -m eval.analyze_arms --base $M --ref g_v1 --arms g_F --out $HOME/lca_artifacts/reports/guard_medium_vs_v1.md | sed -n '3,13p'
echo; echo "--- ES, all lines, medium: F vs path_distance ---"
sed -n '/### ES/,/^$/p' $HOME/lca_artifacts/reports/guard_medium_vs_pd.md | sed -n '1,5p'
echo; for a in g_path_distance g_v1 g_F; do $PY -c "import json; s=json.load(open('$M/$a/summary.json')); print('medium', '$a', 'lines', sum(s['line_counts'].values()), 'wall', s['wall_seconds'], 's', s['code'])"; done
$PY -c "import json; s=json.load(open('$S/g_F/summary.json')); print('small g_F lines', sum(s['line_counts'].values()), 'wall', s['wall_seconds'], 's', s['code'])"
