#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
export PYTHONPATH=$PROJECT PYTHONDONTWRITEBYTECODE=1
B=$HOME/lca_artifacts/python/codellama-7b/medium_context/arms
R=$HOME/lca_artifacts/reports/s3a; mkdir -p $R ~/empty_cwd; cd ~/empty_cwd
echo "--- provenance ---"
for a in s3a_cl7b_zero s3a_cl7b_pd s3a_cl7b_F16; do
  $PY -c "import json; s=json.load(open('$B/$a/summary.json')); c=s['code']; print('$a'.ljust(16), 'model', s['model'], 'budget', s['token_budget'], 'lines', sum(s['line_counts'].values()), 'commit', c['commit'], 'dirty', c['dirty'], 'wall', round(s['wall_seconds']/60), 'min')"
done
$PY -m eval.analyze_arms --base $B --ref s3a_cl7b_pd --arms s3a_cl7b_zero s3a_cl7b_F16 --out $R/s3a_vs_pd.md > /dev/null
sed -n '/### EM/,/### ES/p' $R/s3a_vs_pd.md | sed '$d'
grep -E '^\| all ' $R/s3a_vs_pd.md | tail -1 | sed 's/^/ES  /'
