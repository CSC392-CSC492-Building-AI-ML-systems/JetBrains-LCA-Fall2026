#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
export PYTHONPATH=$PROJECT PYTHONDONTWRITEBYTECODE=1
S=$HOME/lca_artifacts/python/starcoderbase-1b/small_context/arms
M=$HOME/lca_artifacts/python/starcoderbase-1b/medium_context/arms
mkdir -p $HOME/lca_artifacts/reports ~/empty_cwd; cd ~/empty_cwd
echo "--- R0-T log ---"; grep -E "=====" $HOME/lca_r0t.log | sed 's/overrides.*//'; grep -c Traceback $HOME/lca_r0t.log
echo; echo "--- wall time and lines per run ---"
for set in small medium; do
  if [ $set = small ]; then D=$S; else D=$M; fi
  for a in r0t_zero r0t_path_distance r0t_imports_first r0t_v1; do
    $PY -c "import json,sys; s=json.load(open('$D/$a/summary.json')); print('$set', '$a', 'lines', sum(s['line_counts'].values()), 'wall', s['wall_seconds'], 's', s['code'])"
  done
done
for spec in "POOLED small+medium:$S $M" "MEDIUM only:$M" "SMALL only:$S"; do
  NAME=${spec%%:*}; BASES=${spec#*:}
  echo; echo "=================== T_dev ($NAME): arms vs path_distance ==================="
  $PY -m eval.analyze_arms --base $BASES --ref r0t_path_distance --arms r0t_zero r0t_imports_first r0t_v1 \
      --out $HOME/lca_artifacts/reports/r0t_$(echo $NAME | cut -d' ' -f1 | tr 'A-Z' 'a-z').md
done
