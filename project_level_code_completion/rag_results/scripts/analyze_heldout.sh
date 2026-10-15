#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""
export PROJECT=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
PY=~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python
export PYTHONPATH=$PROJECT PYTHONDONTWRITEBYTECODE=1
A=$HOME/lca_artifacts/python
S=$A/starcoderbase-1b/small_context/arms
M=$A/starcoderbase-1b/medium_context/arms
M7=$A/starcoderbase-7b/medium_context/arms
R=$HOME/lca_artifacts/reports/heldout; mkdir -p $R ~/empty_cwd; cd ~/empty_cwd

echo "=================== PROVENANCE ==================="
for spec in "$S:small 1B" "$M:medium 1B" "$M7:medium 7B"; do
  D=${spec%%:*}; NAME=${spec#*:}
  for a in $(ls $D | grep -E '^h7?_'); do
    $PY -c "import json; s=json.load(open('$D/$a/summary.json')); c=s['code']; print('$NAME'.ljust(10), '$a'.ljust(18), 'commit', c['commit'], 'dirty', c['dirty'], '| split', s['split'], 'lines=%s' % s['lines'], '| datapoints/lines', sum(s['line_counts'].values()), '| wall', round(s['wall_seconds']/60,1), 'min | model', s['model'])"
  done
done

echo; echo "=================== TARGETS (held-out T lines; snapshot only) ==================="
$PY -m eval.make_targets --dataset small  --split heldout --dev-size 20 --dev-extra 28 --out $R/targets_small_heldout.json  2>&1 | grep -E "targeted"
$PY -m eval.make_targets --dataset medium --split heldout --dev-size 28 --dev-extra 28 --out $R/targets_medium_heldout.json 2>&1 | grep -E "targeted"

show() {  # title, then analyze_arms args; prints the EM table and the ES all-row
  TITLE=$1; OUT=$2; shift 2
  echo; echo "=================== $TITLE ==================="
  $PY -m eval.analyze_arms "$@" --out $R/$OUT.md > /dev/null
  sed -n '1p' $R/$OUT.md | cut -c1-160
  sed -n '/### EM/,/### ES/p' $R/$OUT.md | sed '$d'
  grep -E '^\| all ' $R/$OUT.md | tail -1 | sed 's/^/ES  /'
}
ARMS="h_zero h_imports_first h_v1 h_F"
show "P1  MEDIUM held-out, ALL lines, ref path_distance"  p1_medium_all   --base $M --ref h_path_distance --arms $ARMS
show "P3  SMALL held-out, ALL lines, ref path_distance"   p3_small_all    --base $S --ref h_path_distance --arms $ARMS
show "P2  T lines, POOLED small+medium held-out, ref path_distance" p2_T_pooled --base $S $M --targets $R/targets_small_heldout.json $R/targets_medium_heldout.json --ref h_path_distance --arms $ARMS
show "T lines, MEDIUM held-out only" T_medium --base $M --targets $R/targets_medium_heldout.json --ref h_path_distance --arms $ARMS
show "T lines, SMALL held-out only"  T_small  --base $S --targets $R/targets_small_heldout.json --ref h_path_distance --arms $ARMS
show "F - v1, MEDIUM held-out all lines" f_vs_v1_medium --base $M --ref h_v1 --arms h_F
show "F - v1, SMALL held-out all lines"  f_vs_v1_small  --base $S --ref h_v1 --arms h_F
show "F - v1, T lines pooled" f_vs_v1_T --base $S $M --targets $R/targets_small_heldout.json $R/targets_medium_heldout.json --ref h_v1 --arms h_F
show "7B, medium held-out T lines (ref path_distance)" b7_T --base $M7 --ref h7_path_distance --arms h7_zero h7_F
