#!/usr/bin/env bash
# Phase 1 of RAG_EXPERIMENT_LOG section 8.93: does the 32B planner fit next to CodeLlama-7B, and how fast is it? Runs arm d3
# (agent=v3) on the development split for MINUTES minutes (default 30), samples GPU memory every 20 s, then prints the lines
# done, seconds per line (context + generation), planner tokens and the peak GPU memory. The arm folder is e5v3_speed (partial,
# not a result). Needs a GPU with >= 48 GB.
# Usage: LCA_USD_PER_HOUR=<price> bash run_E5v3_speed.sh [MINUTES]
source $(dirname "$0")/box_lib.sh
MINUTES=${1:-30}
LOG=$HOME/lca_box_E5v3speed.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "${GPU_MB:-0}" -ge 45000 ] || { echo "needs a GPU with >= 48 GB (this one has ${GPU_MB} MB)" | tee -a $LOG; exit 1; }
M="params=codellama7b dataset=medium split=dev dev_size=8 dev_extra=16 lines=all"
F="rag=F rag.prefix_cap=15900 context_cap=1753"
MEM=$HOME/lca_box_E5v3speed_gpu.csv
rm -f $MEM
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $MEM; sleep 20; done ) &
SAMPLER=$!
echo "===== speed test start $(date +%T), $MINUTES minutes =====" | tee -a $LOG
(cd ~/ts_test && PYTHONPATH=$PROJECT timeout ${MINUTES}m $PY -m eval.compare_arms arm_name=e5v3_speed artifacts_dir=$ART overwrite=true $M arm=agent agent=v3 $F >> $LOG 2>&1)
kill $SAMPLER 2> /dev/null
python3 - <<PY | tee -a $LOG
import json, os
f = os.path.expanduser("$ART/python/codellama-7b/medium_context/arms/e5v3_speed/generation_results.jsonl")
rows = [json.loads(l) for l in open(f)] if os.path.exists(f) else []
mem = [int(x) for x in open(os.path.expanduser("$MEM")).read().split()]
print("lines done:", len(rows))
if rows:
    secs = [(r["metrics"]["context_ms"] + r["metrics"]["gen_ms"]) / 1000 for r in rows]
    print("seconds per line: mean %.1f, median %.1f, max %.1f" % (sum(secs) / len(secs), sorted(secs)[len(secs) // 2], max(secs)))
    print("planner steps per line %.2f, tool calls %.2f" % (sum(r["retrieval"]["steps"] for r in rows) / len(rows), sum(r["retrieval"]["tool_calls"] for r in rows) / len(rows)))
    print("planner tokens per line (in+out) %.0f" % (sum(r["retrieval"]["planner_input_tokens"] + r["retrieval"]["planner_output_tokens"] for r in rows) / len(rows)))
    print("changed share %.1f%%, fallback %d" % (100 * sum(r["retrieval"]["changed"] for r in rows) / len(rows), sum(r["retrieval"]["fallback"] for r in rows)))
    print("full dev arm (1372 lines) would take about %.1f hours at this speed" % (1372 * sum(secs) / len(secs) / 3600))
print("peak GPU memory used: %.1f GB of %.1f GB" % (max(mem) / 1000, $GPU_MB / 1000) if mem else "no memory samples")
PY
mark "MACHINE E5v3speed complete"
