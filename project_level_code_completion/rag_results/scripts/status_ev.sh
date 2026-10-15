#!/usr/bin/env bash
date +%T
grep -E "=====|-----" ~/lca_evidence.log | sed 's/overrides.*//'
echo "tracebacks: $(grep -c Traceback ~/lca_evidence.log)"
echo "chain running: $(pgrep -f run_evidence | wc -l) processes"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
