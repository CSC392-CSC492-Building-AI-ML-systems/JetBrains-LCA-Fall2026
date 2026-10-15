#!/usr/bin/env bash
date +%T
echo "--- run markers ---"
grep -E "=====|REFUS|held-out evaluation" ~/lca_heldout.log | sed 's/overrides.*//'
echo "--- tracebacks: $(grep -c Traceback ~/lca_heldout.log)"
echo "--- last error-looking lines ---"
grep -E "Error|error:|Exception" ~/lca_heldout.log | tail -5
echo "--- processes ---"
pgrep -af "eval.compare_arms|run_heldout" | cut -c1-150
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
