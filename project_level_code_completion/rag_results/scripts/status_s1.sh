#!/usr/bin/env bash
date +%T
echo "--- S1 markers ---"
grep -E "=====" ~/lca_s1.log | sed 's/overrides.*//'
echo "--- tracebacks: $(grep -c Traceback ~/lca_s1.log)"
echo "--- processes ---"
pgrep -af "eval.compare_arms|run_s1" | cut -c1-140 || true
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "--- finished S1 arm directories ---"
ls ~/lca_artifacts/python/starcoderbase-1b/medium_context/arms 2>/dev/null | grep '^s1_'
ls ~/lca_artifacts/python/starcoderbase-1b/small_context/arms 2>/dev/null | grep '^s1_'
