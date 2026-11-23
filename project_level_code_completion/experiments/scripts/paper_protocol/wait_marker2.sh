#!/usr/bin/env bash
# usage: wait_marker2.sh "V3 done"  -> exits when the marker is logged, or when no evidence chain / switch controller is alive.
until grep -q "$1" ~/lca_evidence.log || { ! pgrep -f "run_evidence" > /dev/null && ! pgrep -f "switch_chain.sh" > /dev/null; }; do sleep 60; done
grep -E "=====|-----" ~/lca_evidence.log | tail -4 | sed 's/overrides.*//'
