#!/usr/bin/env bash
# usage: wait_marker.sh "S3a complete"   exits when the marker is logged (or the chain died)
until grep -q "$1" ~/lca_evidence.log || ! pgrep -f run_evidence.sh > /dev/null; do sleep 60; done
grep -E "=====" ~/lca_evidence.log | tail -4 | sed 's/overrides.*//'
