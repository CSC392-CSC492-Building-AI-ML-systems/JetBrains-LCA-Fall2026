#!/usr/bin/env bash
# Exits when the S1 script has logged completion (or its process is gone), so a notification fires.
until grep -q "S1 complete" ~/lca_s1.log || ! pgrep -f run_s1.sh > /dev/null; do sleep 60; done
grep -E "=====" ~/lca_s1.log | tail -3 | sed 's/overrides.*//'
