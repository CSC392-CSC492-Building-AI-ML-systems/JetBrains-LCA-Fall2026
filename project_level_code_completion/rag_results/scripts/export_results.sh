#!/usr/bin/env bash
# Copy the small result artifacts (reports, targets, per-arm summary.json + config.json) into the repository.
P="${PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
A=$HOME/lca_artifacts
DEST=$P/rag_results
mkdir -p "$DEST/reports"
cp "$A"/reports/*.md "$DEST/reports/" 2>/dev/null
cp -r "$A/reports/heldout" "$DEST/reports/" 2>/dev/null
cp "$A/reports/targets_small_dev.json" "$DEST/reports/" 2>/dev/null
for model in starcoderbase-1b starcoderbase-7b; do
  for ds in small_context medium_context; do
    SRC=$A/python/$model/$ds/arms
    [ -d "$SRC" ] || continue
    for arm_dir in "$SRC"/*; do
      [ -d "$arm_dir" ] || continue
      arm=${arm_dir##*/}
      for f in summary.json config.json; do
        [ -f "$arm_dir/$f" ] || continue
        mkdir -p "$DEST/arms/$model/$ds/$arm"
        cp "$arm_dir/$f" "$DEST/arms/$model/$ds/$arm/"
      done
    done
  done
done
echo "files: $(find "$DEST" -type f | wc -l)  size: $(du -sh "$DEST" | cut -f1)"
echo "largest files:"; find "$DEST" -type f -printf '%s %p\n' | sort -rn | head -4 | awk '{size=$1; sub(/^[^ ]+ /,""); printf "  %.0f KB  %s\n", size/1024, $0}' | sed "s#$P/##"
echo "arms exported: $(find "$DEST/arms" -name summary.json | wc -l)"
