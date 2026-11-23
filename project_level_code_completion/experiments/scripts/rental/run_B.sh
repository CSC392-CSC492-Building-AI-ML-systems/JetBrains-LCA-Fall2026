#!/usr/bin/env bash
# Machine B: E2/E3 part 2 (chunking sweep and F). Same model/data/settings as machine A; the anchor check (5 datapoints) must give
# the same predictions-hash as machine A before B's arms are compared with A's (log section 8.44).
source $(dirname "$0")/box_lib.sh
LOG=$HOME/lca_box_B.log
anchor_check
run_once e23_fine10 $ALL $V1 rag.chunking=fine rag.fine_window=10 rag.fine_stride=5
run_once e23_fine20 $ALL $V1 rag.chunking=fine rag.fine_window=20 rag.fine_stride=10
run_once e23_fine40 $ALL $V1 rag.chunking=fine rag.fine_window=40 rag.fine_stride=20
run_once e23_file   $ALL $V1 rag.chunking=fine rag.fine_window=100000 rag.fine_stride=100000
run_once e23_F      $ALL arm=rag rag=F rag.prefix_cap=15900
mark "MACHINE B complete"
