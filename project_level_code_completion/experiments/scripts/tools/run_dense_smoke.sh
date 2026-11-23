#!/usr/bin/env bash
D=$(dirname "$0")
for spec in "jinaai/jina-embeddings-v2-base-code cls" "microsoft/unixcoder-base mean" "Alibaba-NLP/gte-modernbert-base cls" "BAAI/bge-small-en-v1.5 cls"; do
  echo "################ $spec"
  bash $D/run_py.sh smoke_dense.py $spec 2>&1 | tail -4
done
