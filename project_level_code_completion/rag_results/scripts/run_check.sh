#!/usr/bin/env bash
export PYTHONPATH="${PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd ~/ts_test || exit 1
echo "grammar ABI:"
grep -m1 "define LANGUAGE_VERSION" tree-sitter-python/src/parser.c
~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python verify_env.py
echo "exit=$?"
