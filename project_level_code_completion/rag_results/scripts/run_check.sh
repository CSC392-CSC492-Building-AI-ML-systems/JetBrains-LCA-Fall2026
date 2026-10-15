#!/usr/bin/env bash
cd ~/ts_test || exit 1
echo "grammar ABI:"
grep -m1 "define LANGUAGE_VERSION" tree-sitter-python/src/parser.c
export PYTHONPATH=/mnt/c/Users/Adriana/Documents/School/CSC398/JetBrains-LCA-Fall2026/project_level_code_completion
~/.cache/pypoetry/virtualenvs/lca-L6V1CIw0-py3.11/bin/python verify_env.py
echo "exit=$?"
