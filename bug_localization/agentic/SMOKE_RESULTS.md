# Smoke Verification (2026-11-09)

Verification summary for the offline mocked investigation and a single live training-issue smoke run. For the full 50-issue Python test evaluation, see [RESULTS.md](RESULTS.md).

## Local Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s bug_localization/agentic/tests -v
.\.venv\Scripts\python.exe -m unittest discover -s bug_localization/rag/tests -v
.\.venv\Scripts\python.exe -m unittest discover -s bug_localization/long_context/tests -v
.\.venv\Scripts\python.exe -m compileall -q bug_localization/agentic
.\.venv\Scripts\python.exe -m bug_localization.agentic.smoke
```

Results: **6 agentic tests, 90 RAG tests, and 19 long-context tests passed** (115 total). Compilation and the offline smoke test passed. Tests cover derived follow-up queries, base/head and label isolation, invalid paths/ranges, tool-budget exhaustion, disabled-tool final ranking, context/output bounds, full BM25 fallback, partial ranking completion, saved results, and scoring through the existing evaluator.

The offline mocked smoke run writes a trace under `outputs/offline_smoke/runs/<signature>/examples/000.json` demonstrating this evidence-driven sequence:

1. Search `crash`; receive an import of `resolve_alias` in `router.py`.
2. Extract that identifier from the returned source and search for `resolve_alias`.
3. Select `symbols.py` from the returned definition snippet and read its range.
4. Submit `symbols.py` first; validate and append remaining candidates by BM25.

The evaluator reports `AP=0.5` because one of the two synthetic gold files is deliberately excluded by the shared test-directory filter, verifying denominator behavior rather than benchmark quality.

## Live Training Smoke Run

One live training issue was run and scored to verify end-to-end API behavior before running the full test evaluation:

- Dataset: `JetBrains-Research/lca-bug-localization`, `py/train`, original index 621.
- Issue: <https://github.com/pre-commit/pre-commit/issues/1076>.
- Base SHA: `95afd642f52b4d2a12ad605683c8f752e63530d0`.
- Model: `gpt-4.1-2025-04-14`, using [`settings.example.json`](settings.example.json).
- Initial source: 4,026 serialized tokens within the 4,096-token source budget.
- Tool limit: 6; actual tool calls: 5; model requests: 6.
- Outcome: `repaired`, stop reason `submit_ranking`, **no full BM25 fallback**.
- Submitted prefix: `pre_commit/commands/autoupdate.py`, `pre_commit/store.py`.
- Cumulative API usage: 19,925 prompt + 138 completion = **20,063 tokens**.
- Cumulative model-request latency: **8.866 seconds**.
- Single-issue MAP/AP: `0.6875`; MRR and Hit@1/5/10: `1.0`.

The investigation trace recorded a search for `shallow fetch tag`, a follow-up search for `_update_repo` returning `pre_commit/commands/autoupdate.py:21-60`, targeted reads of `store.py` and successive ranges of `autoupdate.py`, and early submission via `submit_ranking`. Actual returned line ranges reflect whole-line output clipping and are saved separately from requested ranges.

Reproduce with:

```powershell
.\.venv\Scripts\python.exe -m bug_localization.agentic.run --config bug_localization/agentic/settings.example.json --split train --start 621 --limit 1 --output-dir bug_localization/agentic/outputs/live_smoke
.\.venv\Scripts\python.exe -m bug_localization.agentic.evaluate bug_localization/agentic/outputs/live_smoke --allow-partial --output-dir bug_localization/agentic/outputs/live_smoke/scoring
```

Generated artifacts live under the gitignored `outputs/` directory. Matched one-shot RAG/direct arms at the same 4,096-token budget are still required to measure relative improvement; this single smoke example verifies implementation behavior only. See [RESULTS.md](RESULTS.md) for the full 50-issue Python test evaluation.
