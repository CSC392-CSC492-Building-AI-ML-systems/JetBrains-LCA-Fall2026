# Verification on 2026-11-09

All added files are under `bug_localization/agentic/`; existing experiment source
and entry points were unchanged. No `AGENTS.md` was found in the repository or
its ancestor directories. The repository README and the existing RAG,
long-context and baseline preparation/evaluation code were inspected first.

## Local checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s bug_localization/agentic/tests -v
.\.venv\Scripts\python.exe -m unittest discover -s bug_localization/rag/tests -v
.\.venv\Scripts\python.exe -m unittest discover -s bug_localization/long_context/tests -v
.\.venv\Scripts\python.exe -m compileall -q bug_localization/agentic
.\.venv\Scripts\python.exe -m bug_localization.agentic.smoke
```

Results: **6 agentic tests, 90 RAG tests, and 19 long-context tests passed**.
Compilation and offline smoke passed. Tests cover derived follow-up queries,
base/head and label isolation, invalid paths/ranges, tool-budget exhaustion,
disabled-tool final ranking, context/output bounds, full BM25 fallback, partial
ranking completion, saved results and scoring through the existing evaluator.

The offline trace is saved at
`outputs/offline_smoke/runs/50b51b02f9e34d8180a424e83625c808c4aa8b64788517e93329950260561a6a/examples/000.json`.
It records this evidence-driven mocked sequence:

1. Search `crash`; receive an import of `resolve_alias` in `router.py`.
2. Extract that identifier from the actual returned source and search it.
3. Select `symbols.py` from the returned definition snippet and read its range.
4. Submit `symbols.py` first; validate and append remaining candidates by BM25.

The existing evaluator reports AP=0.5 because one of the two synthetic gold files
is deliberately excluded by the shared test-directory filter. This demonstrates
the intended denominator behavior, not benchmark quality.

## Live training smoke

Credentials were configured, so one live training issue was run and scored:

- Dataset: `JetBrains-Research/lca-bug-localization`, `py/train`, original index 621.
- Issue: <https://github.com/pre-commit/pre-commit/issues/1076>.
- Base SHA: `95afd642f52b4d2a12ad605683c8f752e63530d0`.
- Model: `gpt-4.1-2025-04-14`, using `settings.example.json`.
- Initial source: 4,026 serialized tokens within the 4,096-token source budget.
- Tool limit: six; actual tool calls: five; model requests: six.
- Outcome: `repaired`, stop reason `submit_ranking`, **no full BM25 fallback**.
- Submitted prefix: `pre_commit/commands/autoupdate.py`, `pre_commit/store.py`.
- Cumulative API usage: 19,925 prompt + 138 completion = **20,063 tokens**.
- Cumulative model-request latency: **8.866 seconds**.
- Single-issue MAP/AP: 0.6875; MRR and Hit@1/5/10: 1.0.

The live trace is saved at
`outputs/live_smoke/e806401c4a99c2cd41d74f310fcf285a105c2abcb4f6d03334a015b7e5c44791/examples/621.json`.
It records a search for `shallow fetch tag`, a search for `_update_repo` that
returns `pre_commit/commands/autoupdate.py:21-60`, reads of `store.py` and
successive ranges of `autoupdate.py`, then submission. Actual returned ranges
reflect whole-line output clipping and are recorded separately from requests.
The structured trace contains every exact request, response, tool result,
usage record and latency. Scoring output is in `outputs/live_smoke/scoring/`.

The live command followed the README with `--split train --start 621 --limit 1`
and `--output-dir bug_localization/agentic/outputs/live_smoke`. Run and scoring
both used the existing local Arrow file
`C:/Users/Admin/.cache/huggingface/datasets/JetBrains-Research___lca-bug-localization/py/0.0.0/f2bb26a1c3320ccc1d7b02d2630077ff327a80f6/lca-bug-localization-train.arrow`
via `--dataset-file`. Restricted execution required elevated read access to the
cached Git history and network access for the API; no repository source was
checked out or executed. No extra model setup remains in this environment.

Generated artifacts are retained locally under the ignored `outputs/` directory.
The complete 50-issue paid evaluation was **not run during smoke verification**.
It was subsequently run at the user's request; see [RESULTS.md](RESULTS.md).
Matched one-shot RAG/direct runs are still required to measure improvement;
this single smoke result does not establish an improvement.
