# Controlled source-context experiment

Run from the repository root in PowerShell with Python 3.11. All implementation,
tests, documentation and default experiment artifacts live here. Existing RAG and
TF-IDF entry points are unchanged. No full API runs are included in this change.

```powershell
.\.venv\Scripts\python.exe -m pip install -r bug_localization/requirements-rag.txt
.\.venv\Scripts\python.exe -m bug_localization.long_context.run --help
.\.venv\Scripts\python.exe -m bug_localization.long_context.compare --help
.\.venv\Scripts\python.exe -m unittest discover -s bug_localization/long_context/tests -v
.\.venv\Scripts\python.exe -m bug_localization.long_context.smoke
```

The last command runs five **offline fixtures with mocked responses**, including a
real Git repository with distinct base/head commits. It writes a comparison table
to `bug_localization/long_context/outputs/offline_smoke/comparison.md` plus
`comparison.json`, `per_example_metrics.json` and content-addressed per-example runs
under `outputs/offline_smoke/runs/<signature>/`. These scores verify the machinery;
they are not benchmark scores. No credentials or network are needed for the tests
or offline smoke. The example configuration below requires an available API model;
adjust it once for your account before running, then keep it fixed for every arm.

## Benchmark commands

The default dataset is `JetBrains-Research/lca-bug-localization`, configuration
`py`, split `test`, indices 0–49 (`--limit 50`). `--limit 1` or `--limit 2` makes a
limited run; `--start` chooses its first original dataset index. `--dataset-file`
accepts a local HF Arrow file; otherwise the existing dataset loader downloads or
uses the HF cache. Repository access uses the existing `get_repository` clone/fetch
cache at `bug_localization/repos/`; it reads Git blobs at **each issue's base_sha**,
without checking out or executing repository code.

`settings.example.json` is a pinned experiment configuration with a deliberately
configured 128,000-token context allowance. This allowance is an experiment setting,
not model-capability discovery; choose a supported model and context/output limits
for your endpoint. The CLI otherwise inherits the existing RAG defaults (`gpt-4`,
8,192 context, 2,048 response tokens). Use the same config for all five runs.
Only these live benchmark commands need `OPENAI_API_KEY` in the environment.

First inspect one prompt without calling the API:

```powershell
.\.venv\Scripts\python.exe -m bug_localization.long_context.run --mode direct --repo-token-budget 8192 --config bug_localization/long_context/settings.example.json --limit 1 --dry-run --output-dir bug_localization/long_context/outputs/prompt_smoke
```

For a live smoke run, use any full-run command below with `--limit 2` and a separate
`--output-dir`, such as `outputs/live_smoke/direct_small`. The five full 50-example
API runs are intentionally left for the user to launch:

```powershell
.\.venv\Scripts\python.exe -m bug_localization.long_context.run --mode direct --repo-token-budget 8192 --config bug_localization/long_context/settings.example.json --output-dir bug_localization/long_context/outputs/direct_small
.\.venv\Scripts\python.exe -m bug_localization.long_context.run --mode rag --repo-token-budget 8192 --config bug_localization/long_context/settings.example.json --output-dir bug_localization/long_context/outputs/rag_small
.\.venv\Scripts\python.exe -m bug_localization.long_context.run --mode direct --repo-token-budget 65536 --config bug_localization/long_context/settings.example.json --output-dir bug_localization/long_context/outputs/direct_large
.\.venv\Scripts\python.exe -m bug_localization.long_context.run --mode rag --repo-token-budget 65536 --config bug_localization/long_context/settings.example.json --output-dir bug_localization/long_context/outputs/rag_large
.\.venv\Scripts\python.exe -m bug_localization.long_context.run --mode issue_only --repo-token-budget 0 --config bug_localization/long_context/settings.example.json --output-dir bug_localization/long_context/outputs/issue_only
.\.venv\Scripts\python.exe -m bug_localization.long_context.compare bug_localization/long_context/outputs/direct_small bug_localization/long_context/outputs/rag_small bug_localization/long_context/outputs/direct_large bug_localization/long_context/outputs/rag_large bug_localization/long_context/outputs/issue_only --output-dir bug_localization/long_context/outputs/comparison
```

The comparator accepts explicit `summary.json` paths, run folders, or the latest-run
parents shown above. To compare only direct versus RAG at one budget, pass those two
parents. It rejects changes in model/settings, tokenizer, retrieval, source versions,
dataset identity or candidate snapshots. A run may select a different subset; only
shared completed examples enter its head-to-head table. Supply the same local Arrow
file to both CLIs when using `--dataset-file`.

## Controls and outputs

All three arms receive the same full issue and lexicographic candidate list.
`direct` packs whole files; `rag` reuses BM25 chunks; `issue_only` sends zero
source tokens. Packing reserves the complete fixed prompt, response allowance and
safety margin before source. **All modes use the same BM25 completion tail**,
including `issue_only`. Ground-truth paths excluded by candidate filters remain
misses. Model successes, repaired responses, request/validation/context fallbacks
and preparation failures are logged separately.

Each output parent has `latest_run.json` and `<signature>/run_metadata.json`,
`summary.json`, and `examples/000.json` etc. These record snapshot/input/prompt
hashes, source inclusion decisions, token counts, coverage, truncation, latency,
raw responses, validation and rankings. Reruns reuse validated successes and retry
failures; `--no-resume` reruns every selected example. Dry runs cannot be scored.

The comparator writes `comparison.md`, `comparison.json` and
`per_example_metrics.json` in the specified output folder. Tables show MAP, MRR,
Hit@1/5/10, exact scoring coverage, model successes, fallback/failure counts and
remaining examples. Head-to-head scores use a common set of completed examples;
a separate sensitivity table uses common model successes. Empty sets show N/A.
The unchanged TF-IDF evaluator computes all ranking metrics.

See [METHODS.md](METHODS.md) for token accounting, shared components, failure
semantics, provenance/resume details and metric definitions.
