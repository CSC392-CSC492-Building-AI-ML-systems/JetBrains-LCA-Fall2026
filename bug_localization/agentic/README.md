# Agentic bug localization prototype

Run all commands from the repository root in PowerShell (Python 3.11). All new
implementation, configuration, tests, documentation and default output folders
are under `bug_localization/agentic/`. Existing experiment entry points are intact.

## Local verification

```powershell
.\.venv\Scripts\python.exe -m pip install -r bug_localization/requirements-rag.txt
.\.venv\Scripts\python.exe -m unittest discover -s bug_localization/agentic/tests -v
.\.venv\Scripts\python.exe -m bug_localization.agentic.smoke
```

The smoke command needs no network or API key. It creates a real Git repository
with different base and fix commits, runs an evidence-driven **mocked** model,
saves rankings and traces, then scores with the unchanged existing evaluator.
The model searches `crash`, discovers `resolve_alias` in the returned import,
searches that identifier, reads the discovered implementation path, and submits
it. Its follow-up arguments are derived from returned snippets, rather than a
fixed sequence of model outputs. The excluded test-file label still counts as a
miss. These synthetic scores are verification, not benchmark results.

Artifacts are in `outputs/offline_smoke/runs/<signature>/examples/000.json` and
`outputs/offline_smoke/scoring/`. Each output parent has `latest_run.json` pointing
to its current content-addressed run. `outputs/` is ignored by Git.

## Live single issue and training pilot

Set `OPENAI_API_KEY` in the process environment. No dotenv file is loaded and no
credentials are serialized. The configuration copies the existing long-context
comparison's pinned model settings. Choose settings supported by your account,
then hold that same file fixed for all comparison arms. Context-window values
are experiment settings, not automatic model-capability discovery.

Training index 621 is `pre-commit/pre-commit#1076` in the dataset version used for
the verified smoke run. `--start` is a zero-based index in the chosen split;
`--start N --limit 1` selects a single issue. Inspect dataset provenance if using
a different dataset revision. The default is one training example, avoiding an
accidental full paid evaluation.

```powershell
# Optional: prepare and save initial input without a model call (not scorable).
.\.venv\Scripts\python.exe -m bug_localization.agentic.run --config bug_localization/agentic/settings.example.json --split train --start 621 --limit 1 --dry-run --output-dir bug_localization/agentic/outputs/prompt_smoke

# Live single training issue, default 4096 source tokens and six tool calls.
.\.venv\Scripts\python.exe -m bug_localization.agentic.run --config bug_localization/agentic/settings.example.json --split train --start 621 --limit 1 --output-dir bug_localization/agentic/outputs/live_smoke

# Score the saved ranking. Labels are loaded only by this scoring step.
.\.venv\Scripts\python.exe -m bug_localization.agentic.evaluate bug_localization/agentic/outputs/live_smoke --allow-partial --output-dir bug_localization/agentic/outputs/live_smoke/scoring

# A small training pilot: do tuning here, then freeze the settings for test.
.\.venv\Scripts\python.exe -m bug_localization.agentic.run --config bug_localization/agentic/settings.example.json --split train --start 620 --limit 5 --tool-calls 6 --output-dir bug_localization/agentic/outputs/train_pilot
.\.venv\Scripts\python.exe -m bug_localization.agentic.evaluate bug_localization/agentic/outputs/train_pilot --allow-partial --output-dir bug_localization/agentic/outputs/train_pilot/scoring
```

The shared Hugging Face loader uses `JetBrains-Research/lca-bug-localization`,
configuration `py`. Use `--dataset-file C:/path/to/lca-bug-localization-train.arrow`
on **both** run and scoring commands for an existing Arrow file. Use the correct
split's file; the shared loader trusts the supplied file's split. For downloading,
`--dataset-cache-dir` selects the HF cache. Git clones/fetches use the existing
`bug_localization/repos/` cache. Preparation reads blobs at `base_sha`, never
checks out code, and never runs repository source. Network and local Git-history
read access are required for a live run.

## Controlled 50-issue comparison (paid; not run automatically)

These commands compare the agent with **one-shot RAG in the existing controlled
long-context comparison**, and direct long context, on the same 50 `py/test`
issues. This RAG arm supplies the full candidate list and packs BM25 chunks under
an explicit source budget. The older `rag.run_ranking` arm instead supplies only
retrieved candidates, so it is a separate experiment and is not interchangeable.

```powershell
.\.venv\Scripts\python.exe -m bug_localization.agentic.run --config bug_localization/agentic/settings.example.json --split test --start 0 --limit 50 --repo-token-budget 4096 --tool-calls 6 --window 40 --stride 20 --output-dir bug_localization/agentic/outputs/test50/agent
.\.venv\Scripts\python.exe -m bug_localization.long_context.run --mode rag --config bug_localization/agentic/settings.example.json --split test --start 0 --limit 50 --repo-token-budget 4096 --window 40 --stride 20 --output-dir bug_localization/agentic/outputs/test50/rag
.\.venv\Scripts\python.exe -m bug_localization.long_context.run --mode direct --config bug_localization/agentic/settings.example.json --split test --start 0 --limit 50 --repo-token-budget 4096 --window 40 --stride 20 --output-dir bug_localization/agentic/outputs/test50/direct
.\.venv\Scripts\python.exe -m bug_localization.agentic.evaluate bug_localization/agentic/outputs/test50/agent bug_localization/agentic/outputs/test50/rag bug_localization/agentic/outputs/test50/direct --output-dir bug_localization/agentic/outputs/test50/comparison
```

For local data, add the identical test Arrow `--dataset-file` to all four
commands. The comparator verifies the dataset fingerprint, selected indices,
issue hashes, model settings, tokenizer, source budget, candidate policy,
chunking, shared implementation versions, resolved snapshots, candidate lists
and BM25 tails. The default scoring command requires all 50 completed test
predictions. `--allow-partial` explicitly permits a pilot or incomplete run and
uses only common completed examples. Report coverage and failures alongside
partial scores; never describe them as the full benchmark.

`comparison.md` and `comparison.json` report MAP, MRR, Hit@1/5/10, cumulative
prompt/completion/total tokens across **every model turn**, total request latency,
mean request latency per issue, elapsed wall time, tool calls, fallback counts
and fallback rates. The table uses matched examples for metrics and resources;
JSON also includes run-wide totals, status counts and a sensitivity analysis
restricted to common model successes. API calls with unavailable usage are
counted separately; failed SDK retries can have unreported billable usage.
Metrics use `baseline_eval.evaluate_tfidf.score_example` and `aggregate` through
the existing long-context scoring adapter. All gold files, including those
excluded by candidate filtering, stay in metric denominators.

Equal initial source budgets do not imply equal total computation: the agent
can gather more evidence and resends its retained conversation. Compare quality
against cumulative tokens and latency, not just the initial 4096-token budget.
Direct mode packs whole files and may underfill a small budget. Do not tune on
the test set or claim an improvement from the smoke example alone.

## Loop and limits

`loop.py` imports the shared `prepare_input`, `pack_prompt(mode="rag")`,
`ChunkBM25Index`, `OpenAIClient`, `ModelSettings`, `TokenCounter`,
`validate_response` and `complete_ranking`. The issue, lexicographic Python
candidate paths and serialized initial BM25 source payload are identical to the
one-shot comparison. Only the agent's system instructions/schema differ.
The baseline reader preserves `.py` blobs, excludes paths containing `test/` or
`tests/` case-insensitively, and retains empty candidate files. The chunk index
uses 40-line windows, stride 20, BM25 k1=1.2 and b=.75 by default.

The Python loop requests one structured JSON decision per turn, with an explicit
schema embedded in the prompt. In `json_schema` mode the same schema is also
sent as the strict response format, using a small `ModelSettings` subclass and
the unchanged OpenAI client. This is a JSON action protocol, not native API
function calling or an agent framework. In text/json-object modes the local
parser still validates decisions. Decisions have these shapes:

```json
{"decision":{"action":"search_code","query":"new_identifier"}}
{"decision":{"action":"read_file","path":"package/module.py","start_line":1,"end_line":40}}
{"decision":{"action":"submit_ranking","paths":["package/module.py"]}}
```

Search uses the existing chunk-level BM25 index over the same prepared snapshot.
Read validates against the exact prepared candidate set and uses physical,
1-based inclusive lines. No tool accesses the live worktree, fix SHA, patch,
gold paths, or arbitrary filesystem locations. Prediction columns are projected
before dataset row materialization. Source and issue text are marked untrusted.

Defaults are `--tool-calls 6`, `--search-results 5`, `--read-lines 120`,
`--tool-output-tokens 1500`, and `--query-characters 512`. Invalid tool names,
paths, queries and ranges spend a call and return an error. Oversized source is
shortened by whole lines, with the actual returned line range and truncation
flag saved. A line too large to fit is omitted. Tool-output budgets must be at
least 256 tokens to fit bounded validation errors. `--tool-calls 0` goes directly
to the final request. An immediate `submit_ranking` is always permitted.

Every request is counted with the shared counter and reserves the configured
output allowance plus at least 5% context safety margin. Full issue and
candidates are retained. A tool result is accepted into conversation history
only if both another action request and a final ranking request fit; otherwise
the trace records that it was not fed back and the agent finalizes from retained
history. At the tool/context limit there is one final request with tool actions
disabled and the existing ranking schema. If even the final input cannot fit,
the result explicitly falls back to BM25. Initial packing uses the same source
selection as one-shot RAG; the agent's additional schema overhead can exhaust
context earlier, which is recorded rather than silently changing that evidence.

Valid partial rankings use shared validation/repair and append missing paths in
BM25 max-chunk-score order, ties lexicographic. This completion is **not** a full
BM25 fallback. Empty/all-invalid rankings, malformed actions, refusal, incomplete
responses, request failures, and unfit context produce separately marked full
BM25 fallbacks. Preparation failures have no ranking and remain pipeline failures.

`run_metadata.json` records model/request settings, split and selected indices,
dataset identity, source budget, tokenizer, retrieval policy, tool limits and
implementation hashes. `examples/<index>.json` holds the final ranking,
validation/repairs, fallback flag/reason, stop reason, initial packing, cumulative
usage, wall/request latency, and structured per-turn traces with exact submitted
messages, raw model responses, usage/attempts, tool arguments, results, paths,
actual line ranges, output tokens and tool latency. The trace records sensitive
issue/source content; store artifacts accordingly.

Writes are atomic and checksummed. Matching validated successes are resumed;
fallbacks and preparation failures are retried. `--no-resume` makes new paid
requests. Exit code 1 indicates any preparation failure or full fallback, even
though fallback rankings remain scorable. Scoring refuses dry runs. See
[SMOKE_RESULTS.md](SMOKE_RESULTS.md) for verification performed in this workspace.
