# Experiment methods and artifacts

## Controls and packing

- All modes receive identical instructions, the complete title/body, the complete
  lexicographically sorted candidate list, and the same `ranked_files` JSON schema.
  Prediction projects only the issue and repository provenance fields, never labels,
  `head_sha`, diffs or fix commits. Ground truth is read only during scoring.
- The existing `baseline_eval.run_all_tfidf.read_content` filters candidates: UTF-8
  `.py` blobs, excluding case-insensitive `test/` and `tests/` substrings. Empty files
  remain candidates. All raw ground-truth files remain in evaluation denominators,
  including excluded tests, other extensions and absent paths; they count as misses.
- `direct` visits normalized paths lexicographically, labels each block with its
  path, and includes whole files that fit. It skips an oversized file and continues
  with later files; there are no partial-file prefixes. Included and omitted paths
  and every packing decision are saved.
- `rag` reuses `prepare_repositories.chunk_files` (40 lines, stride 20),
  `bm25_index.ChunkBM25Index` (k1=1.2, b=.75), `make_query` and `retrieve`.
  It considers **all positive matching chunks**, highest score first, ties by chunk
  ID, then packs whole chunks with the same source-token cap. No extra top-30 cap
  confounds the larger budget. Oversized chunks are skipped and later ones tried.
- `issue_only` has zero source tokens; it still makes a model request. Empty RAG
  retrieval or zero packed source also still makes an issue-and-path request.
- Source tokens count serialized JSON source blocks, including paths, escaping,
  line labels and separators, minus the empty array. Raw source-text tokens are
  separately logged. They are not counts of just the unescaped code. Source caps
  are requested maxima; actual source can be lower from whole-unit packing,
  retrieval misses, or the complete-request context constraint.
- The existing `TokenCounter` counts the **entire** request, including instructions,
  issue, candidates, source, message overhead and response-format schema. It uses
  model-specific tiktoken when available, otherwise a conservative UTF-8-byte
  estimate. Logs identify which. The runner additionally reserves the full
  configured response allowance and at least 5% of context for server framing,
  via unchanged `ModelSettings.input_budget`. It recounts every trial request and
  the final prompt. These are local estimates; provider usage is also preserved.
- The full issue and candidate list are reserved **before** source packing. They
  are never truncated. If they alone exceed the input allowance, no API call is
  made and `fallback_context` is recorded. All arms have identical fixed overhead.
  Model names/context sizes are user configuration; the runner does not query
  provider capabilities. Response `length`/refusal is treated as invalid.

## Rankings, failures and reporting

Every mode uses the existing `file_ranking.validate_response` and `complete_ranking`.
The final ranking is always a complete permutation of the same candidates.
**Every mode, including `issue_only`, uses the shared BM25 completion tail**: the
maximum chunk score per file, descending, ties by path. Candidate input order is
lexicographic and no BM25 scores or tail ordering are sent to the model. Thus
`issue_only` describes the model's input; its final pipeline ranking can benefit
from BM25. `model_ranked_paths` distinguishes the returned prefix from this tail.

`success` means a valid full response; `repaired` means a usable response with
normalization, removals or a partial ranking completed by BM25. Both are explicit
model successes; raw responses and all repairs remain saved. Invalid/all-invalid
JSON, refusal or incomplete output gets `fallback_validation`. Request failures get
`fallback_request`. These and `fallback_context` retain the full BM25 ranking with
`model_ranking_success=false` and `bm25_fallback=true`. A snapshot/preparation error
gets `failed` and no score. `dry_run` artifacts are never scored. CLI exit code is
nonzero for any pipeline failure or fallback. Transport retries reuse the existing
bounded client; validation failures do not trigger new model calls.

Each output parent contains `latest_run.json` and `<signature>/run_metadata.json`,
`summary.json`, and `examples/000.json` etc. Manifests record selected indices,
label-free input hashes, model/request settings, tokenizer identity, package versions,
candidate policy, chunking, and implementation hashes. Example records include the
resolved base SHA, snapshot digest, prompt and hash, included/omitted files/chunks,
requested/actual/available source tokens, total prompt tokens and reservations,
candidate/source-file/source-line coverage (overlap deduplicated), truncation,
latency, raw model response, provider usage, status and failure reason. Failures before
packing record unavailable counts as null. Exception text is not serialized.
Candidate coverage describes the constructed prompt; `submitted_messages=null`
identifies requests that were never sent.

Rerunning resumes checksummed, validated model successes; failed/fallback examples
are retried. `--no-resume` reruns all selected examples. A changed input, config,
budget or implementation creates another content-addressed directory. Dry and live
runs have different signatures. Interrupted in-flight requests can be sent again;
the API has no experiment-level exactly-once guarantee. Stable settings/seed make
inputs reproducible, not provider responses guaranteed identical.

`outputs/comparison/comparison.md` contains MAP, MRR, Hit@1/5/10, coverage against
both selected examples and the full dataset, model successes, fallback counts,
pipeline/request/validation/context failures and remaining examples. It contains:

1. Available-prediction scores for each run, with their individual counts.
2. Head-to-head scores on the **intersection of completed examples**, including
   explicitly identified fallback rankings.
3. A sensitivity table restricted to examples with usable model responses in every
   run. Empty intersections display N/A.

Failure/outcome counts in all three tables are run-wide, not subset counts. The
JSON report saves exact common indices and the per-example metric artifact exposes
every scored issue. Metrics call the unchanged
`baseline_eval.evaluate_tfidf.score_example/aggregate`: AP divides by **all** gold
files, MRR uses the first gold rank, and Hit@k means at least one gold path in top k.
Pipeline fallbacks count toward evaluation coverage but never model-success counts.
No 50-example benchmark scores are claimed until those runs are completed.
