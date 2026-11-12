# Review of Talha's RAG branch

Reviewed `origin/rag-bug-localization` at `23129da` (Talha), including commits
`c9a8bc0`, `fb0c6ea`, `dc74cba`, `8f8a724`, and `23129da`.

## Verified behavior

The implementation prepares immutable failed-revision snapshots, splits them
into 40-line windows with stride 20, and ranks them with BM25 (`k1=1.2`,
`b=0.75`). Issue title/body form the retrieval query. Predictions are projected
away from labels before retrieval. The LLM receives complete retrieved chunks
within an explicit context budget and reranks allowed file paths. Saved
predictions are scored later with the existing file-ranking metrics. Full
candidate rankings, fallbacks, partial coverage, and failed requests are
reported separately. Content-addressed metadata and atomic writes support
validated resumption.

All **90 branch RAG unit tests passed** in an isolated archive checkout using
this task's Python environment, including its Git fixtures. This verifies tested
logic, not model quality. The branch README explicitly says real provider calls
and the full model experiment have not been run; the branch does not contain
evidence of a completed live benchmark. The default model is legacy `gpt-4`,
temperature 0, with an 8,192-token context budget.

## Adaptations for CI repair

The staged design, deterministic BM25, complete-chunk packing, source identities,
and separate saved-result evaluation are appropriate to reuse. The task-specific
parts cannot be copied unchanged:

- CI repair queries come from failing logs/workflow names, not issue reports.
- Tests must remain eligible source evidence: many CI failures occur in tests.
  The bug-localization baseline deliberately excludes test directories.
- The output is replacement code and a patch, not a ranked file list. Its
  MAP/MRR/Hit@k metrics do not establish repair correctness.
- Use the installed Mistral model in both comparison conditions so retrieval,
  rather than a model/provider change, is the experimental treatment.
- The paper's oracle target locations are supplied to both conditions. Gold
  replacement text stays out of prediction. File recall is computed afterwards.
- Local reference-patch and syntax metrics replace GitHub Actions evaluation
  at the user's request, with explicit limits on what they demonstrate.

No unverified paper-score equivalence or completed provider run is inferred
from commit titles or passing synthetic tests.
