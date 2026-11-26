# Qwen planner and GPT-4.1 orchestrator

The default run uses the same 68 cached CI-repair cases and seeds 0, 1, and 2
as the full Qwen comparisons. It generates 204 attempts. Each uses the oracle
edit blocks and seven-line failed log excerpt from the earlier benchmarks.

1. The local `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` checkpoint
   diagnoses the failure and proposes a plan. Its effective random seed is
   the configured seed plus case ID. Its context limit is 32,768 tokens,
   its plan cap is 768 tokens, and its temperature is 0.2.
2. `gpt-4.1-2025-04-14` receives the plan, failed log excerpt, and old
   blocks through the OpenAI Responses API. It has up to six calls at
   temperature 0.2, with an 8,192-token output cap per call. The Responses API
   calls use the same attempt labels as Qwen and do not set a server-side seed.
3. GPT-4.1 can look up filenames and inspect bounded line ranges from the
   cached failed Git commit. Its patch tool accepts sparse structured block
   replacements. The tool previews the patch in memory and returns introduced
   Python syntax errors so GPT-4.1 can revise. A proposal that passes this
   check becomes the sole submission for that attempt.
4. The local CI evaluator runs after generation. It uses the same guarded
   whitespace composition and supported workflows as earlier comparisons.
   Unsupported workflows retain their place in the full denominator.

The orchestrator receives bounded excerpts from the public benchmark data
and Qwen's plan. Its tools read the failed snapshot and validate patches
without shell access or CI feedback. The comparison changes the model,
output format, and inference protocol together. A separate GPT-4.1-only arm
would be needed to estimate the contribution of Qwen's plan.

From the repository root, use the existing benchmark Python environment.
Set `OPENAI_API_KEY` in the root `.env.local` or your environment. The runner
reads the key for API authentication and uses `store=false` on requests.

```sh
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python \
  ci-builds-repair/baseline/run_agentic_benchmark.py
```

`--prepare-only` pins the manifest and verifies the model/runtime without
inference. `--limit-new 1` saves a single new attempt for a live smoke test.
Restarting the same command resumes the exact run, reuses saved plans and API
turns, and checks the transcript before counting completed submissions.
`--audit-run RUN_DIR` replays the transcripts and recomputes metrics offline.

Run directories live in the ignored
`ci-builds-repair/ci-builds-repair-benchmark/out/qwen-gpt41-agentic/` folder.
They store per-attempt plans, API usage and turns, accepted submissions,
offline metrics, and status. The completion supervisor
(`finish_rag_benchmark.py`) can freeze the code, audit generation, evaluate
saved predictions locally, and save audited workflow evidence in the run
directory for later collection. The completed full lean-RAG evaluation is
a compatible source of verified controls for `--reuse-evaluation`.

## Results

Run `e9b65a333929f1c53b12` completed on November 22, 2026. The generation
and local workflow audits verified all 204 attempts and all seven
failed/reference control pairs. The local CI evaluation is `ac5f6d067797cd26ebe5`.

| Approach | Valid submissions / 204 | Exact reference / 204 | Mean edit F1 across 68 cases | Local CI Pass@1 on seven cases, mean across seeds |
| --- | ---: | ---: | ---: | ---: |
| Standard Qwen | 184 | 21 | 0.281071 | 19.0% (4/21) |
| Lean RAG | 183 | 21 | 0.229798 | 28.6% (6/21) |
| Long context | 146 | 7 | 0.191581 | 0.0% (0/21) |
| Qwen planner + GPT-4.1 orchestrator | 104 | 20 | 0.261520 | 19.0% (4/21) |

| Seed | Valid submissions / 68 | Exact reference / 68 | Mean edit F1 | Local CI passes / 7 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 34 | 6 | 0.266798 | 0 |
| 1 | 37 | 7 | 0.257440 | 2 |
| 2 | 33 | 7 | 0.260322 | 2 |

The agentic run tied standard Qwen's CI total with a different mix of repairs.
It passed SkyPilot case 36 on seed 1, SkyPilot case 37 on seeds 1 and 2, and
Starlette case 144 on seed 2. Standard passed case 37 on all three seeds and
case 144 on seed 2. Of the 21 agentic attempts on supported workflows, nine
ended without a valid patch, eight reached CI and failed at YAPF or linting,
and four passed.

The six-turn limit accounted for all 100 attempts without a valid patch.
In 82 of those attempts GPT-4.1 spent every turn on path lookups or file
reads. The other 18 attempts produced 30 patch proposals that the static
check rejected for introduced Python syntax errors. None of those 18
attempts submitted a revised patch that passed the check within the turn
limit. Qwen also reached its 768-token plan cap on 51 attempts. GPT-4.1 still
received those partial plans, and 20 of the 51 attempts produced valid
submissions. The run does not isolate the effect of plan truncation.

Across the run, the API returned 784 responses and reported 1,523,237
input tokens, including 756,992 cached input tokens, and 39,088 output
tokens. These are usage counts from saved responses.

Full-dataset CI Pass@1 remains unavailable because 61 workflows are
unsupported. Edit F1 compares reference edits across all 68 cases. The
agentic setup changes the model, patch format, and evidence-gathering process
together, so the observed difference cannot be attributed to Qwen planning
alone. The workflow audit used local CI.
