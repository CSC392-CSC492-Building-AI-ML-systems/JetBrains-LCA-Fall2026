# Module summarization with 8K direct context

See the [standalone results report](RESULTS.md) for scores, usage, and findings.

The [evaluation protocol](../docs/RAG_EVALUATION_PROTOCOL.md) defines the comparison. This arm uses the first 8,000 GPT‑2 code tokens from each task's supplied context. It reuses the 2K prefix pipeline, documentation prompt, decoding settings, and 2,000-token output allowance. The full config pins the archived 216 task IDs in the same order and checks the dataset and tokenizer fingerprints before running.

The [full run config](full.json) requests the pinned model for all 216 tasks. The [five-task smoke config and outputs](smoke/) document the live preflight.

## Offline preparation

Run from `module_summarization/` with the shared [pipeline dependencies](../rag_pipeline/rag_requirements.txt) installed. Set `RAG_DATA` to the original Arrow dataset, `RAG_TOKENIZER` to its pinned GPT‑2 tokenizer JSON, and `RAG_WORK_DIR` to a local working directory. Use a persistent working directory when generating so the runner can reuse verified outputs after an interruption.

```sh
python -m rag_pipeline.rag_replication --mode retrieve \
  --config long_context/full.json --dataset "$RAG_DATA" \
  --tokenizer "$RAG_TOKENIZER" --output-root "$RAG_WORK_DIR"
```

Offline validation covered all 216 cases. All contexts were contiguous prefixes within budget. The five smoke rows each reached 8,000 tokens. Across the full set, 204 exceeded 2,000 tokens, 174 reached 8,000, and the mean was 7,202.7.

## Model access and generation

Set `RAG_ENV_FILE` to the private API env file and `RAG_LEDGER` to the existing experiment's spending ledger. Coordinate one full-run owner and use that shared ledger for smoke and full generation. Keep the key and ledger outside the tracked results.

```sh
python -m rag_pipeline.rag_replication --mode check-model \
  --config long_context/full.json --env-file "$RAG_ENV_FILE"

python -m rag_pipeline.rag_replication --mode generate \
  --config long_context/full.json --dataset "$RAG_DATA" \
  --tokenizer "$RAG_TOKENIZER" --env-file "$RAG_ENV_FILE" \
  --output-root "$RAG_WORK_DIR" --ledger "$RAG_LEDGER"
```

The model check reports whether the account can list the snapshot. Use `--config long_context/smoke/config.json` for the five-task preflight before launching the full command. Check the returned snapshot, finish reasons, usage, and remaining budget. [OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini)

For each case, the runner records the source span, prompt and prediction hashes, returned model, finish reason, usage, API charge, and latency. The full run should produce 216 predictions in its `prefix/` directory. Use the printed full run directory as `RAG_RUN_DIR`.

## Analyze and score

Run analysis locally to verify source offsets and record diagnostic ROUGE-L. Prepare CompScore locally to check that every prediction has matching provenance. Use a GPU only for the scoring command with `--execute`. The full arm needs 432 judge evaluations. The earlier 2K arms already have scores.

```sh
python -m rag_pipeline.rag_analyze --run-dir "$RAG_RUN_DIR" --dataset "$RAG_DATA"

python -m rag_pipeline.rag_compscore --run-dir "$RAG_RUN_DIR" --dataset "$RAG_DATA" \
  --judge-revision 63a8b081895390a26e140280378bc85ec8bce07a \
  --judge-tokenizer-revision f5db02db724555f92da89c216ac04704f23d4590
```

On the GPU, run the second command with `--execute` using the same dataset and full run directory. Keep the original 6,000-token judge context. Once the result is complete, archive the full run directory under `long_context/results/` for the dashboard. The archived smoke run is under [smoke/](smoke/). Compare 8K prefix with the saved 2K prefix to measure the effect of the larger input and with 2K BM25 to compare the two strategies.

See the [GPU setup guide](../docs/RAG_GPU_GUIDE.md) and [saved scoring environment](../benchmark_results/paired-full-gpt41-mini-15055eac3076fca7/compscore_environment.json) for the tested judge stack.
