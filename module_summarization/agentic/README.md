# Module summarization with agentic code search

See the [standalone results report](RESULTS.md) for scores, usage, and findings.

This arm uses the same 216 tasks, GPT‑4.1 mini snapshot, supplied source, documentation prompt, output cap, and judge specified in the [evaluation protocol](../docs/RAG_EVALUATION_PROTOCOL.md). It starts with the intent and filename, then chooses which parts of the supplied code to inspect. Reference documentation is read by the evaluator.

## Search and final writing

- `search_code(query)` returns up to three ranked BM25 previews and task-local chunk IDs. `read_code(chunk_id)` reads from a previously returned hit and can extend into the following code. Every excerpt records its character offsets in the supplied source.
- The loop permits six search turns with at most one tool call per turn. Each response can return up to 1,500 GPT‑2 code tokens, with an 8,000-token aggregate cap. Previews and overlapping code returned on different calls count toward the cap. Repeated calls, invalid arguments, and no matches are recorded and consume a turn.
- On early completion or a limit, including a capped search draft, a separate final request uses the original baseline system message and documentation prompt with the observed source excerpts, in source order with overlaps removed. The final request has a 2,000-token output cap. The stored context includes only text actually returned by the tools.
- A task can use up to seven API requests, or 1,512 across the full set. Tool schemas, tool outputs, and accumulated search history are included in per-request spending reservations. Report provider tokens and cost across all turns, including code the API receives again in history.

## Run and resume

From `module_summarization/`, install the shared [pipeline dependencies](../rag_pipeline/rag_requirements.txt). Use the archived Arrow dataset and pinned GPT‑2 tokenizer. Set `RAG_DATA`, `RAG_TOKENIZER`, `RAG_ENV_FILE`, and `RAG_LEDGER` to their local paths. Use the same ledger as the previous experiments and a persistent `RAG_WORK_DIR` outside the checkout.

```sh
python -m unittest discover -s agentic/tests -v

python -m agentic.run --config "$RAG_CONFIG" \
  --dataset "$RAG_DATA" --tokenizer "$RAG_TOKENIZER" \
  --env-file "$RAG_ENV_FILE" --ledger "$RAG_LEDGER" \
  --output-root "$RAG_WORK_DIR" --cache-dir "$RAG_WORK_DIR/index_cache"
```

Use `RAG_CONFIG=agentic/smoke/config.json` and add `--force-tool-limit-id 4` to the command for the five-task smoke run. Use the [development configs](development/) for the follow-up cases. The archived smoke and development outputs record retrieval behavior and measured usage. Use actual provider usage to estimate the full cost against the remaining ledger balance before setting `RAG_CONFIG=agentic/full.json`.

Each successful API response is checkpointed before its charge settles. Tool responses are saved turn by turn and checked during replay. Run the same command with the same working directory and ledger to replay completed turns or resume verified tasks. Requests without a persisted response retain their reservation when billing is uncertain and require reconciliation. Failed requests stop the run and are logged. Automatic retries are disabled. A nonempty final response cut off at the output cap retains its length finish reason for review. Final outputs use the shared manifest, prediction, and context formats. The per-task generation JSON includes tool arguments, returned excerpts, stop reason, per-turn usage, cost, latency, and retry count.

Offline validation searched task 159, whose supplied code contains 23,189,964 characters. It located and verified a code excerpt starting at offset 23,188,083. A mocked smoke run also verified six tool calls, a seventh final request, and replay without new requests.

## Evaluate and compare

Use [the long-context scoring commands](../long_context/README.md#analyze-and-score) with the agent run directory. The same CompScore judge and 6K judge context apply. Publish the completed full run under `agentic/results/` for the dashboard. The agent and 8K direct arms together require 864 judge evaluations.

Report CompScore and paired uncertainty alongside actual API tokens, cost, latency, tool-call counts, and failure cases. The primary comparison measures the entire agent workflow against 8K direct context with the same maximum code allowance. Record development IDs in the full manifest before generation. The initial list covers the 12 tasks reviewed during design. If more tasks inform tuning, add them before freezing the full run and report a paired sensitivity analysis that excludes those IDs from every arm.
