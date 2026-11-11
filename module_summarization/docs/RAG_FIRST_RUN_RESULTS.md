# First RAG run — October 22, 2026

The first implementation and a live paired pilot are complete. Official CompScore has not been run, and improved documentation quality has not been established.

## Measured results

| Check | Result |
| --- | --- |
| Existing generation model | `gpt-3.5-turbo-0125`, accessible and used successfully |
| Offline full-dataset preparation | 216 tasks, 43 repositories, 432 contexts, 339.4 seconds |
| Full-dataset source-span audit | Passed; contexts within the 2,000-token maximum |
| Full-dataset context changed by BM25 | 203 of 216 tasks |
| Full-dataset BM25 includes source beyond the prefix | 186 of 216 tasks |
| Full-dataset no-match fallback | 12 of 216 tasks |
| Live smoke test | 5 tasks, 5 repositories, 10 summaries, 34.6 seconds |
| Live paired pilot | 20 tasks, 20 repositories, 40 summaries, 157.6 seconds |
| Pilot output termination | All 40 ended normally; none hit the output-token limit |
| Offline tests | 23 passed, including budget persistence and original-row scoring alignment |
| Total estimated API charge, smoke plus pilot | US$0.055571; no unresolved reservations |
| Remaining experiment allowance | US$3.944429 of the US$4 cap |

Charges are calculated from returned API token usage and the model's published rates; the provider's account dashboard is the billing authority. The smoke and pilot share five tasks but were separate requests, and both are included in the total. No new full-dataset API generation was performed.

## Preliminary diagnostic comparison

The pilot compares newly generated outputs under matched model/prompt settings:

| Local metric, scale 0–1 | Prefix baseline | BM25 RAG |
| --- | --- | --- |
| Mean whitespace-token ROUGE-L F1 | 0.09450 | 0.10160 |
| Mean supplied code tokens | 1,929.85 | 1,825.20 |

The absolute diagnostic difference is +0.00710. This small development sample and lexical metric do not establish factual accuracy, statistical significance, or generalization. Do not describe it as a measured improvement in official benchmark performance. These numbers are also not directly comparable to the older baseline mean over all 216 tasks, because the task sets differ.

BM25 changed the context in 19 of the pilot's 20 pairs and included source beyond the prefix in 17. One pair fell back to the prefix. Source audits validate where excerpts came from, not whether the chosen excerpts were the most relevant. Spot inspection of three smoke pairs showed different documentation emphasis and formatting, including code-fence wrappers; a systematic factual review against source is still pending.

## What to do next

1. Review the paired pilot predictions for unsupported claims, omitted APIs, irrelevant context, broken code examples and formatting. Include the no-match fallback and short/long contexts. Keep a table of errors and proposed fixes.
2. Ask the mentor for a university GPU and Hugging Face tokenizer access using `RAG_GPU_GUIDE.md`. The pilot's official-scoring preparation reports all 40 predictions present and 80 judge evaluations required. Actual GPU execution is untested.
3. Run official CompScore and report paired differences, uncertainty, and subgroup results before claiming a quality benefit. Freeze settings before extending evaluation.
4. If justified, generate both conditions on all 216 tasks with the same spending ledger. Extrapolating the pilot suggests roughly 25–40 minutes for generation and about US$0.50 in API tokens; both are estimates, not guarantees. The ledger checks each request against the remaining US$4 allowance. GPU evaluation time is unknown until a small GPU timing run.

There is no training phase in this version. Syntax-aware chunking or semantic retrieval would be later experiments if the error review shows that keyword windows are a bottleneck.

## Saved artifacts on this computer

- Full retrieval audit: `/Users/haifakhan/Documents/ChatGPT/project 2/rag_runs/paired-full-4d6080d9e211317e/analysis.json`
- Smoke predictions: `/Users/haifakhan/Documents/ChatGPT/project 2/rag_runs/paired-smoke-5dc72376799de381/`
- Pilot predictions and analysis: `/Users/haifakhan/Documents/ChatGPT/project 2/rag_runs/paired-pilot-a88dc251ae9473d2/`
- Shared spending ledger: `/Users/haifakhan/Documents/ChatGPT/project 2/rag_runs/spending.sqlite`

Prediction folders are named `prefix` and `bm25`; filenames use original dataset IDs. Each generated `.txt` has matching context and generation provenance JSON. Keep those together for evaluation.
