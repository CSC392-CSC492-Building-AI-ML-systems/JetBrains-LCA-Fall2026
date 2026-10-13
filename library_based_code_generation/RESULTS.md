# Library-based Code Generation BM25 baseline results

The BM25 top-20 baseline results below cover all 150 test examples for each of the six released models. These scores come from the existing per-model `metadata.json` files alongside the checked-in predictions. No new model API calls were made to create this report, so the original generation environment and dataset revision were not recorded with those existing outputs.

| Model | ChrF | API Recall |
| --- | ---: | ---: |
| CodeLlama 70B Instruct | 0.112160 | 0.035625 |
| CodeLlama 7B Instruct | 0.289920 | 0.149372 |
| GPT-3.5 Turbo 0125 | 0.260515 | 0.192027 |
| GPT-4 0125 Preview | 0.394617 | 0.358043 |
| Mistral 7B Instruct v0.3 | 0.311918 | 0.130139 |
| Mixtral 8x7B Instruct v0.1 | 0.293856 | 0.128218 |

For a fresh run, follow the BM25 replication instructions in [README.md](README.md#reproducible-baseline-replication). The runner writes predictions, per-model summaries, and run metadata under the existing `results/` directory.
