# Local baseline versus BM25 RAG benchmark

Run signature: `2d9349bb16247814246b`. Dataset: JetBrains-Research/lca-ci-builds-repair, 68/68 selected records. Seeds: [0, 1, 2]. Completed predictions: 408/408.

This is an offline reference-patch and syntax benchmark. No project code, GitHub workflow, or GitHub Actions evaluator was executed. These numbers are not CI Pass@1.

Model: `mlx-community/Mistral-7B-Instruct-v0.2-4bit`; temperature 0.2; context 8192 tokens; output cap 2048 tokens. Both methods use the same oracle edit locations and effective seed = seed + dataset ID. RAG adds BM25 context from the failing snapshot. Settings were fixed before generation.

| Method | Seed | Evaluated | Applied response | Exact reference | Edit-token F1 | Nonempty | Syntax preserved among nonempty |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 68/68 | 0.5735 | 0.0000 | 0.1928 | 0.5735 | 0.3077 (n=39) |
| baseline | 1 | 68/68 | 0.6029 | 0.0000 | 0.1943 | 0.6029 | 0.2927 (n=41) |
| baseline | 2 | 68/68 | 0.6324 | 0.0000 | 0.1986 | 0.6324 | 0.2558 (n=43) |
| rag | 0 | 68/68 | 0.3088 | 0.0000 | 0.1079 | 0.3088 | 0.3333 (n=21) |
| rag | 1 | 68/68 | 0.3382 | 0.0000 | 0.1158 | 0.3382 | 0.3478 (n=23) |
| rag | 2 | 68/68 | 0.3529 | 0.0000 | 0.1299 | 0.3529 | 0.5000 (n=24) |

## Across seeds

| Method | Exact reference, mean ± std | Edit-token F1, mean ± std |
| --- | --- | --- |
| baseline | 0.0000 ± 0.0000 | 0.1952 ± 0.0025 |
| rag | 0.0000 ± 0.0000 | 0.1179 ± 0.0091 |

RAG reference-changed-file retrieval recall: 0.4772 before packing; 0.2659 after packing. Targets already supplied as editable blocks are excluded from extra evidence.

Mean edit-token F1 difference (RAG minus baseline): **-0.0774**. This describes reference-edit agreement, not semantic improvement or statistical significance.

## Interpretation and limitations

Exact reference match requires the resulting target-file text to equal the dataset repair; valid alternative fixes can score zero. Token F1 compares only added/removed tokens, keyed by file and edit direction, so unchanged context earns no credit. Applied response means the replacement format validated. Syntax preservation uses Python 3.13 AST parsing only; it does not execute imports, linters, dependency setup, tests, or platform-specific builds.

Malformed/truncated/over-budget model outputs remain in denominators as unapplied, unchanged predictions. Pipeline/source failures are listed separately and reduce coverage. Population standard deviations describe variation across the configured seeds.

The historical paper used 77 items and did not publish CI repair seeds/decoding settings. This cached dataset, local 4-bit model, reconstructed prompt, and proxy metrics cannot establish reproduction of the paper's build-success percentage.

Source/pipeline failure records: 0. All selected predictions evaluated: True.

See `metrics.json` for per-seed denominators, error reasons, syntax regressions, token counts, and latency. `run_metadata.json` records dataset/configuration checksums and source hashes. Raw prompts, responses, patch files, and prediction checksums remain in the ignored benchmark output under `out/rag-local/2d9349bb16247814246b/`.
