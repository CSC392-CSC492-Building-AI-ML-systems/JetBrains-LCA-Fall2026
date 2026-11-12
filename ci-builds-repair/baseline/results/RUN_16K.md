# Local Mistral baseline with 16,000-token context

Run `88a9149b9da105e57f04` completed on 2026-11-09. The standalone verifier
independently checked all **204/204** predictions (68 current test records,
three seeds), with no source failures or retrieved chunks. Run output and
verification details are under the ignored
`ci-builds-repair/ci-builds-repair-benchmark/out/rag-local/88a9149b9da105e57f04/`
directory.

```bash
ci-builds-repair/ci-builds-repair-benchmark/.venv/bin/python \
  ci-builds-repair/rag/run_benchmark.py \
  --methods baseline --seeds 0 1 2 --context-window 16000
```

| Seed | Evaluated | Applied responses | Exact reference repairs | Edit-token F1 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 68 | 39 | 0 | 0.192789 |
| 1 | 68 | 41 | 0 | 0.194282 |
| 2 | 68 | 43 | 0 | 0.198628 |

Mean edit-token F1: **0.195233**, with population standard deviation 0.002477.
There were 123 valid applied responses, 65 marker-validation failures, and 16
output-token-limit failures. The scores are identical to the earlier 8,192-token
run: that run skipped three predictions whose prompts exceeded its input budget,
while the 16,000-token run generated all three but their answers reached the
output limit and could not be applied.

The new run reused 201 saved baseline answers after checking that their prompts,
responses, statuses, model, temperature, seeds, output limit, dataset checksum,
and implementation hashes matched; 25 of those were also regenerated and
matched byte-for-byte in this run. The three formerly over-budget answers were
generated with the local 4-bit MLX Mistral model. The exact reuse counts and
source run ID are recorded in `prediction_reuse.json` beside the outputs.

These are **offline reference-edit metrics, not CI Pass@1**. The current cached
split has 68 records, whereas the published benchmark describes 77. The local
runner uses a 4-bit model, a seven-line first-`error` log excerpt, marked block
responses, and a 2,048-token output cap. The later OpenReview version describes
a 200-line log excerpt around `error`/`failure`/`failed`/`traceback`, file
markers, and unified-diff replies. The original arXiv v1 associated with the
0.065 Mistral result instead describes seven lines around the first `error` and
edited code sections. The authors' exact Mistral prompt assembly and decoding
settings are not available here. Docker is not running locally, and
GitHub Actions was excluded by request, so no CI build-success score was
measured. The measured F1 cannot be compared numerically to the paper's Pass@1.
