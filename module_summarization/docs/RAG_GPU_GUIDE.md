# Running official CompScore later

You can build retrieval and generate documentation on your laptop. The GPU is needed for the **evaluation judge**, not training. The full benchmark was scored on an RTX A6000. GPU costs are outside the $4 API allowance.

## What to ask your mentor

“Our paired baseline and RAG predictions are ready. Could we use a university Linux GPU machine to run the benchmark's Mistral-7B CompScore judge? An NVIDIA A100 with 40 GB memory would be a comfortable target. We also need Hugging Face access to the Llama-2 tokenizer used by the original evaluation. Is there a project environment or GPU allocation we should use?”

The A100 recommendation is a capacity estimate, not a measured minimum. The existing scorer loads the 7B judge in bfloat16 with FlashAttention-2. Weights alone are roughly 14 GB, and inference needs additional memory for long prompts and intermediate tensors. A 16 GB T4 is not a drop-in match for this scorer. [FlashAttention's supported GPU architectures and datatypes](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support) explain the Ampere-or-newer requirement for its bfloat16 CUDA path. [Colab](https://research.google.com/colaboratory/faq.html) does not guarantee a particular free GPU or availability.

## Preparation

1. Get a university allocation first. If a paid cloud machine is the only option, decide its separate budget and shutdown procedure before renting it.
2. Copy the module summarization source, the cached dataset, and the completed `paired-pilot-*` run directory. Include its manifest, predictions, and generation provenance. The API key and spending ledger are not needed for GPU scoring.
3. Obtain permission/access for `meta-llama/Llama-2-7b-chat-hf` on Hugging Face and authenticate on that machine. The judge itself is `mistralai/Mistral-7B-Instruct-v0.2`.
4. Use Linux with a supported NVIDIA GPU, CUDA and a compatible PyTorch/Transformers/FlashAttention installation. Ask for the lab's working environment. The repository's old dependency constraints are not a verified modern GPU recipe; do not blindly upgrade everything or assume the laptop's lightweight environment can score.
5. Start with one or two paired tasks in a separate experiment configuration and measure memory and seconds per judge evaluation before estimating a full run. The official full comparison requires 864 judge evaluations: 216 tasks × two generation conditions × two option-order evaluations.

## Commands on the GPU machine

From the copied `module_summarization` directory, with the GPU environment active:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.is_bf16_supported())"

# Inspect readiness first; no model downloads or scoring.
python -m rag_pipeline.rag_compscore --run-dir /path/to/paired-pilot-RUN_ID \
  --dataset /path/to/lca-module-summarization-test.arrow

# Only after the environment is ready: downloads/loads the judge and evaluates.
python -m rag_pipeline.rag_compscore --run-dir /path/to/paired-pilot-RUN_ID \
  --dataset /path/to/lca-module-summarization-test.arrow --execute
```

The adapter calls the existing `metrics.get_metric`, preserves original row IDs, and applies the original judge-context trimming equally to both conditions. The [full run report](RAG_FULL_RUN_RESULTS.md) records the measured GPU runtime. It writes partial checkpoints and final results, but does not automatically resume a partial GPU run. Keep logs and capture package versions and GPU details with results. If an out-of-memory error occurs, investigate memory use with the mentor before changing the judge, context limit, precision, or quantization; those changes affect comparability and must be documented.
