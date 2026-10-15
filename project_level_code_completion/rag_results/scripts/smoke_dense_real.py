"""Real embedder on real repositories: speed, cache, and how much the dense selection overlaps with BM25's."""
import os
import sys
import time

from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

from eval.line_generators import dev_indices
from eval.preprocessors import load_datapoints
from rag.builder import RagContextBuilder
from rag.dense import DenseEmbedder

name, pooling = sys.argv[1], sys.argv[2]
dataset, dev_size, dev_extra, n_datapoints = sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
with initialize_config_dir(config_dir=os.path.join(os.environ["PROJECT"], "eval", "config"), version_base=None):
    cfg = compose(config_name="compare_arms", overrides=[f"dataset={dataset}"])
tok = AutoTokenizer.from_pretrained(cfg.params.preprocess_params["tokenizer"])
count = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
data = load_datapoints(cfg.dataset)
dev = [data[i] for i in dev_indices(len(data), dev_size, dev_extra)][:n_datapoints]  # development datapoints only
embedder = DenseEmbedder(name, pooling=pooling, cache_dir="~/lca_artifacts/embeddings")
for dp in dev:
    path = next(iter(dp.get_completion()))
    files = dp.get_context()
    t0 = time.time()
    bm = RagContextBuilder(files, path, dp.repo_name, count)
    t_bm = time.time() - t0
    t0 = time.time()
    dense = RagContextBuilder(files, path, dp.repo_name, count, retriever="dense", embedder=embedder)
    t_dense = time.time() - t0
    t0 = time.time()
    RagContextBuilder(files, path, dp.repo_name, count, retriever="dense", embedder=embedder)
    t_cached = time.time() - t0
    lines = [n for nums in dp.completion_lines.values() for n in nums][:15]
    overlaps, t_query = [], []
    for n in lines:
        prefix = dp.get_prefix(n)
        a = {(c.path, c.start_line) for c in bm.context_for(prefix, 7900, prefix_cap=7900).chunks}
        t1 = time.time()
        r = dense.context_for(prefix, 7900, prefix_cap=7900)
        t_query.append(time.time() - t1)
        b = {(c.path, c.start_line) for c in r.chunks}
        overlaps.append(len(a & b) / max(1, len(a | b)))
    print(f"{dp.repo_name[:34]:34s} chunks={len(bm.chunks):5d} index: bm25 {t_bm:5.1f}s | dense first {t_dense:6.1f}s, "
          f"cached {t_cached:5.1f}s | per-line {sum(t_query) / len(t_query) * 1000:4.0f}ms | "
          f"selection overlap with BM25 (Jaccard) {sum(overlaps) / len(overlaps):.2f}")
