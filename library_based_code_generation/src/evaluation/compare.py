"""Controlled comparison: plain vs BM25 vs dense vs hybrid (vs hybrid with normalized BM25 tokens).

Same generator, same dataset split, same k, same prompt. Plain and BM25 are the repo's
existing baselines and are left untouched (their cached outputs are reused).

Run:  poetry run python -m src.evaluation.compare
"""
from ..metrics.chrf import ChrF
from ..metrics.overlap import Overlap
from ..models.openai_model import OpenAIModel
from ..models.rag_model import RAGModel
from .evaluate import evaluate

K = 20  # the official BM25 baseline uses 20 names, so keep k=20 for a like-for-like table


def run_comparison(model_name: str = "gpt-3.5-turbo-0125", k: int = K, limit: int | None = None):
    conditions = [
        ("plain", OpenAIModel(model_name)),
        ("bm25 (official)", OpenAIModel(model_name, use_bm25=True)),
        ("dense", RAGModel(model_name, k=k, mode="dense")),
        ("hybrid", RAGModel(model_name, k=k, mode="hybrid")),
        ("hybrid + normalized tokens", RAGModel(model_name, k=k, mode="hybrid", bm25_tokenizer="normalized")),
    ]
    rows = []
    for label, model in conditions:
        result = evaluate(model, [ChrF(), Overlap()], "results", limit=limit)
        rows.append((label, result["metrics"], result["n_samples"]))

    metric_names = list(rows[0][1].keys())
    print(f"\nGenerator: {model_name} | k={k} | n_samples={rows[0][2]}"
          + ("  (LIMITED RUN - not for reporting)" if limit else ""))
    print(f"{'condition':<30}" + "".join(f"{m:>14}" for m in metric_names))
    for label, metrics, _ in rows:
        print(f"{label:<30}" + "".join(f"{metrics[m]['mean']:>14.3f}" for m in metric_names))


if __name__ == "__main__":
    run_comparison(limit=10)  # pipeline check; set limit=None for the full test split
