"""Controlled comparison: plain vs BM25 vs dense vs hybrid (vs hybrid with normalized BM25 tokens).

Run:  poetry run python -m src.evaluation.compare
"""
from ..metrics.chrf import ChrF
from ..metrics.overlap import Overlap
from ..models.openai_model import OpenAIModel
from ..models.rag_model import RAGModel, TogetherRAGModel
from ..models.together_model import TogetherModel
from .evaluate import evaluate

K = 20  # the official BM25 baseline uses 20 names, so keep k=20 for a like-for-like table


def run_comparison(model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", provider: str = "together",
                   k: int = K, limit: int | None = None):
    Base, RAG = (TogetherModel, TogetherRAGModel) if provider == "together" else (OpenAIModel, RAGModel)
    conditions = [
        ("plain", Base(model_name)),
        ("bm25 (official)", Base(model_name, use_bm25=True)),
        ("dense", RAG(model_name, k=k, mode="dense")),
        ("hybrid", RAG(model_name, k=k, mode="hybrid")),
        ("hybrid + normalized tokens", RAG(model_name, k=k, mode="hybrid", bm25_tokenizer="normalized")),
    ]
    rows = []
    for label, model in conditions:
        result = evaluate(model, [ChrF(), Overlap()], "results", limit=limit)
        rows.append((label, result["metrics"], result["n_samples"]))

    metric_names = list(rows[0][1].keys())
    print(f"\nGenerator: {model_name} ({provider}) | k={k} | n_samples={rows[0][2]}"
          + ("  (LIMITED RUN - not for reporting)" if limit else ""))
    print(f"{'condition':<30}" + "".join(f"{m:>14}" for m in metric_names))
    for label, metrics, _ in rows:
        print(f"{label:<30}" + "".join(f"{metrics[m]['mean']:>14.3f}" for m in metric_names))


if __name__ == "__main__":
    run_comparison(model_name="gpt-4.1", provider="openai", limit=None)
