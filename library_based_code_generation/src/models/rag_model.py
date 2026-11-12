from ..context.rag_retriever import DEFAULT_EMBEDDING_MODEL, OpenAIEmbedder, RAGRetriever
from .openai_model import OpenAIModel
from .together_model import TogetherModel


class _RAGMixin:
    """Retrieval step shared by every provider."""

    def _init_rag(self, k, mode, embedding_model, bm25_tokenizer, rrf_k, embedder):
        self.k, self.mode, self.rrf_k = k, mode, rrf_k
        self.embedding_model, self.bm25_tokenizer = embedding_model, bm25_tokenizer
        self.embedder = (embedder or OpenAIEmbedder(embedding_model)) if mode != "bm25" else None
        if type(k) is not int or k < 0:
            raise ValueError("k must be a nonnegative integer")
        self.evidence_budget_bytes = 8000
        self._retrievers = {}  # one index per library, not per task

    def get_bm25_prompt(self, instruction: str, project_apis: list[str], n_selections: int = 20):
        key = tuple(dict.fromkeys(project_apis or []))
        if key not in self._retrievers:
            self._retrievers[key] = RAGRetriever(
                list(key), self.embedder, self.mode,
                rrf_k=self.rrf_k, bm25_tokenizer=self.bm25_tokenizer)
        apis = self._retrievers[key].top_k(instruction, self.k)
        selected = []
        for api in apis:
            if len(", ".join(selected + [api]).encode("utf-8")) <= self.evidence_budget_bytes:
                selected.append(api)
        self.last_retrieval = {
            "query": instruction, "candidate_count": len(key),
            "retrieved_apis": apis, "selected_apis": selected,
            "evidence_bytes": len(", ".join(selected).encode("utf-8")),
        }
        text = instruction
        if selected:
            text += "\n\nYou can find the following APIs from the library helpful:\n" + ", ".join(selected)
        return self.get_prompt(text)

    def name(self):
        tag = f"rag_v2_{self.mode}"
        if self.mode in ("bm25", "hybrid"):
            tag += f"-{self.bm25_tokenizer}"
        if self.embedding_model != DEFAULT_EMBEDDING_MODEL:
            tag += f"-{self.embedding_model}"
        if self.mode == "hybrid":
            tag += f"-rrf{self.rrf_k}"
        return f"{tag}_k{self.k}_b{self.evidence_budget_bytes}/{self.model_name}"

    def config(self) -> dict:
        """Recorded in metadata.json so every result says how it was produced."""
        hybrid = self.mode == "hybrid"
        return {
            "generator": self.model_name,
            "retrieval_mode": self.mode,
            "k": self.k,
            "retrieval_version": 2,
            "evidence_budget_bytes": self.evidence_budget_bytes,
            "retrieval_source": "project_defined_elements (API names only)",
            "embedding_model": self.embedding_model if self.mode != "bm25" else None,
            "bm25_tokenizer": self.bm25_tokenizer if self.mode != "dense" else None,
            "rrf_k": self.rrf_k if hybrid else None,
        }


class RAGModel(_RAGMixin, OpenAIModel):
    """RAG with an OpenAI generator."""

    def __init__(self, model_name: str, k: int = 20, mode: str = "bm25",
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 bm25_tokenizer: str = "normalized", rrf_k: int = 60, embedder=None):
        OpenAIModel.__init__(self, model_name, use_bm25=True)  # reuses the project_apis plumbing
        self._init_rag(k, mode, embedding_model, bm25_tokenizer, rrf_k, embedder)


class TogetherRAGModel(_RAGMixin, TogetherModel):
    """RAG with a Together.AI generator."""

    def __init__(self, model_name: str, k: int = 20, mode: str = "bm25",
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 bm25_tokenizer: str = "normalized", rrf_k: int = 60, embedder=None):
        TogetherModel.__init__(self, model_name, use_bm25=True)
        self._init_rag(k, mode, embedding_model, bm25_tokenizer, rrf_k, embedder)
