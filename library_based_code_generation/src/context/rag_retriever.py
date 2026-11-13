import hashlib
import os
import re

import numpy as np
from rank_bm25 import BM25Okapi

from ..models.utils import split_identifier

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class OpenAIEmbedder:
    """Embeds texts with OpenAI, caching each text on disk so re-runs cost nothing."""

    def __init__(self, model: str = DEFAULT_EMBEDDING_MODEL, cache_dir: str = ".cache/embeddings"):
        from openai import OpenAI 

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.cache_dir = os.path.join(cache_dir, model)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, text: str) -> str:
        return os.path.join(self.cache_dir, hashlib.sha256(text.encode()).hexdigest() + ".npy")

    def embed(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if not os.path.exists(self._path(t))]
        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            for text, item in zip(batch, resp.data):
                np.save(self._path(text), np.array(item.embedding, dtype=np.float32))
        return np.stack([np.load(self._path(t)) for t in texts])


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-12)


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def normalized_tokens(text: str) -> list[str]:
    """Use the same identifier normalization for API names and instructions."""
    return re.findall(r"[a-z0-9]+", _CAMEL_BOUNDARY.sub(" ", text).lower())


def official_query_tokens(instruction: str) -> list[str]:
    """The official baseline's query tokenization."""
    return "".join(c for c in instruction if c.isalnum() or c == " ").lower().split(" ")



def reciprocal_rank_fusion(rankings: list[np.ndarray], rrf_k: int = 60) -> np.ndarray:
    """Each ranking is an array of item indices, best first.
    fused[i] = sum over rankings of 1 / (rrf_k + rank_of_i)."""
    n = len(rankings[0])
    fused = np.zeros(n)
    for ranking in rankings:
        fused[ranking] += 1.0 / (rrf_k + np.arange(1, n + 1))
    return fused


class RAGRetriever:
    """Retrieves the top-k API entries for an instruction. """

    def __init__(self, apis: list[str], embedder=None, mode: str = "bm25",
                 docs: list[str] | None = None, rrf_k: int = 60,
                 bm25_tokenizer: str = "normalized"):
        if mode not in ("bm25", "dense", "hybrid"):
            raise ValueError("mode must be bm25, dense, or hybrid")
        if bm25_tokenizer not in ("official", "normalized"):
            raise ValueError("Unknown BM25 tokenizer")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if docs is not None and len(docs) != len(apis):
            raise ValueError("docs must align with apis")
        self.apis, self.embedder, self.mode, self.rrf_k = apis, embedder, mode, rrf_k
        self.bm25_tokenizer = bm25_tokenizer
        texts = docs if docs is not None else [" ".join(split_identifier(a)) or a for a in apis]
        self.matrix = None
        if apis and mode != "bm25":
            if embedder is None:
                raise ValueError("Dense and hybrid retrieval require an embedder")
            self.matrix = _normalize(embedder.embed(texts))
        self.bm25 = None
        if apis and mode in ("bm25", "hybrid"):
            index_tok = normalized_tokens if bm25_tokenizer == "normalized" else split_identifier
            corpus = [index_tok(a) for a in apis]
            if any(corpus):
                self.bm25 = BM25Okapi(corpus)

    def _query_tokens(self, instruction: str) -> list[str]:
        if self.bm25_tokenizer == "normalized":
            return normalized_tokens(instruction)
        return official_query_tokens(instruction)

    def top_k(self, instruction: str, k: int = 20) -> list[str]:
        if type(k) is not int or k < 0:
            raise ValueError("k must be a nonnegative integer")
        if not self.apis or k == 0:
            return []
        if self.mode == "bm25":
            scores = (self.bm25.get_scores(self._query_tokens(instruction))
                      if self.bm25 is not None else np.zeros(len(self.apis)))
            order = np.argsort(-scores, kind="stable")[:k]
            return [self.apis[i] for i in order]
        q = _normalize(self.embedder.embed([instruction])[0])
        dense_rank = np.argsort(-(self.matrix @ q), kind="stable")
        if self.mode == "dense":
            order = dense_rank[:k]
        else:
            scores = (self.bm25.get_scores(self._query_tokens(instruction))
                      if self.bm25 is not None else np.zeros(len(self.apis)))
            bm25_rank = np.argsort(-scores, kind="stable")
            fused = reciprocal_rank_fusion([dense_rank, bm25_rank], self.rrf_k)
            order = np.argsort(-fused, kind="stable")[:k]
        return [self.apis[i] for i in order]
