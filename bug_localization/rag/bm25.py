"""Okapi BM25 for pre-tokenized chunks, adapted from the code completion baseline."""

import math
from collections import Counter, defaultdict


class BM25Index:
    """Use an inverted index to score only documents that match the query."""

    def __init__(self, documents: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(documents)
        self.doc_len = [len(document) for document in documents]
        self.avg_len = sum(self.doc_len) / self.n_docs if self.n_docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, document in enumerate(documents):
            for term, frequency in Counter(document).items():
                self.postings[term].append((doc_id, frequency))

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: list[str]) -> dict[int, float]:
        """Score matching documents, counting each query term once."""
        result: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            idf = self.idf(term)
            for doc_id, frequency in self.postings.get(term, ()):
                norm = self.k1 * (
                    1.0 - self.b + self.b * self.doc_len[doc_id] / (self.avg_len or 1.0)
                )
                result[doc_id] += idf * frequency * (self.k1 + 1.0) / (frequency + norm)
        return result

    def rank(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        """Return matching document IDs and scores, with ties in document order."""
        return sorted(self.scores(query_tokens).items(), key=lambda item: (-item[1], item[0]))
