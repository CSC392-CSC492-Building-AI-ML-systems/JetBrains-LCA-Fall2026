import keyword
import math
import re
from collections import Counter, defaultdict

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
PYTHON_KEYWORDS = frozenset(keyword.kwlist)


def tokenize_code(text: str) -> list[str]:
    """Lower-cased identifiers, each kept whole and also split on `_` and camelCase. Keywords and 1-char tokens dropped."""
    tokens = []
    for identifier in IDENTIFIER.findall(text):
        parts = [p for p in CAMEL_BOUNDARY.sub("_", identifier).split("_") if p]
        candidates = [identifier] + (parts if len(parts) > 1 else [])
        tokens += [t.lower() for t in candidates if len(t) > 1 and t not in PYTHON_KEYWORDS]
    return tokens


class BM25Index:
    """Okapi BM25 over pre-tokenized documents, with an inverted index so only matching documents are scored."""

    def __init__(self, documents: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(documents)
        self.doc_len = [len(d) for d in documents]
        self.avg_len = sum(self.doc_len) / self.n_docs if self.n_docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, doc in enumerate(documents):
            for term, tf in Counter(doc).items():
                self.postings[term].append((doc_id, tf))

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: list[str]) -> dict[int, float]:
        """Scores of documents sharing at least one term with the query; repeated query terms count once."""
        result: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            idf = self.idf(term)
            for doc_id, tf in self.postings.get(term, ()):
                norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[doc_id] / (self.avg_len or 1.0))
                result[doc_id] += idf * tf * (self.k1 + 1.0) / (tf + norm)
        return result

    def rank(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        """(doc_id, score) with positive score, best first; ties keep document order."""
        return sorted(self.scores(query_tokens).items(), key=lambda item: (-item[1], item[0]))
