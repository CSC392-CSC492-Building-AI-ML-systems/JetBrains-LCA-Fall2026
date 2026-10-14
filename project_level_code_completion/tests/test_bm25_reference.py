import math
import random
import unittest

from rag.bm25 import BM25Index


def brute_force_scores(documents, query, k1=1.2, b=0.75):
    """Okapi BM25 straight from the formula, no inverted index, for every document."""
    n = len(documents)
    avg = sum(len(d) for d in documents) / n if n else 0.0
    scores = {}
    for doc_id, doc in enumerate(documents):
        total = 0.0
        for term in set(query):
            tf = doc.count(term)
            if tf == 0:
                continue
            df = sum(1 for d in documents if term in d)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            norm = k1 * (1.0 - b + b * len(doc) / (avg or 1.0))
            total += idf * tf * (k1 + 1.0) / (tf + norm)
        if total > 0:
            scores[doc_id] = total
    return scores


class BM25AgainstReferenceTests(unittest.TestCase):
    VOCAB = [f"t{i}" for i in range(25)]

    def random_case(self, rng):
        documents = [[rng.choice(self.VOCAB) for _ in range(rng.randint(0, 40))] for _ in range(rng.randint(1, 30))]
        query = [rng.choice(self.VOCAB + ["absent"]) for _ in range(rng.randint(1, 12))]
        return documents, query

    def test_scores_equal_the_brute_force_reference_on_random_corpora(self):
        rng = random.Random(0)
        for trial in range(200):
            documents, query = self.random_case(rng)
            fast = BM25Index(documents).scores(query)
            slow = brute_force_scores(documents, query)
            self.assertEqual(set(fast), set(slow), f"trial {trial}: different matching documents")
            for doc_id, value in slow.items():
                self.assertAlmostEqual(fast[doc_id], value, places=9, msg=f"trial {trial}, document {doc_id}")

    def test_ranking_is_descending_score_with_ties_by_document_order(self):
        rng = random.Random(1)
        for trial in range(100):
            documents, query = self.random_case(rng)
            ranked = BM25Index(documents).rank(query)
            slow = brute_force_scores(documents, query)
            expected = sorted(slow.items(), key=lambda item: (-item[1], item[0]))
            self.assertEqual([d for d, _ in ranked], [d for d, _ in expected], f"trial {trial}")

    def test_custom_parameters_match_the_reference_too(self):
        rng = random.Random(2)
        for k1, b in ((0.5, 0.2), (2.0, 1.0), (1.2, 0.0)):
            documents, query = self.random_case(rng)
            fast = BM25Index(documents, k1=k1, b=b).scores(query)
            slow = brute_force_scores(documents, query, k1=k1, b=b)
            for doc_id, value in slow.items():
                self.assertAlmostEqual(fast[doc_id], value, places=9)


if __name__ == "__main__":
    unittest.main()
