import unittest

from rag.bm25 import BM25Index, tokenize_code


class TokenizeTests(unittest.TestCase):
    def test_identifier_kept_whole_and_split(self):
        tokens = tokenize_code("compute_total = getUserName(x)")
        for expected in ("compute_total", "compute", "total", "getusername", "get", "user", "name"):
            self.assertIn(expected, tokens)

    def test_keywords_and_single_chars_dropped(self):
        tokens = tokenize_code("def f(x): return None if x else y")
        self.assertEqual(tokens, [])

    def test_acronym_boundary(self):
        self.assertIn("http", tokenize_code("HTTPServer"))
        self.assertIn("server", tokenize_code("HTTPServer"))


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.docs = [["alpha", "beta"], ["alpha"], ["gamma", "gamma", "delta"], []]
        self.index = BM25Index(self.docs)

    def test_only_matching_docs_returned(self):
        ranked = self.index.rank(["alpha"])
        self.assertEqual({doc_id for doc_id, _ in ranked}, {0, 1})
        self.assertTrue(all(score > 0 for _, score in ranked))

    def test_shorter_doc_wins_on_equal_tf(self):
        self.assertEqual(self.index.rank(["alpha"])[0][0], 1)

    def test_rare_term_outweighs_common_term(self):
        scores = self.index.scores(["alpha", "beta"])
        self.assertGreater(scores[0], scores[1])

    def test_repeated_query_terms_count_once(self):
        self.assertEqual(self.index.scores(["alpha"]), self.index.scores(["alpha", "alpha", "alpha"]))

    def test_ties_keep_document_order(self):
        index = BM25Index([["same"], ["same"], ["same"]])
        self.assertEqual([doc_id for doc_id, _ in index.rank(["same"])], [0, 1, 2])

    def test_empty_index_and_unknown_term(self):
        self.assertEqual(BM25Index([]).rank(["x"]), [])
        self.assertEqual(self.index.rank(["nothing"]), [])


if __name__ == "__main__":
    unittest.main()
