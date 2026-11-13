import unittest

from src.context.rag_retriever import RAGRetriever, normalized_tokens


class RetrievalTests(unittest.TestCase):
    def test_identifier_query_alignment(self):
        self.assertEqual(normalized_tokens("loadHTTPServer"), ["load", "http", "server"])
        retriever = RAGRetriever(["loadHTTPServer", "write_file", "close_socket"])
        self.assertEqual(retriever.top_k("load http server", 1), ["loadHTTPServer"])

    def test_empty_and_tokenless_candidates(self):
        self.assertEqual(RAGRetriever([]).top_k("anything"), [])
        self.assertEqual(RAGRetriever(["!!!", "???"]).top_k("anything"), ["!!!", "???"])

    def test_stable_no_match_and_zero_k(self):
        retriever = RAGRetriever(["write_file", "load_file", "close_socket"])
        self.assertEqual(retriever.top_k("unmatched", 2), ["write_file", "load_file"])
        self.assertEqual(retriever.top_k("file", 0), [])
        with self.assertRaises(ValueError):
            retriever.top_k("file", -1)

    def test_bm25_never_calls_embedder(self):
        class ForbiddenEmbedder:
            def embed(self, texts):
                raise AssertionError("BM25 must not request embeddings")

        retriever = RAGRetriever(["load_file", "save_file"], ForbiddenEmbedder())
        self.assertEqual(len(retriever.top_k("load")), 2)


if __name__ == "__main__":
    unittest.main()
