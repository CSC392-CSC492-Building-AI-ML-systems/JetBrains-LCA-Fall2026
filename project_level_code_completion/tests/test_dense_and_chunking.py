import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from rag.builder import RagContextBuilder, reciprocal_rank_fusion
from rag.chunker import chunk_repo
from rag.dense import DenseEmbedder
from rag.provider import make_rag_provider_factory

LANG, META = "LANGSEP\n", "METASEP\n"


def count(text: str) -> int:
    return len(text.split())


class FakeEmbedder:
    """Bag-of-words vectors over a fixed vocabulary, unit length: cosine = word overlap."""
    VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]

    def __init__(self):
        self.chunk_calls = 0

    def _vec(self, text):
        words = text.lower().replace("\n", " ").split()
        v = np.array([words.count(w) for w in self.VOCAB], dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_chunks(self, texts):
        self.chunk_calls += 1
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def make(files, **kwargs):
    return RagContextBuilder(files, "app/main.txt", "repo", count, lang_sep=LANG, meta_sep=META, **kwargs)


class RankFusionTests(unittest.TestCase):
    def test_scores_and_order(self):
        fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1]])
        self.assertEqual([i for i, _ in fused], [1, 3, 2])  # 1: 1/61+1/62; 3: 1/63+1/61; 2: 1/62
        self.assertAlmostEqual(dict(fused)[2], 1 / 62)

    def test_an_item_in_both_rankings_beats_one_in_only_the_first(self):
        fused = dict(reciprocal_rank_fusion([[5, 6], [6]]))
        self.assertGreater(fused[6], fused[5])

    def test_ties_keep_lower_id(self):
        fused = reciprocal_rank_fusion([[2], [1]])
        self.assertEqual([i for i, _ in fused], [1, 2])


class DenseRetrievalTests(unittest.TestCase):
    FILES = {"app/main.txt": "t", "a/one.txt": "alpha beta gamma", "b/two.txt": "alpha delta", "c/three.txt": "zeta eta"}

    def test_dense_ranks_by_cosine_and_returns_every_chunk(self):
        b = make(self.FILES, retriever="dense", embedder=FakeEmbedder())
        result = b.context_for("alpha beta", budget=1000)
        self.assertEqual([c.path for c in result.chunks][:2], ["a/one.txt", "b/two.txt"])
        self.assertEqual(len(result.chunks), 3)  # dense retrieval ranks everything, it never drops non-matching chunks

    def test_hybrid_combines_both_rankings(self):
        files = {"app/main.txt": "t", "a/lexical.txt": "uniqueword rare", "b/semantic.txt": "alpha beta gamma delta"}
        embedder = FakeEmbedder()
        hybrid = make(files, retriever="hybrid", embedder=embedder)
        result = hybrid.context_for("uniqueword alpha", budget=1000)
        self.assertEqual({c.path for c in result.chunks}, {"a/lexical.txt", "b/semantic.txt"})

    def test_dense_and_hybrid_need_an_embedder(self):
        for retriever in ("dense", "hybrid"):
            with self.assertRaises(ValueError):
                make(self.FILES, retriever=retriever)
        with self.assertRaises(ValueError):
            make(self.FILES, retriever="semantic", embedder=FakeEmbedder())

    def test_embeddings_are_computed_once_per_builder(self):
        embedder = FakeEmbedder()
        b = make(self.FILES, retriever="dense", embedder=embedder)
        for _ in range(3):
            b.context_for("alpha beta", budget=1000)
        self.assertEqual(embedder.chunk_calls, 1)

    def test_default_retriever_is_plain_bm25_and_never_touches_an_embedder(self):
        b = make(self.FILES)
        self.assertIsNone(b.chunk_vectors)
        self.assertEqual([c.path for c in b.context_for("alpha beta", budget=1000).chunks], ["a/one.txt", "b/two.txt"])

    def test_budget_still_holds_with_dense_retrieval(self):
        b = make(self.FILES, retriever="dense", embedder=FakeEmbedder())
        for budget in (8, 12, 20, 1000):
            r = b.context_for("alpha beta", budget=budget)
            self.assertLessEqual(count(r.context + "\n" + r.prefix), budget)

    def test_provider_factory_accepts_an_embedder_object(self):
        from data_classes.datapoint_commit_dataset import DatapointCommitDataset

        dp = DatapointCommitDataset(repo_id=0, repo_name="r", completion_lines={"infile": [1]},
                                    context_dict={"a.txt": "alpha beta"},
                                    completion_dict={"main.txt": "start\nalpha line\n"})
        factory = make_rag_provider_factory(count, 1000, lang_sep=LANG, meta_sep=META, retriever="dense",
                                            embedder=FakeEmbedder())
        prompt, _, info = factory(dp)(dp, 1)
        self.assertEqual(info["paths"], ["a.txt"])
        self.assertIn("alpha beta", prompt)


class ContinuationTests(unittest.TestCase):
    CODE = "\n".join(f"line{i} alpha" if i == 3 else f"filler{i}" for i in range(12))

    def test_following_lines_are_appended_to_the_chunk(self):
        b = make({"app/main.txt": "t", "a/file.txt": self.CODE}, chunking="fine", fine_window=2, fine_stride=2)
        plain = b.context_for("alpha", budget=1000)
        extended = b.context_for("alpha", budget=1000, continuation_lines=3)
        self.assertEqual([c.start_line for c in plain.chunks][:1], [2])  # window [line2, line3 alpha]
        self.assertNotIn("filler4", plain.context)
        self.assertIn("filler4\nfiller5\nfiller6", extended.context)
        self.assertNotIn("filler7", extended.context)

    def test_zero_continuation_changes_nothing(self):
        b = make({"app/main.txt": "t", "a/file.txt": self.CODE})
        self.assertEqual(b.context_for("alpha", budget=1000).context,
                         b.context_for("alpha", budget=1000, continuation_lines=0).context)

    def test_continuation_counts_towards_the_budget(self):
        b = make({"app/main.txt": "t", "a/file.txt": self.CODE}, chunking="fine", fine_window=2, fine_stride=2)
        for budget in (6, 9, 14, 1000):
            r = b.context_for("alpha", budget=budget, continuation_lines=5)
            self.assertLessEqual(count(r.context + "\n" + r.prefix), budget, f"budget={budget}")

    def test_no_continuation_past_the_end_of_the_file(self):
        b = make({"app/main.txt": "t", "a/file.txt": "alpha one\nalpha two"}, chunking="fine", fine_window=2,
                 fine_stride=2)
        self.assertEqual(b.context_for("alpha", budget=1000, continuation_lines=10).context.count("alpha two"), 1)


class FineChunkingTests(unittest.TestCase):
    def test_windows_overlap_by_the_stride(self):
        code = "\n".join(f"l{i}" for i in range(10))
        chunks = chunk_repo({"f.txt": code}, count, chunking="fine", fine_window=4, fine_stride=2)
        self.assertEqual([(c.start_line, c.end_line) for c in chunks], [(0, 3), (2, 5), (4, 7), (6, 9)])

    def test_applies_to_python_files_too_without_parsing(self):
        def never(_code):
            raise AssertionError("tree-sitter must not be used for fine chunking")

        chunks = chunk_repo({"m.py": "def f():\n    return 1\n"}, count, parse=never, chunking="fine", fine_window=5)
        self.assertTrue(chunks and all(c.kind == "window" for c in chunks))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_repo({"f.txt": "x"}, count, chunking="coarse")


class DenseEmbedderCacheTests(unittest.TestCase):
    def test_second_call_is_served_from_disk_and_keyed_by_texts_and_model(self):
        with tempfile.TemporaryDirectory() as cache:
            texts = ["alpha beta", "gamma"]
            vectors = np.eye(2, dtype=np.float32)
            embedder = DenseEmbedder("some/model", cache_dir=cache)
            with mock.patch.object(DenseEmbedder, "_encode", return_value=vectors) as encode:
                first = embedder.embed_chunks(texts)
                second = embedder.embed_chunks(texts)
                self.assertEqual(encode.call_count, 1)
                self.assertTrue(np.allclose(first, second))
            self.assertEqual(len(os.listdir(cache)), 1)
            with mock.patch.object(DenseEmbedder, "_encode", return_value=np.eye(3, dtype=np.float32)) as encode:
                DenseEmbedder("some/model", cache_dir=cache).embed_chunks(texts + ["delta"])
                DenseEmbedder("other/model", cache_dir=cache).embed_chunks(texts)
                self.assertEqual(encode.call_count, 2)  # different texts and a different model are cache misses

    def test_chunks_truncate_on_the_right_and_queries_on_the_left(self):
        embedder = DenseEmbedder("some/model")
        with mock.patch.object(DenseEmbedder, "_encode", return_value=np.ones((1, 4), dtype=np.float32)) as encode:
            embedder.embed_chunks(["x"])
            embedder.embed_query("x")
        self.assertEqual([call.kwargs["truncation_side"] for call in encode.call_args_list], ["right", "left"])

    def test_invalid_pooling_is_rejected(self):
        with self.assertRaises(ValueError):
            DenseEmbedder("some/model", pooling="max")

    def test_empty_input_needs_no_model(self):
        self.assertEqual(DenseEmbedder("some/model").embed_chunks([]).shape[0], 0)


if __name__ == "__main__":
    unittest.main()
