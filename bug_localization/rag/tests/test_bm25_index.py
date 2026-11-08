"""Focused step-2 checks, using only the standard library and local fixtures."""

import copy
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bug_localization.rag import bm25_index as retrieval
from bug_localization.rag import prepare_repositories as preparation
from bug_localization.rag import retrieval_tokenizer as tokenizer


class TokenizerTests(unittest.TestCase):
    def test_identifier_examples_and_ordinary_words(self):
        cases = {
            "parse_expression": ["parse_expression", "parse", "expression"],
            "loadUserProfile": ["loaduserprofile", "load", "user", "profile"],
            "HTTPServer": ["httpserver", "http", "server"],
            "parseXMLFile": ["parsexmlfile", "parse", "xml", "file"],
            "HTTP": ["http"], "ordinary": ["ordinary"],
            "_name": ["_name", "name"],
            "foo_foo": ["foo_foo", "foo", "foo"],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(tokenizer.tokenize(text), expected)

    def test_filter_after_case_split_and_lowercasing(self):
        self.assertEqual(tokenizer.tokenize("if IF def class return True False None x Y _"), [])
        self.assertEqual(tokenizer.tokenize("getIfReady"), ["getifready", "get", "ready"])

    def test_comments_docstrings_strings_and_punctuation(self):
        text = 'def loadUserProfile(x):\n # retry request\n """Read user."""\n return "network-error"'
        self.assertEqual(tokenizer.tokenize(text), [
            "loaduserprofile", "load", "user", "profile", "retry", "request",
            "read", "user", "network", "error"])
        self.assertEqual(tokenizer.tokenize("alpha.beta/beta,alpha! 123"),
                         ["alpha", "beta", "beta", "alpha"])


class SnapshotFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.files = {
            "a.py": "if x:\n    pass\n",  # chunk 0 exists but has no tokens
            "b.py": "alpha alpha beta\n", "c.py": "beta gamma\n",
            "d.py": "alpha gamma gamma gamma\n", "empty.py": "",
        }
        self.snapshot = self.prepare(self.files)

    def prepare(self, files, **kwargs):
        repository = SimpleNamespace(commit=lambda ref: SimpleNamespace(hexsha="a" * 40))
        # Different synthetic source trees need distinct preparation locations;
        # step 1 correctly trusts the same repository/SHA to mean the same code.
        location = self.root / "prepared" / retrieval.hashlib.sha256(
            preparation.json_bytes(files)).hexdigest()
        with patch.object(preparation, "read_content", return_value=files):
            return preparation.prepare_snapshot(repository, "fixture__repo", "aaaaaaa",
                                                location, **kwargs)

    def build(self, **kwargs):
        return retrieval.build_or_load_index(self.snapshot, self.root / "indexes", **kwargs)


class SearchTests(SnapshotFixture):
    def test_hand_computed_corpus_statistics_and_scores(self):
        index = self.build()
        self.assertEqual(index.bm25.n_docs, 3)
        self.assertEqual(index.bm25.doc_len, [3, 2, 4])
        self.assertEqual(index.bm25.avg_len, 3)
        self.assertEqual(index.document_frequency, {"alpha": 2, "beta": 2, "gamma": 2})
        self.assertEqual(index.bm25.postings["alpha"], [(0, 2), (2, 1)])
        idf = math.log(1.6)  # N=3, df=2: log(1 + 1.5/2.5)
        self.assertAlmostEqual(index.bm25.idf("alpha"), idf)
        # tf=2,length=3 -> 4.4/3.2; tf=1,length=4 -> 2.2/2.5.
        alpha = index.search("alpha")
        self.assertEqual([r["chunk_id"] for r in alpha], [1, 3])
        self.assertAlmostEqual(alpha[0]["score"], idf * 1.375)
        self.assertAlmostEqual(alpha[1]["score"], idf * .88)
        # tf=1,length=2 -> 2.2/1.9; tf=1,length=3 -> 2.2/2.2.
        beta = index.search("beta")
        self.assertEqual([r["chunk_id"] for r in beta], [2, 1])
        self.assertAlmostEqual(beta[0]["score"], idf * 22 / 19)
        self.assertAlmostEqual(beta[1]["score"], idf)
        combined = index.search("alpha beta")
        self.assertAlmostEqual(combined[0]["score"], idf * 2.375)

    def test_shared_query_tokenization_and_repeated_code_terms(self):
        snapshot = self.prepare({"a.py": "loadUserProfile loadUserProfile\n"})
        index = retrieval.ChunkBM25Index(snapshot)
        self.assertEqual(index.bm25.doc_len, [8])
        self.assertEqual(index.bm25.postings["profile"], [(0, 2)])
        self.assertEqual(index.search("loadUserProfile"),
                         index.search("loaduserprofile load user profile"))
        self.assertEqual(index.search("profile profile profile"), index.search("profile"))

    def test_exclusions_preserve_original_evidence_and_no_path_tokens(self):
        original = copy.deepcopy((self.snapshot.metadata, self.snapshot.chunks))
        index = self.build()
        self.assertEqual(index.chunk_ids, [1, 2, 3])
        self.assertEqual(index.excluded_count, 1)
        self.assertEqual(index.chunks_by_id[0], self.snapshot.chunks[0])
        for result in index.search("alpha", 100):
            self.assertEqual({k: v for k, v in result.items() if k != "score"},
                             self.snapshot.chunks[result["chunk_id"]])
            self.assertGreater(result["score"], 0)
        self.assertEqual(index.search("empty py"), [])
        self.assertEqual((self.snapshot.metadata, self.snapshot.chunks), original)

    def test_empty_filtered_unmatched_and_limits(self):
        index = self.build()
        for query in ("", " \n", "if return x", "unknownterm", "!!!"):
            self.assertEqual(index.search(query), [])
        self.assertEqual(index.search("alpha", 0), [])
        self.assertEqual(len(index.search("alpha", 1)), 1)
        self.assertEqual(len(index.search("alpha", 100)), 2)
        for limit in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                index.search("alpha", limit)

    def test_ties_use_original_ids_and_query_order_is_irrelevant(self):
        snapshot = self.prepare({"a.py": "if x: pass", "b.py": "alpha beta", "c.py": "alpha beta"})
        index = retrieval.ChunkBM25Index(snapshot)
        self.assertEqual([r["chunk_id"] for r in index.search("alpha beta")], [1, 2])
        self.assertEqual(index.search("alpha beta"), index.search("beta alpha alpha"))

    def test_empty_or_fully_filtered_snapshot_can_reload_and_search(self):
        for files in ({"empty.py": ""}, {"a.py": "if x: pass"}):
            with self.subTest(files=files):
                snapshot = self.prepare(files)
                first = retrieval.build_or_load_index(snapshot, self.root / "indexes")
                second = retrieval.build_or_load_index(snapshot, self.root / "indexes")
                self.assertEqual(first.bm25.n_docs, 0)
                self.assertEqual(first.bm25.avg_len, 0)
                self.assertEqual(first.excluded_count, len(snapshot.chunks))
                self.assertEqual(second.search("alpha"), [])
                self.assertTrue(second.reused)

    def test_bm25_settings_validation(self):
        for k1, b in ((0, .75), (-1, .75), (float("nan"), .75),
                      (float("inf"), .75), (1.2, -1), (1.2, 1.1),
                      (1.2, float("nan")), (True, .75), (1.2, True)):
            with self.subTest(k1=k1, b=b), self.assertRaises(ValueError):
                self.build(k1=k1, b=b)
        self.assertEqual(self.build(b=0).bm25.b, 0)
        self.assertEqual(self.build(b=1).bm25.b, 1)


class CacheTests(SnapshotFixture):
    def test_reload_matches_without_tokenizing_source(self):
        first = self.build()
        expected = first.search("alpha beta")
        with patch.object(tokenizer, "tokenize", side_effect=AssertionError("re-tokenized")):
            second = self.build()
        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.assertEqual(expected, second.search("alpha beta"))
        self.assertEqual(first.bm25.doc_len, second.bm25.doc_len)
        self.assertEqual(first.document_frequency, second.document_frequency)
        self.assertEqual(first.bm25.postings, second.bm25.postings)

    def test_settings_versions_and_actual_source_invalidate(self):
        indexes = [self.build(), self.build(k1=1.5), self.build(b=.5)]
        with patch.object(tokenizer, "TOKENIZER_VERSION", 99):
            indexes.append(self.build())
        with patch.object(tokenizer, "MIN_TOKEN_LENGTH", 3):
            indexes.append(self.build())
        with patch.object(retrieval, "SCORING_VERSION", "changed"):
            indexes.append(self.build())
        with patch.object(retrieval, "FORMAT_VERSION", 99):
            indexes.append(self.build())
        self.snapshot.chunks[1]["text"] = "different terms\n"
        indexes.append(self.build())
        self.snapshot.metadata["repository"] = "different__repository"
        indexes.append(self.build())
        self.assertEqual(len({i.cache_path for i in indexes}), len(indexes))
        self.assertTrue(all(not i.reused for i in indexes))

    def test_corrupt_incomplete_and_inconsistent_cache_rebuild(self):
        index = self.build()
        metadata_path = index.cache_path / "metadata.json"
        data_path = index.cache_path / "index.json"
        for corrupt in (
            lambda: metadata_path.unlink(), lambda: data_path.unlink(),
            lambda: metadata_path.write_text("{bad", encoding="utf-8"),
            lambda: data_path.write_text("{bad", encoding="utf-8"),
            lambda: metadata_path.write_text("[]", encoding="utf-8"),
        ):
            corrupt()
            self.assertFalse(self.build().reused)
            self.assertTrue(self.build().reused)
        # Even with a matching checksum, malformed posting/mapping data is rejected.
        for mutate in (
            lambda s: s["chunk_ids"].__setitem__(0, 99),
            lambda s: s["doc_len"].__setitem__(0, 99),
            lambda s: s["postings"]["alpha"][0].__setitem__(1, 0),
            lambda s: s["postings"]["alpha"].append([0, 1]),
        ):
            state = json.loads(data_path.read_bytes())
            mutate(state)
            data = preparation.json_bytes(state)
            metadata = json.loads(metadata_path.read_bytes())
            metadata["index_sha256"] = retrieval.hashlib.sha256(data).hexdigest()
            data_path.write_bytes(data)
            metadata_path.write_bytes(preparation.json_bytes(metadata))
            self.assertFalse(self.build().reused)
            self.assertTrue(self.build().reused)

    def test_interrupted_write_rebuilds(self):
        original_replace = Path.replace

        def interrupted(path, target):
            if path.name == "metadata.json":
                raise OSError("interrupted write")
            return original_replace(path, target)

        with patch.object(Path, "replace", interrupted), self.assertRaises(OSError):
            self.build()
        self.assertFalse(self.build().reused)
        self.assertTrue(self.build().reused)

    def test_prepared_loader_validates_step_one_cache(self):
        loaded = retrieval.load_prepared_snapshot(self.snapshot.cache_path)
        self.assertEqual(loaded.metadata, self.snapshot.metadata)
        self.assertEqual(loaded.chunks, self.snapshot.chunks)
        (loaded.cache_path / "chunks.jsonl").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Invalid or incomplete prepared snapshot"):
            retrieval.load_prepared_snapshot(loaded.cache_path)

    def test_deterministic_across_hash_seeds_and_cache_rebuilds(self):
        code = (
            "import json,sys; "
            "from bug_localization.rag.bm25_index import load_prepared_snapshot,build_or_load_index; "
            "i=build_or_load_index(load_prepared_snapshot(sys.argv[1]),sys.argv[2]); "
            "print(json.dumps(i.search('gamma alpha beta')))"
        )
        outputs, payloads = [], []
        for seed in ("1", "23"):
            cache = self.root / seed
            process = subprocess.run(
                [sys.executable, "-c", code, str(self.snapshot.cache_path), str(cache)],
                cwd=Path(__file__).resolve().parents[3], env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True, text=True, check=True)
            outputs.append(process.stdout)
            payloads.append(next(cache.glob("*/index.json")).read_bytes())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(payloads[0], payloads[1])


class CliTests(SnapshotFixture):
    def test_build_reload_and_diagnostic_output(self):
        args = [str(self.snapshot.cache_path), "--cache-dir", str(self.root / "indexes"),
                "--query", "alpha", "--top-k", "1", "--k1", "1.5", "--b", ".5"]
        for state in ("written", "reused"):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(retrieval.main(args), 0)
            self.assertIn("indexed=3 excluded=1 vocabulary=3", output.getvalue())
            self.assertIn("k1=1.5 b=0.5", output.getvalue())
            self.assertIn(f"cache={state}", output.getvalue())
            self.assertIn("matches=1", output.getvalue())
            self.assertIn("chunk_id=1 b.py:1-1", output.getvalue())

    def test_bad_source_is_clear_and_nonzero(self):
        with redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(retrieval.main([str(self.root / "missing")]), 1)
        self.assertIn("Invalid or incomplete prepared snapshot", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
