"""Focused preparation checks; runnable with unittest without new packages."""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bug_localization.baseline_eval import prepare_repositories as preparation
from bug_localization.baseline_eval.run_all_tfidf import read_content


class ChunkTests(unittest.TestCase):
    def test_lengths_overlap_and_final_window(self):
        for length, ranges in [
            (0, []), (1, [(1, 1)]), (39, [(1, 39)]), (40, [(1, 40)]),
            (41, [(1, 40), (21, 41)]), (60, [(1, 40), (21, 60)]),
            (61, [(1, 40), (21, 60), (41, 61)]),
            (100, [(1, 40), (21, 60), (41, 80), (61, 100)]),
        ]:
            with self.subTest(length=length):
                source = "".join(f"    line {n}\n" for n in range(1, length + 1))
                chunks = preparation.chunk_files({"src/a.py": source})
                self.assertEqual([(c["start_line"], c["end_line"]) for c in chunks], ranges)
                for chunk in chunks:
                    self.assertEqual(chunk["text"], "".join(
                        source.splitlines(keepends=True)[chunk["start_line"] - 1:chunk["end_line"]]))

    def test_whitespace_chunks_and_original_line_numbers(self):
        self.assertEqual(preparation.chunk_files({"empty.py": "", "space.py": " \n\t\n"}), [])
        chunks = preparation.chunk_files({"a.py": "\n" * 40 + "  code\n"})
        self.assertEqual([(c["chunk_id"], c["start_line"], c["end_line"]) for c in chunks],
                         [(0, 21, 41)])
        self.assertTrue(chunks[0]["text"].endswith("  code\n"))

    def test_order_duplicates_normalization_and_no_final_newline(self):
        files = {"z.py": "  same", "src\\a.py": "  same", "empty.py": ""}
        first = preparation.chunk_files(files)
        self.assertEqual(first, preparation.chunk_files(dict(reversed(list(files.items())))))
        self.assertEqual([c["path"] for c in first], ["src/a.py", "z.py"])
        self.assertEqual([c["chunk_id"] for c in first], [0, 1])
        self.assertEqual([c["text"] for c in first], ["  same", "  same"])

    def test_source_content_and_physical_lines(self):
        source = "\tfirst\r\n  second\rthird\nfourth\fvalue\u2028still fourth"
        chunks = preparation.chunk_files({"a.py": source}, window=2, stride=2)
        self.assertEqual([c["text"] for c in chunks],
                         ["\tfirst\r\n  second\r", "third\nfourth\fvalue\u2028still fourth"])
        self.assertEqual([(c["start_line"], c["end_line"]) for c in chunks], [(1, 2), (3, 4)])

    def test_settings_validation_even_for_empty_mapping(self):
        for window, stride in [(0, 1), (-1, 1), (40, 0), (40, -1), (40, 41), (1.5, 1), (True, 1)]:
            with self.subTest(window=window, stride=stride), self.assertRaises(ValueError):
                preparation.chunk_files({}, window, stride)
        self.assertEqual(len(preparation.chunk_files({"a.py": "a\nb\nc"}, 1, 1)), 3)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache = Path(self.temporary.name)
        self.sha = "a" * 40
        self.repository = SimpleNamespace(commit=lambda ref: SimpleNamespace(hexsha=self.sha))
        self.files = {"z.py": "  code\n", "empty.py": "", "space.py": " \n"}
        self.reader = patch.object(preparation, "read_content", return_value=self.files).start()
        self.addCleanup(patch.stopall)

    def prepare(self, **kwargs):
        return preparation.prepare_snapshot(self.repository, "owner__repo", "aaaaaaa", self.cache, **kwargs)

    def test_reuse_and_empty_candidates(self):
        first = self.prepare()
        second = self.prepare()
        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.reader.assert_called_once_with(self.repository, self.sha)
        self.assertEqual(second.candidate_paths, ["empty.py", "space.py", "z.py"])
        self.assertEqual(second.metadata["base_sha"], self.sha)
        self.assertEqual(second.metadata["file_count"], 3)
        self.assertEqual(second.metadata["chunk_count"], 1)
        self.assertEqual(first.metadata, second.metadata)
        self.assertEqual(first.chunks, second.chunks)

    def test_settings_policy_version_repository_and_commit_invalidate(self):
        results = [self.prepare(), self.prepare(window=30), self.prepare(stride=10)]
        with patch.object(preparation, "FORMAT_VERSION", 2):
            results.append(self.prepare())
        with patch.dict(preparation.CANDIDATE_POLICY, {"reader": "new-policy"}):
            results.append(self.prepare())
        self.sha = "b" * 40
        results.append(self.prepare())
        results.append(preparation.prepare_snapshot(self.repository, "other__repo", self.sha, self.cache))
        self.assertEqual(len({result.cache_path for result in results}), len(results))
        self.assertTrue(all(not result.reused for result in results))

    def test_corrupt_or_incomplete_caches_are_rebuilt(self):
        result = self.prepare()
        metadata_path = result.cache_path / "metadata.json"
        chunks_path = result.cache_path / "chunks.jsonl"
        for corrupt in [
            lambda: metadata_path.unlink(),
            lambda: chunks_path.unlink(),
            lambda: chunks_path.write_text("{broken", encoding="utf-8"),
            lambda: metadata_path.write_text("{broken", encoding="utf-8"),
            lambda: metadata_path.write_text(json.dumps({**result.metadata, "file_count": 99}), encoding="utf-8"),
            lambda: metadata_path.write_text(json.dumps({**result.metadata, "base_sha": "f" * 40}), encoding="utf-8"),
            lambda: metadata_path.write_text(json.dumps({**result.metadata, "chunk_count": 99}), encoding="utf-8"),
        ]:
            corrupt()
            self.assertFalse(self.prepare().reused)
            self.assertTrue(self.prepare().reused)

    def test_interruption_before_metadata_cannot_create_valid_cache(self):
        original_replace = Path.replace

        def interrupted(path, target):
            if path.name == "metadata.json":
                raise OSError("simulated interrupted write")
            return original_replace(path, target)

        with patch.object(Path, "replace", interrupted), self.assertRaises(OSError):
            self.prepare()
        self.assertFalse(self.prepare().reused)
        self.assertTrue(self.prepare().reused)

    def test_all_empty_candidates_produce_valid_zero_chunk_cache(self):
        self.reader.return_value = {"empty.py": "", "blank.py": " \n\t"}
        self.assertEqual(self.prepare().chunks, [])
        result = self.prepare()
        self.assertTrue(result.reused)
        self.assertEqual(result.metadata["file_count"], 2)

    def test_cache_payload_is_deterministic(self):
        first = self.prepare()
        original = {name: (first.cache_path / name).read_bytes() for name in ("metadata.json", "chunks.jsonl")}
        self.reader.return_value = dict(reversed(list(self.files.items())))
        (first.cache_path / "metadata.json").unlink()
        rebuilt = self.prepare()
        for name, data in original.items():
            self.assertEqual((rebuilt.cache_path / name).read_bytes(), data)


class CliTests(unittest.TestCase):
    def test_smoke_default_projects_to_provenance_and_closes_repository(self):
        rows = [{"repo_owner": "owner", "repo_name": "repo", "base_sha": "a" * 40}] * 3
        repository = Mock()
        repository.commit.return_value.hexsha = "a" * 40
        loader = Mock(return_value=rows)
        with tempfile.TemporaryDirectory() as cache, redirect_stdout(io.StringIO()) as output:
            with patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=loader)}), \
                    patch.object(preparation, "get_repository", return_value=repository) as get_repo, \
                    patch.object(preparation, "read_content", return_value={"a.py": "code"}):
                self.assertEqual(preparation.main(["--cache-dir", cache]), 0)
                get_repo.assert_called_once_with(rows[0])
        repository.close.assert_called_once()
        self.assertIn("files=1 chunks=1 cache=written", output.getvalue())
        self.assertIn("Prepared 1/1 examples; failures=0", output.getvalue())

    def test_subset_continues_after_failure_and_returns_nonzero(self):
        rows = [{"repo_owner": "owner", "repo_name": "repo", "base_sha": "a" * 40}] * 4
        repository = Mock()
        repository.commit.return_value.hexsha = "a" * 40
        with tempfile.TemporaryDirectory() as cache, redirect_stdout(io.StringIO()) as output, \
                redirect_stderr(io.StringIO()) as errors:
            with patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=Mock(return_value=rows))}), \
                    patch.object(preparation, "get_repository", side_effect=[RuntimeError("missing commit"), repository]), \
                    patch.object(preparation, "read_content", return_value={"a.py": "code"}):
                status = preparation.main(["--start", "1", "--limit", "2", "--cache-dir", cache])
        self.assertEqual(status, 1)
        self.assertIn("[1] FAILED owner__repo: RuntimeError: missing commit", errors.getvalue())
        self.assertIn("[2] owner__repo", output.getvalue())
        self.assertIn("Prepared 1/2 examples; failures=1", output.getvalue())
        repository.close.assert_called_once()

    def test_dataset_loading_failure_is_clear(self):
        with patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=Mock(side_effect=OSError("offline")))}), \
                redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(preparation.main([]), 1)
        self.assertIn("FAILED loading dataset: OSError: offline", errors.getvalue())


@unittest.skipUnless(importlib.util.find_spec("git"), "GitPython unavailable")
class GitSnapshotTests(unittest.TestCase):
    def test_baseline_parity_pre_fix_content_and_untouched_checkout(self):
        from git import Repo

        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            checkout = root / "repository"
            checkout.mkdir()
            repo = Repo.init(checkout)
            try:
                sources = {
                    "src/a.py": "  buggy\r\n" + "code\n" * 60,
                    "empty.py": "", "blank.py": " \n",
                    "test_root.py": "included", "src/tests/a.py": "excluded",
                    "src/TEST/a.py": "excluded", "contest/a.py": "excluded by substring",
                    "src/a.PY": "excluded", "readme.md": "excluded",
                }
                for path, text in sources.items():
                    target = checkout / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(text.encode("utf-8"))
                repo.index.add(list(sources))
                with repo.config_writer() as config:
                    config.set_value("user", "name", "Preparation Test")
                    config.set_value("user", "email", "preparation@example.invalid")
                # Disable checkout transformations; committed blob bytes are the evidence.
                buggy = repo.index.commit("buggy")
                (checkout / "src/a.py").write_text("fixed", encoding="utf-8")
                repo.index.add(["src/a.py"])
                fixed = repo.index.commit("fixed")
                (checkout / "src/a.py").write_text("developer's uncommitted work", encoding="utf-8")
                (checkout / "untracked.txt").write_text("keep me", encoding="utf-8")
                status = repo.git.status("--porcelain")

                result = preparation.prepare_snapshot(repo, "local__fixture", buggy.hexsha[:10], root / "cache")
                baseline = read_content(repo, buggy.hexsha)
                self.assertEqual(result.candidate_paths, sorted(baseline))
                self.assertEqual(result.candidate_paths, ["blank.py", "empty.py", "src/a.py", "test_root.py"])
                self.assertEqual(result.metadata["base_sha"], buggy.hexsha)
                for chunk in result.chunks:
                    lines = preparation.source_lines(baseline[chunk["path"]])
                    self.assertEqual(chunk["text"], "".join(lines[chunk["start_line"] - 1:chunk["end_line"]]))
                self.assertEqual(repo.head.commit.hexsha, fixed.hexsha)
                self.assertEqual(repo.git.status("--porcelain"), status)
                self.assertEqual((checkout / "src/a.py").read_text(encoding="utf-8"), "developer's uncommitted work")
                self.assertEqual((checkout / "untracked.txt").read_text(encoding="utf-8"), "keep me")
            finally:
                repo.close()

    def test_baseline_non_utf8_failure_is_preserved(self):
        from git import Repo

        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            repo = Repo.init(root / "repository")
            try:
                (Path(repo.working_tree_dir) / "bad.py").write_bytes(b"\xff")
                repo.index.add(["bad.py"])
                with repo.config_writer() as config:
                    config.set_value("user", "name", "Preparation Test")
                    config.set_value("user", "email", "preparation@example.invalid")
                commit = repo.index.commit("non-UTF-8")
                with self.assertRaisesRegex(ValueError, "Non-UTF-8 source file"):
                    preparation.prepare_snapshot(repo, "local__fixture", commit.hexsha, root / "cache")
                self.assertFalse(list((root / "cache").glob("*/metadata.json")))
            finally:
                repo.close()


if __name__ == "__main__":
    unittest.main()
