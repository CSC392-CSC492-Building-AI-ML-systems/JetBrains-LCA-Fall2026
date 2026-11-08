"""Step-3 retrieval, resume, provenance, label isolation, and evaluation checks."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bug_localization.baseline_eval import evaluate_tfidf
from bug_localization.rag import bm25_index, evaluate_retrieval as evaluation
from bug_localization.rag import prepare_repositories as preparation
from bug_localization.rag import retrieval, run_retrieval as runner


class QueryTests(unittest.TestCase):
    def test_title_body_separator_and_content_preservation(self):
        title = "loadUserProfile fails"
        body = "Traceback:\r\n  File parser.py\r\n```python\n    parse_expression(x)\n```\n"
        self.assertEqual(retrieval.make_query({"issue_title": title, "issue_body": body}),
                         title + "\n\n" + body)

    def test_missing_null_and_blank_fields(self):
        for fields, expected in (({}, ""), ({"issue_title": None, "issue_body": ""}, ""),
                                 ({"issue_title": " \n", "issue_body": "body"}, "body"),
                                 ({"issue_title": "title", "issue_body": None}, "title")):
            self.assertEqual(retrieval.make_query(fields), expected)
        with self.assertRaises(ValueError):
            retrieval.make_query({"issue_body": 123})


class GuardedRow(dict):
    """Fail immediately if prediction code touches ANY non-allowlisted field."""
    def __getitem__(self, key):
        if key not in retrieval.PREDICTION_FIELDS:
            raise AssertionError(f"Prediction read forbidden field: {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key not in retrieval.PREDICTION_FIELDS:
            raise AssertionError(f"Prediction read forbidden field: {key}")
        return super().get(key, default)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "prepared"
        self.indexes = self.root / "indexes"
        self.settings = runner.Settings(window=1, stride=1)
        self.files = {
            "a.py": "alpha alpha\nalpha alpha\n", "b.py": "alpha beta\n",
            "empty.py": "", "filtered.py": "if x: pass\n", "z.py": "gamma\n",
        }
        self.sha = "a" * 40
        self.snapshot = self.prepare(self.files)
        self.rows = [{
            "repo_owner": "owner", "repo_name": "repo", "base_sha": self.sha,
            "issue_url": "https://example.invalid/issue/1", "issue_title": "alpha",
            "issue_body": None, "changed_files": ["a.py", "b.py", "tests/outside.py"],
            "head_sha": "FORBIDDEN", "diff": "FORBIDDEN", "commit_message": "FORBIDDEN",
        }]

    def prepare(self, files, owner="owner__repo", sha=None):
        repository = SimpleNamespace(commit=lambda ref: SimpleNamespace(hexsha=sha or self.sha))
        with patch.object(preparation, "read_content", return_value=files):
            return preparation.prepare_snapshot(repository, owner, sha or self.sha,
                                                self.prepared, window=1, stride=1)

    def index(self):
        return bm25_index.build_or_load_index(self.snapshot, self.indexes)

    def predict(self, rows=None, **kwargs):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner.run_examples(
                self.rows if rows is None else rows, self.root / "runs",
                settings=kwargs.pop("settings", self.settings),
                prepared_cache=self.prepared, index_cache=self.indexes, **kwargs)

    def result(self, summary_path):
        summary = json.loads(summary_path.read_bytes())
        return json.loads((summary_path.parent / summary["result_files"][0]).read_bytes())


class RetrievalTests(Fixture):
    def test_metadata_order_multiple_chunks_and_max_aggregation(self):
        index = self.index()
        prediction = retrieval.retrieve(index, "alpha", 30)
        chunks = prediction["retrieved_chunks"]
        self.assertEqual([c["rank"] for c in chunks], [1, 2, 3])
        self.assertEqual([c["path"] for c in chunks].count("a.py"), 2)
        self.assertEqual([c["chunk_id"] for c in chunks[:2]], [0, 1])
        for chunk in chunks:
            self.assertEqual({k: chunk[k] for k in self.snapshot.chunks[chunk["chunk_id"]]},
                             self.snapshot.chunks[chunk["chunk_id"]])
        scores = {r["file"]: r["score"] for r in prediction["ranking"]}
        self.assertEqual(scores["a.py"], max(c["score"] for c in chunks if c["path"] == "a.py"))
        self.assertNotEqual(scores["a.py"], sum(c["score"] for c in chunks if c["path"] == "a.py"))
        self.assertEqual(prediction["distinct_retrieved_files"], 2)

    def test_full_ranking_independent_of_limit_and_includes_all_candidates(self):
        index = self.index()
        small = retrieval.retrieve(index, "alpha", 1)
        full = retrieval.retrieve(index, "alpha", 100)
        zero = retrieval.retrieve(index, "alpha", 0)
        self.assertEqual(small["ranking"], full["ranking"])
        self.assertEqual(zero["ranking"], full["ranking"])
        self.assertEqual(small["distinct_retrieved_files"], 1)
        self.assertEqual(zero["retrieved_chunks"], [])
        self.assertGreater(next(r["score"] for r in small["ranking"] if r["file"] == "b.py"), 0)
        self.assertEqual({r["file"] for r in small["ranking"]}, set(self.files))
        self.assertEqual(len(small["ranking"]), len(self.files))
        self.assertEqual([r["file"] for r in small["ranking"] if r["score"] == 0],
                         ["empty.py", "filtered.py", "z.py"])

    def test_empty_unmatched_and_filtered_queries_have_path_order(self):
        for query in ("", "unknownterm", "if x: pass"):
            result = retrieval.retrieve(self.index(), query)
            self.assertEqual(result["retrieved_chunks"], [])
            self.assertEqual([r["file"] for r in result["ranking"]], sorted(self.files))
            self.assertTrue(all(r["score"] == 0 for r in result["ranking"]))

    def test_positive_file_score_ties_use_paths(self):
        snapshot = self.prepare({"z.py": "alpha\n", "a.py": "alpha\n"}, owner="ties__repo")
        result = retrieval.retrieve(bm25_index.ChunkBM25Index(snapshot), "alpha")
        self.assertEqual([r["file"] for r in result["ranking"]], ["a.py", "z.py"])

    def test_no_indexable_chunks_still_produces_complete_file_ranking(self):
        snapshot = self.prepare({"empty.py": "", "filtered.py": "if x: pass\n"}, owner="empty__repo")
        result = retrieval.retrieve(bm25_index.ChunkBM25Index(snapshot), "alpha")
        self.assertEqual(result["retrieved_chunks"], [])
        self.assertEqual(result["ranking"], [
            {"rank": 1, "file": "empty.py", "score": 0},
            {"rank": 2, "file": "filtered.py", "score": 0}])

    def test_invalid_max_chunks_and_settings(self):
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                retrieval.retrieve(self.index(), "alpha", value)
            with self.assertRaises(ValueError):
                runner.Settings(max_chunks=value).validate()


class RunnerTests(Fixture):
    def test_predictions_do_not_access_labels_or_fix_information(self):
        guarded = GuardedRow(self.rows[0])
        summary_path = self.predict([guarded])
        result = self.result(summary_path)
        serialized = json.dumps(result)
        for forbidden in ("changed_files", "correct_files", "head_sha", "FORBIDDEN", "tests/outside.py"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(result["query"], "alpha")
        self.assertEqual(result["requested_max_chunks"], 30)

    def test_arrow_projection_occurs_before_materializing_rows(self):
        class Columns:
            column_names = list(self.rows[0])
            _fingerprint = "fixture"

            def __len__(other):
                return 1

            def select_columns(other, columns):
                self.assertEqual(columns, list(retrieval.PREDICTION_FIELDS))
                return [GuardedRow(self.rows[0])]

            def __getitem__(other, index):
                raise AssertionError("Read unprojected row")

        self.predict(Columns())

    def test_full_sha_snapshot_and_index_reuse_without_git(self):
        example = retrieval.project_example(self.rows[0])
        with patch.object(preparation, "get_repository", side_effect=AssertionError("unnecessary Git")):
            first = runner.load_matching_index(example, self.settings, self.prepared, self.indexes)
            second = runner.load_matching_index(example, self.settings, self.prepared, self.indexes)
        self.assertEqual(second.snapshot.metadata["base_sha"], self.sha)
        self.assertEqual(second.snapshot.metadata["repository"], "owner__repo")
        self.assertEqual(first.cache_path, second.cache_path)
        self.assertTrue(second.reused)

    def test_abbreviated_sha_is_resolved_using_step_one(self):
        row = {**self.rows[0], "base_sha": self.sha[:8]}
        repository = Mock()
        repository.commit.return_value.hexsha = self.sha
        with patch.object(preparation, "get_repository", return_value=repository) as loader:
            index = runner.load_matching_index(row, self.settings, self.prepared, self.indexes)
        self.assertEqual(index.snapshot.metadata["base_sha"], self.sha)
        loader.assert_called_once_with({k: row[k] for k in ("repo_owner", "repo_name", "base_sha")})
        repository.close.assert_called_once()

    def test_different_revisions_and_repositories_never_share_index(self):
        alternate = self.prepare({"other.py": "delta\n"}, sha="b" * 40)
        other_repo = self.prepare({"other.py": "epsilon\n"}, owner="another__repo")
        first = runner.load_matching_index(self.rows[0], self.settings, self.prepared, self.indexes)
        second = runner.load_matching_index({**self.rows[0], "base_sha": "b" * 40},
                                            self.settings, self.prepared, self.indexes)
        third = runner.load_matching_index({**self.rows[0], "repo_owner": "another"},
                                           self.settings, self.prepared, self.indexes)
        self.assertEqual(second.snapshot.metadata, alternate.metadata)
        self.assertEqual(third.snapshot.metadata, other_repo.metadata)
        self.assertEqual(len({i.cache_path for i in (first, second, third)}), 3)
        self.assertEqual(second.search("alpha"), [])

    def test_rejects_mismatched_snapshot_and_index_provenance(self):
        other = self.prepare({"other.py": "delta\n"}, owner="another__repo")
        with patch.object(preparation, "load_cache", return_value=other), self.assertRaisesRegex(ValueError, "provenance"):
            runner.load_matching_index(self.rows[0], self.settings, self.prepared, self.indexes)
        wrong_index = bm25_index.build_or_load_index(other, self.indexes)
        with patch.object(bm25_index, "build_or_load_index", return_value=wrong_index), self.assertRaisesRegex(ValueError, "identity"):
            runner.load_matching_index(self.rows[0], self.settings, self.prepared, self.indexes)

    def test_resume_identical_prediction_without_rescoring(self):
        path = self.predict()
        before = self.result(path)
        with patch.object(runner, "retrieve", side_effect=AssertionError("rescored")):
            second = self.predict()
        self.assertEqual(path, second)
        self.assertEqual(before, self.result(second))
        self.assertEqual(json.loads(path.read_bytes())["reused"], 1)

    def test_resume_invalidates_changed_settings_query_and_source(self):
        first = self.predict()
        second = self.predict(settings=runner.Settings(max_chunks=1, window=1, stride=1))
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.result(second)["retrieved_chunks"]), 1)
        third = self.predict(settings=runner.Settings(window=1, stride=1, k1=2))
        self.assertNotEqual(first, third)
        with patch.object(runner.tokenizer, "TOKENIZER_VERSION", 99):
            fourth = self.predict()
        self.assertNotEqual(first, fourth)
        changed = [{**self.rows[0], "issue_title": "gamma"}]
        fifth = self.predict(changed)
        self.assertEqual(json.loads(fifth.read_bytes())["reused"], 0)
        self.assertEqual(self.result(fifth)["query"], "gamma")
        self.assertNotEqual(self.result(fifth)["signature"], self.result(second)["signature"])
        # Even unchanged requested provenance cannot reuse a prediction after
        # the underlying prepared evidence was rebuilt with different content.
        (self.snapshot.cache_path / "metadata.json").unlink()
        self.prepare({"other.py": "alpha\n"})
        sixth = self.predict()
        self.assertEqual(self.result(sixth)["candidate_count"], 1)
        self.assertEqual(json.loads(sixth.read_bytes())["reused"], 0)

    def test_incomplete_corrupt_or_nonresumed_output_is_regenerated(self):
        path = self.predict()
        result_path = path.parent / "examples/000.json"
        result_path.write_text('{"status":"success"}', encoding="utf-8")
        self.predict()
        self.assertEqual(self.result(path)["candidate_count"], 5)
        result_path.unlink()
        self.predict()
        self.assertTrue(result_path.exists())
        with patch.object(runner, "retrieve", wraps=retrieval.retrieve) as retriever:
            self.predict(resume=False)
        retriever.assert_called_once()

    def test_failure_is_recorded_then_retried_and_summary_covers_every_example(self):
        rows = [self.rows[0], {**self.rows[0], "repo_owner": "missing"}]
        with patch.object(preparation, "get_repository", side_effect=OSError("missing snapshot")):
            path = self.predict(rows, indices=[0, 1])
        summary = json.loads(path.read_bytes())
        self.assertEqual((summary["successful"], summary["failed"], summary["processed"]), (1, 1, 2))
        failed = json.loads((path.parent / "examples/001.json").read_bytes())
        self.assertEqual(failed["status"], "failed")
        self.assertIn("missing snapshot", failed["error"])
        self.prepare(self.files, owner="missing__repo")
        self.predict(rows, indices=[0, 1])
        summary = json.loads(path.read_bytes())
        self.assertEqual((summary["successful"], summary["failed"], summary["reused"]), (2, 0, 1))


class EvaluationTests(Fixture):
    def test_candidate_macro_denominators_and_distinct_file_counts(self):
        rows = [self.rows[0], {**self.rows[0], "changed_files": ["a.py"]}]
        settings = runner.Settings(max_chunks=1, window=1, stride=1)
        path = self.predict(rows, indices=[0, 1], settings=settings)
        metrics = evaluation.evaluate_run(path, rows)
        self.assertEqual(metrics["candidate_file_recall_macro"], .75)
        self.assertEqual(metrics["fraction_at_least_one_eligible_retrieved"], 1)
        self.assertEqual(metrics["fraction_all_eligible_retrieved"], .5)
        self.assertEqual(metrics["distinct_retrieved_files_mean"], 1)
        self.assertEqual(metrics["candidate_recall_applicable_examples"], 2)

    def test_synthetic_candidate_recall_and_outside_paths(self):
        prediction = {
            "example_index": 0, "requested_max_chunks": 30,
            "ranking": [{"file": "a.py"}, {"file": "b.py"}, {"file": "c.py"}],
            "retrieved_chunks": [{"path": "a.py"}, {"path": "a.py"}, {"path": "c.py"}],
        }
        metrics = evaluation.candidate_metrics(prediction, ["a.py", "b.py", "tests/outside.py"])
        self.assertEqual(metrics["candidate_file_recall"], .5)
        self.assertEqual(metrics["distinct_retrieved_files"], 2)
        self.assertEqual(metrics["retrieved_chunk_count"], 3)
        self.assertEqual(metrics["ground_truth_outside_candidates"], ["tests/outside.py"])
        self.assertEqual(metrics["at_least_one_eligible_retrieved"], 1)
        self.assertEqual(metrics["all_eligible_retrieved"], 0)
        metrics = evaluation.candidate_metrics(prediction, ["outside.py"])
        self.assertIsNone(metrics["candidate_file_recall"])
        self.assertIsNone(metrics["all_eligible_retrieved"])

    def test_evaluation_reuses_unchanged_metrics_and_writes_labels_separately(self):
        path = self.predict()
        original_prediction = (path.parent / "examples/000.json").read_bytes()
        metrics = evaluation.evaluate_run(path, self.rows)
        self.assertEqual(metrics["candidate_file_recall_macro"], 1)
        self.assertEqual(metrics["ground_truth_outside_candidates_total"], 1)
        self.assertEqual(metrics["prediction_failures"], 0)
        self.assertEqual(metrics["evaluation_failures"], 0)
        # Gold outside candidates remains in AP's denominator: two correct of three.
        self.assertAlmostEqual(metrics["file_ranking_metrics"]["MAP_full_ranking"], 2 / 3)
        self.assertEqual((path.parent / "examples/000.json").read_bytes(), original_prediction)
        compatible = evaluate_tfidf.evaluate(path.parent / "evaluation/summary.json", allow_partial=True)
        self.assertEqual(compatible["MAP_full_ranking"], metrics["file_ranking_metrics"]["MAP_full_ranking"])

    def test_failed_predictions_and_evaluation_errors_are_reported(self):
        rows = [self.rows[0], {**self.rows[0], "repo_owner": "missing"}]
        with patch.object(preparation, "get_repository", side_effect=OSError("missing snapshot")):
            path = self.predict(rows, indices=[0, 1])
        metrics = evaluation.evaluate_run(path, rows)
        self.assertEqual(metrics["prediction_failures"], 1)
        self.assertEqual(metrics["coverage"], .5)
        invalid_labels = copy.deepcopy(rows)
        invalid_labels[0]["changed_files"] = []
        metrics = evaluation.evaluate_run(path, invalid_labels)
        self.assertEqual(metrics["evaluation_failures"], 1)
        self.assertEqual(metrics["evaluated_examples"], 0)
        self.assertIsNone(metrics["file_ranking_metrics"])

    def test_zero_eligible_gold_is_explicit_and_not_vacuously_successful(self):
        rows = [{**self.rows[0], "changed_files": ["outside.py"]}]
        metrics = evaluation.evaluate_run(self.predict(rows), rows)
        self.assertEqual(metrics["examples_without_eligible_ground_truth"], 1)
        self.assertEqual(metrics["candidate_recall_applicable_examples"], 0)
        self.assertIsNone(metrics["candidate_file_recall_macro"])
        self.assertIsNone(metrics["fraction_all_eligible_retrieved"])
        self.assertEqual(metrics["file_ranking_metrics"]["MAP_full_ranking"], 0)

    def test_wrong_evaluation_row_and_corrupt_prediction_are_failures(self):
        path = self.predict()
        wrong = [{**self.rows[0], "issue_title": "another query"}]
        metrics = evaluation.evaluate_run(path, wrong)
        self.assertEqual(metrics["evaluation_failures"], 1)
        prediction_path = path.parent / "examples/000.json"
        prediction = json.loads(prediction_path.read_bytes())
        prediction["ranking"][0]["score"] = 999
        prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
        metrics = evaluation.evaluate_run(path, self.rows)
        self.assertEqual(metrics["evaluation_failures"], 1)


class CliTests(Fixture):
    def test_subset_and_optional_evaluation(self):
        args = ["--window", "1", "--stride", "1", "--max-chunks", "1", "--evaluate",
                "--output-dir", str(self.root / "cli"), "--prepared-cache-dir", str(self.prepared),
                "--index-cache-dir", str(self.indexes)]
        with patch.object(runner, "load_dataset_input", return_value=self.rows), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(runner.main(args), 0)
        self.assertIn("chunks=1/1 distinct_files=1", output.getvalue())
        self.assertIn("candidate_file_recall_macro: 0.500000", output.getvalue())


if __name__ == "__main__":
    unittest.main()
