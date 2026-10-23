"""End-to-end step-5 checks using prepared snapshots and a mock model."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from unittest.mock import Mock, patch

from bug_localization.rag import evaluate_benchmark, run_benchmark, run_retrieval
from bug_localization.rag.model import ModelResponse, ModelSettings
from bug_localization.rag.run_retrieval import save_json
from bug_localization.rag.tests.test_retrieval import Fixture, GuardedRow


class Dataset(list):
    _fingerprint = "benchmark-fixture"


class BenchmarkTests(Fixture):
    def setUp(self):
        super().setUp()
        self.dataset = Dataset([{**self.rows[0], "issue_url": f"https://example.invalid/{i}"} for i in range(50)])
        self.config = ModelSettings(context_window=10000, max_output_tokens=500, tokenizer="bytes")
        self.client = Mock()
        self.client.generate.return_value = ModelResponse(
            '{"ranked_files":["b.py","a.py"]}', usage={"prompt_tokens": 100},
            latency_seconds=.25, attempts=[{"attempt": 1}], finish_reason="stop")

    def run_pipeline(self, **kwargs):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return run_benchmark.run_benchmark(
                kwargs.pop("dataset", self.dataset), self.root / "benchmark", kwargs.pop("indices", [0, 1]),
                kwargs.pop("settings", self.config), self.settings, client=self.client,
                prepared_cache=self.prepared, index_cache=self.indexes, **kwargs)

    def test_pipeline_label_isolation_full_rankings_stable_ids_and_resume(self):
        guarded = Dataset(GuardedRow(row) for row in self.dataset)
        path = self.run_pipeline(dataset=guarded)
        first = self.result(path)
        self.assertEqual({r["file"] for r in first["ranking"]}, set(self.files))
        self.assertEqual(len(first["ranking"]), len(self.files))
        self.assertIn(first["example_id"], json.loads(path.read_bytes())["result_files"][0])
        self.assertEqual(self.run_pipeline(dataset=guarded), path)
        self.assertEqual(self.client.generate.call_count, 2)
        self.assertEqual(json.loads(path.read_bytes())["reused"], 2)
        self.assertFalse(list(path.parent.rglob(".writing-*")))

    def test_fallback_preserved_explicit_retry_and_configuration_isolation(self):
        self.client.generate.return_value = ModelResponse(None, error={"type": "fixture"})
        path = self.run_pipeline()
        self.assertEqual(json.loads(path.read_bytes())["completed_bm25_fallbacks"], 2)
        self.run_pipeline()
        self.assertEqual(self.client.generate.call_count, 2)
        self.run_pipeline(retry_fallbacks=True)
        self.assertEqual(self.client.generate.call_count, 4)
        self.assertEqual(len(self.result(path)["previous_attempts"]), 1)
        changed = self.run_pipeline(settings=replace(self.config, model_name="other-model"))
        self.assertNotEqual(path, changed)

    def test_failed_preparation_continues_and_retries_without_repeating_success(self):
        original = run_retrieval.load_matching_index
        def fail_one(example, *args):
            if example["issue_url"].endswith("/0"):
                raise RuntimeError("fixture")
            return original(example, *args)
        with patch.object(run_retrieval, "load_matching_index", side_effect=fail_one):
            path = self.run_pipeline()
        summary = json.loads(path.read_bytes())
        self.assertEqual((summary["failed"], summary["successful"], summary["remaining"]), (1, 1, 0))
        self.run_pipeline()
        self.assertEqual(self.client.generate.call_count, 2)
        self.assertEqual(json.loads(path.read_bytes())["failed"], 0)

    def test_incomplete_artifact_retried_and_completed_response_preserved(self):
        path = self.run_pipeline()
        summary = json.loads(path.read_bytes())
        damaged = path.parent / summary["result_files"][0]
        damaged.write_text('{"status": "running"', encoding="utf-8")
        self.run_pipeline()
        self.assertEqual(self.client.generate.call_count, 3)

    def test_dry_run_including_empty_evidence_is_never_evaluated(self):
        self.dataset[0]["issue_title"] = ""
        path = self.run_pipeline(dry_run=True)
        self.client.generate.assert_not_called()
        self.assertEqual(json.loads(path.read_bytes())["dry_run_prompts"], 2)
        self.assertEqual(self.result(path)["status"], "dry_run")
        with self.assertRaises(ValueError):
            evaluate_benchmark.evaluate(path, self.dataset)

    def test_partial_evaluation_fallbacks_raw_gold_and_missing_baseline(self):
        self.client.generate.return_value = ModelResponse("bad json", attempts=[{"attempt": 1}])
        path = self.run_pipeline()
        metrics = evaluate_benchmark.evaluate(path, self.dataset, self.root / "missing")
        self.assertFalse(metrics["complete_50_example_evaluation"])
        self.assertEqual(metrics["bm25_fallbacks"], 2)
        self.assertEqual(metrics["file_ranking_metrics"]["BM25"], metrics["file_ranking_metrics"]["BM25 + LLM RAG"])
        self.assertEqual(metrics["ground_truth_outside_candidates_total"], 2)
        coverage = metrics["retrieval_coverage_before_packing"]
        self.assertEqual(coverage["all_raw_gold"], 0)
        self.assertEqual(coverage["all_eligible_retrieved"], 1)
        self.assertTrue((path.parent / "comparison.csv").is_file())
        self.assertIn("unavailable", metrics["baseline_status"])

    def test_matched_baseline_common_examples_and_incompatibility(self):
        path = self.run_pipeline()
        evaluate_benchmark.evaluate(path, self.dataset)
        baseline = self.root / "tfidf"
        row = json.loads((path.parent / "evaluation/bm25/000.json").read_bytes())
        save_json(baseline / "examples/000.json", row)
        save_json(baseline / "run_metadata.json", {
            "dataset": run_benchmark.preparation.DATASET, "dataset_fingerprint": self.dataset._fingerprint,
            "label_policy": run_benchmark.LABEL_POLICY})
        save_json(baseline / "summary.json", {"configuration": "py", "split": "test", "successful": 1,
                  "label_policy": run_benchmark.LABEL_POLICY, "result_files": ["examples/000.json"]})
        metrics = evaluate_benchmark.evaluate(path, self.dataset, baseline)
        self.assertEqual(metrics["comparison_indices"], [0])
        self.assertEqual(metrics["comparison_metrics"]["TF-IDF"]["evaluated_examples"], 1)
        self.assertEqual(metrics["file_ranking_metrics"]["BM25"]["evaluated_examples"], 2)
        row["base_sha"] = "wrong"
        save_json(baseline / "examples/000.json", row)
        metrics = evaluate_benchmark.evaluate(path, self.dataset, baseline)
        self.assertIn("incompatible", metrics["baseline_status"])
        self.assertEqual(metrics["comparison_indices"], [])

    def test_failure_denominator_and_dataset_size_guard(self):
        with self.assertRaises(ValueError):
            self.run_pipeline(dataset=self.dataset[:1])
        original = run_retrieval.load_matching_index
        def fail_one(example, *args):
            if example["issue_url"].endswith("/1"):
                raise ValueError("fixture")
            return original(example, *args)
        with patch.object(run_retrieval, "load_matching_index", side_effect=fail_one):
            path = self.run_pipeline()
        metrics = evaluate_benchmark.evaluate(path, self.dataset)
        self.assertEqual(metrics["evaluated_examples"], 1)
        self.assertEqual(metrics["coverage"], .5)
        self.assertEqual(metrics["missing_prediction_indices"], [1])

    def test_complete_50_evaluation_and_evaluate_only_never_calls(self):
        path = self.run_pipeline(indices=range(50))
        metrics = evaluate_benchmark.evaluate(path, self.dataset)
        self.assertTrue(metrics["complete_50_example_evaluation"])
        self.assertEqual(metrics["evaluated_examples"], 50)
        with patch.object(run_benchmark.run_retrieval, "load_dataset_input", return_value=self.dataset), \
                patch.object(run_benchmark, "run_benchmark") as inference, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(run_benchmark.main(["--evaluate-only", str(path)]), 0)
        inference.assert_not_called()

    def test_atomic_write_failure_leaves_previous_json_intact(self):
        path = self.root / "atomic.json"
        save_json(path, {"completed": True})
        with patch("pathlib.Path.replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                save_json(path, {"completed": False})
        self.assertEqual(json.loads(path.read_bytes()), {"completed": True})
        self.assertFalse(list(self.root.glob(".writing-*")))

    def test_prompt_identity_changes_run_directory(self):
        first = self.run_pipeline(indices=[0], dry_run=True)
        with patch.object(run_benchmark, "prompt_identity", return_value={"version": "different"}):
            second = self.run_pipeline(indices=[0], dry_run=True)
        self.assertNotEqual(first, second)
