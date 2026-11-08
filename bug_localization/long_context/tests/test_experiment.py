"""Offline checks: snapshots, packing, controlled inputs, outcomes and evaluation."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from git import Repo

from bug_localization.baseline_eval.evaluate_tfidf import aggregate, score_example
from bug_localization.long_context import compare, experiment, run
from bug_localization.long_context.prompt import build_messages, pack_prompt, source_tokens
from bug_localization.long_context.tests.fixtures import GuardedRow, MockClient, make_repository
from bug_localization.rag.model import ModelResponse, ModelSettings
from bug_localization.rag.ranking_prompt import TokenCounter
from bug_localization.rag.run_ranking import read_checked
from bug_localization.rag.run_retrieval import save_json


class PackingTests(unittest.TestCase):
    def setUp(self):
        self.settings = ModelSettings(context_window=12000, max_output_tokens=500, tokenizer="bytes")
        self.counter = TokenCounter("unused", "bytes")
        self.example = {"issue_title": "alpha failure", "issue_body": "body\n🚀"}
        self.files = {"z.py": "last\n", "a.py": "HUGE" * 5000, "b.py": "small\r\n", "empty.py": ""}

    def pack(self, mode="direct", budget=200, **kwargs):
        return pack_prompt(self.example, self.files, kwargs.pop("chunks", []), mode, budget,
                           kwargs.pop("settings", self.settings), self.counter)

    def test_whole_files_lexicographic_skip_oversized_and_continue(self):
        first = self.pack()
        self.assertEqual(first, self.pack())
        self.assertEqual(first["included_files"], ["b.py", "empty.py", "z.py"])
        self.assertEqual(first["omitted_files"], ["a.py"])
        data = json.loads(first["messages"][1]["content"])
        self.assertEqual([b["path"] for b in data["repository_source"]], first["included_files"])
        self.assertEqual(data["repository_source"][0]["text"], "small\r\n")
        self.assertEqual(data["allowed_candidate_paths"], sorted(self.files))
        self.assertLessEqual(first["actual_source_tokens"], 200)
        self.assertTrue(first["source_truncated"])

    def test_exact_source_budget_boundary(self):
        self.files = {"a.py": "alpha\n"}
        block = [{"path": "a.py", "text": "alpha\n"}]
        cost = source_tokens(block, self.counter)
        self.assertEqual(self.pack(budget=cost)["included_files"], ["a.py"])
        self.assertEqual(self.pack(budget=cost - 1)["included_files"], [])

    def test_request_limit_reserves_issue_candidates_schema_and_response(self):
        self.files = {"a.py": "alpha" * 1000, "b.py": "beta"}
        overhead = self.counter.count_request(build_messages(self.example, sorted(self.files), []), self.settings)
        settings = replace(self.settings, context_window=overhead + 1000, max_output_tokens=500)
        packed = self.pack(budget=100000, settings=settings)
        self.assertIn("context_limit", [d["reason"] for d in packed["unit_decisions"]])
        self.assertLessEqual(packed["total_reserved_tokens"], settings.context_window)
        self.assertEqual(packed["total_prompt_tokens"], self.counter.count_request(packed["messages"], settings))
        self.assertEqual(packed["reserved_output_tokens"], 500)
        self.assertGreaterEqual(packed["reserved_safety_tokens"], settings.context_window * .05)

    def test_identical_full_issue_candidates_and_instructions_across_modes(self):
        chunks = [{"path": "z.py", "text": "last\n", "chunk_id": 3, "score": 1,
                   "start_line": 1, "end_line": 1}]
        payloads = []
        for mode in ("direct", "rag", "issue_only"):
            packed = self.pack(mode=mode, budget=0 if mode == "issue_only" else 300, chunks=chunks)
            self.assertEqual(packed["messages"][0], self.pack()["messages"][0])
            data = json.loads(packed["messages"][1]["content"])
            data.pop("repository_source")
            payloads.append(data)
            if mode == "issue_only":
                self.assertEqual(packed["actual_source_tokens"], 0)
                self.assertEqual(packed["source_text_tokens"], 0)
                self.assertEqual(packed["included_files"], [])
        self.assertEqual(payloads, [payloads[0]] * 3)

    def test_oversized_fixed_prompt_is_preserved_and_marked_unfit(self):
        self.example["issue_body"] = "huge 🚀" * 10000
        packed = self.pack()
        self.assertFalse(packed["fits_context"])
        self.assertFalse(packed["issue_truncated"])
        self.assertFalse(packed["candidate_list_truncated"])
        self.assertEqual(packed["included_files"], [])
        self.assertEqual(json.loads(packed["messages"][1]["content"])["issue_body"], self.example["issue_body"])

    def test_candidate_list_alone_can_exhaust_context_without_dropping_paths(self):
        self.files = {f"source/{i:05d}_long_candidate_path.py": "" for i in range(500)}
        packed = self.pack(mode="issue_only", budget=0)
        self.assertFalse(packed["fits_context"])
        self.assertEqual(packed["candidate_paths_supplied"], 500)
        self.assertEqual(packed["allowed_candidate_paths"], sorted(self.files))
        self.assertEqual(packed["actual_source_tokens"], 0)

    def test_rag_score_order_ties_and_overlap_coverage(self):
        self.files = {"a.py": "one\ntwo\nthree\n"}
        chunks = [{"path": "a.py", "text": "two\nthree\n", "chunk_id": 1,
                   "score": 2, "start_line": 2, "end_line": 3},
                  {"path": "a.py", "text": "one\ntwo\n", "chunk_id": 0,
                   "score": 2, "start_line": 1, "end_line": 2}]
        packed = self.pack(mode="rag", budget=1000, chunks=chunks)
        self.assertEqual([d["chunk_id"] for d in packed["unit_decisions"]], [0, 1])
        self.assertEqual(packed["covered_source_lines"], 3)
        self.assertEqual(packed["source_line_coverage"], 1)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo_path, self.rows = make_repository(self.root / "fixture__repo")
        self.get_repository = patch.object(experiment, "get_repository", side_effect=lambda _: Repo(self.repo_path)).start()
        self.addCleanup(patch.stopall)
        self.settings = ModelSettings(context_window=12000, max_output_tokens=500, tokenizer="bytes")
        self.client = MockClient()

    def run_mode(self, mode="direct", budget=1000, rows=None, **kwargs):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return run.run_examples(self.rows if rows is None else rows, mode, budget, self.root / "runs",
                                    kwargs.pop("settings", self.settings), client=kwargs.pop("client", self.client), **kwargs)

    def result(self, path, index=0):
        return read_checked(path.parent / "examples" / f"{index:03d}.json")

    def test_base_snapshot_real_reader_filters_and_label_isolation(self):
        prepared = experiment.prepare_input(GuardedRow(self.rows[0]))
        self.assertEqual(prepared["resolved_base_sha"], self.rows[0]["base_sha"])
        self.assertEqual(list(prepared["files"]), ["a.py", "empty.py", "z.py"])
        self.assertEqual(prepared["files"]["a.py"], "alpha = 'alpha failure'\n")
        supplied = self.get_repository.call_args.args[0]
        self.assertEqual(set(supplied), {"repo_owner", "repo_name", "base_sha"})
        path = self.run_mode(rows=[GuardedRow(row) for row in self.rows])
        saved = json.dumps(self.result(path))
        for forbidden in ("changed_files", "head_sha", "FORBIDDEN", "FIXED_HEAD"):
            self.assertNotIn(forbidden, saved)

    def test_bm25_tail_and_candidate_set_fixed_even_with_zero_source(self):
        results = [self.result(self.run_mode(mode, 0 if mode == "issue_only" else 1000))
                   for mode in ("direct", "rag", "issue_only")]
        self.assertEqual([r["bm25_ranking"] for r in results], [results[0]["bm25_ranking"]] * 3)
        self.assertEqual([r["packing"]["allowed_candidate_paths"] for r in results],
                         [["a.py", "empty.py", "z.py"]] * 3)
        self.assertEqual(results[-1]["packing"]["actual_source_tokens"], 0)
        self.assertTrue(results[-1]["model_ranking_success"])
        # Zero retrieval matches/source still sends an issue-and-path request.
        self.rows[0]["issue_title"] = "unmatchedxyz"
        self.rows[0]["issue_body"] = ""
        before = len(self.client.calls)
        self.run_mode("rag", 0)
        self.assertEqual(len(self.client.calls), before + 2)

    def test_partial_invalid_and_failure_responses_are_distinct(self):
        for response, status, fallback in [
            (ModelResponse('{"ranked_files":["z.py","z.py","invented.py"]}'), "repaired", False),
            (ModelResponse("garbage"), "fallback_validation", True),
            (ModelResponse('{"ranked_files":["a.py"]}', finish_reason="length"), "fallback_validation", True),
            (ModelResponse(None, refusal="refusal"), "fallback_validation", True),
            (ModelResponse(None, error={"type": "Timeout"}), "fallback_request", True),
        ]:
            with self.subTest(status=status, response=response):
                client = Mock(generate=Mock(return_value=response))
                result = self.result(self.run_mode(client=client, resume=False))
                self.assertEqual(result["status"], status)
                self.assertEqual(result["bm25_fallback"], fallback)
                self.assertEqual(result["model_ranking_success"], not fallback)
                self.assertEqual(set(r["file"] for r in result["ranking"]), {"a.py", "empty.py", "z.py"})
                if fallback:
                    self.assertEqual(result["ranking"], result["bm25_ranking"])
                else:
                    self.assertEqual(result["model_ranked_paths"], ["z.py"])
                    tail = [r["file"] for r in result["bm25_ranking"] if r["file"] != "z.py"]
                    self.assertEqual([r["file"] for r in result["ranking"]], ["z.py", *tail])

    def test_request_exception_sanitization_and_retry_on_resume(self):
        client = Mock(generate=Mock(side_effect=TimeoutError("SECRET")))
        path = self.run_mode(client=client)
        self.assertNotIn("SECRET", json.dumps(self.result(path)))
        self.run_mode(client=client)
        self.assertEqual(client.generate.call_count, 4)

    def test_context_failure_never_calls_model_and_preserves_issue(self):
        self.rows[0]["issue_body"] = "large" * 10000
        path = self.run_mode()
        result = self.result(path)
        self.assertEqual(result["status"], "fallback_context")
        self.assertIsNone(result["submitted_messages"])
        self.assertEqual(len(self.client.calls), 1)  # Only the second example fits.

    def test_resume_and_corrupt_result_detection(self):
        first = self.run_mode()
        self.run_mode()
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual(json.loads(first.read_bytes())["reused"], 2)
        saved = self.result(first)
        saved["ranking"] = []
        run.checked_write(first.parent / "examples" / "000.json", saved)
        self.run_mode()
        self.assertEqual(len(self.client.calls), 3)
        new = self.run_mode(budget=999)
        self.assertNotEqual(first, new)
        self.assertEqual(len(self.client.calls), 5)

    def test_dry_run_no_calls_and_rejected_by_evaluator(self):
        path = self.run_mode(dry_run=True)
        self.assertEqual(self.client.calls, [])
        with self.assertRaises(ValueError):
            compare.compare_runs([path], self.rows, self.root / "comparison")
        self.assertNotEqual(path, self.run_mode())

    def test_shared_evaluator_retains_excluded_gold_as_miss(self):
        path = self.run_mode()
        output = compare.compare_runs([path], self.rows, self.root / "comparison")
        metrics = output["runs"][0]["available_metrics"]
        self.assertEqual(metrics["MAP_full_ranking"], .75)  # (1/2 + 1)/2
        self.assertEqual(metrics["MRR"], 1)
        self.assertEqual(metrics["unranked_correct_files_total"], 1)
        rows = [score_example({"status": "success", "correct_files": row["changed_files"],
                               "ranking": self.result(path, i)["ranking"]}) for i, row in enumerate(self.rows)]
        self.assertEqual(metrics, aggregate(rows))
        self.assertTrue((self.root / "comparison" / "comparison.md").is_file())

    def test_incomplete_runs_use_common_set_and_report_fallback_counts(self):
        direct = self.run_mode()
        client = Mock(generate=Mock(return_value=ModelResponse(None, error={"type": "Timeout"})))
        rag = self.run_mode("rag", client=client)
        summary = json.loads(rag.read_bytes())
        summary.update(result_files=summary["result_files"][:1], successful=1, processed=1, remaining=1,
                       status_counts={"fallback_request": 1})
        save_json(rag, summary)  # Simulate interruption after the first saved result.
        output = compare.compare_runs([direct, rag], self.rows, self.root / "comparison")
        self.assertEqual(output["common_example_indices"], [0])
        self.assertEqual(output["common_model_success_example_indices"], [])
        self.assertEqual([r["available_metrics"]["evaluated_examples"] for r in output["runs"]], [2, 1])
        self.assertEqual([r["common_metrics"]["evaluated_examples"] for r in output["runs"]], [1, 1])
        self.assertEqual(output["runs"][1]["fallbacks"], 1)
        self.assertEqual(output["runs"][1]["remaining_examples"], 1)

    def test_comparison_rejects_different_model_settings(self):
        direct = self.run_mode()
        rag = self.run_mode("rag", settings=replace(self.settings, temperature=.3))
        with self.assertRaises(ValueError):
            compare.compare_runs([direct, rag], self.rows, self.root / "comparison")

    def test_snapshot_failure_coverage_and_safe_failure_log(self):
        with patch.object(experiment, "get_repository", side_effect=ValueError("SECRET")):
            path = self.run_mode()
        summary = json.loads(path.read_bytes())
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["successful"], 0)
        self.assertIsNone(self.result(path)["actual_source_tokens"])
        self.assertNotIn("SECRET", json.dumps(self.result(path)))
        output = compare.compare_runs([path], self.rows, self.root / "comparison")
        self.assertIsNone(output["runs"][0]["common_metrics"])

    def test_cli_mocked_run_and_comparison(self):
        with patch.object(run, "load_dataset_input", return_value=self.rows), \
                patch.object(experiment, "OpenAIClient", return_value=self.client), \
                redirect_stdout(io.StringIO()):
            code = run.main(["--mode", "issue_only", "--tokenizer", "bytes", "--limit", "1",
                             "--output-dir", str(self.root / "cli")])
        self.assertEqual(code, 0)
        self.assertEqual(len(self.client.calls), 1)
        with patch.object(compare, "load_dataset_input", return_value=self.rows), redirect_stdout(io.StringIO()):
            code = compare.main([str(self.root / "cli"), "--output-dir", str(self.root / "cli_comparison")])
        self.assertEqual(code, 0)
        with patch.object(run, "load_dataset_input") as loader, redirect_stderr(io.StringIO()):
            self.assertEqual(run.main(["--mode", "issue_only", "--repo-token-budget", "1"]), 1)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
