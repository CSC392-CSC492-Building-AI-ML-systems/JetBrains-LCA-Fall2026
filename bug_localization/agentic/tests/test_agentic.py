import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from git import Repo

from bug_localization.agentic import loop
from bug_localization.agentic.evaluate import compare_runs
from bug_localization.agentic.run import run_examples
from bug_localization.agentic.tests.fixtures import GuardedRow, InvestigatingClient, SequenceClient, make_repository
from bug_localization.baseline_eval.evaluate_tfidf import score_example
from bug_localization.long_context import experiment
from bug_localization.long_context.run import run_examples as run_context
from bug_localization.rag.model import ModelResponse, ModelSettings
from bug_localization.rag.ranking_prompt import TokenCounter
from bug_localization.rag.retrieval import project_example
from bug_localization.rag.run_ranking import read_checked


class AgenticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo_path, self.row = make_repository(self.root / "repo")
        patcher = patch.object(experiment, "get_repository", side_effect=lambda _: Repo(self.repo_path))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.settings = ModelSettings(context_window=16000, max_output_tokens=500, tokenizer="bytes")
        self.counter = TokenCounter(self.settings.model_name, "bytes")
        self.prepared = loop.prepare_input(GuardedRow(self.row), 4, 2)

    def predict(self, client, **kwargs):
        return loop.predict(self.prepared, 1, self.settings, self.counter, client=client, window=4, stride=2, **kwargs)

    def test_follow_up_discovery_and_base_only(self):
        result = self.predict(InvestigatingClient())
        self.assertEqual([t["action"]["action"] for t in result["trace"]],
                         ["search_code", "search_code", "read_file", "submit_ranking"])
        self.assertIn("resolve_alias", result["trace"][0]["tool_result"]["snippets"][0]["text"])
        self.assertEqual(result["trace"][1]["action"]["query"], "resolve_alias")
        self.assertEqual(result["ranking"][0]["file"], "symbols.py")
        self.assertFalse(result["bm25_fallback"])
        self.assertEqual(result["cumulative_usage"]["total_tokens"], 480)
        text = json.dumps(result)
        for forbidden in ("FORBIDDEN_FIX_HEAD", "FORBIDDEN_PATCH", "FORBIDDEN_TEST", "changed_files", self.row["head_sha"]):
            self.assertNotIn(forbidden, text)
        scored = score_example({"status": "success", "ranking": result["ranking"],
                                "correct_files": self.row["changed_files"]})
        self.assertEqual(scored["average_precision"], .5)  # excluded gold remains a miss

    def test_invalid_paths_and_ranges(self):
        cases = [dict(path="../symbols.py", start_line=1, end_line=1),
                 dict(path="tests/excluded.py", start_line=1, end_line=1),
                 dict(path="C:/symbols.py", start_line=1, end_line=1),
                 dict(path="symbols.py", start_line=0, end_line=1),
                 dict(path="symbols.py", start_line=2, end_line=1),
                 dict(path="symbols.py", start_line=1, end_line=3),
                 dict(path="symbols.py", start_line=True, end_line=1)]
        for args in cases:
            with self.subTest(args=args):
                client = SequenceClient({"decision": {"action": "read_file", **args}},
                                        {"decision": {"action": "submit_ranking", "paths": ["symbols.py"]}})
                result = self.predict(client)
                self.assertIn("error", result["trace"][0]["tool_result"])
                self.assertIn("tool_result", client.calls[1][-1]["content"])
                self.assertEqual(result["tool_calls"], 1)

    def test_budget_exhaustion_final_ranking_tools_disabled(self):
        client = SequenceClient({"decision": {"action": "search_code", "query": "crash"}},
                                {"ranked_files": ["symbols.py"]})
        result = self.predict(client, limits=loop.Limits(tool_calls=1))
        self.assertEqual(result["stop_reason"], "tool_budget_exhausted")
        self.assertEqual(result["trace"][-1]["phase"], "final")
        self.assertIn("Tool actions are disabled", client.calls[-1][0]["content"])
        self.assertIn("resolve_alias", client.calls[-1][-2]["content"])
        self.assertFalse(result["bm25_fallback"])
        zero = self.predict(SequenceClient({"ranked_files": ["router.py"]}), limits=loop.Limits(tool_calls=0))
        self.assertEqual(zero["tool_calls"], 0)
        self.assertEqual(len(zero["trace"]), 1)

    def test_fallback_and_completion(self):
        for value in (ModelResponse("bad JSON"), ModelResponse(None, error={"type": "Unavailable"}),
                      {"decision": {"action": "submit_ranking", "paths": ["outside.py"]}}):
            with self.subTest(value=value):
                result = self.predict(SequenceClient(value))
                self.assertTrue(result["bm25_fallback"])
                self.assertEqual(result["ranking"], result["bm25_ranking"])
        result = self.predict(SequenceClient({"decision": {"action": "submit_ranking", "paths": ["symbols.py", "symbols.py", "outside.py"]}}))
        self.assertEqual(result["status"], "repaired")
        self.assertFalse(result["bm25_fallback"])
        self.assertEqual(len(result["ranking"]), len(self.prepared["files"]))

    def test_whole_line_output_and_total_context_bounds(self):
        prepared = {**self.prepared, "files": {**self.prepared["files"], "large.py": "x = '" + "x" * 4000 + "'\nshort = 1\n"}}
        # No single huge line can escape the tool token limit.
        result = loop.predict(prepared, 1, self.settings, self.counter,
                              client=SequenceClient({"decision": {"action": "read_file", "path": "large.py", "start_line": 1, "end_line": 2}},
                                                    {"decision": {"action": "submit_ranking", "paths": ["symbols.py"]}}),
                              limits=loop.Limits(tool_output_tokens=256), window=4, stride=2)
        self.assertLessEqual(result["trace"][0]["tool_output_tokens"], 256)
        for turn in result["trace"]:
            self.assertLessEqual(turn["input_tokens_estimated"], self.settings.input_budget)
        tiny = replace(self.settings, context_window=900)
        result = loop.predict(self.prepared, 1, tiny, self.counter, client=SequenceClient())
        self.assertEqual(result["status"], "fallback_context")
        self.assertEqual(result["trace"], [])

    def test_saved_result_existing_evaluator_and_matched_comparison(self):
        agent = run_examples([GuardedRow(self.row)], 1, self.root / "agent", self.settings,
                             window=4, stride=2, client=InvestigatingClient(), dataset_name="fixture")
        rag = run_context([GuardedRow(self.row)], "rag", 1, self.root / "rag", self.settings,
                          limit=1, split="train", window=4, stride=2,
                          client=SequenceClient({"ranked_files": ["symbols.py"]}), dataset_name="fixture")
        report = compare_runs([agent, rag], [self.row], self.root / "scores", allow_partial=True)
        self.assertEqual(report["runs"][0]["metrics"]["MAP_full_ranking"], .5)
        self.assertEqual(report["runs"][0]["resources_common"]["cumulative_total_tokens"], 480)
        summary = json.loads(agent.read_bytes())
        saved = read_checked(agent.parent / summary["result_files"][0])
        self.assertEqual(saved["example_input"], project_example(self.row))
        with self.assertRaises(ValueError):
            compare_runs([agent], [self.row], self.root / "not_full")


if __name__ == "__main__":
    unittest.main()
