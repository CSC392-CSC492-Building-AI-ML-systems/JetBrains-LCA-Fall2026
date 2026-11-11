import unittest

from agent.llm import ScriptedLLM
from agent.loop import parse_action, plan, task_message
from agent.provider import assemble_context, make_agent_provider_factory
from agent.tools import Limits, RepoTools, json_text
from data_classes.datapoint_base import DatapointBase
from rag.builder import RagContextBuilder

LANG, META = "LANGSEP\n", "METASEP\n"
SECRET = "SECRETANSWER"


def count(text: str) -> int:
    return len(text.split())


TARGET = f"alpha_func beta_func\n{SECRET} alpha_func gamma_func\nlast line"
FILES = {
    "app/target.txt": TARGET,
    "app/a.txt": "alpha_func beta_func gamma_func",
    "app/b.txt": "alpha_func only here",
    "lib/c.txt": "unrelated words only",
    "lib/d.txt": " ".join(f"w{i}" for i in range(50)),
}


def make_builder():
    return RagContextBuilder(FILES, "app/target.txt", "repo", count, lang_sep=LANG, meta_sep=META)


def make_datapoint():
    return DatapointBase(repo_id=1, repo_name="repo", completion_lines={}, context_dict=dict(FILES),
                         completion_dict={"app/target.txt": TARGET})


def submit(*items):
    return json_text({"action": "submit_context", "items": list(items)})


class ParseTests(unittest.TestCase):
    def test_finds_json_inside_noise(self):
        self.assertEqual(parse_action('Sure! {"action":"search_code","query":"x"} done'),
                         {"action": "search_code", "query": "x"})

    def test_none_without_an_object(self):
        self.assertIsNone(parse_action("no json here"))
        self.assertIsNone(parse_action("[1, 2]"))


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tools = RepoTools(make_builder(), Limits(), count)

    def test_search_never_returns_the_completion_file(self):
        out = self.tools.search_code("alpha_func beta_func SECRETANSWER")
        self.assertTrue(out["results"])
        self.assertTrue(all(r["path"] != "app/target.txt" for r in out["results"]))
        self.assertNotIn(SECRET, json_text(out))

    def test_read_file_refuses_the_completion_file_and_bad_ranges(self):
        self.assertIn("error", self.tools.read_file("app/target.txt", 1, 1))
        self.assertIn("error", self.tools.read_file("app/a.txt", 0, 1))
        self.assertIn("error", self.tools.read_file("app/a.txt", 1, 99))
        self.assertIn("error", self.tools.read_file("nope.txt", 1, 1))
        self.assertEqual(self.tools.read_file("app/a.txt", 1, 1)["text"], "alpha_func beta_func gamma_func")

    def test_long_ranges_are_refused(self):
        tools = RepoTools(make_builder(), Limits(read_lines=1), count)
        self.assertIn("error", tools.read_file("app/a.txt", 1, 2))

    def test_find_definition_unknown_name(self):
        self.assertIn("error", self.tools.find_definition("nothing"))

    def test_find_definition_known_name(self):
        self.tools.builder.symbol_index["alpha_func"] = [0]
        self.assertEqual(len(self.tools.find_definition("alpha_func")["results"]), 1)

    def test_only_shown_chunks_can_be_cited(self):
        shown = self.tools.search_code("alpha_func")["results"][0]["chunk"]
        unseen = next(i for i in range(len(self.tools.builder.chunks)) if i not in self.tools.seen)
        got = self.tools.resolve_items([{"chunk": unseen}, {"chunk": shown}])
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0], self.tools.builder.chunks[shown])

    def test_ranges_are_validated_deduplicated_and_capped(self):
        items = [{"path": "app/a.txt", "start_line": 1, "end_line": 1}, {"path": "app/a.txt", "start_line": 1, "end_line": 1},
                 {"path": "app/target.txt", "start_line": 1, "end_line": 1}, {"path": "app/a.txt", "start_line": 5, "end_line": 6},
                 "junk", {"chunk": "x"}]
        got = self.tools.resolve_items(items)
        self.assertEqual([(c.path, c.start_line, c.end_line) for c in got], [("app/a.txt", 0, 0)])
        many = RepoTools(make_builder(), Limits(max_items=1), count).resolve_items(
            [{"path": "app/b.txt", "start_line": 1, "end_line": 1}, {"path": "app/a.txt", "start_line": 1, "end_line": 1}])
        self.assertEqual(len(many), 1)


class LoopTests(unittest.TestCase):
    def run_plan(self, replies, limits=Limits()):
        tools = RepoTools(make_builder(), limits, count)
        llm = ScriptedLLM(replies)
        return plan("task", tools, llm, limits), llm

    def test_search_then_submit(self):
        result, llm = self.run_plan([json_text({"action": "search_code", "query": "alpha_func"}), submit({"chunk": 0})])
        self.assertEqual((result.status, result.steps, result.tool_calls), ("submitted", 2, 1))
        self.assertEqual(result.items, [{"chunk": 0}])
        self.assertIn("tool_result", llm.calls[1][-1]["content"])

    def test_immediate_submit_is_allowed(self):
        result, _ = self.run_plan([submit({"path": "app/a.txt", "start_line": 1, "end_line": 1})])
        self.assertEqual((result.status, result.steps, result.tool_calls), ("submitted", 1, 0))

    def test_tool_budget_forces_a_submit_request(self):
        search = json_text({"action": "search_code", "query": "alpha_func"})
        result, llm = self.run_plan([search, search, submit({"chunk": 0})], Limits(tool_calls=2))
        self.assertEqual((result.status, result.tool_calls, result.steps), ("submitted", 2, 3))
        self.assertIn("Reply now with a submit_context", llm.calls[2][-1]["content"])

    def test_planner_that_keeps_searching_after_the_budget_is_cut_off(self):
        search = json_text({"action": "search_code", "query": "alpha_func"})
        result, _ = self.run_plan([search], Limits(tool_calls=1))
        self.assertEqual(result.status, "no_submit")
        self.assertEqual(result.tool_calls, 1)

    def test_unusable_replies_end_the_run(self):
        result, _ = self.run_plan(["no json"], Limits(invalid_actions=2))
        self.assertEqual((result.status, result.steps), ("invalid_replies", 3))

    def test_empty_submit(self):
        result, _ = self.run_plan([submit()])
        self.assertEqual((result.status, result.items), ("no_items", []))

    def test_invalid_tool_arguments_cost_a_call_and_return_an_error(self):
        result, _ = self.run_plan([json_text({"action": "read_file", "path": "app/target.txt", "start_line": 1, "end_line": 1}),
                                   submit({"chunk": 0})])
        self.assertIn("error", result.trace[0]["tool_result"])
        self.assertEqual(result.tool_calls, 1)


class AssembleTests(unittest.TestCase):
    def setUp(self):
        self.builder = make_builder()
        self.tools = RepoTools(self.builder, Limits(), count)

    def chunks_for(self, *paths):
        return [c for p in paths for c in self.builder.chunks if c.path == p]

    def test_best_chunk_sits_nearest_the_prefix_and_budget_holds(self):
        first, second = self.chunks_for("app/a.txt")[0], self.chunks_for("app/b.txt")[0]
        context, used = assemble_context(self.builder, [first, second], "prefix words", 1000, count)
        self.assertEqual(used, [first, second])
        self.assertLess(context.index("alpha_func only here"), context.index("alpha_func beta_func gamma_func"))
        self.assertLessEqual(count(context + "\n" + "prefix words"), 1000)

    def test_a_chunk_that_does_not_fit_is_skipped_not_cut(self):
        big, small = self.chunks_for("lib/d.txt")[0], self.chunks_for("lib/c.txt")[0]
        context, used = assemble_context(self.builder, [big, small], "p", 12, count)
        self.assertEqual(used, [small])
        self.assertNotIn("w7", context)

    def test_max_context_tokens_caps_the_chunks(self):
        chunks = self.chunks_for("app/a.txt", "app/b.txt")
        _, used = assemble_context(self.builder, chunks, "p", 1000, count, max_context_tokens=5)
        self.assertLessEqual(len(used), 1)


class ProviderTests(unittest.TestCase):
    def provider(self, replies, **kwargs):
        llm = ScriptedLLM(replies)
        factory = make_agent_provider_factory(count, 1000, llm, lang_sep=LANG, meta_sep=META, **kwargs)
        return factory(make_datapoint()), llm

    def test_target_line_never_reaches_the_planner_or_the_prompt(self):
        provider, llm = self.provider([json_text({"action": "search_code", "query": "alpha_func"}), submit({"chunk": 0})])
        prompt, ground_truth, info = provider(make_datapoint(), 1)  # target is line index 1, which holds the secret
        self.assertEqual(ground_truth, f"{SECRET} alpha_func gamma_func")
        self.assertNotIn(SECRET, prompt)
        self.assertTrue(all(SECRET not in m["content"] for call in llm.calls for m in call))
        self.assertEqual((info["steps"], info["tool_calls"], info["planner_status"], info["fallback"]), (2, 1, "submitted", False))

    def test_planner_choice_ends_up_in_the_prompt_and_prefix_is_last(self):
        provider, _ = self.provider([submit({"path": "lib/c.txt", "start_line": 1, "end_line": 1})])
        prompt, _, info = provider(make_datapoint(), 2)
        self.assertIn("unrelated words only", prompt)
        self.assertTrue(prompt.endswith(f"\n{TARGET.split(chr(10))[0]}\n{TARGET.split(chr(10))[1]}"))
        self.assertEqual(info["paths"], ["lib/c.txt"])

    def test_nothing_usable_falls_back_to_rag_and_is_flagged(self):
        provider, _ = self.provider(["no json"])
        prompt, _, info = provider(make_datapoint(), 2)
        self.assertTrue(info["fallback"])
        self.assertEqual(info["planner_status"], "invalid_replies")
        self.assertIn("alpha_func", prompt)  # RAG's own BM25 context from the last lines

    def test_task_message_lists_files_and_prefix_only(self):
        text = task_message("app/target.txt", "some prefix", ["app/a.txt", "lib/c.txt"])
        self.assertIn("app/a.txt", text)
        self.assertIn("some prefix", text)


if __name__ == "__main__":
    unittest.main()
