import unittest

from agent.llm import ReplayModel, ScriptedLLM
from agent.loop import parse_action, plan, task_message
from agent.provider import assemble_context, make_agent_provider_factory
from agent.tools import Limits, RepoTools, json_text
from data_classes.datapoint_base import DatapointBase
from rag.builder import RagContextBuilder
from rag.provider import make_rag_provider_factory

try:
    from tree_sitter_parser.parser import parser  # noqa: F401  (needs the grammar clone in the working directory)
    HAS_PARSER = True
except Exception:
    HAS_PARSER = False

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
RAG_OPTIONS = {"prefix_cap": 1000, "query_lines": 10, "max_chunks": 30, "continuation_lines": 0}


def make_builder(files=None, completion="app/target.txt"):
    return RagContextBuilder(files or FILES, completion, "repo", count, lang_sep=LANG, meta_sep=META)


def make_datapoint():
    return DatapointBase(repo_id=1, repo_name="repo", completion_lines={}, context_dict=dict(FILES),
                         completion_dict={"app/target.txt": TARGET})


def submit(add=(), drop=()):
    return json_text({"action": "submit_context", "add": list(add), "drop": list(drop)})


def search(query="alpha_func"):
    return json_text({"action": "search_code", "query": query})


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

    def test_find_definition_by_chunk_symbol(self):
        self.tools.builder.symbol_index["alpha_func"] = [0]
        self.assertEqual(len(self.tools.find_definition("alpha_func")["results"]), 1)

    def test_find_definition_by_text_for_functions_classes_and_constants(self):
        files = {"app/target.txt": "x", "lib/m.py": "import os\n\nLIMIT = 3\n\n\nclass Foo:\n    pass\n\n\ndef bar(a):\n    return a\n"}
        tools = RepoTools(make_builder(files), Limits(), count)
        tools.builder.symbol_index.clear()  # force the text search
        for name, line in (("LIMIT", 3), ("Foo", 6), ("bar", 10)):
            found = tools.find_definition(name)["results"][0]
            self.assertEqual((found["path"], found["start_line"]), ("lib/m.py", line), name)

    def test_find_definition_reduces_dotted_names_and_calls(self):
        files = {"app/target.txt": "x", "lib/m.py": "def bar(a):\n    return a\n"}
        tools = RepoTools(make_builder(files), Limits(), count)
        tools.builder.symbol_index.clear()
        for name in ("self.bar", "mod.sub.bar()", " bar "):
            self.assertEqual(tools.find_definition(name)["results"][0]["start_line"], 1, name)

    def test_find_definition_unknown_name_and_never_the_completion_file(self):
        files = {"app/target.txt": "def secret_name():\n    pass\n", "lib/m.py": "x = 1\n"}
        tools = RepoTools(make_builder(files), Limits(), count)
        self.assertIn("error", tools.find_definition("secret_name"))
        self.assertIn("error", tools.find_definition(""))

    def test_only_shown_chunks_can_be_cited(self):
        shown = self.tools.search_code("alpha_func")["results"][0]["chunk"]
        unseen = next(i for i in range(len(self.tools.builder.chunks)) if i not in self.tools.seen)
        got = self.tools.resolve_items([{"chunk": unseen}, {"chunk": shown}])
        self.assertEqual(got, [self.tools.builder.chunks[shown]])

    def test_ranges_are_validated_deduplicated_and_capped(self):
        items = [{"path": "app/a.txt", "start_line": 1, "end_line": 1}, {"path": "app/a.txt", "start_line": 1, "end_line": 1},
                 {"path": "app/target.txt", "start_line": 1, "end_line": 1}, {"path": "app/a.txt", "start_line": 5, "end_line": 6},
                 "junk", {"chunk": "x"}]
        got = self.tools.resolve_items(items)
        self.assertEqual([(c.path, c.start_line, c.end_line) for c in got], [("app/a.txt", 0, 0)])
        capped = RepoTools(make_builder(), Limits(max_items=1), count).resolve_items(
            [{"path": "app/b.txt", "start_line": 1, "end_line": 1}, {"path": "app/a.txt", "start_line": 1, "end_line": 1}])
        self.assertEqual(len(capped), 1)


class EditTests(unittest.TestCase):
    def setUp(self):
        self.builder = make_builder()
        self.tools = RepoTools(self.builder, Limits(), count)
        self.default = self.builder.context_for("alpha_func beta_func", budget=1000, max_chunks=30).chunks
        self.ids = [self.tools.chunk_id[id(c)] for c in self.default]
        self.tools.seed_view(self.default)

    def test_the_seed_view_lists_the_default_best_first_with_ids_and_ranks(self):
        view = self.tools.seed_view(self.default)
        self.assertEqual([v["chunk"] for v in view], self.ids)
        self.assertEqual([v["rank"] for v in view], list(range(1, len(view) + 1)))

    def test_an_empty_edit_returns_the_default_unchanged(self):
        final = self.tools.apply_edit(self.default, [], [])
        self.assertEqual([id(c) for c in final], [id(c) for c in self.default])

    def test_drop_removes_default_chunks_and_ignores_other_ids(self):
        final = self.tools.apply_edit(self.default, [], [self.ids[0], 999, "x", -1])
        self.assertEqual([id(c) for c in final], [id(c) for c in self.default[1:]])

    def test_added_code_goes_first_and_default_chunks_follow(self):
        added = {"path": "lib/c.txt", "start_line": 1, "end_line": 1}
        final = self.tools.apply_edit(self.default, [added], [])
        self.assertEqual((final[0].path, final[0].kind), ("lib/c.txt", "range"))
        self.assertEqual([id(c) for c in final[1:]], [id(c) for c in self.default])

    def test_adding_a_default_chunk_promotes_it_without_duplicating(self):
        last = self.ids[-1]
        final = self.tools.apply_edit(self.default, [{"chunk": last}], [])
        self.assertIs(final[0], self.default[-1])
        self.assertEqual(len(final), len(self.default))

    def test_a_default_chunk_covered_by_an_added_range_is_not_repeated(self):
        first = self.default[0]
        wide = {"path": first.path, "start_line": 1, "end_line": first.end_line + 1}
        final = self.tools.apply_edit(self.default, [wide], [])
        self.assertEqual(sum(1 for c in final if c.path == first.path and c.start_line >= first.start_line
                             and c.end_line <= first.end_line), 1)

    def test_malformed_edits_change_nothing(self):
        final = self.tools.apply_edit(self.default, "junk", {"a": 1})
        self.assertEqual([id(c) for c in final], [id(c) for c in self.default])


class LoopTests(unittest.TestCase):
    def run_plan(self, replies, limits=Limits()):
        tools = RepoTools(make_builder(), limits, count)
        llm = ScriptedLLM(replies)
        return plan("task", tools, llm, limits), llm

    def test_search_then_submit(self):
        result, llm = self.run_plan([search(), submit([{"chunk": 0}])])
        self.assertEqual((result.status, result.steps, result.tool_calls), ("submitted", 2, 1))
        self.assertEqual((result.add, result.drop), ([{"chunk": 0}], []))
        self.assertIn("tool_result", llm.calls[1][-1]["content"])

    def test_an_empty_submit_is_a_valid_keep(self):
        result, _ = self.run_plan([submit()])
        self.assertEqual((result.status, result.add, result.drop, result.steps), ("submitted", [], [], 1))

    def test_tool_budget_forces_a_submit_request(self):
        result, llm = self.run_plan([search(), search(), submit([{"chunk": 0}])], Limits(tool_calls=2))
        self.assertEqual((result.status, result.tool_calls, result.steps), ("submitted", 2, 3))
        self.assertIn("Reply now with a submit_context", llm.calls[2][-1]["content"])

    def test_planner_that_keeps_searching_after_the_budget_is_cut_off(self):
        result, _ = self.run_plan([search()], Limits(tool_calls=1))
        self.assertEqual((result.status, result.tool_calls), ("no_submit", 1))

    def test_unusable_replies_end_the_run(self):
        result, _ = self.run_plan(["no json"], Limits(invalid_actions=2))
        self.assertEqual((result.status, result.steps), ("invalid_replies", 3))

    def test_invalid_tool_arguments_cost_a_call_and_return_an_error(self):
        result, _ = self.run_plan([json_text({"action": "read_file", "path": "app/target.txt", "start_line": 1, "end_line": 1}),
                                   submit()])
        self.assertIn("error", result.trace[0]["tool_result"])
        self.assertEqual(result.tool_calls, 1)


class AssembleTests(unittest.TestCase):
    def setUp(self):
        self.builder = make_builder()

    def chunks_for(self, *paths):
        return [c for p in paths for c in self.builder.chunks if c.path == p]

    def test_first_chunk_sits_nearest_the_prefix_and_budget_holds(self):
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
        _, used = assemble_context(self.builder, self.chunks_for("app/a.txt", "app/b.txt"), "p", 1000, count,
                                   max_context_tokens=5)
        self.assertLessEqual(len(used), 1)

    def test_reassembling_the_default_reproduces_the_rag_context(self):
        for options in ({"max_chunks": 30}, {"max_chunks": 30, "max_context_tokens": 6}):
            result = self.builder.context_for("alpha_func beta_func", budget=1000, **options)
            context, used = assemble_context(self.builder, result.chunks, result.prefix, 1000, count,
                                             options.get("max_context_tokens"))
            self.assertEqual(context, result.context)
            self.assertEqual(used, result.chunks)


class ProviderTests(unittest.TestCase):
    def provider(self, replies, rag_options=None):
        llm = ScriptedLLM(replies)
        factory = make_agent_provider_factory(count, 1000, llm, rag_options=rag_options or RAG_OPTIONS,
                                              lang_sep=LANG, meta_sep=META)
        return factory(make_datapoint()), llm

    def rag_prompt(self, line_num, rag_options=None):
        factory = make_rag_provider_factory(count, 1000, lang_sep=LANG, meta_sep=META, **(rag_options or RAG_OPTIONS))
        return factory(make_datapoint())(make_datapoint(), line_num)

    def test_changing_nothing_gives_the_rag_prompt_byte_for_byte(self):
        provider, _ = self.provider([submit()])
        for line_num in (1, 2):
            prompt, ground_truth, info = provider(make_datapoint(), line_num)
            rag_prompt, rag_truth, rag_info = self.rag_prompt(line_num)
            self.assertEqual((prompt, ground_truth), (rag_prompt, rag_truth))
            self.assertEqual((info["changed"], info["fallback"], info["paths"]), (False, False, rag_info["paths"]))

    def test_the_equivalence_also_holds_with_a_context_cap_and_continuation_lines(self):
        options = {**RAG_OPTIONS, "max_context_tokens": 6, "continuation_lines": 20}
        provider, _ = self.provider([submit()], options)
        for line_num in (1, 2):
            self.assertEqual(provider(make_datapoint(), line_num)[0], self.rag_prompt(line_num, options)[0])

    def test_a_planner_that_fails_gives_the_rag_prompt_and_is_flagged(self):
        provider, _ = self.provider(["no json"])
        prompt, _, info = provider(make_datapoint(), 2)
        self.assertEqual(prompt, self.rag_prompt(2)[0])
        self.assertEqual((info["fallback"], info["changed"], info["planner_status"]), (True, False, "invalid_replies"))

    def test_the_target_line_never_reaches_the_planner_or_the_prompt(self):
        provider, llm = self.provider([search(), submit()])
        prompt, ground_truth, info = provider(make_datapoint(), 1)  # target is line index 1, which holds the secret
        self.assertEqual(ground_truth, f"{SECRET} alpha_func gamma_func")
        self.assertNotIn(SECRET, prompt)
        self.assertTrue(all(SECRET not in m["content"] for call in llm.calls for m in call))
        self.assertEqual((info["steps"], info["tool_calls"], info["planner_status"]), (2, 1, "submitted"))

    def test_the_planner_sees_the_default_context_with_ids(self):
        provider, llm = self.provider([submit()])
        provider(make_datapoint(), 1)
        first_user = llm.calls[0][1]["content"]
        self.assertIn("CURRENT CONTEXT", first_user)
        self.assertIn('"chunk":', first_user)

    def test_an_edit_puts_added_code_nearest_the_prefix(self):
        provider, _ = self.provider([submit([{"path": "lib/c.txt", "start_line": 1, "end_line": 1}])])
        prompt, _, info = provider(make_datapoint(), 2)
        self.assertTrue(info["changed"] and not info["fallback"])
        self.assertEqual(info["paths"][0], "lib/c.txt")
        self.assertLess(prompt.index("alpha_func beta_func gamma_func"), prompt.index("unrelated words only"))
        self.assertTrue(prompt.endswith("\n" + "\n".join(TARGET.split("\n")[:2])))

    def test_an_edit_still_respects_the_budget_and_the_cap(self):
        options = {**RAG_OPTIONS, "max_context_tokens": 6}
        provider, _ = self.provider([submit([{"path": "lib/d.txt", "start_line": 1, "end_line": 1}])], options)
        prompt, _, info = provider(make_datapoint(), 2)
        self.assertLessEqual(count(prompt), 1000)
        self.assertEqual(prompt, self.rag_prompt(2, options)[0])  # the 50-word file cannot fit in 6 tokens
        self.assertEqual((info["changed"], info["fallback"]), (False, False))  # nothing changed, nothing failed

    def test_dropping_a_default_chunk_removes_only_that_chunk(self):
        builder = make_builder()
        default = builder.context_for("\n".join(TARGET.split("\n")[:2]), budget=1000, **RAG_OPTIONS).chunks
        self.assertGreater(len(default), 1)
        provider, _ = self.provider([submit(drop=[builder.chunks.index(default[0])])])
        prompt, _, info = provider(make_datapoint(), 2)
        self.assertEqual(info["paths"], [c.path for c in default[1:]])
        self.assertTrue(info["changed"] and not info["fallback"])

    def test_dropping_every_chunk_gives_an_empty_context_not_a_failure(self):
        provider, _ = self.provider(lambda messages: submit(drop=list(range(50))))
        prompt, _, info = provider(make_datapoint(), 2)
        self.assertEqual(info["paths"], [])
        self.assertTrue(info["changed"] and not info["fallback"])
        self.assertNotIn("alpha_func only here", prompt)
        self.assertTrue(prompt.endswith("\n" + "\n".join(TARGET.split("\n")[:2])))

    def test_only_the_bm25_retriever_is_supported(self):
        with self.assertRaises(ValueError):
            make_agent_provider_factory(count, 1000, ScriptedLLM([submit()]), rag_options={"retriever": "dense"})

    def test_task_message_lists_files_prefix_and_current_context(self):
        text = task_message("app/target.txt", "some prefix", ["app/a.txt", "lib/c.txt"], [{"chunk": 1}])
        for part in ("app/a.txt", "some prefix", "CURRENT CONTEXT", '"chunk":1'):
            self.assertIn(part, text)


class AppendModeTests(unittest.TestCase):
    """Planner v2 (log section 8.85): the planner can only add; RAG's ranking is never reduced except to make room."""
    PREFIX = "\n".join(TARGET.split("\n")[:2])
    C = {"path": "lib/c.txt", "start_line": 1, "end_line": 1}

    def provider(self, replies, rag_options=None, **limits):
        llm = ScriptedLLM(replies)
        factory = make_agent_provider_factory(count, 1000, llm, rag_options=rag_options or RAG_OPTIONS,
                                              limits=Limits(edit_mode="append", **limits), lang_sep=LANG, meta_sep=META)
        return factory(make_datapoint())

    def rag(self, rag_options=None):
        factory = make_rag_provider_factory(count, 1000, lang_sep=LANG, meta_sep=META, **(rag_options or RAG_OPTIONS))
        return factory(make_datapoint())(make_datapoint(), 2)

    def default(self):
        return make_builder().context_for(self.PREFIX, budget=1000, **RAG_OPTIONS).chunks

    def full_cap_options(self):
        builder, default = make_builder(), self.default()
        cap = sum(count(builder._format(c, 0) + LANG) for c in default)
        return {**RAG_OPTIONS, "max_context_tokens": cap}

    def test_drop_is_ignored_and_nothing_added_gives_the_rag_prompt(self):
        provider = self.provider([submit(drop=list(range(50)))])
        prompt, _, info = provider(make_datapoint(), 2)
        self.assertEqual(prompt, self.rag()[0])
        self.assertEqual((info["changed"], info["fallback"]), (False, False))

    def test_adding_code_the_default_already_contains_changes_nothing(self):
        builder = make_builder()
        first = self.default()[0]
        provider = self.provider([submit([{"chunk": builder.chunks.index(first)}])])
        prompt, _, info = provider(make_datapoint(), 2)
        self.assertEqual(prompt, self.rag()[0])
        self.assertFalse(info["changed"])

    def test_added_code_goes_last_in_rank_and_farthest_from_the_prefix(self):
        prompt, _, info = self.provider([submit([self.C])])(make_datapoint(), 2)
        self.assertTrue(info["changed"] and not info["fallback"])
        self.assertEqual(info["paths"], self.rag()[2]["paths"] + ["lib/c.txt"])  # every default chunk keeps its rank
        self.assertLess(prompt.index("unrelated words only"), prompt.index("alpha_func beta_func gamma_func"))
        self.assertTrue(prompt.endswith("\n" + self.PREFIX))

    def test_a_full_context_evicts_only_the_lowest_ranked_chunks(self):
        options = self.full_cap_options()
        default_paths = self.rag(options)[2]["paths"]
        prompt, _, info = self.provider([submit([self.C])], options)(make_datapoint(), 2)
        self.assertTrue(info["changed"])
        kept = info["paths"][:-1]
        self.assertEqual(info["paths"][-1], "lib/c.txt")
        self.assertEqual(kept, default_paths[:len(kept)])  # a prefix of the ranking: the top chunks survive
        self.assertIn(len(default_paths) - len(kept), (1, 2))
        self.assertLessEqual(count(prompt), 1000)

    def test_with_no_eviction_allowed_a_full_context_stays_the_rag_prompt(self):
        options = self.full_cap_options()
        prompt, _, info = self.provider([submit([self.C])], options, max_evict=0)(make_datapoint(), 2)
        self.assertEqual(prompt, self.rag(options)[0])
        self.assertEqual((info["changed"], info["fallback"]), (False, False))

    def test_at_most_max_add_additions(self):
        two = [self.C, {"path": "lib/d.txt", "start_line": 1, "end_line": 1}]
        for max_add in (1, 2):
            _, _, info = self.provider([submit(two)], max_add=max_add)(make_datapoint(), 2)
            self.assertEqual(len(info["paths"]), len(self.rag()[2]["paths"]) + max_add)

    def test_the_planner_prompt_has_no_drop_action(self):
        llm = ScriptedLLM([submit()])
        factory = make_agent_provider_factory(count, 1000, llm, rag_options=RAG_OPTIONS,
                                              limits=Limits(edit_mode="append"), lang_sep=LANG, meta_sep=META)
        factory(make_datapoint())(make_datapoint(), 2)
        system = llm.calls[0][0]["content"]
        self.assertNotIn('"drop"', system)
        self.assertIn("cannot remove", system)

    def test_the_kind_of_every_final_chunk_is_recorded_so_the_prompt_can_be_rebuilt(self):
        _, _, info = self.provider([submit([self.C])])(make_datapoint(), 2)
        self.assertEqual(len(info["kinds"]), len(info["paths"]))
        self.assertEqual(info["kinds"][-1], "range")  # the added code
        self.assertNotIn("range", info["kinds"][:-1])  # RAG's own chunks come from the index

    def test_the_planner_actions_are_recorded(self):
        _, _, info = self.provider([search(), submit()])(make_datapoint(), 2)
        self.assertEqual([a["action"] for a in info["actions"]], ["search_code", "submit_context"])


class ReplayTests(unittest.TestCase):
    """The logged `actions` of a run are enough to rebuild its prompt offline (used by eval/failure_split.py)."""
    C = {"path": "lib/c.txt", "start_line": 1, "end_line": 1}

    def run_and_replay(self, replies, **limits):
        def factory(llm):
            return make_agent_provider_factory(count, 1000, llm, rag_options=RAG_OPTIONS,
                                               limits=Limits(edit_mode="append", **limits), lang_sep=LANG, meta_sep=META)(
                make_datapoint())

        original = factory(ScriptedLLM(replies))(make_datapoint(), 2)
        replay = ReplayModel()
        replay.load(original[2]["actions"])
        replayed = factory(replay)(make_datapoint(), 2)
        self.assertTrue(replay.finished)
        self.assertEqual(replayed[0], original[0])  # the same prompt
        for key in ("paths", "lines", "kinds", "changed", "fallback", "planner_status", "steps", "tool_calls"):
            self.assertEqual(replayed[2][key], original[2][key], key)
        return original[2]

    def test_a_search_and_an_addition_are_replayed_exactly(self):
        info = self.run_and_replay([search(), submit([self.C])])
        self.assertTrue(info["changed"])

    def test_submitting_nothing_is_replayed(self):
        self.assertFalse(self.run_and_replay([submit()])["changed"])

    def test_invalid_replies_are_replayed(self):
        info = self.run_and_replay(["no json here", submit([self.C])])
        self.assertEqual(info["planner_status"], "submitted")
        self.assertIsNone(info["actions"][0])

    def test_giving_up_after_too_many_invalid_replies_is_replayed(self):
        info = self.run_and_replay(["nope"] * 5)
        self.assertEqual((info["planner_status"], info["fallback"]), ("invalid_replies", True))

    def test_a_planner_that_ignores_the_forced_submit_is_replayed(self):
        info = self.run_and_replay([search(), search("beta_func"), search("gamma_func")], tool_calls=1)
        self.assertEqual((info["planner_status"], info["fallback"]), ("no_submit", True))

    def test_replaying_more_replies_than_were_logged_is_an_error(self):
        replay = ReplayModel()
        replay.load([{"action": "submit_context", "add": []}])
        replay([])
        with self.assertRaises(RuntimeError):
            replay([])


@unittest.skipUnless(HAS_PARSER, "tree-sitter grammar not available in the working directory")
class RealPythonTests(unittest.TestCase):
    REPO = {
        "pkg/target.py": "from pkg.util import helper\n\n\ndef run(x):\n    y = helper(x)\n    return y\n",
        "pkg/util.py": "LIMIT = 3\n\n\ndef helper(a):\n    return a + LIMIT\n\n\nclass Store:\n    def get(self, key):\n        return key\n",
        "other/misc.py": "def unrelated():\n    return 1\n",
    }

    def test_symbols_are_found_through_the_parser_and_search_and_find_definition_work(self):
        builder = RagContextBuilder(self.REPO, "pkg/target.py", "repo", count, lang_sep=LANG, meta_sep=META)
        tools = RepoTools(builder, Limits(), count)
        self.assertIn("helper", builder.symbol_index)
        found = tools.find_definition("helper")["results"][0]
        self.assertEqual(found["path"], "pkg/util.py")
        self.assertIn("return a + LIMIT", found["text"])
        self.assertEqual(tools.find_definition("LIMIT")["results"][0]["start_line"], 1)
        self.assertEqual(tools.find_definition("self.store.get")["results"][0]["path"], "pkg/util.py")
        hits = tools.search_code("helper LIMIT")["results"]
        self.assertTrue(hits and all(r["path"] != "pkg/target.py" for r in hits))

    def test_adding_a_found_definition_reaches_the_prompt_and_the_prompt_stays_in_budget(self):
        datapoint = DatapointBase(repo_id=1, repo_name="repo", completion_lines={}, context_dict=dict(self.REPO),
                                  completion_dict={"pkg/target.py": self.REPO["pkg/target.py"]})

        def script(messages):
            if len(messages) >= 4:  # system, task, first reply, tool result: add the first result by its chunk id
                chunk_id = int(messages[-1]["content"].split('"chunk":')[1].split(",")[0])
                return json_text({"action": "submit_context", "add": [{"chunk": chunk_id}]})
            return json_text({"action": "find_definition", "name": "Store"})

        factory = make_agent_provider_factory(count, 400, ScriptedLLM(script), rag_options=RAG_OPTIONS,
                                              lang_sep=LANG, meta_sep=META)
        prompt, _, info = factory(datapoint)(datapoint, 5)
        self.assertTrue(info["changed"])
        self.assertIn("class Store", prompt)
        self.assertLessEqual(count(prompt), 400)


if __name__ == "__main__":
    unittest.main()
