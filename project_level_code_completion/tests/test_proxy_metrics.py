import unittest

from eval.proxy_metrics import (definitions, hit_miss_difference, identifiers, needed_names, recalls,
                                repo_definitions, summarize, targeted_lines, visible_prompt)

LIB = '''import os

LIMIT = 3
MODE: str = "x"

def helper(a):
    return a

async def fetch_data(url):
    pass

class Client(Base):
    timeout = 5
    def connect(self):
        pass
'''


class IdentifierTests(unittest.TestCase):
    def test_keywords_dunders_and_single_chars_are_dropped(self):
        self.assertEqual(identifiers("def __init__(self, x): return None if x else foo_bar"), {"self", "foo_bar"})

    def test_names_are_case_sensitive_and_whole(self):
        self.assertEqual(identifiers("Client.connect(clientId)"), {"Client", "connect", "clientId"})


class DefinitionTests(unittest.TestCase):
    def test_defs_classes_methods_and_top_level_assignments(self):
        self.assertEqual(definitions(LIB), {"helper", "fetch_data", "Client", "connect", "LIMIT", "MODE"})

    def test_class_attribute_is_not_a_top_level_assignment(self):
        self.assertNotIn("timeout", definitions(LIB))

    def test_comparison_is_not_an_assignment(self):
        self.assertEqual(definitions("x == 3\nresult = 1\n"), {"result"})

    def test_repo_definitions_skip_completion_file_and_non_python(self):
        files = {"a.py": "def alpha(): pass", "target.py": "def secret(): pass", "doc.md": "def ignored(): pass"}
        self.assertEqual(repo_definitions(files, exclude_path="target.py"), {"alpha"})


class NeededNamesTests(unittest.TestCase):
    def test_strict_excludes_every_name_in_the_prefix(self):
        defined = {"helper", "Client", "LIMIT"}
        needed = needed_names("x = helper(Client(), LIMIT)", "from lib import LIMIT\n", defined, strict=True)
        self.assertEqual(needed, {"helper", "Client"})  # LIMIT already appears in the prefix

    def test_loose_keeps_imported_names_but_drops_names_the_file_defines(self):
        defined = {"helper", "Client", "LIMIT", "local_fn"}
        prefix = "from lib import LIMIT\n\ndef local_fn():\n    pass\n"
        needed = needed_names("x = helper(Client(), LIMIT, local_fn())", prefix, defined)
        self.assertEqual(needed, {"helper", "Client", "LIMIT"})  # an import does not define; local_fn is defined here

    def test_loose_is_the_default(self):
        self.assertEqual(needed_names("helper()", "import helper\n", {"helper"}), {"helper"})

    def test_names_not_defined_in_repo_are_not_needed(self):
        self.assertEqual(needed_names("print(len(items))", "", {"helper"}), set())


class TargetedLinesTests(unittest.TestCase):
    def datapoint(self):
        from data_classes.datapoint_commit_dataset import DatapointCommitDataset

        return DatapointCommitDataset(
            repo_id=0, repo_name="r", completion_lines={"inproject": [1, 3], "infile": [2]},
            context_dict={"lib.py": "def helper(a):\n    return a\n", "notes.md": "def ignored(): pass"},
            completion_dict={"main.py": "import lib\nresult = helper(1)\nprint(result)\nvalue = lib.helper(2)\n"},
        )

    def test_only_lines_using_cross_file_names_are_kept(self):
        self.assertEqual(targeted_lines(self.datapoint()), {"inproject": [1, 3], "infile": []})

    def test_a_name_defined_in_the_file_prefix_is_not_needed(self):
        dp = self.datapoint()
        dp.completion_dict = {"main.py": "def helper(a):\n    return a\nvalue = helper(2)\n"}
        dp.completion_lines = {"infile": [2]}
        self.assertEqual(targeted_lines(dp), {"infile": []})

    def test_completion_file_definitions_do_not_count_as_elsewhere(self):
        dp = self.datapoint()
        dp.context_dict = {"main.py": "def helper(a):\n    return a\n"}  # the snapshot's old copy of the file itself
        self.assertEqual(targeted_lines(dp), {"inproject": [], "infile": []})


class RecallTests(unittest.TestCase):
    def test_appearance_and_definition_recall(self):
        prompt = "lib.pyMETASEP\ndef helper(a):\n    return a\nuse = Client\n"
        app, dfn = recalls({"helper", "Client"}, prompt)
        self.assertEqual((app, dfn), (1.0, 0.5))  # Client appears but is not defined in the prompt

    def test_nothing_visible(self):
        self.assertEqual(recalls({"helper"}, "unrelated text"), (0.0, 0.0))


class VisiblePromptTests(unittest.TestCase):
    encode = staticmethod(lambda s: [ord(c) for c in s])
    decode = staticmethod(lambda ids: "".join(chr(i) for i in ids))

    def test_short_prompt_is_unchanged(self):
        self.assertEqual(visible_prompt("abc", self.encode, self.decode, 10), "abc")

    def test_keeps_the_last_tokens(self):
        self.assertEqual(visible_prompt("abcdefgh", self.encode, self.decode, 3), "fgh")

    def test_character_crop_applies_first(self):
        text = "x" * 1000 + "END"
        self.assertTrue(visible_prompt(text, self.encode, self.decode, 4).endswith("END"))


class HitMissTests(unittest.TestCase):
    def rows(self, hit_em, miss_em, arm="a", n_dp=6):
        out = []
        for dp in range(n_dp):
            out += [{"arm": arm, "dp": dp, "hit": True, "em": hit_em}, {"arm": arm, "dp": dp, "hit": False, "em": miss_em}]
        return out

    def test_consistent_gap_has_interval_above_zero(self):
        diff, low, high, n_hit, n_miss = hit_miss_difference(self.rows(0.6, 0.2), n_boot=300)
        self.assertAlmostEqual(diff, 0.4)
        self.assertGreater(low, 0)
        self.assertEqual((n_hit, n_miss), (6, 6))

    def test_no_gap_gives_zero(self):
        diff, low, high, _, _ = hit_miss_difference(self.rows(0.3, 0.3), n_boot=300)
        self.assertEqual((diff, low, high), (0.0, 0.0, 0.0))

    def test_averages_over_arms(self):
        rows = self.rows(0.6, 0.2, "a") + self.rows(0.4, 0.2, "b")
        self.assertAlmostEqual(hit_miss_difference(rows, n_boot=100)[0], 0.3)

    def test_returns_none_when_a_group_is_empty(self):
        self.assertIsNone(hit_miss_difference([{"arm": "a", "dp": 0, "hit": True, "em": 1.0}], n_boot=10))

    def test_deterministic(self):
        rows = self.rows(0.5, 0.1) + self.rows(0.2, 0.3, "b")
        self.assertEqual(hit_miss_difference(rows, n_boot=200), hit_miss_difference(rows, n_boot=200))


class SummaryTests(unittest.TestCase):
    def test_summary_table_has_one_row_per_arm(self):
        rows = [{"arm": "a", "app": 1.0, "def": 0.5, "hit": True}, {"arm": "a", "app": 0.0, "def": 0.0, "hit": False}]
        text = summarize(rows)
        self.assertIn("| a | 2 | 0.500 | 0.250 | 0.500 |", text)


if __name__ == "__main__":
    unittest.main()
