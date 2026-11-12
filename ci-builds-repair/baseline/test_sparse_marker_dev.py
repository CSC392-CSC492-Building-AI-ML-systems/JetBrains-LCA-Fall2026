import unittest

from run_sparse_marker_dev import ARM_NAMES, parse_markers, summarize


class SparseMarkerTests(unittest.TestCase):
    blocks = [
        {"path": "a.py", "start": 5, "old": "    original\n"},
        {"path": "b.py", "start": 2, "old": "other"},
    ]
    tag = "a.py#L5#B0"

    def wrap(self, text, tag=None):
        tag = tag or self.tag
        return f"[start of {tag}]\n{text}[end of {tag}]"

    def test_one_changed_block_and_indentation(self):
        self.assertEqual(parse_markers(self.wrap("    new\n\n"), self.blocks),
                         ["    new\n\n", "other"])

    def test_deletion_noop_and_outer_fence(self):
        self.assertEqual(parse_markers(self.wrap(""), self.blocks), ["", "other"])
        self.assertEqual(parse_markers("[no edits]", self.blocks), ["    original\n", "other"])
        fenced = "```text\n" + self.wrap("    replacement\n") + "\n```"
        self.assertEqual(parse_markers(fenced, self.blocks), ["    replacement\n", "other"])

    def test_preserves_unterminated_original_line(self):
        self.assertEqual(parse_markers(self.wrap("replacement\n", "b.py#L2#B1"), self.blocks),
                         ["    original\n", "replacement"])

    def test_invalid_markers_reject_the_entire_answer(self):
        cases = [
            self.wrap("one\n") + "\n" + self.wrap("two\n"),
            self.wrap("new\n", "unknown.py#L5#B9"),
            "[start of a.py#L5#B0]\nunclosed",
            self.wrap("new\n") + "\nexplanation",
            self.wrap("nul\0\n"),
        ]
        for response in cases:
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_markers(response, self.blocks)

    def test_gate_requires_complete_results_and_quality(self):
        row = {"status": "applied", "nonempty_patch": True, "syntax_preserved": False,
               "target_blocks": 3, "partial_edit": True, "edit_token_f1": 0.3,
               "patch_exact_match": False, "error": None}
        legacy = {**row, "arm": ARM_NAMES[0]}
        sparse = {**row, "arm": ARM_NAMES[1], "syntax_preserved": True}
        self.assertIsNone(summarize([legacy], 1)["development_gate_passed"])
        self.assertTrue(summarize([legacy, sparse], 1)["development_gate_passed"])
        self.assertFalse(summarize([legacy, {**sparse, "edit_token_f1": 0.2}], 1)["development_gate_passed"])


if __name__ == "__main__":
    unittest.main()
