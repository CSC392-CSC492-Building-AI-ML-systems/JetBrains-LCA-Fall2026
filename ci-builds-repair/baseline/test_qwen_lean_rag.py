import unittest

import run_qwen_lean_rag as lean


class LeanRagTests(unittest.TestCase):
    def test_query_uses_failed_log_cues(self):
        logs = [{"step_name": "secret_label", "log":
                 "\x1b[31m2024-01-09T01:02:03.123Z TypeError in pkg/core.py and test_render\x1b[0m\n"
                 "##[error]Process completed with exit code 1"}]
        query, cues = lean.make_lean_query(logs)
        self.assertEqual(cues, ["TypeError", "pkg/core.py", "test_render"])
        for term in ("typeerror", "core", "test_render"):
            self.assertIn(term, query.split())
        for term in ("secret_label", "error", "process"):
            self.assertNotIn(term, query.split())

    def test_pack_enforces_deduplication_overlap_and_budget(self):
        def chunk(path, start, end, text):
            return {"path": path, "start": start, "end": end, "text": text}

        ranked = [
            chunk("target.py", 1, 10, "overlaps"),
            chunk("long.py", 1, 20, "x" * 1000),
            chunk("first.py", 1, 20, "a b"),
            chunk("first.py", 21, 40, "repeat path"),
            chunk("other.py", 1, 20, "a  b"),
            chunk("second.py", 1, 20, "c d"),
            chunk("third.py", 1, 20, "e f"),
            chunk("fourth.py", 1, 20, "g h"),
        ]
        blocks = [{"path": "target.py", "start": 10, "count": 2}]
        self.assertEqual([c["path"] for c in lean.pack("baseline", ranked, blocks, len)],
                         ["first.py", "second.py", "third.py"])
        self.assertEqual(lean.assemble("baseline", []), "baseline")
        self.assertTrue(lean.assemble("baseline", ranked[2:3]).endswith("baseline"))

    def test_unmatched_rows_do_not_count_as_pairs(self):
        row = {"id": 1, "seed": 0, "arm": "plain", "error": None, "nonempty_patch": True,
               "syntax_preserved": True, "patch_exact_match": False, "edit_token_f1": 0.2, "snippets": 0}
        self.assertFalse(lean.summarize([row], [1], [0])["complete"])
        comparison = lean.summarize([row, {**row, "arm": "lean_rag", "snippets": 1, "edit_token_f1": 0.3}], [1], [0])
        self.assertTrue(comparison["complete"])
        self.assertEqual(comparison["paired_f1_wins_ties_losses"], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
