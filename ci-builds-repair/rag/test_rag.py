import unittest

from evaluation import apply_edits, diff_blocks, prediction_blocks, score
from retrieval import chunk_files, make_query, retrieve
from run_benchmark import packed_prompt


class CounterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(messages[0]["content"])


class RagTests(unittest.TestCase):
    def test_labels_absent_from_prediction_and_gold_scores_exactly(self):
        files = {"a.py": "x = 1\n", "b.py": "def help():\n    return x\n"}
        blocks = diff_blocks("--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n")
        projected = prediction_blocks(blocks)
        self.assertNotIn("new", projected[0])
        self.assertNotIn("x = 2", repr(projected))
        gold = apply_edits(files, projected, ["x = 2\n"])
        metrics = score(files, blocks, gold, [], [])
        self.assertTrue(metrics["patch_exact_match"])
        self.assertEqual(metrics["edit_token_f1"], 1)
        unchanged = score(files, blocks, {}, [], [])
        self.assertFalse(unchanged["patch_exact_match"])
        self.assertEqual(unchanged["edit_token_f1"], 0)

    def test_insert_delete_and_newline_preservation(self):
        blocks = diff_blocks("--- a/a.py\n+++ b/a.py\n@@ -1,0 +2 @@\n+y = 2\n")
        self.assertEqual(apply_edits({"a.py": "x = 1\n"}, blocks, ["y = 2\n"])["a.py"], "x = 1\ny = 2\n")
        deleted = diff_blocks("--- a/a.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x = 1\n")
        self.assertEqual(apply_edits({"a.py": "x = 1\n"}, deleted, [""])["a.py"], "")
        no_newline = diff_blocks("--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n\\ No newline at end of file\n+x = 2\n\\ No newline at end of file\n")
        self.assertEqual(no_newline[0]["new"], "x = 2")

    def test_retrieval_determinism_tests_retained_and_query_has_no_labels(self):
        files = {"tests/test_ssl.py": "ssl error certificate\n", "src/a.py": "unrelated\n"}
        chunks = chunk_files(files, window=40, stride=20)
        query = make_query([{"step_name": "not_log", "log": "ssl error certificate"}], "tests")
        self.assertNotIn("not_log", query)
        ranked = retrieve(chunks, query)
        self.assertEqual(ranked[0]["path"], "tests/test_ssl.py")
        self.assertEqual(ranked, retrieve(chunks, query))

    def test_packing_skips_overlap_and_oversize_without_truncating_code(self):
        blocks = [{"path": "a.py", "start": 1, "count": 1, "old": "x = 1\n"}]
        row = {"logs": [{"log": "error"}]}
        evidence = [{"path": "a.py", "start": 1, "end": 40, "text": "duplicate"},
                    {"path": "b.py", "start": 1, "end": 1, "text": "too long" * 1000},
                    {"path": "c.py", "start": 1, "end": 1, "text": "small"}]
        _, tokens, selected, skip = packed_prompt(row, blocks, evidence, CounterTokenizer(), 800)
        self.assertIsNone(skip)
        self.assertLessEqual(len(tokens), 800)
        self.assertEqual([c["path"] for c in selected], ["c.py"])

    def test_invalid_output_is_not_a_syntax_success_claim(self):
        files = {"a.py": "x = 1\n"}
        blocks = diff_blocks("--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n")
        metrics = score(files, blocks, {"a.py": "x = (\n"}, [], [])
        self.assertEqual(metrics["syntax_regression_files"], ["a.py"])
        self.assertFalse(metrics["patch_exact_match"])


if __name__ == "__main__":
    unittest.main()
