import unittest

from eval.failure_split import classify, summarize


class ClassifyTests(unittest.TestCase):
    def test_nothing_to_retrieve_is_self_contained(self):
        self.assertEqual(classify(set(), {"a"}, True), "right_self_contained")
        self.assertEqual(classify(set(), {"a"}, False), "wrong_self_contained")

    def test_right_line_with_needed_names(self):
        self.assertEqual(classify({"helper"}, set(), True), "right")

    def test_wrong_line_with_a_missing_name_is_a_retrieval_failure(self):
        self.assertEqual(classify({"helper", "other"}, {"helper"}, False), "retrieval")

    def test_wrong_line_with_every_name_visible_is_a_reasoning_failure(self):
        self.assertEqual(classify({"helper", "other"}, {"helper", "other", "x"}, False), "reasoning")


class SummarizeTests(unittest.TestCase):
    def test_counts_per_arm(self):
        rows = [{"arm": "a", "cls": "right"}, {"arm": "a", "cls": "retrieval"},
                {"arm": "a", "cls": "reasoning"}, {"arm": "a", "cls": "wrong_self_contained"}]
        table = summarize(rows).splitlines()
        self.assertEqual(len(table), 3)
        self.assertIn("| a | 4 | 1 (25.0%) | 1 (25.0%) | 1 (25.0%) | 1 (25.0%) |", table[2])


if __name__ == "__main__":
    unittest.main()
