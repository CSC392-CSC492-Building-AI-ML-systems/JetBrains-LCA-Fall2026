import unittest

from src.context.long_context import pack_context
from src.evaluation.long_context import paired_interval


class LongContextTests(unittest.TestCase):
    def pack(self, apis, budget=100, **kwargs):
        return pack_context("write code", apis, "direct", budget, len,
                            context_window=1000, max_output_tokens=100,
                            safety_tokens=50, **kwargs)

    def test_direct_preserves_order_and_deduplicates(self):
        result = self.pack(["zebra", "apple", "zebra"])
        self.assertEqual(result["selected_apis"], ["zebra", "apple"])
        self.assertEqual(result["candidate_count"], 2)
        self.assertLessEqual(result["actual_context_tokens"], 100)

    def test_skips_oversized_names_without_cutting(self):
        result = self.pack(["x" * 500, "short"])
        self.assertEqual(result["selected_apis"], ["short"])
        self.assertTrue(result["truncated"])
        self.assertEqual(result["decisions"][0]["reason"], "evidence_budget")

    def test_instruction_only_and_empty_library_share_prompt(self):
        plain = pack_context("do work", ["hidden"], "plain", 0, len)
        direct = pack_context("do work", [], "direct", 100, len)
        self.assertEqual(plain["prompt"], direct["prompt"])
        self.assertEqual(plain["actual_context_tokens"], 0)

    def test_reserves_output_and_safety(self):
        result = pack_context("work", ["x" * 200, "small"], "direct", 10000, len,
                              context_window=350, max_output_tokens=50, safety_tokens=30)
        self.assertLessEqual(result["estimated_input_tokens"], 270)
        self.assertEqual(result["selected_apis"], ["small"])

    def test_oversized_instruction_rejected(self):
        with self.assertRaises(ValueError):
            pack_context("x" * 1000, [], "direct", 100, len,
                         context_window=300, max_output_tokens=50, safety_tokens=30)

    def test_budget_growth_retains_more_context(self):
        apis = [f"library_method_{i}" for i in range(20)]
        small = self.pack(apis, 100)
        large = self.pack(apis, 800)
        self.assertGreater(large["included_count"], small["included_count"])

    def test_constant_paired_difference(self):
        self.assertEqual(paired_interval([.25] * 5, draws=100), [.25, .25])


if __name__ == "__main__":
    unittest.main()
