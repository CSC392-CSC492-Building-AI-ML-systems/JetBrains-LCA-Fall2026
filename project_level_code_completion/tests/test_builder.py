import unittest

from rag.builder import RagContextBuilder, crop_prefix, make_query

LANG, META = "LANGSEP\n", "METASEP\n"


def count(text: str) -> int:
    return len(text.split())


def make_builder(files, completion_path="x/target.txt"):
    return RagContextBuilder(files, completion_path, "repo", count, extension="", lang_sep=LANG, meta_sep=META)


FILES = {
    "x/target.txt": "SECRETANSWER alpha_func beta_func",
    "x/a.txt": "alpha_func beta_func gamma_func",
    "x/b.txt": "alpha_func",
    "y/c.txt": "unrelated words only",
}
PREFIX = "alpha_func beta_func gamma_func"


class QueryAndPrefixTests(unittest.TestCase):
    def test_make_query_takes_last_non_empty_lines(self):
        self.assertEqual(make_query("a\n\nb\n  \nc\nd", n_lines=2), "c\nd")

    def test_crop_prefix_drops_leading_lines(self):
        prefix = "one two\nthree four\nfive six"
        self.assertEqual(crop_prefix(prefix, 4, count), "three four\nfive six")
        self.assertEqual(crop_prefix(prefix, 100, count), prefix)

    def test_crop_prefix_keeps_last_line_even_if_over_cap(self):
        self.assertEqual(crop_prefix("a b c d e f", 2, count), "a b c d e f")


class LeakageTests(unittest.TestCase):
    def test_completion_file_never_indexed_or_emitted(self):
        builder = make_builder(FILES)
        self.assertTrue(all(c.path != "x/target.txt" for c in builder.chunks))
        result = builder.context_for(PREFIX, budget=1000)
        self.assertNotIn("SECRETANSWER", result.context)
        self.assertTrue(all(c.path != "x/target.txt" for c in result.chunks))


class SelectionTests(unittest.TestCase):
    def test_relevant_selected_irrelevant_excluded(self):
        result = make_builder(FILES).context_for(PREFIX, budget=1000)
        self.assertEqual([c.path for c in result.chunks], ["x/a.txt", "x/b.txt"])  # best first
        self.assertNotIn("unrelated", result.context)
        self.assertFalse(result.fallback)

    def test_ascending_relevance_best_chunk_nearest_prefix(self):
        context = make_builder(FILES).context_for(PREFIX, budget=1000).context
        self.assertLess(context.index("x/b.txt"), context.index("x/a.txt"))
        self.assertTrue(context.endswith("x/target.txtMETASEP\n"))

    def test_total_tokens_within_budget(self):
        builder = make_builder(FILES)
        for budget in (20, 25, 30, 60, 1000):
            result = builder.context_for(PREFIX, budget=budget)
            total = count(result.context + "\n" + result.prefix)
            self.assertLessEqual(total, budget, f"budget={budget}")

    def test_budget_holds_for_non_additive_tokenizer(self):
        # Superadditive counter: each pair of separators costs one extra token, so summing per-chunk costs
        # underestimates the total. The exact final check must still keep the input within the budget.
        def count_extra(text: str) -> int:
            return len(text.split()) + text.count("LANGSEP") // 2

        files = {"x/target.txt": "t"} | {f"x/f{i}.txt": f"alpha_func v{i}" for i in range(12)}
        builder = RagContextBuilder(files, "x/target.txt", "repo", count_extra, lang_sep=LANG, meta_sep=META)
        for budget in range(10, 80, 7):
            result = builder.context_for("alpha_func", budget=budget)
            self.assertLessEqual(count_extra(result.context + "\n" + result.prefix), budget, f"budget={budget}")

    def test_max_context_tokens_caps_retrieved_chunks_only(self):
        files = {"x/target.txt": "t"} | {f"x/f{i}.txt": f"alpha_func v{i} " + " ".join(["w"] * 8) for i in range(10)}
        builder = make_builder(files)
        uncapped = builder.context_for("alpha_func", budget=10_000)
        capped = builder.context_for("alpha_func", budget=10_000, max_context_tokens=25)
        self.assertEqual(len(uncapped.chunks), 10)
        self.assertEqual(len(capped.chunks), 2)  # each chunk costs 12 tokens (path + 10 words + separator)
        self.assertLess(count(capped.context), count(uncapped.context))
        self.assertEqual(capped.prefix, "alpha_func")  # the prefix is untouched by the cap

    def test_no_cap_by_default(self):
        builder = make_builder(FILES)
        self.assertEqual(builder.context_for(PREFIX, budget=1000).context,
                         builder.context_for(PREFIX, budget=1000, max_context_tokens=None).context)

    def test_too_big_chunk_skipped_smaller_one_used(self):
        files = {
            "x/target.txt": "t",
            "x/big.txt": "alpha_func " + " ".join(["filler"] * 50),
            "x/small.txt": "alpha_func",
        }
        builder = make_builder(files)
        result = builder.context_for("alpha_func", budget=30)
        self.assertEqual([c.path for c in result.chunks], ["x/small.txt"])

    def test_max_chunks_respected(self):
        files = {"x/target.txt": "t"} | {f"x/f{i}.txt": f"alpha_func v{i}" for i in range(10)}
        result = make_builder(files).context_for("alpha_func", budget=10_000, max_chunks=3)
        self.assertEqual(len(result.chunks), 3)

    def test_zero_budget_gives_only_the_tail(self):
        result = make_builder(FILES).context_for(PREFIX, budget=1)
        self.assertEqual(result.chunks, [])
        self.assertTrue(result.context.endswith("x/target.txtMETASEP\n"))

    def test_prefix_cap_applied_before_retrieval(self):
        prefix = "old_line_one old_line_one\n" + PREFIX
        result = make_builder(FILES).context_for(prefix, budget=1000, prefix_cap=3)
        self.assertEqual(result.prefix, PREFIX)


class FallbackTests(unittest.TestCase):
    def test_empty_prefix_orders_by_path_distance(self):
        files = {"x/target.txt": "t", "y/far.txt": "far", "x/near.txt": "near"}
        result = make_builder(files).context_for("", budget=1000)
        self.assertTrue(result.fallback)
        self.assertEqual([c.path for c in result.chunks], ["x/near.txt", "y/far.txt"])


if __name__ == "__main__":
    unittest.main()
