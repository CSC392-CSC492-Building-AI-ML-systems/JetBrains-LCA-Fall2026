import unittest

from rag.builder import RagContextBuilder, make_query

LANG, META = "LANGSEP\n", "METASEP\n"
FILES = {
    "app/main.txt": "target",
    "app/near.txt": "unrelated nearby text",
    "far/deep/er/x.txt": "alpha_func beta_func appears here",
    "lib/mid.txt": "other words entirely",
}
PREFIX = "first line\nsecond line\nalpha_func beta_func"


def count(text: str) -> int:
    return len(text.split())


def make(**kwargs):
    return RagContextBuilder(FILES, "app/main.txt", "repo", count, lang_sep=LANG, meta_sep=META, **kwargs)


class RandomControlTests(unittest.TestCase):
    def test_ignores_relevance_but_is_reproducible(self):
        b = make(retriever="random")
        first = [c.path for c in b.context_for(PREFIX, budget=1000).chunks]
        again = [c.path for c in b.context_for(PREFIX, budget=1000).chunks]
        self.assertEqual(first, again)
        self.assertEqual(sorted(first), sorted(["app/near.txt", "far/deep/er/x.txt", "lib/mid.txt"]))  # every chunk is eligible

    def test_order_changes_with_the_prefix(self):
        b = make(retriever="random")
        orders = {tuple(c.path for c in b.context_for(PREFIX + "\n" + "x" * i, budget=1000).chunks) for i in range(12)}
        self.assertGreater(len(orders), 1)

    def test_it_is_not_just_bm25(self):
        bm25 = [c.path for c in make().context_for(PREFIX, budget=1000).chunks]
        self.assertEqual(bm25, ["far/deep/er/x.txt"])  # BM25 only returns the matching chunk

    def test_respects_the_budget(self):
        b = make(retriever="random")
        for budget in (9, 12, 16, 1000):  # separator, repo name, file path and prefix alone need 9 tokens here
            r = b.context_for(PREFIX, budget=budget)
            self.assertLessEqual(count(r.context + "\n" + r.prefix), budget)

    def test_when_the_prefix_alone_exceeds_the_budget_no_chunk_is_added(self):
        # The known edge case (verification V-B): nothing can be dropped from header, file path and prefix, so the
        # total may exceed the budget; the generator's own left crop then trims it. No retrieved chunk is added.
        r = make(retriever="random").context_for(PREFIX, budget=6)
        self.assertEqual(r.chunks, [])
        self.assertEqual(count(r.context + "\n" + r.prefix), 9)  # over the budget of 6: nothing droppable remains


class ProximityControlTests(unittest.TestCase):
    def test_nearest_files_first_regardless_of_content(self):
        r = make(retriever="proximity").context_for(PREFIX, budget=1000)
        self.assertEqual([c.path for c in r.chunks], ["app/near.txt", "lib/mid.txt", "far/deep/er/x.txt"])

    def test_needs_no_embedder(self):
        make(retriever="proximity")
        make(retriever="random")

    def test_dense_and_hybrid_still_need_one(self):
        for retriever in ("dense", "hybrid"):
            with self.assertRaises(ValueError):
                make(retriever=retriever)


class StaleQueryTests(unittest.TestCase):
    def test_first_mode_takes_the_first_lines(self):
        self.assertEqual(make_query("a\nb\nc\nd", 2, "first"), "a\nb")
        self.assertEqual(make_query("a\nb\nc\nd", 2, "last"), "c\nd")
        self.assertEqual(make_query("a\nb\nc\nd", 2), "c\nd")  # the default is unchanged

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            make_query("a", 1, "middle")

    def test_stale_query_misses_what_was_just_written(self):
        b = make()
        last = [c.path for c in b.context_for(PREFIX, budget=1000, query_lines=1).chunks]
        first = [c.path for c in b.context_for(PREFIX, budget=1000, query_lines=1, query_mode="first").chunks]
        self.assertEqual(last, ["far/deep/er/x.txt"])
        self.assertEqual(first, [])  # "first line" matches nothing

    def test_defaults_are_untouched(self):
        b = make()
        self.assertEqual(b.context_for(PREFIX, budget=1000).context,
                         b.context_for(PREFIX, budget=1000, query_mode="last").context)


if __name__ == "__main__":
    unittest.main()
