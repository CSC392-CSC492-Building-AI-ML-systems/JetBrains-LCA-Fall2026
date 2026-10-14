import unittest

from rag.builder import RagContextBuilder, parse_imports

try:
    from tree_sitter_parser.parser import parser  # noqa: F401  (needs the grammar clone in the working directory)
    HAS_PARSER = True
except Exception:
    HAS_PARSER = False

LANG, META = "LANGSEP\n", "METASEP\n"


def count(text: str) -> int:
    return len(text.split())


def builder(files, completion_path="app/main.py"):
    return RagContextBuilder(files, completion_path, "repo", count, lang_sep=LANG, meta_sep=META)


class ParseImportsTests(unittest.TestCase):
    def test_from_and_plain_imports(self):
        code = "import os, sys as system\nfrom pkg.util import helper, Tool as T\nfrom . import sibling\nx = 1\n"
        modules, names = parse_imports(code)
        self.assertEqual(modules, {"os", "sys", "pkg.util"})
        self.assertEqual(names, {"helper", "Tool", "sibling"})

    def test_star_and_parenthesised_imports(self):
        modules, names = parse_imports("from a import *\nfrom b import (c, d)\n")
        self.assertEqual((modules, names), ({"a", "b"}, {"c", "d"}))

    def test_no_imports(self):
        self.assertEqual(parse_imports("x = 1\n"), (set(), set()))


class ProximityTests(unittest.TestCase):
    FILES = {"app/main.py": "t", "far/deep/x.txt": "alpha_func one", "app/near.txt": "alpha_func two"}

    def test_without_weight_equal_scores_keep_file_order(self):
        result = builder(self.FILES).context_for("alpha_func", budget=1000)
        self.assertEqual([c.path for c in result.chunks], ["far/deep/x.txt", "app/near.txt"])

    def test_proximity_prior_prefers_the_nearer_file(self):
        result = builder(self.FILES).context_for("alpha_func", budget=1000, proximity_weight=0.5)
        self.assertEqual([c.path for c in result.chunks], ["app/near.txt", "far/deep/x.txt"])

    def test_proximity_never_adds_chunks_with_no_lexical_match(self):
        files = self.FILES | {"app/also_near.txt": "nothing relevant"}
        result = builder(files).context_for("alpha_func", budget=1000, proximity_weight=5.0)
        self.assertNotIn("app/also_near.txt", [c.path for c in result.chunks])


class ImportBoostTests(unittest.TestCase):
    FILES = {"app/main.py": "t", "lib/a.txt": "alpha_func one", "pkg/util.txt": "alpha_func two"}

    def test_files_matching_imported_modules_are_boosted(self):
        files = {"app/main.py": "t", "lib/a.py": "alpha_func = 1", "pkg/util.py": "alpha_func = 2"}
        prefix = "from pkg.util import helper\nalpha_func"
        plain = builder(files).context_for(prefix, budget=1000)
        boosted = builder(files).context_for(prefix, budget=1000, import_weight=0.5)
        self.assertEqual(boosted.chunks[0].path, "pkg/util.py")
        self.assertEqual(sorted(c.path for c in plain.chunks), sorted(c.path for c in boosted.chunks))

    def test_package_init_counts_as_the_module(self):
        files = {"app/main.py": "t", "lib/a.py": "alpha_func = 1", "pkg/__init__.py": "alpha_func = 2"}
        result = builder(files).context_for("import pkg\nalpha_func", budget=1000, import_weight=0.5)
        self.assertEqual(result.chunks[0].path, "pkg/__init__.py")


@unittest.skipUnless(HAS_PARSER, "tree-sitter grammar not available in the working directory")
class DefinitionRetrievalTests(unittest.TestCase):
    FILES = {
        "app/main.py": "t",
        "lib/tools.py": "def make_widget(size):\n    return size * 2\n",
        "lib/other.py": "def unrelated():\n    return 0\n",
        "lib/noise.py": "def noise():\n    return make_widget_like_name\n",
    }
    PREFIX = "from lib.tools import make_widget\n\nresult = make_widget(3)\nprint(result)"

    def test_chunks_record_their_symbol(self):
        symbols = {c.symbol for c in builder(self.FILES).chunks if c.symbol}
        self.assertEqual(symbols, {"make_widget", "unrelated", "noise"})

    def test_definition_of_a_used_name_is_selected_first_and_nearest_the_prefix(self):
        result = builder(self.FILES).context_for(self.PREFIX, budget=1000, definitions_first=True)
        self.assertEqual(result.n_definitions, 1)
        self.assertEqual(result.chunks[0].symbol, "make_widget")
        paths = [c.path for c in result.chunks]
        self.assertIn("lib/noise.py", paths)  # also retrieved lexically, so there is something to be ordered against
        self.assertLess(result.context.index("def noise"), result.context.index("def make_widget"))

    def test_off_by_default(self):
        result = builder(self.FILES).context_for(self.PREFIX, budget=1000)
        self.assertEqual(result.n_definitions, 0)

    def test_imported_name_is_found_even_if_not_in_the_last_lines(self):
        prefix = "from lib.tools import make_widget\n" + "\n".join(f"x{i} = {i}" for i in range(30))
        result = builder(self.FILES).context_for(prefix, budget=2000, definitions_first=True, query_lines=5)
        self.assertEqual(result.n_definitions, 1)
        without = builder(self.FILES).context_for(prefix, budget=2000, definitions_first=True, query_lines=5,
                                                  definition_imports=False)
        self.assertEqual(without.n_definitions, 0)

    def test_names_the_prefix_defines_itself_are_not_looked_up(self):
        prefix = "def make_widget(size):\n    return size\n\nresult = make_widget(3)"
        result = builder(self.FILES).context_for(prefix, budget=1000, definitions_first=True)
        self.assertEqual(result.n_definitions, 0)

    def test_definition_token_cap_and_total_budget_hold(self):
        files = {"app/main.py": "t"} | {f"lib/m{i}.py": f"def fn{i}(a):\n    return a + {i}\n" for i in range(6)}
        prefix = "\n".join(f"v{i} = fn{i}(1)" for i in range(6))
        capped = builder(files).context_for(prefix, budget=1000, definitions_first=True, definition_tokens=9)
        self.assertEqual(capped.n_definitions, 1)  # a definition chunk costs 7-8 tokens here
        full = builder(files).context_for(prefix, budget=60, definitions_first=True)
        self.assertLessEqual(count(full.context + "\n" + full.prefix), 60)


class OrderTests(unittest.TestCase):
    def test_best_first_reverses_the_emission_order(self):
        files = {"app/main.py": "t", "a/one.txt": "alpha_func beta_func", "b/two.txt": "alpha_func"}
        last = builder(files).context_for("alpha_func beta_func", budget=1000)
        first = builder(files).context_for("alpha_func beta_func", budget=1000, order="best_first")
        self.assertEqual([c.path for c in last.chunks], [c.path for c in first.chunks])  # same selection
        self.assertLess(last.context.index("b/two.txt"), last.context.index("a/one.txt"))
        self.assertLess(first.context.index("a/one.txt"), first.context.index("b/two.txt"))


if __name__ == "__main__":
    unittest.main()
