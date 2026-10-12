import unittest

from rag.chunker import MAX_CHUNK_TOKENS, chunk_file, chunk_repo, window_chunks

try:
    from tree_sitter_parser.parser import parser  # noqa: F401  (needs the grammar clone in the working directory)
    HAS_PARSER = True
except Exception:
    HAS_PARSER = False

SAMPLE = '''import os

LIMIT = 3


def top(a):
    return a + LIMIT


class Foo(Base):
    """Doc."""
    x = 1

    @property
    def bar(self):
        return self.x

    def baz(self):
        return top(self.x)
'''


def count(text: str) -> int:
    return len(text.split())


class WindowTests(unittest.TestCase):
    def test_overlap_and_coverage(self):
        lines = [f"line{i}" for i in range(100)]
        chunks = window_chunks(lines, "a.txt", 0, count, window=40, stride=20)
        self.assertEqual([(c.start_line, c.end_line) for c in chunks], [(0, 39), (20, 59), (40, 79), (60, 99)])

    def test_short_input_single_window_and_blank_skipped(self):
        self.assertEqual(len(window_chunks(["a", "b"], "a.txt", 0, count)), 1)
        self.assertEqual(window_chunks(["", "  "], "a.txt", 0, count), [])

    def test_offset_is_applied(self):
        chunks = window_chunks(["x"] * 5, "a.txt", 10, count)
        self.assertEqual((chunks[0].start_line, chunks[0].end_line), (10, 14))


class FallbackTests(unittest.TestCase):
    def test_non_python_file_uses_windows(self):
        chunks = chunk_file("README.md", "\n".join(["word"] * 50), count)
        self.assertTrue(chunks and all(c.kind == "window" for c in chunks))

    def test_parse_failure_uses_windows(self):
        def broken(_code):
            raise RuntimeError("no grammar")

        chunks = chunk_file("a.py", "def f():\n    pass\n", count, parse=broken)
        self.assertTrue(chunks and all(c.kind == "window" for c in chunks))


class RepoTests(unittest.TestCase):
    def test_excluded_path_never_indexed(self):
        files = {"a.md": "alpha beta", "target.md": "SECRET answer"}
        chunks = chunk_repo(files, count, exclude_paths=["target.md"])
        self.assertTrue(all(c.path != "target.md" for c in chunks))
        self.assertFalse(any("SECRET" in c.text for c in chunks))

    def test_exact_duplicates_dropped_first_path_wins(self):
        chunks = chunk_repo({"a.md": "same text", "b.md": "same text", "c.md": "other"}, count)
        self.assertEqual([c.path for c in chunks], ["a.md", "c.md"])


@unittest.skipUnless(HAS_PARSER, "tree-sitter grammar not available in the working directory")
class TreeSitterTests(unittest.TestCase):
    def setUp(self):
        self.chunks = chunk_file("pkg/mod.py", SAMPLE, count)
        self.by_kind = {}
        for c in self.chunks:
            self.by_kind.setdefault(c.kind, []).append(c)

    def test_expected_chunk_kinds(self):
        self.assertEqual(len(self.by_kind["function"]), 3)  # top, bar, baz
        self.assertEqual(len(self.by_kind["class_header"]), 1)
        self.assertEqual(len(self.by_kind["module_header"]), 1)

    def test_module_header_has_imports_and_constants_only(self):
        text = self.by_kind["module_header"][0].text
        self.assertIn("import os", text)
        self.assertIn("LIMIT = 3", text)
        self.assertNotIn("def top", text)

    def test_methods_get_class_line_and_decorators(self):
        method_texts = [c.text for c in self.by_kind["function"] if "self" in c.text]
        self.assertTrue(all(t.startswith("class Foo(Base):") for t in method_texts))
        self.assertTrue(any("@property" in t for t in method_texts))

    def test_class_header_has_docstring_and_attrs_not_methods(self):
        text = self.by_kind["class_header"][0].text
        self.assertIn('"""Doc."""', text)
        self.assertIn("x = 1", text)
        self.assertNotIn("def baz", text)

    def test_line_numbers_point_into_the_file(self):
        lines = SAMPLE.split("\n")
        top = next(c for c in self.by_kind["function"] if c.text.startswith("def top"))
        self.assertEqual(lines[top.start_line], "def top(a):")
        self.assertEqual(lines[top.end_line].strip(), "return a + LIMIT")

    def test_syntax_error_falls_back_to_windows(self):
        chunks = chunk_file("bad.py", "def (:\n  oops\n", count)
        self.assertTrue(chunks and all(c.kind == "window" for c in chunks))

    def test_oversized_function_is_windowed(self):
        body = "\n".join(f"    v{i} = {i}" for i in range(MAX_CHUNK_TOKENS * 2))
        chunks = chunk_file("big.py", f"def big():\n{body}\n", lambda t: len(t.split()) * 3)
        self.assertTrue(len(chunks) > 1 and all(c.kind == "window" for c in chunks))


if __name__ == "__main__":
    unittest.main()
