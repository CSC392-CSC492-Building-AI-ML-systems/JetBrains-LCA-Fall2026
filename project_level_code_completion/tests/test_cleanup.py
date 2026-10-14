import os
import subprocess
import sys
import unittest
import warnings
from unittest import mock

from hydra import compose, initialize_config_dir

from rag import chunker
from rag.chunker import ParserUnavailable, chunk_file

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT, "eval", "config")


def count(text: str) -> int:
    return len(text.split())


class MissingGrammarWarningTests(unittest.TestCase):
    CODE = "def f():\n    return 1\n"

    def unavailable_state(self):
        return mock.patch.dict(chunker._PARSER_STATE, {"parser": None, "error": "OSError: no grammar", "warned": False})

    def test_missing_grammar_warns_once_and_falls_back_to_windows(self):
        with self.unavailable_state(), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = chunk_file("a.py", self.CODE, count)
            second = chunk_file("b.py", self.CODE, count)
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(runtime), 1)
        self.assertIn("tree-sitter grammar not available", str(runtime[0].message))
        self.assertIn("OSError: no grammar", str(runtime[0].message))
        self.assertTrue(all(c.kind == "window" for c in first + second))

    def test_a_syntax_error_in_one_file_does_not_warn(self):
        def bad_parse(_code):
            raise ValueError("syntax error")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chunks = chunk_file("a.py", self.CODE, count, parse=bad_parse)
        self.assertEqual([w for w in caught if issubclass(w.category, RuntimeWarning)], [])
        self.assertTrue(chunks and all(c.kind == "window" for c in chunks))

    def test_non_python_files_never_warn(self):
        with self.unavailable_state(), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chunk_file("notes.md", "some text", count)
        self.assertEqual(caught, [])

    def test_the_failure_is_remembered_instead_of_retried(self):
        with self.unavailable_state():
            with self.assertRaises(ParserUnavailable):
                chunker._default_parse(self.CODE)
            self.assertEqual(chunker._PARSER_STATE["error"], "OSError: no grammar")


class PresetTests(unittest.TestCase):
    def rag(self, *overrides):
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            return compose(config_name="compare_arms", overrides=list(overrides)).rag

    def test_default_is_plain_bm25(self):
        rag = self.rag()
        self.assertEqual((rag.prefix_cap, rag.continuation_lines, rag.retriever), (4000, 0, "bm25"))

    def test_v1_preset_keeps_the_prefix_in_full(self):
        rag = self.rag("rag=v1")
        self.assertEqual((rag.prefix_cap, rag.continuation_lines), (7900, 0))

    def test_f_preset_is_the_final_design(self):
        rag = self.rag("rag=F")
        self.assertEqual((rag.prefix_cap, rag.continuation_lines), (7900, 20))
        self.assertEqual((rag.retriever, rag.query_lines, rag.max_chunks, rag.order, rag.chunking),
                         ("bm25", 10, 30, "best_last", "structure"))

    def test_explicit_overrides_beat_the_preset(self):
        rag = self.rag("rag=F", "rag.continuation_lines=0")
        self.assertEqual((rag.prefix_cap, rag.continuation_lines), (7900, 0))

    def test_old_style_overrides_still_work(self):
        rag = self.rag("rag.prefix_cap=7900", "rag.continuation_lines=20")
        self.assertEqual(dict(rag), dict(self.rag("rag=F")))


class DependencyDirectionTests(unittest.TestCase):
    def test_the_retrieval_package_does_not_import_eval(self):
        code = ("import sys; import rag.builder, rag.provider, rag.chunker, rag.dense, rag.bm25; "
                "bad = sorted(m for m in sys.modules if m == 'eval' or m.startswith('eval.')); "
                "print('IMPORTED_EVAL:' + ','.join(bad))")
        env = dict(os.environ, PYTHONPATH=PROJECT, PYTHONDONTWRITEBYTECODE="1")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=PROJECT, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("IMPORTED_EVAL:", out.stdout)
        self.assertEqual(out.stdout.strip().split("IMPORTED_EVAL:")[1], "")

    def test_proxy_metrics_still_exposes_the_helpers(self):
        from eval import proxy_metrics
        from rag import code_names

        self.assertIs(proxy_metrics.identifiers, code_names.identifiers)
        self.assertIs(proxy_metrics.definitions, code_names.definitions)
        self.assertIs(proxy_metrics.KEYWORDS, code_names.KEYWORDS)


if __name__ == "__main__":
    unittest.main()
