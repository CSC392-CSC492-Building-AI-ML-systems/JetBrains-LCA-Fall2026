import importlib.util
import sys
import unittest

import run_raw_edit_smoke as raw


class RawEditTests(unittest.TestCase):
    files = {"x.py": "def f():\n    n = 1\n    return n\n\nTEMP = True\n"}
    blocks = raw.previous.blocks_from(files, [("x.py", 2, 2), ("x.py", 5, 1)])

    def test_actions_and_indentation(self):
        replacement = "B0 REPLACE\nif True:\n    return 2\nreturn 0\nEND\nB1 DELETE\nDONE"
        self.assertEqual(raw.parse_answer(replacement, self.blocks),
                         ["    if True:\n        return 2\n    return 0\n", ""])
        self.assertEqual(raw.parse_answer("B1 KEEP\nDONE", self.blocks), [b["old"] for b in self.blocks])
        self.assertEqual(raw.parse_answer("DONE", self.blocks), [b["old"] for b in self.blocks])
        self.assertIn("n = 1\nreturn n", raw.prompt_for("failure", self.blocks))

    def test_validation(self):
        for bad in ("", "B0 REPLACE\n\nEND\nDONE", "B1 DELETE\nB0 KEEP\nDONE",
                    "B0 REPLACE\nn = 3\nDONE", "B2 DELETE\nDONE", "DONE\ncomment"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                raw.parse_answer(bad, self.blocks)

    def test_actual_grammar(self):
        deps = raw.previous.DEPS
        if deps.is_dir() and str(deps) not in sys.path:
            sys.path.insert(0, str(deps))
        tokenizer_path = raw.previous.MODEL / "tokenizer.json"
        if not importlib.util.find_spec("llguidance") or not tokenizer_path.is_file():
            self.skipTest("Optional local grammar engine and tokenizer unavailable")
        from llguidance import LLMatcher, LLTokenizer
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        eos = tokenizer.token_to_id("</s>")
        backend = LLTokenizer(tokenizer.to_str(), eos_token=eos)
        grammar = raw.grammar_for(2)
        for answer in ("DONE", "B0 DELETE\nB1 KEEP\nDONE",
                       "B0 REPLACE\nreturn 2\nEND\nB1 DELETE\nDONE",
                       "B1 REPLACE\nline = 'café'\nEND\nDONE"):
            with self.subTest(answer=answer):
                matcher = LLMatcher(backend, grammar)
                for token in tokenizer.encode(answer, add_special_tokens=False).ids:
                    mask = matcher.compute_logit_bias()
                    self.assertFalse(mask[eos])
                    self.assertTrue(mask[token])
                    self.assertTrue(matcher.consume_token(token))
                self.assertTrue(matcher.is_accepting())
        for answer in ("B2 KEEP\nDONE", "B1 DELETE\nB0 KEEP\nDONE", "B0 KEEP\nB0 KEEP\nDONE"):
            matcher = LLMatcher(backend, grammar)
            tokens = tokenizer.encode(answer, add_special_tokens=False).ids
            self.assertLess(matcher.validate_tokens(tokens), len(tokens))


if __name__ == "__main__":
    unittest.main()
