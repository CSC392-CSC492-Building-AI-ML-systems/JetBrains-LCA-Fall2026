import importlib.util
import sys
import unittest

import run_constrained_edit_smoke as smoke


class ShortIdTests(unittest.TestCase):
    files = {"x.py": "A = 1\nB = 2\n"}
    blocks = smoke.blocks_from(files, [("x.py", 1, 1), ("x.py", 2, 1)])

    def test_sparse_order_and_deletion(self):
        self.assertEqual(smoke.parse_answer('{"B1":"B = 3"}', self.blocks), ["A = 1\n", "B = 3\n"])
        self.assertEqual(smoke.parse_answer('{"B0":""}', self.blocks), ["", "B = 2\n"])
        self.assertEqual(smoke.parse_answer("{}", self.blocks), ["A = 1\n", "B = 2\n"])

    def test_reject_malformed_and_out_of_order_outputs(self):
        for bad in ('{"B2":"x"}', '{"B0":42}', '{"B0":"","B0":""}', '{"B1":"","B0":""}', '{} more'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                smoke.parse_answer(bad, self.blocks)

    def test_ast_and_output_cap(self):
        gold = {"x.py": "A = 3\nB = 2\n"}
        result = smoke.score_answer('{"B0":"A=3"}', "stop", self.files, self.blocks, gold)
        self.assertTrue(result["reference_ast_match"])
        self.assertFalse(smoke.score_answer('{"B0":"A=3"}', "length", self.files, self.blocks, gold)["format_valid"])
        invalid = smoke.score_answer('{"B0":"A = ("}', "stop", self.files, self.blocks, gold)
        self.assertTrue(invalid["format_valid"])
        self.assertFalse(invalid["syntax_valid"])

    def test_actual_tokenizer_mask(self):
        if smoke.DEPS.is_dir() and str(smoke.DEPS) not in sys.path:
            sys.path.insert(0, str(smoke.DEPS))
        if not importlib.util.find_spec("llguidance") or not (smoke.MODEL / "tokenizer.json").is_file():
            self.skipTest("Optional local model tokenizer or grammar package unavailable")
        from llguidance import LLMatcher, LLTokenizer
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(smoke.MODEL / "tokenizer.json"))
        eos = tokenizer.token_to_id("</s>")
        backend = LLTokenizer(tokenizer.to_str(), eos_token=eos)
        grammar = smoke.grammar_for(2)
        for valid in ('{}', '{"B1":""}', '{"B0":"café\\n","B1":""}'):
            with self.subTest(valid=valid):
                matcher = LLMatcher(backend, grammar)
                for token in tokenizer.encode(valid, add_special_tokens=False).ids:
                    mask = matcher.compute_logit_bias()
                    self.assertFalse(mask[eos])
                    self.assertTrue(mask[token])
                    self.assertTrue(matcher.consume_token(token))
                self.assertTrue(matcher.is_accepting())
        for invalid in ('{"B2":""}', '{"B0":"","B0":""}', '{"B1":"","B0":""}'):
            matcher = LLMatcher(backend, grammar)
            tokens = tokenizer.encode(invalid, add_special_tokens=False).ids
            self.assertLess(matcher.validate_tokens(tokens), len(tokens))


if __name__ == "__main__":
    unittest.main()
