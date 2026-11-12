import importlib.util
import sys
import unittest

import run_qwen_ci_dev as dev


class QwenDevelopmentTests(unittest.TestCase):
    blocks = [{"path": "settings.yaml", "start": 1, "count": 1, "old": "  count: 1\n"}]

    def test_formats_share_log_excerpt_and_block(self):
        logs = "error: followed by raw Python on subsequent lines"
        prompts = dev.prompts_for({"logs": [{"log": logs}]}, self.blocks)
        for prompt in prompts.values():
            self.assertIn(logs, prompt)
            self.assertIn("settings.yaml", prompt)
        self.assertIn("followed by raw replacement text", prompts["raw_plain"])

    def test_validate_and_restore_indent_without_running_code(self):
        answer = {"finish_reason": "stop", "response": "B0 REPLACE\ncount: 2\nEND\nDONE"}
        self.assertEqual(dev.replacements_for(answer, self.blocks, "raw_plain"), (["  count: 2\n"], None))
        self.assertEqual(dev.replacements_for({**answer, "finish_reason": "length"}, self.blocks, "raw_plain"),
                         (["  count: 1\n"], "output_token_limit"))
        self.assertEqual(dev.replacements_for({"skip": "input_budget"}, self.blocks, "raw_plain"),
                         (["  count: 1\n"], "input_budget"))

    def test_large_grammar_with_qwen_tokenizer(self):
        deps = dev.raw.previous.DEPS
        if deps.is_dir() and str(deps) not in sys.path:
            sys.path.insert(0, str(deps))
        if not importlib.util.find_spec("llguidance") or not (dev.MODEL / "tokenizer.json").exists():
            self.skipTest("Optional local Qwen tokenizer or grammar engine unavailable")
        from llguidance import LLMatcher, LLTokenizer
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(dev.MODEL / "tokenizer.json"))
        eos = tokenizer.token_to_id("<|im_end|>")
        n_vocab = dev.pilot.read_json(dev.MODEL / "config.json")["vocab_size"]
        backend = LLTokenizer(tokenizer.to_str(), n_vocab=n_vocab, eos_token=eos)
        m = LLMatcher(backend, dev.raw.grammar_for(19))
        for token in tokenizer.encode("B0 KEEP\nB18 REPLACE\ncount: 2\nEND\nDONE", add_special_tokens=False).ids:
            mask = m.compute_logit_bias()
            self.assertEqual(len(mask), n_vocab)
            self.assertFalse(mask[eos])
            self.assertTrue(mask[token])
            self.assertTrue(m.consume_token(token))
        self.assertTrue(m.is_accepting())


if __name__ == "__main__":
    unittest.main()
