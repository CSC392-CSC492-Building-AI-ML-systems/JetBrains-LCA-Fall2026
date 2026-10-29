import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eval import compare_arms
from eval.run_metrics import cost_block, crop_to_last_tokens, summarize


class WordTokenizer:
    """One token per whitespace-separated word, with character offsets like a fast tokenizer."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offsets, position = [], 0
        for word in text.split(" "):
            offsets.append((position, position + len(word)))
            position += len(word) + 1
        return {"offset_mapping": offsets}


class CropToLastTokensTests(unittest.TestCase):
    def test_keeps_the_last_n_tokens_and_original_text(self):
        self.assertEqual(crop_to_last_tokens("a b c d e", 2, WordTokenizer()), "d e")

    def test_short_text_is_returned_unchanged(self):
        self.assertEqual(crop_to_last_tokens("a b", 5, WordTokenizer()), "a b")

    def test_zero_tokens_gives_empty_context(self):
        self.assertEqual(crop_to_last_tokens("a b c", 0, WordTokenizer()), "")

    def test_exactly_n_tokens_is_unchanged(self):
        self.assertEqual(crop_to_last_tokens("a b c", 3, WordTokenizer()), "a b c")

    def test_very_long_text_is_bounded_before_tokenizing(self):
        text = " ".join(["w"] * 100_000)
        self.assertEqual(crop_to_last_tokens(text, 3, WordTokenizer()), "w w w")


def write_rows(path, rows):
    Path(path).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def metric(input_tokens, output_tokens, context_ms, gen_ms, truncated=False, hit=False):
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "context_ms": context_ms,
            "gen_ms": gen_ms, "truncated": truncated, "hit_max_new_tokens": hit}


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "results.jsonl")

    def test_totals_means_and_percentiles(self):
        write_rows(self.path, [
            {"metrics": metric(100, 10, 1.0, 100.0)},
            {"metrics": metric(300, 30, 3.0, 300.0, truncated=True)},
            {"metrics": metric(200, 20, 2.0, 200.0, hit=True)},
        ])
        s = summarize(self.path, wall_seconds=6)
        self.assertEqual(s["lines_with_metrics"], 3)
        self.assertEqual(s["input_tokens_total"], 600)
        self.assertEqual(s["input_tokens_mean"], 200.0)
        self.assertEqual(s["output_tokens_total"], 60)
        self.assertEqual(s["gen_ms_p50"], 200.0)
        self.assertEqual(s["gen_ms_p95"], 290.0)
        self.assertAlmostEqual(s["share_truncated"], 1 / 3, places=3)
        self.assertAlmostEqual(s["share_hit_max_new_tokens"], 1 / 3, places=3)
        self.assertEqual(s["wall_seconds_per_line"], 2.0)

    def test_rows_without_metrics_are_ignored_and_empty_is_reported(self):
        write_rows(self.path, [{"prediction_line": "x"}])
        self.assertEqual(summarize(self.path), {"lines_with_metrics": 0})

    def test_cost_block_uses_the_hourly_rate_from_the_environment(self):
        write_rows(self.path, [{"metrics": metric(10, 1, 1.0, 1.0)}])
        with mock.patch.dict(os.environ, {"LCA_USD_PER_HOUR": "0.72"}):
            block = cost_block(self.path, 3600.0, "RTX 5090", 27.5)
        self.assertEqual((block["gpu"], block["peak_gpu_memory_gb"]), ("RTX 5090", 27.5))
        self.assertEqual((block["usd_per_gpu_hour"], block["arm_cost_usd"]), (0.72, 0.72))

    def test_cost_block_without_a_rate_claims_no_dollars(self):
        write_rows(self.path, [{"metrics": metric(10, 1, 1.0, 1.0)}])
        env = {k: v for k, v in os.environ.items() if k != "LCA_USD_PER_HOUR"}
        with mock.patch.dict(os.environ, env, clear=True):
            block = cost_block(self.path, 3600.0, None, None)
        self.assertIsNone(block["arm_cost_usd"])


class FakeComposer:
    def __init__(self, **kwargs):
        pass

    def context_composer(self, datapoint):
        return "w1 w2 w3 w4 w5"


class StaticProviderCapTests(unittest.TestCase):
    def provider(self, cap):
        fake_module = SimpleNamespace(FakeComposer=FakeComposer)
        specs = {"fake": {"module": "unused", "name": "FakeComposer"}}
        datapoint = SimpleNamespace(get_prefix=lambda n: "PREFIX", get_line=lambda n: "LINE")
        with mock.patch.object(compare_arms, "COMPOSERS", specs), \
                mock.patch.object(compare_arms.importlib, "import_module", lambda name: fake_module):
            factory = compare_arms.static_provider_factory("fake", {}, cap_tokens=cap, tokenizer=WordTokenizer())
            return factory(datapoint)(datapoint, 1)

    def test_without_a_cap_the_full_context_is_used(self):
        prompt, gt, info = self.provider(None)
        self.assertEqual((prompt, gt, info), ("w1 w2 w3 w4 w5\nPREFIX", "LINE", None))

    def test_with_a_cap_only_the_last_tokens_of_the_context_remain(self):
        prompt, _, _ = self.provider(2)
        self.assertEqual(prompt, "w4 w5\nPREFIX")


if __name__ == "__main__":
    unittest.main()
