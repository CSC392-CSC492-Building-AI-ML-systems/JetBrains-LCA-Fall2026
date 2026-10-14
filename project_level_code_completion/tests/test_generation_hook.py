import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from data_classes.datapoint_commit_dataset import DatapointCommitDataset
from eval import line_generators
from eval.line_generators import GeneratorConfig, LineGeneratorHF, evaluate_generation, select_subset
from rag.provider import make_rag_provider_factory


class FakeTokenizer:
    """Character-level tokenizer: token id = code point."""
    bos_token_id = 0
    eos_token_id = 1
    vocab = {"a": 2, "\n": 3}

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def __call__(self, text, return_tensors=None, padding=False):
        return {"input_ids": torch.tensor([[ord(c) for c in text]])}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["".join(chr(i) for i in ids[0].tolist())]


class StubModel:
    """Records the prompt it receives and 'generates' the line `pred`."""

    def __init__(self):
        self.prompts = []

    def generate(self, input_ids, **kwargs):
        self.prompts.append("".join(chr(i) for i in input_ids[0].tolist()))
        return torch.cat([input_ids, torch.tensor([[ord("p"), ord("r"), ord("e"), ord("d"), ord("\n")]])], dim=-1)


def fake_tokenizer_loader(self):
    self._tokenizer = FakeTokenizer()


def make_datapoint():
    return DatapointCommitDataset(
        repo_id=0, repo_name="repo", completion_lines={"infile": [1, 2]},
        context_dict={"lib.txt": "alpha_func beta_func", "other.txt": "unrelated"},
        completion_dict={"target.txt": "line0\nline1 alpha_func\nline2"}, context="CTX",
    )


class GeneratorHookTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(LineGeneratorHF, "_load_tokenizer", fake_tokenizer_loader)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.results_path = str(Path(self.tmp.name) / "results.jsonl")
        self.model = StubModel()
        self.generator = LineGeneratorHF(self.model, "cpu", max_seq_len=10_000, results_path=self.results_path,
                                         tokenizer_path="unused")

    def saved(self):
        return [json.loads(line) for line in Path(self.results_path).read_text().splitlines()]

    def test_static_prompt_is_unchanged(self):
        self.generator.generate_line(make_datapoint())
        self.assertEqual(self.model.prompts[0], "CTX\nline0")  # context + newline + prefix of line 1
        self.assertNotIn("retrieval", self.saved()[0])

    def test_provider_prompt_is_used_and_info_saved(self):
        calls = []

        def provider(dp, line_num):
            calls.append(line_num)
            return f"PROMPT{line_num}", dp.get_line(line_num), {"paths": ["lib.txt"]}

        self.generator.generate_line(make_datapoint(), context_provider=provider)
        self.assertEqual(calls, [1, 2])
        self.assertEqual(self.model.prompts, ["PROMPT1", "PROMPT2"])
        saved = self.saved()
        self.assertEqual(saved[0]["retrieval"], {"paths": ["lib.txt"]})
        self.assertEqual(saved[0]["ground_truth"], "line1 alpha_func")
        self.assertEqual(saved[0]["prediction_line"], "pred")

    def test_datapoint_idx_and_repo_saved_when_set(self):
        self.generator.datapoint_idx = 7
        self.generator.generate_line(make_datapoint())
        self.assertEqual((self.saved()[0]["datapoint_idx"], self.saved()[0]["repo"]), (7, "repo"))

    def test_datapoint_idx_absent_by_default(self):
        self.generator.generate_line(make_datapoint())
        self.assertNotIn("datapoint_idx", self.saved()[0])

    def test_zero_context_ignores_provider(self):
        def provider(dp, line_num):
            raise AssertionError("provider must not be called for the zero-context pass")

        self.generator.generate_line(make_datapoint(), use_zero_context=True, context_provider=provider)
        self.assertEqual(self.model.prompts[0], "line0")  # prefix only


class RagProviderTests(unittest.TestCase):
    def test_provider_builds_prompt_and_logs_retrieval(self):
        factory = make_rag_provider_factory(lambda text: len(text.split()), budget=1000,
                                            lang_sep="LANGSEP\n", meta_sep="METASEP\n")
        dp = make_datapoint()
        prompt, gt, info = factory(dp)(dp, 1)
        self.assertEqual(gt, "line1 alpha_func")
        self.assertTrue(prompt.endswith("target.txtMETASEP\n\nline0"))  # context ends with the file header
        self.assertNotIn("alpha_func", prompt)  # prefix "line0" shares no identifier with any file
        self.assertEqual(info["paths"], [])
        self.assertFalse(info["fallback"])

    def test_query_uses_prefix_and_target_line_is_not_leaked(self):
        factory = make_rag_provider_factory(lambda text: len(text.split()), budget=1000,
                                            lang_sep="LANGSEP\n", meta_sep="METASEP\n")
        dp = make_datapoint()
        _, _, info = factory(dp)(dp, 2)  # prefix contains "alpha_func"
        self.assertEqual(info["paths"], ["lib.txt"])
        prompt, _, _ = factory(dp)(dp, 1)
        self.assertNotIn("line1 alpha_func", prompt)  # the target line itself is never in the prompt

    def test_non_python_filter(self):
        dp = make_datapoint()
        dp.context_dict["lib.py"] = "alpha_func = 1"
        factory = make_rag_provider_factory(lambda text: len(text.split()), budget=1000, include_non_py=False,
                                            lang_sep="LANGSEP\n", meta_sep="METASEP\n")
        _, _, info = factory(dp)(dp, 2)
        self.assertEqual(info["paths"], ["lib.py"])


class SelectSubsetTests(unittest.TestCase):
    def test_none_and_oversized_limit_return_everything(self):
        self.assertEqual(select_subset([1, 2, 3], None), [1, 2, 3])
        self.assertEqual(select_subset([1, 2, 3], 5), [1, 2, 3])

    def test_evenly_spaced_and_deterministic(self):
        items = list(range(100))
        self.assertEqual(select_subset(items, 4), [0, 25, 50, 75])
        self.assertEqual(select_subset(items, 4), select_subset(items, 4))


class EvaluateGenerationTests(unittest.TestCase):
    def setUp(self):
        for patcher in (
            mock.patch.object(LineGeneratorHF, "_load_tokenizer", fake_tokenizer_loader),
            mock.patch.object(line_generators, "get_model", lambda args: (StubModel(), "cpu")),
            mock.patch.object(line_generators, "load", lambda name: FakeExactMatch()),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = GeneratorConfig(
            input_data_path="", seq_max_len=10_000, context_max=10_000, model="stub", device="cpu",
            best_perplexity=0.0, tokenizer_path="unused", composer="rag", seed=0,
            results_path=str(Path(self.tmp.name) / "results.jsonl"),
        )

    def provider_factory(self, dp):
        return lambda d, n: ("P", d.get_line(n), None)

    def test_context_only_has_no_zero_keys(self):
        scores, lists, em_diff, counts = evaluate_generation(
            self.args, input_data=[make_datapoint()], context_provider_factory=self.provider_factory,
            run_zero=False)
        self.assertEqual(set(scores), {"em", "es"})
        self.assertEqual(set(lists), {"em_list", "es_list"})
        self.assertEqual(em_diff, {})
        self.assertEqual(counts, {"infile": 2})

    def test_zero_only(self):
        scores, _, em_diff, counts = evaluate_generation(self.args, input_data=[make_datapoint()], run_context=False)
        self.assertEqual(set(scores), {"em_zero", "es_zero"})
        self.assertEqual(em_diff, {})
        self.assertEqual(counts, {"infile": 2})

    def test_both_passes_keep_original_output_shape(self):
        scores, lists, em_diff, _ = evaluate_generation(
            self.args, input_data=[make_datapoint()], context_provider_factory=self.provider_factory)
        self.assertEqual(list(scores), ["em_zero", "es_zero", "em", "es"])
        self.assertEqual(list(lists), ["em_zero_list", "es_zero_list", "em_list", "es_list"])
        self.assertIn("infile", em_diff)

    def test_evaluate_generation_records_datapoint_idx(self):
        evaluate_generation(self.args, input_data=[make_datapoint() for _ in range(3)], run_zero=False,
                            context_provider_factory=self.provider_factory)
        rows = [json.loads(line) for line in Path(self.args.results_path).read_text().splitlines()]
        self.assertEqual(sorted({r["datapoint_idx"] for r in rows}), [0, 1, 2])

    def test_heldout_of_evaluates_the_complement_of_limit(self):
        datapoints = [make_datapoint() for _ in range(10)]
        _, lists, _, _ = evaluate_generation(self.args, input_data=datapoints, run_zero=False,
                                             context_provider_factory=self.provider_factory, heldout_of=2)
        self.assertEqual(len(lists["em_list"]["all"]), 8)

    def test_select_overrides_limit_and_keeps_listed_datapoints(self):
        datapoints = [make_datapoint() for _ in range(6)]
        for i, dp in enumerate(datapoints):
            dp.repo_name = f"repo{i}"
        evaluate_generation(self.args, input_data=datapoints, run_zero=False, limit=1,
                            context_provider_factory=self.provider_factory, select=lambda n: [1, 4])
        rows = [json.loads(line) for line in Path(self.args.results_path).read_text().splitlines()]
        self.assertEqual(sorted({r["repo"] for r in rows}), ["repo1", "repo4"])

    def test_limit_selects_evenly_spaced_datapoints(self):
        datapoints = [make_datapoint() for _ in range(6)]
        _, lists, _, _ = evaluate_generation(self.args, input_data=datapoints, run_zero=False,
                                             context_provider_factory=self.provider_factory, limit=3)
        self.assertEqual(len(lists["em_list"]["all"]), 3)


class FakeExactMatch:
    def compute(self, references, predictions):
        hits = sum(r == p for r, p in zip(references, predictions))
        return {"exact_match": hits / len(references)}


if __name__ == "__main__":
    unittest.main()
