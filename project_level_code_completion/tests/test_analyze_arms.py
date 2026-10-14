import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from eval import compare_arms
from eval.analyze_arms import arm_lines, datapoint_means, paired_bootstrap, report
from eval.line_generators import select_subset, subset_indices


def write_arm(base: Path, name: str, rows: list[dict]):
    (base / name).mkdir()
    with open(base / name / "generation_results.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def row(dp, cls, pred, gt, zero=False):
    return {"datapoint_idx": dp, "repo": f"r{dp}", "line_class": cls, "prediction_line": pred, "ground_truth": gt,
            "zero_context": zero}


class SplitTests(unittest.TestCase):
    def test_dev_and_heldout_are_disjoint_and_cover_everything(self):
        items = list(range(144))
        dev = select_subset(items, 20)
        heldout = select_subset(items, None, heldout_of=20)
        self.assertEqual(len(dev), 20)
        self.assertEqual(len(heldout), 124)
        self.assertEqual(set(dev) & set(heldout), set())
        self.assertEqual(sorted(dev + heldout), items)

    def test_dev_indices_match_pilot_selection(self):
        self.assertEqual(subset_indices(144, 20), [i * 144 // 20 for i in range(20)])

    def test_heldout_keeps_dataset_order(self):
        heldout = select_subset(list(range(10)), None, heldout_of=2)
        self.assertEqual(heldout, sorted(heldout))


class DriverSplitTests(unittest.TestCase):
    def cfg(self, **overrides):
        base = {"arm": "rag", "split": "all", "dev_size": 20, "limit_datapoints": None}
        return OmegaConf.create(base | overrides)

    def test_selection_per_split(self):
        self.assertEqual(compare_arms.datapoint_selection(self.cfg(split="dev")), {"limit": 20})
        self.assertEqual(compare_arms.datapoint_selection(self.cfg(split="heldout")), {"heldout_of": 20})
        self.assertEqual(compare_arms.datapoint_selection(self.cfg(limit_datapoints=5)), {"limit": 5})

    def test_invalid_split_and_combination_rejected(self):
        with self.assertRaises(ValueError):
            compare_arms.validate(self.cfg(split="test"))
        with self.assertRaises(ValueError):
            compare_arms.validate(self.cfg(split="dev", limit_datapoints=5))
        compare_arms.validate(self.cfg(split="heldout"))


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def test_benchmark_style_mean_of_datapoint_means(self):
        # dp0: 1 of 3 lines right; dp1: 1 of 1 right -> mean of means = (1/3 + 1) / 2, not 2/4
        write_arm(self.base, "a", [row(0, "infile", "x", "x"), row(0, "infile", "x", "y"), row(0, "infile", "x", "z"),
                                   row(1, "infile", "q", "q")])
        means = datapoint_means(arm_lines(str(self.base), "a"), "em", None)
        self.assertAlmostEqual(sum(means.values()) / len(means), (1 / 3 + 1) / 2)

    def test_category_mean_only_over_datapoints_having_it(self):
        write_arm(self.base, "a", [row(0, "infile", "x", "x"), row(1, "common", "x", "y")])
        lines = arm_lines(str(self.base), "a")
        self.assertEqual(set(datapoint_means(lines, "em", "common")), {1})
        self.assertEqual(set(datapoint_means(lines, "em", None)), {0, 1})

    def test_zero_arm_falls_back_to_zero_pass_and_context_arm_ignores_it(self):
        write_arm(self.base, "zero", [row(0, "infile", "x", "x", zero=True)])
        write_arm(self.base, "ctx", [row(0, "infile", "x", "y", zero=True), row(0, "infile", "x", "x")])
        self.assertEqual(next(iter(arm_lines(str(self.base), "zero").values()))["em"], 1.0)
        self.assertEqual(next(iter(arm_lines(str(self.base), "ctx").values()))["em"], 1.0)  # not the zero row

    def test_paired_bootstrap_detects_consistent_gain(self):
        ref = {i: 0.2 for i in range(20)}
        better = {i: 0.4 for i in range(20)}
        delta, low, high, n = paired_bootstrap(ref, better, n_boot=500)
        self.assertAlmostEqual(delta, 0.2)
        self.assertGreater(low, 0)
        self.assertEqual(n, 20)

    def test_paired_bootstrap_identical_arms_gives_zero_interval(self):
        ref = {i: 0.3 for i in range(10)}
        self.assertEqual(paired_bootstrap(ref, dict(ref), n_boot=200)[:3], (0.0, 0.0, 0.0))

    def test_paired_bootstrap_noisy_difference_interval_includes_zero(self):
        ref = {i: 0.0 for i in range(10)}
        noisy = {i: (1.0 if i % 2 == 0 else -1.0) for i in range(10)}
        _, low, high, _ = paired_bootstrap(ref, noisy, n_boot=2000)
        self.assertLess(low, 0)
        self.assertGreater(high, 0)

    def test_bootstrap_is_deterministic(self):
        ref = {i: i / 10 for i in range(10)}
        other = {i: (i % 3) / 10 for i in range(10)}
        self.assertEqual(paired_bootstrap(ref, other, n_boot=300), paired_bootstrap(ref, other, n_boot=300))

    def test_report_renders_a_table_with_categories(self):
        write_arm(self.base, "ref", [row(i, "infile", "x", "y") for i in range(6)])
        write_arm(self.base, "new", [row(i, "infile", "x", "x") for i in range(6)])
        text = report(str(self.base), "ref", ["new"], n_boot=200)
        self.assertIn("### EM", text)
        self.assertIn("| infile (6) |", text)
        self.assertIn("+1.000", text)


if __name__ == "__main__":
    unittest.main()
