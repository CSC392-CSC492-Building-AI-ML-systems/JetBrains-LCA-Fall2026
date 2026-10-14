import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from eval import compare_arms
from eval.analyze_arms import arm_lines, datapoint_means, paired_bootstrap, report
from eval.line_generators import dev_indices, heldout_indices, select_subset, subset_indices


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


class ExtendedSplitTests(unittest.TestCase):
    def test_default_sizes_on_small_context(self):
        dev, heldout = dev_indices(144, 20, 28), heldout_indices(144, 20, 28)
        self.assertEqual((len(dev), len(heldout)), (48, 96))
        self.assertEqual(set(dev) & set(heldout), set())
        self.assertEqual(sorted(dev + heldout), list(range(144)))

    def test_old_dev_set_stays_inside_dev(self):
        old_dev = set(subset_indices(144, 20))
        self.assertTrue(old_dev <= set(dev_indices(144, 20, 28)))
        self.assertTrue(old_dev.isdisjoint(heldout_indices(144, 20, 28)))

    def test_no_extra_reproduces_the_old_split(self):
        self.assertEqual(dev_indices(144, 20, 0), subset_indices(144, 20))
        self.assertEqual(heldout_indices(144, 20, 0), [i for i in range(144) if i not in set(subset_indices(144, 20))])

    def test_extra_datapoints_are_spread_out(self):
        extra = sorted(set(dev_indices(144, 20, 28)) - set(subset_indices(144, 20)))
        gaps = [b - a for a, b in zip(extra, extra[1:])]
        self.assertEqual(len(extra), 28)
        self.assertLessEqual(max(gaps), 12)

    def test_small_inputs_do_not_crash(self):
        self.assertEqual(sorted(dev_indices(5, 20, 28)), list(range(5)))
        self.assertEqual(heldout_indices(5, 20, 28), [])


class DriverSplitTests(unittest.TestCase):
    def cfg(self, **overrides):
        base = {"arm": "rag", "split": "all", "dev_size": 20, "dev_extra": 28, "limit_datapoints": None,
                "lines": "all", "static_mode": "provider"}
        return OmegaConf.create(base | overrides)

    def test_selection_per_split(self):
        dev = compare_arms.datapoint_selection(self.cfg(split="dev"))["select"](144)
        held = compare_arms.datapoint_selection(self.cfg(split="heldout"))["select"](144)
        self.assertEqual((len(dev), len(held)), (48, 96))
        self.assertEqual(set(dev) & set(held), set())
        self.assertEqual(compare_arms.datapoint_selection(self.cfg(limit_datapoints=5)), {"limit": 5})

    def test_git_state_reports_commit_and_dirty_flag(self):
        state = compare_arms.git_state()
        self.assertEqual(set(state), {"commit", "dirty"})
        self.assertTrue(state["commit"] is None or len(state["commit"]) >= 7)

    def test_invalid_split_and_combination_rejected(self):
        with self.assertRaises(ValueError):
            compare_arms.validate(self.cfg(split="test"))
        with self.assertRaises(ValueError):
            compare_arms.validate(self.cfg(split="dev", limit_datapoints=5))
        compare_arms.validate(self.cfg(split="heldout"))
        with self.assertRaises(ValueError):
            compare_arms.validate(self.cfg(lines="some"))
        compare_arms.validate(self.cfg(lines="targeted", split="dev"))
        with self.assertRaises(ValueError):
            compare_arms.validate(self.cfg(static_mode="fast"))

    def test_static_provider_matches_the_generators_static_prompt(self):
        from data_classes.datapoint_commit_dataset import DatapointCommitDataset

        dp = DatapointCommitDataset(
            repo_id=0, repo_name="repo", completion_lines={"infile": [1]},
            context_dict={"lib.py": "def helper(): pass\n", "other.py": "x = 1\n"},
            completion_dict={"target.py": "line0\nline1\n"})
        args = {"lang_sep_symbol": "", "meta_info_sep_symbol": "METASEP\n", "extension": ""}
        provider = compare_arms.static_provider_factory("path_distance", args)(dp)
        prompt, gt, info = provider(dp, 1)
        from composers.path_distance_composer import PathDistanceComposer

        expected_context = PathDistanceComposer(**args).context_composer(dp)
        self.assertEqual(prompt, expected_context + "\n" + "line0")  # what the generator builds: context, newline, prefix
        self.assertEqual((gt, info), ("line1", None))
        self.assertTrue(prompt.endswith("target.pyMETASEP\n\nline0"))


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

    def test_several_base_directories_are_merged_with_distinct_datapoint_ids(self):
        other = self.base / "second"
        other.mkdir()
        write_arm(self.base, "a", [row(0, "infile", "x", "x")])
        write_arm(other, "a", [row(0, "infile", "x", "y")])
        merged = arm_lines([str(self.base), str(other)], "a")
        self.assertEqual(sorted(merged), [(0, "infile", 0), (10_000, "infile", 0)])
        self.assertEqual([merged[k]["em"] for k in sorted(merged)], [1.0, 0.0])
        self.assertEqual(sorted(arm_lines(str(self.base), "a")), [(0, "infile", 0)])  # a single base still works

    def test_targets_keep_only_listed_lines_and_keep_keys_aligned(self):
        rows = [dict(row(0, "infile", "x", "x"), line_num=3), dict(row(0, "infile", "x", "y"), line_num=7),
                dict(row(1, "common", "x", "x"), line_num=2)]
        write_arm(self.base, "a", rows)
        write_arm(self.base, "b", [dict(r, prediction_line="y") for r in rows])
        targets = {"0": {"infile": [7]}, "1": {"common": [2]}}
        a = arm_lines(str(self.base), "a", [targets])
        b = arm_lines(str(self.base), "b", [targets])
        self.assertEqual(sorted(a), [(0, "infile", 1), (1, "common", 0)])  # k counts the skipped line too
        self.assertEqual(sorted(a), sorted(b))
        self.assertEqual([a[k]["em"] for k in sorted(a)], [0.0, 1.0])

    def test_targets_are_per_base_and_none_means_everything(self):
        other = self.base / "second"
        other.mkdir()
        write_arm(self.base, "a", [dict(row(0, "infile", "x", "x"), line_num=1)])
        write_arm(other, "a", [dict(row(0, "infile", "x", "y"), line_num=1), dict(row(0, "infile", "x", "x"), line_num=2)])
        merged = arm_lines([str(self.base), str(other)], "a", [None, {"0": {"infile": [2]}}])
        self.assertEqual(sorted(merged), [(0, "infile", 0), (10_000, "infile", 1)])

    def test_make_targets_positions_follow_the_split(self):
        from data_classes.datapoint_commit_dataset import DatapointCommitDataset
        from eval.make_targets import targets_for

        def datapoint(i):
            return DatapointCommitDataset(
                repo_id=i, repo_name=f"r{i}", completion_lines={"inproject": [1]},
                context_dict={"lib.py": "def helper(a):\n    return a\n"},
                completion_dict={"main.py": "import lib\nvalue = helper(1)\n"})

        data = [datapoint(i) for i in range(10)]
        held = targets_for(data, "heldout", dev_size=2, dev_extra=2)
        dev = targets_for(data, "dev", dev_size=2, dev_extra=2)
        self.assertEqual((len(dev), len(held)), (4, 6))
        self.assertEqual(sorted(held, key=int), [str(i) for i in range(6)])  # positions within the split
        self.assertEqual(held["0"], {"inproject": [1]})
        with self.assertRaises(ValueError):
            targets_for(data, "all", 2, 2)

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
