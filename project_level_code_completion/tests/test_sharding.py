import importlib.util
import json
import random
import statistics
import tempfile
import unittest
from pathlib import Path

from eval.line_generators import shard_indices

SCRIPT = Path(__file__).resolve().parents[1] / "rag_results" / "scripts" / "merge_shards.py"
spec = importlib.util.spec_from_file_location("merge_shards", SCRIPT)
merge_shards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge_shards)

CLASSES = ["infile", "inproject", "random", "commited"]


def datapoint_lines(original_idx: int) -> list[dict]:
    """Deterministic fake lines for one datapoint; some datapoints lack some line classes."""
    rng = random.Random(original_idx)
    present = [c for c in CLASSES if rng.random() < 0.7] or ["infile"]
    return [{"line_num": n, "line_class": rng.choice(present), "right": rng.random() < 0.5} for n in range(rng.randint(1, 9))]


def write_run(directory: Path, originals: list[int], shard_index=None, shard_count=None, wall=100.0):
    """What `evaluate_generation` + `compare_arms` write for a run over these datapoints (saved index = position)."""
    directory.mkdir(parents=True)
    rows, em = [], {"all": []}
    counts = {}
    for position, original in enumerate(originals):
        lines = datapoint_lines(original)
        for line in lines:
            rows.append({"datapoint_idx": position, "line_num": line["line_num"], "line_class": line["line_class"],
                         "ground_truth": f"gt{original}-{line['line_num']}", "prediction_line": "x",
                         "original_idx_for_test": original,
                         "metrics": {"input_tokens": 100 + original, "output_tokens": 5, "context_ms": 1.0, "gen_ms": 2.0}})
            counts[line["line_class"]] = counts.get(line["line_class"], 0) + 1
        em["all"].append(statistics.fmean(float(l["right"]) for l in lines))
        for category in {l["line_class"] for l in lines}:
            em.setdefault(category, []).append(statistics.fmean(float(l["right"]) for l in lines if l["line_class"] == category))
    es = {k: [v * 100 for v in values] for k, values in em.items()}
    (directory / "generation_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (directory / "per_datapoint_scores.json").write_text(json.dumps({"em_list": em, "es_list": es}))
    (directory / "config.json").write_text(json.dumps({"arm": "agent", "shard_index": shard_index, "shard_count": shard_count}))
    summary = {"arm": "agent", "arm_name": directory.name, "dataset": "medium_context", "model": "m", "token_budget": 15900,
               "context_cap": 1753, "split": "all", "lines": "all", "static_mode": "provider", "code": {"commit": "c"},
               "shard_index": shard_index, "shard_count": shard_count, "wall_seconds": wall,
               "cost": {"gpu": "g", "usd_per_gpu_hour": 0.5, "peak_gpu_memory_gb": 10.0, "arm_cost_usd": wall / 3600 * 0.5},
               "scores": {"em": {k: statistics.fmean(v) for k, v in em.items()}}, "em_difference_vs_zero": {},
               "line_counts": counts}
    (directory / "summary.json").write_text(json.dumps(summary))


class ShardIndexTests(unittest.TestCase):
    def test_shards_are_disjoint_and_cover_everything(self):
        for n, k in ((224, 3), (10, 4), (5, 8), (7, 1)):
            parts = [shard_indices(n, i, k) for i in range(k)]
            self.assertEqual(sorted(sum(parts, [])), list(range(n)))
            self.assertEqual(len(set(sum(parts, []))), n)

    def test_interleaved(self):
        self.assertEqual(shard_indices(10, 1, 3), [1, 4, 7])


class MergeTests(unittest.TestCase):
    def merged_and_reference(self, n=23, k=3):
        tmp = Path(tempfile.mkdtemp())
        write_run(tmp / "ref", list(range(n)))
        for i in range(k):
            write_run(tmp / f"s{i}", shard_indices(n, i, k), i, k, wall=100.0 + i)
        out = tmp / "merged"
        summary = merge_shards.merge(out, [tmp / f"s{i}" for i in reversed(range(k))])  # given out of order on purpose
        return tmp, out, summary

    def test_merged_per_datapoint_lists_equal_an_unsharded_run(self):
        for n, k in ((23, 3), (8, 2), (5, 5), (4, 3)):
            tmp, out, _ = self.merged_and_reference(n, k)
            merged = json.loads((out / "per_datapoint_scores.json").read_text())
            reference = json.loads((tmp / "ref" / "per_datapoint_scores.json").read_text())
            self.assertEqual(merged, reference, (n, k))

    def test_merged_lines_have_the_original_indices_and_no_loss(self):
        tmp, out, _ = self.merged_and_reference(23, 3)
        rows = [json.loads(l) for l in (out / "generation_results.jsonl").read_text().splitlines()]
        reference = [json.loads(l) for l in (tmp / "ref" / "generation_results.jsonl").read_text().splitlines()]
        self.assertEqual([(r["datapoint_idx"], r["line_num"], r["ground_truth"]) for r in rows],
                         [(r["datapoint_idx"], r["line_num"], r["ground_truth"]) for r in reference])
        self.assertTrue(all(r["datapoint_idx"] == r["original_idx_for_test"] for r in rows))

    def test_scores_counts_and_cost_are_recomputed(self):
        tmp, out, summary = self.merged_and_reference(23, 3)
        reference = json.loads((tmp / "ref" / "summary.json").read_text())
        self.assertEqual(summary["line_counts"], reference["line_counts"])
        for category, value in reference["scores"]["em"].items():
            self.assertAlmostEqual(summary["scores"]["em"][category], value)
        self.assertEqual(summary["wall_seconds"], 303.0)
        self.assertAlmostEqual(summary["cost"]["arm_cost_usd"], 303.0 / 3600 * 0.5, places=3)
        self.assertEqual(summary["cost"]["lines_with_metrics"], sum(reference["line_counts"].values()))
        self.assertIsNone(summary["shard_count"])

    def test_a_missing_shard_is_refused(self):
        tmp = Path(tempfile.mkdtemp())
        for i in (0, 2):
            write_run(tmp / f"s{i}", shard_indices(9, i, 3), i, 3)
        with self.assertRaises(SystemExit):
            merge_shards.merge(tmp / "m", [tmp / "s0", tmp / "s2"])

    def test_a_shard_run_twice_is_refused(self):
        tmp = Path(tempfile.mkdtemp())
        for i in range(2):
            write_run(tmp / f"s{i}", shard_indices(6, i, 2), i, 2)
        with open(tmp / "s0" / "generation_results.jsonl", "a") as f:  # a re-run appended to the same file
            f.write((tmp / "s0" / "generation_results.jsonl").read_text())
        with self.assertRaises(SystemExit):
            merge_shards.merge(tmp / "m", [tmp / "s0", tmp / "s1"])

    def test_shards_that_disagree_on_the_setting_are_refused(self):
        tmp = Path(tempfile.mkdtemp())
        for i in range(2):
            write_run(tmp / f"s{i}", shard_indices(6, i, 2), i, 2)
        summary = json.loads((tmp / "s1" / "summary.json").read_text())
        summary["context_cap"] = 256
        (tmp / "s1" / "summary.json").write_text(json.dumps(summary))
        with self.assertRaises(SystemExit):
            merge_shards.merge(tmp / "m", [tmp / "s0", tmp / "s1"])

    def test_it_never_writes_into_a_non_empty_folder(self):
        tmp = Path(tempfile.mkdtemp())
        for i in range(2):
            write_run(tmp / f"s{i}", shard_indices(6, i, 2), i, 2)
        (tmp / "m").mkdir()
        (tmp / "m" / "keep.txt").write_text("x")
        with self.assertRaises(SystemExit):
            merge_shards.merge(tmp / "m", [tmp / "s0", tmp / "s1"])


if __name__ == "__main__":
    unittest.main()
