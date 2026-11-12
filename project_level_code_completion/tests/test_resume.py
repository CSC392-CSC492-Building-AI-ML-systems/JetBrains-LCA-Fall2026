import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from eval.line_generators import (GenerationResults, LineGeneratorHF, prepare_resume, split_finished_datapoints)


@dataclass
class FakeDatapoint:
    completion_lines: dict

    def get_line(self, line_num: int) -> str:
        return f"gt{line_num}"


def row(datapoint_idx, line_class, line_num, prediction="p"):
    return {"datapoint_idx": datapoint_idx, "line_class": line_class, "line_num": line_num,
            "ground_truth": f"gt{line_num}", "prediction_line": prediction, "metrics": {"input_tokens": 1}}


def rows_of(datapoint_idx, datapoint):
    return [row(datapoint_idx, c, n) for c, lines in datapoint.completion_lines.items() for n in lines]


def generator(datapoint_idx):
    gen = object.__new__(LineGeneratorHF)  # no model or tokenizer needed to restore saved rows
    gen.generation_results = {}
    gen.datapoint_idx = datapoint_idx
    return gen


class SplitFinishedTests(unittest.TestCase):
    def test_complete_datapoints_are_kept_and_the_partial_one_is_set_aside(self):
        rows = [row(0, "a", 1), row(0, "a", 2), row(1, "a", 3), row(1, "a", 4), row(2, "a", 5)]
        finished, unfinished = split_finished_datapoints(rows, {0: 2, 1: 2, 2: 3})
        self.assertEqual(sorted(finished), [0, 1])
        self.assertEqual(unfinished, [row(2, "a", 5)])

    def test_more_rows_than_expected_is_not_finished(self):
        finished, unfinished = split_finished_datapoints([row(0, "a", 1), row(0, "a", 1)], {0: 1})
        self.assertEqual(finished, {})
        self.assertEqual(len(unfinished), 2)

    def test_a_datapoint_outside_the_selection_is_never_reused(self):
        finished, unfinished = split_finished_datapoints([row(9, "a", 1)], {0: 1})
        self.assertEqual(finished, {})
        self.assertEqual(len(unfinished), 1)


class PrepareResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "generation_results.jsonl")
        self.data = [FakeDatapoint({"infile": [1, 2], "inproject": [3]}), FakeDatapoint({"infile": [4]}),
                     FakeDatapoint({"infile": [5, 6]})]

    def write(self, rows):
        Path(self.path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")

    def read(self, path):
        return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines()]

    def test_no_file_means_nothing_to_resume(self):
        self.assertEqual(prepare_resume(self.path, self.data), {})
        self.assertFalse(Path(self.path).exists())

    def test_file_is_cut_back_to_the_finished_datapoints(self):
        saved = rows_of(0, self.data[0]) + rows_of(1, self.data[1]) + [row(2, "infile", 5)]
        self.write(saved)
        finished = prepare_resume(self.path, self.data)
        self.assertEqual(sorted(finished), [0, 1])
        self.assertEqual(self.read(self.path), saved[:-1])
        self.assertEqual(self.read(self.path + ".unfinished"), [row(2, "infile", 5)])

    def test_resuming_a_finished_run_changes_nothing(self):
        saved = [r for i, d in enumerate(self.data) for r in rows_of(i, d)]
        self.write(saved)
        self.assertEqual(sorted(prepare_resume(self.path, self.data)), [0, 1, 2])
        self.assertEqual(self.read(self.path), saved)
        self.assertFalse(Path(self.path + ".unfinished").exists())

    def test_resuming_twice_is_the_same_as_once(self):
        self.write(rows_of(0, self.data[0]) + [row(1, "infile", 4)])
        prepare_resume(self.path, self.data)
        after_first = self.read(self.path)
        prepare_resume(self.path, self.data)
        self.assertEqual(self.read(self.path), after_first)


class RestoreLineResultsTests(unittest.TestCase):
    def test_restored_results_equal_what_generate_line_keeps(self):
        datapoint = FakeDatapoint({"infile": [1, 2], "inproject": [3]})
        saved = [row(0, "infile", 1, "x"), row(0, "infile", 2, "y"), row(0, "inproject", 3, "z")]
        gen = generator(0)
        counts = gen.restore_line_results(datapoint, saved)
        self.assertEqual(counts, {"infile": 2, "inproject": 1})
        self.assertEqual(gen.generation_results["infile"], GenerationResults(["x", "y"], ["gt1", "gt2"]))
        self.assertEqual(gen.generation_results["inproject"], GenerationResults(["z"], ["gt3"]))

    def test_rows_of_another_datapoint_are_refused(self):
        datapoint = FakeDatapoint({"infile": [1, 2]})
        for wrong in ([row(0, "infile", 1), row(0, "infile", 7)],          # other line number
                      [row(0, "infile", 1), row(0, "inproject", 2)],        # other class
                      [row(0, "infile", 1)],                                # a row is missing
                      [dict(row(0, "infile", 1), ground_truth="other"), row(0, "infile", 2)]):
            with self.assertRaises(ValueError):
                generator(0).restore_line_results(datapoint, wrong)


if __name__ == "__main__":
    unittest.main()
