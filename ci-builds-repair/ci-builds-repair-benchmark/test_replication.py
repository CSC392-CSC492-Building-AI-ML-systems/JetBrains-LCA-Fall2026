"""Small offline checks for replication accounting and CLI validation."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluate_replication import score_run


class ReplicationTests(unittest.TestCase):
    def test_partial_results_keep_missing_jobs_out_of_pass_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = {
                "baseline": "none", "baseline_kind": "no_repair",
                "dataset_ids": [1, 2, 3],
                "jobs_file": "jobs_ids_none.jsonl",
                "results_file": "jobs_results_none.jsonl",
            }
            (directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            jobs = [{"id": number, "commit": str(number), "branch_name": f"job{number}"}
                    for number in (1, 2)]
            (directory / manifest["jobs_file"]).write_text(
                "".join(json.dumps(row) + "\n" for row in jobs), encoding="utf-8"
            )
            result = {**jobs[0], "difficulty": 1, "conclusion": "success"}
            (directory / manifest["results_file"]).write_text(json.dumps(result) + "\n", encoding="utf-8")

            summary = score_run(directory)
            self.assertEqual(summary["coverage"], 1 / 3)
            self.assertEqual(summary["pass_rate_completed"], 1.0)
            self.assertEqual(summary["pass_rate_selected_lower_bound"], 1 / 3)
            self.assertEqual(summary["unresolved_ids"], [2])
            self.assertEqual(summary["not_submitted_ids"], [3])
            self.assertFalse(summary["complete"])

            (directory / manifest["results_file"]).write_text(
                json.dumps(result) + "\n" + json.dumps(result) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Duplicate or unexpected result"):
                score_run(directory)

if __name__ == "__main__":
    unittest.main()
