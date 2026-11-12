"""Small offline checks for replication accounting and CLI validation."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluate_replication import score_run
from paper_mistral_baseline import apply_blocks, make_prompt, oracle_blocks, parse_response, relevant_logs


class ReplicationTests(unittest.TestCase):
    def test_oracle_prompt_uses_old_code_and_not_gold_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "sample.py"
            source.write_text("before\nwrong\nafter\n", encoding="utf-8")
            diff = ("--- a/sample.py\n+++ b/sample.py\n@@ -1,3 +1,3 @@\n"
                    " before\n-wrong\n+secret_gold_fix\n after\n")
            blocks = oracle_blocks(diff, temporary)
            prompt = make_prompt({"logs": [{"log": "line 1\nERROR failed\nline 3"}]}, blocks)
            self.assertIn("wrong", prompt)
            self.assertNotIn("secret_gold_fix", prompt)
            self.assertEqual(relevant_logs([{"log": "a\nERROR bad\nb"}]), "a\nERROR bad\nb")
            tag = "sample.py#L1#B0"
            answer = f"[start of {tag}]\nbefore\nfixed\nafter\n[end of {tag}]"
            apply_blocks(temporary, blocks, parse_response(answer, blocks))
            self.assertEqual(source.read_text(encoding="utf-8"), "before\nfixed\nafter\n")

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
