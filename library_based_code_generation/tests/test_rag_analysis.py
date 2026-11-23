import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation.rag_analysis import analyze, extract_code, inspect_arm, paired_stats, portable_path


class AnalysisTests(unittest.TestCase):
    def test_portable_paths_remove_windows_and_wsl_home_directories(self):
        suffix = "library_based_code_generation/results/plain/0.py"
        for prefix in ("C:/Users/private/Documents/", "/mnt/c/Users/private/Documents/"):
            self.assertEqual(portable_path(prefix + suffix), "results/plain/0.py")
        self.assertEqual(portable_path("C:\\Users\\private\\secret\\test.arrow"), "external/test.arrow")
        self.assertEqual(portable_path("results/plain/0.py"), "results/plain/0.py")

    def test_paired_cluster_bootstrap(self):
        result = paired_stats([.25, .25, .25], ["library_a", "library_a", "library_b"], 100)
        self.assertEqual(result["ci95"], [.25, .25])
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["wins"], 3)
        self.assertIsNone(paired_stats([], [], 100))

    def test_duplicate_and_out_of_range_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0.py").write_text("pass")
            (root / "00.py").write_text("pass")
            with self.assertRaises(ValueError):
                inspect_arm(root, 2)
            (root / "00.py").unlink()
            (root / "2.py").write_text("pass")
            with self.assertRaises(ValueError):
                inspect_arm(root, 2)

    def test_matched_cohort_excludes_missing_and_empty_but_scores_bad_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arms = {label: root / label for label in ("plain", "bm25", "rag")}
            for folder in arms.values():
                folder.mkdir()
                (folder / "0.py").write_text("not valid python !!!")
                (folder / "1.py").write_text("pass")
            (arms["rag"] / "1.py").write_text("")
            rows = [{"instruction": "test", "repo": "same"}] * 3
            report = analyze(rows, arms, root / "out", lambda code, row: {"ChrF": .5, "API_recall": .25},
                             group_field="repo", draws=100)
            self.assertEqual(report["common_indices"], [0])
            self.assertFalse(report["complete"])
            self.assertEqual(report["coverage"]["rag"]["missing_indices"], [2])
            self.assertEqual(report["coverage"]["rag"]["invalid_predictions"], {"1": "empty_prediction"})
            self.assertEqual(report["coverage"]["rag"]["syntax_valid_on_common"], 0)
            self.assertTrue((root / "out/REPORT.md").exists())
            self.assertEqual(json.loads((root / "out/per_example_metrics.json").read_text())["rag"][0]["index"], 0)

    def test_extraction_parity(self):
        for text in ("```python\nprint(1)\n```", "```\nprint(1)\n```", "<code>print(1)</code>", " print(1) "):
            self.assertEqual(extract_code(text), "print(1)")

    def test_metadata_model_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.json").write_text(json.dumps({"split": "test", "name": "wrong"}))
            with self.assertRaises(ValueError):
                inspect_arm(root, 1, "expected")


if __name__ == "__main__":
    unittest.main()
