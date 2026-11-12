import unittest

from verify_completed_run import (
    apply_edits, diff_blocks, digest, make_prompt, prediction_blocks, score,
    verify_prediction,
)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.row = {"id": 1, "logs": [{"log": "error: wrong x"}],
                    "diff": "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"}
        self.files = {"a.py": "x = 1\n"}
        gold = diff_blocks(self.row["diff"])
        blocks = prediction_blocks(gold)
        self.prediction = {
            "identity": {"run": "test", "id": 1, "seed": 0,
                         "method": "baseline", "snapshot": "snapshot"},
            "packed_chunks": [], "prompt": make_prompt(self.row, blocks),
            "response": "[start of a.py#L1#B0]\nx = 2\n[end of a.py#L1#B0]",
            "replacements": ["x = 2\n"], "status": "applied", "error": None,
            "finish_reason": "stop"}
        self.metric = {"id": 1, "seed": 0, "method": "baseline", "status": "applied",
                       "error": None,
                       **score(self.files, gold, apply_edits(self.files, blocks, ["x = 2\n"]), [], [])}
        self.sign()

    def sign(self):
        self.prediction.pop("payload_sha256", None)
        checksum = digest(self.prediction)
        self.prediction["payload_sha256"] = self.metric["prediction_sha256"] = checksum

    def verify(self):
        return verify_prediction(self.prediction, self.metric, self.row, self.files,
                                 "test", 0, "snapshot")

    def test_valid_prediction_scores_exactly(self):
        self.assertTrue(self.verify()["patch_exact_match"])

    def test_tampering_fails_checksum(self):
        self.prediction["response"] += "tampered"
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.verify()

    def test_retrieved_context_rejected_even_with_valid_checksum(self):
        self.prediction["packed_chunks"] = [{"text": "evidence"}]
        self.sign()
        with self.assertRaisesRegex(ValueError, "unaugmented"):
            self.verify()

    def test_prompt_changes_rejected(self):
        self.prediction["prompt"] += "extra context"
        self.sign()
        with self.assertRaisesRegex(ValueError, "unaugmented"):
            self.verify()

    def test_metric_changes_rejected(self):
        self.metric["edit_token_f1"] = 0.5
        with self.assertRaisesRegex(ValueError, "Recomputed"):
            self.verify()

    def test_saved_replacement_must_match_raw_answer(self):
        self.prediction["replacements"] = ["x = 99\n"]
        self.sign()
        with self.assertRaisesRegex(ValueError, "replacements"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
