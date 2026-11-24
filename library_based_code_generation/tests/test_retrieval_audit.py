import unittest

from retrieval_recall import coverage, summarize, repository_identity_consistent


class RetrievalAuditTests(unittest.TestCase):
    def test_repository_identity_allows_dataset_separator_and_case(self):
        self.assertTrue(repository_identity_consistent({"repo_full_name": "owner__repo", "repo_owner": "Owner", "repo_name": "Repo"}))
        self.assertFalse(repository_identity_consistent({"repo_full_name": "other__repo", "repo_owner": "Owner", "repo_name": "Repo"}))

    def test_coverage_deduplicates_names_and_handles_empty_reference(self):
        self.assertEqual(coverage(["a", "a", "b"], ["a", "c"]), .5)
        self.assertEqual(coverage([], ["a"]), 0)
        self.assertIsNone(coverage(["a"], []))

    def test_summary_reports_macro_gap_without_empty_gold_bias(self):
        rows = [
            {"full_list_recall": 1., "bm25_at20_recall": .5, "retrieval_gap": .5, "full_context_tokens": 100},
            {"full_list_recall": .5, "bm25_at20_recall": .5, "retrieval_gap": 0., "full_context_tokens": 10000},
            {"full_list_recall": None, "bm25_at20_recall": None, "retrieval_gap": None, "full_context_tokens": 20000},
        ]
        result = summarize(rows)
        self.assertEqual(result["full_list_reference_recall_macro"], .75)
        self.assertEqual(result["macro_retrieval_gap"], .25)
        self.assertEqual(result["tasks_with_gold_beyond_top20"], 1)
        self.assertEqual(result["tasks_full_context_fits_8192"], 1)
        self.assertEqual(result["tasks_full_context_fits_65536"], 3)


if __name__ == "__main__":
    unittest.main()
