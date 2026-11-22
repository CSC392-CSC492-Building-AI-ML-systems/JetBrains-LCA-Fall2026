import unittest

from finish_rag_benchmark import proxy_scores, supported_scores


class BenchmarkExportTests(unittest.TestCase):
    def test_unresolved_supported_case_does_not_inflate_ci_rate(self):
        rows = [{'arm': 'plain+markers+boundaries', 'source_arm': 'plain', 'seed': 0,
                 'id': i, 'status': status} for i, status in [(1, 'pass'), (2, 'controls_unresolved')]]
        result = supported_scores(rows, [1, 2], [0])[0]
        self.assertEqual(result['selected'], 2)
        self.assertEqual(result['resolved'], 1)
        self.assertIsNone(result['supported_case_ci_pass_at_1'])

    def test_invalid_prediction_remains_a_failed_attempt(self):
        rows = [{'arm': 'plain', 'seed': 0, 'id': i, 'status': status}
                for i, status in enumerate(['pass', 'fail', 'invalid_prediction'])]
        result = supported_scores(rows, [0, 1, 2], [0])[0]
        self.assertTrue(result['complete'])
        self.assertAlmostEqual(result['supported_case_ci_pass_at_1'], 1 / 3)

    def test_proxy_export_requires_every_selected_case(self):
        row = {'arm': 'plain', 'seed': 0, 'id': 1, 'error': None,
               'nonempty_patch': True, 'edit_token_f1': .5, 'patch_exact_match': False}
        with self.assertRaises(ValueError):
            proxy_scores({'cases': [row]}, [1, 2], [0])


if __name__ == '__main__':
    unittest.main()
