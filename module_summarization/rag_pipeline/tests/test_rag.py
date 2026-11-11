import ast
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bm25_retrieval.rag_retrieval import BM25Index, Chunk, chunk_code, subtract_intervals, terms
from rag_pipeline.rag_replication import build_prompt, diverse_ids, ensure_manifest, run
from rag_pipeline.rag_budget import BudgetLedger


class CharTokenizer:
    fingerprint = 'test-only-character-tokenizer'
    def encode_offsets(self, text):
        return list(range(len(text))), [(i, i + 1) for i in range(len(text))]
    def count(self, text):
        return len(text)
    def truncate(self, text, budget):
        return text[:max(0, budget)]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer()

    def test_identifier_normalization(self):
        words = terms('custom_attributes getHTTPResponse foo-bar Generate documentation')
        self.assertTrue({'custom_attributes', 'custom', 'attributes', 'get', 'http', 'response', 'foo', 'bar'} <= set(words))
        self.assertNotIn('documentation', words)

    def test_actual_bm25_formula(self):
        index = BM25Index('aabb', [Chunk(0, 2, {'needle': 2}), Chunk(2, 4, {'other': 2})])
        self.assertAlmostEqual(index.rank('needle')[0][1], math.log(2) * 2 * 2.5 / (2 + 1.5))

    def test_late_evidence_after_old_512k_cutoff(self):
        code = 'x\n' * 260100 + 'late_unique_symbol = 42\n'
        index = BM25Index.load_or_build(code, self.tok, size=400, overlap=50)
        context, trace = index.retrieve('late_unique_symbol', self.tok, 400)
        self.assertIn('late_unique_symbol', context)
        self.assertGreater(trace['selected'][0]['start_char'], 512000)

    def test_offsets_and_budget(self):
        code = ('alpha_keyword = 1\n' * 20) + ('βγ delta = 2\n' * 20)
        index = BM25Index.load_or_build(code, self.tok, size=60, overlap=20)
        context, trace = index.retrieve('alpha_keyword delta', self.tok, 150)
        self.assertLessEqual(len(context), 150)
        fragments = [code[s['start_char']:s['end_char']] for s in trace['selected']]
        self.assertEqual(context, '\n\n'.join(fragments))
        spans = sorted((s['start_char'], s['end_char']) for s in trace['selected'])
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])))

    def test_smaller_later_candidate_still_fits(self):
        code = 'A' * 8 + 'B' * 8 + 'C' * 2
        index = BM25Index(code, [Chunk(0, 8, {'term': 1}), Chunk(8, 16, {'term': 1}), Chunk(16, 18, {'term': 1})])
        index.rank = lambda query: [(0, 3), (1, 2), (2, 1)]
        context, _ = index.retrieve('term', self.tok, 12)
        self.assertEqual(context, 'A' * 8 + '\n\nCC')

    def test_fallbacks_and_empty_source(self):
        for code, query, expected in [('alpha', 'zebra', 'no_term_match'), ('alpha', 'the documentation', 'empty_query'), ('', 'zebra', 'no_term_match')]:
            context, trace = BM25Index.load_or_build(code, self.tok).retrieve(query, self.tok, 3)
            self.assertEqual(trace['fallback'], expected)
            self.assertEqual(context, code[:3])

    def test_invalid_windows(self):
        for size, overlap in [(0, 0), (5, 5), (5, -1)]:
            with self.assertRaises(ValueError): chunk_code('abc', self.tok, size, overlap)

    def test_interval_subtraction(self):
        self.assertEqual(subtract_intervals(0, 20, [(3, 6), (10, 30)]), [(0, 3), (6, 10)])

    def test_disk_cache_and_tokenizer_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = BM25Index.load_or_build('alpha_beta = 1', self.tok, tmp)
            with patch('bm25_retrieval.rag_retrieval.chunk_code', side_effect=AssertionError('cache should be used')):
                b = BM25Index.load_or_build('alpha_beta = 1', self.tok, tmp)
            self.assertEqual(a.rank('alpha'), b.rank('alpha'))
            other = CharTokenizer(); other.fingerprint = 'other-tokenizer'
            BM25Index.load_or_build('alpha_beta = 1', other, tmp)
            self.assertEqual(len(list(Path(tmp).glob('*.gz'))), 2)

    def test_reference_text_cannot_change_prompt(self):
        row = {'intent': 'Explain alpha', 'docfile_name': 'alpha.rst', 'target_text': 'SECRET'}
        prompt = build_prompt(row, 'alpha = 1')
        row['target_text'] = 'DIFFERENT SECRET'
        self.assertEqual(prompt, build_prompt(row, 'alpha = 1'))
        self.assertNotIn('SECRET', prompt)

    def test_prompt_matches_original_baseline(self):
        # Extract the pure prompt function without importing its heavy dependencies.
        baseline = Path(__file__).resolve().parents[2] / 'baseline' / 'baseline_replication.py'
        if not baseline.exists(): self.skipTest('Original baseline not alongside installation')
        tree = ast.parse(baseline.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_one')
        namespace = {'gpt_generation': lambda client, prompt, model: prompt}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(baseline), 'exec'), namespace)
        row = {'intent': 'Explain alpha', 'docfile_name': 'alpha.rst'}
        self.assertEqual(build_prompt(row, 'alpha = 1'), namespace['generate_one'](row, 'alpha = 1', None, 'test'))

    def test_diverse_selection_stable_and_original_ids(self):
        rows = [{'repo': f'repo{i%6}', 'doc_type': f'type{i%3}', 'intent': 'alpha', 'context_chars': i*10000} for i in range(24)]
        ids = diverse_ids(rows, 6)
        self.assertEqual(ids, diverse_ids(rows, 6))
        self.assertEqual(len({rows[i]['repo'] for i in ids}), 6)
        self.assertEqual(diverse_ids(rows, 3), ids[:3])
        self.assertEqual(len(set(diverse_ids(rows, 24))), 24)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'ledger.sqlite'
        self.messages = [{'role': 'user', 'content': 'hello'}]
    def tearDown(self): self.tmp.cleanup()

    def test_shared_budget_and_atomic_cap(self):
        ledger = BudgetLedger(self.path, .004)
        ledger.reserve('one', 'gpt-3.5-turbo-0125', self.messages, 2000)
        with self.assertRaisesRegex(RuntimeError, 'Budget stop'):
            BudgetLedger(self.path, .004).reserve('two', 'gpt-3.5-turbo-0125', self.messages, 2000)

    def test_usage_settlement_and_duplicate_protection(self):
        ledger = BudgetLedger(self.path)
        reserved = ledger.reserve('one', 'gpt-3.5-turbo-0125', self.messages, 2000)
        cost = ledger.settle('one', {'prompt_tokens': 10, 'completion_tokens': 20})
        self.assertLess(cost, reserved)
        self.assertAlmostEqual(cost, .000035)
        with self.assertRaises(RuntimeError): ledger.reserve('one', 'gpt-3.5-turbo-0125', self.messages, 2000)

    def test_timeout_reservation_survives_restart(self):
        ledger = BudgetLedger(self.path)
        charge = ledger.reserve('timeout', 'gpt-3.5-turbo-0125', self.messages, 2000)
        self.assertEqual(BudgetLedger(self.path).summary()['held_for_unresolved_requests_usd'], charge)
        with self.assertRaises(ValueError): ledger.settle('timeout', {})
        self.assertEqual(ledger.summary()['held_for_unresolved_requests_usd'], charge)

    def test_auth_rejection_not_charged(self):
        ledger = BudgetLedger(self.path)
        ledger.reserve('one', 'gpt-3.5-turbo-0125', self.messages, 2000)
        ledger.reject('one')
        self.assertEqual(ledger.summary()['remaining_after_reservations_usd'], 4)

    def test_cap_cannot_increase_on_restart(self):
        BudgetLedger(self.path, 1)
        with self.assertRaises(ValueError): BudgetLedger(self.path, 4)

    def test_unknown_models_and_invalid_caps(self):
        ledger = BudgetLedger(self.path)
        with self.assertRaises(ValueError): ledger.reserve('one', 'unknown', self.messages, 1)
        for cap in [0, -1, 5]:
            with self.assertRaises(ValueError): BudgetLedger(self.path, cap)


class RunnerTests(unittest.TestCase):
    def test_manifest_rejects_stale_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'run'
            ensure_manifest(folder, {'model': 'one'})
            ensure_manifest(folder, {'model': 'one'})
            with self.assertRaises(RuntimeError): ensure_manifest(folder, {'model': 'two'})

    def test_offline_runner_keeps_original_ids_and_reference_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = [{'repo': f'r{i}', 'doc_type': 'Module doc', 'intent': 'alpha',
                     'docfile_name': 'alpha.rst', 'relevant_code_context': 'padding\nalpha_beta = 42',
                     'target_text': 'NEVER_INCLUDE_ME'} for i in range(3)]
            (folder/'data.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (folder/'config.json').write_text(json.dumps({'name':'test', 'model':'gpt-3.5-turbo-0125', 'ids':[2,0]}))
            args = SimpleNamespace(config=str(folder/'config.json'), mode='retrieve', dataset=str(folder/'data.jsonl'),
                tokenizer='test', output_root=str(folder/'out'), cache_dir=str(folder/'cache'))
            with patch('rag_pipeline.rag_replication.CodeTokenizer', return_value=CharTokenizer()), patch('rag_pipeline.rag_replication.request_generation', side_effect=AssertionError('No API calls in retrieval mode')):
                run(args)
            summary = json.loads(next((folder/'out').glob('*/summary.json')).read_text())
            self.assertEqual(summary['manifest']['original_ids'], [2,0])
            self.assertIsNone(summary['budget'])
            for file in (folder/'out').glob('*/*/*.context.json'):
                self.assertNotIn('NEVER_INCLUDE_ME', file.read_text())


if __name__ == '__main__': unittest.main()
