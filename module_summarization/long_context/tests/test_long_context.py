import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag_pipeline.rag_budget import GPT41_MINI_SNAPSHOT
from rag_pipeline.rag_replication import build_prompt, digest, run


class CharacterTokenizer:
    fingerprint = 'fixture-tokenizer'
    count = staticmethod(len)

    @staticmethod
    def truncate(text, budget):
        return text[:budget]


class LongContextTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.row = {
            'repo': 'test', 'docfile_name': 'module.rst', 'intent': 'Describe the module',
            'relevant_code_context': 'x' * 7999 + 'é' + 'tail', 'target_text': 'reference only',
        }
        dataset = root / 'data.jsonl'
        dataset.write_text(json.dumps(self.row))
        self.config = {
            'name': 'long-context-test', 'model': GPT41_MINI_SNAPSHOT, 'ids': [0],
            'context_tokens': 8000, 'max_output_tokens': 2000, 'strategies': ['prefix'],
        }
        self.args = SimpleNamespace(
            config=root / 'config.json', dataset=dataset, tokenizer='fixture', mode='retrieve',
            output_root=root / 'runs', cache_dir=root / 'cache', ledger=root / 'spend.sqlite', env_file=None,
        )
        tokenizer = patch('rag_pipeline.rag_replication.CodeTokenizer', return_value=CharacterTokenizer())
        tokenizer.start()
        self.addCleanup(tokenizer.stop)

    def test_8k_offline_uses_contiguous_prefix(self):
        self.args.config.write_text(json.dumps(self.config))
        with patch('rag_pipeline.rag_replication.request_generation') as api:
            run(self.args)
            api.assert_not_called()
        evidence = json.loads(next(self.args.output_root.glob('*/prefix/0.context.json')).read_text())
        self.assertEqual(evidence['context'], self.row['relevant_code_context'][:8000])
        self.assertEqual(evidence['trace']['context_tokens'], 8000)
        self.assertEqual(evidence['trace']['selected'][0]['end_char'], 8000)
        self.assertEqual(evidence['trace']['context_sha256'], digest(evidence['context']))

    def test_rejects_changed_pinned_input(self):
        self.config['tokenizer_sha256'] = 'wrong-tokenizer'
        self.args.config.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, 'tokenizer_sha256'):
            run(self.args)

    def test_generation_rejects_unexpected_snapshot_and_accounts_for_usage(self):
        self.args.mode = 'generate'
        self.args.config.write_text(json.dumps(self.config))
        response = {
            'model': 'different-model',
            'usage': {'prompt_tokens': 7900, 'completion_tokens': 10},
            'choices': [{'message': {'content': 'Summary'}, 'finish_reason': 'stop'}],
        }
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fixture-key'}), \
             patch('rag_pipeline.rag_replication.request_generation', return_value=response) as api:
            with self.assertRaisesRegex(RuntimeError, 'different model snapshot'):
                run(self.args)
        _, model, messages, output_cap = api.call_args.args
        self.assertEqual(model, GPT41_MINI_SNAPSHOT)
        self.assertEqual(output_cap, 2000)
        self.assertEqual(messages[1]['content'], build_prompt(self.row, self.row['relevant_code_context'][:8000]))
        self.assertFalse(list(self.args.output_root.glob('*/prefix/0.txt')))
        error = json.loads(next(self.args.output_root.glob('*/prefix/0.error.json')).read_text())
        self.assertEqual(error['returned_model'], 'different-model')
        self.assertGreater(error['cost_usd'], 0)
