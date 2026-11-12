import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agentic.run import TurnJournal, generate
from agentic.tools import SourceTools, TOOLS
from bm25_retrieval.rag_retrieval import BM25Index, Chunk
from rag_pipeline.rag_budget import BudgetLedger, GPT41_MINI_SNAPSHOT, input_token_upper_bound
from rag_pipeline.rag_replication import build_prompt


class CharacterTokenizer:
    count = staticmethod(len)

    @staticmethod
    def truncate(text, budget):
        return text[:budget]


def completion(message, finish_reason='stop'):
    return {
        'model': GPT41_MINI_SNAPSHOT, 'id': 'test',
        'usage': {'prompt_tokens': 20, 'completion_tokens': 8, 'total_tokens': 28},
        'choices': [{'message': message, 'finish_reason': finish_reason}],
    }


class AgentTests(unittest.TestCase):
    def test_tool_scope_offsets_and_caps(self):
        code = 'a' * 2100 + 'target' + 'z' * 9000
        index = BM25Index(code, [Chunk(2100, 2300, {'target': 1})])
        tools = SourceTools(index, CharacterTokenizer(), 1600, 1500)
        self.assertEqual(tools.execute('read_code', {'chunk_id': 0})['status'], 'unknown_chunk')
        search = tools.execute('search_code', {'query': 'target'})
        self.assertEqual(tools.execute('search_code', {'query': '  TARGET  '})['status'], 'repeated_call')
        read = tools.execute('read_code', {'chunk_id': 0})
        self.assertEqual(tools.code_tokens, 1600)
        self.assertEqual(read['code_tokens'], 1400)
        for response in (search, read):
            for item in response['excerpts']:
                self.assertEqual(item['code'], code[item['start_char']:item['end_char']])
        context, spans = tools.final_context()
        self.assertEqual(context, '\n\n'.join(code[s['start_char']:s['end_char']] for s in spans))
        self.assertLessEqual(len(context), 1600)

    def test_six_calls_and_baseline_final_prompt_replay_without_api(self):
        code = 'target statement'
        index = BM25Index(code, [Chunk(0, len(code), {'target': 1})])
        row = {'docfile_name': 'target.rst', 'intent': 'Document target', 'target_text': 'HIDDEN'}
        responses = [
            completion({'tool_calls': [{
                'id': f'call-{i}', 'type': 'function',
                'function': {'name': 'search_code', 'arguments': '{"query":"target"}'},
            }]}, 'tool_calls')
            for i in range(6)
        ] + [completion({'content': 'Valid documentation.'})]
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            ledger = BudgetLedger(path / 'ledger.sqlite')
            self.addCleanup(ledger.db.close)
            with patch('agentic.run.request_completion', side_effect=responses) as api:
                result = generate(row, SourceTools(index, CharacterTokenizer(), 8000, 1500),
                                  TurnJournal(path/'turns', 'task', ledger, 'fake'), 6, 2000,
                                  force_tool_limit=True)
            self.assertEqual(api.call_count, 7)
            self.assertTrue(all(c.args[1]['tool_choice'] == 'required' for c in api.call_args_list[:6]))
            final = api.call_args.args[1]
            self.assertNotIn('tools', final)
            self.assertEqual(final['max_tokens'], 2000)
            self.assertEqual(final['messages'], [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': build_prompt(row, code)},
            ])
            self.assertNotIn('HIDDEN', str(api.call_args_list))
            trace = result[2]['agent_trace']
            self.assertEqual(trace['stop_reason'], 'tool_limit')
            self.assertEqual(trace['tool_calls'], 6)
            self.assertEqual(trace['tools'][1]['response']['status'], 'repeated_call')
            self.assertEqual(result[2]['usage']['prompt_tokens'], 7 * 20)
            with patch('agentic.run.request_completion') as api:
                replay = generate(row, SourceTools(index, CharacterTokenizer(), 8000, 1500),
                                  TurnJournal(path/'turns', 'task', ledger, 'fake'), 6, 2000,
                                  force_tool_limit=True)
            api.assert_not_called()
            self.assertEqual(replay, result)
            self.assertEqual(ledger.summary()['calls_by_state'], {'complete': 7})

    def test_capped_search_draft_gets_normal_final_request(self):
        row = {'docfile_name': 'target.rst', 'intent': 'Document target'}
        index = BM25Index('target', [Chunk(0, 6, {'target': 1})])
        responses = [
            completion({'content': 'Partial draft'}, 'length'),
            completion({'content': 'Documentation at the output cap.'}, 'length'),
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            ledger = BudgetLedger(path / 'ledger.sqlite')
            self.addCleanup(ledger.db.close)
            with patch('agentic.run.request_completion', side_effect=responses) as api:
                _, _, generation = generate(row, SourceTools(index, CharacterTokenizer(), 8000, 1500),
                                              TurnJournal(path/'turns', 'task', ledger, 'fake'), 6, 2000)
            self.assertEqual(api.call_count, 2)
            self.assertEqual(api.call_args.args[1]['messages'][1]['content'], build_prompt(row, ''))
            self.assertEqual(generation['agent_trace']['stop_reason'], 'search_output_limit')
            self.assertEqual(generation['finish_reason'], 'length')

    def test_received_turn_can_settle_after_interruption(self):
        messages = [{'role': 'user', 'content': 'test'}]
        self.assertGreater(input_token_upper_bound(messages, TOOLS), input_token_upper_bound(messages))
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            ledger = BudgetLedger(path / 'ledger.sqlite')
            self.addCleanup(ledger.db.close)
            with patch('agentic.run.request_completion', return_value=completion({'content': 'Saved'})), \
                 patch.object(ledger, 'settle', side_effect=RuntimeError('interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    TurnJournal(path/'turns', 'task', ledger, 'fake').complete(messages, 2000)
            self.assertEqual(ledger.summary()['calls_by_state'], {'reserved': 1})
            with patch('agentic.run.request_completion') as api:
                result = TurnJournal(path/'turns', 'task', ledger, 'fake').complete(messages, 2000)
            api.assert_not_called()
            self.assertEqual(result['message']['content'], 'Saved')
            self.assertEqual(ledger.summary()['calls_by_state'], {'complete': 1})
