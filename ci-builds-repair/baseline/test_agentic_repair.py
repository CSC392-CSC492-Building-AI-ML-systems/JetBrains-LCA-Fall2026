import json
import tempfile
import unittest
from pathlib import Path

import agentic_repair as agent
from run_agentic_benchmark import call_cached


class AgenticRepairTests(unittest.TestCase):
    def setUp(self):
        self.blocks = [{'path': 'example.py', 'start': 1, 'count': 1, 'old': 'x = 1\n'}]
        self.targets = {'example.py': 'x = 1\n'}
        self.files = {**self.targets, 'helpers.py': 'y = 2\n'}

    def test_read_repair_feedback_then_submit_and_replay_without_api(self):
        def call(name, args):
            return {'type': 'function_call', 'name': name, 'arguments': json.dumps(args),
                    'call_id': 'call_' + name}

        def proposal(text):
            return call('propose_patch', {'edits': [{'block_id': 'B0', 'replacement': text}]})

        outputs = iter([call('read_file', {'path': 'helpers.py', 'start': 1, 'end': 10}),
                        proposal('x =\n'), proposal('x = 2\n')])

        def live(request):
            item = next(outputs)
            if item['name'] == 'propose_patch':
                self.assertEqual(request['input'][-1]['type'], 'function_call_output')
            self.assertIs(request['store'], False)
            return {'model': agent.MODEL, 'id': item['call_id'], 'status': 'completed',
                    'usage': {'input_tokens': 5, 'output_tokens': 3}, 'output': [item]}

        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            answer = agent.orchestrate('task', self.files, self.targets, self.blocks,
                                       call_cached(work, live, audit=False))
            self.assertEqual(answer['finish_reason'], 'stop')
            self.assertEqual(agent.parse_submission(answer['response'], self.blocks), ['x = 2\n'])
            self.assertIn('introduced_python_syntax_errors', answer['events'][1]['observation'])
            self.assertEqual(answer['prompt_tokens'], 15)
            replay = agent.orchestrate('task', self.files, self.targets, self.blocks,
                                       call_cached(work, None, audit=True))
            self.assertEqual(replay, answer)
            with self.assertRaisesRegex(ValueError, 'reconstructed request'):
                agent.orchestrate('different task', self.files, self.targets, self.blocks,
                                  call_cached(work, None, audit=True))

    def test_source_access_and_edit_addresses_are_bounded(self):
        lookup, _ = agent.use_tool('read_file', {'path': '../.env.local', 'start': 1, 'end': 1},
                                   self.files, self.targets, self.blocks)
        self.assertIn('error', lookup)
        self.assertEqual(agent.replacements({'edits': []}, self.blocks), ['x = 1\n'])
        edit = {'block_id': 'B0', 'replacement': ''}
        self.assertEqual(agent.replacements({'edits': [edit]}, self.blocks), [''])
        with self.assertRaisesRegex(ValueError, 'duplicate block'):
            agent.replacements({'edits': [edit, edit]}, self.blocks)

    def test_turn_limit_yields_invalid_attempt(self):
        answer = agent.orchestrate('task', self.files, self.targets, self.blocks,
                                   lambda *_: {'id': 'no_patch', 'status': 'completed', 'output': []})
        self.assertEqual(answer['skip'], 'agent_turn_limit')
        self.assertEqual(len(answer['api_responses']), agent.MAX_TURNS)


if __name__ == '__main__':
    unittest.main()
