import copy
from pathlib import Path
import tempfile
import unittest

import long_context as context
import run_qwen_long_context as runner
from finish_rag_benchmark import supported_scores


class LongContextTests(unittest.TestCase):
    def test_reused_baseline_is_checksum_bound_and_prompt_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'source-run'
            answer = {'identity': {'run': folder.name, 'id': 1, 'seed': 0, 'arm': 'plain'},
                      'prompt': 'original prompt', 'response': 'answer', 'finish_reason': 'stop'}
            path = folder / 'examples/1/0/plain.json'
            checksum = runner.pilot.digest(answer)
            runner.pilot.save_json(path, {**answer, 'payload_sha256': checksum})
            copied = runner.reuse_plain(folder, 1, 0, 'original prompt', {'run': 'new'}, {'settings': 'same'})
            self.assertEqual(copied['response'], answer['response'])
            self.assertEqual(copied['reused_from']['payload_sha256'], checksum)
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                runner.reuse_plain(folder, 1, 0, 'changed prompt', {}, {})
            answer['response'] = 'tampered'
            runner.pilot.save_json(path, {**answer, 'payload_sha256': checksum})
            with self.assertRaises(ValueError):
                runner.reuse_plain(folder, 1, 0, 'original prompt', {}, {})

    def test_direct_imports_resolve_without_importing_project_code(self):
        files = {'src/pkg/core.py', 'src/pkg/helpers.py', 'src/pkg/__init__.py', 'src/external.py'}
        found = context.imported_files('src/pkg/core.py',
                                       'from .helpers import helper\nimport external\nimport os\n', files)
        self.assertEqual(found, ['src/external.py', 'src/pkg/helpers.py'])

    def test_complete_target_and_import_use_failed_text(self):
        files = {'pkg/a.py': 'from .b import value\nx = value\n', 'pkg/b.py': 'value = 1\n',
                 'pyproject.toml': '[tool.black]\nline-length = 88\n', 'unrelated.py': 'secret = 1\n'}
        blocks = [{'path': 'pkg/a.py', 'start': 2, 'count': 1}]
        evidence = context.pack('repair', files, blocks, len, context=4000, cap=200)
        context.validate(evidence, files)
        self.assertEqual({s['path'] for s in evidence['sections']}, {'pkg/a.py', 'pkg/b.py', 'pyproject.toml'})
        self.assertTrue(all(s['complete_file'] for s in evidence['sections']))
        self.assertEqual(evidence['partial_paths'], [])
        self.assertNotIn('unrelated.py', context.assemble('repair', evidence['sections']))

    def test_large_target_is_windowed_without_changing_oracle_prompt(self):
        files = {'a.py': ''.join(f'x_{i} = {i}\n' for i in range(1000))}
        base = 'UNCHANGED ORACLE BLOCKS'
        blocks = [{'path': 'a.py', 'start': 500, 'count': 2}]
        evidence = context.pack(base, files, blocks, len, context=1800, cap=200)
        context.validate(evidence, files)
        self.assertEqual(evidence['partial_paths'], ['a.py'])
        self.assertTrue(any(s['start'] <= 500 and s['end'] >= 501 for s in evidence['sections']))
        prompt = context.assemble(base, evidence['sections'])
        self.assertTrue(prompt.endswith(base))
        self.assertLessEqual(len(prompt) + 200 + 256, 1800)

    def test_tampered_context_and_overlaps_are_rejected(self):
        files = {'a.py': 'x = 1\n'}
        evidence = context.pack('repair', files, [{'path': 'a.py', 'start': 1, 'count': 1}], len, 1000, 100)
        changed = copy.deepcopy(evidence)
        changed['sections'][0]['text'] = 'x = 2\n'
        with self.assertRaisesRegex(ValueError, 'failed snapshot'):
            context.validate(changed, files)
        evidence['sections'] *= 2
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            context.validate(evidence, files)

    def test_over_budget_base_is_rejected_instead_of_truncated(self):
        with self.assertRaisesRegex(ValueError, 'Oracle prompt'):
            context.pack('x' * 1000, {}, [], len, context=1000, cap=100)

    def test_new_file_target_has_no_invented_failed_context(self):
        blocks = [{'path': 'new.py', 'start': 0, 'count': 0, 'old': ''}]
        packed = context.pack('create requested block', {}, blocks, len, 1000, 100)
        self.assertEqual(packed['sections'], [])
        self.assertEqual(packed['absent_target_paths'], ['new.py'])
        with self.assertRaisesRegex(ValueError, 'Missing failed target'):
            context.pack('repair', {}, [{**blocks[0], 'count': 1}], len, 1000, 100)

    def test_incomplete_pairs_and_unknown_ci_remain_unavailable(self):
        row = {'id': 1, 'seed': 0, 'arm': 'plain', 'error': None, 'nonempty_patch': True,
               'syntax_preserved': True, 'patch_exact_match': False, 'edit_token_f1': .2}
        self.assertFalse(runner.summarize([row], [1], [0])['complete'])
        result = runner.summarize([row, {**row, 'arm': 'long_context', 'edit_token_f1': .4}], [1], [0])
        self.assertTrue(result['complete'])
        self.assertEqual(result['paired_f1_wins_ties_losses'], [1, 0, 0])
        scores = supported_scores([{'id': 1, 'seed': 0, 'arm': 'long_context', 'status': 'pass'},
                                   {'id': 2, 'seed': 0, 'arm': 'long_context', 'status': 'unsupported'}],
                                  [1, 2], [0], runner.ARMS)
        self.assertIsNone(scores[1]['supported_case_ci_pass_at_1'])


if __name__ == '__main__':
    unittest.main()
