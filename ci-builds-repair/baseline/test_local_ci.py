"""Failure-sensitive checks for workflow compilation and scoring."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from local_ci.workflows import Unsupported, compile_workflow
import run_local_ci as local_ci
from run_local_ci import checked, controls_valid, digest, evaluation_source_runs, summarize, workflow_result


def fixture():
    return {'repo_name': 'starlette', 'workflow': yaml.safe_dump({
        'env': {'PIP_INDEX_URL': 'http://localhost:8629/2024-01-22',
                'UV_INDEX_URL': 'http://localhost:8629/2024-01-22'},
        'jobs': {'tests': {'runs-on': 'ubuntu-latest',
            'services': {'pypi_wayback': {'image': 'ghcr.io/waleko/pypi-wayback', 'ports': ['8629:8080']}},
            'strategy': {'matrix': {'python-version': ['3.8', '3.9', '3.10', '3.11', '3.12']}},
            'steps': [{'uses': 'actions/checkout@v4'}, {'uses': 'actions/setup-python@v5',
                       'with': {'python-version': '${{ matrix.python-version }}'}},
                      {'name': 'Install dependencies', 'run': 'scripts/install'},
                      *[{'name': name, 'run': command} for name, command in [
                          ('Run linting checks', 'scripts/check'), ('Build package & docs', 'scripts/build'),
                          ('Run tests', 'scripts/test'), ('Enforce coverage', 'scripts/coverage')]]]}}})}


def job(code=0, phase='check', name='Run tests'):
    return {'steps': [{'exit_code': code, 'phase': phase, 'name': name}], 'passed': code == 0}


class WorkflowTests(unittest.TestCase):
    def test_full_matrix_and_command_preservation(self):
        plans = compile_workflow(fixture())
        self.assertEqual([p['python'] for p in plans], ['3.8', '3.9', '3.10', '3.11', '3.12'])
        self.assertTrue(all(len(p['steps']) == 5 for p in plans))
        self.assertEqual(plans[0]['steps'][3]['run'], 'scripts/test')
        self.assertEqual(plans[0]['env']['PIP_INDEX_URL'], 'http://package-index:8080/2024-01-22')

    def mutate(self, function):
        row = fixture()
        data = yaml.safe_load(row['workflow'])
        function(data)
        row['workflow'] = yaml.safe_dump(data)
        with self.assertRaises(Unsupported):
            compile_workflow(row)

    def test_unknown_actions_cannot_be_skipped(self):
        self.mutate(lambda d: d['jobs']['tests']['steps'].insert(2, {'uses': 'unknown/action@v1'}))

    def test_conditions_cannot_be_ignored(self):
        self.mutate(lambda d: d['jobs']['tests']['steps'][3].update({'if': 'always()'}))

    def test_missing_matrix_jobs_cannot_be_hidden(self):
        self.mutate(lambda d: d['jobs']['tests']['strategy']['matrix'].update({'exclude': [{'python-version': '3.8'}]}))

    def test_missing_checks_cannot_pass(self):
        self.mutate(lambda d: d['jobs']['tests']['steps'].pop())

    def test_unexpanded_expressions_rejected(self):
        self.mutate(lambda d: d['jobs']['tests']['steps'][3].update({'run': '${{ secrets.TOKEN }}'}))

    def test_host_platform_not_silently_replaced(self):
        self.mutate(lambda d: d['jobs']['tests'].update({'runs-on': 'windows-latest'}))

    def test_workflow_defaults_not_ignored(self):
        self.mutate(lambda d: d.update({'defaults': {'run': {'shell': 'sh'}}}))


class ScoringTests(unittest.TestCase):
    def test_audited_unsupported_cases_allow_control_reuse_without_changing_score(self):
        audited = {'complete': False, 'manifest': {'dataset_sha256': 'same'},
                   'controls': [{'id': 1, 'controls_valid': True}],
                   'predictions': [{'status': 'pass'}, {'status': 'unsupported'}]}
        local_ci.validate_reuse(audited, 'same')
        self.assertFalse(audited['complete'])
        self.assertEqual(audited['predictions'][1]['status'], 'unsupported')
        with self.assertRaisesRegex(ValueError, 'same dataset'):
            local_ci.validate_reuse(audited, 'changed')
        for invalid in ('controls_unresolved', 'infrastructure_error', 'unknown'):
            changed = copy.deepcopy(audited)
            changed['predictions'][0]['status'] = invalid
            with self.assertRaisesRegex(ValueError, 'unresolved workflow'):
                local_ci.validate_reuse(changed, 'same')
        for controls in ([], [{'id': 1, 'controls_valid': False}]):
            changed = {**audited, 'controls': controls}
            with self.assertRaisesRegex(ValueError, 'unresolved failed/reference'):
                local_ci.validate_reuse(changed, 'same')

    def test_cross_model_reuse_audits_original_source_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            meta = {'model': 'original', 'ids': [1], 'seeds': [0]}
            identity = digest(meta)[:20]
            source = base / 'out/old-model' / identity
            source.mkdir(parents=True)
            (source / 'run_metadata.json').write_text(json.dumps(meta))
            evaluation = base / 'evaluation'
            evaluation.mkdir()
            (evaluation / 'manifest.json').write_text(json.dumps({'runs': {identity: digest(meta)}}))
            with patch.object(local_ci, 'BENCHMARK', base):
                self.assertEqual(evaluation_source_runs(evaluation, []), [source])
                (source / 'run_metadata.json').write_text(json.dumps({'model': 'tampered'}))
                with self.assertRaises(ValueError):
                    evaluation_source_runs(evaluation, [])

    def test_all_matrix_jobs_required(self):
        self.assertEqual(workflow_result([job()], 5), 'infrastructure_error')
        self.assertEqual(workflow_result([job(), job(1)], 2), 'fail')

    def test_setup_failure_and_timeout_are_unresolved(self):
        self.assertEqual(workflow_result([job(1, 'setup')], 1), 'infrastructure_error')
        self.assertEqual(workflow_result([job(124)], 1), 'infrastructure_error')
        self.assertEqual(workflow_result([{'steps': [], 'passed': True}], 1), 'infrastructure_error')

    def test_both_controls_and_recorded_step_required(self):
        row = {'logs': [{'step_name': 'Python 3.9/7_Run tests.txt'}]}
        fail = {'status': 'fail', 'jobs': [job(1)]}
        fixed = {'status': 'pass', 'jobs': [job()]}
        self.assertTrue(controls_valid(fail, fixed, row))
        self.assertFalse(controls_valid(fixed, fixed, row))
        self.assertFalse(controls_valid(fail, fail, row))
        other = copy.deepcopy(fail)
        other['jobs'][0]['steps'][0]['name'] = 'Different failure'
        self.assertFalse(controls_valid(other, fixed, row))

    def test_seeds_not_best_of_three(self):
        records = [{'run': 'r', 'model': 'm', 'arm': 'a', 'seed': s, 'id': 1,
                    'status': 'pass' if s == 0 else 'fail'} for s in range(3)]
        report = summarize(records)
        self.assertAlmostEqual(report['across_seeds'][0]['mean_local_ci_pass_at_1'], 1 / 3)

    def test_unresolved_not_dropped_from_denominator(self):
        records = [{'run': 'r', 'model': 'm', 'arm': 'a', 'seed': 0, 'id': i, 'status': status}
                   for i, status in enumerate(['pass', 'controls_unresolved', 'unsupported', 'invalid_prediction'])]
        result = summarize(records)['per_seed'][0]
        self.assertFalse(result['complete'])
        self.assertIsNone(result['local_ci_pass_at_1'])
        self.assertEqual(result['selected'], 4)
        self.assertEqual(result['resolved'], 2)
        self.assertEqual(result['pass_rate_lower_bound'], .25)
        self.assertEqual(result['pass_rate_upper_bound'], .75)

    def test_duplicate_predictions_rejected(self):
        row = {'run': 'r', 'model': 'm', 'arm': 'a', 'seed': 0, 'id': 1, 'status': 'pass'}
        with self.assertRaises(ValueError):
            summarize([row, row])

    def test_changed_answer_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prediction.json'
            payload = {'response': 'original', 'identity': {'id': 1}}
            path.write_text(json.dumps({**payload, 'payload_sha256': digest(payload)}))
            self.assertEqual(checked(path)[0], payload)
            path.write_text(json.dumps({**payload, 'response': 'changed', 'payload_sha256': digest(payload)}))
            with self.assertRaises(ValueError):
                checked(path)

    def test_same_step_with_different_failure_is_not_a_valid_control(self):
        failed = {'status': 'fail', 'jobs': [job(1, name='Running yapf')]}
        reference = {'status': 'pass', 'jobs': [job()]}
        failed['jobs'][0]['steps'][0]['log'] = '@@ -1 +1 @@\n-wrong\n+failure\n'
        row = {'repo_name': 'skypilot', 'logs': [{'step_name': '5_Running yapf.txt',
                                                'log': '@@ -1 +1 @@\n-recorded\n+failure\n'}]}
        self.assertFalse(controls_valid(failed, reference, row))
        failed['jobs'][0]['steps'][0]['log'] = row['logs'][0]['log']
        self.assertTrue(controls_valid(failed, reference, row))

    def test_typecheck_control_requires_recorded_diagnostics(self):
        diagnostic = 'starlette/example.py:12: error: Wrong annotation  [misc]'
        failed = {'status': 'fail', 'jobs': [job(1, name='Run linting checks')]}
        failed['jobs'][0]['steps'][0]['log'] = diagnostic
        fixed = {'status': 'pass', 'jobs': [job()]}
        row = {'repo_name': 'starlette', 'logs': [{'step_name': '5_Run linting checks.txt', 'log': diagnostic}]}
        self.assertTrue(controls_valid(failed, fixed, row))
        failed['jobs'][0]['steps'][0]['log'] = 'Unrelated lint failure'
        self.assertFalse(controls_valid(failed, fixed, row))

    def test_traceback_control_requires_matching_locations_per_python(self):
        diagnostic = 'tests/test_routes.py:12: AssertionError'
        failed = {'status': 'fail', 'jobs': [job(1)]}
        failed['jobs'][0].update(matrix={'python-version': '3.9'})
        failed['jobs'][0]['steps'][0]['log'] = diagnostic
        fixed = {'status': 'pass', 'jobs': [job()]}
        row = {'repo_name': 'starlette', 'logs': [{'step_name': 'Python 3.9/7_Run tests.txt', 'log': diagnostic}]}
        self.assertTrue(controls_valid(failed, fixed, row))
        failed['jobs'][0]['matrix']['python-version'] = '3.10'
        self.assertFalse(controls_valid(failed, fixed, row))

    def test_unsupported_cases_share_the_same_policy_denominator(self):
        p = {'run': 'r', 'model': 'm', 'id': 1, 'seed': 0, 'arm': 'plain', 'edits': None}
        a = local_ci.prediction_entry(p, 'recover-markers') | {'status': 'pass'}
        b = local_ci.prediction_entry({**p, 'id': 2}, 'recover-markers') | {'status': 'unsupported'}
        summary = summarize([a, b])['per_seed'][0]
        self.assertEqual(summary['selected'], 2)
        self.assertIsNone(summary['local_ci_pass_at_1'])


if __name__ == '__main__':
    unittest.main()
