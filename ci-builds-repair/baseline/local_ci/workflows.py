"""Conservative compiler for audited, complete workflow families.

Unknown syntax/actions are unsupported, never implicitly successful. This is
not a general GitHub Actions emulator. Original run commands are preserved.
"""

import itertools
import re

import yaml


class Unsupported(ValueError):
    pass


def interpolate(value, matrix):
    def replace(match):
        key = match[1]
        if key not in matrix or not isinstance(matrix[key], (str, int, float)):
            raise Unsupported('Unknown or non-scalar matrix expression: ' + key)
        return str(matrix[key])
    value = re.sub(r'\$\{\{\s*matrix\.([\w-]+)\s*\}\}', replace, str(value))
    if '${{' in value:
        raise Unsupported('Unsupported expression: ' + value)
    return value


def compile_workflow(row):
    workflow = yaml.safe_load(row['workflow'])
    if set(workflow) - {'name', 'on', True, 'env', 'jobs', 'concurrency', 'permissions'}:
        raise Unsupported('Unsupported workflow semantics')
    # Execution profiles are explicitly limited to reviewed workflow families.
    family = row['repo_name']
    if family not in ('skypilot', 'starlette'):
        raise Unsupported('Workflow family has not been audited for local replay')
    jobs = workflow.get('jobs', {})
    if (family == 'skypilot' and set(jobs) != {'format'}) or (family == 'starlette' and set(jobs) != {'tests'}):
        raise Unsupported('Unreviewed job set')
    plans = []
    for job_id, job in jobs.items():
        if set(job) - {'name', 'runs-on', 'strategy', 'services', 'steps', 'env', 'timeout-minutes'}:
            raise Unsupported('Unsupported job semantics')
        if job['runs-on'] != 'ubuntu-latest':
            raise Unsupported('Unsupported operating system')
        strategy = job.get('strategy', {})
        if set(strategy) - {'matrix', 'fail-fast', 'max-parallel'}:
            raise Unsupported('Unsupported strategy')
        axes = strategy.get('matrix', {})
        if any(not isinstance(v, list) for v in axes.values()) or {'include', 'exclude'} & axes.keys():
            raise Unsupported('Unsupported matrix')
        services = job.get('services', {})
        if services != {'pypi_wayback': {'image': 'ghcr.io/waleko/pypi-wayback', 'ports': ['8629:8080']}}:
            raise Unsupported('Unexpected historical package service')
        for values in itertools.product(*axes.values()):
            matrix = dict(zip(axes, values))
            env = {**workflow.get('env', {}), **job.get('env', {})}
            if set(env) != {'PIP_INDEX_URL', 'UV_INDEX_URL'}:
                raise Unsupported('Unreviewed workflow environment')
            dates = [re.fullmatch(r'http://localhost:8629/(\d{4}-\d\d-\d\d)', str(v)) for v in env.values()]
            if not all(dates) or len({m[1] for m in dates}) != 1:
                raise Unsupported('Unexpected package cutoff')
            env = {k: str(v).replace('localhost:8629', 'package-index:8080') for k, v in env.items()}
            env['PIP_TRUSTED_HOST'] = 'package-index'
            python, steps, setup_done, checkout = None, [], False, False
            for source in job['steps']:
                if set(source) - {'name', 'uses', 'with', 'run', 'env', 'shell'}:
                    raise Unsupported('Unsupported step semantics')
                if 'uses' in source:
                    action = source['uses']
                    if re.fullmatch(r'actions/checkout@v[34]', action) and not checkout and not steps:
                        if source.get('with'):
                            raise Unsupported('Checkout options')
                        checkout = True
                    elif re.fullmatch(r'actions/setup-python@v[45]', action) and checkout and not steps and python is None:
                        opts = source.get('with', {})
                        if set(opts) - {'python-version', 'allow-prereleases'}:
                            raise Unsupported('Python setup options')
                        python = interpolate(opts.get('python-version', ''), matrix)
                        if python not in ('3.8', '3.9', '3.10', '3.11', '3.12'):
                            raise Unsupported('Unsupported Python version')
                    else:
                        raise Unsupported('Unreviewed action: ' + action)
                    continue
                if not checkout or python is None or source.get('with') or 'run' not in source:
                    raise Unsupported('Step order or unsupported run step')
                if source.get('shell', 'bash') != 'bash':
                    raise Unsupported('Unsupported shell')
                name = source.get('name', 'unnamed')
                if family == 'skypilot':
                    is_setup = name == 'Install dependencies'
                else:
                    is_setup = source['run'] == 'scripts/install'
                if is_setup and setup_done:
                    raise Unsupported('Setup after check')
                if not is_setup:
                    setup_done = True
                steps.append({'name': interpolate(name, matrix), 'run': interpolate(source['run'], matrix),
                              'phase': 'setup' if is_setup else 'check',
                              'env': {k: interpolate(v, matrix) for k, v in source.get('env', {}).items()}})
            expected = (['Running yapf', 'Running black', 'Running isort for black formatted files',
                         'Running isort for yapf formatted files'] if family == 'skypilot'
                        else ['Run linting checks', 'Build package & docs', 'Run tests', 'Enforce coverage'])
            if [s['name'] for s in steps if s['phase'] == 'check'] != expected or sum(s['phase'] == 'setup' for s in steps) != 1:
                raise Unsupported('Check sequence differs from reviewed workflow')
            plans.append({'job': job_id, 'matrix': matrix, 'python': python, 'env': env,
                          'steps': steps, 'package_cutoff': dates[0][1],
                          'deviations': ['Debian container replaces Ubuntu hosted runner',
                                         'Latest available patch release within the specified Python minor',
                                         'Jobs run sequentially; no GitHub event or hosted runner',
                                         'Historical package service resolved by image digest at evaluation time']})
    if not plans:
        raise Unsupported('Empty workflow')
    return plans
