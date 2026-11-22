"""Container-only worker. The host never executes benchmark project code."""

import json
import os
import platform
import subprocess
import sys
import time


def main():
    plan = json.load(open('/runner/plan.json'))
    env = dict(os.environ)
    env.update(plan['env'])
    env.update(GITHUB_ACTIONS='true', CI='true', GITHUB_WORKSPACE='/workspace',
               GITHUB_EVENT_NAME='push', GITHUB_STEP_SUMMARY='/tmp/step-summary',
               GITHUB_ENV='/tmp/github-env', GITHUB_PATH='/tmp/github-path',
               GITHUB_OUTPUT='/tmp/github-output')
    results = []
    for step in plan['steps']:
        started = time.monotonic()
        try:
            proc = subprocess.run(
                ['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c', step['run']],
                cwd='/workspace', env={**env, **step.get('env', {})},
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                errors='replace', timeout=plan['step_timeout'])
            code, output = proc.returncode, proc.stdout
        except subprocess.TimeoutExpired as exc:
            code, output = 124, str(exc.output or '')
        results.append({'name': step['name'], 'phase': step['phase'],
                        'command': step['run'], 'exit_code': code, 'log': output,
                        'seconds': time.monotonic() - started})
        if code:
            break
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'],
                            capture_output=True, text=True, timeout=30)
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'steps': results, 'packages': freeze.stdout.splitlines(),
              'completed_steps': len(results), 'expected_steps': len(plan['steps']),
              'passed': len(results) == len(plan['steps']) and all(s['exit_code'] == 0 for s in results)}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
