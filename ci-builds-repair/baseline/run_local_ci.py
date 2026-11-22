"""Replay complete audited workflows against saved model answers in Docker.

No generation, credentials, pushes, or GitHub Actions. Unsupported workflows
and failed environment controls remain unresolved, not model failures/successes.
"""

import argparse
import hashlib
import io
import json
import posixpath
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from local_ci.workflows import Unsupported, compile_workflow
from verify_completed_run import BENCHMARK, apply_edits, diff_blocks, digest, parse_response, prediction_blocks, save_json

HERE = Path(__file__).resolve().parent
DEFAULT_RUNS = ['rag-local/2d9349bb16247814246b', 'qwen-ci-dev/ac114173bcf206333b0a',
                'qwen-lean-rag/bf64963b600ed42de523']
SERVICE = 'ghcr.io/waleko/pypi-wayback@sha256:9f8e60cb4dcc03e5e27b37bacc809492e755ad39b95641aa9d90db3127cd36fd'


def prediction_files(files, blocks, prediction, composition):
    if composition in ('preserve-boundaries', 'recover-markers'):
        from repair_composition import compose_files
        return compose_files(files, blocks, prediction['edits'])
    if composition != 'original':
        raise ValueError('Unknown composition policy')
    return apply_edits(files, blocks, prediction['edits']), {}


def prediction_entry(prediction, composition):
    entry = {k: v for k, v in prediction.items() if k != 'edits'}
    if composition != 'original':
        entry['source_arm'] = entry['arm']
        entry['arm'] += ('+markers+boundaries' if composition == 'recover-markers' else '+boundaries')
    return entry


def evaluation_source_runs(folder, current_runs):
    """Resolve the source evaluation's own inputs before cross-model reuse."""
    declared = read(folder / 'manifest.json')['runs']
    supplied = {run.name: run for run in current_runs}
    result = []
    for identifier, expected in declared.items():
        if not re.fullmatch(r'[0-9a-f]{20}', identifier):
            raise ValueError('Unsafe source run identity')
        candidates = ([supplied[identifier]] if identifier in supplied else
                      [p.parent for p in (BENCHMARK / 'out').glob(f'*/{identifier}/run_metadata.json')])
        matches = [p for p in candidates if digest(read(p / 'run_metadata.json')) == expected]
        if len(matches) != 1:
            raise ValueError('Cannot uniquely locate source evaluation prediction run: ' + identifier)
        result.append(matches[0])
    return result


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path):
    record = read(path)
    expected = record.pop('payload_sha256')
    if digest(record) != expected:
        raise ValueError('Prediction checksum mismatch: ' + str(path))
    return record, expected


def validate_reuse(audited, dataset_sha256):
    """Unsupported cases do not invalidate independently audited executions.

    This only authorizes execution reuse. It never turns the full-dataset CI
    score into a complete score or removes an unsupported case's denominator.
    The caller must obtain this report from audit_run, not a saved summary.
    """
    if audited['manifest']['dataset_sha256'] != dataset_sha256:
        raise ValueError('Need an audited evaluation of the same dataset')
    if not audited['controls'] or not all(c['controls_valid'] for c in audited['controls']):
        raise ValueError('Cannot reuse unresolved failed/reference controls')
    allowed = {'pass', 'fail', 'invalid_prediction', 'unsupported'}
    if any(p['status'] not in allowed for p in audited['predictions']):
        raise ValueError('Cannot reuse unresolved workflow executions')


def saved_predictions(run, row, composition='original'):
    meta = read(run / 'run_metadata.json')
    if digest(meta)[:20] != run.name:
        raise ValueError('Run manifest identity mismatch')
    if row['id'] not in meta['ids']:
        return []
    if 'failing_shas' in meta and meta['failing_shas'][str(row['id'])] != row['sha_fail']:
        raise ValueError('Saved source revision mismatch')
    gold = diff_blocks(row['diff'])
    blocks = prediction_blocks(gold)
    result = []
    arms = meta.get('methods', meta.get('arms'))
    for seed in meta['seeds']:
        for arm in arms:
            mistral = 'methods' in meta
            path = (run / 'examples' / str(row['id']) / f'seed_{seed}' / arm / 'prediction.json'
                    if mistral else run / 'examples' / str(row['id']) / str(seed) / f'{arm}.json')
            answer, checksum = checked(path)
            identity = answer['identity']
            arm_key = 'method' if mistral else 'arm'
            if any(identity[k] != v for k, v in {'run': run.name, 'id': row['id'], 'seed': seed, arm_key: arm}.items()):
                raise ValueError('Prediction identity mismatch')
            if 'sha_fail' in identity and identity['sha_fail'] != row['sha_fail']:
                raise ValueError('Prediction failing revision mismatch')
            if mistral:
                prov = read(run / 'retrieval' / f"{row['id']}.json")['provenance']
                if prov['sha_fail'] != row['sha_fail'] or identity['snapshot'] != prov['files_sha256']:
                    raise ValueError('Historical source provenance mismatch')
                error = answer.get('error') if answer['status'] != 'applied' else None
                if answer['status'] not in ('applied', 'unapplied'):
                    raise ValueError('Unexpected prediction status')
            else:
                error = answer.get('skip')
            edits = None
            if not error:
                try:
                    if answer['finish_reason'] != 'stop':
                        raise ValueError('output_token_limit')
                    if arm == 'raw_plain':
                        from run_raw_edit_smoke import parse_answer
                        edits = parse_answer(answer['response'], blocks)
                    else:
                        edits = parse_response(answer['response'], blocks)
                except ValueError as exc:
                    error = str(exc)
            if mistral and (edits if edits is not None else [b['old'] for b in blocks]) != answer['replacements']:
                raise ValueError('Saved replacements differ from parsed answer')
            extra = {}
            if (composition == 'recover-markers' and error and error.startswith('Expected one replacement')
                    and answer['finish_reason'] == 'stop' and arm != 'raw_plain'):
                from repair_composition import recover_markers
                try:
                    recovered = recover_markers(answer['response'], blocks)
                except ValueError:
                    pass
                else:
                    extra = {'original_error': error, 'markers_recovered': True}
                    edits, error = recovered, None
            result.append({'run': run.name, 'model': meta.get('model_id', meta.get('model')),
                           'id': row['id'], 'seed': seed, 'arm': arm,
                           'prediction_sha256': checksum, 'error': error, 'edits': edits, **extra})
    return result


def source_repo(row):
    from git import Repo
    for key in ('repo_owner', 'repo_name'):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', row[key]):
            raise ValueError('Unsafe repository name')
    if not re.fullmatch(r'[0-9a-f]{40}', row['sha_fail']):
        raise ValueError('Unsafe source revision')
    return Repo(BENCHMARK / '.venv/rag-repos' / f"{row['repo_owner']}__{row['repo_name']}.git")


def source_archive(row):
    with source_repo(row) as repo:
        raw = repo.git.archive(row['sha_fail'], stdout_as_string=False)
    # Reject escaping links and special files before handing archives to Docker.
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or name.parts[0] == '.git':
                raise ValueError('Unsafe archive path')
            if member.issym() or member.islnk():
                target = PurePosixPath(member.linkname)
                resolved = posixpath.normpath(str(name.parent / target) if member.issym() else str(target))
                if target.is_absolute() or resolved == '..' or resolved.startswith('../'):
                    raise ValueError('Unsafe archive link')
            elif not (member.isfile() or member.isdir()):
                raise ValueError('Special archive member')
    return raw


def target_files(row):
    with source_repo(row) as repo:
        tree = repo.commit(row['sha_fail']).tree
        files = {}
        for block in diff_blocks(row['diff']):
            path = block['path']
            try:
                blob = tree / path
            except KeyError:
                if block['count']:
                    raise
                continue
            if blob.mode != 0o100644 and blob.mode != 0o100755:
                raise ValueError('Target is not a regular file')
            # Reject writes through symbolic directory entries as well.
            parent = PurePosixPath(path).parent
            if str(parent) != '.' and (tree / str(parent)).type != 'tree':
                raise ValueError('Target parent is not a directory')
            files[path] = blob.data_stream.read().decode('utf-8')
    return files


def docker(args, timeout=90, **kwargs):
    result = subprocess.run(['docker', *args], capture_output=True,
                            timeout=timeout, **kwargs)
    if result.returncode:
        details = (result.stdout + result.stderr).decode('utf-8', 'replace')
        raise RuntimeError('Docker ' + args[0] + ' failed: ' + details[-6000:])
    return result


class Engine:
    def __init__(self, platform):
        self.platform = platform
        self.network = 'lca-local-' + uuid.uuid4().hex[:12]
        self.service = self.network + '-index'
        self.images = {}
        self.prepared = {}

    def __enter__(self):
        docker(['info'])
        docker(['network', 'create', self.network])
        try:
            docker(['run', '-d', '--name', self.service, '--network', self.network,
                    '--network-alias', 'package-index', '--platform', self.platform,
                    '--memory', '768m', '--pids-limit', '128', '--cap-drop', 'ALL',
                    '--security-opt', 'no-new-privileges', SERVICE], timeout=180)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        subprocess.run(['docker', 'rm', '-f', self.service], capture_output=True)
        subprocess.run(['docker', 'network', 'rm', self.network], capture_output=True)

    def image(self, version):
        if version not in self.images:
            # Full official Python images include Git and build tools, avoiding
            # mutable OS-package installation during environment preparation.
            tag = f'python:{version}-' + ('bullseye' if version == '3.8' else 'bookworm')
            docker(['pull', '--platform', self.platform, tag], timeout=300)
            info = json.loads(docker(['image', 'inspect', tag]).stdout)[0]
            self.images[version] = {'tag': tag, 'id': info['Id'], 'base_digest': info['RepoDigests'][0],
                                    'architecture': info['Architecture'], 'os': info['Os']}
        return self.images[version]

    def execute(self, archive, patched, plan, folder, step_timeout):
        name = self.network + '-' + uuid.uuid4().hex[:10]
        cache_key = (hashlib.sha256(archive).hexdigest(), digest(plan))
        image = self.prepared.get(cache_key) or self.image(plan['python'])
        with tempfile.TemporaryDirectory(prefix='lca-ci-') as work:
            work = Path(work)
            (work / 'source.tar').write_bytes(archive)
            (work / 'worker.py').write_bytes((HERE / 'local_ci/worker.py').read_bytes())
            save_json(work / 'plan.json', {**plan, 'step_timeout': step_timeout})
            # Overlay contains only patched regular target files; no local checkout.
            with tarfile.open(work / 'overlay.tar', 'w') as tar:
                for path, text in patched.items():
                    item = tarfile.TarInfo(path)
                    content = text.encode('utf-8')
                    item.size, item.mode = len(content), 0o644
                    tar.addfile(item, io.BytesIO(content))
            started = time.monotonic()
            try:
                docker(['create', '--name', name, '--network', self.network,
                        '--platform', self.platform, '--memory', '3g', '--cpus', '3',
                        '--pids-limit', '384', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                        '--mount', f'type=bind,source={work},target=/runner,readonly',
                        '--workdir', '/workspace', image['id'], 'sh', '-c',
                        'cd / && rm -rf /workspace && mkdir /workspace && '
                        'tar -xf /runner/source.tar -C /workspace && '
                        'tar -xf /runner/overlay.tar -C /workspace && '
                        'cd /workspace && git init -q && git add . && '
                        'python /runner/worker.py'])
                docker(['start', name])
                code = docker(['wait', name], timeout=step_timeout * len(plan['steps']) + 90).stdout.strip()
                logs = docker(['logs', name]).stdout.decode('utf-8', 'replace')
                if code != b'0':
                    raise RuntimeError('Container worker failed: ' + logs[-2000:])
                result = json.loads(logs)
                result.update(image=image, seconds=time.monotonic() - started)
                save_json(folder / 'execution.json', result)
                # Cache only environments produced by the unmodified control.
                # Every variant gets a fresh source tree and re-runs every setup
                # command; editable installs still resolve the current sources.
                if folder.parent.name == 'failed' and result['steps'] and result['steps'][0]['exit_code'] == 0:
                    prepared_id = docker(['commit', name], timeout=90).stdout.decode().strip()
                    self.prepared[cache_key] = {**image, 'id': prepared_id,
                                                'prepared_from_execution_sha256': digest(result)}
                return result
            finally:
                subprocess.run(['docker', 'rm', '-f', name], capture_output=True)


def workflow_result(jobs, expected):
    if len(jobs) != expected:
        return 'infrastructure_error'
    for job in jobs:
        if not job['steps']:
            return 'infrastructure_error'
        for step in job['steps']:
            if step['exit_code'] and (step['phase'] == 'setup' or step['exit_code'] == 124):
                return 'infrastructure_error'
    return 'pass' if all(j['passed'] for j in jobs) else 'fail'


def controls_valid(failed, reference, row):
    if failed['status'] != 'fail' or reference['status'] != 'pass':
        return False
    # Require the failed workflow to reach a step named in the recorded log.
    failures = [s for j in failed['jobs'] for s in j['steps'] if s['exit_code']]
    matched = [s for s in failures if s['phase'] == 'check' and any(
        s['name'] in log['step_name'] for log in row['logs'])]
    if not matched:
        return False
    # For formatting cases compare the actual diff hunks, not just step names.
    if row.get('repo_name') == 'skypilot':
        def hunks(text):
            result, active = [], False
            for line in text.splitlines():
                if line.startswith(('--- ', '+++ ')):
                    active = False
                elif line.startswith('@@ '):
                    active = True
                    result.append(line)
                elif active and line[:1] in (' ', '+', '-'):
                    result.append(line)
                else:
                    active = False
            return result
        expected = hunks('\n'.join(l['log'] for l in row['logs']))
        observed = hunks('\n'.join(s['log'] for s in matched))
        return bool(expected) and expected == observed
    if row.get('repo_name') == 'starlette':
        recorded = '\n'.join(l['log'] for l in row['logs'])
        observed = '\n'.join(s['log'] for s in matched)
        expected = re.findall(r'^E\s+(TypeError:[^\n]+)', recorded, re.M)
        if not expected:
            expected = re.findall(r'^[\w./-]+:\d+: error: [^\n]+', recorded, re.M)
        if expected:
            return all(e in observed for e in expected)
        # Some pytest logs omit FAILED node summaries. Match the complete set
        # of project-relative traceback locations for each recorded Python job.
        location = r'^(?:tests|starlette)/[^\s:]+\.py:\d+: [A-Za-z][\w.]*[^\n]*$'
        checked_versions = 0
        for log in row['logs']:
            version = re.match(r'Python (\d+\.\d+)/', log['step_name'])
            wanted = set(re.findall(location, log['log'], re.M))
            if not version or not wanted:
                return False
            actual = '\n'.join(s['log'] for j in failed['jobs']
                               if j.get('matrix', {}).get('python-version') == version[1]
                               for s in j['steps'] if s['exit_code'])
            if set(re.findall(location, actual, re.M)) != wanted:
                return False
            checked_versions += 1
        return bool(checked_versions)
    return True


def summarize(records):
    identities = [(r['run'], r['arm'], r['seed'], r['id']) for r in records]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate prediction evaluation')
    groups = {}
    for r in records:
        key = (r['run'], r['model'], r['arm'], r['seed'])
        groups.setdefault(key, []).append(r)
    per_seed = []
    for (run, model, arm, seed), values in groups.items():
        counts = Counter(v['status'] for v in values)
        n, passed = len(values), counts['pass']
        resolved = counts['pass'] + counts['fail'] + counts['invalid_prediction']
        per_seed.append({'run': run, 'model': model, 'arm': arm, 'seed': seed, 'selected': n,
                         'resolved': resolved, 'passed': passed, 'statuses': dict(counts),
                         'complete': resolved == n, 'local_ci_pass_at_1': passed / n if resolved == n else None,
                         'pass_rate_lower_bound': passed / n,
                         'pass_rate_upper_bound': (passed + n - resolved) / n})
    across = []
    for run, model, arm in dict.fromkeys((x['run'], x['model'], x['arm']) for x in per_seed):
        rows = [x for x in per_seed if (x['run'], x['model'], x['arm']) == (run, model, arm)]
        rates = [x['local_ci_pass_at_1'] for x in rows]
        complete = all(x is not None for x in rates)
        across.append({'run': run, 'model': model, 'arm': arm,
                       'mean_local_ci_pass_at_1': statistics.mean(rates) if complete else None,
                       'std_across_seeds': statistics.pstdev(rates) if complete else None})
    return {'scope': 'Complete audited local workflow replay on selected cases; not original hosted CI or paper replication',
            'complete': bool(per_seed) and all(x['complete'] for x in per_seed),
            'per_seed': per_seed, 'across_seeds': across, 'predictions': records}


def audit_run(folder, rows, runs):
    """Reconstruct patches and independently check complete execution evidence."""
    manifest = read(folder / 'manifest.json')
    if digest(manifest)[:20] != folder.name:
        raise ValueError('Evaluation manifest identity mismatch')
    for name, expected in manifest['source_hashes'].items():
        if sha(folder / 'sources' / name) != expected:
            raise ValueError('Archived evaluator source checksum mismatch')
    if {r.name: digest(read(r / 'run_metadata.json')) for r in runs} != manifest['runs']:
        raise ValueError('Prediction manifests changed')
    original = read(folder / 'summary.json')
    composition = manifest.get('composition', 'original')
    records, controls = [], []
    for i in manifest['ids']:
        row = rows[i]
        if row['sha_fail'] != manifest['failing_shas'][str(i)] or hashlib.sha256(row['workflow'].encode()).hexdigest() != manifest['workflows'][str(i)]:
            raise ValueError('Workflow or failed source changed')
        predictions = [p for run in runs for p in saved_predictions(run, row, composition)]
        try:
            plans = compile_workflow(row)
        except Unsupported as exc:
            records.extend(prediction_entry(p, composition) | {'status': 'unsupported', 'reason': str(exc)} for p in predictions)
            continue
        files = target_files(row)
        gold = diff_blocks(row['diff'])
        blocks = prediction_blocks(gold)

        def execution(label, patched):
            value = read(folder / str(i) / label / 'result.json')
            if value['patch_sha256'] != digest(patched) or value['label'] != label:
                raise ValueError('Executed patch identity mismatch')
            if len(value['jobs']) != len(plans):
                raise ValueError('Missing workflow matrix configurations')
            for j, (job, plan) in enumerate(zip(value['jobs'], plans)):
                observed = read(folder / str(i) / label / str(j) / 'execution.json')
                if any(job.get(k) != v for k, v in observed.items()):
                    raise ValueError('Job evidence differs from workflow record')
                if 'error' in job:
                    continue
                if job['matrix'] != plan['matrix'] or job['job'] != plan['job']:
                    raise ValueError('Executed matrix identity mismatch')
                if not job['python'].startswith(plan['python'] + '.') or job['image']['architecture'] != 'amd64' or job['image']['os'] != 'linux':
                    raise ValueError('Wrong execution environment')
                if job['expected_steps'] != len(plan['steps']) or job['completed_steps'] != len(job['steps']):
                    raise ValueError('Step count mismatch')
                if not job['steps'] or len(job['steps']) > len(plan['steps']):
                    raise ValueError('Invalid execution sequence')
                for step, expected in zip(job['steps'], plan['steps']):
                    if (step['name'], step['phase'], step['command']) != (expected['name'], expected['phase'], expected['run']):
                        raise ValueError('Workflow command changed or skipped')
                if any(s['exit_code'] for s in job['steps'][:-1]):
                    raise ValueError('Execution continued after a failing step')
                passed = len(job['steps']) == len(plan['steps']) and all(s['exit_code'] == 0 for s in job['steps'])
                if job['passed'] != passed or (not passed and not job['steps'][-1]['exit_code']):
                    raise ValueError('Unsupported early stop or wrong job verdict')
            status = ('infrastructure_error' if any('error' in j for j in value['jobs'])
                      else workflow_result(value['jobs'], len(plans)))
            if status != value['status']:
                raise ValueError('Workflow status mismatch')
            return value

        failed = execution('failed', files)
        reference = execution('reference', apply_edits(files, blocks, [b['new'] for b in gold]))
        valid = controls_valid(failed, reference, row)
        controls.append({'id': i, 'controls_valid': valid, 'failed_status': failed['status'], 'reference_status': reference['status'],
                         'expected_jobs': len(plans), 'failed_failure_steps': [s['name'] for j in failed['jobs'] for s in j['steps'] if s['exit_code']]})
        for p in predictions:
            entry = prediction_entry(p, composition)
            previous, = [r for r in original['predictions'] if all(r[k] == entry[k] for k in ('id', 'run', 'arm', 'seed'))]
            if previous['prediction_sha256'] != p['prediction_sha256']:
                raise ValueError('Evaluated prediction checksum changed')
            if not valid:
                entry['status'] = 'controls_unresolved'
            elif p['error']:
                entry['status'] = 'invalid_prediction'
            else:
                patched, notes = prediction_files(files, blocks, p, composition)
                if composition != 'original':
                    entry['composition_notes'] = notes
                result = execution(previous['execution'], patched)
                entry.update(status=result['status'], patch_sha256=result['patch_sha256'], execution=result['label'])
            records.append(entry)
    report = {**summarize(records), 'controls': controls, 'manifest': manifest,
              'audit': {'source_sha256': sha(Path(__file__)), 'workflows_sha256': sha(HERE / 'local_ci/workflows.py'),
                        'composition_source_sha256': sha(HERE / 'repair_composition.py') if composition != 'original' else None,
                        'execution_evidence_sha256': {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('execution.json'))},
                        'generation': False, 'execution': False, 'strong_failure_fingerprints': True}}
    save_json(folder / 'audited_summary.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--ids', type=int, nargs='+', default=[33, 34, 36, 37])
    selection.add_argument('--all-cases', action='store_true')
    parser.add_argument('--runs', nargs='+', default=DEFAULT_RUNS)
    parser.add_argument('--dataset-file', type=Path)
    parser.add_argument('--output-dir', type=Path, default=BENCHMARK / 'out/local-ci')
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--audit-run', type=Path, help='Verify saved workflow executions without Docker or model calls')
    parser.add_argument('--step-timeout', type=int, default=600)
    parser.add_argument('--platform', choices=['linux/amd64'], default='linux/amd64')
    parser.add_argument('--composition', choices=['original', 'preserve-boundaries', 'recover-markers'], default='original')
    parser.add_argument('--reuse-evaluation', type=Path, help='Reuse audited executions only for byte-identical case source trees')
    parser.add_argument('--additional-reuse-evaluation', type=Path, action='append', default=[])
    args = parser.parse_args(argv)
    from datasets import Dataset
    if args.dataset_file is None:
        args.dataset_file, = (BENCHMARK / '.venv/dataset-cache').rglob('*test.arrow')
    rows = {r['id']: r for r in Dataset.from_file(str(args.dataset_file))}
    if args.all_cases:
        args.ids = list(rows)
    if len(set(args.ids)) != len(args.ids) or set(args.ids) - rows.keys() or args.step_timeout < 1:
        parser.error('Require distinct existing IDs and a positive timeout')
    inventory = []
    for row in rows.values():
        try:
            plans = compile_workflow(row)
            inventory.append({'id': row['id'], 'repository': row['repo_name'], 'supported': True,
                              'jobs': len(plans), 'python_versions': [p['python'] for p in plans]})
        except Unsupported as exc:
            inventory.append({'id': row['id'], 'repository': row['repo_name'], 'supported': False, 'reason': str(exc)})
    save_json(args.output_dir / 'workflow_inventory.json', inventory)
    if args.inventory_only:
        print(json.dumps(inventory, indent=2))
        return
    runs = [(Path(r) if Path(r).is_absolute() else BENCHMARK / 'out' / r).resolve() for r in args.runs]
    for run in runs:
        if read(run / 'run_metadata.json')['dataset_sha256'] != sha(args.dataset_file):
            raise ValueError('Dataset does not match saved run')
    if args.audit_run:
        if read(args.audit_run / 'manifest.json')['dataset_sha256'] != sha(args.dataset_file):
            raise ValueError('Evaluation dataset mismatch')
        report = audit_run(args.audit_run.resolve(), rows, runs)
        print(json.dumps({k: report[k] for k in ('complete', 'controls', 'across_seeds')}, indent=2))
        return
    sources = [Path(__file__), HERE / 'local_ci/worker.py', HERE / 'local_ci/workflows.py',
               HERE.parent / 'rag/evaluation.py', BENCHMARK / 'paper_mistral_baseline.py', HERE / 'run_raw_edit_smoke.py']
    if args.composition != 'original':
        sources.append(HERE / 'repair_composition.py')
    source_hashes = {p.name: sha(p) for p in sources}
    manifest = {'ids': args.ids, 'dataset_sha256': sha(args.dataset_file), 'platform': args.platform,
                'runs': {r.name: digest(read(r / 'run_metadata.json')) for r in runs},
                'source_hashes': source_hashes, 'step_timeout': args.step_timeout, 'package_service': SERVICE,
                'failing_shas': {str(i): rows[i]['sha_fail'] for i in args.ids},
                'workflows': {str(i): hashlib.sha256(rows[i]['workflow'].encode()).hexdigest() for i in args.ids},
                'composition': args.composition}
    reuse_sources = []
    reuse_paths = ([args.reuse_evaluation] if args.reuse_evaluation else []) + args.additional_reuse_evaluation
    for reuse_path in reuse_paths:
        reuse = reuse_path.resolve()
        audited = audit_run(reuse, rows, evaluation_source_runs(reuse, runs))
        validate_reuse(audited, manifest['dataset_sha256'])
        if audited['manifest']['source_hashes']['worker.py'] != source_hashes['worker.py']:
            raise ValueError('Cannot reuse a changed execution worker')
        if audited['manifest']['source_hashes']['workflows.py'] != source_hashes['workflows.py']:
            raise ValueError('Cannot reuse changed workflow semantics')
        if audited['manifest']['package_service'] != SERVICE or audited['manifest']['platform'] != args.platform:
            raise ValueError('Cannot reuse a different execution environment')
        reuse_sources.append((reuse, audited))
    if reuse_sources:
        manifest['reused_evaluations'] = [{'run': p.name, 'audited_summary_sha256': sha(p / 'audited_summary.json')}
                                          for p, _ in reuse_sources]
    folder = args.output_dir / digest(manifest)[:20]
    if folder.exists():
        raise SystemExit('Evaluation directory already exists. Use --audit-run to verify it, or a new --output-dir for another execution.')
    save_json(folder / 'manifest.json', manifest)
    (folder / 'sources').mkdir()
    for source in sources:
        (folder / 'sources' / source.name).write_bytes(source.read_bytes())
    print('Local workflow evaluation: ' + str(folder), flush=True)
    records, cases = [], []
    with Engine(args.platform) as engine:
        for i in args.ids:
            row = rows[i]
            predictions = [p for run in runs for p in saved_predictions(run, row, args.composition)]
            try:
                plans = compile_workflow(row)
            except Unsupported as exc:
                records.extend(prediction_entry(p, args.composition) | {'status': 'unsupported', 'reason': str(exc)} for p in predictions)
                continue
            raw = source_archive(row)
            files = target_files(row)
            gold = diff_blocks(row['diff'])
            blocks = prediction_blocks(gold)
            reference = apply_edits(files, blocks, [b['new'] for b in gold])
            cache = {}
            reusable = {}
            reuse, audited = next(((p, a) for p, a in reuse_sources if str(i) in a['manifest']['failing_shas']), (None, None))
            if reuse and str(i) in audited['manifest']['failing_shas']:
                if audited['manifest']['failing_shas'][str(i)] != row['sha_fail'] or audited['manifest']['workflows'][str(i)] != manifest['workflows'][str(i)]:
                    raise ValueError('Cannot reuse a changed case')
                for path in (reuse / str(i)).glob('*/result.json'):
                    value = read(path)
                    reusable[value['patch_sha256']] = path
                # Reuse only dependency environments from the unmodified control.
                for j, plan in enumerate(plans):
                    failed_job = read(reuse / str(i) / 'failed' / str(j) / 'execution.json')
                    image = read(reuse / str(i) / 'reference' / str(j) / 'execution.json')['image']
                    if image.get('prepared_from_execution_sha256') == digest(failed_job):
                        try:
                            docker(['image', 'inspect', image['id']])
                        except RuntimeError:
                            continue
                        engine.prepared[hashlib.sha256(raw).hexdigest(), digest(plan)] = image

            def execute(label, patched):
                key = digest(patched)
                if key in cache:
                    return {**cache[key], 'reused_execution': cache[key]['label']}
                destination = folder / str(i) / label
                if key in reusable:
                    source = reusable[key]
                    value = read(source)
                    shutil.copytree(source.parent, destination)
                    value.update(label=label, reused_from={'run': reuse.name,
                                  'path': str(source.relative_to(reuse)), 'sha256': sha(source)})
                    save_json(destination / 'result.json', value)
                    cache[key] = value
                    print(f'  {i}/{label}: {value["status"]} (identical audited source tree)', flush=True)
                    return value
                def run_job(pair):
                    j, plan = pair
                    try:
                        result = engine.execute(raw, patched, plan, destination / str(j), args.step_timeout)
                        return {**result, 'job': plan['job'], 'matrix': plan['matrix'], 'deviations': plan['deviations']}
                    except (subprocess.SubprocessError, RuntimeError, ValueError) as exc:
                        result = {'error': str(exc), 'steps': [], 'passed': False}
                        save_json(destination / str(j) / 'execution.json', result)
                        return result
                with ThreadPoolExecutor(max_workers=2) as pool:
                    jobs = list(pool.map(run_job, enumerate(plans)))
                status = 'infrastructure_error' if any('error' in j for j in jobs) else workflow_result(jobs, len(plans))
                value = {'label': label, 'status': status, 'patch_sha256': key, 'jobs': jobs}
                cache[key] = value
                save_json(destination / 'result.json', value)
                print(f'  {i}/{label}: {status}', flush=True)
                return value

            failed = execute('failed', files)
            fixed = execute('reference', reference)
            valid = controls_valid(failed, fixed, row)
            cases.append({'id': i, 'controls_valid': valid, 'failed_status': failed['status'], 'reference_status': fixed['status'],
                          'expected_jobs': len(plans), 'failed_failure_steps': [s['name'] for j in failed['jobs'] for s in j['steps'] if s['exit_code']]})
            print(f'  {i}: controls_valid={valid}', flush=True)
            for p in predictions:
                entry = prediction_entry(p, args.composition)
                if not valid:
                    entry['status'] = 'controls_unresolved'
                elif p['error']:
                    entry['status'] = 'invalid_prediction'
                else:
                    patched, notes = prediction_files(files, blocks, p, args.composition)
                    if args.composition != 'original':
                        entry['composition_notes'] = notes
                    result = execute(f"{p['run']}-{p['arm']}-{p['seed']}", patched)
                    entry.update(status=result['status'], patch_sha256=result['patch_sha256'],
                                 execution=result.get('reused_execution', result['label']))
                records.append(entry)
            summary = {**summarize(records), 'controls': cases, 'manifest': manifest}
            save_json(folder / 'summary.json', summary)
    summary = {**summarize(records), 'controls': cases, 'manifest': manifest}
    save_json(folder / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('predictions', 'manifest', 'per_seed')}, indent=2))


if __name__ == '__main__':
    main()
