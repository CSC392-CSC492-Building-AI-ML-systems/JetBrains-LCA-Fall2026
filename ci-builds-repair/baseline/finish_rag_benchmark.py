"""Follow an active paired generation run, audit it, evaluate locally, export.

No GitHub submissions. Unsupported workflows retain their denominator and
produce an unavailable full-dataset CI score rather than an inflated score.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import tempfile

import run_local_ci as ci

ARMS = ('plain', 'lean_rag')
RESOLVED = {'pass', 'fail', 'invalid_prediction'}
ROOT = ci.HERE.parent.parent


def freeze_evaluation_code(run, meta, name='evaluation-code'):
    """Keep later audits independent of edits in the active checkout."""
    if Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('Evaluation code name must be a single directory name')
    destination = run / name
    if not destination.exists():
        with tempfile.TemporaryDirectory(prefix='.freezing-', dir=run) as temporary:
            root = Path(temporary) / 'source'
            for name in ('baseline', 'rag', 'ci-builds-repair-benchmark'):
                source = ROOT / 'ci-builds-repair' / name
                files = list(source.glob('*.py'))
                if name == 'baseline':
                    files.extend((source / 'local_ci').glob('*.py'))
                for path in files:
                    target = root / path.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
            for path in (ROOT / 'project_level_code_completion/rag').glob('*.py'):
                target = root / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
            benchmark = root / 'ci-builds-repair/ci-builds-repair-benchmark'
            benchmark.mkdir(parents=True, exist_ok=True)
            for name in ('.venv', 'out'):
                (benchmark / name).symlink_to(ci.BENCHMARK / name, target_is_directory=True)
            hashes = {str(p.relative_to(root)): ci.sha(p) for p in root.rglob('*.py')}
            ci.save_json(root / 'source_hashes.json', hashes)
            root.rename(destination)
    for relative, expected in ci.read(destination / 'source_hashes.json').items():
        if ci.sha(destination / relative) != expected:
            raise ValueError('Frozen evaluator code changed')
    for name, expected in meta['source_hashes'].items():
        matches = list(destination.rglob(name))
        if not any(ci.sha(path) == expected for path in matches):
            raise ValueError('Cannot freeze the recorded generation source: ' + name)
    return destination


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def supported_scores(predictions, supported_ids, seeds, arms=ARMS):
    output = []
    for arm in arms:
        for seed in seeds:
            group = [p for p in predictions if p.get('source_arm', p['arm']) == arm
                     and p['seed'] == seed and p['id'] in supported_ids]
            counts = Counter(p['status'] for p in group)
            resolved = sum(counts[s] for s in RESOLVED)
            n = len(supported_ids)
            complete = len(group) == n and resolved == n
            output.append({'arm': arm, 'seed': seed, 'selected': n, 'resolved': resolved,
                           'passed': counts['pass'], 'statuses': dict(counts), 'complete': complete,
                           'supported_case_ci_pass_at_1': counts['pass'] / n if complete and n else None})
    return output


def proxy_scores(metrics, ids, seeds, arms=ARMS):
    result = []
    for arm in arms:
        for seed in seeds:
            group = [m for m in metrics['cases'] if m['arm'] == arm and m['seed'] == seed]
            if len(group) != len(ids) or {m['id'] for m in group} != set(ids):
                raise ValueError('Missing or duplicate proxy scores')
            result.append({'arm': arm, 'seed': seed, 'evaluated': len(group),
                           'valid': sum(m['error'] is None for m in group),
                           'nonempty': sum(m['nonempty_patch'] for m in group),
                           'edit_f1': statistics.mean(m['edit_token_f1'] for m in group),
                           'exact_reference': sum(m['patch_exact_match'] for m in group)})
    return result


def publish(run, evaluation, evaluation_date):
    meta = ci.read(run / 'run_metadata.json')
    metrics = ci.read(run / 'metrics.json')
    audited = ci.read(evaluation / 'audited_summary.json')
    inventory = ci.read(evaluation.parent / 'workflow_inventory.json')
    supported_ids = [r['id'] for r in inventory if r['supported'] and r['id'] in meta['ids']]
    arms = tuple(meta['arms'])
    long_context = arms == ('plain', 'long_context')
    if arms not in (ARMS, ('plain', 'long_context')):
        raise ValueError('Unknown paired benchmark arms')
    subset = supported_scores(audited['predictions'], supported_ids, meta['seeds'], arms)
    proxies = proxy_scores(metrics, meta['ids'], meta['seeds'], arms)
    report = {'evaluation_date': evaluation_date, 'generation_run': run.name,
              'workflow_evaluation_run': evaluation.name, 'generation_complete': metrics['complete'],
              'full_dataset_ci_complete': audited['complete'], 'paper_replication': False,
              'scope': 'Full cached dataset paired generation; local CI coverage explicitly limited',
              'generation_manifest': meta, 'proxy_summary': {k: v for k, v in metrics.items() if k != 'cases'},
              'proxy_per_seed': proxies, 'proxy_cases': metrics['cases'],
              'ci_full_dataset_per_seed': audited['per_seed'], 'ci_predictions': audited['predictions'],
              'ci_controls': audited['controls'], 'ci_manifest': audited['manifest'], 'ci_audit': audited['audit'],
              'supported_ids': supported_ids, 'unsupported_ids': sorted(set(meta['ids']) - set(supported_ids)),
              'supported_case_ci_per_seed': subset,
              'limitations': ['Four-bit local model and reconstructed prompt',
                              'Current cached split differs from the original paper dataset',
                              'Unsupported or unresolved workflows are not successful or silently excluded',
                              'Edit F1 and syntax are offline proxies, not CI Pass@1']}
    stem = 'full_long_context' if long_context else 'full_rag'
    markdown = 'FULL_LONG_CONTEXT_BENCHMARK.md' if long_context else 'FULL_RAG_BENCHMARK.md'
    if long_context:
        report['baseline_reuse'] = {'run': meta['baseline_run'],
                                    'manifest_sha256': meta['baseline_manifest_sha256']}
        report['context_preflight'] = ci.read(run / 'preflight.json')
        report['limitations'].append('Oversized affected files use bounded excerpts; this is not full-repository context')
        previous_run = ci.BENCHMARK / 'out' / meta['baseline_run']
        previous_meta = ci.read(previous_run / 'run_metadata.json')
        if ci.digest(previous_meta) != meta['baseline_manifest_sha256']:
            raise ValueError('Comparison baseline manifest changed')
        previous_metrics = ci.read(previous_run / 'metrics.json')
        previous_evaluation = Path(ci.read(previous_run / 'benchmark_status.json')['evaluation'])
        previous_ci = ci.read(previous_evaluation / 'audited_summary.json')
        if previous_ci['manifest']['runs'].get(previous_run.name) != ci.digest(previous_meta):
            raise ValueError('Previous CI comparison uses different baseline predictions')
        prior_subset = supported_scores(previous_ci['predictions'], supported_ids, meta['seeds'])
        report['previous_lean_rag_comparison'] = {
            'generation_run': previous_run.name, 'workflow_evaluation_run': previous_evaluation.name,
            'same_cases_and_seeds': previous_meta['ids'] == meta['ids'] and previous_meta['seeds'] == meta['seeds'],
            'proxy_summary': {k: v for k, v in previous_metrics.items() if k != 'cases'},
            'supported_case_ci_per_seed': prior_subset}
    ci.save_json(ci.HERE / f'results/{stem}_metrics.json', report)
    title = 'Full paired long-context benchmark' if long_context else 'Full paired lean-RAG benchmark'
    strategy = ('Long context adds failed-snapshot affected files, direct Python imports, and root configuration. '
                'Oversized targets use bounded windows; omissions are recorded. '
                'The identical verified standard-baseline answers are reused with source checksums.'
                if long_context else 'RAG adds up to three BM25 snippets and 768 actual tokens from the failed snapshot.')
    lines = [f'# {title}', '',
             f'Evaluated on {evaluation_date}. Model: `{meta["model"]}`. Generation run `{run.name}`.', '',
             f'**{len(meta["ids"])} cases × {len(meta["seeds"])} seeds × 2 arms**; all raw answers audited.',
             f'Context: {meta["context"]} tokens; output cap: {meta["cap"]}; temperature: {meta["temperature"]}.',
             'Both arms use the same oracle blocks, decoding settings, and improved composition.',
             strategy, '',
             '| Arm | Valid responses | Mean edit F1 | Exact reference |',
             '| --- | --- | --- | --- |']
    for arm in arms:
        values = metrics['arms'][arm]
        lines.append(f'| {arm} | {values["valid_response"]}/{values["expected"]} | '
                     f'{values["mean_edit_f1"]:.6f} | {values["exact_reference"]} |')
    lines.extend(['', '**These edit scores are not CI Pass@1.**', '',
                  f'The local evaluator recognizes {len(supported_ids)} of {len(meta["ids"])} workflows. '
                  f'{len(report["unsupported_ids"])} remain explicitly unsupported.', '',
                  '| Arm | Seed | Supported cases resolved | Passing | Supported-case CI Pass@1 |',
                  '| --- | --- | --- | --- | --- |'])
    for item in subset:
        rate = item['supported_case_ci_pass_at_1']
        rendered = 'unavailable' if rate is None else f'{rate:.1%}'
        lines.append(f'| {item["arm"]} | {item["seed"]} | {item["resolved"]}/{item["selected"]} | '
                     f'{item["passed"]} | {rendered} |')
    if long_context and report['previous_lean_rag_comparison']['same_cases_and_seeds']:
        previous = report['previous_lean_rag_comparison']
        lines.extend(['', '## Comparison with lean RAG', '',
                      'Same model, cases, seeds, generation budgets, and composition. '
                      'The standard-baseline answers are shared, not generated again.', '',
                      '| Approach | Mean edit F1 | Supported-case CI Pass@1, mean across seeds |',
                      '| --- | --- | --- |'])
        for arm in (*arms, 'lean_rag'):
            values = (previous['proxy_summary'] if arm == 'lean_rag' else metrics)['arms'][arm]
            scores = previous['supported_case_ci_per_seed'] if arm == 'lean_rag' else subset
            rates = [r['supported_case_ci_pass_at_1'] for r in scores if r['arm'] == arm]
            rendered = ('unavailable' if not rates or any(r is None for r in rates)
                        else f'{statistics.mean(rates):.1%}')
            lines.append(f'| {arm} | {values["mean_edit_f1"]:.6f} | {rendered} |')
    lines.extend(['', 'Full-dataset CI Pass@1 remains unavailable unless every selected workflow resolves. '
                  'The machine-readable export retains all unsupported and unresolved cases, full-dataset '
                  'bounds, failed/reference controls, and the supported-subset denominators.', '',
                  'Local Linux containers differ from hosted CI. This is not an exact replication of '
                  'the original paper evaluation. No GitHub Actions evaluator was used.', '',
                  f'[Portable metrics, per-case outcomes, and audit provenance]({stem}_metrics.json)', '',
                  '[Run configuration and reproduction instructions](../local_ci/README.md)', ''])
    (ci.HERE / 'results' / markdown).write_text('\n'.join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--generation-pid', type=int, required=True)
    parser.add_argument('--evaluation-date', required=True)
    parser.add_argument('--reuse-evaluation', type=Path, action='append', default=[])
    parser.add_argument('--evaluation-code-name', default='evaluation-code',
                        help='Use a new snapshot name after an evaluator fix; old snapshots remain immutable')
    args = parser.parse_args()
    run = args.run.resolve()
    meta = ci.read(run / 'run_metadata.json')
    if ci.digest(meta)[:20] != run.name or meta['composition'] != 'recover-markers':
        raise ValueError('Unexpected paired benchmark run')
    expected = len(meta['ids']) * len(meta['seeds']) * len(meta['arms'])
    status_file = run / 'benchmark_status.json'
    log_file = run / 'evaluation.log'
    state = {'run': run.name, 'generation_pid': args.generation_pid, 'supervisor_pid': os.getpid(),
             'expected_predictions': expected}

    def status(phase, **values):
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
        ci.save_json(status_file, state)
        print(phase, flush=True)

    def execute(argv):
        with log_file.open('a') as log:
            subprocess.run([sys.executable, *map(str, argv)], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    try:
        frozen = freeze_evaluation_code(run, meta, args.evaluation_code_name)
        scripts = frozen / 'ci-builds-repair/baseline'
        state['evaluation_code'] = str(frozen)
        while True:
            path = run / 'metrics.json'
            metrics = ci.read(path) if path.exists() else {'complete': False, 'cases': []}
            status('generating', completed_predictions=len(metrics['cases']))
            if metrics['complete']:
                break
            if not alive(args.generation_pid):
                raise RuntimeError('Generation ended before all predictions were saved; resume the recorded run')
            time.sleep(20)
        status('auditing_generation', completed_predictions=expected)
        audit_script = meta.get('audit_script', 'run_qwen_lean_rag.py')
        if audit_script not in ('run_qwen_lean_rag.py', 'run_qwen_long_context.py'):
            raise ValueError('Unknown generation audit script')
        execute([scripts / audit_script, '--audit-run', run])
        status('evaluating_local_workflows')
        root = run / 'local-ci'
        candidates = list(root.glob('*/summary.json'))
        finished = [p.parent for p in candidates if len(ci.read(p)['predictions']) == expected]
        if finished:
            evaluation = finished[-1]
        else:
            if list(root.glob('*/manifest.json')):
                root = run / ('local-ci-retry-' + str(int(time.time())))
            command = [scripts / 'run_local_ci.py', '--all-cases', '--runs', run,
                       '--composition', 'recover-markers', '--output-dir', root]
            for i, previous in enumerate(args.reuse_evaluation):
                command.extend(['--reuse-evaluation' if i == 0 else '--additional-reuse-evaluation', previous])
            execute(command)
            evaluation, = [p.parent for p in root.glob('*/summary.json')]
        status('auditing_local_workflows', evaluation=str(evaluation))
        execute([scripts / 'run_local_ci.py', '--audit-run', evaluation, '--runs', run])
        status('publishing_results')
        report = publish(run, evaluation, args.evaluation_date)
        markdown = ('FULL_LONG_CONTEXT_BENCHMARK.md' if 'long_context' in meta['arms']
                    else 'FULL_RAG_BENCHMARK.md')
        status('complete' if report['full_dataset_ci_complete'] else 'complete_with_limited_ci_coverage',
               results=str(ci.HERE / 'results' / markdown))
    except BaseException as exc:
        status('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
