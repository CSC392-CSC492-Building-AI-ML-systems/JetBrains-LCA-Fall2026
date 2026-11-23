"""Paired full-file context benchmark; reuse audited identical baseline answers."""

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import long_context as context
import run_qwen_lean_rag as lean

pilot, dev = lean.pilot, lean.dev
ARMS = ('plain', 'long_context')
DEFAULT_BASELINE = pilot.BENCHMARK / 'out/qwen-lean-rag/372e8aa90832060ccc54'


def baseline_manifest(folder):
    meta = pilot.read_json(folder / 'run_metadata.json')
    if folder.name != pilot.digest(meta)[:20] or not pilot.read_json(folder / 'metrics.json')['complete']:
        raise ValueError('Baseline manifest mismatch or incomplete baseline')
    if (meta['context'], meta['cap'], meta['temperature'], meta['composition']) != (32768, 8192, .2, 'recover-markers'):
        raise ValueError('Baseline settings differ from this comparison')
    return meta


def audit_baseline(folder, meta):
    frozen = folder / 'evaluation-code'
    for name, checksum in meta['source_hashes'].items():
        if not any(pilot.file_sha(p) == checksum for p in frozen.rglob(name)):
            raise ValueError('Missing frozen baseline audit code: ' + name)
    script = frozen / 'ci-builds-repair/baseline/run_qwen_lean_rag.py'
    subprocess.run([sys.executable, str(script), '--audit-run', str(folder)], check=True,
                   stdout=subprocess.DEVNULL)


def reuse_plain(folder, i, seed, prompt, identity, meta):
    original, checksum = pilot.read_checked(folder / 'examples' / str(i) / str(seed) / 'plain.json')
    if (original['identity']['run'] != folder.name or original['identity']['id'] != i
            or original['identity']['seed'] != seed or original['identity']['arm'] != 'plain'
            or original['prompt'] != prompt):
        raise ValueError('Source baseline answer identity differs')
    return {**original, 'identity': identity,
            'reused_from': {'run': folder.name, 'id': i, 'seed': seed, 'arm': 'plain',
                            'payload_sha256': checksum, 'manifest_sha256': pilot.digest(meta)}}


def summarize(rows, ids, seeds):
    indexed = {(r['id'], r['seed'], r['arm']): r for r in rows}
    expected_keys = {(i, s, a) for i in ids for s in seeds for a in ARMS}
    if len(indexed) != len(rows) or set(indexed) - expected_keys:
        raise ValueError('Duplicate or unexpected benchmark row')
    n = len(ids) * len(seeds)
    totals = {}
    for arm in ARMS:
        group = [r for r in rows if r['arm'] == arm]
        totals[arm] = {'scored': len(group), 'expected': n,
                       'valid_response': sum(r['error'] is None for r in group),
                       'nonempty_patch': sum(r['nonempty_patch'] for r in group),
                       'nonempty_no_detected_syntax_regression': sum(r['nonempty_patch'] and r['syntax_preserved'] for r in group),
                       'exact_reference': sum(r['patch_exact_match'] for r in group),
                       'mean_edit_f1': statistics.mean(r['edit_token_f1'] for r in group) if group else None,
                       'errors': dict(Counter(r['error'] for r in group if r['error']))}
    delta = [indexed[i, s, 'long_context']['edit_token_f1'] - indexed[i, s, 'plain']['edit_token_f1']
             for i in ids for s in seeds if (i, s, 'long_context') in indexed and (i, s, 'plain') in indexed]
    return {'scope': 'Offline text and syntax proxies, not CI Pass@1',
            'complete': set(indexed) == expected_keys, 'arms': totals, 'matched_pairs': len(delta),
            'paired_f1_delta': statistics.mean(delta) if delta else None,
            'paired_f1_wins_ties_losses': [sum(d > 0 for d in delta), sum(d == 0 for d in delta), sum(d < 0 for d in delta)],
            'cases': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--out', type=Path, default=pilot.BENCHMARK / 'out/qwen-long-context')
    parser.add_argument('--ids', type=int, nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--audit-run', type=Path)
    args = parser.parse_args()
    from datasets import Dataset
    if args.audit_run:
        saved = pilot.read_json(args.audit_run / 'run_metadata.json')
        args.baseline_run = pilot.BENCHMARK / 'out' / saved['baseline_run']
        args.ids, args.seeds, args.out = saved['ids'], saved['seeds'], args.audit_run.parent
    baseline = args.baseline_run.resolve()
    previous = baseline_manifest(baseline)
    ids = args.ids if args.ids is not None else previous['ids']
    if (not ids or len(set(ids)) != len(ids) or set(ids) - set(previous['ids'])
            or not args.seeds or len(set(args.seeds)) != len(args.seeds) or set(args.seeds) - set(previous['seeds'])):
        parser.error('Require distinct IDs/seeds available in the paired baseline')
    audit_baseline(baseline, previous)
    arrow, = [p for p in (pilot.BENCHMARK / '.venv/dataset-cache').rglob('*test.arrow')
              if pilot.file_sha(p) == previous['dataset_sha256']]
    rows = {r['id']: r for r in Dataset.from_file(str(arrow)) if r['id'] in ids}
    modules = [Path(__file__), Path(context.__file__), Path(lean.__file__), Path(dev.__file__), Path(pilot.__file__)]
    modules.extend(Path(sys.modules[name].__file__) for name in
                   ('retrieval', 'evaluation', 'paper_mistral_baseline', 'project_level_code_completion.rag.bm25'))
    import repair_composition
    modules.append(Path(repair_composition.__file__))
    meta = {k: previous[k] for k in ('model', 'revision', 'model_hashes', 'packages', 'dataset_sha256',
                                    'context', 'cap', 'temperature', 'composition')}
    meta.update(ids=ids, seeds=args.seeds, arms=list(ARMS),
                failing_shas={str(i): rows[i]['sha_fail'] for i in ids},
                context_strategy=context.POLICY, audit_script=Path(__file__).name,
                baseline_run=str(baseline.relative_to((pilot.BENCHMARK / 'out').resolve())),
                baseline_manifest_sha256=pilot.digest(previous),
                source_hashes={p.name: pilot.file_sha(p) for p in modules},
                effective_seed='seed + dataset ID',
                selection='All cached dataset cases unless explicitly selected; no score-based selection')
    folder = args.out / pilot.digest(meta)[:20]
    print('Long context: ' + str(folder), flush=True)
    if args.audit_run:
        if folder.resolve() != args.audit_run.resolve() or pilot.read_json(folder / 'run_metadata.json') != meta:
            raise ValueError('Recorded run identity or generation sources differ')
    else:
        if (folder / 'run_metadata.json').exists() and pilot.read_json(folder / 'run_metadata.json') != meta:
            raise ValueError('Existing run metadata mismatch')
        pilot.save_json(folder / 'run_metadata.json', meta)
    for source in modules:
        archived = folder / 'sources' / source.name
        if args.audit_run or archived.exists():
            if pilot.file_sha(archived) != meta['source_hashes'][source.name]:
                raise ValueError('Archived generation source changed')
        else:
            archived.parent.mkdir(exist_ok=True)
            archived.write_bytes(source.read_bytes())
    sys.path.insert(0, str(dev.raw.previous.DEPS))
    if not args.audit_run:
        if {n: importlib.metadata.version(n) for n in meta['packages']} != meta['packages']:
            raise ValueError('Runtime differs from paired baseline')
        for name, checksum in meta['model_hashes'].items():
            if pilot.file_sha(dev.MODEL / name) != checksum:
                raise ValueError('Model artifact differs from paired baseline')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(dev.MODEL), local_files_only=True)

    def tokens(text):
        return lean.encode_prompt(tokenizer, text)

    network = None
    results, preflight = [], []
    for i in ids:
        row = rows[i]
        if previous['failing_shas'][str(i)] != row['sha_fail']:
            raise ValueError('Failed snapshot differs from baseline')
        gold = pilot.diff_blocks(row['diff'])
        blocks = pilot.prediction_blocks(gold)
        target_files = pilot.target_files(row, blocks, lean.CACHE)
        files = {**lean.cached_source(row), **target_files}
        base = pilot.make_prompt({'logs': row['logs']}, blocks)
        evidence_path = folder / 'evidence' / f'{i}.json'
        packed = context.pack(base, files, blocks, lambda t: len(tokens(t)), meta['context'], meta['cap'])
        evidence = {'sha_fail': row['sha_fail'], 'policy': context.POLICY, **packed}
        if args.audit_run or evidence_path.exists():
            stored, evidence_sha = pilot.read_checked(evidence_path)
            if stored != evidence:
                raise ValueError('Context packing differs from saved evidence')
        else:
            evidence_sha = pilot.digest(evidence)
            pilot.save_json(evidence_path, {**evidence, 'payload_sha256': evidence_sha})
        context.validate(evidence, files)
        prompt = context.assemble(base, evidence['sections'])
        input_ids = tokens(prompt)
        if len(input_ids) != evidence['input_tokens'] or len(input_ids) + meta['cap'] + 256 > meta['context']:
            raise ValueError('Input/output budget mismatch')
        preflight.append({'id': i, 'plain_tokens': len(tokens(base)), 'long_context_tokens': len(input_ids),
                          'complete_files': sum(s['complete_file'] for s in evidence['sections']),
                          'partial_paths': evidence['partial_paths'], 'omitted_paths': evidence['omitted_paths'],
                          'absent_target_paths': evidence['absent_target_paths'],
                          'fits': True})
        if not args.audit_run:
            pilot.save_json(folder / 'preflight.json', {'complete': len(preflight) == len(ids), 'cases': preflight,
                                                      'expected_predictions': len(ids) * len(args.seeds) * 2,
                                                      'new_generations': len(ids) * len(args.seeds)})
        print(f'Prepared {i}: {len(input_ids)} input tokens, {preflight[-1]["complete_files"]} complete files', flush=True)
        if args.prepare_only:
            continue
        for seed in args.seeds:
            for arm in ARMS:
                identity = {'run': folder.name, 'id': i, 'seed': seed, 'arm': arm,
                            'sha_fail': row['sha_fail'], 'context_sha256': evidence_sha}
                path = folder / 'examples' / str(i) / str(seed) / f'{arm}.json'
                expected_plain = reuse_plain(baseline, i, seed, base, identity, previous) if arm == 'plain' else None
                if args.audit_run or path.exists():
                    answer, checksum = pilot.read_checked(path)
                    if (answer['identity'] != identity or answer['prompt'] != (base if arm == 'plain' else prompt)
                            or (arm == 'plain' and answer != expected_plain)):
                        raise ValueError('Saved answer identity or baseline provenance differs')
                elif arm == 'plain':
                    answer = expected_plain
                    checksum = pilot.digest(answer)
                    pilot.save_json(path, {**answer, 'payload_sha256': checksum})
                else:
                    if network is None:
                        import mlx.core as mx
                        from mlx_lm import load, stream_generate
                        from mlx_lm.sample_utils import make_sampler
                        network, mlx_tokenizer = load(str(dev.MODEL))
                        if lean.encode_prompt(mlx_tokenizer, prompt) != input_ids:
                            raise ValueError('Inference and preflight tokenizers differ')
                    mx.random.seed(seed + i)
                    response, finish, count = '', None, 0
                    started = time.monotonic()
                    for piece in stream_generate(network, mlx_tokenizer, input_ids, max_tokens=meta['cap'],
                                                 sampler=make_sampler(temp=meta['temperature'])):
                        response += piece.text
                        finish, count = piece.finish_reason, piece.generation_tokens
                        if count and count % 512 == 0:
                            print(f'  {i}/{seed}: {count} output tokens', flush=True)
                    mx.clear_cache()
                    answer = {'identity': identity, 'prompt': prompt, 'prompt_tokens': len(input_ids),
                              'response': response, 'finish_reason': finish, 'generation_tokens': count,
                              'skip': None, 'elapsed_seconds': time.monotonic() - started}
                    checksum = pilot.digest(answer)
                    pilot.save_json(path, {**answer, 'payload_sha256': checksum})
                if answer['prompt_tokens'] != len(tokens(answer['prompt'])):
                    raise ValueError('Saved prompt token count differs')
                measured = lean.score_answer(answer, blocks, target_files, gold, meta['composition'])
                results.append({'id': i, 'seed': seed, 'arm': arm, 'prediction_sha256': checksum,
                                'prompt_tokens': answer['prompt_tokens'], 'generation_tokens': answer['generation_tokens'],
                                **measured})
                if not args.audit_run:
                    pilot.save_json(folder / 'metrics.json', summarize(results, ids, args.seeds))
                print(f'{i}/{seed}/{arm}: {measured["error"] or "valid"} F1={measured["edit_token_f1"]:.3f}', flush=True)
    if args.prepare_only:
        pilot.save_json(folder / 'preflight.json', {'complete': True, 'cases': preflight,
                                                  'expected_predictions': len(ids) * len(args.seeds) * 2,
                                                  'new_generations': len(ids) * len(args.seeds)})
    else:
        metrics = summarize(results, ids, args.seeds)
        if args.audit_run and metrics != pilot.read_json(folder / 'metrics.json'):
            raise ValueError('Recomputed scores differ from saved results')
        print(json.dumps({k: v for k, v in metrics.items() if k != 'cases'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
