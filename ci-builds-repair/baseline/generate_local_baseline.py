"""Generate fresh no-retrieval, oracle-block baseline answers with local models.

Raw answers remain immutable. Original and whitespace-preserving composition
are evaluated separately by run_local_ci.py. No project code executes here.
"""

import argparse
import importlib.metadata
import sys
import time
from pathlib import Path

import run_local_ci as ci
from paper_mistral_baseline import make_prompt


def audit_generation(folder, rows, data):
    meta = ci.read(folder / 'run_metadata.json')
    if folder.name != ci.digest(meta)[:20] or meta['dataset_sha256'] != ci.sha(data) or meta['retrieval'] is not False:
        raise ValueError('Generation manifest or dataset mismatch')
    for name, checksum in meta['source_hashes'].items():
        if ci.sha(folder / 'sources' / name) != checksum:
            raise ValueError('Archived generation source differs')
    records = []
    for i in meta['ids']:
        if rows[i]['sha_fail'] != meta['failing_shas'][str(i)]:
            raise ValueError('Generation source revision mismatch')
        prompt = make_prompt({'logs': rows[i]['logs']}, ci.prediction_blocks(ci.diff_blocks(rows[i]['diff'])))
        if ci.digest(prompt) != meta['prompt_sha256'][str(i)]:
            raise ValueError('Reconstructed prompt mismatch')
        for seed in meta['seeds']:
            answer, checksum = ci.checked(folder / 'examples' / str(i) / str(seed) / 'plain.json')
            if answer['identity'] != {'run': folder.name, 'id': i, 'seed': seed, 'arm': 'plain', 'sha_fail': rows[i]['sha_fail']} or answer['prompt'] != prompt:
                raise ValueError('Prediction prompt or identity mismatch')
            records.append({'id': i, 'seed': seed, 'finish_reason': answer['finish_reason'],
                            'generation_tokens': answer['generation_tokens'], 'skip': answer['skip'], 'sha256': checksum})
    if ci.read(folder / 'generation_summary.json') != {'complete': True, 'records': records}:
        raise ValueError('Generation summary differs or is incomplete')
    return {'complete': True, 'predictions': len(records), 'model': meta['model'], 'retrieval': False,
            'model_loaded': False, 'source_run': folder.name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--ids', type=int, nargs='+', default=[142])
    selection.add_argument('--all-cases', action='store_true', help='Generate the full cached test split')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--model', choices=['qwen', 'mistral'], default='qwen')
    parser.add_argument('--context-window', type=int, default=16000)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=.2)
    parser.add_argument('--output-dir', type=Path, default=ci.BENCHMARK / 'out/standard-baseline')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--audit-run', type=Path, help='Audit a saved run by path without loading model weights')
    args = parser.parse_args()
    from datasets import Dataset
    data, = (ci.BENCHMARK / '.venv/dataset-cache').rglob('*test.arrow')
    rows = {r['id']: r for r in Dataset.from_file(str(data))}
    if args.audit_run:
        print(audit_generation(args.audit_run.resolve(), rows, data))
        return
    if args.all_cases:
        args.ids = list(rows)
    if (set(args.ids) - rows.keys() or len(set(args.ids)) != len(args.ids)
            or not args.ids or not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or args.max_tokens < 1 or args.context_window <= args.max_tokens + 256 or args.temperature < 0):
        parser.error('Require distinct valid IDs/seeds and positive input/output budgets')
    model = ci.BENCHMARK / '.venv' / ('qwen-coder-model' if args.model == 'qwen' else 'mistral-model')
    model_id = ('mlx-community/Qwen2.5-Coder-7B-Instruct-4bit' if args.model == 'qwen'
                else 'mlx-community/Mistral-7B-Instruct-v0.2-4bit')
    hashes = {name: ci.sha(model / name) for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors')}
    prepared = {}
    for i in args.ids:
        blocks = ci.prediction_blocks(ci.diff_blocks(rows[i]['diff']))
        # Verify each old hunk against its failed Git blob before prompting.
        ci.apply_edits(ci.target_files(rows[i]), blocks, [b['old'] for b in blocks])
        prepared[i] = make_prompt({'logs': rows[i]['logs']}, blocks)
    sources = [Path(__file__), Path(ci.__file__), Path(sys.modules['paper_mistral_baseline'].__file__)]
    manifest = {'model': model_id, 'model_hashes': hashes, 'ids': args.ids, 'seeds': args.seeds,
                'arms': ['plain'], 'retrieval': False, 'dataset_sha256': ci.sha(data),
                'failing_shas': {str(i): rows[i]['sha_fail'] for i in args.ids},
                'prompt_sha256': {str(i): ci.digest(prepared[i]) for i in args.ids},
                'context_window': args.context_window, 'max_tokens': args.max_tokens,
                'temperature': args.temperature, 'effective_seed': 'seed + dataset ID',
                'source_hashes': {p.name: ci.sha(p) for p in sources},
                'packages': {n: importlib.metadata.version(n) for n in ['mlx', 'mlx-lm', 'transformers', 'datasets']},
                'protocol': 'Existing paper-style oracle-block prompt; all blocks in one request; original seven-line logs',
                'deviations': ['Local four-bit checkpoint', 'Larger output cap than earlier 2048-token experiments',
                               'Original paper decoding parameters and full dataset identities are unavailable']}
    folder = args.output_dir / ci.digest(manifest)[:20]
    if (folder / 'run_metadata.json').exists():
        if ci.read(folder / 'run_metadata.json') != manifest:
            raise ValueError('Existing run metadata mismatch')
    elif args.audit_only:
        raise ValueError('No matching saved baseline')
    else:
        ci.save_json(folder / 'run_metadata.json', manifest)
        (folder / 'sources').mkdir()
        for source in sources:
            (folder / 'sources' / source.name).write_bytes(source.read_bytes())
    print('Standard baseline: ' + str(folder), flush=True)
    if args.prepare_only:
        return
    if not args.audit_only:
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        network, tokenizer = load(str(model))
    records = []
    for i in args.ids:
        for seed in args.seeds:
            identity = {'run': folder.name, 'id': i, 'seed': seed, 'arm': 'plain', 'sha_fail': rows[i]['sha_fail']}
            path = folder / 'examples' / str(i) / str(seed) / 'plain.json'
            if path.exists() or args.audit_only:
                answer, checksum = ci.checked(path)
                if answer['identity'] != identity or answer['prompt'] != prepared[i]:
                    raise ValueError('Saved prediction identity or prompt differs')
            else:
                tokens = tokenizer.apply_chat_template([{'role': 'user', 'content': prepared[i]}],
                                                       add_generation_prompt=True, tokenize=True)
                skip = 'input_budget' if len(tokens) + args.max_tokens + 256 > args.context_window else None
                response, finish, count = '', None, 0
                started = time.monotonic()
                if not skip:
                    mx.random.seed(seed + i)
                    for piece in stream_generate(network, tokenizer, tokens, max_tokens=args.max_tokens,
                                                 sampler=make_sampler(temp=args.temperature)):
                        response += piece.text
                        finish, count = piece.finish_reason, piece.generation_tokens
                        if count and count % 512 == 0:
                            print(f'  {i}/{seed}: {count} output tokens generated', flush=True)
                    mx.clear_cache()
                answer = {'identity': identity, 'prompt': prepared[i], 'prompt_tokens': len(tokens),
                          'response': response, 'finish_reason': finish, 'generation_tokens': count,
                          'skip': skip, 'elapsed_seconds': time.monotonic() - started}
                checksum = ci.digest(answer)
                ci.save_json(path, {**answer, 'payload_sha256': checksum})
            records.append({'id': i, 'seed': seed, 'finish_reason': answer['finish_reason'],
                            'generation_tokens': answer['generation_tokens'], 'skip': answer['skip'], 'sha256': checksum})
            print(f"  {i}/{seed}: {answer['skip'] or answer['finish_reason']}, {answer['generation_tokens']} output tokens", flush=True)
    ci.save_json(folder / 'generation_summary.json', {'complete': len(records) == len(args.ids) * len(args.seeds), 'records': records})


if __name__ == '__main__':
    main()
