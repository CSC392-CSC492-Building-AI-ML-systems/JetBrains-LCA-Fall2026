"""Paired prefix/BM25 experiment. Retrieval mode is offline and costs nothing."""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
import urllib.error
import urllib.request

from bm25_retrieval.rag_retrieval import BM25Index, CodeTokenizer, VERSION, digest
from rag_pipeline.rag_budget import BudgetLedger


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def file_hash(path):
    result = hashlib.sha256()
    with open(path, 'rb') as f:
        while data := f.read(1024 * 1024):
            result.update(data)
    return result.hexdigest()


def load_env(path):
    # No shell evaluation, interpolation, logging, or persistence of credentials.
    if not path:
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.removeprefix('export ').split('=', 1)
        if key.strip() == 'OPENAI_API_KEY':
            os.environ.setdefault('OPENAI_API_KEY', value.strip().strip('\"\''))


class Dataset:
    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = file_hash(path)
        self.table, self.rows = None, None
        if self.path.suffix == '.arrow':
            import pyarrow as pa
            self.source = pa.memory_map(str(path), 'r')
            self.table = pa.ipc.open_stream(self.source).read_all()
        else:
            self.rows = [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]

    def __len__(self):
        return len(self.table) if self.table is not None else len(self.rows)

    def __getitem__(self, idx):
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        return self.table.slice(idx, 1).to_pylist()[0] if self.table is not None else self.rows[idx]

    def metadata(self):
        if self.table is not None:
            import pyarrow.compute as pc
            keys = ['repo', 'docfile_name', 'doc_type', 'intent']
            meta = self.table.select(keys).to_pylist()
            lengths = pc.utf8_length(self.table['relevant_code_context']).to_pylist()
            for row, length in zip(meta, lengths):
                row['context_chars'] = length
            return meta
        return [{k: r.get(k, '') for k in ['repo', 'docfile_name', 'doc_type', 'intent']} |
                {'context_chars': len(r['relevant_code_context'])} for r in self.rows]


def diverse_ids(metadata, count, seed=20261022):
    if not isinstance(count, int) or not 0 < count <= len(metadata):
        raise ValueError('Selection size outside dataset')
    items = list(enumerate(metadata))
    rng = random.Random(seed)
    rng.shuffle(items)
    groups = defaultdict(deque)
    for idx, row in items:
        n = row['context_chars']
        band = 0 if n <= 20000 else (1 if n <= 100000 else 2)
        generic = row['intent'].lower().startswith('generate documentation')
        groups[(row['doc_type'], band, generic)].append((idx, row['repo']))
    groups = {k: groups[k] for k in sorted(groups)}
    selected, repositories = [], set()
    while len(selected) < count:
        progress = False
        for queue in groups.values():
            pos = next((i for i, (_, repo) in enumerate(queue) if repo not in repositories), None)
            if pos is None:
                continue
            idx, repo = queue[pos]
            del queue[pos]
            repositories.add(repo)
            selected.append(idx)
            progress = True
            if len(selected) == count:
                return selected
        if not progress:
            for queue in groups.values():
                if queue:
                    selected.append(queue.popleft()[0])
                    if len(selected) == count:
                        return selected
    return selected


def build_prompt(row, context):
    # Byte-for-byte original baseline prompt; only code context changes.
    name, intent = row['docfile_name'], row['intent']
    prompt = 'I have code collected from one or more files joined into one string. '
    prompt += f'Using the code generate text for {name} file with documentation about {intent}.\n\n'
    prompt += f'My code:\n\n{context}'
    prompt += (f'\n\n\n\nAs answer return text for {name} file about {intent}. '
               'Do not return the instruction how to make documentation, return only documentation itself.')
    return prompt


def request_generation(key, model, messages, output_cap):
    payload = {'model': model, 'messages': messages, 'temperature': 0.0,
               'max_tokens': output_cap, 'n': 1, 'stream': False, 'top_p': 1.0,
               'frequency_penalty': 0.0, 'presence_penalty': 0.0}
    req = urllib.request.Request('https://api.openai.com/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + key,
                                                    'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as result:
        return json.load(result)


def check_model(key, model):
    # A model listing request is not a paid generation or proof inference works.
    req = urllib.request.Request('https://api.openai.com/v1/models/' + model,
                                  headers={'Authorization': 'Bearer ' + key})
    try:
        with urllib.request.urlopen(req, timeout=30) as result:
            body = json.load(result)
        return {'model': body['id'], 'listed_for_account': True}
    except urllib.error.HTTPError as exc:
        try:
            code = json.loads(exc.read()).get('error', {}).get('code')
        except Exception:
            code = None
        return {'model': model, 'listed_for_account': False, 'http_status': exc.code,
                'error_code': code if isinstance(code, str) and re.fullmatch('[A-Za-z_0-9-]+', code) else None}


def ensure_manifest(directory, manifest):
    directory = Path(directory)
    path = directory / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError('Manifest mismatch: refusing stale-output reuse')
    elif directory.exists() and any(directory.iterdir()):
        raise RuntimeError('Nonempty output without manifest: refusing reuse')
    else:
        atomic_json(path, manifest)


def run(args):
    config = json.loads(Path(args.config).read_text())
    if config.get('model') not in ['gpt-3.5-turbo-0125', 'gpt-4.1-mini']:
        raise ValueError('Unsupported/unpriced generation model')
    if args.mode == 'check-model':
        load_env(args.env_file)
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            raise RuntimeError('OPENAI_API_KEY is missing; keep it in a local env file')
        result = check_model(key, config['model'])
        if args.report:
            atomic_json(args.report, result)
        print(json.dumps(result))
        return
    if not args.dataset or not args.tokenizer:
        raise ValueError('Provide the local dataset and GPT-2 tokenizer JSON')
    dataset, tokenizer = Dataset(args.dataset), CodeTokenizer(args.tokenizer)
    ids = config.get('ids')
    if ids is None:
        ids = diverse_ids(dataset.metadata(), config['count'], config.get('seed', 20261022))
    if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 or i >= len(dataset) for i in ids):
        raise ValueError('Invalid or duplicate original dataset IDs')
    settings = config.get('retrieval', {})
    budget, output_cap = config.get('context_tokens', 2000), config.get('max_output_tokens', 2000)
    if not isinstance(budget, int) or budget <= 0 or budget > 2000:
        raise ValueError('Context allowance must be an integer from 1 to 2000')
    if type(output_cap) is not int or not 1 <= output_cap <= 2000:
        raise ValueError('Output allowance must be 1 to 2000')
    strategies = config.get('strategies', ['prefix', 'bm25'])
    if not strategies or len(set(strategies)) != len(strategies) or any(x not in ['prefix', 'bm25'] for x in strategies):
        raise ValueError('Invalid strategies')
    # Evaluation references are deliberately excluded from the generation identity.
    pipeline_dir = Path(__file__).resolve().parent
    implementation_files = (
        pipeline_dir / 'rag_replication.py',
        pipeline_dir.parent / 'bm25_retrieval' / 'rag_retrieval.py',
        pipeline_dir / 'rag_budget.py',
    )
    implementation = digest(''.join(path.read_text() for path in implementation_files))
    manifest = {'schema_version': 1, 'algorithm': VERSION, 'implementation_sha256': implementation,
                'dataset': 'JetBrains-Research/lca-module-summarization', 'dataset_sha256': dataset.sha256,
                'dataset_rows': len(dataset), 'original_ids': ids, 'tokenizer': 'gpt2-tokenizers-json',
                'tokenizer_sha256': tokenizer.fingerprint, 'model': config['model'],
                'context_tokens': budget, 'max_output_tokens': output_cap, 'temperature': 0,
                'retrieval': settings, 'strategies': strategies,
                'scope': 'retrieval_within_supplied_relevant_code_context'}
    identity = digest(json.dumps(manifest, sort_keys=True))[:16]
    root = Path(args.output_root) / (config.get('name', 'paired') + '-' + identity)
    ensure_manifest(root, manifest)
    ledger, key = None, None
    if args.mode == 'generate':
        load_env(args.env_file)
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            raise RuntimeError('OPENAI_API_KEY missing')
        ledger = BudgetLedger(args.ledger, config.get('budget_usd', 4))
    records = []
    start = time.perf_counter()
    previous_hash, index = None, None
    for idx in ids:
        raw = dataset[idx]
        # Pass only allowed fields onward; no target_text in query, index or prompt.
        row = {k: raw[k] for k in ['repo', 'docfile_name', 'intent', 'relevant_code_context']}
        code = row['relevant_code_context'] or ''
        for strategy in strategies:
            tick = time.perf_counter()
            folder = root / strategy
            folder.mkdir(exist_ok=True)
            evidence_path = folder / f'{idx}.context.json'
            code_hash = digest(code)
            if evidence_path.exists():
                evidence = json.loads(evidence_path.read_text())
                context, trace = evidence['context'], evidence['trace']
                if evidence['source_sha256'] != code_hash or digest(context) != trace['context_sha256']:
                    raise RuntimeError('Cached evidence changed; refusing reuse')
            else:
                if strategy == 'prefix':
                    context = tokenizer.truncate(code, budget)
                    trace = {'selected': [{'start_char': 0, 'end_char': len(context), 'score': None}],
                             'fallback': None, 'context_tokens': tokenizer.count(context),
                             'context_sha256': digest(context)}
                else:
                    if previous_hash != code_hash:
                        index = BM25Index.load_or_build(code, tokenizer, args.cache_dir,
                            size=settings.get('chunk_tokens', 400), overlap=settings.get('overlap_tokens', 50),
                            k1=settings.get('k1', 1.5), b=settings.get('b', .75))
                        previous_hash = code_hash
                    query = row['intent'] + ' ' + Path(row['docfile_name']).stem
                    context, trace = index.retrieve(query, tokenizer, budget)
                evidence = {'original_id': idx, 'repo': row['repo'], 'docfile_name': row['docfile_name'],
                            'source_sha256': code_hash, 'context': context, 'trace': trace,
                            'preparation_seconds': time.perf_counter() - tick}
                atomic_json(evidence_path, evidence)
            if tokenizer.count(context) > budget:
                raise RuntimeError('Context exceeds budget')
            prompt = build_prompt(row, context)
            prediction_path, response_path = folder / f'{idx}.txt', folder / f'{idx}.generation.json'
            generated = False
            if args.mode == 'generate':
                if response_path.exists():
                    response = json.loads(response_path.read_text())
                    if response['prompt_sha256'] != digest(prompt):
                        raise RuntimeError('Prediction/prompt mismatch')
                    if prediction_path.exists() and digest(prediction_path.read_text()) != response['prediction_sha256']:
                        raise RuntimeError('Prediction content changed')
                    prediction_path.write_text(response['prediction'])
                else:
                    if prediction_path.exists():
                        raise RuntimeError('Prediction without generation provenance')
                    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                                {'role': 'user', 'content': prompt}]
                    job = identity + ':' + strategy + ':' + str(idx)
                    ledger.reserve(job, config['model'], messages, output_cap)
                    tick_api = time.perf_counter()
                    try:
                        api = request_generation(key, config['model'], messages, output_cap)
                    except urllib.error.HTTPError as exc:
                        if exc.code in [400, 401, 403, 404]:
                            ledger.reject(job)
                        # Do not print response bodies or credentials; no automatic retries.
                        atomic_json(folder / f'{idx}.error.json', {'http_status': exc.code,
                            'budget': ledger.summary(), 'retry': 'manual reconciliation required'})
                        raise RuntimeError(f'Generation rejected with HTTP {exc.code}; stopped without retry') from None
                    except Exception as exc:
                        atomic_json(folder / f'{idx}.error.json', {'exception': type(exc).__name__,
                            'budget': ledger.summary(), 'retry': 'reservation retained; do not retry blindly'})
                        raise RuntimeError('Network/response failure; reservation retained and run stopped') from None
                    prediction = api['choices'][0]['message']['content']
                    if not isinstance(prediction, str) or not prediction.strip():
                        raise RuntimeError('Empty generation; retain reservation and investigate')
                    response = {'model_requested': config['model'], 'model_returned': api.get('model'),
                                'request_id': api.get('id'), 'prompt_sha256': digest(prompt),
                                'usage': api.get('usage', {}), 'prediction': prediction,
                                'prediction_sha256': digest(prediction),
                                'finish_reason': api['choices'][0].get('finish_reason'),
                                'generation_seconds': time.perf_counter() - tick_api}
                    # Persist output before releasing the reservation; interruption stays conservative.
                    atomic_json(response_path, response)
                    response['cost_usd'] = ledger.settle(job, response['usage'])
                    atomic_json(response_path, response)
                    prediction_path.write_text(prediction)
                    generated = True
            record = {'original_id': idx, 'repo': row['repo'], 'strategy': strategy,
                      'context_tokens': trace['context_tokens'], 'fallback': trace.get('fallback'),
                      'context_sha256': trace['context_sha256'], 'prediction_exists': prediction_path.exists(),
                      'preparation_seconds': evidence['preparation_seconds']}
            records.append(record)
            print(json.dumps({'id': idx, 'strategy': strategy, 'tokens': trace['context_tokens'],
                              'new_generation': generated, 'progress': len(records), 'total': len(ids)*len(strategies)}), flush=True)
            atomic_json(root / 'progress.json', {'records': records, 'budget': ledger.summary() if ledger else None})
    summary = {'manifest': manifest, 'run_dir': str(root.resolve()), 'mode': args.mode,
               'tasks': len(ids), 'repositories': len(set(r['repo'] for r in records)),
               'records': records, 'elapsed_seconds': time.perf_counter() - start,
               'budget': ledger.summary() if ledger else None,
               'official_compscore': 'not_run', 'quality_improvement': 'not_established'}
    atomic_json(root / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['manifest', 'records']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['retrieve', 'generate', 'check-model'], default='retrieve')
    parser.add_argument('--dataset', help='Local Hugging Face .arrow or one-row-per-line JSON dataset')
    parser.add_argument('--tokenizer', help='Local GPT-2 tokenizer.json')
    parser.add_argument('--env-file')
    parser.add_argument('--output-root', default='data/rag_runs')
    parser.add_argument('--cache-dir', default='data/rag_index_cache')
    parser.add_argument('--ledger', default='data/rag_spending.sqlite')
    parser.add_argument('--report', help='Optional model-access JSON output')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
