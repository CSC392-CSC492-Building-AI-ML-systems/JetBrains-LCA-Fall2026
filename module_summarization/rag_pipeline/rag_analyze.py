"""Audit source provenance and budgets; score outputs only when they exist."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
from rag_pipeline.rag_replication import Dataset, atomic_json, digest


def rouge_l_f1(prediction, reference):
    """Same whitespace-token LCS F1 as baseline, using a bitset LCS algorithm."""
    pred, ref = prediction.split(), reference.split()
    if not pred or not ref:
        return 0.0
    masks = {}
    for i, token in enumerate(ref):
        masks[token] = masks.get(token, 0) | (1 << i)
    state = 0
    for token in pred:
        union = masks.get(token, 0) | state
        state = union & ~(union - ((state << 1) | 1))
    lcs = state.bit_count()
    return 2 * lcs / (len(pred) + len(ref))


def analyze(directory, dataset):
    root = Path(directory)
    manifest = json.loads((root/'manifest.json').read_text())
    if dataset.sha256 != manifest['dataset_sha256']:
        raise ValueError('Dataset does not match generation manifest')
    metadata = dataset.metadata()
    records, different, late = [], 0, 0
    for idx in manifest['original_ids']:
        row = dataset[idx]
        contexts = {}
        for strategy in manifest['strategies']:
            folder = root/strategy
            evidence = json.loads((folder/f'{idx}.context.json').read_text())
            trace = evidence['trace']
            source = row['relevant_code_context'] or ''
            if digest(source) != evidence['source_sha256']:
                raise ValueError('Source changed')
            spans = [(x['start_char'], x['end_char']) for x in trace['selected']]
            if any(not 0 <= a <= b <= len(source) for a,b in spans):
                raise ValueError('Invalid source offsets')
            ordered = sorted(spans)
            if any(a[1] > b[0] for a,b in zip(ordered, ordered[1:])):
                raise ValueError('Duplicate source overlap')
            rebuilt = '\n\n'.join(source[a:b] for a,b in spans)
            if rebuilt != evidence['context'] or digest(rebuilt) != trace['context_sha256']:
                raise ValueError('Retrieved text does not match source offsets')
            if trace['context_tokens'] > manifest['context_tokens']:
                raise ValueError('Token budget exceeded')
            contexts[strategy] = evidence
            prediction = folder/f'{idx}.txt'
            record = {'original_id':idx, 'repo':row['repo'], 'doc_type':metadata[idx]['doc_type'],
                      'context_chars':len(source), 'strategy':strategy,
                      'tokens':trace['context_tokens'], 'fallback':trace.get('fallback'),
                      'prediction_exists':prediction.exists(), 'preparation_seconds':evidence['preparation_seconds']}
            if prediction.exists():
                generation = json.loads((folder/f'{idx}.generation.json').read_text())
                text = prediction.read_text()
                if digest(text) != generation['prediction_sha256']:
                    raise ValueError('Prediction hash mismatch')
                record.update(rouge_l_f1_diagnostic=rouge_l_f1(text,row.get('target_text') or ''),
                              generation_seconds=generation['generation_seconds'],
                              usage=generation['usage'], returned_model=generation['model_returned'])
            records.append(record)
        if 'prefix' in contexts and 'bm25' in contexts:
            different += contexts['prefix']['context'] != contexts['bm25']['context']
            end = max((s['end_char'] for s in contexts['prefix']['trace']['selected']), default=0)
            late += any(s['end_char'] > end for s in contexts['bm25']['trace']['selected'])
    summaries = {}
    for strategy in manifest['strategies']:
        rows = [r for r in records if r['strategy']==strategy]
        scored = [r for r in rows if 'rouge_l_f1_diagnostic' in r]
        summaries[strategy] = {'contexts':len(rows), 'predictions':len(scored),
            'tokens_min':min(r['tokens'] for r in rows), 'tokens_max':max(r['tokens'] for r in rows),
            'tokens_mean':mean(r['tokens'] for r in rows),
            'fallbacks':dict(Counter(r['fallback'] for r in rows if r['fallback'])),
            'rouge_l_f1_diagnostic_mean':mean(r['rouge_l_f1_diagnostic'] for r in scored) if scored else None}
    result = {'run_dir':str(root.resolve()), 'source_offset_audit':'passed',
              'tasks':len(manifest['original_ids']), 'repositories':len({r['repo'] for r in records}),
              'different_context_pairs':different, 'bm25_pairs_using_source_beyond_prefix':late,
              'strategies':summaries, 'records':records,
              'official_compscore':'not_run', 'quality_improvement':'not_established',
              'note':'Source provenance and token budget checks are not retrieval relevance or documentation quality scores.'}
    atomic_json(root/'analysis.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--dataset',required=True)
    args=parser.parse_args()
    result=analyze(args.run_dir,Dataset(args.dataset))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
