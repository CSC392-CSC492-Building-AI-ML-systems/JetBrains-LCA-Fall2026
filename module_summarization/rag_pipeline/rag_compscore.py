"""Prepare or run the original CompScore metric with explicit original row IDs.

No judge is downloaded and no GPU is used unless --execute is supplied.
"""
import argparse
import json
from pathlib import Path
from statistics import mean
from rag_pipeline.rag_replication import Dataset, atomic_json, digest


def prepare(root, dataset, *, judge_revision=None, judge_tokenizer_revision=None):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    if dataset.sha256!=manifest['dataset_sha256']:
        raise ValueError('Wrong dataset for these predictions')
    ids=manifest['original_ids']
    if not ids or len(set(ids))!=len(ids) or any(type(i) is not int or not 0 <= i < len(dataset) for i in ids):
        raise ValueError('Invalid original dataset IDs')
    missing=[]
    for strategy in manifest['strategies']:
        for idx in manifest['original_ids']:
            pred=root/strategy/f'{idx}.txt'
            provenance=root/strategy/f'{idx}.generation.json'
            if not pred.exists() or not provenance.exists():
                missing.append({'strategy':strategy,'original_id':idx})
            elif digest(pred.read_text())!=json.loads(provenance.read_text())['prediction_sha256']:
                raise ValueError('Prediction changed since generation')
    plan={'ready':not missing,'original_ids':manifest['original_ids'],
          'strategies':manifest['strategies'],'missing_predictions':missing,
          'judge_evaluations':len(manifest['original_ids'])*len(manifest['strategies'])*2,
          'judge':'mistralai/Mistral-7B-Instruct-v0.2',
          'judge_tokenizer':'meta-llama/Llama-2-7b-chat-hf','judge_code_tokens':6000,
          'metric':'original metrics.get_metric; probability scale 0 to 1',
          'status':'not_run'}
    if judge_revision:
        plan['judge_revision'] = judge_revision
    if judge_tokenizer_revision:
        plan['judge_tokenizer_revision'] = judge_tokenizer_revision
    atomic_json(root/'compscore_plan.json',plan)
    return manifest,plan


def execute(root,dataset,plan,device):
    if not plan['ready']:
        raise RuntimeError('Predictions missing; GPU scoring is not ready')
    # Original project functions; don't replace the metric with free-text voting.
    from metrics import get_metric
    from utils.scorer import OptionsScoringModel
    from utils.context_utils import trim_context
    from transformers import AutoTokenizer
    tokenizer_options = {'revision': plan['judge_tokenizer_revision']} if 'judge_tokenizer_revision' in plan else {}
    judge_options = {'revision': plan['judge_revision']} if 'judge_revision' in plan else {}
    tokenizer = AutoTokenizer.from_pretrained(plan['judge_tokenizer'], **tokenizer_options)
    scorer = OptionsScoringModel(plan['judge'], device, **judge_options)
    results={strategy:[] for strategy in plan['strategies']}
    for idx in plan['original_ids']:
        row=dataset[idx]
        # Same source context for every strategy; NOT its selected generation chunks.
        code=trim_context(row['relevant_code_context'],tokenizer,plan['judge_code_tokens'])
        for strategy in plan['strategies']:
            prediction=(root/strategy/f'{idx}.txt').read_text()
            score=get_metric(scorer,row['intent'],code,row['target_text'],prediction)
            results[strategy].append({'original_id':idx,'score':score})
        atomic_json(root/'compscore_progress.json',{'status':'partial','results':results})
    output={'status':'complete','plan':dict(plan, status='complete'),'results':results,
            'means':{k:mean(x['score'] for x in v) for k,v in results.items()}}
    atomic_json(root/'compscore_results.json',output)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--judge-revision')
    parser.add_argument('--judge-tokenizer-revision')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    root=Path(args.run_dir)
    data=Dataset(args.dataset)
    _, plan = prepare(root, data, judge_revision=args.judge_revision,
                      judge_tokenizer_revision=args.judge_tokenizer_revision)
    if args.execute:
        print(json.dumps(execute(root,data,plan,args.device)['means'],indent=2))
    else:
        print(json.dumps({k:v for k,v in plan.items() if k!='missing_predictions'}|
                         {'missing_count':len(plan['missing_predictions'])},indent=2))
