import json, random, sys, os
home = os.path.expanduser('~')
split = [json.loads(l) for l in open(home + '/lca_remote/E4/failure_split.jsonl')]
base = home + '/lca_remote/E4/codellama-7b/medium_context/arms/'

def results(arm):
    d = {}
    for l in open(base + arm + '/generation_results.jsonl', encoding='utf8'):
        r = json.loads(l)
        d[(r['datapoint_idx'], r['line_num'])] = r
    return d

arm = 'e4_F_N12000'
res = results(arm)
res_pd = results('e4_pd_N12000')
rows = [r for r in split if r['arm'] == arm]
pd_ok = {(r['dp'], r['line_num']) for r in split if r['arm'] == 'e4_pd_N12000' and r['correct']}
groups = {
    'retrieval (name missing)': [r for r in rows if r['cls'] == 'retrieval'],
    'reasoning (names visible)': [r for r in rows if r['cls'] == 'reasoning'],
    'wrong self-contained': [r for r in rows if r['cls'] == 'wrong_self_contained'],
    'F wrong, pd right': [r for r in rows if not r['correct'] and (r['dp'], r['line_num']) in pd_ok],
}
rng = random.Random(0)
out = []
for name, g in groups.items():
    out.append(f'\n######## {name}: {len(g)} lines (random sample of 25, seed 0)')
    for r in rng.sample(g, min(25, len(g))):
        x = res[(r['dp'], r['line_num'])]
        out.append(f"[{r['repo']} L{r['line_num']} {r['line_class']} needed={r['n_needed']} visible={r['n_visible']}]")
        out.append('  GT  : ' + x['ground_truth'].strip()[:200])
        out.append('  F   : ' + x['prediction_line'].strip()[:200])
        out.append('  PD  : ' + res_pd[(r['dp'], r['line_num'])]['prediction_line'].strip()[:200])
open(sys.argv[1], 'w', encoding='utf8').write('\n'.join(out))
print('written', len(out))
