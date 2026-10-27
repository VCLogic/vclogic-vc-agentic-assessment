"""Evaluate the memory-swap pilot with the main evaluation's measures.

Metric definitions are copied from outputs/final-solution/report.py (recall/precision/F1 at three levels,
decision-linked and investor-fit recall, target-investor identification, overlap between twins, within-panel AUC).
Rationales are the investor lens output; the weighted score is the tally of investigator + coverage rationales.
Balanced accuracy uses each investor's 70th-percentile cutoff of the MAIN evaluation's scores on their other
pitches (all pitches if fewer than 8), so the cutoff is the same in every condition.
Paired differences vs OWN: 2,000 bootstrap samples, seed 0, resampling episodes.

Run from the repo root: .venv/bin/python experiments/memory_swap_pilot/evaluate.py
"""
import itertools
import json
from pathlib import Path

exec(open('outputs/free-tests/_common.py').read())  # rows (main evaluation), cat, tally, top_cutoff, avg
from vc_clone_graph.components.cli import _lens_path

HERE = Path('experiments/memory_swap_pilot')
OUT = Path('outputs/memory-swap-pilot/runs')
crit = json.loads((HERE / 'criteria.json').read_text())
CONDS = ['OWN', 'SWAP', 'NAME_ONLY', 'GENERIC']
NB = 2000
main = {(r['vc'], r['ep']): r for r in rows}

# Main-evaluation cutoffs (fixed across conditions)
main_byvc = collections.defaultdict(list)
for r in rows:
    main_byvc[r['vc']].append(r)
def cutoff(vc, ep):
    own = [o['T'] for o in main_byvc[vc] if o['ep'] != ep]
    return top_cutoff(own if len(own) >= 8 else [o['T'] for o in rows if o['ep'] != ep], .3)

sample = crit['sample']['decisions']
data, missing, costs, results = {}, collections.defaultdict(list), collections.defaultdict(list), {}
for c in CONDS + ['MAIN']:
    data[c] = []
    for d in sample:
        vc, ep = d['vc_slug'], d['episode_slug']
        m = main[(vc, ep)]
        if c == 'MAIN':
            lens = json.load(open(_lens_path(Path('outputs/components-lens'), 'contrastive', vc, ep)))['rationales']
            S = m['T']
        else:
            p = OUT / c / vc / ep / 'result.json'
            if not p.is_file():
                missing[c].append(d['decision_id']); continue
            res = json.loads(p.read_text()); results[(c, d['decision_id'])] = res; costs[c].append(res['cost_total_usd'])
            run = Path(res['run_dir'])
            p1 = json.load(open(run / 'phase1/investigation.json'))['rationales']
            cov = json.load(open(run / 'components/coverage.json'))['additions']
            lens = json.load(open(run / 'components/lens.json'))['rationales']
            S = tally(p1 + cov)
        data[c].append(dict(vc=vc, ep=ep, y=bool(m['y']), ms=m['ms'], lens=lens, S=S, D=S >= cutoff(vc, ep)))

L = lambda it: {x['taxonomy_label'] for x in it}
f1 = lambda X, Z: 2 * len(X & Z) / max(1, len(X) + len(Z))
def bal(R):
    y = [r['y'] for r in R]; d = [r['D'] for r in R]
    tp = sum(a and b for a, b in zip(d, y)); tn = sum((not a) and (not b) for a, b in zip(d, y))
    return (tp / max(1, sum(y)) + tn / max(1, len(y) - sum(y))) / 2
def auc(groups):
    ok = tot = 0
    for g in groups:
        a = [r['S'] for r in g if r['y']]; b = [r['S'] for r in g if not r['y']]
        for x in a:
            for z in b: ok += (x > z) + .5 * (x == z); tot += 1
    return ok / tot if tot else float('nan')

def measures(groups):
    """groups: list of per-episode lists of decision dicts (one condition)."""
    I = [r for g in groups for r in g]; A = [r for r in I if r['ms'] is not None]
    multi = [m for m in ([r for r in g if r['ms'] is not None] for g in groups) if len(m) >= 2]
    o = {}
    for lvl, name in enumerate(['dimension', 'rationale', 'rationale + direction']):
        tp = nr = npd = 0
        for r in A:
            it, ms = r['lens'], r['ms']
            P = [{cat.get(l) for l in L(it)}, L(it), {(x['taxonomy_label'], x['direction']) for x in it}][lvl]
            R = [{m['rationale_category'] for m in ms}, {m['rationale_label'] for m in ms},
                 {(m['rationale_label'], m['direction']) for m in ms}][lvl]
            tp += len(P & R); nr += len(R); npd += len(P)
        o[f'{name} recall'] = tp / nr; o[f'{name} precision'] = tp / max(1, npd); o[f'{name} F1'] = 2 * tp / (nr + npd)
    def rec(pred):
        h = t = 0
        for r in A:
            R = {m['rationale_label'] for m in r['ms'] if pred(m)}; h += len(L(r['lens']) & R); t += len(R)
        return h / t if t else float('nan')
    o['decision-linked recall'] = rec(lambda m: m['decision_link'] != 'none')
    o['investor-fit recall'] = rec(lambda m: m['rationale_category'] == 'investor_fit_constraints')
    o['target-investor identification'] = avg([
        f1(L(a['lens']), {x['rationale_label'] for x in a['ms']}) > f1(L(a['lens']), {x['rationale_label'] for x in b['ms']})
        for m in multi for a, b in itertools.permutations(m, 2)])
    o['overlap between twins'] = avg([f1(L(a['lens']), L(b['lens'])) for m in multi for a, b in itertools.combinations(m, 2)])
    o['overlap between investors'] = avg([f1({x['rationale_label'] for x in a['ms']}, {x['rationale_label'] for x in b['ms']})
                                          for m in multi for a, b in itertools.combinations(m, 2)])
    o['within-panel AUC (weighted score)'] = auc(groups)
    o['balanced accuracy (weighted score, main cutoffs)'] = bal(I)
    o['rationales per pitch'] = avg([len(L(r['lens'])) for r in I])
    o['share in favour'] = avg([x['direction'] == 'positive' for r in I for x in r['lens']])
    return o

conds = [c for c in CONDS if data[c]]
# Paired analysis on the decisions completed in every condition (runs the pipeline gave up on are missing data)
keys = lambda c: {(r['vc'], r['ep']) for r in data[c]}
common = set.intersection(*[keys(c) for c in conds + ['MAIN']])
eps = sorted({e for _, e in common})
def groups_for(c, ep_list):
    by = collections.defaultdict(list)
    for r in data[c]:
        if (r['vc'], r['ep']) in common: by[r['ep']].append(r)
    return [by[e] for e in ep_list]
point = {c: measures(groups_for(c, eps)) for c in conds + ['MAIN']}
rng = np.random.default_rng(0); boot = []
for _ in range(NB):
    s = list(rng.choice(eps, len(eps)))
    boot.append({c: measures(groups_for(c, s)) for c in conds + ['MAIN']})
def ci(vals):
    v = [x for x in vals if x == x]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [None, None]
table = {}
for k in point['OWN']:
    table[k] = {c: {'value': point[c][k], 'ci': ci([b[c][k] for b in boot])} for c in conds + ['MAIN']}
    for c in conds[1:]:
        table[k][c]['diff_vs_OWN'] = point[c][k] - point['OWN'][k]
        table[k][c]['diff_ci'] = ci([b[c][k] - b['OWN'][k] for b in boot])
    table[k]['MAIN']['diff_OWN_minus_MAIN'] = point['OWN'][k] - point['MAIN'][k]
    table[k]['MAIN']['diff_ci'] = ci([b['OWN'][k] - b['MAIN'][k] for b in boot])

def met(c, k):
    d = table[k][c]; return point['OWN'][k] - point[c][k] > 0 and d['diff_ci'][1] < 0
primary = {k: met('SWAP', k) for k in ['target-investor identification', 'decision-linked recall']} if 'SWAP' in conds else {}
secondary = {f'OWN > {c}: {k}': met(c, k) for c in conds[1:] if c != 'SWAP'
             for k in ['target-investor identification', 'decision-linked recall']}
# Sensitivity: OWN vs SWAP on every decision completed in both (not only those completed in all conditions)
common_all, eps_all = common, eps
common = keys('OWN') & keys('SWAP'); eps = sorted({e for _, e in common})
rng = np.random.default_rng(0)
sens_point = {c: measures(groups_for(c, eps)) for c in ['OWN', 'SWAP']}
sens_boot = []
for _ in range(NB):
    s = list(rng.choice(eps, len(eps)))
    sens_boot.append({c: measures(groups_for(c, s)) for c in ['OWN', 'SWAP']})
sensitivity = {'n_decisions': len(common), 'n_episodes': len(eps), 'OWN_minus_SWAP': {
    k: {'diff': sens_point['OWN'][k] - sens_point['SWAP'][k],
        'ci': ci([b['OWN'][k] - b['SWAP'][k] for b in sens_boot])}
    for k in ['target-investor identification', 'decision-linked recall', 'rationale F1']}}
common, eps = common_all, eps_all

empty = collections.Counter(e for res in results.values() for e in res['inputs']['empty_inputs'])
mentions = {f'{c}:{d}': res['inputs']['overlay_mentions_of_pitched_company'] for (c, d), res in results.items()
            if res['inputs']['overlay_mentions_of_pitched_company']}
out = {
    'status': 'PILOT; exploratory', 'conditions_evaluated': conds, 'missing': dict(missing),
    'n_decisions': len(common), 'n_episodes': len(eps), 'bootstrap': {'samples': NB, 'seed': 0, 'unit': 'episode'},
    'threshold_note': "balanced accuracy uses each investor's 70th-percentile cutoff of the MAIN evaluation's weighted "
                      'scores on their other pitches (all pitches if fewer than 8), identical in every condition',
    'metrics': table, 'primary_criteria_met': primary, 'secondary_criteria_met': secondary,
    'primary_overall_met': bool(primary) and all(primary.values()),
    'sensitivity_OWN_vs_SWAP_pairwise': sensitivity,
    'sample': crit['sample'], 'donors': crit['donors'],
    'cost_usd': {c: {'total': round(sum(v), 4), 'per_decision': round(avg(v), 4), 'n': len(v)} for c, v in costs.items()},
    'cost_total_usd': round(sum(sum(v) for v in costs.values()), 4),
    'empty_inputs': dict(empty), 'pitched_company_mentions_in_supplied_evidence': mentions,
    'config': {'model': 'openai/gpt-5.6-luna via OpenRouter', 'run_config': "each decision's main-evaluation run-config.json",
               'components': 'configs/components.toml (coverage, contrastive lens)'},
}
(HERE / 'results.json').write_text(json.dumps(out, indent=2, default=float) + '\n')
fmt = lambda k, x: 'n/a' if x is None or x != x else (f'{x:.2f}' if 'AUC' in k else f'{x:.1f}' if 'per pitch' in k else f'{100*x:.1f}%')
fd = lambda k, x: 'n/a' if x is None or x != x else (f'{x:+.2f}' if 'AUC' in k else f'{x:+.1f}' if 'per pitch' in k else f'{100*x:+.1f}')
print(f"{len(common)} decisions, {len(eps)} episodes; conditions {conds}")
for k in table:
    cells = [fmt(k, table[k]['OWN']['value'])] + [
        f"{fmt(k, table[k][c]['value'])} ({fd(k, table[k][c]['diff_vs_OWN'])}; {fd(k, table[k][c]['diff_ci'][0])} to {fd(k, table[k][c]['diff_ci'][1])})"
        for c in conds[1:]] + [fmt(k, table[k]['MAIN']['value'])]
    print(f'{k:48s} ' + ' | '.join(cells))
print('primary', primary, 'secondary', secondary)
print('sensitivity', json.dumps(sensitivity, default=float))
print('cost', out['cost_usd'], 'total', out['cost_total_usd'])
print('empty inputs', dict(empty)); print('company mentions in supplied evidence', len(mentions))
