"""Draw the pre-registered pilot sample and write criteria.json. No model calls.

Run from the repo root: .venv/bin/python experiments/memory_swap_pilot/sample.py
"""
import json
from pathlib import Path

exec(open('outputs/free-tests/_common.py').read())  # main-evaluation loader: `rows` = the 400 decisions

HERE = Path('experiments/memory_swap_pilot')
SPLIT = open('outputs/components-lens/split-episodes.txt').read().split()
by_ep = collections.defaultdict(list)
for r in rows:
    by_ep[r['ep']].append(r)
assert len(SPLIT) == 57 and all(0 < sum(r['y'] for r in by_ep[e]) < len(by_ep[e]) for e in SPLIT)

rng = np.random.default_rng(0)
episodes = sorted(rng.choice(SPLIT, 20, replace=False).tolist())
decisions, donors = [], {}
donor_rng = np.random.default_rng(0)
for ep in episodes:
    panel = sorted(r['vc'] for r in by_ep[ep])
    for r in sorted(by_ep[ep], key=lambda r: r['vc']):
        decisions.append({'decision_id': f"{r['vc']}/{ep}", 'vc_slug': r['vc'], 'episode_slug': ep,
                          'actual': 'In' if r['y'] else 'Out', 'has_reference_rationales': r['ms'] is not None})
        others = [v for v in panel if v != r['vc']]
        donors[f"{r['vc']}/{ep}"] = others[0] if len(others) == 1 else str(donor_rng.choice(others))

criteria = {
    'schema': 'memory-swap-pilot-criteria-v1',
    'status': 'PILOT. Pre-registered before any model call. All results are exploratory.',
    'research_question': ("Does giving a twin the correct investor's evidence make its assessment closer to that "
                          "investor than giving it another investor's evidence, or none?"),
    'sample': {
        'frame': '57 split panels of the main evaluation (outputs/components-lens/split-episodes.txt): '
                 'panels where investors disagree, from the 400-decision main evaluation',
        'draw': 'numpy.random.default_rng(0).choice(split_panels, 20, replace=False), sorted',
        'episodes': episodes,
        'decisions': decisions,
        'n_decisions': len(decisions),
        'n_with_reference_rationales': sum(d['has_reference_rationales'] for d in decisions),
    },
    'conditions': {
        'OWN': "Target investor's name + target's own evidence. Rerun with the pilot code; old outputs not reused.",
        'SWAP': "Target investor's name + evidence of another investor on the same panel (donor). Donor: the only "
                "other panelist, or numpy.random.default_rng(0).choice over the other panelists (sorted), drawn in "
                "decision order.",
        'NAME_ONLY': "Target investor's name, no investor-specific evidence.",
        'GENERIC': "No name ('an early-stage venture investor'), no investor-specific evidence.",
        'OWN_ANON': 'Optional; not run unless stated in results.',
    },
    'donors': donors,
    'investor_specific_evidence': [
        'Investment Memory (wiki and its index), used by the investigator, coverage, decision agent and lens',
        "prior cases and decisions from the investor's other episodes (precedent corpus and index)",
        "on-show rationale profile used by the investor lens (from the investor's other episodes)",
        "panel-mates' profiles used by the contrastive lens (OWN: all other panelists; SWAP: panelists other than "
        'target and donor; NAME_ONLY/GENERIC: none)',
        'uniqueness pages: not used by the presented pipeline, so not used in any condition',
    ],
    'leakage_rules': 'In every condition, donor evidence included: the current episode is excluded from prior cases, '
                     'Investment Memory sections mentioning the pitched company are filtered, and the pitch is the '
                     "main evaluation's founders-only pitch cut at the first panel decision.",
    'measures': {
        'rationales': 'investor lens output; recall/precision/F1 at dimension, rationale, rationale+direction level',
        'decision_linked_recall': "recall of rationales the investor links to the decision",
        'investor_fit_recall': 'recall of investor-fit rationales',
        'target_investor_identification': "share of ordered pairs on a panel where the twin's rationales are closer "
                                          '(F1) to its own investor than to the other panelist; chance 50%',
        'overlap_between_twins': 'F1 between twins on the same pitch, compared with investors (main-evaluation definition)',
        'within_panel_auc': 'AUC of the weighted score (investigator + coverage rationales) within panels',
        'balanced_accuracy': "weighted score, In if above the investor's 70th percentile of the MAIN evaluation's "
                             'scores on their other pitches (all pitches if fewer than 8)',
        'code': 'metric definitions copied from outputs/final-solution/report.py',
    },
    'primary': {
        'criteria': ['OWN - SWAP on target-investor identification > 0, 95% CI excludes 0',
                     'OWN - SWAP on decision-linked recall > 0, 95% CI excludes 0'],
        'met_if': 'both criteria are met',
    },
    'secondary': ['OWN vs NAME_ONLY and OWN vs GENERIC on target-investor identification and decision-linked recall',
                  'rationale F1 at all three levels', 'within-panel AUC of the weighted score',
                  'rationale overlap between twins on the same pitch'],
    'inference': 'paired differences vs OWN; 2,000 bootstrap samples, seed 0, resampling episodes; 95% percentile CIs',
    'model': 'openai/gpt-5.6-luna via OpenRouter, configs/investors/<vc>/canonical.toml (contract v4.1); coverage and '
             'contrastive lens with configs/components.toml defaults',
    'budget': 'dry run on 2 decisions x 4 conditions; stop if projected total exceeds USD 60',
}
(HERE / 'criteria.json').write_text(json.dumps(criteria, indent=2) + '\n')
print(len(decisions), 'decisions;', sum(d['has_reference_rationales'] for d in decisions), 'with references')
