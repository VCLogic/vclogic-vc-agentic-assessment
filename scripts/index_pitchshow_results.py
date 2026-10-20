#!/usr/bin/env python3
"""Index preserved legacy and new results without rewriting native artifacts."""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from prepare_pitchshow_evaluation import read,write

VC_MAP={'charles-hudson':'charles-hudson-precursor-ventures',
'cyan-banister':'cyan-banister-long-journey-ventures','elizabeth-yin':'elizabeth-yin-hustle-fund',
'jesse-middleton':'jesse-middleton-flybridge','jillian-manus':'jillian-manus-structure-capital',
'phil-nadel':'phil-nadel-forefront-venture-partners'}
LABELS={'charles-hudson':'charles','cyan-banister':'cyan_banister','elizabeth-yin':'elizabeth_yin',
'jesse-middleton':'jesse_middleton','jillian-manus':'jillian_manus','phil-nadel':'phil_nadel'}
ARCHIVE=Path('outputs/legacy-langgraph/workspace')


def digest(path): return sha256(Path(path).read_bytes()).hexdigest()


def execution_cohort(status_path):
    metadata_path = status_path.parent / 'experiment.json'
    metadata = read(metadata_path) if metadata_path.exists() else {}
    return metadata['experiment'] if metadata.get('evaluation_role') in {'regression', 'validation'} else 'current-pitchshow'


def result_record(path, *, vc, episode, mode, cohort, actual):
    summary=read(path);decision=summary.get('decision') or {};root=path.parent
    if summary.get('episode_slug')!=episode: raise ValueError('result episode mismatch')
    from evaluate_pitchshow_results import provider_metadata
    return {**provider_metadata(root), 'execution_status':'completed', 'schema':'pitchshow-result-v1','vc_slug':vc,'episode_slug':episode,
            'input_mode':mode,'cohort':cohort,'contract_version':summary.get('contract_version'),
            'phase1_status':summary.get('phase1_status'),'phase2_status':summary.get('phase2_status'),
            'predicted_decision':decision.get('decision'),
            'investment_likelihood':decision.get('investment_likelihood'),
            'actual_pitch_decision':actual,'usage':summary.get('usage',{}),
            'native_artifact_root':str(root.resolve()),'summary_sha256':digest(path),
            'phase1_path':str((root/'phase1/investigation.json').resolve()) if (root/'phase1/investigation.json').exists() else None,
            'phase2_path':str((root/'phase2/decision.json').resolve()) if (root/'phase2/decision.json').exists() else None}


def build_catalog():
    root=Path('evaluation/pitchshow');rows=[]
    canonical=ARCHIVE/'outputs/canonical-v4-v41-portfolio-2026-08-15'
    if (canonical/'manifest.json').exists():
        labels={vc:{r['episode_slug']:r['pitch_window_decision'] for r in read(ARCHIVE/f'evaluation/labels/{name}_pitch_window_decisions.json')} for vc,name in LABELS.items()}
        for original in read(canonical/'manifest.json')['episode_rows']:
            vc=original['vc_slug'];slug=original['episode_slug'];artifact=canonical/original['snapshot_artifact_path']
            record=result_record(artifact/'summary.json',vc=VC_MAP[vc],episode=slug,
                mode='legacy-canonical',cohort='legacy-canonical-v4-v41',actual=labels[vc].get(slug))
            provenance=read(artifact/'input-provenance.json')
            original_root=Path('/home/dpasch01/vc-digital-twins/langgraph-vc-clone-framework')
            pitch=ARCHIVE/Path(provenance['pitch_path']).relative_to(original_root)
            if not pitch.exists() or digest(pitch)!=provenance['pitch_sha256']:
                pitch=Path('outputs/legacy-langgraph/recovered-inputs')/vc/f'{slug}.txt'
            matched=pitch.exists() and digest(pitch)==provenance['pitch_sha256']
            record.update(input_hash_verified=matched, pitch_path=str(pitch.resolve()) if matched else None,
                          input_mode_equivalence='legacy input: inspect original speaker content; not assumed equivalent to new mode A')
            if matched: record['has_neutral_investor_questions']='Investor question:' in pitch.read_text()
            rows.append(record)
        panel=ARCHIVE/'outputs/panel-question-phase1-canary-2026-08-20'
        if (panel/'batch-status.json').exists():
            selections={(r['vc_slug'],r['episode_slug']):r for r in read(panel/'selection.json')['cases']}
            for case in read(panel/'batch-status.json')['cases'].values():
                completed=[a for a in case['attempts'] if a['status']=='completed']
                if not completed:continue
                attempt=completed[-1];artifact=ARCHIVE/attempt['artifact_root']
                vc=next(k for k,v in VC_MAP.items() if v==case['vc_slug'] or k==case['vc_slug'] or (k=='phil-nadel' and case['vc_slug']=='phil-nadel'))
                row=result_record(artifact/'summary.json',vc=VC_MAP[vc],episode=case['episode_slug'],
                    mode='legacy-panel-question-augmentation',cohort='legacy-panel-question-phase1',actual=labels[vc].get(case['episode_slug']))
                selection=selections[(case['vc_slug'],case['episode_slug'])]
                pitch=ARCHIVE/selection['pitch_path']
                if digest(pitch)!=selection['pitch_sha256']: raise ValueError('legacy panel input hash mismatch')
                row.update(input_hash_verified=True,pitch_path=str(pitch.resolve()),
                           selected_attempt=attempt['attempt'],input_mode_equivalence='other-investor question augmentation; older boundaries and input text',
                           full_decision_comparison_available=False)
                rows.append(row)
    for status_path in Path('outputs/pitchshow-evaluation/batch/jobs').glob('*/status.json'):
        state=read(status_path)
        if state['status']!='completed':continue
        summary=Path(state['summary_path'])
        if digest(summary)!=state['summary_sha256']:raise ValueError('new result summary changed')
        annotations={r['episode_slug']:r for r in read(root/'investors'/state['vc_slug']/'annotations.json')}
        row=result_record(summary,vc=state['vc_slug'],episode=state['episode_slug'],mode=state['mode'],
            cohort=execution_cohort(status_path),actual=annotations[state['episode_slug']]['pitch_window_decision'])
        row.update(job_id=state['job_id'],selected_attempt=state['selected_attempt'],input_hash_verified=True,
                   input_mode_equivalence='current speaker-filtered dataset',label_provenance='machine-reviewed-current-annotations')
        rows.append(row)
    rows.sort(key=lambda r:(r['vc_slug'],r['episode_slug'],r['cohort'],r['input_mode'],r.get('job_id','')))
    write(root/'results/catalog.json',{'schema':'pitchshow-result-catalog-v1','results':rows})
    for vc in sorted({r['vc_slug'] for r in rows}):
        write(root/'investors'/vc/'results.json',{'schema':'pitchshow-result-catalog-v1','results':[r for r in rows if r['vc_slug']==vc]})
    counts=Counter(r['cohort'] for r in rows)
    write(root/'results/summary.json',{'counts':dict(counts),'total':len(rows),
        'warning':'Legacy cohorts have different input boundaries, memory snapshots and stage coverage. Do not pool as a controlled A/B benchmark.'})
    from evaluate_pitchshow_results import build_report
    build_report(root, rows=rows)
    print(json.dumps(dict(counts)))
    return rows


if __name__=='__main__': build_catalog()
