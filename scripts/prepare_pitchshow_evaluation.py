#!/usr/bin/env python3
"""Review downloaded Pitch Show episodes and materialize per-VC evaluation inputs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import re

from vc_clone_graph.providers.openrouter import OpenRouterProvider
from pitchshow_evaluation_review import fingerprint, review_episode


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(path)


def gather(root):
    episodes={};targets={};locations={}
    for folder in sorted((root/'investors').iterdir()):
        profile_path=folder/'source/profile.json'
        if not profile_path.exists(): continue
        profile=read(profile_path)
        name=profile['name']; aliases=list(dict.fromkeys([name,name.split()[0]]))
        for path in sorted((folder/'source/episodes').glob('*.json')):
            episode=read(path);slug=episode['slug']
            if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',slug): raise ValueError('unsafe episode slug')
            for panel in episode.get('panel') or []:
                if panel.get('slug')==profile['id'] and panel.get('name'):
                    aliases.extend([panel['name'],panel['name'].split()[0]])
            if slug in episodes and episodes[slug].get('transcript')!=episode.get('transcript'):
                raise ValueError(f'conflicting transcripts across VCs: {slug}')
            episodes[slug]=episode
            targets.setdefault(slug,[]).append({'vc_slug':folder.name,'name':name,'aliases':sorted(set(aliases))})
            locations.setdefault(slug,[]).append((folder,path))
    return episodes,targets,locations


LEGACY={'charles-hudson-precursor-ventures':'charles','cyan-banister-long-journey-ventures':'cyan_banister',
'elizabeth-yin-hustle-fund':'elizabeth_yin','jesse-middleton-flybridge':'jesse_middleton',
'jillian-manus-structure-capital':'jillian_manus','phil-nadel-forefront-venture-partners':'phil_nadel'}


def publish(root,episodes,targets,locations,results):
    legacy={}
    for vc,prefix in LEGACY.items():
        path=Path('evaluation/labels')/f'{prefix}_pitch_window_decisions.json'
        if path.exists(): legacy[vc]={r['episode_slug']:r for r in read(path)}
    ledgers={};counts={}
    for number,slug in enumerate(sorted(episodes),1):
        episode=episodes[slug];neutral='pitch-'+sha256(slug.encode()).hexdigest()[:16]
        result=results.get(slug)
        if result:
            result_by_vc={r['vc_slug']:r for r in result['annotations']}
        else: result_by_vc={}
        for folder,path in locations[slug]:
            transcript=episode.get('transcript') or ''
            if transcript:
                dest=folder/'transcripts'/f'{slug}.txt';dest.parent.mkdir(exist_ok=True);dest.write_text(transcript)
            row=dict(result_by_vc.get(folder.name,{'pitch_window_decision':'Unobserved','later_decision':'Unknown',
                'evaluation_eligible':False,'review_status':'missing_transcript' if not transcript else 'review_failed_or_pending'}))
            row.update(episode_slug=slug,neutral_pitch_id=neutral,vc_slug=folder.name,
                       source_url=episode.get('url'),source_sha256=sha256(path.read_bytes()).hexdigest(),
                       source_path=f'source/episodes/{slug}.json',
                       transcript_sha256=sha256(transcript.encode()).hexdigest())
            reported={r['episode_slug']:r for r in read(folder/'source/decisions.json')}
            row['reported_profile_investment']=reported.get(slug,{}).get('reported_decision','Unknown')
            old=legacy.get(folder.name,{}).get(slug)
            if old:
                row['prior_audit']={k:old.get(k) for k in ('pitch_window_decision','final_decision','decision_context','audit_notes','evidence_quote')}
                row['prior_audit']['source']=f'evaluation/labels/{LEGACY[folder.name]}_pitch_window_decisions.json'
                previous=old.get('pitch_window_decision')
                row['prior_audit_conflict']=previous in {'In','Out','Unobserved'} and row['pitch_window_decision'] in {'In','Out'} and previous!=row['pitch_window_decision']
                if row['prior_audit_conflict']:
                    row['evaluation_eligible']=False
                    row['review_status']='conflicts_with_prior_audit'
            eligible=bool(row['evaluation_eligible'] and result and result['pitch_safe'])
            row['evaluation_eligible']=eligible
            row['episode_kind']=result['review']['episode_kind'] if result else 'unknown'
            row['exclusion_reason']=(None if eligible else
                'missing_transcript' if not transcript else
                'review_failed_or_pending' if not result else
                'prior_audit_conflict' if row.get('prior_audit_conflict') else
                'not_an_original_pitch' if result['review']['episode_kind']!='pitch' else
                'unsafe_or_empty_pitch' if not result['pitch_safe'] else 'decision_unobserved_or_disputed')
            row['pitch_review_status']='verified_redacted' if eligible else 'not_published'
            if result and not result['pitch_safe'] and row['review_status']=='machine_reviewed_and_verified':
                row['review_status']='decision_verified_pitch_not_safe'
            pitch_path=folder/'pitches'/f'{neutral}.txt'
            if eligible:
                pitch_path.parent.mkdir(exist_ok=True);pitch_path.write_text(result['pitch'])
                row['pitch_path']=f'pitches/{neutral}.txt';row['pitch_sha256']=sha256(pitch_path.read_bytes()).hexdigest()
            elif pitch_path.exists(): pitch_path.unlink()
            if result:
                write(folder/'audits'/f'{neutral}.json',{'episode_slug':slug,'vc_slug':folder.name,
                    'review_status':row['review_status'],'evaluation_eligible':bool(eligible),
                    'source_sha256':row['source_sha256'],'pitch_sha256':row.get('pitch_sha256'),
                    'review_receipt':f'../../../reviews/{slug}.json','boundary':{k:result['review'][k] for k in ('pitch_start_turn','pitch_end_turn','boundary_reason')},
                    'segments':result['segments'],'removed_candidate_lines':result['removed_candidate_lines'],
                    'semantic_review':result['verification']})
            ledgers.setdefault(folder.name,[]).append(row)
    for vc,rows in ledgers.items():
        folder=root/'investors'/vc;write(folder/'annotations.json',rows)
        expected_pitches={r['neutral_pitch_id'] for r in rows if r['evaluation_eligible']}
        expected_audits={r['neutral_pitch_id'] for r in rows if r['episode_slug'] in results}
        for kind, expected in [('pitches',expected_pitches),('audits',expected_audits)]:
            for old in (folder/kind).glob('pitch-*'):
                if old.stem not in expected:
                    old.unlink()
        collection=read(folder/'source/collection.json')
        counts[vc]={'episodes':len(rows),'available_transcripts':sum(bool(episodes[r['episode_slug']].get('transcript')) for r in rows),
            'eligible_pitches':sum(bool(r['evaluation_eligible']) for r in rows),
            'pitch_decisions':dict(Counter(r['pitch_window_decision'] for r in rows)),
            'later_decisions':dict(Counter(r['later_decision'] for r in rows)),
            'prior_audit_conflicts':sum(r.get('prior_audit_conflict',False) for r in rows),
            'collection_complete':collection.get('complete'), 'missing_transcripts':collection.get('missing_transcripts',[])}
        write(folder/'summary.json',counts[vc])
    summary={'schema':'pitchshow-evaluation-dataset-v1','generated_at':datetime.now(timezone.utc).isoformat(),'investors':counts,'unique_episodes':len(episodes),
             'reviewed_unique_episodes':len(results),'human_reviewed':False,
             'review_complete':all(not e.get('transcript') or slug in results for slug,e in episodes.items()),
             'unreviewed_available_transcripts':[slug for slug,e in episodes.items() if e.get('transcript') and slug not in results],
             'evaluation_target':'pitch_window_decision','later_outcomes_are_separate':True}
    write(root/'summary.json',summary)
    lines=['# Pitch Show Evaluation Coverage', '',
           'Machine-reviewed annotations; eligible pitches passed the independent model and deterministic checks.', '',
           '| Investor | Episodes | Transcripts | Pitch In | Pitch Out | Unobserved | Eligible Pitches | Prior Audit Conflicts |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for vc,row in counts.items():
        c=row['pitch_decisions']
        lines.append(f"| {vc} | {row['episodes']} | {row['available_transcripts']} | {c.get('In',0)} | {c.get('Out',0)} | {c.get('Unobserved',0)} | {row['eligible_pitches']} | {row['prior_audit_conflicts']} |")
    lines.extend(['',f"Unique episodes: {len(episodes)}. Completed model reviews: {len(results)}.",
                  'Missing transcripts and non-pitch content remain documented but are not scored.',
                  'Decisions shown include cases withheld from evaluation; use evaluation_eligible to select scored rows.'])
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('evaluation/pitchshow'))
    parser.add_argument('--model',default='openai/gpt-5.6-luna')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--publish-only',action='store_true')
    args=parser.parse_args()
    episodes,targets,locations=gather(args.root)
    results={};pending=[];errors={}
    for slug,episode in episodes.items():
        path=args.root/'reviews'/f'{slug}.json'
        if path.exists():
            saved=read(path)
            if saved.get('fingerprint')==fingerprint(episode,targets[slug]) and saved.get('model')==args.model:
                results[slug]=saved;continue
        if episode.get('transcript'): pending.append(slug)
    if args.limit is not None: pending=pending[:args.limit]
    print(f'{len(episodes)} unique episodes; {len(results)} cached reviews; {len(pending)} to review',flush=True)
    def work(slug):
        provider=OpenRouterProvider(args.model,max_output_tokens=16384)
        try:
            old_path=args.root/'reviews'/f'{slug}.json'
            previous=read(old_path) if old_path.exists() else None
            return review_episode(episodes[slug],targets[slug],provider,previous=previous)
        finally: provider.client.close()
    if not args.publish_only:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(work,slug):slug for slug in pending}
            for future in as_completed(futures):
                slug=futures[future]
                try:
                    result=future.result();write(args.root/'reviews'/f'{slug}.json',result);results[slug]=result
                    print(f'REVIEWED {slug} safe={result["pitch_safe"]} eligible={sum(r["evaluation_eligible"] for r in result["annotations"])}',flush=True)
                except Exception as exc:
                    errors[slug]=f'{type(exc).__name__}: {exc}';print(f'FAILED {slug}: {type(exc).__name__}',flush=True)
                write(args.root/'review-progress.json',{'completed':len(results),'errors':errors,'total':len(episodes)})
    summary=publish(args.root,episodes,targets,locations,results)
    print(json.dumps({'investors':len(summary['investors']),'unique_episodes':len(episodes),
                      'reviewed':len(results),'eligible_pitches':sum(v['eligible_pitches'] for v in summary['investors'].values()),'errors':errors},indent=2))
    return 1 if errors else 0


if __name__=='__main__':
    raise SystemExit(main())
