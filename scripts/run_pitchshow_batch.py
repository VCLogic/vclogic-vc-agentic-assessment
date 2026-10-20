#!/usr/bin/env python3
"""Plan or execute resumable Pitch Show assessments; default is a no-call dry run."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

from vc_clone_graph.cli import command_preflight, command_run, command_verify
from vc_clone_graph.config import load_config
from prepare_pitchshow_evaluation import gather, read, write
from prepare_pitchshow_variants import MODES, SUPPORTED_MODES, package_case, render_variant
from pitchshow_evaluation_review import fingerprint
from pitchshow_company_aliases import validate_aliases


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()



def asset_signature(workspace, vc):
    config_path = workspace / f'configs/investors/{vc}/canonical.toml'
    config = load_config(config_path)
    if config.run.contract_version == 'v4.4':
        raise ValueError('batch decisions require a full Phase 2 contract, not v4.4')
    inputs = workspace / 'inputs'
    if config.resolve_path(config.run.input_root).resolve() != inputs.resolve():
        raise ValueError(f'canonical template does not select installed inputs: {vc}')
    paths = [config_path, inputs/f'investors/{vc}.toml', inputs/'taxonomy/codebook_v_final.json',
             inputs/f'indexes/{vc}.json']
    wiki = inputs/'wiki'/vc
    if not wiki.is_dir(): raise ValueError(f'missing wiki: {vc}')
    paths.extend(p for p in wiki.rglob('*') if p.is_file())
    if config.precedents.enabled:
        paths.extend([inputs/f'indexes/{vc}.precedents.json',
                      inputs/f'data/investors/{vc}/precedents/corpus-manifest.json'])
    if config.portfolio_memory.enabled:
        paths.extend(p for p in (inputs/f'data/investors/{vc}/portfolio-memory').rglob('*') if p.is_file())
    # Code changes get a fresh execution identity rather than reusing old predictions.
    paths.extend((workspace/'src/vc_clone_graph').rglob('*.py'))
    paths.extend(workspace/'scripts'/name for name in ('run_pitchshow_batch.py','prepare_pitchshow_variants.py','pitchshow_company_aliases.py'))
    return identity({str(p.relative_to(workspace)): digest(p) for p in sorted(set(paths))})


def plan(root, workspace, modes, vcs=(), episodes_filter=()):
    episodes, targets, _ = gather(root)
    selected = []
    all_vcs = set()
    for path in sorted((root/'investors').glob('*/annotations.json')):
        vc = path.parent.name; all_vcs.add(vc)
        if vcs and vc not in vcs: continue
        for row in read(path):
            if row['evaluation_eligible'] and (not episodes_filter or row['episode_slug'] in episodes_filter):
                selected.append((vc, row['episode_slug']))
    if set(vcs) - all_vcs: raise ValueError('unknown VC filter')
    if set(episodes_filter) - {s for _,s in selected}: raise ValueError('episode filter is not eligible for selected VCs')
    if not selected: raise ValueError('no eligible cases selected')
    mapping = read(root/'company-aliases.json')
    aliases = validate_aliases(root, mapping, {s for _,s in selected})
    signatures = {vc: asset_signature(workspace,vc) for vc in sorted({v for v,_ in selected})}
    jobs = []
    for vc, slug in selected:
        receipt_path=root/'reviews'/f'{slug}.json'; receipt=read(receipt_path)
        if receipt['fingerprint'] != fingerprint(episodes[slug],targets[slug]):
            raise ValueError(f'stale review: {slug}')
        target=next(t for t in targets[slug] if t['vc_slug']==vc)
        for mode in modes:
            case=root/'investors'/vc/'cases'/slug/mode
            text, _ = render_variant(receipt,target['aliases'],mode)
            if (case/'pitch.txt').read_text()!=text or digest(case/'pitch.txt')!=read(case/'audit.json')['pitch_sha256']:
                raise ValueError(f'stale pitch variant: {vc}/{slug}/{mode}')
            job={'vc_slug':vc,'episode_slug':slug,'mode':mode,'aliases':aliases[slug],
                 'pitch_sha256':digest(case/'pitch.txt'),'review_sha256':digest(receipt_path),
                 'asset_signature':signatures[vc], 'alias_record_sha256':identity(mapping['episodes'][slug])}
            job['job_id']=identity(job)
            jobs.append(job)
    return jobs


def completed_summary(config):
    command_verify(config)
    path=config.resolve_path(config.run.output_root)/config.run.episode_slug/'summary.json'
    summary=read(path)
    if not isinstance(summary.get('decision'),dict) or summary['decision'].get('decision') not in {'In','Out'}:
        raise ValueError('assessment has no completed In/Out decision')
    return summary, path


def run_job(job, status_path, prepare):
    state = read(status_path) if status_path.exists() else dict(job, status='pending', attempts=[])
    if state['job_id'] != job['job_id']:
        raise ValueError('batch job identity mismatch')
    if state['status']=='completed':
        try:
            config=prepare(job,state['selected_attempt'])
            summary,path=completed_summary(config)
            if digest(path)!=state['summary_sha256']:
                raise ValueError('completed summary changed')
            return state
        except Exception as exc:
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            write(status_path,state)
            return state
    number=len(state['attempts'])+1
    attempt={'number':number,'status':'running','started_at':datetime.now(timezone.utc).isoformat()}
    state['attempts'].append(attempt);state.update(status='running');state.pop('error',None)
    write(status_path,state)
    try:
        config=prepare(job,number)
        log=status_path.parent/f'attempt-{number}.log'
        with log.open('w') as stream, redirect_stdout(stream), redirect_stderr(stream):
            command_run(config)
            summary,path=completed_summary(config)
        attempt.update(status='completed',summary_path=str(path),usage=summary.get('usage',{}))
        state.update(status='completed',selected_attempt=number,summary_path=str(path),summary_sha256=digest(path))
    except Exception as exc:
        attempt.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        state.update(status='failed',error=attempt['error'])
    write(status_path,state)
    return state


def prepare_attempt(root,workspace,job,attempt):
    if asset_signature(workspace,job['vc_slug']) != job['asset_signature']:
        raise ValueError('installed assets or code changed since planning; rerun the batch command')
    config=package_case(root,workspace,job['vc_slug'],job['episode_slug'],job['mode'],job['aliases'],
                        execution_key=job['job_id'])
    session=config.resolve_path(config.run.output_root).parent
    payload=config.model_dump(mode='json')
    if job.get('backend')=='codex':
        payload['provider'].update(kind='codex',model=payload['provider']['model'].removeprefix('openai/'),base_url=None,api_key_env=None)
        for phase in ('phase1','phase2'):
            if payload[phase].get('model'):
                payload[phase]['model']=payload[phase]['model'].removeprefix('openai/')
    payload['run'].update(output_root=str((session/f'attempt-{attempt}/runs').relative_to(workspace)),
                          checkpoint_path=str((session/f'attempt-{attempt}/checkpoints.sqlite').relative_to(workspace)))
    from vc_clone_graph.config import RunConfig
    config=RunConfig.model_validate(payload);config._workspace=workspace
    path=session/f'attempt-{attempt}/run-config.json';write(path,payload)
    with redirect_stdout(StringIO()) as output:
        command_preflight(config,path)
    report=json.loads(output.getvalue())
    write(path.parent/'preflight.json',report)
    if report['status']!='ready' or report.get('target_accessible'):
        raise ValueError('assessment preflight is not ready or exposes target')
    return config


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('evaluation/pitchshow'))
    parser.add_argument('--mode',choices=(*SUPPORTED_MODES,'both'),default='founders-only')
    parser.add_argument('--vc',action='append',default=[])
    parser.add_argument('--episode',action='append',default=[])
    parser.add_argument('--limit',type=int)
    parser.add_argument('--backend',choices=['configured','codex'],default='configured',help='Use configured provider or Codex ChatGPT login with the same models')
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--execute',action='store_true',help='Authorize paid assessments')
    action.add_argument('--preflight',action='store_true',help='Package and validate selected cases without generation calls')
    args=parser.parse_args(argv)
    if args.limit is not None and args.limit<1: parser.error('--limit must be positive')
    workspace=Path.cwd().resolve(); root=args.root.resolve()
    modes=MODES if args.mode=='both' else (args.mode,)
    jobs=plan(root,workspace,modes,args.vc,args.episode)
    if args.limit: jobs=jobs[:args.limit]
    if args.backend=='codex':
        for job in jobs:
            job.pop('job_id')
            job['backend']='codex'
            job['job_id']=identity(job)
    output=workspace/'outputs/pitchshow-evaluation/batch';output.mkdir(parents=True,exist_ok=True)
    with (output/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=output/f'plan-{identity(jobs)[:16]}.json'
        write(manifest,{'schema':'pitchshow-batch-plan-v1','jobs':jobs})
        print(f'{len(jobs)} jobs; plan: {manifest}',flush=True)
        if not args.execute and not args.preflight:
            print('Dry run: aliases, variants, templates and asset fingerprints validated; no assessments executed.')
            return 0
        counts=Counter()
        for job in jobs:
            status=output/'jobs'/job['job_id']/'status.json'
            if args.preflight:
                try:
                    prepare_attempt(root,workspace,job,1);counts['ready']+=1
                except Exception as exc:
                    counts['failed']+=1
                    write(status.parent/'preflight-error.json',{'error':f'{type(exc).__name__}: {exc}'})
                    print(f'FAILED {job["vc_slug"]}/{job["episode_slug"]}: {exc}',flush=True)
            else:
                result=run_job(job,status,lambda j,n: prepare_attempt(root,workspace,j,n))
                if result['status']=='completed':
                    from index_pitchshow_results import result_record
                    from evaluate_pitchshow_results import evaluate_record
                    annotations={r['episode_slug']:r for r in read(root/'investors'/job['vc_slug']/'annotations.json')}
                    row=result_record(Path(result['summary_path']),vc=job['vc_slug'],episode=job['episode_slug'],
                        mode=job['mode'],cohort='current-pitchshow',
                        actual=annotations[job['episode_slug']]['pitch_window_decision'])
                    row.update(job_id=job['job_id'],selected_attempt=result['selected_attempt'])
                    evaluation=evaluate_record(row,root)
                    write(status.parent/f"evaluation-{identity(evaluation)[:16]}.json",evaluation)
                counts[result['status']]+=1
                print(f'{result["status"]}: {job["vc_slug"]}/{job["episode_slug"]}/{job["mode"]}',flush=True)
        write(output/'latest-summary.json',{'plan':str(manifest),'counts':dict(counts),
              'action':'execute' if args.execute else 'preflight'})
        if args.execute:
            from index_pitchshow_results import build_catalog
            build_catalog()
        print(json.dumps(dict(counts)))
        return 1 if counts['failed'] else 0


if __name__=='__main__':
    raise SystemExit(main())
