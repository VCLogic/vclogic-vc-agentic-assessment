"""Memory-swap pilot runner. Reuses the main pipeline unchanged; only the evidence it is given changes.

``evidence_source`` (per condition) decides what investor-specific evidence every step receives:
  own          the target's Investment Memory, prior cases and on-show profile
  donor        the same three blocks from another investor on the same panel (SWAP)
  none         an explicit placeholder memory, no prior cases, no profile (NAME_ONLY, GENERIC)
The name the twin acts as is set separately (``name``: target | generic). All evidence moves together.

Usage (repo root):
  .venv/bin/python experiments/memory_swap_pilot/run_pilot.py prepare
  .venv/bin/python experiments/memory_swap_pilot/run_pilot.py run --decisions ID,ID [--conditions ...] [--workers N] --execute
Outputs go to outputs/memory-swap-pilot/ (git-ignored, like the main evaluation's run outputs).
Nothing under inputs/, evaluation/ or rationale_annotation/ is written: every overlay is a fresh copy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 16 GPU processes crashed the driver (NVML assert); CPU is safe
WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / 'src'))
HERE = WORKSPACE / 'experiments/memory_swap_pilot'
OUT = WORKSPACE / 'outputs/memory-swap-pilot'
INPUTS = WORKSPACE / 'inputs'
ANNOTATIONS = WORKSPACE / 'rationale_annotation/annotations'
CONDITIONS = {
    'OWN': {'evidence_source': 'own', 'name': 'target'},
    'SWAP': {'evidence_source': 'donor', 'name': 'target'},
    'NAME_ONLY': {'evidence_source': 'none', 'name': 'target'},
    'GENERIC': {'evidence_source': 'none', 'name': 'generic'},
}
GENERIC_NAME = 'an early-stage venture investor'
PLACEHOLDER = ('# No investor-specific evidence\n\nNo investor-specific evidence is provided for this assessment. '
               'Assess the pitch as a typical early-stage venture investor would.\n')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def criteria():
    return read(HERE / 'criteria.json')


def main_runs():
    from vc_clone_graph.components.cli import _latest_runs, _read_evaluation
    ev = _read_evaluation(WORKSPACE / 'evaluation/pitchshow/results/automated-evaluation/latest.json')
    return _latest_runs(ev, 'openai/gpt-5.6-luna')


def evidence_owner(cond, vc, episode, donors):
    source = CONDITIONS[cond]['evidence_source']
    return vc if source == 'own' else donors[f'{vc}/{episode}'] if source == 'donor' else None


def overlay_dir(cond, vc, owner):
    return OUT / 'overlays' / cond / (vc if owner in (None, vc) else f'{vc}__{owner}')


def copy_tree(src, dst):
    """Real copies (never hard links), so nothing downstream can write into inputs/."""
    if src.exists():
        shutil.copytree(src, dst, copy_function=shutil.copy2)


def build_overlay(cond, vc, owner, embedder):
    """A trusted source tree for one (condition, target investor, evidence owner)."""
    dst = overlay_dir(cond, vc, owner)
    if (dst / 'overlay.json').is_file():
        return dst
    tmp = dst.with_name(dst.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    registry = tomllib.loads((INPUTS / f'investors/{vc}.toml').read_text(encoding='utf-8'))
    text = (INPUTS / f'investors/{vc}.toml').read_text(encoding='utf-8')
    if CONDITIONS[cond]['name'] == 'generic':
        text = re.sub(r'^display_name = .*$', f'display_name = "{GENERIC_NAME}"', text, flags=re.M)
        text = re.sub(r'^firm = .*$', 'firm = ""', text, flags=re.M)
        text = re.sub(r'^speaker_names = .*$', 'speaker_names = []', text, flags=re.M)
    (tmp / 'investors').mkdir(parents=True)
    (tmp / f'investors/{vc}.toml').write_text(text, encoding='utf-8')
    (tmp / 'taxonomy').mkdir()
    shutil.copy2(INPUTS / 'taxonomy/codebook_v_final.json', tmp / 'taxonomy/codebook_v_final.json')
    (tmp / 'indexes').mkdir()
    if owner is None:
        wiki = tmp / f'wiki/{vc}'
        wiki.mkdir(parents=True)
        (wiki / 'persona.md').write_text(PLACEHOLDER, encoding='utf-8')
        from vc_clone_graph.retrieval import HybridWikiIndex
        HybridWikiIndex.build(wiki, embedder, require_complete_embeddings=True).save(tmp / f'indexes/{vc}.json')
    else:
        copy_tree(INPUTS / f'wiki/{owner}', tmp / f'wiki/{vc}')
        shutil.copy2(INPUTS / f'indexes/{owner}.json', tmp / f'indexes/{vc}.json')
        shutil.copy2(INPUTS / f'indexes/{owner}.precedents.json', tmp / f'indexes/{vc}.precedents.json')
        copy_tree(INPUTS / f'data/investors/{owner}/precedents', tmp / f'data/investors/{vc}/precedents')
    files = {p.relative_to(tmp).as_posix(): digest(p) for p in sorted(tmp.rglob('*')) if p.is_file()}
    write(tmp / 'overlay.json', {'condition': cond, 'target': vc, 'evidence_owner': owner,
                                 'display_name': GENERIC_NAME if CONDITIONS[cond]['name'] == 'generic'
                                 else registry['display_name'], 'files': files})
    tmp.rename(dst)
    return dst


def annotation_root(cond, vc, owner, episode):
    """Profile source for the lens. OWN: the real annotations. SWAP: target and donor folders swapped,
    so the target slot holds the donor's other episodes and population counts are unchanged.
    none: the target folder is absent, so the target has no profile."""
    if cond == 'OWN':
        return ANNOTATIONS
    root = OUT / 'annotations' / cond / f'{vc}__{owner or "none"}'
    if not root.is_dir():
        tmp = root.with_name(root.name + '.tmp')
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for folder in sorted(p for p in ANNOTATIONS.iterdir() if p.is_dir()):
            name = folder.name
            if owner is None and name == vc:
                continue
            if owner is not None and name in (vc, owner):
                folder = ANNOTATIONS / (owner if name == vc else vc)
            (tmp / name).symlink_to(folder, target_is_directory=True)
        tmp.rename(root)
    return root


def panel_for(cond, vc, owner, episode, sample):
    """(slug, name) of panel-mates whose profiles the contrastive lens contrasts against."""
    if CONDITIONS[cond]['evidence_source'] == 'none':
        return []
    mates = sorted(d['vc_slug'] for d in sample if d['episode_slug'] == episode and d['vc_slug'] not in (vc, owner))
    return [(m, tomllib.loads((INPUTS / f'investors/{m}.toml').read_text())['display_name']) for m in mates]


def alias_hits(directory, aliases):
    """Memory and prior-case files in the overlay that mention the pitched company (filtered at run time)."""
    hits = {}
    pattern = re.compile('|'.join(rf'\b{re.escape(a)}\b' for a in aliases), re.I) if aliases else None
    for p in sorted(directory.rglob('*')):
        if pattern and p.is_file() and p.suffix in ('.md', '.json', '.jsonl') and 'indexes' not in p.parts:
            n = len(pattern.findall(p.read_text(encoding='utf-8', errors='ignore')))
            if n:
                hits[p.relative_to(directory).as_posix()] = n
    return hits


def session_state(session):
    """fresh: nothing paid yet. interrupted: checkpoint exists, pipeline unfinished (resume it).
    pipeline_done: In/Out decision saved (reuse it). gave_up: the pipeline itself failed (do not pay again)."""
    summaries = list(session.glob('runs/*/summary.json'))
    if summaries:
        summary = read(summaries[0])
        if (summary.get('decision') or {}).get('decision') in ('In', 'Out'):
            return 'pipeline_done'
        if 'failed' in (summary.get('phase1_status'), summary.get('phase2_status')):
            return 'gave_up'
    if (session / 'checkpoints.sqlite').is_file() and (session / 'run-config.json').is_file():
        return 'interrupted'
    return 'fresh'


def spent(session):
    """Measured spend of a session, finished or not."""
    if (session / 'result.json').is_file():
        return read(session / 'result.json')['cost_total_usd']
    total = 0.0
    for f in session.glob('runs/*/summary.json'):
        total += read(f).get('usage', {}).get('cost_usd', 0.0)
    for f in session.glob('runs/*/components/*.json'):
        total += read(f).get('usage', {}).get('cost_usd', 0.0)
    return total


def run_job(cond, decision_id):
    from vc_clone_graph.cli import command_preflight, command_run, command_verify
    from vc_clone_graph.components.config import CoverageSettings, load_components_config
    from vc_clone_graph.components.coverage import openrouter_provider, run_coverage
    from vc_clone_graph.components.lens import run_lens
    from vc_clone_graph.components.run_inputs import load_run
    from vc_clone_graph.config import RunConfig
    from vc_clone_graph.firewall import verify_package
    from vc_clone_graph.manifest_builder import build_episode_manifest
    from vc_clone_graph.rehearsal_bootstrap import prepare_live_package

    crit = criteria()
    vc, episode = decision_id.split('/')
    owner = evidence_owner(cond, vc, episode, crit['donors'])
    session = OUT / 'runs' / cond / vc / episode
    if (session / 'result.json').is_file():
        return read(session / 'result.json')
    state = session_state(session)
    if state == 'gave_up':
        raise ValueError('pipeline gave up on this run earlier; not retried (no paid retries)')
    if state == 'fresh':
        shutil.rmtree(session, ignore_errors=True)  # only sessions with no model output
        session.mkdir(parents=True)
    log = (session / 'run.log').open('a')

    main = main_runs()[(vc, episode)]
    main_dir = Path(main['phase1_path']).parent.parent
    main_config = read(main_dir / 'run-config.json')
    main_pitch = Path(read(main_dir / 'input-provenance.json')['pitch_path'])
    main_audit = read(Path(main_config['run']['input_root']) / f'data/investors/{vc}/audits/{episode}.json')
    aliases = main_audit['target_company_aliases']
    overlay = overlay_dir(cond, vc, owner)

    package = session / 'inputs'
    if state == 'fresh':
        prepare_live_package(source_root=overlay, destination_root=package, vc_slug=vc, episode_slug=episode,
                             pitch=main_pitch.read_text(encoding='utf-8'), target_company_aliases=aliases)
        audit_path = package / f'data/investors/{vc}/audits/{episode}.json'
        audit = read(audit_path)
        audit.update(audit_source='memory-swap-pilot (pitch and aliases from the main-evaluation run)',
                     human_reviewed=False, variant_mode='founders-only')
        write(audit_path, audit)
        build_episode_manifest(package, vc, episode)
    verify_package(package, vc, episode)

    payload = json.loads(json.dumps(main_config))
    rel = lambda p: str(Path(p).relative_to(WORKSPACE))
    payload['run'].update(input_root=rel(package), output_root=rel(session / 'runs'),
                          checkpoint_path=rel(session / 'checkpoints.sqlite'))
    if owner is None:
        payload['precedents']['enabled'] = False
    write(session / 'run-config.json', payload)
    config = RunConfig.model_validate(payload)
    config._workspace = WORKSPACE
    with redirect_stdout(StringIO()) as out:
        command_preflight(config, session / 'run-config.json')
    preflight = json.loads(out.getvalue())
    write(session / 'preflight.json', preflight)
    if preflight['status'] != 'ready' or preflight.get('target_accessible'):
        raise ValueError(f'preflight not ready or exposes target: {preflight}')

    run_dir = session / 'runs' / episode
    with redirect_stdout(log), redirect_stderr(log):
        if state == 'pipeline_done':
            print('reusing the completed pipeline run; only components are run')
        else:
            command_run(config, resume=(state == 'interrupted'))
        command_verify(config)
    summary = read(run_dir / 'summary.json')
    if summary.get('decision', {}).get('decision') not in ('In', 'Out'):
        raise ValueError('no completed In/Out decision')

    comp = load_components_config(WORKSPACE / 'configs/components.toml')
    run = load_run(run_dir, workspace=WORKSPACE)
    if (run_dir / 'components/coverage.json').is_file():
        coverage = read(run_dir / 'components/coverage.json')
    else:
        coverage = run_coverage(run, comp.coverage, openrouter_provider(comp.coverage))
        write(run_dir / 'components/coverage.json', coverage)

    ann = annotation_root(cond, vc, owner, episode)
    panel = panel_for(cond, vc, owner, episode, crit['sample']['decisions'])
    lens_settings = comp.lens.model_validate({**comp.lens.model_dump(), 'annotation_root': str(ann)})
    lens = run_lens(run, lens_settings, openrouter_provider(lens_settings), coverage=coverage, panel=panel,
                    workspace=WORKSPACE)
    write(run_dir / 'components/lens.json', lens)

    policy = read(run_dir / 'assessment-policy.json')
    empty = []
    if owner is None:
        empty = ['investigator: Investment Memory (placeholder only)', 'investigator: prior cases (disabled)',
                 'decision agent: Investment Memory (placeholder only)', 'decision agent: prior cases (disabled)',
                 'coverage: Investment Memory excerpts (placeholder only)', 'lens: prior cases', 'lens: own profile',
                 'lens: panel profiles']
    if not lens['profile'] and 'lens: own profile' not in empty:
        empty.append('lens: own profile (fewer than 2 other annotated episodes)')
    if not lens['precedent_episodes'] and 'lens: prior cases' not in empty:
        empty.append('lens: prior cases (none retrieved)')
    usage = {'pipeline': summary.get('usage', {}).get('cost_usd', 0.0),
             'coverage': coverage.get('usage', {}).get('cost_usd', 0.0),
             'lens': lens.get('usage', {}).get('cost_usd', 0.0)}
    inputs = {
        'investor_name_in_prompts': run.investor_name,
        'evidence_owner': owner,
        'investment_memory': {'source_investor': owner, 'wiki_files': sorted(read(overlay / 'overlay.json')['files']),
                              'policy_excerpts_selected': len(policy.get('evidence', [])),
                              'policy_excerpt_chars': sum(len(e.get('text', '')) for e in policy.get('evidence', []))},
        'prior_cases': {'source_investor': owner, 'enabled': owner is not None,
                        'lens_precedent_episodes': [p['episode_slug'] for p in lens['precedent_episodes']],
                        'current_episode_excluded': episode not in [p['episode_slug'] for p in lens['precedent_episodes']]},
        'lens': {'memory_items': len(lens['memory_ids']), 'memory_source': lens['memory_source'],
                 'profile_source_investor': owner, 'profile': [p['taxonomy_label'] for p in lens['profile']],
                 'panel': lens['panel'], 'prompt_chars': lens['prompt_chars']},
        'coverage': {'prompt_chars': coverage.get('prompt_chars'), 'additions': len(coverage.get('additions', []))},
        'pitch': {'path': str(main_pitch), 'sha256': digest(main_pitch), 'chars': len(run.pitch_text)},
        'pitched_company_aliases': aliases,
        'overlay_mentions_of_pitched_company': alias_hits(overlay, aliases),
        'empty_inputs': empty,
    }
    result = {'condition': cond, 'decision_id': decision_id, 'vc_slug': vc, 'episode_slug': episode,
              'evidence_owner': owner, 'run_dir': rel(run_dir), 'cost_usd': usage,
              'cost_total_usd': round(sum(usage.values()), 6), 'inputs': inputs,
              'checksums': {'overlay': digest(overlay / 'overlay.json'),
                            'package_manifest': digest(package / f'data/investors/{vc}/manifests/{episode}.json'),
                            'summary': digest(run_dir / 'summary.json'),
                            'coverage': digest(run_dir / 'components/coverage.json'),
                            'lens': digest(run_dir / 'components/lens.json')}}
    write(session / 'result.json', result)
    return result


def prepare(decisions, conditions):
    from vc_clone_graph.cli import _indexing_provider
    from vc_clone_graph.config import RunConfig
    crit = criteria()
    main = main_runs()
    some = read(Path(main[tuple(decisions[0].split('/'))]['phase1_path']).parent.parent / 'run-config.json')
    config = RunConfig.model_validate(some)
    config._workspace = WORKSPACE
    embedder = _indexing_provider(config)
    for cond in conditions:
        for d in decisions:
            vc, ep = d.split('/')
            owner = evidence_owner(cond, vc, ep, crit['donors'])
            build_overlay(cond, vc, owner, embedder)
            if cond != 'OWN':
                annotation_root(cond, vc, owner, ep)
    print(f'overlays ready for {len(decisions)} decisions x {len(conditions)} conditions')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['prepare', 'run', 'job'])
    parser.add_argument('--decisions', help='comma-separated decision IDs (default: the whole sample)')
    parser.add_argument('--conditions', default=','.join(CONDITIONS))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--budget', type=float, default=35.0, help='stop starting jobs once measured spend reaches this (USD)')
    parser.add_argument('--execute', action='store_true', help='authorize paid model calls')
    parser.add_argument('--cond'); parser.add_argument('--decision')
    args = parser.parse_args(argv)
    decisions = args.decisions.split(',') if args.decisions else [d['decision_id'] for d in criteria()['sample']['decisions']]
    conditions = args.conditions.split(',')
    if args.action == 'prepare':
        prepare(decisions, conditions)
        return 0
    if args.action == 'job':
        try:
            r = run_job(args.cond, args.decision)
            print(json.dumps({'status': 'completed', 'cost': r['cost_total_usd']}))
            return 0
        except Exception as exc:
            session = OUT / 'runs' / args.cond / args.decision
            write(session / 'error.json', {'error': f'{type(exc).__name__}: {exc}', 'state': session_state(session),
                                           'spent_usd': round(spent(session), 6)})
            print(json.dumps({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}))
            return 1
    sessions = [(c, d, OUT / 'runs' / c / d) for d in decisions for c in conditions]
    jobs = [(c, d) for c, d, s in sessions if not (s / 'result.json').is_file() and session_state(s) != 'gave_up']
    gave_up = [f'{c}/{d}' for c, d, s in sessions if session_state(s) == 'gave_up' and not (s / 'result.json').is_file()]
    already = sum(spent(s) for _, _, s in sessions if s.is_dir())
    print(f'{len(jobs)} pending jobs ({len(decisions)} decisions x {len(conditions)} conditions); '
          f'{len(gave_up)} given up by the pipeline (not retried); spent so far USD {already:.2f}; '
          f'budget USD {args.budget:.2f}', flush=True)
    if not args.execute:
        print('dry listing only; pass --execute to make paid calls')
        return 0
    prepare(decisions, conditions)
    import threading
    lock = threading.Lock()
    stop = threading.Event()

    def one(job):
        c, d = job
        with lock:
            total = sum(spent(s) for _, _, s in sessions if s.is_dir())
            if total >= args.budget:
                stop.set()
        if stop.is_set():
            print(f'{c:9s} {d}: skipped, budget reached (USD {total:.2f})', flush=True)
            return 2
        p = subprocess.run([sys.executable, __file__, 'job', '--cond', c, '--decision', d],
                           capture_output=True, text=True, cwd=WORKSPACE)
        line = (p.stdout.strip().splitlines() or ['{}'])[-1]
        print(f'{c:9s} {d}: {line}', flush=True)
        return p.returncode
    with ThreadPoolExecutor(args.workers) as pool:
        codes = list(pool.map(one, jobs))
    total = sum(spent(s) for _, _, s in sessions if s.is_dir())
    print(f'done: {codes.count(0)} completed, {codes.count(1)} failed, {codes.count(2)} skipped for budget; '
          f'spent USD {total:.2f}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
