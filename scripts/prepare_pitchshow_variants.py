#!/usr/bin/env python3
"""Prepare speaker-filtered historical inputs and optionally run one held-out case."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from prepare_pitchshow_evaluation import gather, read, write
from pitchshow_evaluation_review import fingerprint
from pitchshow_context_quality import audit_context, restore_context

MODES = ('founders-only', 'founders-and-other-vcs')
CONTEXT_MODE = 'founders-context-restored'
SUPPORTED_MODES = (*MODES, CONTEXT_MODE)


def render_variant(receipt, aliases, mode):
    if mode not in SUPPORTED_MODES or not receipt['pitch_safe']:
        raise ValueError('invalid mode or unsafe reviewed pitch')
    segments = [s for s in receipt['segments'] if s['rendered_text']]
    if '\n'.join(s['rendered_text'] for s in segments) + '\n' != receipt['candidate_pitch']:
        raise ValueError('candidate lines do not match source segments')
    removed = set(receipt['removed_candidate_lines'])
    surviving = [s['rendered_text'] for i, s in enumerate(segments) if i not in removed]
    if '\n'.join(surviving) + '\n' != receipt['pitch']:
        raise ValueError('verified pitch differs from audited segments')
    target = {a.strip().casefold() for a in aliases}
    kept = []
    for i, segment in enumerate(segments):
        if i in removed or segment['speaker'].strip().casefold() in target:
            continue
        kind = segment['retained_content_class']
        if kind == 'explicit_founder' or (mode == MODES[1] and kind == 'explicit_investor_question'):
            kept.append(segment)
    if not any(s['retained_content_class'] == 'explicit_founder' for s in kept):
        raise ValueError('variant has no founder content')
    if mode == CONTEXT_MODE:
        return restore_context(receipt, kept, aliases)[0], kept
    return '\n'.join(s['rendered_text'] for s in kept) + '\n', kept


def prepare(root, *, restore=False):
    episodes, targets, _ = gather(root)
    total = 0
    context_reports = []
    modes = SUPPORTED_MODES if restore else MODES
    for folder in sorted((root / 'investors').iterdir()):
        annotations = folder / 'annotations.json'
        if not annotations.exists():
            continue
        expected = set()
        for row in read(annotations):
            if not row['evaluation_eligible']:
                continue
            slug = row['episode_slug']
            receipt = read(root / 'reviews' / f'{slug}.json')
            if receipt['fingerprint'] != fingerprint(episodes[slug], targets[slug]):
                raise ValueError(f'stale review: {slug}')
            target = next(t for t in targets[slug] if t['vc_slug'] == folder.name)
            case = folder / 'cases' / slug
            for mode in modes:
                pitch, segments = render_variant(receipt, target['aliases'], mode)
                destination = case / mode
                destination.mkdir(parents=True, exist_ok=True)
                (destination / 'pitch.txt').write_text(pitch)
                write(destination / 'audit.json', {
                    'schema': 'pitchshow-speaker-variant-v1', 'vc_slug': folder.name,
                    'episode_slug': slug, 'mode': mode, 'human_reviewed': False,
                    'pitch_sha256': sha256(pitch.encode()).hexdigest(),
                    'review_sha256': sha256((root / 'reviews' / f'{slug}.json').read_bytes()).hexdigest(),
                    'target_speaker_aliases': target['aliases'],
                    'source_turn_indices': [s['source_turn_index'] for s in segments],
                    'retained_speakers': sorted({s['speaker'] for s in segments}),
                    'target_speech_removed': True,
                    'company_aliases_required_at_packaging': True,
                })
                context = audit_context(receipt, segments, target['aliases'])
                context.update(vc_slug=folder.name, episode_slug=slug, mode=mode,
                               pitch_sha256=sha256(pitch.encode()).hexdigest(),
                               restoration_applied=mode == CONTEXT_MODE)
                write(destination / 'context-quality.json', context)
                context_reports.append(context)
                expected.add(destination / 'pitch.txt')
                total += 1
        # Remove only this generator's stale published inputs, never source material.
        for path in (folder / 'cases').glob('*/*/pitch.txt'):
            if path not in expected and path.parent.name in MODES:
                path.unlink()
                path.with_name('audit.json').unlink(missing_ok=True)
    write(root / 'variants-summary.json', {'eligible_vc_episode_pairs': total // len(modes),
                                         'prepared_inputs': total // len(modes) * len(MODES), 'modes': MODES})
    write(root / 'context-quality-summary.json', {
        'schema': 'pitchshow-context-quality-summary-v1',
        'audited_inputs': total, 'eligible_vc_episode_pairs': total // len(modes),
        'modes': modes, 'reports': context_reports})
    print(f'Prepared {total} inputs for {total // len(modes)} VC–episode pairs')


def package_case(root, workspace, vc, episode, mode, companies, *, execution_key=None):
    from vc_clone_graph.config import load_config, RunConfig
    from vc_clone_graph.firewall import safe_slug, verify_package
    from vc_clone_graph.rehearsal_bootstrap import prepare_live_package
    from vc_clone_graph.manifest_builder import build_episode_manifest

    safe_slug(vc, 'vc_slug'); safe_slug(episode, 'episode_slug')
    if mode not in SUPPORTED_MODES or not companies or not all(c.strip() for c in companies):
        raise ValueError('select a valid mode and provide --company aliases')
    rows = read(root / 'investors' / vc / 'annotations.json')
    if not any(r['episode_slug'] == episode and r['evaluation_eligible'] for r in rows):
        raise ValueError('case is not eligible for evaluation')
    case = root / 'investors' / vc / 'cases' / episode / mode
    pitch = (case / 'pitch.txt').read_text()
    audit = read(case / 'audit.json')
    if sha256(pitch.encode()).hexdigest() != audit['pitch_sha256']:
        raise ValueError('variant pitch hash mismatch')
    episodes, targets, _ = gather(root)
    receipt = read(root / 'reviews' / f'{episode}.json')
    if receipt['fingerprint'] != fingerprint(episodes[episode], targets[episode]):
        raise ValueError('stale source review')
    target = next(t for t in targets[episode] if t['vc_slug'] == vc)
    if render_variant(receipt, target['aliases'], mode)[0] != pitch:
        raise ValueError('variant does not match current speaker-filtered review')
    aliases = list(dict.fromkeys(c.strip() for c in companies))
    identity = sha256(json.dumps([audit['pitch_sha256'], aliases]).encode()).hexdigest()[:16]
    if execution_key is not None:
        if len(execution_key) != 64 or any(c not in '0123456789abcdef' for c in execution_key):
            raise ValueError('invalid execution identity')
        identity = execution_key
    session = workspace / 'outputs/pitchshow-evaluation' / vc / episode / mode / identity
    package = session / 'inputs'
    prepare_live_package(source_root=workspace / 'inputs', destination_root=package,
                         vc_slug=vc, episode_slug=episode, pitch=pitch,
                         target_company_aliases=aliases)
    verified = verify_package(package, vc, episode)
    if tuple(aliases) != verified.target_company_aliases:
        raise ValueError('package exclusion aliases do not match')
    # Correct the live-package helper's provenance; this is machine-reviewed history.
    audit_path = package / f'data/investors/{vc}/audits/{episode}.json'
    package_audit = read(audit_path)
    package_audit.update(audit_source='pitchshow-machine-reviewed-speaker-variant',
                         human_reviewed=False, variant_mode=mode)
    write(audit_path, package_audit)
    build_episode_manifest(package, vc, episode)
    verify_package(package, vc, episode)
    template = load_config(workspace / f'configs/investors/{vc}/canonical.toml')
    payload = template.model_dump(mode='json')
    payload['run'].update(vc_slug=vc, episode_slug=episode, mode='full',
        input_root=str(package.relative_to(workspace)),
        output_root=str((session / 'runs').relative_to(workspace)),
        checkpoint_path=str((session / 'checkpoints.sqlite').relative_to(workspace)))
    config = RunConfig.model_validate(payload)
    config._workspace = workspace
    write(session / 'run-config.json', payload)
    write(session / 'selection.json', {'vc_slug': vc, 'episode_slug': episode, 'mode': mode,
        'target_company_aliases': aliases, 'pitch_sha256': audit['pitch_sha256'],
        'preparation_status': 'packaged'})
    print(f'Verified package: {package}')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('evaluation/pitchshow'))
    parser.add_argument('--restore-context', action='store_true',
                        help='Also prepare the separate opt-in founders-context-restored mode')
    parser.add_argument('--vc'); parser.add_argument('--episode')
    parser.add_argument('--mode', choices=SUPPORTED_MODES)
    parser.add_argument('--company', action='append', default=[])
    parser.add_argument('--run', action='store_true', help='Execute paid assessment after packaging')
    args = parser.parse_args()
    selected = any((args.vc, args.episode, args.mode, args.company, args.run))
    if not selected:
        prepare(args.root, restore=args.restore_context)
        return
    if not all((args.vc, args.episode, args.mode)):
        parser.error('packaging requires --vc, --episode and --mode')
    if not args.company:
        from pitchshow_company_aliases import validate_aliases
        args.company = validate_aliases(args.root, read(args.root/'company-aliases.json'), {args.episode})[args.episode]
    config = package_case(args.root, Path.cwd().resolve(), args.vc, args.episode, args.mode, args.company)
    if args.run:
        from vc_clone_graph.cli import command_run
        command_run(config)


if __name__ == '__main__':
    main()
