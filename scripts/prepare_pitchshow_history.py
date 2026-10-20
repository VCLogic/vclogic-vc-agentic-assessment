#!/usr/bin/env python3
"""Stage source-bound, reviewed historical dialogue corpora without network calls."""
from __future__ import annotations

import argparse
import fcntl
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tomllib

from vc_clone_graph.precedent_builder import _canonical_json_bytes, _chunk_rows, _locator_text, parse_speaker_turns
from vc_clone_graph.precedents import DecisionEvidence, PrecedentDecision, PrecedentEpisode
from vc_clone_graph.historical_conversation import clean_historical_conversation


def read(path):
    return json.loads(path.read_bytes())


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(payload))


def decision_verified(annotation, review, vc_slug):
    """Support both receipt formats without overriding an explicit rejection."""
    if 'pitch_evidence_verified' in annotation:
        return annotation['pitch_evidence_verified'] is True
    verdicts = [v for v in review.get('verification', {}).get('decisions', [])
                if v.get('vc_slug') == vc_slug]
    return (len(verdicts) == 1
            and verdicts[0].get('pitch_decision_supported') is True
            and bool(annotation.get('pitch_evidence'))
            and verdicts[0].get('pitch_evidence') == annotation['pitch_evidence'])


def clean_record(record, source, review, vc_slug, review_path):
    """Bind the verified in-room decision to the retained exact dialogue."""
    cleaned = clean_historical_conversation(source, review, participant_aliases=record.investor_aliases)
    if not cleaned.included:
        raise ValueError(cleaned.exclusion_reason)
    annotations = [a for a in review.get('annotations', []) if a.get('vc_slug') == vc_slug]
    if len(annotations) != 1:
        raise ValueError('missing_or_ambiguous_investor_review')
    annotation = annotations[0]
    status = annotation.get('pitch_window_decision')
    context = annotation.get('decision_context')
    if status not in {'In', 'Out', 'Unobserved'}:
        raise ValueError('invalid_review_decision')
    if status != 'Unobserved' and context not in {'initial_panel', 'same_session_reversal'}:
        raise ValueError('decision_not_in_room')
    turns = parse_speaker_turns(cleaned.transcript)
    evidence = []
    if status in {'In', 'Out'}:
        if not decision_verified(annotation, review, vc_slug):
            raise ValueError('decision_not_verified')
        aliases = {_locator_text(a).casefold() for a in record.investor_aliases}
        for item in annotation.get('pitch_evidence', []):
            original = item.get('turn_index')
            if type(original) is not int or original not in cleaned.original_to_clean:
                raise ValueError('decision_turn_removed')
            index = cleaned.original_to_clean[original]
            turn = turns[index]
            quote = _locator_text(item.get('quote', ''))
            if not quote or quote not in _locator_text(turn.text):
                raise ValueError('decision_quote_removed')
            if _locator_text(turn.speaker).casefold() not in aliases:
                raise ValueError('decision_speaker_mismatch')
            evidence.append(DecisionEvidence(turn_start=index, turn_end=index, text=turn.text))
        if not evidence:
            raise ValueError('decision_evidence_missing')
    # Keep only identity and dialogue; descriptions and other source metadata can summarize later outcomes.
    snapshot = {'slug': record.episode_slug, 'transcript': cleaned.transcript}
    source_hash = sha256(_canonical_json_bytes(snapshot)).hexdigest()
    result = PrecedentEpisode(
        episode_slug=record.episode_slug, episode_number=record.episode_number,
        source_path=f'sources/{record.episode_slug}.json', source_sha256=source_hash,
        investor_aliases=record.investor_aliases,
        investor_present=any(_locator_text(t.speaker).casefold() in {_locator_text(a).casefold() for a in record.investor_aliases} for t in turns),
        turns=turns, decision=PrecedentDecision(
            status='unobserved' if status == 'Unobserved' else status,
            context=context if status != 'Unobserved' else 'unclear', check_tier=None, conditions=(), evidence=tuple(evidence),
            audit_source=review_path.as_posix(),
            audit_notes='Verified current-session dialogue; decision evidence remapped to cleaned source.'
        ))
    return result, snapshot, cleaned


def prepare(project_root: Path, output_root: Path, review_root: Path, source_backup: Path | None = None):
    if output_root.exists():
        raise ValueError(f'staging destination already exists: {output_root}')
    configs = sorted((project_root / 'configs/investors').glob('*/canonical.toml'))
    if not configs:
        raise ValueError('no canonical investor configs')
    report = {'schema': 'pitchshow-history-clean-v1', 'investors': [], 'included': 0, 'excluded': 0}
    for config_path in configs:
        config = tomllib.loads(config_path.read_text())
        vc = config['run']['vc_slug']
        canonical_corpus = project_root / config['run']['input_root'] / config['precedents']['corpus_path']
        corpus = source_backup / vc / 'precedents' if source_backup else canonical_corpus
        destination = output_root / vc / 'precedents'
        records = []
        exclusions = []
        for entry in read(corpus / 'corpus-manifest.json')['records']:
            slug = entry['episode_slug']
            source_path = corpus / entry['source_path']
            raw = source_path.read_bytes()
            original_hash = sha256(raw).hexdigest()
            # Archive even excluded episodes; never modify or replace original snapshots in place.
            archive = output_root / 'raw-originals' / vc / f'{slug}.json'
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(raw)
            try:
                if original_hash != entry['source_sha256']:
                    raise ValueError('original_source_hash_mismatch')
                record_bytes = (corpus / entry['record_path']).read_bytes()
                if sha256(record_bytes).hexdigest() != entry['record_sha256']:
                    raise ValueError('original_record_hash_mismatch')
                record = PrecedentEpisode.model_validate_json(record_bytes)
                if record.source_sha256 != original_hash:
                    raise ValueError('record_source_hash_mismatch')
                source = json.loads(raw)
                if record.turns != parse_speaker_turns(source.get('transcript', '')):
                    raise ValueError('original_turns_mismatch')
                receipt = review_root / 'reviews' / f'{slug}.json'
                reviewed_source = review_root / 'investors' / vc / 'source/episodes' / f'{slug}.json'
                if not receipt.is_file() or not reviewed_source.is_file():
                    raise ValueError('missing_review_or_review_source')
                reviewed_bytes = reviewed_source.read_bytes()
                if json.loads(reviewed_bytes).get('transcript') != source.get('transcript'):
                    raise ValueError('review_source_transcript_mismatch')
                review = read(receipt)
                if review.get('episode_slug') != slug:
                    raise ValueError('review_episode_mismatch')
                result, snapshot, cleaned = clean_record(record, source, review, vc, receipt)
            except (ValueError, KeyError, TypeError) as exc:
                exclusions.append({'episode_slug': slug, 'reason': str(exc), 'original_source_sha256': original_hash})
                continue
            write(destination / result.source_path, snapshot)
            write(destination / 'records' / f'{slug}.json', result.model_dump(mode='json'))
            write(destination.parent / 'cleaning' / f'{slug}.json', {
                'original_source_sha256': original_hash,
                'original_source_path': str(source_path),
                'archived_original_path': str(archive),
                'review_source_sha256': sha256(reviewed_bytes).hexdigest(),
                'review_sha256': sha256(receipt.read_bytes()).hexdigest(),
                'clean_source_sha256': result.source_sha256,
                'original_to_clean': cleaned.original_to_clean, 'audit': cleaned.audit,
            })
            records.append(result)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'chunks.jsonl').write_text(''.join(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n' for r in _chunk_rows(records)))
        write(destination / 'corpus-manifest.json', {'schema': 'precedent-corpus-v1', 'records': [{
            'decision_status': r.decision.status, 'episode_slug': r.episode_slug,
            'record_path': f'records/{r.episode_slug}.json',
            'record_sha256': sha256((destination / 'records' / f'{r.episode_slug}.json').read_bytes()).hexdigest(),
            'source_path': r.source_path, 'source_sha256': r.source_sha256,
        } for r in records]})
        write(destination.parent / 'cleaning-exclusions.json', exclusions)
        report['investors'].append({'vc_slug': vc, 'canonical_corpus': str(canonical_corpus), 'source_corpus': str(corpus), 'staged_corpus': str(destination), 'included': len(records), 'excluded': len(exclusions), 'decision_counts': {status: sum(r.decision.status == status for r in records) for status in ('In', 'Out', 'unobserved')}, 'canonical_index': str(project_root / config['run']['input_root'] / 'indexes' / f'{vc}.precedents.json')})
        report['included'] += len(records)
        report['excluded'] += len(exclusions)
    write(output_root / 'preparation-report.json', report)
    return report


def install(output_root: Path, project_root: Path):
    """Publish under the batch lock; rename full backups and restore on any error."""
    report = read(output_root / 'preparation-report.json')
    backup_root = output_root / ('install-backup-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    operations = []
    for item in report['investors']:
        vc = item['vc_slug']
        stage = Path(item['staged_corpus'])
        index = stage.parent / 'precedents.index.json'
        if not index.is_file():
            raise ValueError(f'missing staged index: {index}')
        operations.extend([(stage, Path(item['canonical_corpus']), backup_root / vc / 'precedents'),
                           (index, Path(item['canonical_index']), backup_root / vc / 'precedents.index.json')])
    lock_path = project_root / 'outputs/pitchshow-evaluation/batch/execution.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    installed = []
    temporaries = []
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for source, target, backup in operations:
                temporary = target.with_name(target.name + '.clean-install-tmp')
                if temporary.exists():
                    raise ValueError(f'install temporary already exists: {temporary}')
                temporaries.append(temporary)
                if source.is_dir():
                    shutil.copytree(source, temporary)
                else:
                    shutil.copy2(source, temporary)
                backup.parent.mkdir(parents=True, exist_ok=True)
                existed = target.exists()
                if existed:
                    target.rename(backup)
                installed.append((target, backup, existed))
                temporary.rename(target)
        except BaseException:
            for temporary in temporaries:
                if temporary.is_dir():
                    shutil.rmtree(temporary)
                elif temporary.exists():
                    temporary.unlink()
            for target, backup, existed in reversed(installed):
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
                if existed:
                    backup.rename(target)
            raise
        write(backup_root / 'rollback-manifest.json', [{'target': str(t), 'backup': str(b), 'existed': e} for t, b, e in installed])
    return backup_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--review-root', type=Path)
    parser.add_argument('--source-backup', type=Path, help='Original installation backup containing <vc>/precedents; rebuild without replacing installed data first.')
    parser.add_argument('--install', action='store_true', help='Install prepared corpora and their staged indexes; preserve rollback backups.')
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (args.output_root or root / 'outputs/pitchshow-history-clean').resolve()
    if args.install:
        print(f'Installed; rollback backup: {install(output, root)}.')
    else:
        report = prepare(root, output, (args.review_root or root / 'evaluation/pitchshow').resolve(), args.source_backup.resolve() if args.source_backup else None)
        print(f"Staged {report['included']} records; excluded {report['excluded']}; {len(report['investors'])} investors.")


if __name__ == '__main__':
    main()
