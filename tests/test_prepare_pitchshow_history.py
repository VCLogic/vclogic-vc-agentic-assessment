"""Offline history publication keeps source hashes and decisions bound to dialogue."""
import importlib.util
from hashlib import sha256
import json
from pathlib import Path

import pytest

from vc_clone_graph.precedent_builder import _canonical_json_bytes, parse_speaker_turns
from vc_clone_graph.precedents import PrecedentEpisode

spec = importlib.util.spec_from_file_location('prepare_pitchshow_history', Path(__file__).parents[1] / 'scripts/prepare_pitchshow_history.py')
history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history)


def fixture(tmp_path):
    vc = 'alice-investor'
    slug = '1-company'
    source = {'slug': slug, 'panel': [{'name': 'Alice Investor'}], 'transcript': 'Narrator: Next on the show\nFounder: We sell software.\nAlice: I am in.\nMonths later they failed.\nNarrator: After the pitch'}
    decision = {'vc_slug': vc, 'pitch_window_decision': 'In', 'decision_context': 'initial_panel', 'pitch_evidence_verified': True, 'pitch_evidence': [{'turn_index': 2, 'quote': 'I am in.'}]}
    review = {'episode_slug': slug, 'review': {'episode_kind': 'pitch', 'founder_speakers': ['Founder'], 'pitch_start_turn': 1, 'pitch_end_turn': 2, 'decisions': [decision]}, 'annotations': [decision]}
    record = PrecedentEpisode(episode_slug=slug, episode_number=1, source_path=f'sources/{slug}.json', source_sha256=sha256(_canonical_json_bytes(source)).hexdigest(), investor_aliases=('Alice', 'Alice Investor'), investor_present=True, turns=parse_speaker_turns(source['transcript']), decision={'status': 'In', 'context': 'initial_panel', 'check_tier': None, 'conditions': ['later contaminated condition'], 'evidence': [{'turn_start': 2, 'turn_end': 2, 'text': 'I am in.\nMonths later they failed.'}], 'audit_source': 'old', 'audit_notes': 'later contaminated notes'})
    corpus = tmp_path / 'inputs/data/investors' / vc / 'precedents'
    history.write(corpus / record.source_path, source)
    history.write(corpus / 'records' / f'{slug}.json', record.model_dump(mode='json'))
    history.write(corpus / 'corpus-manifest.json', {'records': [{'episode_slug': slug, 'source_path': record.source_path, 'source_sha256': record.source_sha256, 'record_path': f'records/{slug}.json', 'record_sha256': sha256((corpus / 'records' / f'{slug}.json').read_bytes()).hexdigest()}]})
    review_root = tmp_path / 'evaluation/pitchshow'
    history.write(review_root / 'reviews' / f'{slug}.json', review)
    history.write(review_root / 'investors' / vc / 'source/episodes' / f'{slug}.json', source)
    config = tmp_path / 'configs/investors' / vc / 'canonical.toml'
    config.parent.mkdir(parents=True)
    config.write_text(f'[run]\nvc_slug="{vc}"\ninput_root="inputs"\n[precedents]\ncorpus_path="data/investors/{vc}/precedents"\n')
    return vc, slug, source, record, review, review_root, corpus


def test_stage_hashes_remapped_decision_and_archive(tmp_path):
    vc, slug, source, original, review, review_root, corpus = fixture(tmp_path)
    out = tmp_path / 'staging'
    report = history.prepare(tmp_path, out, review_root)
    assert report['included'] == 1
    staged = out / vc / 'precedents'
    record = PrecedentEpisode.model_validate_json((staged / 'records' / f'{slug}.json').read_bytes())
    source_bytes = (staged / record.source_path).read_bytes()
    assert record.source_sha256 == sha256(source_bytes).hexdigest()
    assert record.decision.evidence[0].turn_start == 1
    assert record.decision.evidence[0].text == 'I am in.'
    assert record.decision.conditions == ()
    assert 'Months later' not in source_bytes.decode()
    assert (out / 'raw-originals' / vc / f'{slug}.json').read_bytes() == (corpus / original.source_path).read_bytes()
    sidecar = history.read(out / vc / 'cleaning' / f'{slug}.json')
    assert sidecar['original_source_sha256'] == original.source_sha256
    assert sidecar['original_to_clean'] == {'1': 0, '2': 1}


@pytest.mark.parametrize('unsafe', ['review_mismatch', 'decision_quote_removed', 'missing_review'])
def test_unsafe_rows_excluded(tmp_path, unsafe):
    vc, slug, source, record, review, review_root, corpus = fixture(tmp_path)
    receipt = review_root / 'reviews' / f'{slug}.json'
    if unsafe == 'review_mismatch':
        history.write(review_root / 'investors' / vc / 'source/episodes' / f'{slug}.json', {**source, 'transcript': 'different'})
    elif unsafe == 'decision_quote_removed':
        review['annotations'][0]['pitch_evidence'][0]['quote'] = 'Months later they failed.'
        history.write(receipt, review)
    else:
        receipt.unlink()
    out = tmp_path / 'staging'
    report = history.prepare(tmp_path, out, review_root)
    assert report['included'] == 0
    assert report['excluded'] == 1
    assert history.read(out / vc / 'precedents/corpus-manifest.json')['records'] == []
    assert history.read(out / vc / 'cleaning-exclusions.json')[0]['reason']


def test_install_refuses_missing_stage_index_without_changing_original(tmp_path):
    *_, review_root, corpus = fixture(tmp_path)
    before = (corpus / 'corpus-manifest.json').read_bytes()
    out = tmp_path / 'staging'
    history.prepare(tmp_path, out, review_root)
    with pytest.raises(ValueError, match='missing staged index'):
        history.install(out, tmp_path)
    assert (corpus / 'corpus-manifest.json').read_bytes() == before


def test_install_backs_up_corpus_and_index(tmp_path):
    vc, slug, source, record, review, review_root, corpus = fixture(tmp_path)
    old = (corpus / record.source_path).read_bytes()
    out = tmp_path / 'staging'
    history.prepare(tmp_path, out, review_root)
    history.write(out / vc / 'precedents.index.json', {'new': True})
    index = tmp_path / 'inputs/indexes' / f'{vc}.precedents.json'
    history.write(index, {'old': True})
    backup = history.install(out, tmp_path)
    assert (backup / vc / 'precedents' / record.source_path).read_bytes() == old
    assert history.read(backup / vc / 'precedents.index.json') == {'old': True}
    assert history.read(index) == {'new': True}
    assert 'Months later' not in (corpus / record.source_path).read_text()


def test_failed_install_restores_corpus_and_index(tmp_path, monkeypatch):
    vc, slug, source, record, review, review_root, corpus = fixture(tmp_path)
    old = (corpus / record.source_path).read_bytes()
    out = tmp_path / 'staging'
    history.prepare(tmp_path, out, review_root)
    history.write(out / vc / 'precedents.index.json', {'new': True})
    index = tmp_path / 'inputs/indexes' / f'{vc}.precedents.json'
    history.write(index, {'old': True})
    rename = Path.rename

    def fail_index_publish(path, target):
        if path.name == index.name + '.clean-install-tmp':
            raise OSError('simulated index publish error')
        return rename(path, target)

    monkeypatch.setattr(Path, 'rename', fail_index_publish)
    with pytest.raises(OSError, match='simulated'):
        history.install(out, tmp_path)
    assert (corpus / record.source_path).read_bytes() == old
    assert history.read(index) == {'old': True}


@pytest.mark.parametrize('supported,evidence_matches,explicit_rejection,expected', [
    (True, True, False, True), (False, True, False, False),
    (True, False, False, False), (True, True, True, False),
])
def test_alternate_review_verification_is_bound_to_exact_evidence(tmp_path, supported, evidence_matches, explicit_rejection, expected):
    vc, slug, source, record, review, review_root, corpus = fixture(tmp_path)
    annotation = review['annotations'][0]
    annotation.pop('pitch_evidence_verified')
    if explicit_rejection:
        annotation['pitch_evidence_verified'] = False
    review['verification'] = {'decisions': [{'vc_slug': vc,
        'pitch_decision_supported': supported,
        'pitch_evidence': annotation['pitch_evidence'] if evidence_matches else []}]}
    if expected:
        assert history.clean_record(record, source, review, vc, Path('review.json'))[0].decision.status == 'In'
    else:
        with pytest.raises(ValueError, match='decision_not_verified'):
            history.clean_record(record, source, review, vc, Path('review.json'))


def test_source_backup_is_read_only_and_publication_targets_canonical(tmp_path):
    import shutil
    vc, slug, source, record, review, review_root, corpus = fixture(tmp_path)
    backup = tmp_path / 'original-backup'
    shutil.copytree(corpus, backup / vc / 'precedents')
    original = (backup / vc / 'precedents' / record.source_path).read_bytes()
    history.write(corpus / 'corpus-manifest.json', {'records': []})
    result = history.prepare(tmp_path, tmp_path / 'stage', review_root, backup)
    assert result['included'] == 1
    assert result['investors'][0]['canonical_corpus'] == str(corpus)
    assert result['investors'][0]['decision_counts'] == {'In': 1, 'Out': 0, 'unobserved': 0}
    assert (backup / vc / 'precedents' / record.source_path).read_bytes() == original
