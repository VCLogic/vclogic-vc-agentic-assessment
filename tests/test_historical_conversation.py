"""Historical dialogue retains in-room decisions, not show commentary."""
import json
from pathlib import Path

import pytest

from vc_clone_graph.historical_conversation import clean_historical_conversation
from vc_clone_graph.precedent_builder import parse_speaker_turns

ROOT = Path(__file__).resolve().parents[1]


def _fixture(slug):
    path = ROOT / 'onboarding/jake-chapman-marque-ventures/source/pitch-show/episodes' / f'{slug}.json'
    review = ROOT / 'evaluation/pitchshow/reviews' / f'{slug}.json'
    return json.loads(path.read_text()), json.loads(review.read_text())


@pytest.mark.parametrize('slug', ['7-shimmur', '10-sudden-coffee'])
def test_real_conversations_remove_intro_and_keep_all_reviewed_decisions(slug):
    episode, review = _fixture(slug)
    original = episode['transcript']
    result = clean_historical_conversation(episode, review)
    assert result.included, result.exclusion_reason
    assert episode['transcript'] == original
    assert 'iMirror' not in result.transcript
    assert 'Babyscripts' not in result.transcript
    assert 'From Gimlet' not in result.transcript
    for decision in review['review']['decisions']:
        for receipt in decision['pitch_evidence']:
            assert receipt['quote'] in result.transcript
            assert receipt['turn_index'] in result.original_to_clean
    turns = parse_speaker_turns(result.transcript)
    assert [t.turn_index for t in turns] == list(range(len(turns)))
    assert set(result.original_to_clean.values()) == set(range(len(turns)))
    if slug == '7-shimmur':
        assert 87 in result.original_to_clean
        assert 123 in result.original_to_clean
        assert all(index >= 17 for index in result.original_to_clean)


def _synthetic():
    episode = {'slug': 'example', 'panel': [{'name': 'Investor Person'}],
               'founders': [{'name': 'Founder Person'}], 'transcript': (
                   'Investor: Last episode I invested.\n'
                   'Founder\n: Our company makes widgets.\n'
                   'The host explains the company.\n'
                   'Host: This week is special.\n'
                   'Investor: Tell me about revenue.\n'
                   'Founder: We sell widgets.\n'
                   'Investor: I will invest.\n'
                   'Months later everything changed.\n'
                   'Founder: Postpitch interview!\n'
                   'Investor: An unrelated postpitch quote.\n')}
    review = {'episode_slug': 'example', 'review': {
        'episode_kind': 'pitch', 'founder_speakers': ['Founder Person'],
        'pitch_start_turn': 1, 'pitch_end_turn': 5,
        'decisions': [{'decision_context': 'initial_panel', 'pitch_evidence': [
            {'turn_index': 5, 'quote': 'I will invest.'}]}]}}
    return episode, review


def test_strict_lines_split_speakers_and_postpitch_participant_quotes():
    episode, review = _synthetic()
    result = clean_historical_conversation(episode, review)
    assert result.included
    assert result.transcript == ('Founder: Our company makes widgets.\n'
                                 'Investor: Tell me about revenue.\n'
                                 'Founder: We sell widgets.\n'
                                 'Investor: I will invest.')
    assert result.original_to_clean == {1: 0, 3: 1, 4: 2, 5: 3}
    assert len(result.audit['trimmed_turns']) == 2


def test_unattributed_decision_evidence_fails_closed():
    episode, review = _synthetic()
    episode['transcript'] = episode['transcript'].replace(
        'Investor: I will invest.', 'Investor: Let me think.\nI will invest.')
    result = clean_historical_conversation(episode, review)
    assert result.exclusion_reason == 'decision_evidence_not_preserved'
    assert result.transcript == ''
    assert result.original_to_clean == {}


def test_missing_review_excludes_even_an_unobserved_source():
    episode, _ = _synthetic()
    result = clean_historical_conversation(episode, None)
    assert result.exclusion_reason == 'missing_review'
    assert not result.included
    assert not result.transcript


@pytest.mark.parametrize('start,end', [(True, 5), (5, 1), (-1, 5), (1, 999)])
def test_invalid_review_bounds_fail_closed(start, end):
    episode, review = _synthetic()
    review['review'].update(pitch_start_turn=start, pitch_end_turn=end)
    assert not clean_historical_conversation(episode, review).included


def test_later_evidence_does_not_extend_conversation():
    episode, review = _synthetic()
    review['review']['decisions'][0]['later_evidence'] = [
        {'turn_index': 7, 'quote': 'An unrelated postpitch quote.'}]
    assert 'postpitch' not in clean_historical_conversation(episode, review).transcript


def test_header_on_separate_line_preserves_first_statement_only():
    episode, review = _synthetic()
    episode['transcript'] = episode['transcript'].replace(
        'Investor: I will invest.', 'Investor:\nI will invest.')
    result = clean_historical_conversation(episode, review)
    assert result.included
    assert 'Investor: I will invest.' in result.transcript
    assert 'Months later' not in result.transcript


def test_participant_labelled_show_commentary_is_removed_inside_window():
    episode, review = _synthetic()
    episode['transcript'] = episode['transcript'].replace(
        'Tell me about revenue.', 'Last episode I invested in another company.')
    result = clean_historical_conversation(episode, review)
    assert result.included
    assert 'Last episode' not in result.transcript
    assert any(t['reason'] == 'episode_commentary' for t in result.audit['removed_turns'])


def test_live_cross_episode_investment_reference_is_removed():
    episode, review = _synthetic()
    episode['transcript'] = episode['transcript'].replace(
        'Tell me about revenue.',
        'There was a company on, on the show last season that I invested in called Vital Audio.')
    result = clean_historical_conversation(episode, review)
    assert result.included
    assert 'Vital Audio' not in result.transcript
    assert 'I will invest.' in result.transcript
