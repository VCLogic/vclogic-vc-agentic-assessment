import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_pitchshow_variants import MODES, CONTEXT_MODE, render_variant
from pitchshow_context_quality import audit_context


def fixture(decision=False):
    rows = [dict(speaker='Jenny', retained_content_class='explicit_investor_question',
                 rendered_text="Investor question: What's the burn?", source_turn_index=67,
                 preceded_by_investor_decision=decision),
            dict(speaker='Ariel', retained_content_class='explicit_founder',
                 rendered_text="Ariel: Right now with the four, it's about seven.",
                 source_turn_index=68, preceded_by_investor_decision=decision),
            dict(speaker='Jenny', retained_content_class=None, rendered_text='',
                 source_turn_index=81, preceded_by_investor_decision=True,
                 reason='Decision explains $7K/month burn. Invest!')]
    pitch = '\n'.join(s['rendered_text'] for s in rows if s['rendered_text']) + '\n'
    return dict(pitch_safe=True, segments=rows, removed_candidate_lines=[],
                candidate_pitch=pitch, pitch=pitch)


def test_restores_only_topic_never_target_speech_or_decision_units():
    r = fixture()
    originals = [render_variant(r, ['Jenny'], m)[0] for m in MODES]
    pitch, kept = render_variant(r, ['Jenny'], CONTEXT_MODE)
    assert pitch == "[Neutral question topic: burn]\nAriel: Right now with the four, it's about seven.\n"
    assert all(s['speaker'] == 'Ariel' for s in kept)
    assert originals == [render_variant(r, ['Jenny'], m)[0] for m in MODES]
    report = audit_context(r, kept, ['Jenny'])
    assert report['findings'][0]['context_source_turn_index'] == 67
    assert report['orphaned_answer_candidates'] == 1


def test_post_decision_and_rejected_questions_cannot_restore():
    for decision in (True, False):
        r = fixture(decision)
        if not decision:
            r['removed_candidate_lines'] = [0]
            r['pitch'] = r['segments'][1]['rendered_text'] + '\n'
        pitch, _ = render_variant(r, ['Jenny'], CONTEXT_MODE)
        assert 'Neutral question' not in pitch


def test_present_question_is_not_orphaned():
    r = fixture()
    _, kept = render_variant(r, ['Other VC'], MODES[1])
    assert audit_context(r, kept, ['Other VC'])['orphaned_answer_candidates'] == 0
