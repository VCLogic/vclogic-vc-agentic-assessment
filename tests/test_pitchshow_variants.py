import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_pitchshow_variants import render_variant, MODES


def receipt():
    rows = [('Founder', 'explicit_founder', 'Founder: Revenue is growing.'),
            ('Mac', 'explicit_investor_question', 'Investor question: How fast?'),
            ('Phil', 'explicit_investor_question', 'Investor question: What is the margin?'),
            ('Founder', 'explicit_founder', 'Founder: Thanks for investing!'),
            ('Founder', 'explicit_founder', 'Founder: Eighty percent.')]
    segments = [dict(speaker=s, retained_content_class=c, rendered_text=t, source_turn_index=i)
                for i, (s, c, t) in enumerate(rows)]
    return dict(pitch_safe=True, segments=segments, removed_candidate_lines=[3],
                candidate_pitch='\n'.join(t for _, _, t in rows)+'\n',
                pitch='\n'.join(t for i, (_, _, t) in enumerate(rows) if i != 3)+'\n')


def test_founders_only_and_other_vcs_keep_source_identity():
    r = receipt()
    founders, kept = render_variant(r, ['Mac Conwell', 'Mac'], MODES[0])
    assert founders == 'Founder: Revenue is growing.\nFounder: Eighty percent.\n'
    assert all(s['speaker'] == 'Founder' for s in kept)
    other, kept = render_variant(r, ['Mac Conwell', 'Mac'], MODES[1])
    assert 'How fast?' not in other
    assert 'What is the margin?' in other
    assert 'investing' not in other
    assert [s['source_turn_index'] for s in kept] == [0, 2, 4]
    phil, _ = render_variant(r, ['Phil'], MODES[1])
    assert 'How fast?' in phil and 'margin' not in phil


def test_refuse_tampered_or_unsafe_receipt():
    r = receipt(); r['pitch'] += 'Extra text\n'
    with pytest.raises(ValueError): render_variant(r, ['Mac'], MODES[0])
    r = receipt(); r['pitch_safe'] = False
    with pytest.raises(ValueError): render_variant(r, ['Mac'], MODES[0])
