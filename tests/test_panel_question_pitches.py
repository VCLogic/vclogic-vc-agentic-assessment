from __future__ import annotations

from vc_clone_graph.panel_question_pitches import build_panel_question_pitch


def episode():
    return {
        "founders": [{"name": "Ada Founder"}],
        "panel": [
            {"name": "Target VC", "slug": "target-vc-fund"},
            {"name": "Other VC", "slug": "other-vc-fund"},
        ],
    }


def record():
    return {
        "episode_slug": "1-example",
        "investor_aliases": ["Target", "Target VC"],
        "turns": [
            {"speaker": "Ada", "text": "We sell useful software.", "turn_index": 1},
            {"speaker": "Other", "text": "How many customers pay today?", "turn_index": 2},
            {"speaker": "Ada", "text": "Ten customers pay us today.", "turn_index": 3},
            {"speaker": "Target", "text": "What is retention?", "turn_index": 4},
            {"speaker": "Ada", "text": "Retention is ninety percent.", "turn_index": 5},
            {"speaker": "Other", "text": "Would you accept my investment offer?", "turn_index": 6},
            {"speaker": "Ada", "text": "Yes.", "turn_index": 7},
            {"speaker": "Other", "text": "I am out.", "turn_index": 8},
        ],
    }


def audit():
    return {
        "boundary_cleanup": {
            "boundary_event": {"evidence": "I am out.", "speaker": "Other"}
        }
    }


def test_adds_only_neutral_safe_non_target_question() -> None:
    canonical = (
        "Ada: We sell useful software.\n"
        "Ada: Ten customers pay us today.\n"
        "Ada: Retention is ninety percent.\n"
    )
    result = build_panel_question_pitch(
        canonical_pitch=canonical, source_record=record(), episode=episode(),
        audit=audit(), target_vc_slug="target-vc-fund",
    )

    assert result.pitch_text == (
        "Ada: We sell useful software.\n"
        "Panel Investor: How many customers pay today?\n"
        "Ada: Ten customers pay us today.\n"
        "Ada: Retention is ninety percent.\n"
    )
    assert [row.source_turn_index for row in result.insertions] == [2]
    assert result.insertions[0].answer_turn_index == 3
    assert "Other" not in result.pitch_text
    assert "Target" not in result.pitch_text


def test_uses_canonical_founder_membership_when_boundary_metadata_is_absent() -> None:
    broken = audit()
    broken["boundary_cleanup"]["boundary_event"]["evidence"] = "missing evidence"
    source = record()
    source["turns"] = source["turns"][:5]
    result = build_panel_question_pitch(
        canonical_pitch="Ada: We sell useful software.\n",
        source_record=source, episode=episode(), audit=broken,
        target_vc_slug="target-vc-fund",
    )
    assert result.boundary_turn_index == 6
    assert not result.insertions


def test_records_unmapped_question_instead_of_inserting_it() -> None:
    result = build_panel_question_pitch(
        canonical_pitch="Ada: We sell useful software.\n",
        source_record=record(), episode=episode(), audit=audit(),
        target_vc_slug="target-vc-fund",
    )
    assert not result.insertions
    assert any(row.reason == "founder_answer_not_in_canonical_pitch" for row in result.omitted)
