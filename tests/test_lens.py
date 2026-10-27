from pathlib import Path

import pytest

from vc_clone_graph.components.config import ComponentsConfig, FactsheetSettings, LensSettings
from vc_clone_graph.components.consolidate import tally
from vc_clone_graph.components.factsheet import (
    check_transcript,
    founder_turns,
    in_room_window,
    validate_facts,
)
from vc_clone_graph.components.lens import apply_lens, candidates
from vc_clone_graph.components.run_inputs import RunArtifacts
from vc_clone_graph.precedents import TranscriptTurn

PITCH = "We sell to schools. We are raising $2 million at a $20 million valuation."
TAXONOMY = (
    {"label": "market_size_assessment", "coarse_parent": "market_opportunity"},
    {"label": "valuation_concern", "coarse_parent": "deal_terms_valuation"},
    {"label": "thesis_fit", "coarse_parent": "investor_fit_constraints"},
)


def run() -> RunArtifacts:
    return RunArtifacts(
        run_dir=Path("."), vc_slug="vc-a", episode_slug="ep-1", investor_name="Investor A",
        pitch_text=PITCH,
        phase1={"rationales": [
            {"rationale_id": "R1", "taxonomy_label": "market_size_assessment", "direction": "positive",
             "salience": "primary", "confidence": 0.8, "justification": "Big market."},
            {"rationale_id": "R2", "taxonomy_label": "valuation_concern", "direction": "positive",
             "salience": "secondary", "confidence": 0.6, "justification": "Fair price."},
        ]},
        phase2=None, policy_evidence=(), taxonomy=TAXONOMY, hashes={},
    )


def test_lens_is_on_by_default_with_contrastive_variant():
    lens = ComponentsConfig().lens
    assert lens.enabled is True and lens.variant == "contrastive"


def test_lens_drops_flips_keeps_unreviewed_and_grounds_additions():
    rows = candidates(run(), None)
    raw = {
        "investor_focus": "Price.",
        "reviews": [
            {"candidate_id": "R1", "action": "drop", "direction": "positive", "salience": "primary"},
            {"candidate_id": "R1", "action": "keep", "direction": "positive", "salience": "primary"},
        ],
        "additions": [
            {"taxonomy_label": "thesis_fit", "direction": "negative", "salience": "primary", "confidence": 2,
             "pitch_quote": "We sell to schools.", "memory_ids": ["W-1"]},
            {"taxonomy_label": "thesis_fit", "direction": "negative", "salience": "primary",
             "pitch_quote": "We sell to schools.", "memory_ids": ["W-1"]},
            {"taxonomy_label": "market_size_assessment", "direction": "negative", "salience": "primary",
             "pitch_quote": "invented quote here", "memory_ids": ["W-1"]},
        ],
    }
    out = apply_lens(raw, run(), rows, LensSettings(), memory_ids={"W-1"})
    assert [d["candidate_id"] for d in out["dropped"]] == ["R1"]  # first review wins
    assert out["unreviewed_candidate_ids"] == ["R2"]
    labels = [(r["rationale_id"], r["taxonomy_label"], r["direction"]) for r in out["rationales"]]
    assert labels == [("R2", "valuation_concern", "positive"), ("L1", "thesis_fit", "negative")]
    assert out["rationales"][1]["confidence"] == 1.0
    assert [r["rejection_reason"] for r in out["rejected_additions"]] == ["duplicate_label", "evidence_not_found"]
    assert tally(out["rationales"]) == pytest.approx(0.5 * 0.6 - 1.0)


def test_lens_additions_need_memory_and_accept_fact_or_precedent_grounding():
    rows = candidates(run(), None)
    base = {"taxonomy_label": "thesis_fit", "direction": "neutral", "salience": "secondary"}
    raw = {"reviews": [], "additions": [{**base, "memory_ids": ["W-9"], "fact_ids": ["F1"]}]}
    out = apply_lens(raw, run(), rows, LensSettings(), memory_ids={"W-1"}, fact_ids={"F1"})
    assert out["rejected_additions"][0]["rejection_reason"] == "no_valid_memory_id"
    raw["additions"][0]["memory_ids"] = ["W-1"]
    assert apply_lens(raw, run(), rows, LensSettings(), memory_ids={"W-1"}, fact_ids={"F1"})["rationales"][-1]["fact_ids"] == ["F1"]
    raw["additions"][0] = {**base, "memory_ids": ["W-1"], "precedent_ids": ["E2"]}
    out = apply_lens(raw, run(), rows, LensSettings(), memory_ids={"W-1"}, precedent_ids={"E2"})
    assert out["rationales"][-1]["precedent_ids"] == ["E2"]


REVIEW = {
    "episode_slug": "ep-1",
    "review": {"episode_kind": "pitch", "founder_speakers": ["Ann"], "pitch_start_turn": 0, "pitch_end_turn": 2},
    "annotations": [
        {"vc_slug": "vc-a", "decision_context": "initial_panel", "pitch_evidence": [{"turn_index": 3, "quote": "I'm out"}]},
        {"vc_slug": "vc-b", "decision_context": "initial_panel", "pitch_evidence": [{"turn_index": 5, "quote": "I'm in"}]},
        {"vc_slug": "vc-c", "decision_context": "later_diligence", "pitch_evidence": [{"turn_index": 8, "quote": "later"}]},
    ],
}
LINES = [
    ("Ann", "We sell to schools in Ohio."),
    ("Bea", "What is your churn rate?"),
    ("Ann", "Monthly churn is two percent across all of our districts."),
    ("Bea", "I'm out, sorry."),
    ("Ann", "Thank you Bea, I appreciate the feedback, we hear that a lot."),
    ("Cal", "I'm in for 25k."),
    ("Ann", "Our revenue was 400 thousand dollars last year from 30 districts."),
    ("Ann", "Thanks everyone, this was great."),
    ("Cal", "later"),
]
TURNS = tuple(TranscriptTurn(turn_index=i, speaker=s, text=t) for i, (s, t) in enumerate(LINES))


def test_fact_window_is_in_room_founder_answers_only():
    assert in_room_window(REVIEW) == (2, 5)  # later diligence does not extend the room
    check_transcript(TURNS, REVIEW)
    rows = founder_turns(TURNS, REVIEW, ["Bea Smith", "Cal Jones"])
    assert [r["turn_index"] for r in rows] == [2, 4]
    bad = {**REVIEW, "annotations": [{**REVIEW["annotations"][0], "pitch_evidence": [{"turn_index": 1, "quote": "I'm out"}]}]}
    with pytest.raises(ValueError):
        check_transcript(TURNS, bad)


def test_facts_must_be_neutral_quoted_and_free_of_reactions():
    rows = founder_turns(TURNS, REVIEW, ["Bea Smith", "Cal Jones"])
    raw = {"facts": [
        {"category": "traction_growth", "fact": "Monthly churn is two percent.", "quote": "Monthly churn is two percent", "turn": 2},
        {"category": "traction_growth", "fact": "Churn is low.", "quote": "churn is one percent", "turn": 2},
        {"category": "traction_growth", "fact": "The founders hear churn concerns often.", "quote": "I appreciate the feedback", "turn": 4},
        {"category": "traction_growth", "fact": "Your churn question was answered.", "quote": "across all of our districts", "turn": 2},
        {"category": "investor_fit_constraints", "fact": "x", "quote": "Monthly churn is two percent", "turn": 2},
    ]}
    kept, rejected = validate_facts(raw, rows, ["Bea Smith"], FactsheetSettings())
    assert [k["fact_id"] for k in kept] == ["F1"]
    assert [r["rejection_reason"] for r in rejected] == [
        "quote_not_found_in_turn", "reaction_or_panel_reference", "second_person_in_fact", "invalid_category"]


def test_selective_lens_caps_total_reasons_keeping_primary_first():
    rows = candidates(run(), None)
    raw = {"reviews": [], "additions": [
        {"taxonomy_label": "thesis_fit", "direction": "negative", "salience": "primary", "confidence": 0.9,
         "pitch_quote": "We sell to schools.", "memory_ids": ["W-1"]}]}
    out = apply_lens(raw, run(), rows, LensSettings(max_kept=2), memory_ids={"W-1"})
    assert [r["rationale_id"] for r in out["rationales"]] == ["R1", "L1"]
    assert out["dropped"] == [{"candidate_id": "R2", "taxonomy_label": "valuation_concern", "reason": "over_max_kept"}]


def test_memory_rebuilds_from_run_wiki_only_when_saved_index_is_stale(tmp_path):
    from types import SimpleNamespace

    from vc_clone_graph.components.lens import _memory_index
    from vc_clone_graph.retrieval import HybridWikiIndex

    wiki = tmp_path / "wiki" / "vc-a"
    wiki.mkdir(parents=True)
    (wiki / "persona.md").write_text("# Checks\nWrites small first checks.\n", encoding="utf-8")
    (tmp_path / "indexes").mkdir()
    HybridWikiIndex.build(wiki, None).save(tmp_path / "indexes" / "vc-a.json")
    package = SimpleNamespace(wiki=wiki)
    assert _memory_index(run(), tmp_path, package)[1] == "saved_index"

    (wiki / "persona.md").write_text("# Checks\nWrites large lead checks.\n", encoding="utf-8")
    current = {c.chunk_id for c in HybridWikiIndex.build(wiki, None).chunks}
    used = RunArtifacts(**{**run().__dict__, "policy_evidence": tuple({"evidence_id": i} for i in current)})
    index, source = _memory_index(used, tmp_path, package)
    assert source == "rebuilt_from_run_wiki" and "large lead" in index.chunks[0].text
    with pytest.raises(ValueError, match="does not contain"):
        _memory_index(RunArtifacts(**{**run().__dict__, "policy_evidence": ({"evidence_id": "W-old"},)}), tmp_path, package)


def test_profile_excludes_the_current_episode_for_every_investor(tmp_path):
    import json as _json

    from vc_clone_graph.components.profile import build_profile

    def write(vc, ep, labels):
        path = tmp_path / vc / f"{ep}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps({"rationale_mentions": [
            {"rationale_label": l, "scope": "original_pitch", "direction": "negative", "quote": f"{l} quote"}
            for l in labels]}), encoding="utf-8")

    for ep in ("e1", "e2", "e3"):
        write("vc-a", ep, ["thesis_fit", "market_size_assessment"])
        write("vc-b", ep, ["market_size_assessment"])
    write("vc-a", "now", ["valuation_concern"])
    write("vc-b", "now", ["valuation_concern", "thesis_fit"])
    profile = build_profile(tmp_path, "vc-a", "now")
    assert [v["taxonomy_label"] for v in profile] == ["thesis_fit", "market_size_assessment"]
    assert profile[0]["ratio"] == 2.0 and profile[0]["episodes"] == 3
    assert all("now" not in e for v in profile for e in v["examples"])
