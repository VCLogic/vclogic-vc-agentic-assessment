from hashlib import sha256

from vc_clone_graph.config import PrecedentSettings
from vc_clone_graph.precedents import (
    DecisionEvidence,
    PrecedentCorpus,
    PrecedentDecision,
    PrecedentEpisode,
    TranscriptTurn,
)

FOUNDERS = {"1-alpha": "Ann", "2-beta": "Bob", "3-gamma": "Cid"}


def episode(slug: str) -> PrecedentEpisode:
    founder = FOUNDERS[slug]
    lines = [
        (founder, "We sell quasar software to schools."),
        ("Elizabeth", "I worry about quasar churn."),
        ("Charles", "How do schools buy quasar software?"),
        (founder, "District budgets, renewed yearly."),
        ("Elizabeth", "What is your quasar margin?"),
        (founder, "Seventy percent."),
        ("Charles", "Quasar sales cycles are too slow for me, I'm out."),
    ]
    turns = tuple(TranscriptTurn(turn_index=i, speaker=s, text=t) for i, (s, t) in enumerate(lines))
    return PrecedentEpisode(
        episode_slug=slug,
        episode_number=int(slug.split("-", 1)[0]),
        source_path=f"source/{slug}.json",
        source_sha256=sha256(slug.encode()).hexdigest(),
        investor_aliases=("Charles", "Charles Hudson"),
        investor_present=True,
        turns=turns,
        decision=PrecedentDecision(
            status="Out",
            context="initial_panel",
            check_tier=None,
            conditions=(),
            evidence=(DecisionEvidence(turn_start=6, turn_end=6, text=turns[6].text),),
            audit_source="fixture",
            audit_notes="fixture",
        ),
    )


def corpus() -> PrecedentCorpus:
    return PrecedentCorpus.from_episodes([episode(slug) for slug in FOUNDERS])


def test_default_shows_every_speaker():
    read = corpus().open_transcript("1-alpha", query_id="Q1")
    assert [t.speaker for t in read.turns] == ["Ann", "Elizabeth", "Charles", "Ann", "Elizabeth", "Ann", "Charles"]


def test_target_turns_only_keeps_investor_and_founder_replies_to_them():
    view = corpus().restrict_to_target_turns()
    read = view.open_transcript("1-alpha", query_id="Q1")
    assert [(t.speaker, t.turn_index) for t in read.turns] == [("Charles", 2), ("Ann", 3), ("Charles", 6)]
    assert "Elizabeth" not in read.evidence.text and "churn" not in read.evidence.text
    hits = view.search("quasar", top_k=3, query_id="Q2").hits
    assert hits and all("Elizabeth" not in hit.excerpt for hit in hits)
    assert any("Charles:" in hit.excerpt for hit in hits)


def test_range_without_target_turns_is_marked_not_empty():
    read = corpus().restrict_to_target_turns().open_transcript("1-alpha", 0, 1, query_id="Q3")
    assert read.turns == () and read.evidence.text.startswith("[No statements by the target investor")


def test_setting_is_omitted_from_saved_config_when_off():
    assert "target_turns_only" not in PrecedentSettings().model_dump(mode="json")
    assert PrecedentSettings(target_turns_only=True).model_dump(mode="json")["target_turns_only"] is True
    assert PrecedentSettings.model_validate({"target_turns_only": True}).target_turns_only is True
