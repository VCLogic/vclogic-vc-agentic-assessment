from hashlib import sha256
import json

from vc_clone_graph.precedents import (
    DecisionEvidence,
    PrecedentCorpus,
    PrecedentDecision,
    PrecedentEpisode,
    TranscriptTurn,
)
from vc_clone_graph.rehearsal_canary import (
    build_canary,
    canary_precedents,
    score_question,
    score_rationales,
)


class TinyEmbedder:
    def embed(self, texts):
        return [[1.0, float("retention" in text.casefold())] for text in texts]

    def embed_documents(self, texts):
        return self.embed(texts)

    def embed_queries(self, texts):
        return self.embed(texts)


def episode(slug: str = "20-harper-wilde") -> PrecedentEpisode:
    turns = (
        TranscriptTurn(turn_index=0, speaker="Founder", text="We sell subscriptions."),
        TranscriptTurn(turn_index=1, speaker="Charles", text="What does retention look like?"),
        TranscriptTurn(turn_index=2, speaker="Founder", text="Founder response"),
        TranscriptTurn(turn_index=3, speaker="Other Investor", text="I disagree."),
        TranscriptTurn(turn_index=4, speaker="Charles", text="I am out."),
    )
    return PrecedentEpisode(
        episode_slug=slug,
        episode_number=int(slug.split("-", 1)[0]),
        source_path=f"sources/{slug}.json",
        source_sha256=sha256(slug.encode()).hexdigest(),
        investor_aliases=("Charles",),
        investor_present=True,
        turns=turns,
        decision=PrecedentDecision(
            status="Out",
            context="initial_panel",
            check_tier=None,
            conditions=(),
            evidence=(
                DecisionEvidence(turn_start=4, turn_end=4, text="I am out."),
            ),
            audit_source="fixture",
            audit_notes="fixture",
        ),
    )


def test_canary_excludes_target_decision_and_non_founder_answers() -> None:
    canary = build_canary(
        episode(),
        target_vc="Charles",
        pitch_text="We sell subscriptions.",
        founder_speakers={"Founder"},
        investor_speakers={"Charles", "Other Investor"},
        observed_question_labels={1: {"traction_customer_validation"}},
    )
    serialized = json.dumps(canary.model_dump(mode="json"))

    assert "actual_decision" not in serialized
    assert "I am out" not in serialized
    assert "Other Investor" not in serialized
    assert canary.founder_answers == ("Founder response",)
    assert canary.observed_questions[0].rationale_labels == (
        "traction_customer_validation",
    )


def test_question_metrics_include_semantic_and_rationale_overlap() -> None:
    metrics = score_question(
        generated="How often do customers return?",
        generated_labels={"traction_customer_validation"},
        observed="What does retention look like?",
        observed_labels={"traction_customer_validation"},
        embedder=TinyEmbedder(),
    )

    assert 0 <= metrics.semantic_similarity <= 1
    assert metrics.rationale_precision == 1.0
    assert metrics.rationale_recall == 1.0
    assert metrics.rationale_f1 == 1.0


def test_rationale_metrics_measure_coverage_direction_and_salience() -> None:
    metrics = score_rationales(
        predicted=[
            {"taxonomy_label": "founder_execution", "direction": "positive", "salience": "primary"},
            {"taxonomy_label": "market_size_assessment", "direction": "negative", "salience": "secondary"},
        ],
        observed=[
            {"taxonomy_label": "founder_execution", "direction": "positive", "salience": "primary"},
        ],
    )

    assert metrics.precision == 0.5
    assert metrics.recall == 1.0
    assert metrics.direction_accuracy == 1.0
    assert metrics.salience_accuracy == 1.0


def test_canary_runtime_filters_current_episode() -> None:
    corpus = PrecedentCorpus.from_episodes(
        (episode("20-harper-wilde"), episode("21-another-company")),
        TinyEmbedder(),
    )

    filtered = canary_precedents(corpus, "20-harper-wilde")

    assert "20-harper-wilde" not in {
        row.episode_slug for row in filtered.list_episodes()
    }
