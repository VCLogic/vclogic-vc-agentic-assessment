from __future__ import annotations

from vc_clone_graph.personalized_cases import PersonalizedCase
from vc_clone_graph.personalized_features import (
    build_structured_feature_views,
    pitch_indicator_features,
)


def test_structured_views_emit_every_taxonomy_label_and_missingness() -> None:
    case = PersonalizedCase(
        vc_slug="alpha",
        vc_input_slug="alpha-fund",
        vc_name="Alpha",
        episode_slug="1-example",
        group="1-example",
        target=1,
        pitch_text="We have $2m ARR, growing 30%, but CAC is unknown.",
        phase1_text="market rationale",
        phase2_text="decision synthesis",
        phase1_features={
            "label__market_size__signed_confidence": 0.8,
            "label__market_size__confidence_sum": 0.8,
            "label_direction__market_size__positive__primary__count": 1.0,
            "label__market_size__pitch_evidence_count": 2.0,
        },
        phase2_features={
            "decision_in": 1.0,
            "investment_likelihood": 0.7,
        },
        wiki_text="early stage investor",
        actual_rationale_targets={"market_size": 1},
        actual_rationale_available=True,
        actual_rationale_source_tier="test",
        actual_rationale_source_format="test",
        source_hashes={"pitch": "a" * 64},
    )

    views = build_structured_feature_views(
        case, taxonomy_labels=("market_size", "founder_execution")
    )

    assert views.phase1["p1__market_size__activated"] == 1.0
    assert views.phase1["p1__market_size__direction_positive"] == 1.0
    assert views.phase1["p1__market_size__salience_primary"] == 1.0
    assert views.phase1["p1__market_size__pitch_evidence_count"] == 2.0
    assert views.phase1["p1__founder_execution__activated"] == 0.0
    assert views.phase2["p2__investment_likelihood"] == 0.7
    assert views.phase2["p2_missing__decision_confidence"] == 1.0
    assert views.pitch["pitch__currency_or_revenue_mentions"] >= 1.0
    assert views.pitch["pitch__percentage_mentions"] == 1.0
    assert views.pitch["pitch__unknown_mentions"] == 1.0


def test_pitch_indicators_are_label_blind_and_deterministic() -> None:
    text = "Founder built two companies. We have 12 customers and $50k MRR. Competitor data is unknown."

    first = pitch_indicator_features(text)
    second = pitch_indicator_features(text)

    assert first == second
    assert first["pitch__founder_history_mentions"] == 1.0
    assert first["pitch__traction_mentions"] >= 2.0
    assert first["pitch__competitor_mentions"] == 1.0
    assert not any("target" in key or "actual" in key for key in first)
