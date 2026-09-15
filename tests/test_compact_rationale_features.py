from __future__ import annotations

from vc_clone_graph.compact_rationale_features import (
    compact_decision_view,
    compact_feature_dictionary,
    compact_rationale_view,
)
from vc_clone_graph.phase2_calibration_evaluation import Phase2CalibrationRecord


LABELS = ("founder_execution", *(f"label_{index}" for index in range(43)))


def record() -> Phase2CalibrationRecord:
    semantic = {
        "rationale__founder_execution__signed_confidence": 0.8,
        "rationale__founder_execution__salience__secondary": 1.0,
        "rationale__founder_execution__direction__positive": 1.0,
        "rationale__founder_execution__count": 1.0,
        "rationale__founder_execution__pitch_evidence_count": 2.0,
        "rationale__founder_execution__wiki_evidence_count": 1.0,
        "rationale__founder_execution__historical_evidence_count": 3.0,
        "rationale__founder_execution__controlling": 1.0,
        "constraint__portfolio_conflict__triggered__blocking": 1.0,
        "vc_rationale__alpha__founder_execution__signed_confidence": 0.9,
    }
    phase2 = {
        "investment_likelihood": 0.7,
        "review_priority_score": 0.8,
        "decision_confidence": 0.75,
        "signed_decision_confidence": 0.75,
        "decision_in": 1.0,
        "vc__alpha": 1.0,
    }
    return Phase2CalibrationRecord(
        vc_slug="alpha",
        vc_name="Alpha",
        episode_slug="1-example",
        group="1-example",
        target=1,
        raw_decision=1,
        raw_likelihood=0.7,
        phase2_features=phase2,
        combined_features={**semantic, **phase2},
        semantic_features={**semantic, **phase2},
        artifact_root="runs/alpha/1-example",
        phase1_sha256="a" * 64,
        phase2_sha256="b" * 64,
    )


def test_compact_rationale_view_has_one_signed_value_per_taxonomy_label() -> None:
    features = compact_rationale_view(record(), LABELS)

    rationale_keys = [name for name in features if name.startswith("rationale_core__")]
    assert len(rationale_keys) == 44
    assert set(rationale_keys) == {f"rationale_core__{label}" for label in LABELS}
    assert features["rationale_core__founder_execution"] == 0.4
    assert features["aggregate__positive_count"] == 1.0
    assert features["aggregate__pitch_evidence_total"] == 2.0
    assert features["aggregate__external_evidence_total"] == 4.0
    assert features["constraint__triggered_hard"] == 1.0
    assert features["constraint__portfolio_conflict_triggered"] == 1.0
    assert len(features) == 58


def test_compact_views_exclude_identity_target_and_phase2_from_rationale_only() -> None:
    rationale = compact_rationale_view(record(), LABELS)
    decision = compact_decision_view(record(), LABELS)

    assert not any(
        "target" in name or "actual" in name or "episode" in name
        for name in decision
    )
    assert not any(name.startswith("vc__") for name in decision)
    assert "phase2__investment_likelihood" not in rationale
    assert decision["phase2__investment_likelihood"] == 0.7
    assert "phase2__decision_in" not in decision
    assert len(decision) == 62


def test_compact_feature_dictionary_describes_every_feature() -> None:
    rows = compact_feature_dictionary(LABELS)

    assert len(rows) == 62
    assert len({row["feature"] for row in rows}) == 62
    founder = next(
        row for row in rows if row["feature"] == "rationale_core__founder_execution"
    )
    assert founder["family"] == "rationale"
    assert founder["directionality"] == "signed"
