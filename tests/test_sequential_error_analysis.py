from __future__ import annotations

from vc_clone_graph.phase1_evaluation import (
    Phase1Case,
    PredictedRationale,
    ReferenceRationale,
)
from vc_clone_graph.phase2_calibration_evaluation import MethodPrediction
from vc_clone_graph.sequential_error_analysis import (
    analyze_error_case,
    analyze_error_cases,
    error_summary_rows,
)


def predicted(label: str, direction: str = "positive", salience: str = "primary"):
    return PredictedRationale(label, direction, salience, 0.8, f"Why {label}")


def reference(
    label: str,
    direction: str = "positive",
    salience: str = "primary",
    decision_link: str = "explicit",
):
    return ReferenceRationale(
        label, direction, salience, 0.9, "evaluated", "decision_reason",
        decision_link, (f"Evidence {label}",),
    )


def case(predictions, references, *, source_tier="newly_extracted"):
    return Phase1Case(
        vc_slug="alpha", vc_name="Alpha", episode_slug="1-example",
        actual_decision="In", source_tier=source_tier,
        source_format="transcript-observed-rationales-v1",
        predicted=tuple(predictions), reference=tuple(references),
        artifact_path=None, reference_path=None,
    )


def missed_in_prediction() -> MethodPrediction:
    return MethodPrediction(
        vc_slug="alpha", vc_name="Alpha", episode_slug="1-example",
        group="1-example", target=1, method="raw_phase2", score=0.3,
        predicted=0,
    )


def test_error_flags_are_observable_and_do_not_claim_tacit_causality() -> None:
    item = case(
        [predicted("founder_execution")],
        [reference("founder_execution"), reference("business_viability_concern")],
    )

    row = analyze_error_case(item, missed_in_prediction())

    assert row["missing_key_rationales"] == "business_viability_concern"
    assert "phase1_key_rationale_missing" in row["diagnostic_flags"]
    assert "tacit" not in row["diagnostic_flags"]
    assert row["error_type"] == "missed_in"


def test_high_coverage_decision_error_is_distinguished_from_phase1_failure() -> None:
    item = case(
        [predicted("founder_execution"), predicted("market_size_assessment")],
        [reference("founder_execution"), reference("market_size_assessment")],
    )

    row = analyze_error_case(item, missed_in_prediction(), high_coverage_threshold=0.70)

    assert row["rationale_recall"] == 1.0
    assert "decision_error_despite_high_rationale_coverage" in row["diagnostic_flags"]


def test_error_collection_and_summary_keep_episode_rows() -> None:
    item = case(
        [predicted("founder_execution", direction="negative", salience="secondary")],
        [reference("founder_execution")],
        source_tier="legacy_fallback",
    )
    rows = analyze_error_cases([item], {"raw_phase2": [missed_in_prediction()]})
    summary = error_summary_rows(rows)

    assert len(rows) == 1
    assert "phase1_direction_or_salience_mismatch" in rows[0]["diagnostic_flags"]
    assert "limited_reference_observability" in rows[0]["diagnostic_flags"]
    assert summary[0]["missed_ins"] == 1
