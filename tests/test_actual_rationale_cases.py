from __future__ import annotations

from dataclasses import replace

import pytest

from vc_clone_graph.actual_rationale_cases import build_rationale_model_cases
from vc_clone_graph.phase1_evaluation import (
    Phase1Case,
    PredictedRationale,
    ReferenceRationale,
)


LABELS = ("founder_execution", *(f"label_{index}" for index in range(43)))


def phase1_case() -> Phase1Case:
    return Phase1Case(
        vc_slug="alpha",
        vc_name="Alpha",
        episode_slug="1-example",
        actual_decision="In",
        source_tier="newly_extracted",
        source_format="transcript-observed-rationales-v1",
        predicted=(PredictedRationale("founder_execution", "positive", "primary", 0.7, "why"),),
        reference=(ReferenceRationale("founder_execution", "positive", "primary", 0.99, "evaluated", "assessment", "supports", ("words",)),),
        artifact_path=None,
        reference_path=None,
    )


def test_build_cases_aligns_actual_and_predicted_features_without_target() -> None:
    result = build_rationale_model_cases([phase1_case()], LABELS)

    assert len(result) == 1
    case = result[0]
    assert case.target == 1
    assert case.actual_features == case.predicted_features
    assert case.actual_items == case.predicted_items
    assert not any("decision" in key for key in case.actual_features)


def test_duplicate_case_keys_are_rejected() -> None:
    case = phase1_case()
    with pytest.raises(ValueError, match="duplicate"):
        build_rationale_model_cases([case, case], LABELS)


def test_invalid_outcome_is_rejected() -> None:
    with pytest.raises(ValueError, match="decision"):
        build_rationale_model_cases(
            [replace(phase1_case(), actual_decision="Maybe")], LABELS
        )
