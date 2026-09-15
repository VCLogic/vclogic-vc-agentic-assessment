from pathlib import Path

import pytest

from vc_clone_graph.phase1_calibration_cases import (
    build_calibration_cases,
    resolve_pitch_path,
)
from vc_clone_graph.phase1_evaluation import (
    Phase1Case,
    PredictedRationale,
    ReferenceRationale,
    TaxonomyLabel,
)


def _case() -> Phase1Case:
    return Phase1Case(
        vc_slug="example-vc",
        vc_name="Example VC",
        episode_slug="1-example",
        actual_decision="In",
        source_tier="newly_extracted",
        source_format="transcript-observed-rationales-v1",
        predicted=(
            PredictedRationale(
                "founder_execution", "positive", "primary", 0.8, "Built it."
            ),
        ),
        reference=(
            ReferenceRationale(
                "founder_execution",
                "positive",
                "primary",
                0.95,
                "evaluated",
                "decision_reason",
                "explicit",
                ("Built it.",),
            ),
        ),
        artifact_path=Path("artifact.json"),
        reference_path=Path("reference.json"),
    )


def test_build_calibration_case_encodes_all_labels_and_raw_signal(tmp_path: Path) -> None:
    pitch = tmp_path / "inputs/data/investors/example-vc-fund/pitches/1-example.txt"
    pitch.parent.mkdir(parents=True)
    pitch.write_text("Founder: We built the product.\n", encoding="utf-8")
    taxonomy = {
        "founder_execution": TaxonomyLabel(
            "founder_execution", "Evidence of execution.", "founder_team"
        ),
        "market_size_assessment": TaxonomyLabel(
            "market_size_assessment", "Market scale.", "market_opportunity"
        ),
    }

    result = build_calibration_cases(tmp_path, [_case()], taxonomy)

    assert len(result) == 1
    row = result[0]
    assert row.key == ("example-vc", "1-example")
    assert row.pitch_text == "Founder: We built the product."
    assert row.reference_targets == {
        "founder_execution": 1,
        "market_size_assessment": 0,
    }
    assert row.raw_signals["founder_execution"].confidence == 0.8
    assert row.raw_signals["founder_execution"].signed_salience == 2.0
    assert row.raw_signals["market_size_assessment"].predicted is False
    assert row.reference_attributes["founder_execution"].decision_link == "explicit"


def test_resolve_pitch_path_rejects_ambiguous_packages(tmp_path: Path) -> None:
    for package in ("example-vc-one", "example-vc-two"):
        pitch = tmp_path / f"inputs/data/investors/{package}/pitches/1-example.txt"
        pitch.parent.mkdir(parents=True)
        pitch.write_text("pitch", encoding="utf-8")

    with pytest.raises(ValueError, match="ambiguous audited pitch"):
        resolve_pitch_path(tmp_path, "example-vc", "1-example")


def test_resolve_pitch_path_rejects_missing_pitch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing audited pitch"):
        resolve_pitch_path(tmp_path, "example-vc", "1-example")
