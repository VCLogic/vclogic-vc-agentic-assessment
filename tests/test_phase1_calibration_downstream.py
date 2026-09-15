from pathlib import Path

from vc_clone_graph.phase1_calibration_cases import CalibrationCase, RawLabelSignal
from vc_clone_graph.phase1_calibration_downstream import build_downstream_cases
from vc_clone_graph.phase1_calibration_models import CalibrationPrediction
from vc_clone_graph.phase1_evaluation import (
    Phase1Case,
    PredictedRationale,
    ReferenceRationale,
)


def test_downstream_adapter_keeps_raw_attributes_and_marks_rescues_neutral() -> None:
    labels = ("a", "b", *(f"c{index:02d}" for index in range(42)))
    phase1 = Phase1Case(
        vc_slug="vc",
        vc_name="VC",
        episode_slug="e1",
        actual_decision="In",
        source_tier="newly_extracted",
        source_format="transcript-observed-rationales-v1",
        predicted=(PredictedRationale("a", "positive", "primary", 0.8, "why"),),
        reference=(
            ReferenceRationale(
                "a", "positive", "primary", 0.9, "evaluated", "assessment", "none", ("x",)
            ),
        ),
        artifact_path=Path("artifact"),
        reference_path=Path("reference"),
    )
    case = CalibrationCase(
        "vc",
        "VC",
        "e1",
        "In",
        "newly_extracted",
        "transcript-observed-rationales-v1",
        "pitch",
        Path("pitch"),
        "hash",
        labels,
        {label: f"family-{label}" for label in labels},
        {
            label: (
                RawLabelSignal(True, 0.8, "positive", "primary", 2.0)
                if label == "a"
                else RawLabelSignal(False, 0.0, "absent", "absent", 0.0)
            )
            for label in labels
        },
        {label: int(label == "a") for label in labels},
        {},
        phase1,
    )
    predictions = [
        CalibrationPrediction(
            "vc",
            "e1",
            label,
            "selector_expansion",
            0.7 if label in {"a", "b"} else 0.1,
            label in {"a", "b"},
            label == "a",
            label == "a",
        )
        for label in labels
    ]

    result = build_downstream_cases([case], predictions, labels, "selector_expansion")

    assert result[0].predicted_items == (
        ("a", "positive", "primary"),
        ("b", "neutral", "secondary"),
    )
    assert result[0].target == 1
    assert not any("decision" in name for name in result[0].predicted_features)
