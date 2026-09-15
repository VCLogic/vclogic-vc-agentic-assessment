from pathlib import Path

from vc_clone_graph.phase1_calibration_cases import (
    CalibrationCase,
    RawLabelSignal,
    ReferenceAttribute,
)
from vc_clone_graph.phase1_calibration_evaluation import evaluate_calibration
from vc_clone_graph.phase1_calibration_models import CalibrationPrediction


def _case(index: int, reference: str, raw: str) -> CalibrationCase:
    labels = ("a", "b")
    return CalibrationCase(
        vc_slug=f"vc{index % 2}",
        vc_name=f"VC {index % 2}",
        episode_slug=f"e{index}",
        actual_decision="In" if index % 2 == 0 else "Out",
        source_tier="newly_extracted" if index != 3 else "legacy_fallback",
        source_format="transcript-observed-rationales-v1" if index != 3 else "legacy",
        pitch_text="pitch",
        pitch_path=Path("pitch.txt"),
        pitch_sha256="hash",
        labels=labels,
        families={"a": "family-a", "b": "family-b"},
        raw_signals={
            label: RawLabelSignal(
                label == raw,
                0.8 if label == raw else 0.0,
                "positive" if label == raw else "absent",
                "primary" if label == raw else "absent",
                2.0 if label == raw else 0.0,
            )
            for label in labels
        },
        reference_targets={label: int(label == reference) for label in labels},
        reference_attributes={
            reference: ReferenceAttribute("evaluated", "decision_reason", "explicit", "primary")
        },
        phase1_case=None,  # type: ignore[arg-type]
    )


def test_evaluation_reports_metrics_sensitivity_and_rescues() -> None:
    cases = [
        _case(0, "a", "b"),
        _case(1, "b", "b"),
        _case(2, "a", "a"),
        _case(3, "b", "a"),
    ]
    predictions = []
    for case in cases:
        for condition in ("filtering_only", "selector_expansion", "semantic_control"):
            for label in case.labels:
                predicted = (
                    case.reference_targets[label] == 1
                    if condition == "selector_expansion"
                    else case.raw_signals[label].predicted
                )
                predictions.append(
                    CalibrationPrediction(
                        case.vc_slug,
                        case.episode_slug,
                        label,
                        condition,
                        0.9 if predicted else 0.1,
                        predicted,
                        case.raw_signals[label].predicted,
                        bool(case.reference_targets[label]),
                    )
                )

    result = evaluate_calibration(cases, predictions, bootstrap_iterations=20)

    expansion = next(
        row
        for row in result["metric_rows"]
        if row["scope"] == "all" and row["method"] == "selector_expansion"
    )
    assert expansion["micro_f1"] == 1.0
    assert expansion["average_predicted_size"] == 1.0
    diagnostic = next(
        row for row in result["action_rows"] if row["method"] == "selector_expansion"
    )
    assert diagnostic["correct_rescues"] == 2
    assert diagnostic["false_rescues"] == 0
    assert {row["scope"] for row in result["metric_rows"]} >= {
        "all",
        "rich",
        "legacy",
        "vc",
    }
    assert any(row["subset"] == "decision_link:explicit" for row in result["subset_rows"])
    assert result["uncertainty_rows"]
