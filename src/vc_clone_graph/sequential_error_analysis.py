"""Observable case-level diagnosis for rationale and decision errors."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from .phase1_evaluation import Phase1Case, PredictedRationale, ReferenceRationale
from .phase2_calibration_evaluation import MethodPrediction


def _highest_confidence(items: Sequence[object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in items:
        label = str(item.label)
        if label not in result or float(item.confidence) > float(result[label].confidence):
            result[label] = item
    return result


def analyze_error_case(
    case: Phase1Case,
    prediction: MethodPrediction,
    *,
    high_coverage_threshold: float = 0.70,
) -> dict[str, object]:
    """Describe one wrong decision using only observable comparison signals."""
    if (
        case.vc_slug != prediction.vc_slug
        or case.episode_slug != prediction.episode_slug
    ):
        raise ValueError("Phase 1 case and decision prediction do not align")
    expected_target = int(case.actual_decision == "In")
    if prediction.target != expected_target:
        raise ValueError("Phase 1 case and decision target do not align")
    predicted = _highest_confidence(case.predicted)
    reference = _highest_confidence(case.reference)
    predicted_labels, reference_labels = set(predicted), set(reference)
    matched = predicted_labels & reference_labels
    missing = sorted(reference_labels - predicted_labels)
    false = sorted(predicted_labels - reference_labels)
    key_labels = {
        label
        for label, rationale in reference.items()
        if rationale.salience == "primary" or rationale.decision_link == "explicit"
    }
    missing_key = sorted(key_labels - predicted_labels)
    direction_mismatches = sorted(
        label
        for label in matched
        if predicted[label].direction != reference[label].direction
    )
    salience_mismatches = sorted(
        label
        for label in matched
        if predicted[label].salience != reference[label].salience
    )
    rationale_recall = (
        len(matched) / len(reference_labels) if reference_labels else 1.0
    )
    wrong = prediction.predicted != prediction.target
    flags: set[str] = set()
    if missing_key:
        flags.add("phase1_key_rationale_missing")
    if len(predicted_labels) > 2 * max(1, len(reference_labels)):
        flags.add("phase1_overproduction")
    if direction_mismatches or salience_mismatches:
        flags.add("phase1_direction_or_salience_mismatch")
    if wrong and rationale_recall >= high_coverage_threshold:
        flags.add("decision_error_despite_high_rationale_coverage")
    if case.source_tier == "legacy_fallback" or len(reference_labels) < 2:
        flags.add("limited_reference_observability")
    if prediction.target == 1 and prediction.predicted == 0:
        error_type = "missed_in"
    elif prediction.target == 0 and prediction.predicted == 1:
        error_type = "false_in"
    else:
        error_type = "correct"
    return {
        "vc_slug": case.vc_slug,
        "vc_name": case.vc_name,
        "episode_slug": case.episode_slug,
        "method": prediction.method,
        "actual_decision": "In" if prediction.target else "Out",
        "predicted_decision": "In" if prediction.predicted else "Out",
        "score": prediction.score,
        "error_type": error_type,
        "source_tier": case.source_tier,
        "source_format": case.source_format,
        "predicted_rationale_count": len(predicted_labels),
        "reference_rationale_count": len(reference_labels),
        "rationale_recall": rationale_recall,
        "missing_rationales": ";".join(missing),
        "false_rationales": ";".join(false),
        "missing_key_rationales": ";".join(missing_key),
        "direction_mismatches": ";".join(direction_mismatches),
        "salience_mismatches": ";".join(salience_mismatches),
        "diagnostic_flags": ";".join(sorted(flags)),
    }


def analyze_error_cases(
    cases: Sequence[Phase1Case],
    methods: Mapping[str, Sequence[MethodPrediction]],
    *,
    high_coverage_threshold: float = 0.70,
) -> list[dict[str, object]]:
    """Return every false In and missed In for every supplied method."""
    lookup = {(case.vc_slug, case.episode_slug): case for case in cases}
    rows: list[dict[str, object]] = []
    for name, predictions in methods.items():
        for prediction in predictions:
            if prediction.predicted == prediction.target:
                continue
            key = (prediction.vc_slug, prediction.episode_slug)
            if key not in lookup:
                raise ValueError(f"missing aligned Phase 1 case: {key}")
            row = analyze_error_case(
                lookup[key],
                prediction,
                high_coverage_threshold=high_coverage_threshold,
            )
            row["method"] = name
            rows.append(row)
    return sorted(rows, key=lambda row: (
        str(row["method"]), str(row["vc_slug"]), str(row["episode_slug"])
    ))


def error_summary_rows(
    error_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Count decision errors and observable diagnostic flags by VC and method."""
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in error_rows:
        key = (str(row["method"]), str(row["vc_slug"]), str(row["vc_name"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for (method, vc_slug, vc_name), rows in sorted(grouped.items()):
        flags = Counter(
            flag
            for row in rows
            for flag in str(row["diagnostic_flags"]).split(";")
            if flag
        )
        output.append({
            "method": method,
            "vc_slug": vc_slug,
            "vc_name": vc_name,
            "false_ins": sum(row["error_type"] == "false_in" for row in rows),
            "missed_ins": sum(row["error_type"] == "missed_in" for row in rows),
            "phase1_key_rationale_missing": flags["phase1_key_rationale_missing"],
            "phase1_overproduction": flags["phase1_overproduction"],
            "phase1_direction_or_salience_mismatch": flags[
                "phase1_direction_or_salience_mismatch"
            ],
            "decision_error_despite_high_rationale_coverage": flags[
                "decision_error_despite_high_rationale_coverage"
            ],
            "limited_reference_observability": flags[
                "limited_reference_observability"
            ],
        })
    return output
