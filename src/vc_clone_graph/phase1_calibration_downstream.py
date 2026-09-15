"""Decision-model adapters for calibrated Phase 1 rationale sets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from .actual_rationale_cases import RationaleModelCase, build_rationale_model_cases
from .actual_rationale_features import structured_rationale_features
from .actual_rationale_models import RationalePrediction, nested_per_vc_predictions
from .phase1_calibration_cases import CalibrationCase
from .phase1_calibration_models import CalibrationPrediction


def build_downstream_cases(
    cases: Sequence[CalibrationCase],
    predictions: Sequence[CalibrationPrediction],
    taxonomy: Mapping[str, object] | Sequence[str],
    condition: str,
) -> list[RationaleModelCase]:
    base = build_rationale_model_cases([case.phase1_case for case in cases], taxonomy)
    selected: dict[tuple[str, str], set[str]] = {case.key: set() for case in cases}
    observed: set[tuple[str, str, str]] = set()
    for row in predictions:
        if row.condition != condition:
            continue
        key = row.vc_slug, row.episode_slug, row.label
        if key in observed:
            raise ValueError(f"duplicate calibrated downstream row: {key}")
        observed.add(key)
        if row.predicted:
            selected[(row.vc_slug, row.episode_slug)].add(row.label)
    expected = {(case.vc_slug, case.episode_slug, label) for case in cases for label in case.labels}
    if observed != expected:
        raise ValueError(f"calibrated downstream coverage mismatch: {condition}")
    result: list[RationaleModelCase] = []
    for case, original in zip(cases, base, strict=True):
        rows: list[dict[str, object]] = []
        items: list[tuple[str, str, str]] = []
        for label in sorted(selected[case.key]):
            signal = case.raw_signals[label]
            direction = signal.direction if signal.predicted else "neutral"
            salience = signal.salience if signal.predicted else "secondary"
            rows.append(
                {
                    "taxonomy_label": label,
                    "direction": direction,
                    "salience": salience,
                }
            )
            items.append((label, direction, salience))
        predicted_features = structured_rationale_features(
            rows, taxonomy, label_field="taxonomy_label"
        )
        if set(predicted_features) != set(original.actual_features):
            raise ValueError(f"calibrated feature contract mismatch: {case.key}")
        if not all(math.isfinite(value) for value in predicted_features.values()):
            raise ValueError(f"non-finite calibrated feature: {case.key}")
        result.append(
            RationaleModelCase(
                vc_slug=original.vc_slug,
                vc_name=original.vc_name,
                episode_slug=original.episode_slug,
                group=original.group,
                target=original.target,
                source_tier=original.source_tier,
                source_format=original.source_format,
                actual_features=original.actual_features,
                predicted_features=predicted_features,
                actual_items=original.actual_items,
                predicted_items=tuple(items),
            )
        )
    return result


def run_downstream_models(
    cases: Sequence[CalibrationCase],
    predictions: Sequence[CalibrationPrediction],
    taxonomy: Mapping[str, object] | Sequence[str],
    *,
    conditions: Sequence[str],
    inner_splits: int = 3,
    seed: int = 20260817,
    n_jobs: int = 1,
) -> dict[str, list[RationalePrediction]]:
    raw_cases = build_rationale_model_cases(
        [case.phase1_case for case in cases], taxonomy
    )
    result: dict[str, list[RationalePrediction]] = {
        "raw_phase1": nested_per_vc_predictions(
            raw_cases,
            model_family="logistic",
            source_condition="actual_to_predicted",
            inner_splits=inner_splits,
            seed=seed - 2,
            n_jobs=n_jobs,
        ),
        "actual_rationale_oracle": nested_per_vc_predictions(
            raw_cases,
            model_family="logistic",
            source_condition="actual_to_actual",
            inner_splits=inner_splits,
            seed=seed - 1,
            n_jobs=n_jobs,
        ),
    }
    for offset, condition in enumerate(conditions):
        adapted = build_downstream_cases(cases, predictions, taxonomy, condition)
        rows = nested_per_vc_predictions(
            adapted,
            model_family="logistic",
            source_condition="actual_to_predicted",
            inner_splits=inner_splits,
            seed=seed + offset * 10_000,
            n_jobs=n_jobs,
        )
        result[condition] = rows
    return result
