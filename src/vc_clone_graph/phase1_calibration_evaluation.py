"""Evaluation for calibrated Phase 1 rationale predictions."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .phase1_calibration_cases import CalibrationCase, ReferenceAttribute
from .phase1_calibration_models import CalibrationPrediction, CONDITIONS


METHODS = ("raw_phase1", *CONDITIONS)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _prediction_maps(
    cases: Sequence[CalibrationCase], predictions: Sequence[CalibrationPrediction]
) -> tuple[dict[str, dict[tuple[str, str], set[str]]], dict[str, dict[tuple[str, str, str], float]]]:
    selected: dict[str, dict[tuple[str, str], set[str]]] = {
        method: defaultdict(set) for method in METHODS
    }
    scores: dict[str, dict[tuple[str, str, str], float]] = {method: {} for method in METHODS}
    expected = {(case.vc_slug, case.episode_slug, label) for case in cases for label in case.labels}
    for case in cases:
        for label in case.labels:
            key = case.vc_slug, case.episode_slug, label
            signal = case.raw_signals[label]
            scores["raw_phase1"][key] = signal.confidence
            if signal.predicted:
                selected["raw_phase1"][case.key].add(label)
    observed: dict[str, set[tuple[str, str, str]]] = {condition: set() for condition in CONDITIONS}
    for row in predictions:
        if row.condition not in CONDITIONS:
            raise ValueError(f"unknown calibration condition: {row.condition}")
        key = row.vc_slug, row.episode_slug, row.label
        if key in observed[row.condition]:
            raise ValueError(f"duplicate calibration prediction: {row.condition}/{key}")
        observed[row.condition].add(key)
        scores[row.condition][key] = row.score
        if row.predicted:
            selected[row.condition][(row.vc_slug, row.episode_slug)].add(row.label)
    for condition in CONDITIONS:
        if observed[condition] != expected:
            raise ValueError(f"calibration prediction coverage mismatch: {condition}")
    return selected, scores


def _scope_metric(
    cases: Sequence[CalibrationCase],
    method: str,
    selected: dict[str, dict[tuple[str, str], set[str]]],
    scores: dict[str, dict[tuple[str, str, str], float]],
) -> dict[str, Any]:
    tp = fp = fn = 0
    jaccards: list[float] = []
    sizes: list[int] = []
    family_tp = family_fp = family_fn = 0
    true_flat: list[int] = []
    score_flat: list[float] = []
    for case in cases:
        predicted = selected[method][case.key]
        reference = {label for label, value in case.reference_targets.items() if value}
        tp += len(predicted & reference)
        fp += len(predicted - reference)
        fn += len(reference - predicted)
        union = predicted | reference
        jaccards.append(len(predicted & reference) / len(union) if union else 1.0)
        sizes.append(len(predicted))
        predicted_families = {case.families[label] for label in predicted}
        reference_families = {case.families[label] for label in reference}
        family_tp += len(predicted_families & reference_families)
        family_fp += len(predicted_families - reference_families)
        family_fn += len(reference_families - predicted_families)
        for label in case.labels:
            true_flat.append(case.reference_targets[label])
            score_flat.append(scores[method][(case.vc_slug, case.episode_slug, label)])
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    family_precision = _ratio(family_tp, family_tp + family_fp)
    family_recall = _ratio(family_tp, family_tp + family_fn)
    positives = sum(true_flat)
    return {
        "n": len(cases),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": _f1(precision, recall),
        "macro_jaccard": fmean(jaccards),
        "family_f1": _f1(family_precision, family_recall),
        "average_predicted_size": fmean(sizes),
        "average_reference_size": fmean(sum(case.reference_targets.values()) for case in cases),
        "average_precision": (
            float(average_precision_score(true_flat, score_flat)) if positives else None
        ),
        "roc_auc": (
            float(roc_auc_score(true_flat, score_flat))
            if positives and positives < len(true_flat)
            else None
        ),
    }


def _attribute_matches(attribute: ReferenceAttribute, subset: str) -> bool:
    field, value = subset.split(":", 1)
    if field == "primary_and_explicit":
        return attribute.salience == "primary" and attribute.decision_link == "explicit"
    return getattr(attribute, field) == value


def _subset_rows(
    cases: Sequence[CalibrationCase],
    selected: dict[str, dict[tuple[str, str], set[str]]],
) -> list[dict[str, Any]]:
    subsets = (
        "activation:queried",
        "activation:evaluated",
        "utterance_type:question",
        "utterance_type:assessment",
        "utterance_type:decision_reason",
        "decision_link:explicit",
        "primary_and_explicit:true",
    )
    rows: list[dict[str, Any]] = []
    rich = [case for case in cases if case.source_format == "transcript-observed-rationales-v1"]
    for method in METHODS:
        for subset in subsets:
            reference = [
                (case, label)
                for case in rich
                for label, attribute in case.reference_attributes.items()
                if _attribute_matches(attribute, subset)
            ]
            recovered = sum(label in selected[method][case.key] for case, label in reference)
            rows.append(
                {
                    "method": method,
                    "subset": subset.replace(":true", ""),
                    "reference_count": len(reference),
                    "recovered": recovered,
                    "recall": _ratio(recovered, len(reference)),
                }
            )
    return rows


def _action_rows(
    cases: Sequence[CalibrationCase],
    selected: dict[str, dict[tuple[str, str], set[str]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in CONDITIONS:
        removed = correct_removed = harmful_removed = 0
        rescued = correct_rescues = false_rescues = 0
        for case in cases:
            raw = selected["raw_phase1"][case.key]
            calibrated = selected[method][case.key]
            reference = {label for label, value in case.reference_targets.items() if value}
            removed_set = raw - calibrated
            rescued_set = calibrated - raw
            removed += len(removed_set)
            correct_removed += len(removed_set - reference)
            harmful_removed += len(removed_set & reference)
            rescued += len(rescued_set)
            correct_rescues += len(rescued_set & reference)
            false_rescues += len(rescued_set - reference)
        rows.append(
            {
                "method": method,
                "removed": removed,
                "correct_removals": correct_removed,
                "harmful_removals": harmful_removed,
                "rescued": rescued,
                "correct_rescues": correct_rescues,
                "false_rescues": false_rescues,
            }
        )
    return rows


def _bootstrap_rows(
    cases: Sequence[CalibrationCase],
    selected: dict[str, dict[tuple[str, str], set[str]]],
    scores: dict[str, dict[tuple[str, str, str], float]],
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    if iterations <= 0:
        return []
    by_group: dict[str, list[CalibrationCase]] = defaultdict(list)
    for case in cases:
        by_group[case.episode_slug].append(case)
    groups = sorted(by_group)
    rng = np.random.default_rng(seed)
    distributions = {method: [] for method in METHODS}
    for _ in range(iterations):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        rows = [case for group in sampled for case in by_group[str(group)]]
        for method in METHODS:
            distributions[method].append(
                float(_scope_metric(rows, method, selected, scores)["micro_f1"])
            )
    return [
        {
            "method": method,
            "metric": "micro_f1",
            "estimate": _scope_metric(cases, method, selected, scores)["micro_f1"],
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "iterations": iterations,
        }
        for method, values in distributions.items()
    ]


def evaluate_calibration(
    cases: Sequence[CalibrationCase],
    predictions: Sequence[CalibrationPrediction],
    *,
    bootstrap_iterations: int = 1000,
    seed: int = 17,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("calibration evaluation requires cases")
    selected, scores = _prediction_maps(cases, predictions)
    scopes: list[tuple[str, str, list[CalibrationCase]]] = [
        ("all", "all", list(cases)),
        ("rich", "rich", [case for case in cases if case.source_tier != "legacy_fallback"]),
        ("legacy", "legacy", [case for case in cases if case.source_tier == "legacy_fallback"]),
    ]
    for vc_slug in sorted({case.vc_slug for case in cases}):
        scopes.append(("vc", vc_slug, [case for case in cases if case.vc_slug == vc_slug]))
    metric_rows: list[dict[str, Any]] = []
    for scope, value, scoped_cases in scopes:
        if not scoped_cases:
            continue
        for method in METHODS:
            metric_rows.append(
                {
                    "scope": scope,
                    "scope_value": value,
                    "method": method,
                    **_scope_metric(scoped_cases, method, selected, scores),
                }
            )
    return {
        "metric_rows": metric_rows,
        "subset_rows": _subset_rows(cases, selected),
        "action_rows": _action_rows(cases, selected),
        "uncertainty_rows": _bootstrap_rows(
            cases, selected, scores, bootstrap_iterations, seed
        ),
    }
