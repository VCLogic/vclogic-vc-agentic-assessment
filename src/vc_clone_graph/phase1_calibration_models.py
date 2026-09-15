"""Grouped nested multi-label models for Phase 1 rationale calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .phase1_calibration_cases import CalibrationCase
from .phase1_calibration_features import FoldFeatures, build_fold_features


CONDITIONS = ("filtering_only", "selector_expansion", "semantic_control")


@dataclass(frozen=True)
class CalibrationPrediction:
    vc_slug: str
    episode_slug: str
    label: str
    condition: str
    score: float
    predicted: bool
    raw_predicted: bool
    reference: bool


@dataclass(frozen=True)
class CalibrationFoldProvenance:
    condition: str
    held_episode_slug: str
    training_episode_slugs: tuple[str, ...]
    regularization_c: float
    threshold: float
    inner_objective_f1: float


@dataclass(frozen=True)
class CalibrationRunResult:
    predictions: tuple[CalibrationPrediction, ...]
    provenance: tuple[CalibrationFoldProvenance, ...]


def apply_condition(
    scores: np.ndarray,
    raw_candidates: np.ndarray,
    *,
    threshold: float,
    condition: str,
) -> np.ndarray:
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported calibration condition: {condition}")
    selected = np.asarray(scores, dtype=float) >= threshold
    if condition == "filtering_only":
        selected &= np.asarray(raw_candidates, dtype=bool)
    return selected


def _targets(cases: Sequence[CalibrationCase], indices: Sequence[int]) -> np.ndarray:
    labels = cases[0].labels
    return np.asarray(
        [[cases[index].reference_targets[label] for label in labels] for index in indices],
        dtype=int,
    )


def _raw_candidates(cases: Sequence[CalibrationCase], indices: Sequence[int]) -> np.ndarray:
    labels = cases[0].labels
    return np.asarray(
        [[cases[index].raw_signals[label].predicted for label in labels] for index in indices],
        dtype=bool,
    )


def _tier_weights(cases: Sequence[CalibrationCase], indices: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [0.35 if cases[index].source_tier == "legacy_fallback" else 1.0 for index in indices],
        dtype=float,
    )


def _fit_predict(
    features: FoldFeatures,
    evaluate: np.ndarray,
    base_weights: np.ndarray,
    regularization_c: float,
) -> np.ndarray:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(features.train)
    evaluate_x = scaler.transform(evaluate)
    scores = np.zeros((evaluate.shape[0], features.targets.shape[1]), dtype=float)
    for label_index in range(features.targets.shape[1]):
        target = features.targets[:, label_index]
        positives = target == 1
        negatives = ~positives
        if not positives.any() or not negatives.any():
            scores[:, label_index] = (
                float(np.average(target, weights=base_weights))
                if base_weights.sum()
                else float(target.mean())
            )
            continue
        positive_mass = float(base_weights[positives].sum())
        negative_mass = float(base_weights[negatives].sum())
        weights = base_weights.copy()
        weights[positives] *= 0.5 / positive_mass
        weights[negatives] *= 0.5 / negative_mass
        model = LogisticRegression(
            C=regularization_c,
            solver="liblinear",
            max_iter=600,
            random_state=17,
        )
        model.fit(train_x, target, sample_weight=weights)
        scores[:, label_index] = model.predict_proba(evaluate_x)[:, 1]
    return scores


def _micro_metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    tp = int(np.logical_and(predicted, target).sum())
    fp = int(np.logical_and(predicted, ~target).sum())
    fn = int(np.logical_and(~predicted, target).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _select_parameters(
    cases: Sequence[CalibrationCase],
    embeddings: np.ndarray,
    outer_train: Sequence[int],
    *,
    condition: str,
    include_phase1: bool,
    c_grid: Sequence[float],
    thresholds: Sequence[float],
    inner_splits: int,
    pca_components: int,
) -> tuple[float, float, float]:
    groups = np.asarray([cases[index].episode_slug for index in outer_train])
    unique_groups = np.unique(groups)
    splits = min(inner_splits, len(unique_groups))
    if splits < 2:
        return float(c_grid[0]), float(thresholds[0]), 0.0
    splitter = GroupKFold(n_splits=splits)
    predictions = {
        float(c): np.zeros((len(outer_train), len(cases[0].labels)), dtype=float)
        for c in c_grid
    }
    for fit_local, validate_local in splitter.split(np.zeros(len(outer_train)), groups=groups):
        fit_indices = [outer_train[index] for index in fit_local]
        validate_indices = [outer_train[index] for index in validate_local]
        fold = build_fold_features(
            cases,
            embeddings,
            fit_indices,
            validate_indices,
            include_phase1=include_phase1,
            pca_components=pca_components,
        )
        weights = _tier_weights(cases, fit_indices)
        for regularization_c in c_grid:
            scores = _fit_predict(fold, fold.evaluate, weights, float(regularization_c))
            predictions[float(regularization_c)][validate_local] = scores

    target = _targets(cases, outer_train)
    raw = _raw_candidates(cases, outer_train)
    rich = np.asarray(
        [cases[index].source_tier != "legacy_fallback" for index in outer_train],
        dtype=bool,
    )
    if not rich.any():
        rich[:] = True
    best: tuple[tuple[float, float, float, float, float], float, float, float] | None = None
    for regularization_c in c_grid:
        score = predictions[float(regularization_c)]
        for threshold in thresholds:
            selected = apply_condition(
                score, raw, threshold=float(threshold), condition=condition
            )
            precision, recall, f1 = _micro_metrics(selected[rich], target[rich])
            rank = (f1, min(recall, 0.50), precision, recall, -float(regularization_c))
            candidate = (rank, float(regularization_c), float(threshold), f1)
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    return best[1], best[2], best[3]


def _outer_fold(
    held_group: str,
    cases: Sequence[CalibrationCase],
    embeddings: np.ndarray,
    c_grid: Sequence[float],
    thresholds: Sequence[float],
    inner_splits: int,
    pca_components: int,
) -> tuple[list[CalibrationPrediction], list[CalibrationFoldProvenance]]:
    train_indices = [index for index, case in enumerate(cases) if case.episode_slug != held_group]
    held_indices = [index for index, case in enumerate(cases) if case.episode_slug == held_group]
    training_groups = tuple(sorted({cases[index].episode_slug for index in train_indices}))
    results: list[CalibrationPrediction] = []
    provenance: list[CalibrationFoldProvenance] = []
    for condition, include_phase1 in (
        ("filtering_only", True),
        ("selector_expansion", True),
        ("semantic_control", False),
    ):
        regularization_c, threshold, inner_f1 = _select_parameters(
            cases,
            embeddings,
            train_indices,
            condition=condition,
            include_phase1=include_phase1,
            c_grid=c_grid,
            thresholds=thresholds,
            inner_splits=inner_splits,
            pca_components=pca_components,
        )
        fold = build_fold_features(
            cases,
            embeddings,
            train_indices,
            held_indices,
            include_phase1=include_phase1,
            pca_components=pca_components,
        )
        scores = _fit_predict(
            fold, fold.evaluate, _tier_weights(cases, train_indices), regularization_c
        )
        raw = _raw_candidates(cases, held_indices)
        selected = apply_condition(scores, raw, threshold=threshold, condition=condition)
        for local_index, case_index in enumerate(held_indices):
            case = cases[case_index]
            for label_index, label in enumerate(case.labels):
                results.append(
                    CalibrationPrediction(
                        vc_slug=case.vc_slug,
                        episode_slug=case.episode_slug,
                        label=label,
                        condition=condition,
                        score=float(scores[local_index, label_index]),
                        predicted=bool(selected[local_index, label_index]),
                        raw_predicted=bool(raw[local_index, label_index]),
                        reference=bool(case.reference_targets[label]),
                    )
                )
        provenance.append(
            CalibrationFoldProvenance(
                condition=condition,
                held_episode_slug=held_group,
                training_episode_slugs=training_groups,
                regularization_c=regularization_c,
                threshold=threshold,
                inner_objective_f1=inner_f1,
            )
        )
    return results, provenance


def run_nested_calibration(
    cases: Sequence[CalibrationCase],
    pitch_embeddings: np.ndarray,
    *,
    jobs: int = 1,
    c_grid: Sequence[float] = (0.01, 0.1, 1.0),
    thresholds: Sequence[float] = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
    inner_splits: int = 3,
    pca_components: int = 32,
) -> CalibrationRunResult:
    if not cases:
        raise ValueError("calibration requires at least one case")
    embeddings = np.asarray(pitch_embeddings, dtype=float)
    if embeddings.shape[0] != len(cases):
        raise ValueError("pitch embedding row mismatch")
    groups = sorted({case.episode_slug for case in cases})
    folds = Parallel(n_jobs=jobs, prefer="processes")(
        delayed(_outer_fold)(
            group,
            cases,
            embeddings,
            tuple(float(value) for value in c_grid),
            tuple(float(value) for value in thresholds),
            inner_splits,
            pca_components,
        )
        for group in groups
    )
    predictions = sorted(
        (row for fold, _ in folds for row in fold),
        key=lambda row: (row.condition, row.vc_slug, row.episode_slug, row.label),
    )
    provenance = sorted(
        (row for _, fold in folds for row in fold),
        key=lambda row: (row.condition, row.held_episode_slug),
    )
    return CalibrationRunResult(tuple(predictions), tuple(provenance))
