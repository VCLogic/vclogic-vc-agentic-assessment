"""Nested compact per-VC and hierarchical partial-pooling evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    fbeta_score,
    precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .compact_rationale_features import compact_decision_view, compact_rationale_view
from .phase2_calibration_evaluation import Phase2CalibrationRecord


@dataclass(frozen=True)
class CompactPrediction:
    vc_slug: str
    vc_name: str
    episode_slug: str
    held_group: str
    target: int
    method: str
    score: float
    balanced_decision: int
    precision_decision: int
    balanced_threshold: float
    precision_threshold: float
    selected_c: float
    deviation_scale: float
    training_episode_slugs: tuple[str, ...]
    training_groups: tuple[str, ...]
    training_vcs: tuple[str, ...]
    training_indices: tuple[int, ...]


def select_threshold(
    targets: np.ndarray, scores: np.ndarray, objective: str
) -> float:
    """Choose a deterministic classification threshold on training scores."""
    candidates = sorted({0.0, 0.5, 1.0, *(float(value) for value in scores)})
    if objective == "balanced_accuracy":
        return max(
            candidates,
            key=lambda threshold: (
                balanced_accuracy_score(targets, scores >= threshold),
                threshold,
            ),
        )
    if objective == "f0_5":
        return max(
            candidates,
            key=lambda threshold: (
                fbeta_score(
                    targets, scores >= threshold, beta=0.5, zero_division=0
                ),
                precision_score(targets, scores >= threshold, zero_division=0),
                threshold,
            ),
        )
    raise ValueError(f"unknown threshold objective: {objective}")


def _compact_view(
    record: Phase2CalibrationRecord,
    taxonomy: Mapping[str, object] | Sequence[str],
    family: str,
) -> dict[str, float]:
    if family == "decision":
        return compact_decision_view(record, taxonomy)
    if family == "rationale":
        return compact_rationale_view(record, taxonomy)
    raise ValueError(f"unknown compact feature family: {family}")


def _estimator(c_value: float, seed: int):
    return make_pipeline(
        DictVectorizer(sparse=True),
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=c_value,
            solver="liblinear",
            l1_ratio=0.0,
            class_weight="balanced",
            max_iter=2_000,
            random_state=seed,
        ),
    )


def _inner_folds(
    targets: np.ndarray, inner_splits: int, seed: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    splits = min(inner_splits, int(targets.sum()), int(len(targets) - targets.sum()))
    if splits < 2:
        raise ValueError("nested compact evaluation needs two cases per class")
    return tuple(
        StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed).split(
            np.zeros(len(targets)), targets
        )
    )


def _select_compact_model(
    maps: Sequence[dict[str, float]],
    targets: np.ndarray,
    *,
    inner_splits: int,
    seed: int,
) -> tuple[float, np.ndarray, float, float]:
    folds = _inner_folds(targets, inner_splits, seed)
    candidates: list[tuple[float, float, float, np.ndarray]] = []
    for c_value in (0.1, 1.0, 10.0):
        scores = np.zeros(len(targets), dtype=float)
        for inner_index, (fit_indices, validation_indices) in enumerate(folds):
            estimator = _estimator(c_value, seed + inner_index)
            estimator.fit(
                [maps[int(index)] for index in fit_indices], targets[fit_indices]
            )
            scores[validation_indices] = estimator.predict_proba(
                [maps[int(index)] for index in validation_indices]
            )[:, 1]
        candidates.append(
            (average_precision_score(targets, scores), -c_value, c_value, scores)
        )
    _, _, selected_c, selected_scores = max(
        candidates, key=lambda row: (row[0], row[1])
    )
    return (
        selected_c,
        selected_scores,
        select_threshold(targets, selected_scores, "balanced_accuracy"),
        select_threshold(targets, selected_scores, "f0_5"),
    )


def nested_per_vc_compact_predictions(
    records: Sequence[Phase2CalibrationRecord],
    taxonomy: Mapping[str, object] | Sequence[str],
    *,
    family: str,
    inner_splits: int = 3,
    seed: int = 20260816,
    n_jobs: int = 1,
) -> list[CompactPrediction]:
    """Hold each pitch out and train only on that investor's other pitches."""
    predictions: list[CompactPrediction | None] = [None] * len(records)
    tasks: list[tuple[int, tuple[int, ...], int]] = []
    fold_index = 0
    for vc_slug in sorted({record.vc_slug for record in records}):
        vc_indices = [
            index for index, record in enumerate(records) if record.vc_slug == vc_slug
        ]
        if {records[index].target for index in vc_indices} != {0, 1}:
            raise ValueError(f"per-VC compact model needs both classes: {vc_slug}")
        for held_index in vc_indices:
            tasks.append((held_index, tuple(vc_indices), fold_index))
            fold_index += 1
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_per_vc_fold_prediction)(
            held_index,
            vc_indices,
            records,
            taxonomy,
            family,
            inner_splits,
            seed + task_fold_index,
        )
        for held_index, vc_indices, task_fold_index in tasks
    )
    for held_index, prediction in results:
        predictions[held_index] = prediction
    if any(prediction is None for prediction in predictions):
        raise RuntimeError("per-VC compact model did not predict every case")
    return [prediction for prediction in predictions if prediction is not None]


def _per_vc_fold_prediction(
    held_index: int,
    vc_indices: tuple[int, ...],
    records: Sequence[Phase2CalibrationRecord],
    taxonomy: Mapping[str, object] | Sequence[str],
    family: str,
    inner_splits: int,
    fold_seed: int,
) -> tuple[int, CompactPrediction]:
    training_indices = tuple(index for index in vc_indices if index != held_index)
    training_records = [records[index] for index in training_indices]
    targets = np.asarray([record.target for record in training_records], dtype=int)
    maps = [_compact_view(record, taxonomy, family) for record in training_records]
    c_value, _, balanced_threshold, precision_threshold = _select_compact_model(
        maps, targets, inner_splits=inner_splits, seed=fold_seed
    )
    estimator = _estimator(c_value, fold_seed)
    estimator.fit(maps, targets)
    held = records[held_index]
    score = float(
        estimator.predict_proba([_compact_view(held, taxonomy, family)])[0, 1]
    )
    return held_index, CompactPrediction(
        vc_slug=held.vc_slug,
        vc_name=held.vc_name,
        episode_slug=held.episode_slug,
        held_group=held.group,
        target=held.target,
        method=f"per_vc_compact_{family}",
        score=score,
        balanced_decision=int(score >= balanced_threshold),
        precision_decision=int(score >= precision_threshold),
        balanced_threshold=balanced_threshold,
        precision_threshold=precision_threshold,
        selected_c=c_value,
        deviation_scale=0.0,
        training_episode_slugs=tuple(
            record.episode_slug for record in training_records
        ),
        training_groups=tuple(record.group for record in training_records),
        training_vcs=tuple(record.vc_slug for record in training_records),
        training_indices=training_indices,
    )


def partial_pooling_view(
    record: Phase2CalibrationRecord,
    taxonomy: Mapping[str, object] | Sequence[str],
    *,
    family: str,
    deviation_scale: float,
) -> dict[str, float]:
    """Emit global compact effects and shrunken investor deviations."""
    if deviation_scale not in {0.0, 0.25, 0.5, 1.0}:
        raise ValueError(f"unsupported deviation scale: {deviation_scale}")
    compact = _compact_view(record, taxonomy, family)
    return _partial_pooling_from_compact(record, compact, deviation_scale)


def _partial_pooling_from_compact(
    record: Phase2CalibrationRecord,
    compact: Mapping[str, float],
    deviation_scale: float,
) -> dict[str, float]:
    result = {f"global__{name}": value for name, value in compact.items()}
    result.update({
        f"deviation__{record.vc_slug}__{name}": value * deviation_scale
        for name, value in compact.items()
    })
    result[f"deviation__{record.vc_slug}__intercept"] = deviation_scale
    return result


def _partial_estimator(c_value: float, seed: int):
    return make_pipeline(
        DictVectorizer(sparse=False),
        LogisticRegression(
            C=c_value,
            solver="liblinear",
            l1_ratio=0.0,
            class_weight="balanced",
            max_iter=2_000,
            random_state=seed,
        ),
    )


def _grouped_inner_folds(
    targets: np.ndarray,
    groups: np.ndarray,
    inner_splits: int,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    maximum = min(inner_splits, len(set(str(group) for group in groups)))
    for split_count in range(maximum, 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=split_count, shuffle=True, random_state=seed
        )
        folds = tuple(splitter.split(np.zeros(len(targets)), targets, groups))
        if all(len(set(targets[fit])) == 2 for fit, _ in folds):
            return folds
    raise ValueError("partial pooling needs grouped inner folds with both classes")


def _select_partial_model(
    records: Sequence[Phase2CalibrationRecord],
    taxonomy: Mapping[str, object] | Sequence[str],
    *,
    family: str,
    objective: str,
    inner_splits: int,
    seed: int,
) -> tuple[float, float, np.ndarray, float, float]:
    targets = np.asarray([record.target for record in records], dtype=int)
    groups = np.asarray([record.group for record in records], dtype=object)
    folds = _grouped_inner_folds(targets, groups, inner_splits, seed)
    candidates: list[
        tuple[float, float, float, float, float, np.ndarray]
    ] = []
    compact_maps = [_compact_view(record, taxonomy, family) for record in records]
    for c_value in (0.1, 1.0, 10.0):
        for deviation_scale in (0.0, 0.25, 0.5, 1.0):
            maps = [
                _partial_pooling_from_compact(
                    record,
                    compact,
                    deviation_scale,
                )
                for record, compact in zip(records, compact_maps, strict=True)
            ]
            scores = np.zeros(len(records), dtype=float)
            for inner_index, (fit_indices, validation_indices) in enumerate(folds):
                estimator = _partial_estimator(c_value, seed + inner_index)
                estimator.fit(
                    [maps[int(index)] for index in fit_indices],
                    targets[fit_indices],
                )
                scores[validation_indices] = estimator.predict_proba(
                    [maps[int(index)] for index in validation_indices]
                )[:, 1]
            balanced_threshold = select_threshold(
                targets, scores, "balanced_accuracy"
            )
            precision_threshold = select_threshold(targets, scores, "f0_5")
            if objective == "average_precision":
                metric = average_precision_score(targets, scores)
            elif objective == "balanced_accuracy":
                metric = balanced_accuracy_score(
                    targets, scores >= balanced_threshold
                )
            else:
                raise ValueError(f"unknown partial-pooling objective: {objective}")
            candidates.append((
                float(metric),
                -c_value,
                -deviation_scale,
                c_value,
                deviation_scale,
                scores,
            ))
    _, _, _, selected_c, selected_scale, selected_scores = max(
        candidates, key=lambda row: (row[0], row[1], row[2])
    )
    return (
        selected_c,
        selected_scale,
        selected_scores,
        select_threshold(targets, selected_scores, "balanced_accuracy"),
        select_threshold(targets, selected_scores, "f0_5"),
    )


def nested_partial_pooling_predictions(
    records: Sequence[Phase2CalibrationRecord],
    taxonomy: Mapping[str, object] | Sequence[str],
    *,
    family: str,
    objective: str,
    inner_splits: int = 3,
    seed: int = 20260816,
    n_jobs: int = 1,
) -> list[CompactPrediction]:
    """Hold a shared pitch episode out across all investors and partially pool."""
    if {record.target for record in records} != {0, 1}:
        raise ValueError("partial pooling needs both outcome classes")
    predictions: list[CompactPrediction | None] = [None] * len(records)
    groups = sorted({record.group for record in records})
    suffix = "ranking" if objective == "average_precision" else "classification"
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_partial_pooling_fold_predictions)(
            records,
            taxonomy,
            family,
            objective,
            inner_splits,
            seed + fold_index,
            held_group,
            suffix,
        )
        for fold_index, held_group in enumerate(groups)
    )
    for fold_predictions in results:
        for held_index, prediction in fold_predictions:
            predictions[held_index] = prediction
    if any(prediction is None for prediction in predictions):
        raise RuntimeError("partial pooling did not predict every case")
    return [prediction for prediction in predictions if prediction is not None]


def _partial_pooling_fold_predictions(
    records: Sequence[Phase2CalibrationRecord],
    taxonomy: Mapping[str, object] | Sequence[str],
    family: str,
    objective: str,
    inner_splits: int,
    fold_seed: int,
    held_group: str,
    suffix: str,
) -> list[tuple[int, CompactPrediction]]:
        held_indices = tuple(
            index for index, record in enumerate(records)
            if record.group == held_group
        )
        training_indices = tuple(
            index for index, record in enumerate(records)
            if record.group != held_group
        )
        training_records = [records[index] for index in training_indices]
        (
            selected_c,
            selected_scale,
            _,
            balanced_threshold,
            precision_threshold,
        ) = _select_partial_model(
            training_records,
            taxonomy,
            family=family,
            objective=objective,
            inner_splits=inner_splits,
            seed=fold_seed,
        )
        estimator = _partial_estimator(selected_c, fold_seed)
        training_maps = [
            partial_pooling_view(
                record,
                taxonomy,
                family=family,
                deviation_scale=selected_scale,
            )
            for record in training_records
        ]
        estimator.fit(
            training_maps,
            np.asarray([record.target for record in training_records], dtype=int),
        )
        held_records = [records[index] for index in held_indices]
        held_scores = estimator.predict_proba([
            partial_pooling_view(
                record,
                taxonomy,
                family=family,
                deviation_scale=selected_scale,
            )
            for record in held_records
        ])[:, 1]
        training_groups = tuple(sorted({record.group for record in training_records}))
        training_vcs = tuple(sorted({record.vc_slug for record in training_records}))
        training_episodes = tuple(record.episode_slug for record in training_records)
        output: list[tuple[int, CompactPrediction]] = []
        for held_index, held, score_value in zip(
            held_indices, held_records, held_scores, strict=True
        ):
            score = float(score_value)
            output.append((held_index, CompactPrediction(
                vc_slug=held.vc_slug,
                vc_name=held.vc_name,
                episode_slug=held.episode_slug,
                held_group=held.group,
                target=held.target,
                method=f"partial_pooling_{family}_{suffix}",
                score=score,
                balanced_decision=int(score >= balanced_threshold),
                precision_decision=int(score >= precision_threshold),
                balanced_threshold=balanced_threshold,
                precision_threshold=precision_threshold,
                selected_c=selected_c,
                deviation_scale=selected_scale,
                training_episode_slugs=training_episodes,
                training_groups=training_groups,
                training_vcs=training_vcs,
                training_indices=training_indices,
            )))
        return output


def fit_partial_pooling_explanatory_model(
    records: Sequence[Phase2CalibrationRecord],
    taxonomy: Mapping[str, object] | Sequence[str],
    *,
    family: str,
    inner_splits: int = 3,
    seed: int = 20260816,
) -> dict[str, object]:
    """Fit a full-data model for explanation/deployment, never performance."""
    (
        selected_c,
        selected_scale,
        _,
        balanced_threshold,
        precision_threshold,
    ) = _select_partial_model(
        records,
        taxonomy,
        family=family,
        objective="average_precision",
        inner_splits=inner_splits,
        seed=seed,
    )
    maps = [
        partial_pooling_view(
            record,
            taxonomy,
            family=family,
            deviation_scale=selected_scale,
        )
        for record in records
    ]
    estimator = _partial_estimator(selected_c, seed)
    estimator.fit(maps, np.asarray([record.target for record in records], dtype=int))
    vectorizer = estimator[0]
    classifier = estimator[-1]
    rows: list[dict[str, object]] = []
    for name, coefficient in zip(
        vectorizer.get_feature_names_out(), classifier.coef_[0], strict=True
    ):
        feature = str(name)
        if feature.startswith("global__"):
            scope, vc_slug, compact_feature = "global", "", feature.removeprefix("global__")
        else:
            parts = feature.split("__", 2)
            scope = "vc_deviation"
            vc_slug = parts[1] if len(parts) > 1 else ""
            compact_feature = parts[2] if len(parts) > 2 else feature
        value = float(coefficient)
        rows.append({
            "effect_scope": scope,
            "vc_slug": vc_slug,
            "feature": compact_feature,
            "coefficient": value,
            "odds_ratio": float(np.exp(np.clip(value, -20.0, 20.0))),
        })
    return {
        "schema": "partial-pooling-explanatory-model-v1",
        "scientific_status": "full_data_explanatory_not_performance_estimate",
        "family": family,
        "selected_c": selected_c,
        "deviation_scale": selected_scale,
        "balanced_threshold": balanced_threshold,
        "precision_threshold": precision_threshold,
        "coefficient_rows": rows,
        "estimator": estimator,
    }
