"""Leakage-controlled advanced models over frozen calibration features."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from .artifacts import write_json
from .calibration import CalibrationRecord, evaluate_predictions, feature_names


@dataclass(frozen=True)
class CandidateSpec:
    """A finite, predeclared advanced-model search space."""

    name: str
    feature_family: str
    objective: Literal["classification", "ranking", "hybrid"]
    parameter_grid: tuple[Mapping[str, object], ...]


LOGISTIC_BASELINES: Mapping[str, str] = {
    "legacy_phase1": "legacy_phase1",
    "semantic_phase1": "semantic_phase1",
    "phase2": "phase2",
    "legacy_plus_phase2": "legacy_plus_phase2",
    "semantic_plus_phase2": "semantic_plus_phase2",
}
RAW_REFERENCES = (
    "majority_out",
    "raw_any_check",
    "raw_standard_check",
    "raw_ranking_score",
)


def candidate_manifest() -> dict[str, CandidateSpec]:
    """Return deterministic candidate definitions for the expanded benchmark."""
    specifications = (
        CandidateSpec(
            "late_fusion",
            "semantic_phase1+phase2_scores",
            "hybrid",
            tuple(
                {"base_c": base_c, "meta_c": meta_c}
                for base_c in (0.03, 0.1, 0.3)
                for meta_c in (0.03, 0.1, 0.3)
            ),
        ),
        CandidateSpec(
            "elastic_net_combined",
            "semantic_plus_phase2",
            "classification",
            (
                {"c": 0.03, "l1_ratio": 0.25},
                {"c": 0.1, "l1_ratio": 0.25},
                {"c": 0.1, "l1_ratio": 0.75},
                {"c": 0.3, "l1_ratio": 0.5},
            ),
        ),
        CandidateSpec(
            "shallow_gradient_boosting",
            "semantic_plus_phase2",
            "classification",
            (
                {"learning_rate": 0.03, "max_depth": 1, "max_leaf_nodes": 3},
                {"learning_rate": 0.05, "max_depth": 2, "max_leaf_nodes": 5},
                {"learning_rate": 0.1, "max_depth": 1, "max_leaf_nodes": 3},
                {"learning_rate": 0.1, "max_depth": 2, "max_leaf_nodes": 5},
            ),
        ),
        CandidateSpec(
            "linear_svm",
            "semantic_plus_phase2",
            "classification",
            ({"c": 0.03}, {"c": 0.1}, {"c": 0.3}),
        ),
        CandidateSpec(
            "rbf_svm",
            "semantic_plus_phase2",
            "classification",
            tuple(
                {"c": c_value, "gamma": gamma}
                for c_value in (0.1, 1.0, 10.0)
                for gamma in ("scale", 0.01)
            ),
        ),
        CandidateSpec(
            "pairwise_ranker",
            "legacy_plus_phase2",
            "ranking",
            ({"c": 0.03}, {"c": 0.1}, {"c": 0.3}),
        ),
        CandidateSpec(
            "two_stage_diamond",
            "semantic_phase1+phase2_scores",
            "hybrid",
            tuple(
                {"base_c": base_c, "phase1_weight": phase1_weight}
                for base_c in (0.03, 0.1, 0.3)
                for phase1_weight in (0.0, 0.25, 0.5, 0.75, 1.0)
            ),
        ),
    )
    return {specification.name: specification for specification in specifications}


@dataclass(frozen=True)
class AdvancedPrediction:
    """One outer-fold prediction with its training-only model choices."""

    method: str
    episode_slug: str
    target: int
    score: float
    predicted: int
    threshold: float
    selected_config: Mapping[str, object]
    training_slugs: tuple[str, ...]
    provenance: Mapping[str, object]


def _matrix(
    records: Sequence[CalibrationRecord],
    feature_family: str,
    names: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [
            [float(record.features[feature_family].get(name, 0.0)) for name in names]
            for record in records
        ],
        dtype=float,
    )


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidates = sorted({float("-inf"), 0.0, *(float(value) for value in scores)})
    best_score = -1.0
    best_threshold = 0.0
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        score = float(balanced_accuracy_score(y_true, predicted))
        if score > best_score or (
            np.isclose(score, best_score) and threshold > best_threshold
        ):
            best_score = score
            best_threshold = threshold
    return best_threshold


def _splitter(y: np.ndarray) -> StratifiedKFold:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    splits = min(3, positives, negatives)
    if splits < 2:
        raise ValueError("nested evaluation requires at least two cases per class")
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=20260803)


def _estimator(method: str, configuration: Mapping[str, object]):
    if method == "elastic_net_combined":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(configuration["c"]),
                class_weight="balanced",
                l1_ratio=float(configuration["l1_ratio"]),
                max_iter=20_000,
                random_state=20260803,
                solver="saga",
            ),
        )
    if method == "shallow_gradient_boosting":
        return HistGradientBoostingClassifier(
            class_weight="balanced",
            early_stopping=False,
            l2_regularization=1.0,
            learning_rate=float(configuration["learning_rate"]),
            max_depth=int(configuration["max_depth"]),
            max_iter=100,
            max_leaf_nodes=int(configuration["max_leaf_nodes"]),
            min_samples_leaf=5,
            random_state=20260803,
        )
    if method == "linear_svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=float(configuration["c"]),
                class_weight="balanced",
                dual="auto",
                max_iter=20_000,
                random_state=20260803,
            ),
        )
    if method == "rbf_svm":
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=float(configuration["c"]),
                class_weight="balanced",
                gamma=configuration["gamma"],
                kernel="rbf",
                random_state=20260803,
            ),
        )
    raise ValueError(f"no generic estimator for method: {method}")


def _score(estimator: object, x_values: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(x_values)[:, 1], dtype=float)
    return np.asarray(estimator.decision_function(x_values), dtype=float)


def _pairwise_training(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positives = x_values[y_values == 1]
    negatives = x_values[y_values == 0]
    if not len(positives) or not len(negatives):
        raise ValueError("pairwise ranking requires both classes")
    differences = []
    targets = []
    for positive in positives:
        for negative in negatives:
            delta = positive - negative
            differences.extend((delta, -delta))
            targets.extend((1, 0))
    return np.asarray(differences, dtype=float), np.asarray(targets, dtype=int)


def _fit_pairwise(
    x_values: np.ndarray,
    y_values: np.ndarray,
    configuration: Mapping[str, object],
):
    pair_x, pair_y = _pairwise_training(x_values, y_values)
    estimator = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=float(configuration["c"]),
            class_weight="balanced",
            dual="auto",
            fit_intercept=False,
            max_iter=20_000,
            random_state=20260803,
        ),
    )
    estimator.fit(pair_x, pair_y)
    return estimator


def _inner_scores(
    method: str,
    configuration: Mapping[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    scores = np.zeros(len(y_train), dtype=float)
    for fit_indices, validation_indices in _splitter(y_train).split(x_train, y_train):
        if method == "pairwise_ranker":
            estimator = _fit_pairwise(
                x_train[fit_indices], y_train[fit_indices], configuration
            )
        else:
            estimator = _estimator(method, configuration)
            estimator.fit(x_train[fit_indices], y_train[fit_indices])
        scores[validation_indices] = _score(estimator, x_train[validation_indices])
    return scores


def _select_configuration(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[Mapping[str, object], np.ndarray]:
    specification = candidate_manifest()[method]
    selected = specification.parameter_grid[0]
    selected_scores = np.zeros(len(y_train), dtype=float)
    best_average_precision = -1.0
    for configuration in specification.parameter_grid:
        scores = _inner_scores(method, configuration, x_train, y_train)
        average_precision = float(average_precision_score(y_train, scores))
        if average_precision > best_average_precision:
            best_average_precision = average_precision
            selected = configuration
            selected_scores = scores
    return selected, selected_scores


def nested_advanced_leave_one_out(
    records: Sequence[CalibrationRecord], method: str
) -> list[AdvancedPrediction]:
    """Evaluate a generic or pairwise advanced candidate using nested LOOCV."""
    manifest = candidate_manifest()
    if method not in manifest:
        raise ValueError(f"unknown advanced method: {method}")
    if method in {"late_fusion", "two_stage_diamond"}:
        return _nested_score_composition(records, method)
    if len(records) < 5:
        raise ValueError("advanced leave-one-out evaluation needs at least five records")

    family = manifest[method].feature_family
    names = feature_names(records, family)
    x_all = _matrix(records, family, names)
    y_all = np.asarray([record.target for record in records], dtype=int)
    predictions: list[AdvancedPrediction] = []
    for held_index, held_record in enumerate(records):
        train_indices = np.asarray(
            [index for index in range(len(records)) if index != held_index], dtype=int
        )
        x_train = x_all[train_indices]
        y_train = y_all[train_indices]
        configuration, inner_scores = _select_configuration(
            method, x_train, y_train
        )
        threshold = _best_threshold(y_train, inner_scores)
        if method == "pairwise_ranker":
            estimator = _fit_pairwise(x_train, y_train, configuration)
        else:
            estimator = _estimator(method, configuration)
            estimator.fit(x_train, y_train)
        held_score = float(_score(estimator, x_all[[held_index]])[0])
        predictions.append(
            AdvancedPrediction(
                method=method,
                episode_slug=held_record.episode_slug,
                target=int(held_record.target),
                score=held_score,
                predicted=int(held_score >= threshold),
                threshold=float(threshold),
                selected_config=dict(configuration),
                training_slugs=tuple(records[index].episode_slug for index in train_indices),
                provenance={"inner_cross_fitted": True},
            )
        )
    return predictions


FoldMaps = tuple[tuple[tuple[tuple[int, ...], tuple[int, ...]], ...], ...]


def make_repeated_stratified_folds(
    records: Sequence[CalibrationRecord],
    *,
    repeats: int = 5,
    splits: int = 5,
) -> FoldMaps:
    """Create fixed outer folds; callers may reuse them across label definitions."""
    if repeats < 1 or splits < 2:
        raise ValueError("repeated stratification needs positive repeats and >=2 splits")
    targets = np.asarray([record.target for record in records], dtype=int)
    positives = int(targets.sum())
    negatives = int(len(targets) - positives)
    actual_splits = min(splits, positives, negatives)
    if actual_splits < 2:
        raise ValueError("repeated stratification requires both classes")
    dummy = np.zeros((len(records), 1), dtype=float)
    repetitions = []
    for repeat in range(repeats):
        splitter = StratifiedKFold(
            n_splits=actual_splits,
            shuffle=True,
            random_state=20260803 + repeat,
        )
        repetitions.append(
            tuple(
                (tuple(int(index) for index in train), tuple(int(index) for index in held))
                for train, held in splitter.split(dummy, targets)
            )
        )
    return tuple(repetitions)


def _platt_calibrator(raw_scores: np.ndarray, targets: np.ndarray):
    calibrator = LogisticRegression(C=1.0, max_iter=20_000, random_state=20260803)
    calibrator.fit(raw_scores.reshape(-1, 1), targets)
    return calibrator


def _calibrated_scores(calibrator: object, raw_scores: np.ndarray) -> np.ndarray:
    return np.asarray(
        calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1], dtype=float
    )


def _generic_fold_prediction(
    records: Sequence[CalibrationRecord],
    method: str,
    train_indices: np.ndarray,
    held_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object], float]:
    if method in {"late_fusion", "two_stage_diamond"}:
        return _composition_fold_prediction(
            records, method, train_indices, held_indices
        )
    family = (
        LOGISTIC_BASELINES[method]
        if method in LOGISTIC_BASELINES
        else candidate_manifest()[method].feature_family
    )
    names = feature_names(records, family)
    x_all = _matrix(records, family, names)
    y_all = np.asarray([record.target for record in records], dtype=int)
    x_train = x_all[train_indices]
    y_train = y_all[train_indices]
    if method in LOGISTIC_BASELINES:
        configuration, inner_raw_scores = _select_logistic_configuration(
            x_train, y_train
        )
    else:
        configuration, inner_raw_scores = _select_configuration(
            method, x_train, y_train
        )
    calibrator = _platt_calibrator(inner_raw_scores, y_train)
    inner_scores = _calibrated_scores(calibrator, inner_raw_scores)
    threshold = _best_threshold(y_train, inner_scores)
    if method in LOGISTIC_BASELINES:
        estimator = _logistic(float(configuration["c"]))
        estimator.fit(x_train, y_train)
    elif method == "pairwise_ranker":
        estimator = _fit_pairwise(x_train, y_train, configuration)
    else:
        estimator = _estimator(method, configuration)
        estimator.fit(x_train, y_train)
    held_raw_scores = _score(estimator, x_all[held_indices])
    held_scores = _calibrated_scores(calibrator, held_raw_scores)
    decisions = (held_scores >= threshold).astype(int)
    return held_scores, decisions, configuration, float(threshold)


def _select_logistic_configuration(
    x_train: np.ndarray, y_train: np.ndarray
) -> tuple[Mapping[str, object], np.ndarray]:
    selected: Mapping[str, object] = {"c": 0.01}
    selected_scores = np.zeros(len(y_train), dtype=float)
    best_ap = -1.0
    for c_value in (0.01, 0.1, 1.0, 10.0):
        scores = _crossfit_logistic_scores(x_train, y_train, c_value)
        average_precision = float(average_precision_score(y_train, scores))
        if average_precision > best_ap:
            best_ap = average_precision
            selected = {"c": c_value}
            selected_scores = scores
    return selected, selected_scores


def _composition_fold_prediction(
    records: Sequence[CalibrationRecord],
    method: str,
    train_indices: np.ndarray,
    held_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object], float]:
    p1_names = feature_names(records, "semantic_phase1")
    p2_names = feature_names(records, "phase2")
    p1_all = _matrix(records, "semantic_phase1", p1_names)
    p2_all = _matrix(records, "phase2", p2_names)
    raw_any_all = np.asarray(
        [record.features["phase2"]["any_check_likelihood"] for record in records],
        dtype=float,
    )
    targets = np.asarray([record.target for record in records], dtype=int)
    y_train = targets[train_indices]
    p1_train = p1_all[train_indices]
    p2_train = p2_all[train_indices]
    raw_any_train = raw_any_all[train_indices]
    score_cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def base_scores(base_c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if base_c not in score_cache:
            p1_oof = _crossfit_logistic_scores(p1_train, y_train, base_c)
            p2_oof = _crossfit_logistic_scores(p2_train, y_train, base_c)
            p1_model = _logistic(base_c)
            p2_model = _logistic(base_c)
            p1_model.fit(p1_train, y_train)
            p2_model.fit(p2_train, y_train)
            score_cache[base_c] = (
                p1_oof,
                p2_oof,
                _score(p1_model, p1_all[held_indices]),
                _score(p2_model, p2_all[held_indices]),
            )
        return score_cache[base_c]

    best_ap = -1.0
    selected_config: Mapping[str, object] = candidate_manifest()[method].parameter_grid[0]
    selected_inner_raw = np.zeros(len(y_train), dtype=float)
    selected_held_raw = np.zeros(len(held_indices), dtype=float)
    for configuration in candidate_manifest()[method].parameter_grid:
        p1_oof, p2_oof, p1_held, p2_held = base_scores(
            float(configuration["base_c"])
        )
        if method == "late_fusion":
            meta_train = _fusion_matrix(p1_oof, p2_oof, raw_any_train)
            meta_held = _fusion_matrix(
                p1_held, p2_held, raw_any_all[held_indices]
            )
            meta_c = float(configuration["meta_c"])
            inner_raw = _crossfit_logistic_scores(meta_train, y_train, meta_c)
            meta_model = _logistic(meta_c)
            meta_model.fit(meta_train, y_train)
            held_raw = _score(meta_model, meta_held)
        else:
            weight = float(configuration["phase1_weight"])
            inner_raw = weight * p1_oof + (1.0 - weight) * p2_oof
            held_raw = weight * p1_held + (1.0 - weight) * p2_held
        average_precision = float(average_precision_score(y_train, inner_raw))
        if average_precision > best_ap:
            best_ap = average_precision
            selected_config = configuration
            selected_inner_raw = inner_raw
            selected_held_raw = held_raw

    calibrator = _platt_calibrator(selected_inner_raw, y_train)
    inner_scores = _calibrated_scores(calibrator, selected_inner_raw)
    held_scores = _calibrated_scores(calibrator, selected_held_raw)
    threshold = _best_threshold(y_train, inner_scores)
    decisions = (held_scores >= threshold).astype(int)
    return held_scores, decisions, selected_config, float(threshold)


def repeated_stratified_predictions(
    records: Sequence[CalibrationRecord],
    method: str,
    *,
    fold_maps: FoldMaps | None = None,
) -> list[AdvancedPrediction]:
    """Average calibrated out-of-sample predictions across fixed stratified folds."""
    if method not in candidate_manifest() and method not in LOGISTIC_BASELINES:
        raise ValueError(f"unknown advanced method: {method}")
    folds = fold_maps or make_repeated_stratified_folds(records)
    score_values: list[list[float]] = [[] for _ in records]
    decision_values: list[list[int]] = [[] for _ in records]
    configurations: list[list[Mapping[str, object]]] = [[] for _ in records]
    training_folds: list[list[tuple[str, ...]]] = [[] for _ in records]
    for repetition in folds:
        covered: set[int] = set()
        for train_tuple, held_tuple in repetition:
            train_indices = np.asarray(train_tuple, dtype=int)
            held_indices = np.asarray(held_tuple, dtype=int)
            overlap = set(train_tuple) & set(held_tuple)
            if overlap:
                raise ValueError("outer train and held indices overlap")
            if covered & set(held_tuple):
                raise ValueError("outer repetition holds an episode more than once")
            covered.update(held_tuple)
            held_scores, held_decisions, configuration, _ = _generic_fold_prediction(
                records, method, train_indices, held_indices
            )
            train_slugs = tuple(records[index].episode_slug for index in train_indices)
            for offset, held_index in enumerate(held_indices):
                index = int(held_index)
                score_values[index].append(float(held_scores[offset]))
                decision_values[index].append(int(held_decisions[offset]))
                configurations[index].append(dict(configuration))
                training_folds[index].append(train_slugs)
        if covered != set(range(len(records))):
            raise ValueError("outer repetition does not cover every episode")

    predictions = []
    repeat_count = len(folds)
    for index, record in enumerate(records):
        if len(score_values[index]) != repeat_count:
            raise ValueError(f"episode lacks repeated predictions: {record.episode_slug}")
        mean_score = float(np.mean(score_values[index]))
        predicted = int(float(np.mean(decision_values[index])) >= 0.5)
        predictions.append(
            AdvancedPrediction(
                method=method,
                episode_slug=record.episode_slug,
                target=int(record.target),
                score=mean_score,
                predicted=predicted,
                threshold=0.5,
                selected_config={"fold_configurations": configurations[index]},
                training_slugs=training_folds[index][0],
                provenance={
                    "outer_training_folds": training_folds[index],
                    "outer_repeats": repeat_count,
                    "score_calibration": "training_only_platt",
                },
            )
        )
    return predictions


def repeated_stratified_logistic_predictions(
    records: Sequence[CalibrationRecord],
    *,
    method: str,
    feature_family: str,
    fold_maps: FoldMaps | None = None,
) -> list[AdvancedPrediction]:
    """Evaluate an externally constructed feature family with the baseline protocol."""
    if not method.strip():
        raise ValueError("dynamic logistic method needs a non-empty name")
    names = feature_names(records, feature_family)
    x_all = _matrix(records, feature_family, names)
    y_all = np.asarray([record.target for record in records], dtype=int)
    folds = fold_maps or make_repeated_stratified_folds(records)
    score_values: list[list[float]] = [[] for _ in records]
    decision_values: list[list[int]] = [[] for _ in records]
    configurations: list[list[Mapping[str, object]]] = [[] for _ in records]
    training_folds: list[list[tuple[str, ...]]] = [[] for _ in records]

    for repetition in folds:
        covered: set[int] = set()
        for train_tuple, held_tuple in repetition:
            train_indices = np.asarray(train_tuple, dtype=int)
            held_indices = np.asarray(held_tuple, dtype=int)
            if set(train_tuple) & set(held_tuple):
                raise ValueError("outer train and held indices overlap")
            if covered & set(held_tuple):
                raise ValueError("outer repetition holds an episode more than once")
            covered.update(held_tuple)
            x_train = x_all[train_indices]
            y_train = y_all[train_indices]
            configuration, inner_raw_scores = _select_logistic_configuration(
                x_train, y_train
            )
            calibrator = _platt_calibrator(inner_raw_scores, y_train)
            inner_scores = _calibrated_scores(calibrator, inner_raw_scores)
            threshold = _best_threshold(y_train, inner_scores)
            estimator = _logistic(float(configuration["c"]))
            estimator.fit(x_train, y_train)
            held_raw_scores = _score(estimator, x_all[held_indices])
            held_scores = _calibrated_scores(calibrator, held_raw_scores)
            held_decisions = (held_scores >= threshold).astype(int)
            train_slugs = tuple(records[index].episode_slug for index in train_indices)
            for offset, held_index in enumerate(held_indices):
                index = int(held_index)
                score_values[index].append(float(held_scores[offset]))
                decision_values[index].append(int(held_decisions[offset]))
                configurations[index].append(dict(configuration))
                training_folds[index].append(train_slugs)
        if covered != set(range(len(records))):
            raise ValueError("outer repetition does not cover every episode")

    predictions = []
    repeat_count = len(folds)
    for index, record in enumerate(records):
        if len(score_values[index]) != repeat_count:
            raise ValueError(f"episode lacks repeated predictions: {record.episode_slug}")
        predictions.append(
            AdvancedPrediction(
                method=method,
                episode_slug=record.episode_slug,
                target=int(record.target),
                score=float(np.mean(score_values[index])),
                predicted=int(float(np.mean(decision_values[index])) >= 0.5),
                threshold=0.5,
                selected_config={"fold_configurations": configurations[index]},
                training_slugs=training_folds[index][0],
                provenance={
                    "feature_family": feature_family,
                    "outer_training_folds": training_folds[index],
                    "outer_repeats": repeat_count,
                    "score_calibration": "training_only_platt",
                },
            )
        )
    return predictions


def _nested_score_composition(
    records: Sequence[CalibrationRecord], method: str
) -> list[AdvancedPrediction]:
    if len(records) < 5:
        raise ValueError("score composition needs at least five records")
    p1_names = feature_names(records, "semantic_phase1")
    p2_names = feature_names(records, "phase2")
    p1_all = _matrix(records, "semantic_phase1", p1_names)
    p2_all = _matrix(records, "phase2", p2_names)
    raw_any_all = np.asarray(
        [record.features["phase2"]["any_check_likelihood"] for record in records],
        dtype=float,
    )
    y_all = np.asarray([record.target for record in records], dtype=int)
    predictions: list[AdvancedPrediction] = []
    for held_index, held_record in enumerate(records):
        train_indices = np.asarray(
            [index for index in range(len(records)) if index != held_index], dtype=int
        )
        y_train = y_all[train_indices]
        p1_train = p1_all[train_indices]
        p2_train = p2_all[train_indices]
        raw_any_train = raw_any_all[train_indices]
        score_cache: dict[float, tuple[np.ndarray, np.ndarray, float, float]] = {}

        def base_scores(base_c: float) -> tuple[np.ndarray, np.ndarray, float, float]:
            if base_c not in score_cache:
                p1_oof = _crossfit_logistic_scores(p1_train, y_train, base_c)
                p2_oof = _crossfit_logistic_scores(p2_train, y_train, base_c)
                p1_model = _logistic(base_c)
                p2_model = _logistic(base_c)
                p1_model.fit(p1_train, y_train)
                p2_model.fit(p2_train, y_train)
                p1_held = float(_score(p1_model, p1_all[[held_index]])[0])
                p2_held = float(_score(p2_model, p2_all[[held_index]])[0])
                score_cache[base_c] = (p1_oof, p2_oof, p1_held, p2_held)
            return score_cache[base_c]

        best_ap = -1.0
        selected_config: Mapping[str, object] = candidate_manifest()[method].parameter_grid[0]
        selected_inner_scores = np.zeros(len(y_train), dtype=float)
        selected_held_score = 0.0
        for configuration in candidate_manifest()[method].parameter_grid:
            base_c = float(configuration["base_c"])
            p1_oof, p2_oof, p1_held, p2_held = base_scores(base_c)
            if method == "late_fusion":
                meta_train = _fusion_matrix(
                    p1_oof, p2_oof, raw_any_train
                )
                meta_held = _fusion_matrix(
                    np.asarray([p1_held]),
                    np.asarray([p2_held]),
                    np.asarray([raw_any_all[held_index]]),
                )
                meta_c = float(configuration["meta_c"])
                inner_scores = _crossfit_logistic_scores(
                    meta_train, y_train, meta_c
                )
                meta_model = _logistic(meta_c)
                meta_model.fit(meta_train, y_train)
                held_score = float(_score(meta_model, meta_held)[0])
            else:
                weight = float(configuration["phase1_weight"])
                inner_scores = weight * p1_oof + (1.0 - weight) * p2_oof
                held_score = weight * p1_held + (1.0 - weight) * p2_held
            average_precision = float(
                average_precision_score(y_train, inner_scores)
            )
            if average_precision > best_ap:
                best_ap = average_precision
                selected_config = configuration
                selected_inner_scores = inner_scores
                selected_held_score = held_score

        threshold = _best_threshold(y_train, selected_inner_scores)
        predictions.append(
            AdvancedPrediction(
                method=method,
                episode_slug=held_record.episode_slug,
                target=int(held_record.target),
                score=float(selected_held_score),
                predicted=int(selected_held_score >= threshold),
                threshold=float(threshold),
                selected_config=dict(selected_config),
                training_slugs=tuple(records[index].episode_slug for index in train_indices),
                provenance={
                    "outer_held_out_excluded": True,
                    "each_meta_row_cross_fitted": True,
                    "crossfit_row_count": len(train_indices),
                },
            )
        )
    return predictions


def _logistic(c_value: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=20_000,
            random_state=20260803,
        ),
    )


def _crossfit_logistic_scores(
    x_values: np.ndarray, y_values: np.ndarray, c_value: float
) -> np.ndarray:
    scores = np.zeros(len(y_values), dtype=float)
    for fit_indices, validation_indices in _splitter(y_values).split(
        x_values, y_values
    ):
        estimator = _logistic(c_value)
        estimator.fit(x_values[fit_indices], y_values[fit_indices])
        scores[validation_indices] = _score(
            estimator, x_values[validation_indices]
        )
    return scores


def _fusion_matrix(
    phase1_scores: np.ndarray,
    phase2_scores: np.ndarray,
    raw_any_scores: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        (
            phase1_scores,
            phase2_scores,
            phase1_scores * phase2_scores,
            phase1_scores - phase2_scores,
            raw_any_scores,
        )
    )


def bootstrap_intervals(
    predictions: Sequence[AdvancedPrediction],
    *,
    iterations: int = 2_000,
    seed: int = 20260803,
) -> dict[str, tuple[float, float]]:
    """Compute deterministic stratified percentile intervals over predictions."""
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    targets = np.asarray([prediction.target for prediction in predictions], dtype=int)
    positive_indices = np.flatnonzero(targets == 1)
    negative_indices = np.flatnonzero(targets == 0)
    if not len(positive_indices) or not len(negative_indices):
        raise ValueError("bootstrap intervals require both classes")
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "balanced_accuracy": [],
        "in_precision": [],
        "in_recall": [],
        "in_f1": [],
        "average_precision": [],
        "roc_auc": [],
        "hits_at_5": [],
        "hits_at_10": [],
        "hits_at_20": [],
    }
    for _ in range(iterations):
        sampled_indices = np.concatenate(
            (
                rng.choice(positive_indices, size=len(positive_indices), replace=True),
                rng.choice(negative_indices, size=len(negative_indices), replace=True),
            )
        )
        sample = [predictions[int(index)] for index in sampled_indices]
        metrics = evaluate_predictions(sample)
        for key in (
            "balanced_accuracy",
            "in_precision",
            "in_recall",
            "in_f1",
            "average_precision",
            "roc_auc",
        ):
            values[key].append(float(metrics[key]))
        for budget in (5, 10, 20):
            values[f"hits_at_{budget}"].append(
                float(metrics["ranking"]["hits_at_k"][str(budget)])
            )
    return {
        key: (
            float(np.percentile(metric_values, 2.5)),
            float(np.percentile(metric_values, 97.5)),
        )
        for key, metric_values in values.items()
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def load_baseline_predictions(
    path: Path, records: Sequence[CalibrationRecord]
) -> dict[str, list[AdvancedPrediction]]:
    """Load prior out-of-fold predictions after an exact slug/label join."""
    expected = {record.episode_slug: int(record.target) for record in records}
    loaded: dict[str, list[AdvancedPrediction]] = {}
    seen: dict[str, set[str]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            method = str(row["method"])
            slug = str(row["episode_slug"])
            if slug not in expected:
                raise ValueError(f"baseline prediction has unknown episode: {slug}")
            target = int(row["target"])
            if target != expected[slug]:
                raise ValueError(f"baseline prediction label mismatch: {slug}")
            method_seen = seen.setdefault(method, set())
            if slug in method_seen:
                raise ValueError(f"duplicate baseline prediction: {method}/{slug}")
            method_seen.add(slug)
            training_slugs = tuple(json.loads(row.get("training_slugs") or "[]"))
            loaded.setdefault(method, []).append(
                AdvancedPrediction(
                    method=method,
                    episode_slug=slug,
                    target=target,
                    score=float(row["score"]),
                    predicted=int(row["predicted"]),
                    threshold=float(row["threshold"]),
                    selected_config={"chosen_c": float(row.get("chosen_c") or 0.0)},
                    training_slugs=training_slugs,
                    provenance={"source": str(path)},
                )
            )
    expected_slugs = set(expected)
    for method, method_seen in seen.items():
        if method_seen != expected_slugs:
            missing = expected_slugs - method_seen
            raise ValueError(
                f"baseline method {method} is incomplete: {', '.join(sorted(missing))}"
            )
    return loaded


def _raw_reference_predictions(
    records: Sequence[CalibrationRecord], method: str
) -> list[AdvancedPrediction]:
    if method not in RAW_REFERENCES:
        raise ValueError(f"unknown raw reference: {method}")
    predictions = []
    for record in records:
        phase2 = record.features["phase2"]
        if method == "majority_out":
            score, predicted, threshold = 0.0, 0, 1.0
        elif method == "raw_any_check":
            score = float(phase2["any_check_likelihood"])
            predicted = int(phase2["any_check_in"])
            threshold = 0.5
        elif method == "raw_standard_check":
            score = float(phase2["standard_check_likelihood"])
            predicted = int(phase2["standard_check_in"])
            threshold = 0.5
        else:
            score = float(phase2["ranking_score"])
            predicted = int(score >= 0.5)
            threshold = 0.5
        predictions.append(
            AdvancedPrediction(
                method=method,
                episode_slug=record.episode_slug,
                target=int(record.target),
                score=score,
                predicted=predicted,
                threshold=threshold,
                selected_config={"kind": "fixed_reference"},
                training_slugs=(),
                provenance={"source": "frozen_phase2"},
            )
        )
    return predictions


def _generated_report(result: Mapping[str, object]) -> str:
    lines = [
        "# Expanded calibration comparison",
        "",
        f"Population: {result['dataset']['episodes']} episodes, "
        f"{result['dataset']['ins']} Ins, {result['dataset']['outs']} Outs.",
        "",
        "| Method | Balanced accuracy | In precision | In recall | In F1 | AP | ROC AUC | Top 5 / 10 / 20 hits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in result["methods"].items():
        hits = metrics["ranking"]["hits_at_k"]
        lines.append(
            f"| {method} | {metrics['balanced_accuracy']:.3f} | "
            f"{metrics['in_precision']:.3f} | {metrics['in_recall']:.3f} | "
            f"{metrics['in_f1']:.3f} | {metrics['average_precision']:.3f} | "
            f"{metrics['roc_auc']:.3f} | {hits['5']} / {hits['10']} / {hits['20']} |"
        )
    lines.extend(
        [
            "",
            "Nested development estimates; not an untouched holdout evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_advanced_analysis(
    records: Sequence[CalibrationRecord],
    output_root: Path,
    *,
    methods: Sequence[str] | None = None,
    bootstrap_iterations: int = 2_000,
    baseline_predictions_path: Path | None = None,
    fold_maps: FoldMaps | None = None,
) -> dict[str, object]:
    """Evaluate selected advanced candidates and write reproducibility artifacts."""
    all_native_methods = (
        tuple(candidate_manifest()) + tuple(LOGISTIC_BASELINES) + RAW_REFERENCES
    )
    selected_methods = tuple(methods or all_native_methods)
    unknown = set(selected_methods) - set(all_native_methods)
    if unknown:
        raise ValueError("unknown advanced methods: " + ", ".join(sorted(unknown)))
    predictions_by_method = {
        method: (
            _raw_reference_predictions(records, method)
            if method in RAW_REFERENCES
            else repeated_stratified_predictions(
                records,
                method,
                fold_maps=fold_maps,
            )
        )
        for method in selected_methods
    }
    baseline_methods: set[str] = set()
    if baseline_predictions_path is not None:
        baselines = load_baseline_predictions(baseline_predictions_path, records)
        overlap = set(predictions_by_method) & set(baselines)
        if overlap:
            raise ValueError("baseline methods collide: " + ", ".join(sorted(overlap)))
        predictions_by_method.update(baselines)
        baseline_methods = set(baselines)
    metrics = {
        method: evaluate_predictions(predictions)
        for method, predictions in predictions_by_method.items()
    }
    intervals = {
        method: bootstrap_intervals(
            predictions, iterations=bootstrap_iterations, seed=20260803
        )
        for method, predictions in predictions_by_method.items()
    }
    result: dict[str, object] = {
        "dataset": {
            "episodes": len(records),
            "ins": sum(record.target for record in records),
            "outs": len(records) - sum(record.target for record in records),
        },
        "evaluation": "repeated_stratified_nested_development_not_untouched_holdout",
        "outer_protocol": {
            "repeats": len(fold_maps or make_repeated_stratified_folds(records)),
            "fixed_fold_maps_reusable_across_label_definitions": True,
            "training_only_score_calibration": "platt",
        },
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "seed": 20260803,
            "kind": "stratified_episode_percentile_fixed_oof_predictions",
        },
        "methods": metrics,
        "intervals_95": intervals,
    }
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", result)
    manifest_rows: dict[str, object] = {}
    for method in selected_methods:
        if method in candidate_manifest():
            specification = candidate_manifest()[method]
            manifest_rows[method] = {
                "feature_family": specification.feature_family,
                "objective": specification.objective,
                "parameter_grid": [
                    dict(configuration)
                    for configuration in specification.parameter_grid
                ],
            }
        elif method in LOGISTIC_BASELINES:
            manifest_rows[method] = {
                "feature_family": LOGISTIC_BASELINES[method],
                "objective": "classification",
                "parameter_grid": [
                    {"c": c_value} for c_value in (0.01, 0.1, 1.0, 10.0)
                ],
            }
        else:
            manifest_rows[method] = {
                "kind": "fixed_reference",
                "source": "frozen_phase2",
            }
    manifest_rows.update(
        {
            method: {
                "source": str(baseline_predictions_path),
                "kind": "existing_out_of_fold_reference",
            }
            for method in sorted(baseline_methods)
        }
    )
    write_json(output / "method-manifest.json", manifest_rows)
    prediction_rows = [
        {
            "method": method,
            "episode_slug": prediction.episode_slug,
            "target": prediction.target,
            "score": prediction.score,
            "predicted": prediction.predicted,
            "threshold": prediction.threshold,
            "selected_config": dict(prediction.selected_config),
            "training_slugs": list(prediction.training_slugs),
            "provenance": dict(prediction.provenance),
        }
        for method, predictions in predictions_by_method.items()
        for prediction in predictions
    ]
    _write_csv(output / "predictions.csv", prediction_rows)
    ranking_rows = [
        {"method": method, **row}
        for method, method_metrics in metrics.items()
        for row in method_metrics["ranking"]["rows"]
    ]
    _write_csv(output / "rankings.csv", ranking_rows)
    (output / "report.md").write_text(_generated_report(result), encoding="utf-8")
    return result
