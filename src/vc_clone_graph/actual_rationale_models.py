"""Nested out-of-fold models for observed and predicted rationale features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from joblib import Parallel, delayed
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .actual_rationale_cases import RationaleModelCase
from .partial_pooling_evaluation import select_threshold


SOURCE_CONDITIONS = {
    "actual_to_actual": ("actual", "actual"),
    "actual_to_predicted": ("actual", "predicted"),
    "predicted_to_predicted": ("predicted", "predicted"),
}


@dataclass(frozen=True)
class RationalePrediction:
    vc_slug: str
    vc_name: str
    episode_slug: str
    held_group: str
    target: int
    method: str
    model_family: str
    source_condition: str
    training_source: str
    test_source: str
    score: float
    balanced_decision: int
    precision_decision: int
    balanced_threshold: float
    precision_threshold: float
    selected_params: tuple[tuple[str, float], ...]
    training_episode_slugs: tuple[str, ...]
    training_groups: tuple[str, ...]
    training_vcs: tuple[str, ...]
    training_indices: tuple[int, ...]


def _source(condition: str) -> tuple[str, str]:
    try:
        return SOURCE_CONDITIONS[condition]
    except KeyError as error:
        raise ValueError(f"unknown source condition: {condition}") from error


def _features(case: RationaleModelCase, source: str) -> dict[str, float]:
    if source == "actual":
        return case.actual_features
    if source == "predicted":
        return case.predicted_features
    raise ValueError(f"unknown rationale source: {source}")


def _estimator(model_family: str, params: Mapping[str, float], seed: int):
    if model_family == "logistic":
        return make_pipeline(
            DictVectorizer(sparse=True),
            StandardScaler(with_mean=False),
            LogisticRegression(
                C=float(params["c"]), solver="liblinear", class_weight="balanced",
                max_iter=2_000, random_state=seed,
            ),
        )
    if model_family == "tree":
        return make_pipeline(
            DictVectorizer(sparse=False),
            RandomForestClassifier(
                n_estimators=64,
                max_depth=int(params["max_depth"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
        )
    raise ValueError(f"unknown per-VC model family: {model_family}")


def _candidates(model_family: str) -> tuple[dict[str, float], ...]:
    if model_family == "logistic":
        return tuple({"c": value} for value in (0.1, 1.0, 10.0))
    if model_family == "tree":
        return tuple(
            {"max_depth": float(depth), "min_samples_leaf": float(leaf)}
            for depth in (2, 3, 4)
            for leaf in (2, 4)
        )
    raise ValueError(f"unknown per-VC model family: {model_family}")


def _inner_folds(targets: np.ndarray, splits: int, seed: int):
    count = min(splits, int(targets.sum()), int(len(targets) - targets.sum()))
    if count < 2:
        raise ValueError("nested rationale model needs at least two cases per class")
    return tuple(StratifiedKFold(
        n_splits=count, shuffle=True, random_state=seed
    ).split(np.zeros(len(targets)), targets))


def _group_inner_folds(
    targets: np.ndarray, groups: np.ndarray, splits: int, seed: int
):
    maximum = min(splits, len(set(str(group) for group in groups)))
    for count in range(maximum, 1, -1):
        folds = tuple(StratifiedGroupKFold(
            n_splits=count, shuffle=True, random_state=seed
        ).split(np.zeros(len(targets)), targets, groups))
        if all(len(set(targets[fit])) == 2 for fit, _ in folds):
            return folds
    raise ValueError("grouped rationale model needs valid two-class inner folds")


def _select_model(
    training_maps: Sequence[dict[str, float]],
    validation_maps: Sequence[dict[str, float]],
    targets: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    model_family: str,
    seed: int,
) -> tuple[dict[str, float], float, float]:
    candidates: list[tuple[float, int, dict[str, float], np.ndarray]] = []
    for candidate_index, params in enumerate(_candidates(model_family)):
        scores = np.zeros(len(targets), dtype=float)
        for fold_index, (fit, validation) in enumerate(folds):
            estimator = _estimator(model_family, params, seed + fold_index)
            estimator.fit(
                [training_maps[int(index)] for index in fit], targets[fit]
            )
            scores[validation] = estimator.predict_proba(
                [validation_maps[int(index)] for index in validation]
            )[:, 1]
        candidates.append((
            float(average_precision_score(targets, scores)),
            -candidate_index,
            params,
            scores,
        ))
    _, _, selected, scores = max(candidates, key=lambda row: (row[0], row[1]))
    return (
        selected,
        select_threshold(targets, scores, "balanced_accuracy"),
        select_threshold(targets, scores, "f0_5"),
    )


def _per_vc_fold(
    held_index: int,
    vc_indices: tuple[int, ...],
    cases: Sequence[RationaleModelCase],
    model_family: str,
    source_condition: str,
    inner_splits: int,
    fold_seed: int,
) -> tuple[int, RationalePrediction]:
    training_source, test_source = _source(source_condition)
    training_indices = tuple(index for index in vc_indices if index != held_index)
    training_cases = [cases[index] for index in training_indices]
    targets = np.asarray([case.target for case in training_cases], dtype=int)
    fit_maps = [_features(case, training_source) for case in training_cases]
    validation_maps = [_features(case, test_source) for case in training_cases]
    params, balanced_threshold, precision_threshold = _select_model(
        fit_maps,
        validation_maps,
        targets,
        _inner_folds(targets, inner_splits, fold_seed),
        model_family=model_family,
        seed=fold_seed,
    )
    estimator = _estimator(model_family, params, fold_seed)
    estimator.fit(fit_maps, targets)
    held = cases[held_index]
    score = float(estimator.predict_proba([_features(held, test_source)])[0, 1])
    return held_index, RationalePrediction(
        vc_slug=held.vc_slug,
        vc_name=held.vc_name,
        episode_slug=held.episode_slug,
        held_group=held.group,
        target=held.target,
        method=f"per_vc_{model_family}__{source_condition}",
        model_family=model_family,
        source_condition=source_condition,
        training_source=training_source,
        test_source=test_source,
        score=score,
        balanced_decision=int(score >= balanced_threshold),
        precision_decision=int(score >= precision_threshold),
        balanced_threshold=balanced_threshold,
        precision_threshold=precision_threshold,
        selected_params=tuple(sorted((name, float(value)) for name, value in params.items())),
        training_episode_slugs=tuple(case.episode_slug for case in training_cases),
        training_groups=tuple(case.group for case in training_cases),
        training_vcs=tuple(case.vc_slug for case in training_cases),
        training_indices=training_indices,
    )


def nested_per_vc_predictions(
    cases: Sequence[RationaleModelCase],
    *,
    model_family: str,
    source_condition: str,
    inner_splits: int = 3,
    seed: int = 20260817,
    n_jobs: int = 1,
) -> list[RationalePrediction]:
    _source(source_condition)
    _candidates(model_family)
    tasks: list[tuple[int, tuple[int, ...], int]] = []
    fold_index = 0
    for vc_slug in sorted({case.vc_slug for case in cases}):
        vc_indices = tuple(
            index for index, case in enumerate(cases) if case.vc_slug == vc_slug
        )
        if {cases[index].target for index in vc_indices} != {0, 1}:
            raise ValueError(f"per-VC rationale model needs both classes: {vc_slug}")
        for held_index in vc_indices:
            tasks.append((held_index, vc_indices, fold_index))
            fold_index += 1
    outputs = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_per_vc_fold)(
            held_index, vc_indices, cases, model_family, source_condition,
            inner_splits, seed + task_index,
        )
        for held_index, vc_indices, task_index in tasks
    )
    result: list[RationalePrediction | None] = [None] * len(cases)
    for held_index, prediction in outputs:
        result[held_index] = prediction
    if any(item is None for item in result):
        raise RuntimeError("per-VC rationale model did not predict every case")
    return [item for item in result if item is not None]


def _hier_map(
    case: RationaleModelCase, source: str, deviation_scale: float
) -> dict[str, float]:
    base = _features(case, source)
    result = {f"global__{name}": value for name, value in base.items()}
    result.update({
        f"deviation__{case.vc_slug}__{name}": value * deviation_scale
        for name, value in base.items()
    })
    result[f"deviation__{case.vc_slug}__intercept"] = deviation_scale
    return result


def _select_hierarchical(
    cases: Sequence[RationaleModelCase],
    source_condition: str,
    inner_splits: int,
    seed: int,
) -> tuple[dict[str, float], float, float]:
    training_source, test_source = _source(source_condition)
    targets = np.asarray([case.target for case in cases], dtype=int)
    groups = np.asarray([case.group for case in cases], dtype=object)
    folds = _group_inner_folds(targets, groups, inner_splits, seed)
    candidates: list[tuple[float, int, dict[str, float], np.ndarray]] = []
    candidate_index = 0
    for c_value in (0.1, 1.0, 10.0):
        for deviation_scale in (0.0, 0.25, 0.5, 1.0):
            params = {"c": c_value, "deviation_scale": deviation_scale}
            fit_maps = [_hier_map(case, training_source, deviation_scale) for case in cases]
            validation_maps = [_hier_map(case, test_source, deviation_scale) for case in cases]
            scores = np.zeros(len(cases), dtype=float)
            for inner_index, (fit, validation) in enumerate(folds):
                estimator = _estimator("logistic", params, seed + inner_index)
                estimator.fit([fit_maps[int(index)] for index in fit], targets[fit])
                scores[validation] = estimator.predict_proba(
                    [validation_maps[int(index)] for index in validation]
                )[:, 1]
            candidates.append((
                float(average_precision_score(targets, scores)),
                -candidate_index,
                params,
                scores,
            ))
            candidate_index += 1
    _, _, params, scores = max(candidates, key=lambda row: (row[0], row[1]))
    return (
        params,
        select_threshold(targets, scores, "balanced_accuracy"),
        select_threshold(targets, scores, "f0_5"),
    )


def _hierarchical_fold(
    held_group: str,
    cases: Sequence[RationaleModelCase],
    source_condition: str,
    inner_splits: int,
    fold_seed: int,
) -> list[tuple[int, RationalePrediction]]:
    training_source, test_source = _source(source_condition)
    held_indices = tuple(index for index, case in enumerate(cases) if case.group == held_group)
    training_indices = tuple(index for index, case in enumerate(cases) if case.group != held_group)
    training_cases = [cases[index] for index in training_indices]
    params, balanced_threshold, precision_threshold = _select_hierarchical(
        training_cases, source_condition, inner_splits, fold_seed
    )
    scale = float(params["deviation_scale"])
    estimator = _estimator("logistic", params, fold_seed)
    estimator.fit(
        [_hier_map(case, training_source, scale) for case in training_cases],
        np.asarray([case.target for case in training_cases], dtype=int),
    )
    held_cases = [cases[index] for index in held_indices]
    scores = estimator.predict_proba(
        [_hier_map(case, test_source, scale) for case in held_cases]
    )[:, 1]
    training_groups = tuple(sorted({case.group for case in training_cases}))
    training_vcs = tuple(sorted({case.vc_slug for case in training_cases}))
    output: list[tuple[int, RationalePrediction]] = []
    for held_index, held, raw_score in zip(held_indices, held_cases, scores, strict=True):
        score = float(raw_score)
        output.append((held_index, RationalePrediction(
            vc_slug=held.vc_slug,
            vc_name=held.vc_name,
            episode_slug=held.episode_slug,
            held_group=held.group,
            target=held.target,
            method=f"hierarchical_logistic__{source_condition}",
            model_family="hierarchical_logistic",
            source_condition=source_condition,
            training_source=training_source,
            test_source=test_source,
            score=score,
            balanced_decision=int(score >= balanced_threshold),
            precision_decision=int(score >= precision_threshold),
            balanced_threshold=balanced_threshold,
            precision_threshold=precision_threshold,
            selected_params=tuple(sorted((name, float(value)) for name, value in params.items())),
            training_episode_slugs=tuple(case.episode_slug for case in training_cases),
            training_groups=training_groups,
            training_vcs=training_vcs,
            training_indices=training_indices,
        )))
    return output


def nested_hierarchical_predictions(
    cases: Sequence[RationaleModelCase],
    *,
    source_condition: str,
    inner_splits: int = 3,
    seed: int = 20260817,
    n_jobs: int = 1,
) -> list[RationalePrediction]:
    _source(source_condition)
    groups = sorted({case.group for case in cases})
    outputs = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_hierarchical_fold)(
            held_group, cases, source_condition, inner_splits, seed + fold_index
        )
        for fold_index, held_group in enumerate(groups)
    )
    result: list[RationalePrediction | None] = [None] * len(cases)
    for fold in outputs:
        for held_index, prediction in fold:
            result[held_index] = prediction
    if any(item is None for item in result):
        raise RuntimeError("hierarchical rationale model did not predict every case")
    return [item for item in result if item is not None]


def run_all_rationale_models(
    cases: Sequence[RationaleModelCase],
    *,
    inner_splits: int = 3,
    seed: int = 20260817,
    n_jobs: int = 1,
) -> dict[str, list[RationalePrediction]]:
    """Run the three model families under all three source conditions."""
    result: dict[str, list[RationalePrediction]] = {}
    offset = 0
    for source_condition in SOURCE_CONDITIONS:
        for model_family in ("logistic", "tree"):
            predictions = nested_per_vc_predictions(
                cases,
                model_family=model_family,
                source_condition=source_condition,
                inner_splits=inner_splits,
                seed=seed + offset,
                n_jobs=n_jobs,
            )
            result[predictions[0].method] = predictions
            offset += 10_000
        predictions = nested_hierarchical_predictions(
            cases,
            source_condition=source_condition,
            inner_splits=inner_splits,
            seed=seed + offset,
            n_jobs=n_jobs,
        )
        result[predictions[0].method] = predictions
        offset += 10_000
    return result
