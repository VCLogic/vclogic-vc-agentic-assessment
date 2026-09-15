"""Leakage-safe nested per-investor models for structured decision fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .structured_fusion_features import StructuredFusionCase, feature_view


DEFAULT_ELASTIC_CONFIGS = tuple(
    (c, ratio)
    for c in (0.1, 1.0, 10.0)
    for ratio in (0.5, 1.0)
)
DEFAULT_FOREST_CONFIGS = ((2, 0.10), (3, 0.10), (3, 0.15))


@dataclass(frozen=True)
class StructuredFusionPrediction:
    vc_slug: str
    vc_name: str
    episode_slug: str
    target: int
    condition: str
    model_family: str
    classification_score: float
    ranking_score: float
    predicted: int
    classification_threshold: float
    classification_config: str
    ranking_config: str
    training_episode_slugs: tuple[str, ...]
    inner_validation_episode_slugs: tuple[tuple[str, ...], ...]
    classification_selected_features: tuple[tuple[str, float], ...]
    ranking_selected_features: tuple[tuple[str, float], ...]


@dataclass
class _Fit:
    model: object
    vectorizer: DictVectorizer
    scaler: StandardScaler | None
    names: np.ndarray


def _fit(
    rows: Sequence[Mapping[str, float]],
    y: np.ndarray,
    family: str,
    config: tuple[float, float],
    seed: int,
    forest_trees: int,
) -> _Fit:
    vectorizer = DictVectorizer(sparse=False)
    x = vectorizer.fit_transform(rows)
    names = np.asarray(vectorizer.get_feature_names_out())
    varying = np.ptp(x, axis=0) > 0
    if not varying.any():
        x = np.zeros((len(rows), 1), dtype=float)
        names = np.asarray(["__constant__"])
    else:
        x = x[:, varying]
        names = names[varying]
    # DictVectorizer must still transform the complete dictionary at prediction;
    # retain the fold-local varying-column mask on the fitted model.
    if family == "elastic":
        scaler = StandardScaler().fit(x)
        # SGD gives the same elastic-net logistic objective while making the
        # 301 outer folds computationally practical and deterministic.
        model = SGDClassifier(
            loss="log_loss", penalty="elasticnet",
            alpha=1.0 / (float(config[0]) * max(1, len(rows))),
            l1_ratio=float(config[1]), class_weight="balanced", max_iter=200,
            random_state=seed, tol=1e-2,
        ).fit(scaler.transform(x), y)
    else:
        scaler = None
        min_leaf = max(1, int(np.ceil(float(config[1]) * len(rows))))
        model = RandomForestClassifier(
            n_estimators=forest_trees, max_depth=int(config[0]),
            min_samples_leaf=min_leaf, class_weight="balanced_subsample",
            max_features="sqrt", random_state=seed, n_jobs=1,
        ).fit(x, y)
    setattr(model, "_structured_varying", varying)
    return _Fit(model, vectorizer, scaler, names)


def _score(fit: _Fit, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
    x = fit.vectorizer.transform(rows)
    varying = getattr(fit.model, "_structured_varying")
    if varying.any():
        x = x[:, varying]
    else:
        x = np.zeros((len(rows), 1), dtype=float)
    if fit.scaler is not None:
        x = fit.scaler.transform(x)
    return fit.model.predict_proba(x)[:, 1]


def _importance(fit: _Fit) -> tuple[tuple[str, float], ...]:
    if hasattr(fit.model, "coef_"):
        values = np.asarray(fit.model.coef_[0])
    else:
        values = np.asarray(fit.model.feature_importances_)
    ranked = sorted(
        ((str(name), float(value)) for name, value in zip(fit.names, values)
         if abs(float(value)) > 1e-12),
        key=lambda row: (-abs(row[1]), row[0]),
    )
    return tuple(ranked[:50]) or (("__constant__", 0.0),)


def _threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, tuple[float, float, float]]:
    candidates = sorted({0.0, 0.5, 1.0, *map(float, scores)})
    best = (0.5, (-1.0, -1.0, -1.0))
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        specificity = float(((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum()))
        metrics = (
            float(balanced_accuracy_score(y, pred)),
            specificity,
            float(precision_score(y, pred, zero_division=0)),
        )
        if metrics > best[1]:
            best = (float(threshold), metrics)
    return best


def _folds(y: np.ndarray, seed: int, requested: int) -> list[tuple[np.ndarray, np.ndarray]]:
    minority = int(min(np.bincount(y, minlength=2)))
    n_splits = min(requested, minority)
    if n_splits < 2:
        raise ValueError("nested evaluation requires at least two cases in each class")
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(y, y))


def _config_text(family: str, config: tuple[float, float]) -> str:
    if family == "elastic":
        return f"C={config[0]:g};l1_ratio={config[1]:g}"
    return f"max_depth={int(config[0])};min_leaf_fraction={config[1]:g}"


def _oof(
    rows: Sequence[Mapping[str, float]], y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]], family: str,
    config: tuple[float, float], seed: int, forest_trees: int,
) -> np.ndarray:
    result = np.zeros(len(rows), dtype=float)
    for fold_index, (train, valid) in enumerate(folds):
        fit = _fit([rows[i] for i in train], y[train], family, config,
                   seed + fold_index, forest_trees)
        result[valid] = _score(fit, [rows[i] for i in valid])
    return result


def _select_family(
    rows: Sequence[Mapping[str, float]], y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]], family: str,
    configs: Sequence[tuple[float, float]], seed: int, forest_trees: int,
) -> tuple[tuple[float, float], np.ndarray, float, tuple[float, float], dict[tuple[float, float], np.ndarray]]:
    cache: dict[tuple[float, float], np.ndarray] = {}
    classification_best: tuple[tuple[float, float], np.ndarray, float, tuple[float, float]] | None = None
    ranking_best: tuple[tuple[float, float], np.ndarray, float] | None = None
    for config in configs:
        scores = _oof(rows, y, folds, family, config, seed, forest_trees)
        cache[config] = scores
        threshold, class_metrics = _threshold(y, scores)
        class_key = (class_metrics[0], class_metrics[1], class_metrics[2], -config[0], -config[1])
        if classification_best is None or class_key > classification_best[3]:
            classification_best = (config, scores, threshold, class_key)
        ap = float(average_precision_score(y, scores))
        rank_key = (ap, -config[0], -config[1])
        if ranking_best is None or rank_key > (ranking_best[2], -ranking_best[0][0], -ranking_best[0][1]):
            ranking_best = (config, scores, ap)
    assert classification_best is not None and ranking_best is not None
    return (
        classification_best[0], classification_best[1], classification_best[2],
        ranking_best[0], cache,
    )


def _one_outer(
    all_cases: Sequence[StructuredFusionCase], held_index: int, condition: str,
    model_family: str, inner_splits: int, seed: int,
    elastic_configs: Sequence[tuple[float, float]],
    forest_configs: Sequence[tuple[float, float]], forest_trees: int,
    blend_weights: Sequence[float],
) -> StructuredFusionPrediction:
    held = all_cases[held_index]
    train = [row for index, row in enumerate(all_cases) if index != held_index]
    rows = [feature_view(row, condition) for row in train]
    y = np.asarray([row.target for row in train], dtype=int)
    folds = _folds(y, seed + held_index, inner_splits)
    validation = tuple(tuple(train[i].episode_slug for i in valid) for _, valid in folds)

    families = ("elastic", "forest") if model_family == "blend" else (model_family,)
    selected: dict[str, tuple] = {}
    for family in families:
        configs = elastic_configs if family == "elastic" else forest_configs
        selected[family] = _select_family(
            rows, y, folds, family, configs, seed + held_index * 17,
            forest_trees,
        )

    if model_family != "blend":
        c_config, c_oof, threshold, r_config, _ = selected[model_family]
        c_fit = _fit(rows, y, model_family, c_config, seed + held_index, forest_trees)
        r_fit = _fit(rows, y, model_family, r_config, seed + held_index, forest_trees)
        c_score = float(_score(c_fit, [feature_view(held, condition)])[0])
        r_score = float(_score(r_fit, [feature_view(held, condition)])[0])
        c_text = _config_text(model_family, c_config)
        r_text = _config_text(model_family, r_config)
    else:
        ec, ecoof, _, er, _ = selected["elastic"]
        fc, fcoof, _, fr, _ = selected["forest"]
        er_oof = selected["elastic"][4][er]
        fr_oof = selected["forest"][4][fr]
        best_c = max(
            blend_weights,
            key=lambda w: (*_threshold(y, w * ecoof + (1 - w) * fcoof)[1], -w),
        )
        best_r = max(
            blend_weights,
            key=lambda w: (average_precision_score(y, w * er_oof + (1 - w) * fr_oof), -w),
        )
        c_oof = best_c * ecoof + (1 - best_c) * fcoof
        threshold = _threshold(y, c_oof)[0]
        ec_fit = _fit(rows, y, "elastic", ec, seed + held_index, forest_trees)
        fc_fit = _fit(rows, y, "forest", fc, seed + held_index, forest_trees)
        er_fit = _fit(rows, y, "elastic", er, seed + held_index, forest_trees)
        fr_fit = _fit(rows, y, "forest", fr, seed + held_index, forest_trees)
        held_row = [feature_view(held, condition)]
        c_score = float(best_c * _score(ec_fit, held_row)[0] + (1-best_c) * _score(fc_fit, held_row)[0])
        r_score = float(best_r * _score(er_fit, held_row)[0] + (1-best_r) * _score(fr_fit, held_row)[0])
        c_fit, r_fit = ec_fit, er_fit
        c_text = f"elastic_weight={best_c:g};elastic[{_config_text('elastic', ec)}];forest[{_config_text('forest', fc)}]"
        r_text = f"elastic_weight={best_r:g};elastic[{_config_text('elastic', er)}];forest[{_config_text('forest', fr)}]"

    return StructuredFusionPrediction(
        vc_slug=held.vc_slug, vc_name=held.vc_name,
        episode_slug=held.episode_slug, target=held.target, condition=condition,
        model_family=model_family, classification_score=c_score,
        ranking_score=r_score, predicted=int(c_score >= threshold),
        classification_threshold=float(threshold), classification_config=c_text,
        ranking_config=r_text,
        training_episode_slugs=tuple(row.episode_slug for row in train),
        inner_validation_episode_slugs=validation,
        classification_selected_features=_importance(c_fit),
        ranking_selected_features=_importance(r_fit),
    )


def nested_two_head_predictions(
    cases: Sequence[StructuredFusionCase], *, condition: str,
    model_family: str, inner_splits: int = 3, seed: int = 20260820,
    n_jobs: int = 1,
    elastic_configs: Sequence[tuple[float, float]] = DEFAULT_ELASTIC_CONFIGS,
    forest_configs: Sequence[tuple[float, float]] = DEFAULT_FOREST_CONFIGS,
    forest_trees: int = 64,
    blend_weights: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> list[StructuredFusionPrediction]:
    """Generate nested leave-one-out predictions separately within each VC."""
    if model_family not in {"elastic", "forest", "blend"}:
        raise ValueError(f"unknown model family: {model_family}")
    result: list[StructuredFusionPrediction] = []
    for vc_slug in sorted({row.vc_slug for row in cases}):
        vc_cases = sorted(
            (row for row in cases if row.vc_slug == vc_slug),
            key=lambda row: row.episode_slug,
        )
        predictions = Parallel(n_jobs=n_jobs)(
            delayed(_one_outer)(
                vc_cases, index, condition, model_family, inner_splits, seed,
                tuple(elastic_configs), tuple(forest_configs), forest_trees,
                tuple(blend_weights),
            )
            for index in range(len(vc_cases))
        )
        result.extend(predictions)
    return result
