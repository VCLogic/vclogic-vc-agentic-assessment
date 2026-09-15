"""Nested personalized TabPFN classification and ranking heads."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import time
from typing import Callable, Protocol, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score

from .personalized_cases import PersonalizedCase
from .personalized_folds import EpisodeFold
from .personalized_matrices import RawFeatureViews, fit_ablation_preprocessor


class Classifier(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> "Classifier": ...
    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...


TabPFNFactory = Callable[..., Classifier]


def default_tabpfn_factory(*, n_estimators: int, seed: int, device: str) -> Classifier:
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(
        n_estimators=n_estimators, random_state=seed, device=device
    )


@dataclass(frozen=True)
class TabPFNCandidate:
    config: str
    scores: np.ndarray


@dataclass(frozen=True)
class SelectedTabPFNHeads:
    classification_config: str
    classification_threshold: float
    ranking_config: str


@dataclass(frozen=True)
class PersonalizedPrediction:
    family: str
    condition: str
    vc_slug: str
    vc_name: str
    episode_slug: str
    target: int
    classification_score: float
    ranking_score: float
    predicted: int
    classification_threshold: float
    classification_config: str
    ranking_config: str
    training_episodes: tuple[str, ...]
    inner_validation_episodes: tuple[tuple[str, ...], ...]
    feature_schema_sha256: str
    runtime_seconds: float
    device: str
    status: str
    classification_inner_oof: tuple[tuple[str, str, float], ...]
    ranking_inner_oof: tuple[tuple[str, str, float], ...]


def _macro_balanced_accuracy(
    targets: np.ndarray, predictions: np.ndarray, vcs: np.ndarray
) -> float:
    values = []
    for vc in sorted(set(vcs.tolist())):
        selected = vcs == vc
        if len(set(targets[selected].tolist())) == 2:
            values.append(balanced_accuracy_score(targets[selected], predictions[selected]))
    return float(np.mean(values)) if values else float(
        balanced_accuracy_score(targets, predictions)
    )


def _macro_average_precision(
    targets: np.ndarray, scores: np.ndarray, vcs: np.ndarray
) -> float:
    values = []
    for vc in sorted(set(vcs.tolist())):
        selected = vcs == vc
        if len(set(targets[selected].tolist())) == 2:
            values.append(average_precision_score(targets[selected], scores[selected]))
    return float(np.mean(values)) if values else float(
        average_precision_score(targets, scores)
    )


def _best_threshold(
    targets: np.ndarray, scores: np.ndarray, vcs: np.ndarray
) -> tuple[float, tuple[float, float, float]]:
    unique = sorted(set(float(value) for value in scores))
    candidates = [0.0, 1.0, *unique]
    candidates.extend((left + right) / 2 for left, right in zip(unique, unique[1:]))
    best = (0.5, (-1.0, -1.0, -1.0))
    for threshold in sorted(set(candidates)):
        predicted = (scores >= threshold).astype(int)
        negatives = targets == 0
        specificity = float(np.mean(predicted[negatives] == 0)) if negatives.any() else 0.0
        positive_predictions = int(predicted.sum())
        precision = float(((predicted == 1) & (targets == 1)).sum() / positive_predictions) if positive_predictions else 0.0
        metrics = (
            _macro_balanced_accuracy(targets, predicted, vcs),
            specificity,
            precision,
        )
        if metrics > best[1]:
            best = (float(threshold), metrics)
    return best


def select_tabpfn_heads(
    targets: np.ndarray,
    vcs: np.ndarray,
    candidates: Sequence[TabPFNCandidate],
) -> SelectedTabPFNHeads:
    """Select classification and within-VC ranking candidates independently."""
    if not candidates:
        raise ValueError("TabPFN selection requires candidates")
    classification = None
    ranking = None
    for candidate in candidates:
        scores = np.asarray(candidate.scores, dtype=float)
        if scores.shape != targets.shape or not np.isfinite(scores).all():
            raise ValueError("TabPFN candidate scores are invalid")
        threshold, metrics = _best_threshold(targets, scores, vcs)
        classification_key = (*metrics, candidate.config)
        if classification is None or classification_key > classification[0]:
            classification = (classification_key, candidate.config, threshold)
        ap = _macro_average_precision(targets, scores, vcs)
        ranking_key = (ap, candidate.config)
        if ranking is None or ranking_key > ranking[0]:
            ranking = (ranking_key, candidate.config)
    assert classification is not None and ranking is not None
    return SelectedTabPFNHeads(
        classification_config=classification[1],
        classification_threshold=float(classification[2]),
        ranking_config=ranking[1],
    )


def _configuration(n_estimators: int, pca_components: int) -> str:
    return f"n_estimators={n_estimators};pca_components={pca_components}"


def _parse_configuration(value: str) -> tuple[int, int]:
    fields = dict(item.split("=", 1) for item in value.split(";"))
    return int(fields["n_estimators"]), int(fields["pca_components"])


def _score_model(model: Classifier, matrix: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.predict_proba(matrix), dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 2 or not np.isfinite(scores).all():
        raise ValueError("TabPFN predict_proba returned an invalid matrix")
    result = scores[:, 1]
    if ((result < 0) | (result > 1)).any():
        raise ValueError("TabPFN probability lies outside [0, 1]")
    return result


def _fit_selected(
    cases: Sequence[PersonalizedCase],
    views: Sequence[RawFeatureViews],
    fold: EpisodeFold,
    condition: str,
    config: str,
    *,
    device: str,
    seed: int,
    factory: TabPFNFactory,
) -> tuple[np.ndarray, tuple[str, ...]]:
    estimators, components = _parse_configuration(config)
    preprocessor = fit_ablation_preprocessor(
        views,
        train_indices=fold.train_indices,
        condition=condition,
        pca_components=components,
        maximum_features=2000,
    )
    x_train = preprocessor.transform(views, fold.train_indices)
    x_held = preprocessor.transform(views, fold.test_indices)
    y_train = np.asarray([cases[index].target for index in fold.train_indices], dtype=int)
    model = factory(n_estimators=estimators, seed=seed, device=device).fit(x_train, y_train)
    return _score_model(model, x_held), preprocessor.feature_names


def run_tabpfn_fold(
    cases: Sequence[PersonalizedCase],
    views: Sequence[RawFeatureViews],
    fold: EpisodeFold,
    *,
    condition: str,
    n_estimators: Sequence[int],
    pca_components: Sequence[int],
    device: str,
    seed: int,
    factory: TabPFNFactory = default_tabpfn_factory,
) -> list[PersonalizedPrediction]:
    """Select both heads inside an outer fold and predict all held investor rows."""
    started = time.monotonic()
    if len(cases) != len(views):
        raise ValueError("TabPFN cases and views are misaligned")
    oof_indices = tuple(index for _, valid in fold.inner_splits for index in valid)
    if set(oof_indices) != set(fold.train_indices) or len(oof_indices) != len(set(oof_indices)):
        raise ValueError("inner folds must cover each outer-training row exactly once")
    position = {index: offset for offset, index in enumerate(fold.train_indices)}
    targets = np.asarray([cases[index].target for index in fold.train_indices], dtype=int)
    vcs = np.asarray([cases[index].vc_slug for index in fold.train_indices], dtype=object)
    candidates = []
    for estimator_count in n_estimators:
        for component_count in pca_components:
            scores = np.zeros(len(fold.train_indices), dtype=float)
            for split_index, (inner_train, inner_valid) in enumerate(fold.inner_splits):
                preprocessor = fit_ablation_preprocessor(
                    views,
                    train_indices=inner_train,
                    condition=condition,
                    pca_components=component_count,
                    maximum_features=2000,
                )
                x_train = preprocessor.transform(views, inner_train)
                x_valid = preprocessor.transform(views, inner_valid)
                y_train = np.asarray([cases[index].target for index in inner_train], dtype=int)
                model = factory(
                    n_estimators=estimator_count,
                    seed=seed + split_index,
                    device=device,
                ).fit(x_train, y_train)
                predicted = _score_model(model, x_valid)
                for index, score in zip(inner_valid, predicted, strict=True):
                    scores[position[index]] = score
            candidates.append(TabPFNCandidate(
                _configuration(estimator_count, component_count), scores
            ))
    selected = select_tabpfn_heads(targets, vcs, candidates)
    candidate_by_config = {candidate.config: candidate.scores for candidate in candidates}
    classification_inner_oof = tuple(
        (
            cases[index].vc_slug,
            cases[index].episode_slug,
            float(candidate_by_config[selected.classification_config][position[index]]),
        )
        for index in fold.train_indices
    )
    ranking_inner_oof = tuple(
        (
            cases[index].vc_slug,
            cases[index].episode_slug,
            float(candidate_by_config[selected.ranking_config][position[index]]),
        )
        for index in fold.train_indices
    )
    classification_scores, feature_names = _fit_selected(
        cases, views, fold, condition, selected.classification_config,
        device=device, seed=seed + 50000, factory=factory,
    )
    ranking_scores, _ = _fit_selected(
        cases, views, fold, condition, selected.ranking_config,
        device=device, seed=seed + 60000, factory=factory,
    )
    feature_digest = sha256(json.dumps(
        feature_names, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    training_episodes = tuple(sorted({cases[index].episode_slug for index in fold.train_indices}))
    validation_episodes = tuple(
        tuple(sorted({cases[index].episode_slug for index in valid}))
        for _, valid in fold.inner_splits
    )
    runtime = time.monotonic() - started
    return [
        PersonalizedPrediction(
            family="tabpfn", condition=condition,
            vc_slug=cases[index].vc_slug, vc_name=cases[index].vc_name,
            episode_slug=cases[index].episode_slug, target=cases[index].target,
            classification_score=float(classification_score),
            ranking_score=float(ranking_score),
            predicted=int(classification_score >= selected.classification_threshold),
            classification_threshold=selected.classification_threshold,
            classification_config=selected.classification_config,
            ranking_config=selected.ranking_config,
            training_episodes=training_episodes,
            inner_validation_episodes=validation_episodes,
            feature_schema_sha256=feature_digest,
            runtime_seconds=runtime,
            device=device,
            status="complete",
            classification_inner_oof=classification_inner_oof,
            ranking_inner_oof=ranking_inner_oof,
        )
        for index, classification_score, ranking_score in zip(
            fold.test_indices, classification_scores, ranking_scores, strict=True
        )
    ]
