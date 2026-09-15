"""Conditional cross-fitted stacking for complementary personalized models."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import time
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from .personalized_cases import PersonalizedCase
from .personalized_folds import EpisodeFold
from .personalized_tabpfn import (
    PersonalizedPrediction,
    TabPFNCandidate,
    select_tabpfn_heads,
)


CaseKey = tuple[str, str]


@dataclass(frozen=True)
class ComponentFoldScores:
    family: str
    classification_train_oof: Mapping[CaseKey, float]
    ranking_train_oof: Mapping[CaseKey, float]
    classification_held: Mapping[CaseKey, float]
    ranking_held: Mapping[CaseKey, float]


@dataclass(frozen=True)
class EnsembleFoldResult:
    status: str
    reason: str
    predictions: tuple[PersonalizedPrediction, ...]


def _score_map(rows: object, *, field: str) -> dict[CaseKey, float]:
    if not isinstance(rows, list):
        raise ValueError(f"component {field} must be a list")
    result: dict[CaseKey, float] = {}
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"component {field} row is invalid")
        key = (str(row[0]), str(row[1]))
        if key in result:
            raise ValueError(f"component {field} contains duplicate keys")
        result[key] = float(row[2])
    return result


def component_scores_from_checkpoint(
    payload: Mapping[str, object], *, family: str
) -> ComponentFoldScores:
    """Load one completed component fold without losing OOF provenance."""
    if payload.get("status") != "complete":
        raise ValueError(f"{family} component checkpoint is not complete")
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError(f"{family} component checkpoint has no predictions")
    first = predictions[0]
    if not isinstance(first, Mapping):
        raise ValueError(f"{family} component prediction is invalid")
    classification_train = _score_map(
        first.get("classification_inner_oof"), field="classification_inner_oof"
    )
    ranking_train = _score_map(
        first.get("ranking_inner_oof"), field="ranking_inner_oof"
    )
    classification_held: dict[CaseKey, float] = {}
    ranking_held: dict[CaseKey, float] = {}
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            raise ValueError(f"{family} component prediction is invalid")
        if (
            _score_map(
                prediction.get("classification_inner_oof"),
                field="classification_inner_oof",
            ) != classification_train
            or _score_map(
                prediction.get("ranking_inner_oof"), field="ranking_inner_oof"
            ) != ranking_train
        ):
            raise ValueError(f"{family} component has inconsistent inner OOF scores")
        key = (str(prediction["vc_slug"]), str(prediction["episode_slug"]))
        if key in classification_held:
            raise ValueError(f"{family} component has duplicate held keys")
        classification_held[key] = float(prediction["classification_score"])
        ranking_held[key] = float(prediction["ranking_score"])
    return ComponentFoldScores(
        family=family,
        classification_train_oof=classification_train,
        ranking_train_oof=ranking_train,
        classification_held=classification_held,
        ranking_held=ranking_held,
    )


def _probability(model: LogisticRegression, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)


def _cross_fit_meta_head(
    matrix: np.ndarray,
    targets: np.ndarray,
    fold: EpisodeFold,
    *,
    c_value: float,
    seed: int,
) -> np.ndarray:
    """Return meta-model predictions where every row is scored out of fold."""
    position = {index: offset for offset, index in enumerate(fold.train_indices)}
    scores = np.zeros(len(fold.train_indices), dtype=float)
    covered: set[int] = set()
    for split_index, (inner_train, inner_valid) in enumerate(fold.inner_splits):
        train_offsets = [position[index] for index in inner_train]
        valid_offsets = [position[index] for index in inner_valid]
        model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=500,
            random_state=seed + split_index,
            solver="liblinear",
        ).fit(matrix[train_offsets], targets[train_offsets])
        scores[valid_offsets] = _probability(model, matrix[valid_offsets])
        covered.update(inner_valid)
    if covered != set(fold.train_indices):
        raise ValueError("ensemble inner splits do not cover outer-training rows")
    return scores


def conditional_ensemble_fold(
    cases: Sequence[PersonalizedCase],
    fold: EpisodeFold,
    components: Sequence[ComponentFoldScores],
    raw_phase2_likelihood: Mapping[CaseKey, float],
    *,
    minimum_disagreement_rate: float,
    require_inner_improvement: bool,
    seed: int,
) -> EnsembleFoldResult:
    """Fit separate classification/ranking stackers from inner OOF scores only."""
    started = time.monotonic()
    if len(components) != 2:
        raise ValueError("ensemble requires exactly two component families")
    train_keys = tuple(
        (cases[index].vc_slug, cases[index].episode_slug) for index in fold.train_indices
    )
    held_keys = tuple(
        (cases[index].vc_slug, cases[index].episode_slug) for index in fold.test_indices
    )
    expected_train = set(train_keys)
    expected_held = set(held_keys)
    for component in components:
        if (
            set(component.classification_train_oof) != expected_train
            or set(component.ranking_train_oof) != expected_train
        ):
            raise ValueError(f"{component.family} does not have exact outer-training keys")
        if (
            set(component.classification_held) != expected_held
            or set(component.ranking_held) != expected_held
        ):
            raise ValueError(f"{component.family} does not have exact held keys")
    if not (expected_train | expected_held) <= set(raw_phase2_likelihood):
        raise ValueError("raw Phase 2 likelihood is incomplete")
    disagreement = float(np.mean([
        (components[0].classification_train_oof[key] >= 0.5)
        != (components[1].classification_train_oof[key] >= 0.5)
        for key in train_keys
    ]))
    if disagreement < minimum_disagreement_rate:
        return EnsembleFoldResult(
            "skipped",
            f"component disagreement {disagreement:.3f} is below eligibility threshold",
            (),
        )

    x_class = np.asarray([[ 
        components[0].classification_train_oof[key],
        components[1].classification_train_oof[key],
        components[0].ranking_train_oof[key],
        components[1].ranking_train_oof[key],
        raw_phase2_likelihood[key],
    ] for key in train_keys], dtype=float)
    x_held = np.asarray([[ 
        components[0].classification_held[key],
        components[1].classification_held[key],
        components[0].ranking_held[key],
        components[1].ranking_held[key],
        raw_phase2_likelihood[key],
    ] for key in held_keys], dtype=float)
    y = np.asarray([cases[index].target for index in fold.train_indices], dtype=int)
    vcs = np.asarray([cases[index].vc_slug for index in fold.train_indices], dtype=object)
    classification_oof = _cross_fit_meta_head(
        x_class, y, fold, c_value=0.1, seed=seed,
    )
    ranking_oof = _cross_fit_meta_head(
        x_class, y, fold, c_value=0.03, seed=seed + 1000,
    )
    classification_model = LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=500,
        random_state=seed, solver="liblinear",
    ).fit(x_class, y)
    ranking_model = LogisticRegression(
        C=0.03, class_weight="balanced", max_iter=500,
        random_state=seed + 1, solver="liblinear",
    ).fit(x_class, y)
    selected = select_tabpfn_heads(
        y, vcs, (TabPFNCandidate("ensemble", classification_oof),)
    )
    if require_inner_improvement:
        ensemble_ap = average_precision_score(y, ranking_oof)
        component_ap = max(
            average_precision_score(
                y, np.asarray([component.ranking_train_oof[key] for key in train_keys])
            )
            for component in components
        )
        if ensemble_ap <= component_ap:
            return EnsembleFoldResult(
                "skipped",
                f"inner ensemble AP {ensemble_ap:.3f} did not exceed component AP {component_ap:.3f}",
                (),
            )
    classification_held = _probability(classification_model, x_held)
    ranking_held = _probability(ranking_model, x_held)
    training_episodes = tuple(sorted({cases[index].episode_slug for index in fold.train_indices}))
    classification_inner = tuple(
        (key[0], key[1], float(score))
        for key, score in zip(train_keys, classification_oof, strict=True)
    )
    ranking_inner = tuple(
        (key[0], key[1], float(score))
        for key, score in zip(train_keys, ranking_oof, strict=True)
    )
    schema_digest = sha256(
        b"tabpfn_class,setfit_class,tabpfn_rank,setfit_rank,raw_phase2"
    ).hexdigest()
    runtime = time.monotonic() - started
    predictions = tuple(
        PersonalizedPrediction(
            family="ensemble", condition="full", vc_slug=cases[index].vc_slug,
            vc_name=cases[index].vc_name, episode_slug=cases[index].episode_slug,
            target=cases[index].target,
            classification_score=float(classification_score),
            ranking_score=float(ranking_score),
            predicted=int(classification_score >= selected.classification_threshold),
            classification_threshold=selected.classification_threshold,
            classification_config="C=0.1",
            ranking_config="C=0.03",
            training_episodes=training_episodes,
            inner_validation_episodes=tuple(
                tuple(sorted({cases[row].episode_slug for row in valid}))
                for _, valid in fold.inner_splits
            ),
            feature_schema_sha256=schema_digest,
            runtime_seconds=runtime,
            device="cpu",
            status="complete",
            classification_inner_oof=classification_inner,
            ranking_inner_oof=ranking_inner,
        )
        for index, classification_score, ranking_score in zip(
            fold.test_indices, classification_held, ranking_held, strict=True
        )
    )
    return EnsembleFoldResult("complete", "eligible and fitted", predictions)
