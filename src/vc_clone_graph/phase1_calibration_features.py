"""Leakage-safe fold features for Phase 1 rationale calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA

from .phase1_calibration_cases import CalibrationCase


@dataclass(frozen=True)
class FoldFeatures:
    train: np.ndarray
    evaluate: np.ndarray
    targets: np.ndarray
    feature_names: tuple[str, ...]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def _raw_phase1_features(
    cases: Sequence[CalibrationCase], indices: Sequence[int], labels: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    names: list[str] = []
    for label in labels:
        names.extend(
            (
                f"phase1__{label}__predicted",
                f"phase1__{label}__confidence",
                f"phase1__{label}__signed_salience",
                f"phase1__{label}__positive",
                f"phase1__{label}__negative",
                f"phase1__{label}__neutral",
                f"phase1__{label}__primary",
                f"phase1__{label}__secondary",
            )
        )
    names.extend(
        (
            "phase1__aggregate__predicted_count",
            "phase1__aggregate__positive_count",
            "phase1__aggregate__negative_count",
            "phase1__aggregate__neutral_count",
            "phase1__aggregate__primary_count",
        )
    )
    rows: list[list[float]] = []
    for index in indices:
        case = cases[index]
        row: list[float] = []
        predicted = positive = negative = neutral = primary = 0
        for label in labels:
            signal = case.raw_signals[label]
            predicted += int(signal.predicted)
            positive += int(signal.direction == "positive")
            negative += int(signal.direction == "negative")
            neutral += int(signal.direction == "neutral")
            primary += int(signal.salience == "primary")
            row.extend(
                (
                    float(signal.predicted),
                    signal.confidence,
                    signal.signed_salience,
                    float(signal.direction == "positive"),
                    float(signal.direction == "negative"),
                    float(signal.direction == "neutral"),
                    float(signal.salience == "primary"),
                    float(signal.salience == "secondary"),
                )
            )
        row.extend((predicted, positive, negative, neutral, primary))
        rows.append(row)
    return np.asarray(rows, dtype=float), tuple(names)


def _prototype_row(
    vector: np.ndarray,
    fit_indices: Sequence[int],
    cases: Sequence[CalibrationCase],
    normalized_embeddings: np.ndarray,
    labels: Sequence[str],
    families: dict[str, str],
    excluded_group: str | None,
) -> list[float]:
    similarities: dict[str, float] = {}
    for label in labels:
        positive = [
            index
            for index in fit_indices
            if cases[index].reference_targets[label]
            and cases[index].episode_slug != excluded_group
        ]
        if not positive:
            similarities[label] = 0.0
            continue
        centroid = normalized_embeddings[positive].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        similarities[label] = float(vector @ (centroid / norm)) if norm else 0.0
    result: list[float] = []
    for label in labels:
        alternatives = [
            similarities[other]
            for other in labels
            if other != label and families[other] == families[label]
        ]
        competing = max(alternatives, default=0.0)
        result.extend((similarities[label], similarities[label] - competing))
    return result


def _prototype_features(
    cases: Sequence[CalibrationCase],
    normalized_embeddings: np.ndarray,
    fit_indices: Sequence[int],
    transform_indices: Sequence[int],
    labels: Sequence[str],
    *,
    exclude_own_group: bool,
) -> np.ndarray:
    families = dict(cases[0].families)
    dimension = normalized_embeddings.shape[1]
    totals: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    group_totals: dict[tuple[str, str], np.ndarray] = {}
    group_counts: dict[tuple[str, str], int] = {}
    for label in labels:
        positive = [index for index in fit_indices if cases[index].reference_targets[label]]
        totals[label] = (
            normalized_embeddings[positive].sum(axis=0)
            if positive
            else np.zeros(dimension, dtype=float)
        )
        counts[label] = len(positive)
        for index in positive:
            key = label, cases[index].episode_slug
            group_totals[key] = group_totals.get(key, np.zeros(dimension, dtype=float)) + normalized_embeddings[index]
            group_counts[key] = group_counts.get(key, 0) + 1

    rows: list[list[float]] = []
    for index in transform_indices:
        group = cases[index].episode_slug if exclude_own_group else None
        similarities: dict[str, float] = {}
        for label in labels:
            key = label, group or ""
            total = totals[label] - group_totals.get(key, 0.0)
            count = counts[label] - group_counts.get(key, 0)
            centroid = total / count if count else np.zeros(dimension, dtype=float)
            norm = float(np.linalg.norm(centroid))
            similarities[label] = (
                float(normalized_embeddings[index] @ (centroid / norm)) if norm else 0.0
            )
        row: list[float] = []
        for label in labels:
            alternatives = [
                similarities[other]
                for other in labels
                if other != label and families[other] == families[label]
            ]
            competing = max(alternatives, default=0.0)
            row.extend((similarities[label], similarities[label] - competing))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def build_fold_features(
    cases: Sequence[CalibrationCase],
    pitch_embeddings: np.ndarray,
    train_indices: Sequence[int],
    evaluate_indices: Sequence[int],
    *,
    include_phase1: bool = True,
    pca_components: int = 32,
) -> FoldFeatures:
    if not cases or not train_indices or not evaluate_indices:
        raise ValueError("fold features require cases, training rows, and evaluation rows")
    labels = cases[0].labels
    if any(case.labels != labels for case in cases):
        raise ValueError("calibration taxonomy order mismatch")
    embeddings = _normalize(np.asarray(pitch_embeddings, dtype=float))
    if embeddings.shape[0] != len(cases):
        raise ValueError("pitch embedding row mismatch")
    train_source = embeddings[list(train_indices)]
    evaluate_source = embeddings[list(evaluate_indices)]
    components = min(pca_components, len(train_indices), embeddings.shape[1])
    if float(np.var(train_source, axis=0).sum()) <= np.finfo(float).eps:
        components = 1
        train_embedding = np.zeros((len(train_indices), 1), dtype=float)
        evaluate_embedding = np.zeros((len(evaluate_indices), 1), dtype=float)
    else:
        pca = PCA(n_components=components, random_state=17)
        train_embedding = pca.fit_transform(train_source)
        evaluate_embedding = pca.transform(evaluate_source)
    feature_names: list[str] = [f"pitch_pca__{index:03d}" for index in range(components)]

    train_prototype = _prototype_features(
        cases, embeddings, train_indices, train_indices, labels, exclude_own_group=True
    )
    evaluate_prototype = _prototype_features(
        cases, embeddings, train_indices, evaluate_indices, labels, exclude_own_group=False
    )
    prototype_names = [
        name
        for label in labels
        for name in (
            f"prototype__{label}__similarity",
            f"prototype__{label}__family_margin",
        )
    ]
    feature_names.extend(prototype_names)

    vcs = tuple(sorted({case.vc_slug for case in cases}))
    train_vc = np.asarray(
        [[float(cases[index].vc_slug == vc) for vc in vcs] for index in train_indices]
    )
    evaluate_vc = np.asarray(
        [[float(cases[index].vc_slug == vc) for vc in vcs] for index in evaluate_indices]
    )
    feature_names.extend(f"vc__{vc}" for vc in vcs)

    train_parts = [train_embedding, train_prototype, train_vc]
    evaluate_parts = [evaluate_embedding, evaluate_prototype, evaluate_vc]
    if include_phase1:
        train_raw, raw_names = _raw_phase1_features(cases, train_indices, labels)
        evaluate_raw, _ = _raw_phase1_features(cases, evaluate_indices, labels)
        train_parts.append(train_raw)
        evaluate_parts.append(evaluate_raw)
        feature_names.extend(raw_names)

    targets = np.asarray(
        [[cases[index].reference_targets[label] for label in labels] for index in train_indices],
        dtype=int,
    )
    return FoldFeatures(
        train=np.column_stack(train_parts),
        evaluate=np.column_stack(evaluate_parts),
        targets=targets,
        feature_names=tuple(feature_names),
    )
