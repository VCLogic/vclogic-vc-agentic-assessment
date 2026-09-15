"""Cross-fitted auxiliary teacher for transcript-observed VC rationales."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from .personalized_cases import PersonalizedCase
from .personalized_folds import EpisodeFold


@dataclass(frozen=True)
class RationaleTeacherOutput:
    probabilities: Mapping[str, float]
    training_episodes: tuple[str, ...]
    status_by_label: Mapping[str, str]


def _safe_rows(rows: Sequence[Mapping[str, float]]) -> list[dict[str, float]]:
    result = []
    for row in rows:
        unsafe = [
            name for name in row
            if "actual" in name.lower() or "target" in name.lower()
        ]
        if unsafe:
            raise ValueError(f"teacher deployable rows contain forbidden fields: {unsafe}")
        values = {name: float(value) for name, value in row.items()}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("teacher deployable rows contain non-finite values")
        result.append(values)
    return result


def _label_predictions(
    cases: Sequence[PersonalizedCase],
    rows: Sequence[Mapping[str, float]],
    fit_indices: Sequence[int],
    predict_indices: Sequence[int],
    label: str,
    *,
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, str]:
    supervised = [index for index in fit_indices if cases[index].actual_rationale_available]
    y = np.asarray([
        int(cases[index].actual_rationale_targets.get(label, 0))
        for index in supervised
    ], dtype=int)
    prevalence = float((int(y.sum()) + 1) / (len(y) + 2))
    if len(set(y.tolist())) < 2:
        return np.full(len(predict_indices), prevalence, dtype=float), "untrainable_prevalence"
    vectorizer = DictVectorizer(sparse=False)
    x_fit = vectorizer.fit_transform([rows[index] for index in supervised])
    x_predict = vectorizer.transform([rows[index] for index in predict_indices])
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=500,
        random_state=seed,
        solver="liblinear",
    ).fit(x_fit, y)
    return np.asarray(model.predict_proba(x_predict)[:, 1], dtype=float), "trained"


def cross_fit_rationale_teacher(
    cases: Sequence[PersonalizedCase],
    fold: EpisodeFold,
    deployable_rows: Sequence[Mapping[str, float]],
    taxonomy_labels: Sequence[str],
    *,
    c_value: float,
    seed: int,
) -> tuple[list[RationaleTeacherOutput], list[RationaleTeacherOutput]]:
    """Return inner-OOF outer-training outputs and outer-fit held outputs."""
    if len(cases) != len(deployable_rows) or c_value <= 0:
        raise ValueError("teacher inputs are misaligned or invalid")
    rows = _safe_rows(deployable_rows)
    labels = tuple(taxonomy_labels)
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("teacher taxonomy labels must be unique and non-empty")

    train_probabilities: dict[int, dict[str, float]] = {
        index: {} for index in fold.train_indices
    }
    train_statuses: dict[int, dict[str, str]] = {
        index: {} for index in fold.train_indices
    }
    train_provenance: dict[int, tuple[str, ...]] = {}
    for split_index, (inner_train, inner_valid) in enumerate(fold.inner_splits):
        episodes = tuple(sorted({cases[index].episode_slug for index in inner_train}))
        if fold.held_episode in episodes:
            raise ValueError("held episode entered rationale-teacher inner training")
        for index in inner_valid:
            train_provenance[index] = episodes
        for label_index, label in enumerate(labels):
            probabilities, status = _label_predictions(
                cases, rows, inner_train, inner_valid, label,
                c_value=c_value, seed=seed + split_index * 101 + label_index,
            )
            for index, probability in zip(inner_valid, probabilities, strict=True):
                train_probabilities[index][label] = float(probability)
                train_statuses[index][label] = status
    if set(train_probabilities) != set(train_provenance) or any(
        set(values) != set(labels) for values in train_probabilities.values()
    ):
        raise ValueError("inner folds did not cross-fit every outer-training row")

    held_probabilities = {index: {} for index in fold.test_indices}
    held_statuses = {index: {} for index in fold.test_indices}
    held_training_episodes = tuple(sorted({
        cases[index].episode_slug for index in fold.train_indices
    }))
    if fold.held_episode in held_training_episodes:
        raise ValueError("held episode entered rationale-teacher outer training")
    for label_index, label in enumerate(labels):
        probabilities, status = _label_predictions(
            cases, rows, fold.train_indices, fold.test_indices, label,
            c_value=c_value, seed=seed + 10000 + label_index,
        )
        for index, probability in zip(fold.test_indices, probabilities, strict=True):
            held_probabilities[index][label] = float(probability)
            held_statuses[index][label] = status

    training_outputs = [
        RationaleTeacherOutput(
            probabilities=train_probabilities[index],
            training_episodes=train_provenance[index],
            status_by_label=train_statuses[index],
        )
        for index in fold.train_indices
    ]
    held_outputs = [
        RationaleTeacherOutput(
            probabilities=held_probabilities[index],
            training_episodes=held_training_episodes,
            status_by_label=held_statuses[index],
        )
        for index in fold.test_indices
    ]
    return training_outputs, held_outputs


def teacher_comparison_features(
    output: RationaleTeacherOutput,
    phase1_features: Mapping[str, float],
    taxonomy_labels: Sequence[str],
) -> dict[str, float]:
    """Compare teacher hypotheses with deployable Phase 1 activations."""
    result = {}
    for label in taxonomy_labels:
        probability = float(output.probabilities[label])
        signed = float(phase1_features.get(
            f"label__{label}__signed_confidence", 0.0
        ))
        confidence = float(phase1_features.get(
            f"label__{label}__confidence_sum", abs(signed)
        ))
        activation = float(confidence > 0)
        result.update({
            f"teacher__{label}__probability": probability,
            f"teacher__{label}__agreement": 1.0 - abs(probability - activation),
            f"teacher__{label}__missing_rationale": probability * (1.0 - activation),
            f"teacher__{label}__spurious_rationale": (1.0 - probability) * activation,
            f"teacher__{label}__confidence_weighted_disagreement": (
                abs(probability - activation) * max(confidence, 1.0 - confidence)
            ),
            f"teacher__{label}__untrainable": float(
                output.status_by_label[label] != "trained"
            ),
        })
    return result
