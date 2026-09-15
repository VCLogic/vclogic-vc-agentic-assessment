"""Leakage-safe post-hoc Phase 2 fusion and ranking models."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from .phase2_calibration_evaluation import FoldMaps, Phase2CalibrationRecord


@dataclass(frozen=True)
class AdvancedPrediction:
    vc_slug: str
    episode_slug: str
    group: str
    target: int
    method: str
    score: float
    predicted: int
    outer_repeats: int
    thresholds: tuple[float, ...]
    outer_training_groups: tuple[tuple[str, ...], ...]
    selected_weights: tuple[tuple[float, ...], ...] = ()
    selected_parameters: tuple[str, ...] = ()


def simplex_weight_grid(
    *, dimensions: int, denominator: int
) -> tuple[tuple[float, ...], ...]:
    """Return deterministic non-negative simplex weights on an integer grid."""
    if dimensions < 1 or denominator < 1:
        raise ValueError("simplex dimensions and denominator must be positive")
    rows = []
    for values in product(range(denominator + 1), repeat=dimensions):
        if sum(values) == denominator:
            rows.append(tuple(value / denominator for value in values))
    return tuple(rows)


def _fusion_components(record: Phase2CalibrationRecord) -> np.ndarray:
    features = record.phase2_features
    signed_confidence = float(features.get("signed_decision_confidence", 0.0))
    return np.asarray(
        [
            float(features.get("investment_likelihood", record.raw_likelihood)),
            float(features.get("review_priority_score", 0.0)),
            float(np.clip((signed_confidence + 1.0) / 2.0, 0.0, 1.0)),
            float(features.get("decision_in", record.raw_decision)),
        ],
        dtype=float,
    )


def _macro_ap(
    records: Sequence[Phase2CalibrationRecord], scores: np.ndarray
) -> float:
    values = []
    for vc_slug in sorted({record.vc_slug for record in records}):
        indices = [index for index, record in enumerate(records) if record.vc_slug == vc_slug]
        targets = np.asarray([records[index].target for index in indices], dtype=int)
        if len(set(targets)) == 2:
            values.append(average_precision_score(targets, scores[indices]))
    return float(np.mean(values)) if values else 0.0


def _macro_balanced_accuracy(
    records: Sequence[Phase2CalibrationRecord], decisions: np.ndarray
) -> float:
    values = []
    for vc_slug in sorted({record.vc_slug for record in records}):
        indices = [index for index, record in enumerate(records) if record.vc_slug == vc_slug]
        targets = np.asarray([records[index].target for index in indices], dtype=int)
        if len(set(targets)) == 2:
            values.append(balanced_accuracy_score(targets, decisions[indices]))
    return float(np.mean(values)) if values else 0.0


def _select_threshold(
    records: Sequence[Phase2CalibrationRecord], scores: np.ndarray
) -> float:
    candidates = sorted({0.0, 0.5, 1.0, *(float(value) for value in scores)})
    return max(
        candidates,
        key=lambda threshold: (
            _macro_balanced_accuracy(records, scores >= threshold),
            threshold,
        ),
    )


def constrained_fusion_predictions(
    records: Sequence[Phase2CalibrationRecord],
    *,
    folds: FoldMaps,
    seed: int = 20260816,
    denominator: int = 4,
) -> list[AdvancedPrediction]:
    """Cross-fit a non-negative convex fusion of native Phase 2 scores."""
    del seed  # The complete deterministic grid needs no randomized fitting.
    components = np.vstack([_fusion_components(record) for record in records])
    weight_grid = simplex_weight_grid(dimensions=components.shape[1], denominator=denominator)
    score_values: list[list[float]] = [[] for _ in records]
    decisions: list[list[int]] = [[] for _ in records]
    thresholds: list[list[float]] = [[] for _ in records]
    weights: list[list[tuple[float, ...]]] = [[] for _ in records]
    training_groups: list[list[tuple[str, ...]]] = [[] for _ in records]
    for repetition in folds:
        for train_tuple, held_tuple in repetition:
            train_indices = np.asarray(train_tuple, dtype=int)
            held_indices = np.asarray(held_tuple, dtype=int)
            train_records = [records[int(index)] for index in train_indices]
            candidate_rows = []
            for candidate in weight_grid:
                candidate_array = np.asarray(candidate, dtype=float)
                train_scores = components[train_indices] @ candidate_array
                candidate_rows.append(
                    (
                        _macro_ap(train_records, train_scores),
                        candidate,
                        train_scores,
                    )
                )
            _, selected, train_scores = max(
                candidate_rows,
                key=lambda row: (row[0], row[1]),
            )
            threshold = _select_threshold(train_records, train_scores)
            held_scores = components[held_indices] @ np.asarray(selected, dtype=float)
            train_group_values = tuple(
                sorted({records[int(index)].group for index in train_indices})
            )
            for offset, held_index in enumerate(held_indices):
                index = int(held_index)
                score = float(held_scores[offset])
                score_values[index].append(score)
                decisions[index].append(int(score >= threshold))
                thresholds[index].append(float(threshold))
                weights[index].append(tuple(float(value) for value in selected))
                training_groups[index].append(train_group_values)
    repeat_count = len(folds)
    return [
        AdvancedPrediction(
            vc_slug=record.vc_slug,
            episode_slug=record.episode_slug,
            group=record.group,
            target=record.target,
            method="constrained_phase2_fusion",
            score=float(np.mean(score_values[index])),
            predicted=int(float(np.mean(decisions[index])) >= 0.5),
            outer_repeats=repeat_count,
            thresholds=tuple(thresholds[index]),
            outer_training_groups=tuple(training_groups[index]),
            selected_weights=tuple(weights[index]),
        )
        for index, record in enumerate(records)
    ]


def make_pairwise_examples(
    matrix: np.ndarray,
    targets: np.ndarray,
    vcs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create symmetric In-minus-Out examples within each investor."""
    examples: list[np.ndarray] = []
    labels: list[int] = []
    for vc_slug in sorted(set(str(value) for value in vcs)):
        positives = np.flatnonzero((vcs == vc_slug) & (targets == 1))
        negatives = np.flatnonzero((vcs == vc_slug) & (targets == 0))
        for positive in positives:
            for negative in negatives:
                difference = matrix[int(positive)] - matrix[int(negative)]
                examples.extend((difference, -difference))
                labels.extend((1, 0))
    if not examples:
        raise ValueError("pairwise ranking needs at least one In-Out pair within a VC")
    return np.asarray(examples, dtype=float), np.asarray(labels, dtype=int)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_pairwise_scores(
    train_records: Sequence[Phase2CalibrationRecord],
    held_records: Sequence[Phase2CalibrationRecord],
    *,
    c_value: float,
    seed: int,
) -> np.ndarray:
    vectorizer = DictVectorizer(sparse=False)
    train_matrix = vectorizer.fit_transform(
        [dict(record.semantic_features) for record in train_records]
    )
    held_matrix = vectorizer.transform(
        [dict(record.semantic_features) for record in held_records]
    )
    pair_matrix, pair_targets = make_pairwise_examples(
        train_matrix,
        np.asarray([record.target for record in train_records], dtype=int),
        np.asarray([record.vc_slug for record in train_records], dtype=object),
    )
    estimator = LogisticRegression(
        C=c_value,
        fit_intercept=False,
        max_iter=20_000,
        random_state=seed,
        solver="liblinear",
    )
    estimator.fit(pair_matrix, pair_targets)
    return _sigmoid(estimator.decision_function(held_matrix))


def _select_pairwise_c(
    records: Sequence[Phase2CalibrationRecord],
    *,
    inner_splits: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    targets = np.asarray([record.target for record in records], dtype=int)
    groups = np.asarray([record.group for record in records], dtype=object)
    split_count = min(
        inner_splits,
        len(set(groups)),
        int(targets.sum()),
        int(len(targets) - targets.sum()),
    )
    if split_count < 2:
        raise ValueError("pairwise inner selection needs both classes and two groups")
    splitter = StratifiedGroupKFold(
        n_splits=split_count,
        shuffle=True,
        random_state=seed,
    )
    dummy = np.zeros((len(records), 1), dtype=float)
    folds = tuple(splitter.split(dummy, targets, groups))
    candidates = (0.01, 0.1, 1.0, 10.0)
    best_c = candidates[0]
    best_ap = float("-inf")
    best_scores = np.zeros(len(records), dtype=float)
    for c_value in candidates:
        scores = np.zeros(len(records), dtype=float)
        valid = True
        for train_indices, held_indices in folds:
            try:
                scores[held_indices] = _fit_pairwise_scores(
                    [records[int(index)] for index in train_indices],
                    [records[int(index)] for index in held_indices],
                    c_value=c_value,
                    seed=seed,
                )
            except ValueError:
                valid = False
                break
        if not valid:
            continue
        value = _macro_ap(records, scores)
        if (value, -c_value) > (best_ap, -best_c):
            best_ap = value
            best_c = c_value
            best_scores = scores
    if not np.isfinite(best_ap):
        raise ValueError("no valid pairwise inner model")
    return best_c, best_scores


def pairwise_grouped_predictions(
    records: Sequence[Phase2CalibrationRecord],
    *,
    folds: FoldMaps,
    inner_splits: int = 3,
    seed: int = 20260816,
) -> list[AdvancedPrediction]:
    """Produce nested grouped scores from a within-VC pairwise ranker."""
    score_values: list[list[float]] = [[] for _ in records]
    decisions: list[list[int]] = [[] for _ in records]
    thresholds: list[list[float]] = [[] for _ in records]
    parameters: list[list[str]] = [[] for _ in records]
    training_groups: list[list[tuple[str, ...]]] = [[] for _ in records]
    for repeat_index, repetition in enumerate(folds):
        for train_tuple, held_tuple in repetition:
            train_indices = np.asarray(train_tuple, dtype=int)
            held_indices = np.asarray(held_tuple, dtype=int)
            train_records = [records[int(index)] for index in train_indices]
            held_records = [records[int(index)] for index in held_indices]
            chosen_c, inner_scores = _select_pairwise_c(
                train_records,
                inner_splits=inner_splits,
                seed=seed + repeat_index,
            )
            threshold = _select_threshold(train_records, inner_scores)
            held_scores = _fit_pairwise_scores(
                train_records,
                held_records,
                c_value=chosen_c,
                seed=seed + repeat_index,
            )
            train_group_values = tuple(sorted({record.group for record in train_records}))
            for offset, held_index in enumerate(held_indices):
                index = int(held_index)
                score = float(held_scores[offset])
                score_values[index].append(score)
                decisions[index].append(int(score >= threshold))
                thresholds[index].append(float(threshold))
                parameters[index].append(f"C={chosen_c:g}")
                training_groups[index].append(train_group_values)
    repeat_count = len(folds)
    return [
        AdvancedPrediction(
            vc_slug=record.vc_slug,
            episode_slug=record.episode_slug,
            group=record.group,
            target=record.target,
            method="within_vc_pairwise_ranker",
            score=float(np.mean(score_values[index])),
            predicted=int(float(np.mean(decisions[index])) >= 0.5),
            outer_repeats=repeat_count,
            thresholds=tuple(thresholds[index]),
            outer_training_groups=tuple(training_groups[index]),
            selected_parameters=tuple(parameters[index]),
        )
        for index, record in enumerate(records)
    ]


def top_k_rerank_scores(
    records: Sequence[Phase2CalibrationRecord],
    learned_scores: np.ndarray,
    *,
    candidate_depth: int,
) -> np.ndarray:
    """Rerank raw top-K candidates while preserving the remaining raw order."""
    if len(records) != len(learned_scores) or candidate_depth < 1:
        raise ValueError("top-K reranking requires aligned records and positive depth")
    scores = np.zeros(len(records), dtype=float)
    for vc_slug in sorted({record.vc_slug for record in records}):
        indices = [index for index, record in enumerate(records) if record.vc_slug == vc_slug]
        raw_order = sorted(
            indices,
            key=lambda index: records[index].raw_likelihood,
            reverse=True,
        )
        depth = min(candidate_depth, len(raw_order))
        candidates = raw_order[:depth]
        remaining = raw_order[depth:]
        learned_order = sorted(
            candidates,
            key=lambda index: (float(learned_scores[index]), records[index].episode_slug),
            reverse=True,
        )
        for rank, index in enumerate(learned_order):
            scores[index] = 1.0 - 0.5 * rank / max(1, depth)
        for rank, index in enumerate(remaining):
            scores[index] = 0.49 - 0.49 * rank / max(1, len(remaining) + 1)
    return scores


def _apply_training_top_k_cutoff(
    train_records: Sequence[Phase2CalibrationRecord],
    held_records: Sequence[Phase2CalibrationRecord],
    learned_scores: np.ndarray,
    *,
    candidate_depth: int,
) -> np.ndarray:
    """Apply a training-derived raw-score candidate cutoff to held records."""
    result = np.zeros(len(held_records), dtype=float)
    for vc_slug in sorted({record.vc_slug for record in held_records}):
        train_raw = sorted(
            (
                record.raw_likelihood
                for record in train_records
                if record.vc_slug == vc_slug
            ),
            reverse=True,
        )
        if not train_raw:
            raise ValueError(f"no training records for held VC {vc_slug}")
        cutoff = train_raw[min(candidate_depth, len(train_raw)) - 1]
        for index, record in enumerate(held_records):
            if record.vc_slug != vc_slug:
                continue
            result[index] = (
                1.0 + float(learned_scores[index])
                if record.raw_likelihood >= cutoff
                else record.raw_likelihood
            )
    return result


def top_k_grouped_predictions(
    records: Sequence[Phase2CalibrationRecord],
    *,
    folds: FoldMaps,
    inner_splits: int = 3,
    candidate_depths: Sequence[int] = (5, 10, 20),
    seed: int = 20260816,
) -> list[AdvancedPrediction]:
    """Cross-fit a training-selected raw-candidate/pairwise reranker."""
    if not candidate_depths or any(depth < 1 for depth in candidate_depths):
        raise ValueError("candidate depths must be positive")
    score_values: list[list[float]] = [[] for _ in records]
    decisions: list[list[int]] = [[] for _ in records]
    thresholds: list[list[float]] = [[] for _ in records]
    parameters: list[list[str]] = [[] for _ in records]
    training_groups: list[list[tuple[str, ...]]] = [[] for _ in records]
    for repeat_index, repetition in enumerate(folds):
        for train_tuple, held_tuple in repetition:
            train_indices = np.asarray(train_tuple, dtype=int)
            held_indices = np.asarray(held_tuple, dtype=int)
            train_records = [records[int(index)] for index in train_indices]
            held_records = [records[int(index)] for index in held_indices]
            chosen_c, inner_pairwise = _select_pairwise_c(
                train_records,
                inner_splits=inner_splits,
                seed=seed + repeat_index,
            )
            depth_rows = []
            for depth in sorted(set(int(value) for value in candidate_depths)):
                reranked = top_k_rerank_scores(
                    train_records,
                    inner_pairwise,
                    candidate_depth=depth,
                )
                depth_rows.append((_macro_ap(train_records, reranked), -depth, depth, reranked))
            _, _, selected_depth, train_reranked = max(depth_rows)
            threshold = _select_threshold(train_records, train_reranked)
            held_pairwise = _fit_pairwise_scores(
                train_records,
                held_records,
                c_value=chosen_c,
                seed=seed + repeat_index,
            )
            held_scores = _apply_training_top_k_cutoff(
                train_records,
                held_records,
                held_pairwise,
                candidate_depth=selected_depth,
            )
            train_group_values = tuple(sorted({record.group for record in train_records}))
            for offset, held_index in enumerate(held_indices):
                index = int(held_index)
                score = float(held_scores[offset])
                score_values[index].append(score)
                decisions[index].append(int(score >= threshold))
                thresholds[index].append(float(threshold))
                parameters[index].extend((f"C={chosen_c:g}", f"K={selected_depth}"))
                training_groups[index].append(train_group_values)
    repeat_count = len(folds)
    return [
        AdvancedPrediction(
            vc_slug=record.vc_slug,
            episode_slug=record.episode_slug,
            group=record.group,
            target=record.target,
            method="training_selected_top_k_reranker",
            score=float(np.mean(score_values[index])),
            predicted=int(float(np.mean(decisions[index])) >= 0.5),
            outer_repeats=repeat_count,
            thresholds=tuple(thresholds[index]),
            outer_training_groups=tuple(training_groups[index]),
            selected_parameters=tuple(parameters[index]),
        )
        for index, record in enumerate(records)
    ]
