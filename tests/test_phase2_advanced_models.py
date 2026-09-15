from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vc_clone_graph.phase2_advanced_models import (
    constrained_fusion_predictions,
    make_pairwise_examples,
    pairwise_grouped_predictions,
    simplex_weight_grid,
    top_k_rerank_scores,
    top_k_grouped_predictions,
)
from vc_clone_graph.phase2_calibration_evaluation import (
    Phase2CalibrationRecord,
    make_repeated_grouped_folds,
    nested_grouped_predictions,
)


def _records() -> list[Phase2CalibrationRecord]:
    rows: list[Phase2CalibrationRecord] = []
    for episode in range(1, 25):
        for vc_index, vc in enumerate(("alpha", "beta")):
            target = int((episode + vc_index) % 4 == 0)
            raw_likelihood = 0.65 if episode % 3 == 0 else 0.25
            review_priority = 0.85 if target else 0.15
            raw_decision = int(raw_likelihood >= 0.5)
            phase2 = {
                "investment_likelihood": raw_likelihood,
                "review_priority_score": review_priority,
                "decision_confidence": 0.8,
                "signed_decision_confidence": 0.8 if raw_decision else -0.8,
                "decision_in": float(raw_decision),
                f"vc__{vc}": 1.0,
            }
            semantic = {
                **phase2,
                "rationale__founder_execution__signed_confidence": (
                    0.9 if target else -0.4
                ),
                f"vc_rationale__{vc}__founder_execution__signed_confidence": (
                    0.9 if target else -0.4
                ),
            }
            rows.append(
                Phase2CalibrationRecord(
                    vc_slug=vc,
                    vc_name=vc.title(),
                    episode_slug=f"{episode}-example",
                    group=f"{episode}-example",
                    target=target,
                    raw_decision=raw_decision,
                    raw_likelihood=raw_likelihood,
                    phase2_features=phase2,
                    combined_features=phase2,
                    artifact_root=f"runs/{vc}/{episode}-example",
                    phase1_sha256=str(episode).zfill(64),
                    phase2_sha256=str(episode + 1).zfill(64),
                    semantic_features=semantic,
                )
            )
    return rows


def test_simplex_weight_grid_is_nonnegative_and_sums_to_one() -> None:
    grid = simplex_weight_grid(dimensions=4, denominator=4)

    assert len(grid) == 35
    assert (1.0, 0.0, 0.0, 0.0) in grid
    assert (0.25, 0.25, 0.25, 0.25) in grid
    assert all(all(value >= 0 for value in weights) for weights in grid)
    assert all(sum(weights) == pytest.approx(1.0) for weights in grid)


def test_constrained_fusion_is_deterministic_and_never_trains_on_held_group() -> None:
    records = _records()
    folds = make_repeated_grouped_folds(records, repeats=2, splits=3, seed=71)

    first = constrained_fusion_predictions(records, folds=folds, seed=71)
    second = constrained_fusion_predictions(records, folds=folds, seed=71)

    assert first == second
    assert len(first) == len(records)
    assert all(0.0 <= row.score <= 1.0 for row in first)
    assert all(
        record.group not in training
        for record, prediction in zip(records, first, strict=True)
        for training in prediction.outer_training_groups
    )
    assert all(
        sum(weights) == pytest.approx(1.0)
        for prediction in first
        for weights in prediction.selected_weights
    )


def test_pairwise_examples_are_symmetric_and_within_vc_only() -> None:
    matrix = np.asarray([[2.0, 1.0], [0.0, 0.5], [4.0, 2.0], [3.0, 1.0]])
    targets = np.asarray([1, 0, 1, 0])
    vcs = np.asarray(["alpha", "alpha", "beta", "beta"], dtype=object)

    x_pairs, y_pairs = make_pairwise_examples(matrix, targets, vcs)

    assert x_pairs.shape == (4, 2)
    assert y_pairs.tolist() == [1, 0, 1, 0]
    assert np.allclose(x_pairs[0], -x_pairs[1])
    assert np.allclose(x_pairs[2], -x_pairs[3])


def test_hierarchical_semantic_classifier_uses_grouped_held_out_predictions() -> None:
    records = _records()
    folds = make_repeated_grouped_folds(records, repeats=2, splits=3, seed=73)

    predictions = nested_grouped_predictions(
        records,
        "semantic",
        folds=folds,
        inner_splits=3,
        seed=73,
    )

    assert len(predictions) == len(records)
    assert all(prediction.method == "semantic" for prediction in predictions)
    assert all(
        record.group not in training
        for record, prediction in zip(records, predictions, strict=True)
        for training in prediction.outer_training_groups
    )


def test_pairwise_ranker_is_deterministic_and_leakage_safe() -> None:
    records = _records()
    folds = make_repeated_grouped_folds(records, repeats=2, splits=3, seed=79)

    first = pairwise_grouped_predictions(records, folds=folds, inner_splits=3, seed=79)
    second = pairwise_grouped_predictions(records, folds=folds, inner_splits=3, seed=79)

    assert first == second
    assert len(first) == len(records)
    assert all(0.0 <= prediction.score <= 1.0 for prediction in first)
    assert all(
        record.group not in training
        for record, prediction in zip(records, first, strict=True)
        for training in prediction.outer_training_groups
    )


def test_top_k_reranker_preserves_non_candidate_raw_order() -> None:
    records = [replace(row, vc_slug="alpha", vc_name="Alpha") for row in _records()[:8]]
    learned = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])

    scores = top_k_rerank_scores(records, learned, candidate_depth=3)

    raw_order = sorted(range(len(records)), key=lambda index: records[index].raw_likelihood, reverse=True)
    candidates = set(raw_order[:3])
    non_candidates = [index for index in raw_order if index not in candidates]
    reranked_non_candidates = sorted(non_candidates, key=lambda index: scores[index], reverse=True)
    assert reranked_non_candidates == non_candidates
    assert min(scores[index] for index in candidates) > max(
        scores[index] for index in non_candidates
    )


def test_top_k_grouped_reranker_selects_depth_without_held_groups() -> None:
    records = _records()
    folds = make_repeated_grouped_folds(records, repeats=2, splits=3, seed=83)

    predictions = top_k_grouped_predictions(
        records,
        folds=folds,
        inner_splits=3,
        candidate_depths=(3, 5, 10),
        seed=83,
    )

    assert len(predictions) == len(records)
    assert all(
        any(part.startswith("K=") for part in prediction.selected_parameters)
        for prediction in predictions
    )
    assert all(
        record.group not in training
        for record, prediction in zip(records, predictions, strict=True)
        for training in prediction.outer_training_groups
    )
