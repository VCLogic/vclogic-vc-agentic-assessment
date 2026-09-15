from __future__ import annotations

import numpy as np

from vc_clone_graph.partial_pooling_evaluation import (
    fit_partial_pooling_explanatory_model,
    nested_partial_pooling_predictions,
    nested_per_vc_compact_predictions,
    partial_pooling_view,
    select_threshold,
)
from vc_clone_graph.phase2_calibration_evaluation import Phase2CalibrationRecord


LABELS = ("founder_execution", *(f"label_{index}" for index in range(43)))


def model_records() -> list[Phase2CalibrationRecord]:
    records: list[Phase2CalibrationRecord] = []
    for vc_offset, vc_slug in enumerate(("alpha", "beta")):
        for episode in range(1, 13):
            target = int((episode + vc_offset) % 3 == 0)
            raw = int(episode % 4 == 0)
            signed = 0.85 if target else -0.55
            semantic = {
                "rationale__founder_execution__signed_confidence": signed,
                "rationale__founder_execution__salience__primary": 1.0,
                f"rationale__founder_execution__direction__{'positive' if target else 'negative'}": 1.0,
                "rationale__founder_execution__pitch_evidence_count": 1.0,
            }
            phase2 = {
                "investment_likelihood": 0.7 if raw else 0.25,
                "review_priority_score": 0.8 if target else 0.2,
                "decision_confidence": 0.75,
                "signed_decision_confidence": 0.75 if raw else -0.75,
                "decision_in": float(raw),
                f"vc__{vc_slug}": 1.0,
            }
            records.append(
                Phase2CalibrationRecord(
                    vc_slug=vc_slug,
                    vc_name=vc_slug.title(),
                    episode_slug=f"{episode}-example",
                    group=f"{episode}-example",
                    target=target,
                    raw_decision=raw,
                    raw_likelihood=phase2["investment_likelihood"],
                    phase2_features=phase2,
                    combined_features={**semantic, **phase2},
                    semantic_features={**semantic, **phase2},
                    artifact_root=f"runs/{vc_slug}/{episode}-example",
                    phase1_sha256=str(episode).zfill(64),
                    phase2_sha256=str(episode + 1).zfill(64),
                )
            )
    return records


def test_threshold_selection_supports_balanced_and_precision_objectives() -> None:
    targets = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.6, 0.55, 0.9])

    balanced = select_threshold(targets, scores, "balanced_accuracy")
    precision = select_threshold(targets, scores, "f0_5")

    assert balanced in set(scores) | {0.0, 0.5, 1.0}
    assert precision in set(scores) | {0.0, 0.5, 1.0}
    assert precision >= balanced


def test_per_vc_compact_predictions_hold_one_pitch_and_train_only_same_vc() -> None:
    records = model_records()
    predictions = nested_per_vc_compact_predictions(
        records, LABELS, family="decision", inner_splits=2, seed=17
    )

    assert len(predictions) == len(records)
    for record, prediction in zip(records, predictions, strict=True):
        assert record.episode_slug not in prediction.training_episode_slugs
        assert set(prediction.training_vcs) == {record.vc_slug}
        assert 0 <= prediction.score <= 1
        assert prediction.balanced_decision in {0, 1}
        assert prediction.precision_decision in {0, 1}
        assert prediction.method == "per_vc_compact_decision"


def test_per_vc_compact_predictions_are_deterministic_for_both_families() -> None:
    records = model_records()
    for family in ("decision", "rationale"):
        first = nested_per_vc_compact_predictions(
            records, LABELS, family=family, inner_splits=2, seed=19
        )
        second = nested_per_vc_compact_predictions(
            records, LABELS, family=family, inner_splits=2, seed=19
        )
        assert first == second
        if family == "rationale":
            assert all(prediction.method == "per_vc_compact_rationale" for prediction in first)


def test_partial_pooling_map_contains_global_and_matching_vc_deviation() -> None:
    item = model_records()[0]

    rationale = partial_pooling_view(
        item, LABELS, family="rationale", deviation_scale=0.5
    )
    decision = partial_pooling_view(
        item, LABELS, family="decision", deviation_scale=0.5
    )

    name = "rationale_core__founder_execution"
    assert rationale[f"global__{name}"] != 0
    assert rationale[f"deviation__{item.vc_slug}__{name}"] == 0.5 * rationale[f"global__{name}"]
    assert rationale[f"deviation__{item.vc_slug}__intercept"] == 0.5
    assert not any("phase2__" in feature for feature in rationale)
    assert any("phase2__investment_likelihood" in feature for feature in decision)


def test_hierarchical_predictions_hold_shared_episode_for_every_vc() -> None:
    records = model_records()
    predictions = nested_partial_pooling_predictions(
        records,
        LABELS,
        family="decision",
        objective="average_precision",
        inner_splits=2,
        seed=23,
    )

    assert len(predictions) == len(records)
    for record, prediction in zip(records, predictions, strict=True):
        assert prediction.held_group == record.group
        assert prediction.held_group not in prediction.training_groups
        assert prediction.held_group not in {
            records[index].group for index in prediction.training_indices
        }
        assert prediction.deviation_scale in {0.0, 0.25, 0.5, 1.0}
        assert set(prediction.training_vcs) == {"alpha", "beta"}
        assert prediction.method == "partial_pooling_decision_ranking"


def test_parallel_outer_folds_match_serial_predictions() -> None:
    records = model_records()
    serial = nested_partial_pooling_predictions(
        records,
        LABELS,
        family="rationale",
        objective="average_precision",
        inner_splits=2,
        seed=29,
        n_jobs=1,
    )
    parallel = nested_partial_pooling_predictions(
        records,
        LABELS,
        family="rationale",
        objective="average_precision",
        inner_splits=2,
        seed=29,
        n_jobs=2,
    )

    assert parallel == serial


def test_hierarchical_classification_and_ranking_are_deterministic() -> None:
    records = model_records()
    for objective in ("balanced_accuracy", "average_precision"):
        first = nested_partial_pooling_predictions(
            records, LABELS, family="rationale", objective=objective,
            inner_splits=2, seed=29,
        )
        second = nested_partial_pooling_predictions(
            records, LABELS, family="rationale", objective=objective,
            inner_splits=2, seed=29,
        )
        assert first == second


def test_full_partial_pooling_fit_exports_global_and_vc_coefficients() -> None:
    result = fit_partial_pooling_explanatory_model(
        model_records(), LABELS, family="decision", inner_splits=2, seed=31
    )

    assert result["scientific_status"] == "full_data_explanatory_not_performance_estimate"
    assert result["selected_c"] in {0.1, 1.0, 10.0}
    assert result["deviation_scale"] in {0.0, 0.25, 0.5, 1.0}
    rows = result["coefficient_rows"]
    assert any(row["effect_scope"] == "global" for row in rows)
    assert any(row["effect_scope"] == "vc_deviation" for row in rows)
    assert all("odds_ratio" in row for row in rows)
