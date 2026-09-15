from __future__ import annotations

import csv
from dataclasses import replace
import json
import math

import pytest

from vc_clone_graph.advanced_calibration import (
    bootstrap_intervals,
    candidate_manifest,
    make_repeated_stratified_folds,
    load_baseline_predictions,
    nested_advanced_leave_one_out,
    repeated_stratified_logistic_predictions,
    repeated_stratified_predictions,
    write_advanced_analysis,
)
from vc_clone_graph.calibration import CalibrationRecord


def _records(*, flip_first: bool = False) -> list[CalibrationRecord]:
    records = []
    for index in range(15):
        target = int(index % 4 == 0)
        if flip_first and index == 0:
            target = 1 - target
        legacy = {
            "positive_primary_count": float(index % 5),
            "negative_primary_count": float((index + 1) % 4),
            "primary_signed_confidence_sum": float(index - 7) / 7,
        }
        semantic = {
            **legacy,
            "label__founder_execution__signed_confidence": float(index % 6) / 5,
            "severity__material__count": float(index % 3 == 0),
        }
        phase2 = {
            "any_check_in": float(index % 3 == 0),
            "any_check_likelihood": 0.1 + index * 0.045,
            "standard_check_in": 0.0,
            "standard_check_likelihood": 0.04 + index * 0.015,
            "ranking_score": 0.85 - index * 0.035,
        }
        records.append(
            CalibrationRecord(
                episode_slug=f"{index + 1}-advanced",
                target=target,
                target_decision="In" if target else "Out",
                artifact_root=f"runs/{index + 1}-advanced",
                phase1_sha256=str(index).zfill(64),
                phase2_sha256=str(index + 1).zfill(64),
                features={
                    "legacy_phase1": legacy,
                    "semantic_phase1": semantic,
                    "phase2": phase2,
                    "legacy_plus_phase2": {
                        **{f"p1__{key}": value for key, value in legacy.items()},
                        **{f"p2__{key}": value for key, value in phase2.items()},
                    },
                    "semantic_plus_phase2": {
                        **{f"p1__{key}": value for key, value in semantic.items()},
                        **{f"p2__{key}": value for key, value in phase2.items()},
                    },
                },
            )
        )
    return records


def test_candidate_manifest_declares_every_advanced_method() -> None:
    manifest = candidate_manifest()

    assert set(manifest) == {
        "late_fusion",
        "elastic_net_combined",
        "shallow_gradient_boosting",
        "linear_svm",
        "rbf_svm",
        "pairwise_ranker",
        "two_stage_diamond",
    }
    for name, specification in manifest.items():
        assert specification.name == name
        assert specification.feature_family
        assert specification.objective in {"classification", "ranking", "hybrid"}
        assert specification.parameter_grid
        assert all(configuration for configuration in specification.parameter_grid)


@pytest.mark.parametrize(
    "method",
    [
        "elastic_net_combined",
        "shallow_gradient_boosting",
        "linear_svm",
        "rbf_svm",
        "pairwise_ranker",
    ],
)
def test_nested_advanced_models_exclude_the_held_episode(method: str) -> None:
    records = _records()

    predictions = nested_advanced_leave_one_out(records, method)

    assert len(predictions) == len(records)
    for prediction in predictions:
        assert prediction.episode_slug not in prediction.training_slugs
        assert len(prediction.training_slugs) == len(records) - 1
        assert math.isfinite(prediction.score)
        assert math.isfinite(prediction.threshold)
        assert prediction.selected_config


@pytest.mark.parametrize(
    "method",
    [
        "elastic_net_combined",
        "shallow_gradient_boosting",
        "linear_svm",
        "rbf_svm",
        "pairwise_ranker",
    ],
)
def test_held_out_label_cannot_change_advanced_model_output(method: str) -> None:
    original = nested_advanced_leave_one_out(_records(), method)[0]
    flipped = nested_advanced_leave_one_out(_records(flip_first=True), method)[0]

    assert flipped.score == pytest.approx(original.score)
    assert flipped.threshold == pytest.approx(original.threshold)
    assert flipped.selected_config == original.selected_config


@pytest.mark.parametrize("method", ["late_fusion", "two_stage_diamond"])
def test_cross_fitted_score_compositions_exclude_the_held_episode(method: str) -> None:
    records = _records()

    predictions = nested_advanced_leave_one_out(records, method)

    assert len(predictions) == len(records)
    for prediction in predictions:
        assert prediction.episode_slug not in prediction.training_slugs
        assert len(prediction.training_slugs) == len(records) - 1
        assert prediction.provenance["outer_held_out_excluded"] is True
        assert prediction.provenance["each_meta_row_cross_fitted"] is True
        assert prediction.provenance["crossfit_row_count"] == len(records) - 1
        assert math.isfinite(prediction.score)


@pytest.mark.parametrize("method", ["late_fusion", "two_stage_diamond"])
def test_held_out_label_cannot_change_score_composition(method: str) -> None:
    original = nested_advanced_leave_one_out(_records(), method)[0]
    flipped = nested_advanced_leave_one_out(_records(flip_first=True), method)[0]

    assert flipped.score == pytest.approx(original.score)
    assert flipped.threshold == pytest.approx(original.threshold)
    assert flipped.selected_config == original.selected_config


def test_bootstrap_intervals_are_deterministic_and_bounded() -> None:
    predictions = nested_advanced_leave_one_out(_records(), "linear_svm")

    first = bootstrap_intervals(predictions, iterations=50, seed=7)
    second = bootstrap_intervals(predictions, iterations=50, seed=7)

    assert first == second
    for interval in first.values():
        assert len(interval) == 2
        assert math.isfinite(interval[0])
        assert math.isfinite(interval[1])
        assert interval[0] <= interval[1]


def test_write_advanced_analysis_outputs_reproducibility_artifacts(
    tmp_path,
) -> None:
    output_root = tmp_path / "advanced"

    result = write_advanced_analysis(
        _records(),
        output_root,
        methods=("linear_svm", "semantic_phase1", "raw_any_check"),
        bootstrap_iterations=20,
        fold_maps=make_repeated_stratified_folds(
            _records(), repeats=1, splits=3
        ),
    )

    assert {path.name for path in output_root.iterdir()} == {
        "metrics.json",
        "predictions.csv",
        "rankings.csv",
        "method-manifest.json",
        "report.md",
    }
    assert result["dataset"] == {"episodes": 15, "ins": 4, "outs": 11}
    assert set(result["methods"]) == {
        "linear_svm",
        "semantic_phase1",
        "raw_any_check",
    }


def test_dynamic_logistic_family_uses_fixed_held_out_folds() -> None:
    records = [
        replace(
            record,
            features={
                **record.features,
                "custom_override": {
                    **record.features["semantic_plus_phase2"],
                    "override_signal": float(index % 3 == 0),
                },
            },
        )
        for index, record in enumerate(_records())
    ]
    folds = make_repeated_stratified_folds(records, repeats=2, splits=3)

    predictions = repeated_stratified_logistic_predictions(
        records,
        method="custom_override_logistic",
        feature_family="custom_override",
        fold_maps=folds,
    )

    assert len(predictions) == len(records)
    assert {prediction.method for prediction in predictions} == {
        "custom_override_logistic"
    }
    for prediction in predictions:
        assert prediction.episode_slug not in prediction.training_slugs
        assert prediction.provenance["feature_family"] == "custom_override"
        assert prediction.provenance["outer_repeats"] == 2


def test_load_baseline_predictions_requires_exact_label_join(tmp_path) -> None:
    records = _records()
    path = tmp_path / "baseline.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "method",
                "episode_slug",
                "target",
                "score",
                "predicted",
                "threshold",
                "chosen_c",
                "training_slugs",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "method": "existing_baseline",
                    "episode_slug": record.episode_slug,
                    "target": record.target,
                    "score": 0.25,
                    "predicted": 0,
                    "threshold": 0.5,
                    "chosen_c": 0.1,
                    "training_slugs": json.dumps(
                        [
                            candidate.episode_slug
                            for candidate in records
                            if candidate.episode_slug != record.episode_slug
                        ]
                    ),
                }
            )

    loaded = load_baseline_predictions(path, records)

    assert set(loaded) == {"existing_baseline"}
    assert len(loaded["existing_baseline"]) == len(records)
    assert loaded["existing_baseline"][0].target == records[0].target


def test_repeated_stratified_svm_scores_are_calibrated_and_out_of_sample() -> None:
    records = _records()
    folds = make_repeated_stratified_folds(records, repeats=2, splits=3)

    predictions = repeated_stratified_predictions(
        records, "rbf_svm", fold_maps=folds
    )

    assert len(predictions) == len(records)
    for prediction in predictions:
        assert 0.0 <= prediction.score <= 1.0
        training_folds = prediction.provenance["outer_training_folds"]
        assert len(training_folds) == 2
        assert all(
            prediction.episode_slug not in training_slugs
            for training_slugs in training_folds
        )
        assert prediction.provenance["score_calibration"] == "training_only_platt"


def test_fixed_outer_folds_prevent_held_label_from_changing_its_score() -> None:
    original_records = _records()
    flipped_records = list(original_records)
    flipped_records[-1] = replace(
        flipped_records[-1], target=1, target_decision="In"
    )
    folds = make_repeated_stratified_folds(original_records, repeats=2, splits=3)

    original = repeated_stratified_predictions(
        original_records, "rbf_svm", fold_maps=folds
    )[-1]
    flipped = repeated_stratified_predictions(
        flipped_records, "rbf_svm", fold_maps=folds
    )[-1]

    assert flipped.score == pytest.approx(original.score)
    assert flipped.predicted == original.predicted
    assert flipped.selected_config == original.selected_config


@pytest.mark.parametrize(
    "method", ["late_fusion", "two_stage_diamond", "semantic_phase1", "phase2"]
)
def test_repeated_stratified_protocol_supports_compositions_and_logistic_baselines(
    method: str,
) -> None:
    records = _records()
    folds = make_repeated_stratified_folds(records, repeats=1, splits=3)

    predictions = repeated_stratified_predictions(records, method, fold_maps=folds)

    assert len(predictions) == len(records)
    assert all(0.0 <= prediction.score <= 1.0 for prediction in predictions)
    assert all(
        prediction.episode_slug
        not in prediction.provenance["outer_training_folds"][0]
        for prediction in predictions
    )
