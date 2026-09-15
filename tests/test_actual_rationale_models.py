from __future__ import annotations

from dataclasses import replace

from vc_clone_graph.actual_rationale_cases import RationaleModelCase
from vc_clone_graph.actual_rationale_models import (
    nested_hierarchical_predictions,
    nested_per_vc_predictions,
    run_all_rationale_models,
)


def model_cases() -> list[RationaleModelCase]:
    cases: list[RationaleModelCase] = []
    for vc_offset, vc_slug in enumerate(("alpha", "beta")):
        for episode in range(1, 13):
            target = int((episode + vc_offset) % 3 == 0)
            actual = {"signal": 1.0 if target else -1.0, "actual_marker": float(episode)}
            predicted = {"signal": 0.8 if target else -0.7, "predicted_marker": float(episode)}
            cases.append(RationaleModelCase(
                vc_slug=vc_slug,
                vc_name=vc_slug.title(),
                episode_slug=f"{episode}-example",
                group=f"{episode}-example",
                target=target,
                source_tier="newly_extracted",
                source_format="transcript-observed-rationales-v1",
                actual_features=actual,
                predicted_features=predicted,
                actual_items=(("signal", "positive" if target else "negative", "primary"),),
                predicted_items=(("signal", "positive" if target else "negative", "primary"),),
            ))
    return cases


def test_per_vc_source_conditions_select_training_and_test_maps() -> None:
    cases = model_cases()
    for condition, train_source, test_source in (
        ("actual_to_actual", "actual", "actual"),
        ("actual_to_predicted", "actual", "predicted"),
        ("predicted_to_predicted", "predicted", "predicted"),
    ):
        predictions = nested_per_vc_predictions(
            cases, model_family="logistic", source_condition=condition,
            inner_splits=2, seed=7,
        )
        assert len(predictions) == len(cases)
        assert {item.training_source for item in predictions} == {train_source}
        assert {item.test_source for item in predictions} == {test_source}
        for case, prediction in zip(cases, predictions, strict=True):
            assert case.episode_slug not in prediction.training_episode_slugs
            assert set(prediction.training_vcs) == {case.vc_slug}


def test_hierarchical_folds_hold_episode_across_vcs() -> None:
    cases = model_cases()
    predictions = nested_hierarchical_predictions(
        cases, source_condition="actual_to_predicted", inner_splits=2, seed=11
    )

    assert len(predictions) == len(cases)
    for case, prediction in zip(cases, predictions, strict=True):
        assert prediction.held_group == case.group
        assert case.group not in prediction.training_groups
        assert {cases[index].group for index in prediction.training_indices}.isdisjoint({case.group})
        assert set(prediction.training_vcs) == {"alpha", "beta"}


def test_parallel_predictions_match_serial_predictions() -> None:
    cases = model_cases()
    serial = nested_per_vc_predictions(
        cases, model_family="tree", source_condition="actual_to_predicted",
        inner_splits=2, seed=13, n_jobs=1,
    )
    parallel = nested_per_vc_predictions(
        cases, model_family="tree", source_condition="actual_to_predicted",
        inner_splits=2, seed=13, n_jobs=2,
    )

    assert serial == parallel


def test_all_model_runner_emits_nine_complete_methods() -> None:
    cases = model_cases()
    result = run_all_rationale_models(cases, inner_splits=2, seed=17, n_jobs=1)

    assert len(result) == 9
    assert all(len(predictions) == len(cases) for predictions in result.values())
    assert "per_vc_logistic__actual_to_predicted" in result
    assert "per_vc_tree__actual_to_predicted" in result
    assert "hierarchical_logistic__actual_to_predicted" in result
    assert all(0.0 <= item.score <= 1.0 for predictions in result.values() for item in predictions)
