from __future__ import annotations

from dataclasses import replace

import pytest

from vc_clone_graph.personalized_cases import PersonalizedCase
from vc_clone_graph.personalized_ensemble import (
    ComponentFoldScores,
    component_scores_from_checkpoint,
    conditional_ensemble_fold,
)
from vc_clone_graph.personalized_folds import make_episode_folds


def cases() -> list[PersonalizedCase]:
    return [PersonalizedCase(
        vc_slug="alpha", vc_input_slug="alpha-fund", vc_name="Alpha",
        episode_slug=f"{index}-episode", group=f"{index}-episode",
        target=index % 2, pitch_text="pitch", phase1_text="p1", phase2_text="p2",
        phase1_features={}, phase2_features={}, wiki_text="wiki",
        actual_rationale_targets={}, actual_rationale_available=True,
        actual_rationale_source_tier="test", actual_rationale_source_format="test",
        source_hashes={"pitch": f"{index:064x}"},
    ) for index in range(8)]


def components(population, fold, *, identical: bool = False):
    keys = [(population[index].vc_slug, population[index].episode_slug) for index in fold.train_indices]
    held_keys = [(population[index].vc_slug, population[index].episode_slug) for index in fold.test_indices]
    tab = {key: 0.8 if population[index].target else 0.2 for key, index in zip(keys, fold.train_indices)}
    setfit = dict(tab) if identical else {
        key: 0.3 if population[index].target else 0.7
        for key, index in zip(keys, fold.train_indices)
    }
    return (
        ComponentFoldScores("tabpfn", tab, tab, {key: 0.6 for key in held_keys}, {key: 0.7 for key in held_keys}),
        ComponentFoldScores("setfit", setfit, setfit, {key: 0.4 for key in held_keys}, {key: 0.3 for key in held_keys}),
    )


def test_conditional_ensemble_uses_exact_outer_training_oof_rows() -> None:
    population = cases()
    fold = make_episode_folds(population, inner_splits=3, seed=51)[0]
    raw = {(row.vc_slug, row.episode_slug): 0.5 for row in population}

    result = conditional_ensemble_fold(
        population, fold, components(population, fold), raw,
        minimum_disagreement_rate=0.1, require_inner_improvement=False, seed=51,
    )

    assert result.status == "complete"
    assert len(result.predictions) == len(fold.test_indices)
    assert all(fold.held_episode not in row.training_episodes for row in result.predictions)
    assert all(0 <= row.classification_score <= 1 for row in result.predictions)


def test_ensemble_skips_when_components_are_not_complementary() -> None:
    population = cases()
    fold = make_episode_folds(population, inner_splits=3, seed=53)[0]
    raw = {(row.vc_slug, row.episode_slug): 0.5 for row in population}

    result = conditional_ensemble_fold(
        population, fold, components(population, fold, identical=True), raw,
        minimum_disagreement_rate=0.1, require_inner_improvement=False, seed=53,
    )

    assert result.status == "skipped"
    assert "disagreement" in result.reason
    assert result.predictions == ()


def test_ensemble_rejects_held_episode_in_component_training_scores() -> None:
    population = cases()
    fold = make_episode_folds(population, inner_splits=3, seed=55)[0]
    tabpfn, setfit = components(population, fold)
    held_key = (population[fold.test_indices[0]].vc_slug, fold.held_episode)
    leaking = replace(
        tabpfn,
        classification_train_oof={**tabpfn.classification_train_oof, held_key: 0.9},
    )
    raw = {(row.vc_slug, row.episode_slug): 0.5 for row in population}

    with pytest.raises(ValueError, match="outer-training keys"):
        conditional_ensemble_fold(
            population, fold, (leaking, setfit), raw,
            minimum_disagreement_rate=0.1, require_inner_improvement=False, seed=55,
        )


def test_component_checkpoint_loader_preserves_inner_and_held_scores() -> None:
    payload = {
        "status": "complete",
        "predictions": [{
            "vc_slug": "alpha", "episode_slug": "8-held",
            "classification_score": 0.7, "ranking_score": 0.8,
            "classification_inner_oof": [["alpha", "1-train", 0.2]],
            "ranking_inner_oof": [["alpha", "1-train", 0.3]],
        }],
    }

    scores = component_scores_from_checkpoint(payload, family="tabpfn")

    assert scores.classification_train_oof == {("alpha", "1-train"): 0.2}
    assert scores.ranking_held == {("alpha", "8-held"): 0.8}


def test_component_checkpoint_loader_rejects_inconsistent_inner_scores() -> None:
    payload = {
        "status": "complete",
        "predictions": [
            {
                "vc_slug": "alpha", "episode_slug": "8-held",
                "classification_score": 0.7, "ranking_score": 0.8,
                "classification_inner_oof": [["alpha", "1-train", 0.2]],
                "ranking_inner_oof": [["alpha", "1-train", 0.3]],
            },
            {
                "vc_slug": "beta", "episode_slug": "8-held",
                "classification_score": 0.6, "ranking_score": 0.7,
                "classification_inner_oof": [["alpha", "1-train", 0.9]],
                "ranking_inner_oof": [["alpha", "1-train", 0.3]],
            },
        ],
    }

    with pytest.raises(ValueError, match="inconsistent"):
        component_scores_from_checkpoint(payload, family="tabpfn")
