from __future__ import annotations

from dataclasses import replace

import numpy as np

from vc_clone_graph.personalized_cases import PersonalizedCase
from vc_clone_graph.personalized_folds import make_episode_folds
from vc_clone_graph.personalized_matrices import RawFeatureViews
from vc_clone_graph.personalized_tabpfn import (
    TabPFNCandidate,
    run_tabpfn_fold,
    select_tabpfn_heads,
)


class FakeTabPFN:
    def __init__(self, n_estimators: int) -> None:
        self.n_estimators = n_estimators

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FakeTabPFN":
        self.direction = x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        margin = np.clip(x @ self.direction, -10, 10)
        score = 1.0 / (1.0 + np.exp(-margin))
        return np.column_stack((1.0 - score, score))


def factory(*, n_estimators: int, seed: int, device: str) -> FakeTabPFN:
    del seed, device
    return FakeTabPFN(n_estimators)


def cases_and_views() -> tuple[list[PersonalizedCase], list[RawFeatureViews]]:
    cases = []
    views = []
    for episode_index in range(8):
        for vc_index, vc in enumerate(("alpha", "beta")):
            target = int(vc_index == 0)
            cases.append(PersonalizedCase(
                vc_slug=vc, vc_input_slug=f"{vc}-fund", vc_name=vc.title(),
                episode_slug=f"{episode_index}-shared", group=f"{episode_index}-shared",
                target=target, pitch_text="same pitch", phase1_text="p1", phase2_text="p2",
                phase1_features={}, phase2_features={}, wiki_text="wiki",
                actual_rationale_targets={}, actual_rationale_available=True,
                actual_rationale_source_tier="test", actual_rationale_source_format="test",
                source_hashes={"pitch": f"{episode_index:064x}"},
            ))
            views.append(RawFeatureViews(
                pitch_embedding=np.asarray([1.0, 0.0]),
                phase1_embedding=np.asarray([0.0, 1.0]),
                phase2_embedding=np.asarray([0.5, 0.5]),
                wiki_embedding=np.asarray([1.0 - vc_index, float(vc_index)]),
                pitch_structured={"pitch__length": 10.0 + episode_index},
                phase1_structured={"p1__signal": 0.5},
                phase2_structured={"p2__likelihood": 0.5},
                investor_structured={
                    "vc_identity__alpha": float(vc == "alpha"),
                    "vc_identity__beta": float(vc == "beta"),
                },
                teacher_structured={}, interaction_structured={},
            ))
    return cases, views


def test_classification_and_ranking_candidates_are_selected_independently() -> None:
    targets = np.asarray([1, 1, 1, 0, 0, 0] * 2)
    vcs = np.asarray(["alpha"] * 6 + ["beta"] * 6)
    classification_scores = np.asarray([.9, .8, .4, .7, .3, .2] * 2)
    ranking_scores = np.asarray([
        .3, .25, .2, .15, .1, .05,
        .95, .9, .85, .8, .75, .7,
    ])

    selected = select_tabpfn_heads(
        targets,
        vcs,
        (
            TabPFNCandidate("classification", classification_scores),
            TabPFNCandidate("ranking", ranking_scores),
        ),
    )

    assert selected.classification_config == "classification"
    assert selected.ranking_config == "ranking"
    assert 0.0 <= selected.classification_threshold <= 1.0


def test_tabpfn_fold_excludes_held_labels_and_conditions_on_vc_identity() -> None:
    cases, views = cases_and_views()
    fold = make_episode_folds(cases, inner_splits=3, seed=21)[0]
    first = run_tabpfn_fold(
        cases, views, fold, condition="all_no_teacher",
        n_estimators=(4,), pca_components=(1,), device="cpu", seed=21,
        factory=factory,
    )

    changed = list(cases)
    for index in fold.test_indices:
        changed[index] = replace(changed[index], target=1 - changed[index].target)
    second = run_tabpfn_fold(
        changed, views, fold, condition="all_no_teacher",
        n_estimators=(4,), pca_components=(1,), device="cpu", seed=21,
        factory=factory,
    )

    assert len(first) == 2
    assert [row.classification_score for row in first] == [
        row.classification_score for row in second
    ]
    assert all(fold.held_episode not in row.training_episodes for row in first)
    assert all(0.0 <= row.classification_score <= 1.0 for row in first)
    assert all(0.0 <= row.ranking_score <= 1.0 for row in first)
    by_vc = {row.vc_slug: row.classification_score for row in first}
    assert by_vc["alpha"] != by_vc["beta"]
    assert all(row.feature_schema_sha256 for row in first)
    assert all(len(row.classification_inner_oof) == len(fold.train_indices) for row in first)
    assert all(len(row.ranking_inner_oof) == len(fold.train_indices) for row in first)
