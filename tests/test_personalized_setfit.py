from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="requires the optional setfit/personalized extra")

from dataclasses import replace

import numpy as np
import torch

from vc_clone_graph.personalized_cases import PersonalizedCase
from vc_clone_graph.personalized_folds import make_episode_folds
from vc_clone_graph.personalized_matrices import RawFeatureViews
from vc_clone_graph.personalized_setfit import (
    build_setfit_pairs,
    multitask_loss,
    run_setfit_fold,
    serialize_case,
)


def cases() -> list[PersonalizedCase]:
    rows = []
    for vc in ("alpha", "beta"):
        for index in range(6):
            rows.append(PersonalizedCase(
                vc_slug=vc, vc_input_slug=f"{vc}-fund", vc_name=vc.title(),
                episode_slug=f"{index}-{vc}", group=f"{index}-{vc}",
                target=int(index in {1, 4}),
                pitch_text=f"Founder pitch {index}",
                phase1_text=f"Investigation {index}",
                phase2_text=f"Synthesis {index}",
                phase1_features={}, phase2_features={}, wiki_text=f"{vc} thesis",
                actual_rationale_targets={"market": int(index % 2)},
                actual_rationale_available=True,
                actual_rationale_source_tier="test",
                actual_rationale_source_format="test",
                source_hashes={"pitch": f"{index:064x}"},
            ))
    return rows


def test_pair_generation_is_deterministic_bounded_and_training_only() -> None:
    population = cases()
    train_indices = tuple(range(1, len(population)))

    first = build_setfit_pairs(
        population, train_indices, max_pairs=20, seed=31
    )
    second = build_setfit_pairs(
        population, train_indices, max_pairs=20, seed=31
    )

    assert first == second
    assert len(first.contrastive) <= 20
    assert len(first.ranking) <= 20
    assert all(0 not in (pair.left_index, pair.right_index) for pair in first.contrastive)
    assert all(0 not in (pair.in_index, pair.out_index) for pair in first.ranking)
    assert {pair.similar for pair in first.contrastive} == {0, 1}
    for pair in first.ranking:
        assert population[pair.in_index].vc_slug == population[pair.out_index].vc_slug
        assert population[pair.in_index].target == 1
        assert population[pair.out_index].target == 0


def test_serialized_case_contains_deployable_views_not_actual_reference() -> None:
    case = cases()[0]
    text = serialize_case(case)

    assert "[INVESTOR] Alpha" in text
    assert "[PROFILE] alpha thesis" in text
    assert "[PITCH] Founder pitch" in text
    assert "[PHASE1] Investigation" in text
    assert "[PHASE2] Synthesis" in text
    assert "actual_rationale_targets" not in text
    assert "source_tier" not in text


def test_multitask_loss_is_weighted_sum_of_three_objectives() -> None:
    result = multitask_loss(
        decision_logits=torch.tensor([1.0, -0.5, 0.25]),
        decisions=torch.tensor([1.0, 0.0, 1.0]),
        ranking_pairs=((0, 1),),
        rationale_logits=torch.tensor([[0.5, -0.2], [0.1, 0.3], [-0.4, 0.8]]),
        rationale_targets=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        rationale_mask=torch.tensor([True, True, False]),
        lambda_rank=0.5,
        lambda_aux=0.25,
        positive_weight=1.5,
    )

    expected = result.classification + 0.5 * result.ranking + 0.25 * result.auxiliary
    assert torch.allclose(result.total, expected)
    assert result.ranking.item() > 0
    assert result.auxiliary.item() > 0


class TinyEncoder:
    def adapt(self, population, pairs, *, epochs: int, seed: int, condition: str) -> None:
        del population, pairs, epochs, seed, condition

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([
            [float("Alpha" in text), float("Beta" in text), len(text) / 100.0]
            for text in texts
        ])


def views(population: list[PersonalizedCase]) -> list[RawFeatureViews]:
    return [RawFeatureViews(
        pitch_embedding=np.asarray([1.0, 0.0]),
        phase1_embedding=np.asarray([0.0, 1.0]),
        phase2_embedding=np.asarray([0.5, 0.5]),
        wiki_embedding=np.asarray([float(row.vc_slug == "alpha"), float(row.vc_slug == "beta")]),
        pitch_structured={"pitch__length": float(index + 1)},
        phase1_structured={"p1__market": float(row.actual_rationale_targets["market"])},
        phase2_structured={"p2__likelihood": 0.6 if row.target else 0.4},
        investor_structured={
            "vc_identity__alpha": float(row.vc_slug == "alpha"),
            "vc_identity__beta": float(row.vc_slug == "beta"),
        },
        teacher_structured={}, interaction_structured={},
    ) for index, row in enumerate(population)]


def test_setfit_fold_is_deterministic_and_held_labels_are_inert() -> None:
    population = cases()
    fold = make_episode_folds(population, inner_splits=2, seed=37)[0]
    source_views = views(population)
    kwargs = dict(
        condition="all_no_teacher",
        epochs=(1,), lambda_rank=(0.25,), lambda_aux=(0.1,),
        max_pairs=32, taxonomy_labels=("market",), seed=37,
        encoder_factory=TinyEncoder,
    )
    first = run_setfit_fold(population, source_views, fold, **kwargs)
    changed = list(population)
    for index in fold.test_indices:
        changed[index] = replace(changed[index], target=1 - changed[index].target)
    second = run_setfit_fold(changed, source_views, fold, **kwargs)

    assert [row.classification_score for row in first] == [
        row.classification_score for row in second
    ]
    assert all(fold.held_episode not in row.training_episodes for row in first)
    assert all(0 <= row.classification_score <= 1 for row in first)
    assert all(row.family == "setfit" for row in first)
    assert all(len(row.classification_inner_oof) == len(fold.train_indices) for row in first)


def test_setfit_reuses_identical_inner_encoder_adaptations() -> None:
    population = cases()
    fold = make_episode_folds(population, inner_splits=2, seed=39)[0]

    class CountingEncoder(TinyEncoder):
        adaptations = 0
        conditions = []

        def adapt(self, population, pairs, *, epochs: int, seed: int, condition: str) -> None:
            type(self).adaptations += 1
            type(self).conditions.append(condition)
            super().adapt(
                population, pairs, epochs=epochs, seed=seed, condition=condition
            )

    run_setfit_fold(
        population, views(population), fold, condition="all_no_teacher",
        epochs=(1,), lambda_rank=(0.25, 0.5), lambda_aux=(0.1, 0.25),
        max_pairs=32, taxonomy_labels=("market",), seed=39,
        encoder_factory=CountingEncoder,
    )

    assert CountingEncoder.adaptations == 4  # two inner splits plus two seeded refits
    assert set(CountingEncoder.conditions) == {"all_no_teacher"}
