from __future__ import annotations

from vc_clone_graph.structured_fusion_features import StructuredFusionCase
from vc_clone_graph.structured_fusion_models import nested_two_head_predictions


def cases() -> list[StructuredFusionCase]:
    rows = []
    for vc_index, vc in enumerate(("alpha", "beta")):
        for index in range(10):
            target = int(index in ({1, 4, 7} if vc_index == 0 else {2, 5, 8}))
            signal = 0.85 if target else -0.65
            rows.append(StructuredFusionCase(
                vc_slug=vc, vc_name=vc.title(), episode_slug=f"{index}-{vc}",
                group=f"{index}-{vc}", target=target,
                phase1_features={"p1__signal": signal, "p1__noise": float(index % 2)},
                association_features={"assoc__signal": 0.8 if target else 0.2},
                phase2_features={
                    "p2__investment_likelihood": 0.75 if target else 0.25,
                    "p2__review_priority_score": 0.7 if target else 0.3,
                },
                interaction_features={"interaction__signal": signal * 0.5},
                association_training_episode_slugs=tuple(
                    f"{other}-{vc}" for other in range(10) if other != index
                ),
                phase1_sha256="a" * 64, phase2_sha256="b" * 64,
                source_investigation_sha256="c" * 64,
            ))
    return rows


def test_nested_elastic_is_deterministic_and_excludes_every_held_episode() -> None:
    first = nested_two_head_predictions(
        cases(), condition="full", model_family="elastic", seed=41, n_jobs=1,
        elastic_configs=((0.1, 0.5), (1.0, 1.0)),
    )
    second = nested_two_head_predictions(
        cases(), condition="full", model_family="elastic", seed=41, n_jobs=1,
        elastic_configs=((0.1, 0.5), (1.0, 1.0)),
    )

    assert first == second
    assert len(first) == len(cases())
    assert all(row.episode_slug not in row.training_episode_slugs for row in first)
    assert all(
        row.episode_slug not in fold
        for row in first for fold in row.inner_validation_episode_slugs
    )
    assert all(0 <= row.classification_score <= 1 for row in first)
    assert all(0 <= row.ranking_score <= 1 for row in first)
    assert all(row.classification_config for row in first)
    assert all(row.ranking_config for row in first)
    assert all(row.classification_selected_features for row in first)


def test_nested_forest_and_blend_are_bounded_and_leakage_safe() -> None:
    forest = nested_two_head_predictions(
        cases(), condition="full", model_family="forest", seed=43, n_jobs=1,
        forest_configs=((2, 0.1),), forest_trees=12,
    )
    blend = nested_two_head_predictions(
        cases(), condition="full", model_family="blend", seed=43, n_jobs=1,
        elastic_configs=((0.1, 0.5),), forest_configs=((2, 0.1),),
        forest_trees=12, blend_weights=(0.0, 0.5, 1.0),
    )

    assert all("max_depth=2" in row.classification_config for row in forest)
    assert all(row.episode_slug not in row.training_episode_slugs for row in blend)
    assert all("elastic_weight=" in row.classification_config for row in blend)
    assert all("elastic_weight=" in row.ranking_config for row in blend)
