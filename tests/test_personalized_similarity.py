from __future__ import annotations

import numpy as np

from vc_clone_graph.personalized_cases import PersonalizedCase
from vc_clone_graph.personalized_similarity import similarity_features


def case(vc: str, episode: str, target: int) -> PersonalizedCase:
    return PersonalizedCase(
        vc_slug=vc, vc_input_slug=f"{vc}-fund", vc_name=vc.title(),
        episode_slug=episode, group=episode, target=target,
        pitch_text="pitch", phase1_text="p1", phase2_text="p2",
        phase1_features={}, phase2_features={}, wiki_text="wiki",
        actual_rationale_targets={}, actual_rationale_available=True,
        actual_rationale_source_tier="test", actual_rationale_source_format="test",
        source_hashes={"pitch": "a" * 64},
    )


def test_similarity_history_is_vc_specific_and_excludes_query_episode() -> None:
    cases = [
        case("alpha", "1-shared", 1),
        case("beta", "1-shared", 0),
        case("alpha", "2-in", 1),
        case("alpha", "3-out", 0),
        case("beta", "4-in", 1),
    ]
    embeddings = np.asarray([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.8, 0.2],
    ])
    result = similarity_features(
        cases,
        embeddings,
        wiki_embeddings={"alpha": np.asarray([1.0, 0.0]), "beta": np.asarray([0.0, 1.0])},
        portfolio_embeddings={"alpha": np.asarray([[0.7, 0.3]]), "beta": np.empty((0, 2))},
        reference_indices=(0, 1, 2, 3, 4),
        query_indices=(0, 1),
    )

    alpha = result.features_by_index[0]
    assert alpha["similarity__prior_in_count"] == 1.0
    assert alpha["similarity__prior_out_count"] == 1.0
    assert result.provenance_by_index[0]["prior_in_episodes"] == ("2-in",)
    assert "1-shared" not in result.provenance_by_index[0]["prior_out_episodes"]
    beta = result.features_by_index[1]
    assert beta["similarity__prior_in_count"] == 1.0
    assert beta["similarity__prior_out_unavailable"] == 1.0
    assert beta["similarity__portfolio_unavailable"] == 1.0


def test_similarity_is_finite_for_empty_history_and_portfolio() -> None:
    cases = [case("alpha", "1-only", 1)]
    result = similarity_features(
        cases,
        np.asarray([[1.0, 0.0]]),
        wiki_embeddings={"alpha": np.asarray([1.0, 0.0])},
        portfolio_embeddings={},
        reference_indices=(0,),
        query_indices=(0,),
    )

    values = result.features_by_index[0]
    assert all(np.isfinite(value) for value in values.values())
    assert values["similarity__prior_in_unavailable"] == 1.0
    assert values["similarity__prior_out_unavailable"] == 1.0
    assert values["similarity__portfolio_unavailable"] == 1.0
