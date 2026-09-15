from __future__ import annotations

from vc_clone_graph.personalized_cases import PersonalizedCase
from vc_clone_graph.personalized_folds import EpisodeFold
from vc_clone_graph.personalized_graph import (
    build_inductive_graphs,
    graph_execution_gate,
    default_graph_scorer,
    run_graph_fold,
)
from vc_clone_graph.personalized_matrices import RawFeatureViews


def cases() -> list[PersonalizedCase]:
    return [PersonalizedCase(
        vc_slug=vc, vc_input_slug=f"{vc}-fund", vc_name=vc.title(),
        episode_slug=episode, group=episode, target=target,
        pitch_text=f"{episode} pitch", phase1_text="market positive", phase2_text="decision",
        phase1_features={"rationale__market__confidence": 0.8},
        phase2_features={"investment_likelihood": 0.6}, wiki_text="wiki claim",
        actual_rationale_targets={"market": 1}, actual_rationale_available=True,
        actual_rationale_source_tier="test", actual_rationale_source_format="test",
        source_hashes={"pitch": episode},
    ) for episode, target in (
        ("1-train", 0), ("2-held", 1), ("3-positive", 1), ("4-negative", 0)
    ) for vc in ("alpha", "beta")]


def held_fold() -> EpisodeFold:
    return EpisodeFold(
        held_episode="2-held", train_indices=(0, 1, 4, 5, 6, 7), test_indices=(2, 3),
        inner_splits=(
            ((4, 5, 6, 7), (0, 1)),
            ((0, 1, 6, 7), (4, 5)),
            ((0, 1, 4, 5), (6, 7)),
        ),
    )


def test_inductive_graph_removes_held_pitch_and_all_derived_edges() -> None:
    population = cases()
    fold = held_fold()
    graphs = build_inductive_graphs(
        population,
        fold,
        taxonomy_labels=("market", "team"),
        portfolio_entities={"alpha": ("Acme",), "beta": ("BetaCo",)},
        wiki_claims={"alpha": ("seed investor",), "beta": ("consumer investor",)},
        sectors={"1-train": ("software",), "2-held": ("health",)},
    )

    assert not any("2-held" in node for node in graphs.training.node_ids["pitch"])
    assert all("2-held" not in endpoint for edges in graphs.training.edges.values() for edge in edges for endpoint in edge)
    assert set(graphs.inference.node_ids["pitch"]) >= {
        "pitch:1-train", "pitch:2-held"
    }
    held_edges = [
        (relation, edge) for relation, edges in graphs.inference.edges.items()
        for edge in edges if "2-held" in edge[0] or "2-held" in edge[1]
    ]
    assert held_edges
    assert all(relation not in {"observed_rationale", "decision_in", "decision_out"} for relation, _ in held_edges)
    assert all("2-held" not in edge for edge in graphs.training.target_edges)
    assert {edge[1] for edge in graphs.inference.target_edges} == {
        "pitch:2-held", "pitch:2-held"
    }


def test_graph_relations_are_undirected_only_after_target_removal() -> None:
    population = cases()
    fold = held_fold()
    graphs = build_inductive_graphs(population, fold, taxonomy_labels=("market",))

    for relation, edges in graphs.training.edges.items():
        if relation.startswith("reverse__"):
            continue
        assert f"reverse__{relation}" in graphs.training.edges
        assert {(right, left) for left, right in edges} == set(
            graphs.training.edges[f"reverse__{relation}"]
        )
    assert all(
        relation not in graphs.inference.edges
        for relation in ("decision_in", "decision_out", "reverse__decision_in", "reverse__decision_out")
    )


def test_graph_gate_requires_enablement_and_primary_report(tmp_path) -> None:
    assert graph_execution_gate(enabled=False, run_only_after_primary=True, output=tmp_path) == (
        False, "graph challenger is disabled"
    )
    assert graph_execution_gate(enabled=True, run_only_after_primary=True, output=tmp_path)[0] is False
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    assert graph_execution_gate(enabled=True, run_only_after_primary=True, output=tmp_path) == (
        True, "eligible"
    )


def test_graph_fold_cross_fits_threshold_and_scores_held_rows() -> None:
    import numpy as np

    population = cases()
    views = [RawFeatureViews(
        pitch_embedding=np.asarray([float(index), 1.0]),
        phase1_embedding=np.asarray([0.0]), phase2_embedding=np.asarray([0.0]),
        wiki_embedding=np.asarray([1.0, float(index)]), pitch_structured={},
        phase1_structured={}, phase2_structured={}, investor_structured={},
        teacher_structured={}, interaction_structured={},
    ) for index in range(len(population))]
    calls = []

    def fake_scorer(cases, views, train_indices, query_indices, **kwargs):
        calls.append((tuple(train_indices), tuple(query_indices), kwargs["task"]))
        return np.asarray([0.8 if cases[index].target else 0.2 for index in query_indices])

    predictions = run_graph_fold(
        population, views, held_fold(), condition="all", taxonomy_labels=("market",),
        hidden_dimensions=8, layers=2, dropout=0.1, weight_decay=1e-4,
        epochs=2, seed=11, device="cpu", scorer=fake_scorer,
    )

    assert len(predictions) == 2
    assert all(row.predicted == 1 for row in predictions)
    assert all("2-held" not in row.training_episodes for row in predictions)
    assert len(calls) == 8  # two heads x (three inner folds + outer refit)


def test_small_relation_graphsage_scores_inductive_rows() -> None:
    import numpy as np
    import pytest
    pytest.importorskip("torch")

    population = cases()
    views = [RawFeatureViews(
        pitch_embedding=np.asarray([float(index), 1.0]),
        phase1_embedding=np.asarray([0.0]), phase2_embedding=np.asarray([0.0]),
        wiki_embedding=np.asarray([1.0, float(index)]), pitch_structured={},
        phase1_structured={}, phase2_structured={}, investor_structured={},
        teacher_structured={}, interaction_structured={},
    ) for index in range(len(population))]

    scores = default_graph_scorer(
        population, views, held_fold().train_indices, held_fold().test_indices,
        taxonomy_labels=("market",), task="classification", hidden_dimensions=4,
        layers=2, dropout=0.1, weight_decay=1e-4, epochs=1, seed=13,
        device="cpu",
    )

    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
