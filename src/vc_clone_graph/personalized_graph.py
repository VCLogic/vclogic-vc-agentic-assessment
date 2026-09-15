"""Leakage-safe inductive heterogeneous graphs and a small GraphSAGE challenger."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import numpy as np

from .personalized_cases import PersonalizedCase
from .personalized_folds import EpisodeFold
from .personalized_matrices import RawFeatureViews
from .personalized_tabpfn import (
    PersonalizedPrediction,
    TabPFNCandidate,
    select_tabpfn_heads,
)


Edge = tuple[str, str]


@dataclass(frozen=True)
class HeterogeneousGraph:
    node_ids: Mapping[str, tuple[str, ...]]
    edges: Mapping[str, tuple[Edge, ...]]
    target_edges: tuple[Edge, ...]
    target_labels: tuple[int, ...]


@dataclass(frozen=True)
class InductiveGraphs:
    training: HeterogeneousGraph
    inference: HeterogeneousGraph


def graph_execution_gate(
    *, enabled: bool, run_only_after_primary: bool, output: Path
) -> tuple[bool, str]:
    if not enabled:
        return False, "graph challenger is disabled"
    if run_only_after_primary and not (Path(output) / "summary.json").is_file():
        return False, "primary report is required before graph execution"
    return True, "eligible"


def _predicted_labels(case: PersonalizedCase, labels: Sequence[str]) -> tuple[str, ...]:
    active = []
    for label in labels:
        prefix = f"rationale__{label}__"
        values = [
            float(value) for name, value in case.phase1_features.items()
            if name.startswith(prefix) and isinstance(value, (int, float))
        ]
        if values and max(values) > 0:
            active.append(label)
    return tuple(active)


def _base_nodes(
    cases: Sequence[PersonalizedCase],
    included_indices: Sequence[int],
    taxonomy_labels: Sequence[str],
    portfolio_entities: Mapping[str, Sequence[str]],
    wiki_claims: Mapping[str, Sequence[str]],
    sectors: Mapping[str, Sequence[str]],
) -> dict[str, set[str]]:
    vcs = sorted({case.vc_slug for case in cases})
    episodes = sorted({cases[index].episode_slug for index in included_indices})
    return {
        "investor": {f"investor:{vc}" for vc in vcs},
        "pitch": {f"pitch:{episode}" for episode in episodes},
        "rationale": {f"rationale:{label}" for label in taxonomy_labels},
        "portfolio_company": {
            f"portfolio:{vc}:{company}" for vc in vcs
            for company in portfolio_entities.get(vc, ())
        },
        "sector": {
            f"sector:{sector}" for episode in episodes
            for sector in sectors.get(episode, ())
        },
        "wiki_claim": {
            f"wiki:{vc}:{offset}" for vc in vcs
            for offset, _ in enumerate(wiki_claims.get(vc, ()))
        },
    }


def _add_static_edges(
    nodes: Mapping[str, set[str]],
    cases: Sequence[PersonalizedCase],
    included_indices: Sequence[int],
    taxonomy_labels: Sequence[str],
    portfolio_entities: Mapping[str, Sequence[str]],
    wiki_claims: Mapping[str, Sequence[str]],
    sectors: Mapping[str, Sequence[str]],
) -> dict[str, set[Edge]]:
    edges: dict[str, set[Edge]] = {}

    def add(relation: str, left: str, right: str) -> None:
        edges.setdefault(relation, set()).add((left, right))

    for vc in sorted({case.vc_slug for case in cases}):
        for company in portfolio_entities.get(vc, ()):
            add("has_portfolio_company", f"investor:{vc}", f"portfolio:{vc}:{company}")
        for offset, _ in enumerate(wiki_claims.get(vc, ())):
            add("supported_by_wiki_claim", f"investor:{vc}", f"wiki:{vc}:{offset}")
    for index in included_indices:
        case = cases[index]
        pitch = f"pitch:{case.episode_slug}"
        add("reviews", f"investor:{case.vc_slug}", pitch)
        for sector in sectors.get(case.episode_slug, ()):
            add("in_sector", pitch, f"sector:{sector}")
        for label in _predicted_labels(case, taxonomy_labels):
            add("predicted_rationale", pitch, f"rationale:{label}")
    return edges


def _undirected(edges: Mapping[str, set[Edge]]) -> dict[str, tuple[Edge, ...]]:
    result: dict[str, tuple[Edge, ...]] = {}
    for relation, values in edges.items():
        forward = tuple(sorted(values))
        result[relation] = forward
        result[f"reverse__{relation}"] = tuple(sorted((right, left) for left, right in values))
    return result


def _graph(
    cases: Sequence[PersonalizedCase],
    included_indices: Sequence[int],
    taxonomy_labels: Sequence[str],
    *,
    portfolio_entities: Mapping[str, Sequence[str]],
    wiki_claims: Mapping[str, Sequence[str]],
    sectors: Mapping[str, Sequence[str]],
    include_observed: bool,
    target_indices: Sequence[int],
) -> HeterogeneousGraph:
    nodes = _base_nodes(
        cases, included_indices, taxonomy_labels, portfolio_entities, wiki_claims, sectors
    )
    edges = _add_static_edges(
        nodes, cases, included_indices, taxonomy_labels,
        portfolio_entities, wiki_claims, sectors,
    )
    if include_observed:
        for index in included_indices:
            case = cases[index]
            investor = f"investor:{case.vc_slug}"
            pitch = f"pitch:{case.episode_slug}"
            relation = "decision_in" if case.target else "decision_out"
            edges.setdefault(relation, set()).add((investor, pitch))
            for label, active in case.actual_rationale_targets.items():
                if active and label in taxonomy_labels:
                    edges.setdefault("observed_rationale", set()).add(
                        (pitch, f"rationale:{label}")
                    )
    target_edges = tuple(
        (f"investor:{cases[index].vc_slug}", f"pitch:{cases[index].episode_slug}")
        for index in target_indices
    )
    target_labels = tuple(cases[index].target for index in target_indices)
    visible_nodes = {node for values in nodes.values() for node in values}
    for relation_edges in edges.values():
        for left, right in relation_edges:
            if left not in visible_nodes or right not in visible_nodes:
                raise ValueError("graph edge endpoint is absent from fold nodes")
    if any(left not in visible_nodes or right not in visible_nodes for left, right in target_edges):
        raise ValueError("graph target endpoint is absent from fold nodes")
    return HeterogeneousGraph(
        node_ids={kind: tuple(sorted(values)) for kind, values in nodes.items()},
        edges=_undirected(edges), target_edges=target_edges, target_labels=target_labels,
    )


def build_inductive_graphs(
    cases: Sequence[PersonalizedCase],
    fold: EpisodeFold,
    *,
    taxonomy_labels: Sequence[str],
    portfolio_entities: Mapping[str, Sequence[str]] | None = None,
    wiki_claims: Mapping[str, Sequence[str]] | None = None,
    sectors: Mapping[str, Sequence[str]] | None = None,
) -> InductiveGraphs:
    """Build a training graph and label-free held-pitch inference graph."""
    portfolio_entities = portfolio_entities or {}
    wiki_claims = wiki_claims or {}
    sectors = sectors or {}
    if any(cases[index].episode_slug == fold.held_episode for index in fold.train_indices):
        raise ValueError("held episode appears in graph training indices")
    training = _graph(
        cases, fold.train_indices, taxonomy_labels,
        portfolio_entities=portfolio_entities, wiki_claims=wiki_claims, sectors=sectors,
        include_observed=True, target_indices=fold.train_indices,
    )
    inference = _graph(
        cases, tuple(fold.train_indices) + tuple(fold.test_indices), taxonomy_labels,
        portfolio_entities=portfolio_entities, wiki_claims=wiki_claims, sectors=sectors,
        include_observed=False, target_indices=fold.test_indices,
    )
    return InductiveGraphs(training=training, inference=inference)


GraphScorer = Callable[..., np.ndarray]


def _project(values: np.ndarray, dimensions: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(source) >= dimensions:
        return source[:dimensions]
    return np.pad(source, (0, dimensions - len(source)))


def _graph_features(
    graph: HeterogeneousGraph,
    cases: Sequence[PersonalizedCase],
    views: Sequence[RawFeatureViews],
    taxonomy_labels: Sequence[str],
    *,
    dimensions: int = 64,
) -> tuple[tuple[str, ...], np.ndarray]:
    nodes = tuple(node for kind in sorted(graph.node_ids) for node in graph.node_ids[kind])
    pitch_vectors: dict[str, list[np.ndarray]] = {}
    investor_vectors: dict[str, list[np.ndarray]] = {}
    for case, view in zip(cases, views, strict=True):
        pitch_vectors.setdefault(f"pitch:{case.episode_slug}", []).append(view.pitch_embedding)
        investor_vectors.setdefault(f"investor:{case.vc_slug}", []).append(view.wiki_embedding)
    rationale_offset = {label: index for index, label in enumerate(taxonomy_labels)}
    rows = []
    for node in nodes:
        if node in pitch_vectors:
            vector = np.mean(pitch_vectors[node], axis=0)
        elif node in investor_vectors:
            vector = np.mean(investor_vectors[node], axis=0)
        elif node.startswith("rationale:"):
            vector = np.zeros(dimensions, dtype=np.float32)
            label = node.split(":", 1)[1]
            vector[rationale_offset.get(label, 0) % dimensions] = 1.0
        else:
            vector = np.zeros(dimensions, dtype=np.float32)
            vector[int(sha256(node.encode()).hexdigest()[:8], 16) % dimensions] = 1.0
        rows.append(_project(vector, dimensions))
    return nodes, np.asarray(rows, dtype=np.float32)


def default_graph_scorer(
    cases: Sequence[PersonalizedCase],
    views: Sequence[RawFeatureViews],
    train_indices: Sequence[int],
    query_indices: Sequence[int],
    *,
    taxonomy_labels: Sequence[str],
    task: str,
    hidden_dimensions: int,
    layers: int,
    dropout: float,
    weight_decay: float,
    epochs: int,
    seed: int,
    device: str,
) -> np.ndarray:
    """Fit one relation-specific inductive GraphSAGE head and score query edges."""
    import torch
    from torch import nn
    import torch.nn.functional as functional

    if task not in {"classification", "ranking"}:
        raise ValueError(f"unknown graph task: {task}")
    torch.manual_seed(seed)
    train_graph = _graph(
        cases, train_indices, taxonomy_labels, portfolio_entities={}, wiki_claims={},
        sectors={}, include_observed=True, target_indices=train_indices,
    )
    inference_graph = _graph(
        cases, tuple(train_indices) + tuple(query_indices), taxonomy_labels,
        portfolio_entities={}, wiki_claims={}, sectors={}, include_observed=False,
        target_indices=query_indices,
    )
    train_nodes, train_features = _graph_features(
        train_graph, cases, views, taxonomy_labels
    )
    query_nodes, query_features = _graph_features(
        inference_graph, cases, views, taxonomy_labels
    )
    relations = tuple(sorted(set(train_graph.edges) | set(inference_graph.edges)))

    class RelationGraphSAGE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_layers = nn.ModuleList()
            self.relation_layers = nn.ModuleList()
            width = train_features.shape[1]
            for _ in range(layers):
                self.self_layers.append(nn.Linear(width, hidden_dimensions))
                self.relation_layers.append(nn.ModuleDict({
                    relation: nn.Linear(width, hidden_dimensions, bias=False)
                    for relation in relations
                }))
                width = hidden_dimensions
            self.decoder = nn.Linear(hidden_dimensions * 2, 1)

        def encode(self, x, edges, node_position):
            for self_layer, relation_layers in zip(
                self.self_layers, self.relation_layers, strict=True
            ):
                updated = self_layer(x)
                counts = torch.ones((len(x), 1), device=x.device)
                for relation, relation_edges in edges.items():
                    if not relation_edges:
                        continue
                    source = torch.tensor(
                        [node_position[left] for left, _ in relation_edges],
                        dtype=torch.long, device=x.device,
                    )
                    target = torch.tensor(
                        [node_position[right] for _, right in relation_edges],
                        dtype=torch.long, device=x.device,
                    )
                    messages = relation_layers[relation](x[source])
                    updated.index_add_(0, target, messages)
                    counts.index_add_(
                        0, target, torch.ones((len(target), 1), device=x.device)
                    )
                x = functional.dropout(
                    functional.relu(updated / counts), p=dropout, training=self.training
                )
            return x

        def score(self, encoded, target_edges, node_position):
            left = torch.tensor(
                [node_position[source] for source, _ in target_edges],
                dtype=torch.long, device=encoded.device,
            )
            right = torch.tensor(
                [node_position[target] for _, target in target_edges],
                dtype=torch.long, device=encoded.device,
            )
            return self.decoder(torch.cat((encoded[left], encoded[right]), dim=1)).squeeze(1)

    selected_device = torch.device(device)
    model = RelationGraphSAGE().to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    x_train = torch.tensor(train_features, device=selected_device)
    y_train = torch.tensor(train_graph.target_labels, dtype=torch.float32, device=selected_device)
    train_position = {node: index for index, node in enumerate(train_nodes)}
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        encoded = model.encode(x_train, train_graph.edges, train_position)
        logits = model.score(encoded, train_graph.target_edges, train_position)
        if task == "ranking":
            pairs = []
            for vc in sorted({cases[index].vc_slug for index in train_indices}):
                vc_positions = [
                    offset for offset, index in enumerate(train_indices)
                    if cases[index].vc_slug == vc
                ]
                positives = [offset for offset in vc_positions if y_train[offset] == 1]
                negatives = [offset for offset in vc_positions if y_train[offset] == 0]
                pairs.extend((positive, negative) for positive in positives for negative in negatives)
            if pairs:
                positive = torch.tensor([left for left, _ in pairs], device=selected_device)
                negative = torch.tensor([right for _, right in pairs], device=selected_device)
                loss = functional.softplus(-(logits[positive] - logits[negative])).mean()
            else:
                loss = functional.binary_cross_entropy_with_logits(logits, y_train)
        else:
            positive_weight = torch.tensor(
                [(y_train == 0).sum() / torch.clamp((y_train == 1).sum(), min=1)],
                device=selected_device,
            )
            loss = functional.binary_cross_entropy_with_logits(
                logits, y_train, pos_weight=positive_weight
            )
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        x_query = torch.tensor(query_features, device=selected_device)
        query_position = {node: index for index, node in enumerate(query_nodes)}
        encoded = model.encode(x_query, inference_graph.edges, query_position)
        scores = torch.sigmoid(
            model.score(encoded, inference_graph.target_edges, query_position)
        )
    return scores.detach().cpu().numpy().astype(float)


def run_graph_fold(
    cases: Sequence[PersonalizedCase],
    views: Sequence[RawFeatureViews],
    fold: EpisodeFold,
    *,
    condition: str,
    taxonomy_labels: Sequence[str],
    hidden_dimensions: int,
    layers: int,
    dropout: float,
    weight_decay: float,
    epochs: int,
    seed: int,
    device: str,
    scorer: GraphScorer = default_graph_scorer,
) -> list[PersonalizedPrediction]:
    """Cross-fit graph scores, select a threshold, refit, and score held pitches."""
    started = time.monotonic()
    position = {index: offset for offset, index in enumerate(fold.train_indices)}
    targets = np.asarray([cases[index].target for index in fold.train_indices], dtype=int)
    vcs = np.asarray([cases[index].vc_slug for index in fold.train_indices], dtype=object)
    oof = {
        task: np.zeros(len(fold.train_indices), dtype=float)
        for task in ("classification", "ranking")
    }
    common = dict(
        taxonomy_labels=taxonomy_labels, hidden_dimensions=hidden_dimensions,
        layers=layers, dropout=dropout, weight_decay=weight_decay, epochs=epochs,
        device=device,
    )
    for task_index, task in enumerate(("classification", "ranking")):
        for split_index, (inner_train, inner_valid) in enumerate(fold.inner_splits):
            predicted = scorer(
                cases, views, inner_train, inner_valid, task=task,
                seed=seed + task_index * 1000 + split_index, **common,
            )
            for index, score in zip(inner_valid, predicted, strict=True):
                oof[task][position[index]] = float(score)
    selected = select_tabpfn_heads(
        targets, vcs,
        (TabPFNCandidate("graph_classification", oof["classification"]),),
    )
    classification_scores = scorer(
        cases, views, fold.train_indices, fold.test_indices, task="classification",
        seed=seed + 5000, **common,
    )
    ranking_scores = scorer(
        cases, views, fold.train_indices, fold.test_indices, task="ranking",
        seed=seed + 6000, **common,
    )
    training_episodes = tuple(sorted({cases[index].episode_slug for index in fold.train_indices}))
    inner_episodes = tuple(
        tuple(sorted({cases[index].episode_slug for index in valid}))
        for _, valid in fold.inner_splits
    )
    classification_inner = tuple(
        (cases[index].vc_slug, cases[index].episode_slug, float(oof["classification"][position[index]]))
        for index in fold.train_indices
    )
    ranking_inner = tuple(
        (cases[index].vc_slug, cases[index].episode_slug, float(oof["ranking"][position[index]]))
        for index in fold.train_indices
    )
    schema = sha256(json.dumps({
        "relations": "typed", "hidden": hidden_dimensions, "layers": layers,
        "dropout": dropout, "weight_decay": weight_decay,
    }, sort_keys=True).encode()).hexdigest()
    runtime = time.monotonic() - started
    return [PersonalizedPrediction(
        family="graph", condition=condition, vc_slug=cases[index].vc_slug,
        vc_name=cases[index].vc_name, episode_slug=cases[index].episode_slug,
        target=cases[index].target, classification_score=float(classification_score),
        ranking_score=float(ranking_score),
        predicted=int(classification_score >= selected.classification_threshold),
        classification_threshold=selected.classification_threshold,
        classification_config="relation_graphsage_classification",
        ranking_config="relation_graphsage_ranking",
        training_episodes=training_episodes,
        inner_validation_episodes=inner_episodes,
        feature_schema_sha256=schema, runtime_seconds=runtime, device=device,
        status="complete", classification_inner_oof=classification_inner,
        ranking_inner_oof=ranking_inner,
    ) for index, classification_score, ranking_score in zip(
        fold.test_indices, classification_scores, ranking_scores, strict=True
    )]
