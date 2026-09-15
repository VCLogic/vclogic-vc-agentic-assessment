"""Typed configuration for the personalized multimodal evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any, Mapping


def _section(raw: Mapping[str, Any], name: str, allowed: set[str]) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing or invalid [{name}] section")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown [{name}] keys: {', '.join(unknown)}")
    return value


def _text(section: Mapping[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _positive_int_tuple(section: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = section.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value)
    ):
        raise ValueError(f"{key} must contain positive integers")
    return tuple(value)


def _float_tuple(
    section: Mapping[str, Any], key: str, *, minimum: float = 0.0
) -> tuple[float, ...]:
    value = section.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty numeric list")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item < minimum for item in result):
        raise ValueError(f"{key} contains an invalid value")
    return result


@dataclass(frozen=True)
class DataConfig:
    source_registry: str
    input_root: str
    rationale_references: str
    taxonomy: str


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    revision: str
    device: str
    batch_size: int
    normalize: bool
    document_prefix: str
    query_prefix: str


@dataclass(frozen=True)
class EvaluationConfig:
    outer_group: str
    seed: int
    inner_splits: int
    classification_metric: str
    ranking_metric: str
    review_budgets: tuple[int, ...]


@dataclass(frozen=True)
class TabPFNConfig:
    enabled: bool
    n_estimators: tuple[int, ...]
    pca_components: tuple[int, ...]
    device: str


@dataclass(frozen=True)
class SetFitConfig:
    enabled: bool
    model: str
    revision: str
    device: str
    lambda_rank: tuple[float, ...]
    lambda_aux: tuple[float, ...]
    epochs: tuple[int, ...]
    max_pairs: int


@dataclass(frozen=True)
class GraphConfig:
    enabled: bool
    run_only_after_primary: bool
    hidden_dimensions: int
    layers: int
    dropout: float
    weight_decay: float
    epochs: int


@dataclass(frozen=True)
class EnsembleConfig:
    enabled: bool
    minimum_disagreement_rate: float
    require_inner_improvement: bool


@dataclass(frozen=True)
class ExecutionConfig:
    output: str
    cache: str
    max_retries: int
    api_cost_ceiling_usd: float


@dataclass(frozen=True)
class PersonalizedConfig:
    data: DataConfig
    embedding: EmbeddingConfig
    evaluation: EvaluationConfig
    tabpfn: TabPFNConfig
    setfit: SetFitConfig
    graph: GraphConfig
    ensemble: EnsembleConfig
    execution: ExecutionConfig


def load_personalized_config(path: Path) -> PersonalizedConfig:
    """Load a strict, bounded personalized-evaluation TOML file."""
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    expected_sections = {
        "data", "embedding", "evaluation", "tabpfn", "setfit", "graph",
        "ensemble", "execution",
    }
    unknown_sections = sorted(set(raw) - expected_sections)
    if unknown_sections:
        raise ValueError(f"unknown configuration sections: {', '.join(unknown_sections)}")

    data = _section(raw, "data", {
        "source_registry", "input_root", "rationale_references", "taxonomy",
    })
    embedding = _section(raw, "embedding", {
        "model", "revision", "device", "batch_size", "normalize",
        "document_prefix", "query_prefix",
    })
    evaluation = _section(raw, "evaluation", {
        "outer_group", "seed", "inner_splits", "classification_metric",
        "ranking_metric", "review_budgets",
    })
    tabpfn = _section(raw, "tabpfn", {
        "enabled", "n_estimators", "pca_components", "device",
    })
    setfit = _section(raw, "setfit", {
        "enabled", "model", "revision", "device", "lambda_rank", "lambda_aux", "epochs",
        "max_pairs",
    })
    graph = _section(raw, "graph", {
        "enabled", "run_only_after_primary", "hidden_dimensions", "layers", "dropout",
        "weight_decay", "epochs",
    })
    ensemble = _section(raw, "ensemble", {
        "enabled", "minimum_disagreement_rate", "require_inner_improvement",
    })
    execution = _section(raw, "execution", {
        "output", "cache", "max_retries", "api_cost_ceiling_usd",
    })

    outer_group = _text(evaluation, "outer_group")
    if outer_group != "episode_slug":
        raise ValueError("outer_group must be episode_slug")
    classification_metric = _text(evaluation, "classification_metric")
    ranking_metric = _text(evaluation, "ranking_metric")
    if classification_metric != "macro_balanced_accuracy":
        raise ValueError("classification_metric must be macro_balanced_accuracy")
    if ranking_metric != "macro_within_vc_average_precision":
        raise ValueError("ranking_metric must be macro_within_vc_average_precision")

    embedding_revision = _text(embedding, "revision")
    setfit_revision = _text(setfit, "revision")
    api_cost = float(execution.get("api_cost_ceiling_usd", -1.0))
    if not math.isfinite(api_cost) or api_cost < 0:
        raise ValueError("API cost ceiling must be finite and non-negative")
    max_retries = int(execution.get("max_retries", -1))
    if max_retries not in {0, 1}:
        raise ValueError("max_retries must be zero or one")
    disagreement = float(ensemble.get("minimum_disagreement_rate", -1.0))
    if not 0.0 <= disagreement <= 1.0:
        raise ValueError("minimum_disagreement_rate must lie in [0, 1]")
    dropout = float(graph.get("dropout", -1.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("graph dropout must lie in [0, 1)")
    weight_decay = float(graph.get("weight_decay", -1.0))
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("graph weight_decay must be finite and non-negative")
    graph_epochs = int(graph.get("epochs", 0))
    if graph_epochs < 1:
        raise ValueError("graph epochs must be positive")
    if int(graph.get("hidden_dimensions", 0)) < 1 or int(graph.get("layers", 0)) < 1:
        raise ValueError("graph dimensions and layers must be positive")

    return PersonalizedConfig(
        data=DataConfig(*(_text(data, key) for key in (
            "source_registry", "input_root", "rationale_references", "taxonomy"
        ))),
        embedding=EmbeddingConfig(
            model=_text(embedding, "model"), revision=embedding_revision,
            device=_text(embedding, "device"), batch_size=int(embedding["batch_size"]),
            normalize=bool(embedding["normalize"]),
            document_prefix=str(embedding["document_prefix"]),
            query_prefix=str(embedding["query_prefix"]),
        ),
        evaluation=EvaluationConfig(
            outer_group=outer_group, seed=int(evaluation["seed"]),
            inner_splits=int(evaluation["inner_splits"]),
            classification_metric=classification_metric, ranking_metric=ranking_metric,
            review_budgets=_positive_int_tuple(evaluation, "review_budgets"),
        ),
        tabpfn=TabPFNConfig(
            enabled=bool(tabpfn["enabled"]),
            n_estimators=_positive_int_tuple(tabpfn, "n_estimators"),
            pca_components=_positive_int_tuple(tabpfn, "pca_components"),
            device=_text(tabpfn, "device"),
        ),
        setfit=SetFitConfig(
            enabled=bool(setfit["enabled"]), model=_text(setfit, "model"),
            revision=setfit_revision, device=_text(setfit, "device"),
            lambda_rank=_float_tuple(setfit, "lambda_rank"),
            lambda_aux=_float_tuple(setfit, "lambda_aux"),
            epochs=_positive_int_tuple(setfit, "epochs"), max_pairs=int(setfit["max_pairs"]),
        ),
        graph=GraphConfig(
            enabled=bool(graph["enabled"]),
            run_only_after_primary=bool(graph["run_only_after_primary"]),
            hidden_dimensions=int(graph["hidden_dimensions"]), layers=int(graph["layers"]),
            dropout=dropout, weight_decay=weight_decay, epochs=graph_epochs,
        ),
        ensemble=EnsembleConfig(
            enabled=bool(ensemble["enabled"]), minimum_disagreement_rate=disagreement,
            require_inner_improvement=bool(ensemble["require_inner_improvement"]),
        ),
        execution=ExecutionConfig(
            output=_text(execution, "output"), cache=_text(execution, "cache"),
            max_retries=max_retries, api_cost_ceiling_usd=api_cost,
        ),
    )
