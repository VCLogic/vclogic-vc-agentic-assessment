"""Strict configuration for the interactive founder-rehearsal workflow."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import tomllib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from .config import EmbeddingSettings, ProviderSettings, RetrievalSettings


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or not value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("path must be a safe relative path")
    return value


class RehearsalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_root: str = "inputs"
    output_root: str = "outputs/rehearsals"
    checkpoint_path: str = "outputs/rehearsals/checkpoints.sqlite"
    taxonomy_path: str = "taxonomy/codebook_v_final.json"
    max_questions: int = Field(default=8, ge=1, le=8)
    max_output_tokens: int = Field(default=8192, ge=1024, le=16384)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    repair_attempts: int = Field(default=1, ge=0, le=1)

    @model_validator(mode="after")
    def validate_paths(self) -> "RehearsalSettings":
        for value in (
            self.input_root,
            self.output_root,
            self.checkpoint_path,
            self.taxonomy_path,
        ):
            _safe_relative(value)
        taxonomy = PurePosixPath(self.taxonomy_path)
        if taxonomy.parts[0] != "taxonomy" or taxonomy.suffix != ".json":
            raise ValueError("taxonomy_path must select a JSON file under taxonomy/")
        return self


class RehearsalPrecedentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    corpus_path: str = "data/investors/<vc_slug>/precedents"
    selection_policy: Literal["semantic", "contrastive"] = "semantic"
    candidate_pool_k: int = Field(default=30, ge=1, le=100)
    in_slots: int = Field(default=2, ge=0, le=20)
    out_slots: int = Field(default=3, ge=0, le=20)

    @model_validator(mode="after")
    def validate_path_and_pool(self) -> "RehearsalPrecedentSettings":
        _safe_relative(self.corpus_path)
        if PurePosixPath(self.corpus_path).parts != (
            "data",
            "investors",
            "<vc_slug>",
            "precedents",
        ):
            raise ValueError(
                "precedent corpus_path must be data/investors/<vc_slug>/precedents"
            )
        if self.in_slots + self.out_slots > self.candidate_pool_k:
            raise ValueError("precedent slots must not exceed candidate_pool_k")
        return self


class RehearsalPortfolioSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    corpus_path: str = "data/investors/<vc_slug>/portfolio-memory"
    retrieval_top_k: int = Field(default=5, ge=1, le=30)
    candidate_pool_k: int = Field(default=15, ge=1, le=100)

    @model_validator(mode="after")
    def validate_path_and_pool(self) -> "RehearsalPortfolioSettings":
        _safe_relative(self.corpus_path)
        if PurePosixPath(self.corpus_path).parts != (
            "data",
            "investors",
            "<vc_slug>",
            "portfolio-memory",
        ):
            raise ValueError(
                "portfolio corpus_path must be "
                "data/investors/<vc_slug>/portfolio-memory"
            )
        if self.retrieval_top_k > self.candidate_pool_k:
            raise ValueError(
                "portfolio retrieval_top_k must not exceed candidate_pool_k"
            )
        return self


class RehearsalQuestionMemorySettings(BaseModel):
    """Local investor-question retrieval and hybrid reranking settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    top_k: int = Field(default=5, ge=1, le=10)
    candidate_pool_k: int = Field(default=30, ge=5, le=100)
    semantic_weight: float = Field(default=0.50, ge=0, le=1)
    rationale_weight: float = Field(default=0.25, ge=0, le=1)
    lexical_weight: float = Field(default=0.15, ge=0, le=1)
    atomicity_bonus: float = Field(default=0.10, ge=0, le=1)
    long_question_penalty: float = Field(default=0.10, ge=0, le=1)
    multi_request_penalty: float = Field(default=0.20, ge=0, le=1)
    duplicate_penalty: float = Field(default=0.40, ge=0, le=1)
    minimum_hybrid_score: float = Field(default=0.25, ge=0, le=1)

    @model_validator(mode="after")
    def validate_pool_and_weights(self) -> "RehearsalQuestionMemorySettings":
        if self.top_k > self.candidate_pool_k:
            raise ValueError("question-memory top_k must not exceed candidate_pool_k")
        if not any(
            (
                self.semantic_weight,
                self.rationale_weight,
                self.lexical_weight,
                self.atomicity_bonus,
            )
        ):
            raise ValueError("question memory requires positive scoring weight")
        return self


class RehearsalClassificationSettings(BaseModel):
    """Controls optional classifier guidance during founder rehearsal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal[
        "classification_informed", "rationale_only", "v41_grounded"
    ] = "classification_informed"
    registry_path: str = "evaluation/rehearsal_classifier_registry/registry.json"
    canonical_registry_path: str = (
        "evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json"
    )
    canonical_config_path: str | None = None
    live_contract_version: Literal["v4", "v4.1"] = "v4.1"
    classifier_tiebreaker: bool = True
    minimum_probability_impact: float = Field(default=0.05, ge=0, le=1)
    minimum_questions: int = Field(default=1, ge=0, le=8)
    maximum_questions: int = Field(default=8, ge=1, le=8)
    fallback_to_rationale_only: bool = True

    @model_validator(mode="after")
    def validate_settings(self) -> "RehearsalClassificationSettings":
        _safe_relative(self.registry_path)
        _safe_relative(self.canonical_registry_path)
        if self.canonical_config_path is not None:
            _safe_relative(self.canonical_config_path)
        if self.minimum_questions > self.maximum_questions:
            raise ValueError("minimum_questions must not exceed maximum_questions")
        return self


class RehearsalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    _workspace: Path | None = PrivateAttr(default=None)

    @property
    def workspace(self) -> Path:
        """Runtime base directory; omitted from serialized configuration."""
        return self._workspace if self._workspace is not None else Path.cwd().resolve()

    def resolve_path(self, value: str | Path) -> Path:
        return (self.workspace / value).resolve()

    rehearsal: RehearsalSettings
    provider: ProviderSettings
    embedding: EmbeddingSettings
    retrieval: RetrievalSettings
    precedents: RehearsalPrecedentSettings = RehearsalPrecedentSettings()
    portfolio_memory: RehearsalPortfolioSettings = RehearsalPortfolioSettings()
    question_memory: RehearsalQuestionMemorySettings = (
        RehearsalQuestionMemorySettings()
    )
    classification: RehearsalClassificationSettings = (
        RehearsalClassificationSettings()
    )

    @model_validator(mode="after")
    def validate_provider_and_embeddings(self) -> "RehearsalConfig":
        if self.embedding.kind not in {"sentence_transformers", "ollama"}:
            raise ValueError("rehearsal retrieval requires a local embedding provider")
        if (
            self.rehearsal.reasoning_effort is not None
            and self.provider.kind != "openrouter"
        ):
            raise ValueError("reasoning_effort requires provider.kind=openrouter")
        if self.classification.maximum_questions > self.rehearsal.max_questions:
            raise ValueError(
                "classification maximum_questions must not exceed rehearsal max_questions"
            )
        return self


def load_rehearsal_config(path: Path, *, workspace: Path | None = None) -> RehearsalConfig:
    """Load a strict rehearsal TOML file."""
    candidate = Path(path)
    if workspace is not None:
        candidate = Path(workspace).resolve() / candidate
    try:
        with candidate.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot load rehearsal configuration: {candidate}") from exc
    config = RehearsalConfig.model_validate(raw)
    if workspace is not None:
        config._workspace = Path(workspace).resolve()
    return config
