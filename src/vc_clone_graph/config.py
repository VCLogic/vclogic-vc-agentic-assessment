"""Strict configuration models and TOML loading."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import tomllib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


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


class RunSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    vc_slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    episode_slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    input_root: str
    output_root: str
    checkpoint_path: str
    taxonomy_path: str = "taxonomy/codebook_v_final.json"
    contract_version: Literal[
        "v1", "v2", "v3", "v4", "v4.1", "v4.2", "v4.3", "v4.4", "v5"
    ] = "v1"
    mode: Literal["full", "phase1_only"] = "full"

    @model_validator(mode="after")
    def validate_paths(self) -> "RunSettings":
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
        if self.mode == "phase1_only" and self.contract_version not in {
            "v4", "v4.1", "v4.2", "v4.3", "v4.4",
        }:
            raise ValueError(
                "phase1_only mode requires a v4-family contract"
            )
        return self


class ProviderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["fake", "ollama", "openai", "openrouter"]
    model: str = Field(min_length=1)
    base_url: str | None = None
    embedding_model: str | None = None
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    require_parameters: bool = True
    data_collection: Literal["allow", "deny"] = "deny"
    max_output_tokens: int = Field(default=4096, ge=256, le=16384)
    enable_thinking: bool = False
    context_window: int = Field(default=32768, ge=4096, le=131072)
    structured_output_mode: Literal["schema", "json"] = "schema"
    request_timeout_seconds: int = Field(default=300, ge=30, le=600)


class PhaseSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    min_iterations: int = Field(ge=1, le=6)
    max_iterations: int = Field(ge=1, le=6)
    model: str | None = Field(default=None, min_length=1)
    max_output_tokens: int | None = Field(default=None, ge=256, le=16384)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    planning_max_output_tokens: int = Field(default=768, ge=256, le=8192)
    planning_reasoning_effort: Literal["low", "medium", "high"] | None = None
    max_precedent_searches: int = Field(default=0, ge=0, le=20)
    max_precedent_reads: int = Field(default=0, ge=0, le=100)

    @model_validator(mode="after")
    def validate_bounds(self) -> "PhaseSettings":
        if self.min_iterations > self.max_iterations:
            raise ValueError("phase minimum must not exceed maximum")
        return self


class Phase1V44Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    taxonomy_top_k: int = Field(default=5, ge=1, le=12)
    claim_retrieval_top_k: int = Field(default=3, ge=1, le=10)
    max_wiki_reads_per_claim: int = Field(default=2, ge=1, le=6)
    max_precedent_reads_per_claim: int = Field(default=2, ge=0, le=6)
    max_revisits: int = Field(default=1, ge=0, le=1)


class PrecedentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = False
    corpus_path: str = "data/investors/precedents"
    allow_full_transcript: bool = False
    selection_policy: Literal["semantic", "contrastive"] = "semantic"
    candidate_pool_k: int = Field(default=30, ge=1, le=100)
    in_slots: int = Field(default=2, ge=0, le=50)
    out_slots: int = Field(default=2, ge=0, le=50)

    @model_validator(mode="after")
    def validate_path(self) -> "PrecedentSettings":
        _safe_relative(self.corpus_path)
        selected = self.in_slots + self.out_slots
        if selected < 1:
            raise ValueError("contrastive precedent slots must not both be zero")
        if selected > self.candidate_pool_k:
            raise ValueError(
                "contrastive precedent slots must not exceed candidate_pool_k"
            )
        return self


class PortfolioMemorySettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = False
    corpus_path: str = "data/investors/<vc_slug>/portfolio-memory"
    retrieval_top_k: int = Field(default=5, ge=1, le=30)
    candidate_pool_k: int = Field(default=15, ge=1, le=100)
    require_complete_embeddings: bool = True

    @model_validator(mode="after")
    def validate_path_and_pool(self) -> "PortfolioMemorySettings":
        _safe_relative(self.corpus_path)
        parts = PurePosixPath(self.corpus_path).parts
        if (
            len(parts) != 4
            or parts[:2] != ("data", "investors")
            or parts[2] != "<vc_slug>"
            or parts[3] != "portfolio-memory"
        ):
            raise ValueError(
                "portfolio corpus_path must be data/investors/<vc_slug>/portfolio-memory"
            )
        if self.retrieval_top_k > self.candidate_pool_k:
            raise ValueError("portfolio retrieval_top_k must not exceed candidate_pool_k")
        return self


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    top_k: int = Field(ge=1, le=30)
    max_exact_reads: int = Field(ge=1, le=100)


class EmbeddingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["sentence_transformers", "ollama", "openai", "openrouter"]
    model: str = Field(min_length=1)
    base_url: str | None = None
    revision: str | None = Field(default=None, min_length=1)
    device: str = Field(default="auto", min_length=1)
    batch_size: int = Field(default=16, ge=1, le=256)
    normalize: bool = True
    document_prefix: str = "search_document: "
    query_prefix: str = "search_query: "
    require_complete_index: bool = True

    @model_validator(mode="after")
    def validate_backend(self) -> "EmbeddingSettings":
        if self.kind == "sentence_transformers" and self.revision is None:
            raise ValueError("sentence_transformers embedding revision must be pinned")
        return self


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    _workspace: Path | None = PrivateAttr(default=None)

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve runtime paths without changing the serialized, relative config."""
        if self._workspace is None:
            return Path(value)
        return (self._workspace / value).resolve()

    run: RunSettings
    provider: ProviderSettings
    embedding: EmbeddingSettings | None = None
    phase1: PhaseSettings
    phase2: PhaseSettings
    retrieval: RetrievalSettings
    precedents: PrecedentSettings = PrecedentSettings()
    portfolio_memory: PortfolioMemorySettings = PortfolioMemorySettings()
    phase1_v44: Phase1V44Settings | None = None

    @model_validator(mode="after")
    def validate_reasoning_provider(self) -> "RunConfig":
        if self.run.contract_version == "v4.4":
            if self.run.mode != "phase1_only":
                raise ValueError("v4.4 requires phase1_only mode")
            if self.phase1_v44 is None:
                raise ValueError("v4.4 requires phase1_v44 settings")
        elif self.phase1_v44 is not None:
            raise ValueError("phase1_v44 requires contract_version=v4.4")

        configured_phases = [
            name
            for name, phase in (("phase1", self.phase1), ("phase2", self.phase2))
            if phase.reasoning_effort is not None
            or phase.planning_reasoning_effort is not None
        ]
        if configured_phases and self.provider.kind != "openrouter":
            phases = ", ".join(configured_phases)
            raise ValueError(
                f"reasoning_effort is only supported with provider.kind=openrouter "
                f"(configured in {phases})"
            )
        if (
            self.precedents.enabled
            and self.embedding is not None
            and self.embedding.kind not in {"ollama", "sentence_transformers"}
        ):
            raise ValueError(
                "precedent embeddings must be local: configure kind=sentence_transformers or ollama "
                "or omit [embedding] for lexical-only retrieval"
            )
        if self.portfolio_memory.enabled and (
            self.embedding is None
            or self.embedding.kind not in {"ollama", "sentence_transformers"}
        ):
            raise ValueError(
                "portfolio memory requires a local ollama or sentence_transformers embedding configuration"
            )
        if self.portfolio_memory.enabled and self.run.contract_version not in {
            "v4.1",
            "v4.2",
            "v4.3",
            "v4.4",
            "v5",
        }:
            raise ValueError(
                "portfolio memory requires contract_version=v4.1, v4.2, v4.3, v4.4, or v5"
            )
        return self


def load_config(path: Path) -> RunConfig:
    """Load strict TOML configuration."""
    candidate = Path(path)
    try:
        with candidate.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot load configuration: {candidate}") from exc
    return RunConfig.model_validate(raw)
