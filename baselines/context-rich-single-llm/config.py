"""Configuration for the context-rich single-model baseline."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictFrozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderConfig(StrictFrozen):
    model: str = "openai/gpt-5.6-luna"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    reasoning_effort: Literal["low", "medium", "high"] = "high"
    max_output_tokens: int = Field(default=16384, ge=256, le=16384)
    require_parameters: bool = True
    data_collection: Literal["allow", "deny"] = "deny"


class RetrievalConfig(StrictFrozen):
    top_k: int = 5
    candidate_pool_k: int = Field(default=40, ge=5, le=100)
    selection_policy: Literal["semantic"] = "semantic"

    @model_validator(mode="after")
    def exactly_five(self):
        if self.top_k != 5:
            raise ValueError("baseline requires exactly five precedents")
        return self


class EmbeddingConfig(StrictFrozen):
    kind: Literal["sentence_transformers"] = "sentence_transformers"
    model: str
    revision: str
    device: str = "auto"
    batch_size: int = 16
    normalize: bool = True
    document_prefix: str = "search_document: "
    query_prefix: str = "search_query: "
    require_complete_index: bool = True


class BaselineConfig(StrictFrozen):
    input_root: str = "inputs"
    vc_slug: str
    taxonomy_path: str = "taxonomy/codebook_v_final.json"
    output_root: str
    provider: ProviderConfig
    retrieval: RetrievalConfig
    embedding: EmbeddingConfig


def load_baseline_config(path: Path) -> BaselineConfig:
    with Path(path).open("rb") as stream:
        return BaselineConfig.model_validate(tomllib.load(stream))
