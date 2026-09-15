"""Provider-neutral structured generation contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field


_NULLABLE_SCALAR_TYPES = {"boolean", "integer", "number", "string"}


def strict_provider_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a provider-safe strict JSON Schema without changing the caller's value."""
    normalized = deepcopy(schema)

    def normalize(value: object, path: tuple[object, ...]) -> object:
        if isinstance(value, list):
            return [normalize(item, (*path, index)) for index, item in enumerate(value)]
        if not isinstance(value, dict):
            return value

        node = {
            key: nested
            for key, nested in value.items()
            if key not in {"default", "uniqueItems", "minLength", "maxLength"}
        }
        if "anyOf" in node:
            alternatives = node.pop("anyOf")
            if not isinstance(alternatives, list) or len(alternatives) != 2:
                raise ValueError(
                    f"unsupported provider schema union at {path}: anyOf must be "
                    "one scalar schema plus null"
                )
            nulls = [
                alternative
                for alternative in alternatives
                if isinstance(alternative, dict) and alternative.get("type") == "null"
            ]
            non_nulls = [
                alternative for alternative in alternatives if alternative not in nulls
            ]
            if (
                len(nulls) != 1
                or len(non_nulls) != 1
                or not isinstance(non_nulls[0], dict)
                or non_nulls[0].get("type") not in _NULLABLE_SCALAR_TYPES
            ):
                raise ValueError(
                    f"unsupported provider schema union at {path}: anyOf must be "
                    "one scalar schema plus null"
                )
            scalar = normalize(non_nulls[0], (*path, "anyOf", "scalar"))
            assert isinstance(scalar, dict)
            scalar_type = scalar.pop("type")
            node.update(scalar)
            node["type"] = [scalar_type, "null"]

        node = {
            key: normalize(nested, (*path, key)) for key, nested in node.items()
        }
        if node.get("type") == "object" or "properties" in node:
            properties = node.get("properties", {})
            if not isinstance(properties, dict):
                raise ValueError(f"invalid object properties at {path}")
            node["additionalProperties"] = False
            node["required"] = list(properties)
        return node

    result = normalize(normalized, ())
    if not isinstance(result, dict):  # pragma: no cover - protected by the public type
        raise ValueError("provider schema must be an object")
    return result


@dataclass(frozen=True)
class GenerationRequest:
    phase: str
    prompt: str
    schema: dict[str, Any]
    max_output_tokens: int | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", strict_provider_schema(self.schema))
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.reasoning_effort not in {None, "low", "medium", "high"}:
            raise ValueError("reasoning_effort must be low, medium, or high")


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)
    input_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)


class GenerationResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    parsed: dict[str, Any] | None
    content: str
    usage: Usage
    elapsed_seconds: float = Field(ge=0)
    raw_metadata: dict[str, Any]


class GenerationProvider(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...

    @property
    def metadata(self) -> dict[str, Any]: ...


class ModelProvider(GenerationProvider, EmbeddingProvider, Protocol):
    pass


class CompositeProvider:
    def __init__(
        self, generator: GenerationProvider, embedder: EmbeddingProvider
    ) -> None:
        self.generator = generator
        self.embedder = embedder

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return self.generator.generate(request)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embedder.embed(texts)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        method = getattr(self.embedder, "embed_documents", self.embedder.embed)
        return method(texts)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        method = getattr(self.embedder, "embed_queries", self.embedder.embed)
        return method(texts)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(getattr(self.embedder, "metadata", {}))


def embed_documents(
    provider: EmbeddingProvider, texts: Sequence[str]
) -> list[list[float]]:
    """Encode corpus documents while retaining legacy provider compatibility."""
    method = getattr(provider, "embed_documents", provider.embed)
    return method(texts)


def embed_queries(
    provider: EmbeddingProvider, texts: Sequence[str]
) -> list[list[float]]:
    """Encode search queries while retaining legacy provider compatibility."""
    method = getattr(provider, "embed_queries", provider.embed)
    return method(texts)


def embedding_identity(provider: object | None) -> dict[str, Any]:
    """Return stable configuration identity without provider credentials."""
    if provider is None:
        return {
            "backend": "none",
            "model": None,
            "revision": None,
            "normalize": False,
            "document_prefix": "",
            "query_prefix": "",
        }
    metadata = dict(getattr(provider, "metadata", {}))
    fallback = f"{provider.__class__.__module__}.{provider.__class__.__qualname__}"
    return {
        "backend": metadata.get("backend", fallback),
        "model": metadata.get("model"),
        "revision": metadata.get("revision"),
        "normalize": bool(metadata.get("normalize", False)),
        "document_prefix": metadata.get("document_prefix", ""),
        "query_prefix": metadata.get("query_prefix", ""),
    }
