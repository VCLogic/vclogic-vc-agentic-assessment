"""Source-bound temporal portfolio memory and company-level hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Literal, Sequence
import unicodedata

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rank_bm25 import BM25Okapi

from .providers.base import embedding_identity


_TOKEN = re.compile(r"[a-z0-9][a-z0-9_-]*")
_DISCLOSURE_SCHEMA = "portfolio-disclosure-v1"
_INDEX_SCHEMA = "portfolio-memory-index-v1"


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DisclosureEvidence(FrozenModel):
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    turn_index: int = Field(ge=0)
    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)


class PortfolioDisclosure(FrozenModel):
    schema_version: Literal["portfolio-disclosure-v1"] = _DISCLOSURE_SCHEMA
    disclosure_id: str = Field(pattern=r"^PM-[0-9a-f]{20}$")
    vc_slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    company_name: str | None = None
    aliases: tuple[str, ...] = ()
    descriptor: str = Field(min_length=1)
    relationship: Literal[
        "investment", "former_investment", "board_role", "eir",
        "indirect_holding", "uncertain",
    ]
    observed_overlap: Literal[
        "direct", "possible", "complementary", "disclosure_only", "unknown"
    ]
    observed_consequence: Literal[
        "out", "conditional_in", "permission_or_check_required", "no_effect", "unclear"
    ]
    source_episode_slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    source_episode_number: int | None = Field(default=None, ge=0)
    evidence: tuple[DisclosureEvidence, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    validation_status: Literal["automated_candidate", "human_validated"]

    @model_validator(mode="after")
    def validate_identity(self) -> "PortfolioDisclosure":
        if self.company_name is not None and not self.company_name.strip():
            raise ValueError("company name cannot be blank")
        if self.company_name is None and not self.descriptor.strip():
            raise ValueError("anonymous holding requires a descriptor")
        if self.company_name and not self.aliases:
            raise ValueError("named holding requires at least one alias")
        prefix = self.source_episode_slug.split("-", 1)[0]
        expected = int(prefix) if prefix.isdecimal() else None
        if self.source_episode_number != expected:
            raise ValueError("source episode number must match the numeric slug prefix")
        return self


class PortfolioEntity(FrozenModel):
    entity_id: str = Field(pattern=r"^PE-[0-9a-f]{20}$")
    company_name: str | None
    aliases: tuple[str, ...]
    descriptor: str
    search_text: str
    disclosure_ids: tuple[str, ...] = Field(min_length=1)
    earliest_episode_number: int = Field(ge=0)
    embedding: tuple[float, ...] | None = None


class PortfolioCandidate(FrozenModel):
    entity_id: str
    company_name: str | None
    aliases: tuple[str, ...]
    descriptor: str
    disclosure_ids: tuple[str, ...]
    score: float
    retrieval_modes: tuple[str, ...]


class FilteredPortfolioManifest(FrozenModel):
    schema_version: Literal["portfolio-filtered-manifest-v1"] = "portfolio-filtered-manifest-v1"
    target_episode_slug: str
    target_episode_number: int | None
    eligible_disclosure_ids: tuple[str, ...]
    excluded_current_or_later_count: int = Field(ge=0)
    excluded_unordered_count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FilteredPortfolioCorpus:
    vc_slug: str
    target_episode_slug: str
    target_episode_number: int | None
    disclosures: tuple[PortfolioDisclosure, ...]
    excluded_current_or_later_count: int
    excluded_unordered_count: int

    def manifest(self) -> FilteredPortfolioManifest:
        payload = {
            "target_episode_slug": self.target_episode_slug,
            "target_episode_number": self.target_episode_number,
            "eligible_disclosure_ids": [row.disclosure_id for row in self.disclosures],
            "excluded_current_or_later_count": self.excluded_current_or_later_count,
            "excluded_unordered_count": self.excluded_unordered_count,
        }
        digest = sha256(_canonical_bytes(payload)).hexdigest()
        return FilteredPortfolioManifest(**payload, sha256=digest)


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def _normal_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def make_disclosure_id(vc_slug: str, episode_slug: str, turn_index: int, identity: str) -> str:
    material = f"{vc_slug}\0{episode_slug}\0{turn_index}\0{_normal_name(identity)}".encode()
    return "PM-" + sha256(material).hexdigest()[:20]


class PortfolioMemoryCorpus:
    def __init__(self, vc_slug: str, disclosures: Iterable[PortfolioDisclosure]):
        materialized = tuple(disclosures)
        if any(row.vc_slug != vc_slug for row in materialized):
            raise ValueError("portfolio disclosure VC mismatch")
        ids = [row.disclosure_id for row in materialized]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate portfolio disclosure ID")
        spans: set[tuple[str, int, str, str]] = set()
        for row in materialized:
            for evidence in row.evidence:
                identity = _normal_name(row.company_name or row.descriptor)
                key = (evidence.source_sha256, evidence.turn_index, evidence.text, identity)
                if key in spans:
                    raise ValueError("duplicate source span across portfolio disclosures")
                spans.add(key)
        self.vc_slug = vc_slug
        self.disclosures = tuple(
            sorted(
                materialized,
                key=lambda row: (
                    row.source_episode_number is None,
                    row.source_episode_number if row.source_episode_number is not None else 10**9,
                    row.source_episode_slug,
                    row.disclosure_id,
                ),
            )
        )

    def eligible_for_target(self, target_slug: str) -> FilteredPortfolioCorpus:
        prefix = target_slug.split("-", 1)[0]
        if not prefix.isdecimal():
            return FilteredPortfolioCorpus(
                vc_slug=self.vc_slug,
                target_episode_slug=target_slug,
                target_episode_number=None,
                disclosures=(),
                excluded_current_or_later_count=sum(
                    row.source_episode_number is not None for row in self.disclosures
                ),
                excluded_unordered_count=sum(
                    row.source_episode_number is None for row in self.disclosures
                ),
            )
        target = int(prefix)
        eligible = tuple(
            row for row in self.disclosures
            if row.source_episode_number is not None and row.source_episode_number < target
        )
        return FilteredPortfolioCorpus(
            vc_slug=self.vc_slug,
            target_episode_slug=target_slug,
            target_episode_number=target,
            disclosures=eligible,
            excluded_current_or_later_count=sum(
                row.source_episode_number is not None and row.source_episode_number >= target
                for row in self.disclosures
            ),
            excluded_unordered_count=sum(row.source_episode_number is None for row in self.disclosures),
        )

    def save_events(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(
            json.dumps(row.model_dump(mode="json"), sort_keys=True, ensure_ascii=False) + "\n"
            for row in self.disclosures
        )
        path.write_text(content, encoding="utf-8")

    @classmethod
    def load_events(cls, vc_slug: str, path: Path) -> "PortfolioMemoryCorpus":
        rows = [
            PortfolioDisclosure.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(vc_slug, rows)


def _embed_documents(provider: Any, texts: Sequence[str]) -> list[list[float]]:
    method = getattr(provider, "embed_documents", None) or getattr(provider, "embed")
    return method(texts)


def _embed_queries(provider: Any, texts: Sequence[str]) -> list[list[float]]:
    method = getattr(provider, "embed_queries", None) or getattr(provider, "embed")
    return method(texts)


def _entity_id(key: str) -> str:
    return "PE-" + sha256(key.encode()).hexdigest()[:20]


def _aggregate(
    disclosures: Sequence[PortfolioDisclosure],
    embeddings: dict[str, tuple[float, ...] | None],
) -> tuple[PortfolioEntity, ...]:
    groups: dict[str, list[PortfolioDisclosure]] = {}
    for row in disclosures:
        key = f"named:{_normal_name(row.company_name)}" if row.company_name else f"anonymous:{row.disclosure_id}"
        groups.setdefault(key, []).append(row)
    entities: list[PortfolioEntity] = []
    for key, rows in sorted(groups.items()):
        names = [row.company_name for row in rows if row.company_name]
        company_name = names[0] if names else None
        aliases = tuple(dict.fromkeys(alias for row in rows for alias in row.aliases))
        descriptors = tuple(dict.fromkeys(row.descriptor for row in rows))
        search_text = " ".join(
            [*(aliases or (() if company_name is None else (company_name,))), *descriptors,
             *(row.observed_overlap for row in rows), *(row.relationship for row in rows)]
        )
        vectors = [embeddings[row.disclosure_id] for row in rows if embeddings.get(row.disclosure_id)]
        mean_vector: tuple[float, ...] | None = None
        if vectors:
            mean = np.mean(np.asarray(vectors, dtype=float), axis=0)
            norm = np.linalg.norm(mean)
            if norm:
                mean = mean / norm
            mean_vector = tuple(float(value) for value in mean)
        entities.append(
            PortfolioEntity(
                entity_id=_entity_id(f"{rows[0].vc_slug}\0{key}"),
                company_name=company_name,
                aliases=aliases,
                descriptor="; ".join(descriptors),
                search_text=search_text,
                disclosure_ids=tuple(row.disclosure_id for row in rows),
                earliest_episode_number=min(row.source_episode_number for row in rows if row.source_episode_number is not None),
                embedding=mean_vector,
            )
        )
    return tuple(entities)


class FilteredPortfolioIndex:
    def __init__(self, filtered: FilteredPortfolioCorpus, entities: Sequence[PortfolioEntity], embedder: Any):
        self.filtered = filtered
        self.entities = tuple(entities)
        self.embedder = embedder
        self._bm25 = BM25Okapi([_tokens(row.search_text) for row in self.entities]) if self.entities else None

    def search(self, query: str, *, limit: int, candidate_pool_k: int | None = None) -> list[PortfolioCandidate]:
        if not query.strip() or limit < 1:
            raise ValueError("portfolio search requires a query and positive limit")
        if not self.entities:
            return []
        pool = min(candidate_pool_k or len(self.entities), len(self.entities))
        lexical_scores = self._bm25.get_scores(_tokens(query)) if self._bm25 is not None else np.zeros(len(self.entities))
        lexical_order = list(np.argsort(-lexical_scores, kind="stable")[:pool])
        rankings = [lexical_order]
        dense_order: list[int] = []
        dense_rows = [(index, row) for index, row in enumerate(self.entities) if row.embedding]
        if dense_rows:
            query_vector = np.asarray(_embed_queries(self.embedder, [query])[0], dtype=float)
            matrix = np.asarray([row.embedding for _, row in dense_rows], dtype=float)
            denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
            similarities = np.divide(matrix @ query_vector, denominator, out=np.zeros(len(matrix)), where=denominator != 0)
            dense_order = [dense_rows[index][0] for index in np.argsort(-similarities, kind="stable")[:pool]]
            rankings.append(dense_order)
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, index in enumerate(ranking, start=1):
                fused[index] = fused.get(index, 0.0) + 1.0 / (60 + rank)
        ordered = sorted(fused, key=lambda index: (-fused[index], self.entities[index].entity_id))[:limit]
        return [
            PortfolioCandidate(
                entity_id=self.entities[index].entity_id,
                company_name=self.entities[index].company_name,
                aliases=self.entities[index].aliases,
                descriptor=self.entities[index].descriptor,
                disclosure_ids=self.entities[index].disclosure_ids,
                score=fused[index],
                retrieval_modes=tuple(
                    mode for mode, ranking in (("lexical", lexical_order), ("dense", dense_order)) if index in ranking
                ),
            )
            for index in ordered
        ]


class PortfolioMemoryIndex:
    def __init__(self, corpus: PortfolioMemoryCorpus, event_embeddings: dict[str, tuple[float, ...] | None], embedder: Any, metadata: dict[str, Any]):
        self.corpus = corpus
        self.event_embeddings = event_embeddings
        self.embedder = embedder
        self.metadata = metadata

    @classmethod
    def build(cls, corpus: PortfolioMemoryCorpus, embedder: Any) -> "PortfolioMemoryIndex":
        texts = [
            " ".join(filter(None, [row.company_name or "", " ".join(row.aliases), row.descriptor, row.relationship, row.observed_overlap]))
            for row in corpus.disclosures
        ]
        vectors = _embed_documents(embedder, texts) if texts else []
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise ValueError("portfolio embedding coverage is incomplete")
        embeddings = {row.disclosure_id: tuple(float(value) for value in vector) for row, vector in zip(corpus.disclosures, vectors, strict=True)}
        metadata = {**embedding_identity(embedder), "embedded_count": len(embeddings), "total_count": len(corpus.disclosures)}
        return cls(corpus, embeddings, embedder, metadata)

    def for_target(self, target_slug: str) -> FilteredPortfolioIndex:
        filtered = self.corpus.eligible_for_target(target_slug)
        entities = _aggregate(filtered.disclosures, self.event_embeddings)
        return FilteredPortfolioIndex(filtered, entities, self.embedder)

    def save(self, path: Path, corpus_path: Path) -> None:
        payload = {
            "schema": _INDEX_SCHEMA,
            "vc_slug": self.corpus.vc_slug,
            "corpus_sha256": sha256(corpus_path.read_bytes()).hexdigest(),
            "embedding": self.metadata,
            "event_embeddings": {key: list(value) if value is not None else None for key, value in sorted(self.event_embeddings.items())},
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, corpus: PortfolioMemoryCorpus, embedder: Any, corpus_path: Path) -> "PortfolioMemoryIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != _INDEX_SCHEMA or payload.get("vc_slug") != corpus.vc_slug:
            raise ValueError("portfolio index identity is invalid")
        if payload.get("corpus_sha256") != sha256(corpus_path.read_bytes()).hexdigest():
            raise ValueError("portfolio corpus hash mismatch")
        expected_identity = embedding_identity(embedder)
        if any(payload.get("embedding", {}).get(key) != value for key, value in expected_identity.items()):
            raise ValueError("portfolio embedding identity mismatch")
        raw = payload.get("event_embeddings")
        if not isinstance(raw, dict) or set(raw) != {row.disclosure_id for row in corpus.disclosures}:
            raise ValueError("portfolio embedding coverage mismatch")
        embeddings = {key: tuple(float(value) for value in vector) if vector is not None else None for key, vector in raw.items()}
        if any(vector is None for vector in embeddings.values()):
            raise ValueError("portfolio embedding coverage is incomplete")
        return cls(corpus, embeddings, embedder, dict(payload["embedding"]))
