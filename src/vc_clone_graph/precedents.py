"""Persistent, source-addressable retrieval over historical precedents."""

from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite, log, sqrt
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Literal, Protocol, Sequence, overload
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .providers.base import embed_documents, embed_queries, embedding_identity


_TOKEN = re.compile(r"[a-z0-9][a-z0-9_-]*")
_CHUNK_TURNS = 8
_INDEX_SCHEMA = "precedent-hybrid-index-v2"
_LEGACY_INDEX_SCHEMA = "precedent-hybrid-index-v1"
_EMBEDDING_FALLBACK = "PRECEDENT_EMBEDDING_FALLBACK"


class Embedder(Protocol):
    """A local-only embedding provider compatible with the wiki retriever."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TranscriptTurn(FrozenModel):
    turn_index: int = Field(ge=0)
    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)


class DecisionEvidence(FrozenModel):
    turn_start: int = Field(ge=0)
    turn_end: int = Field(ge=0)
    text: str = Field(min_length=1)


class PrecedentDecision(FrozenModel):
    status: Literal["In", "Out", "unobserved"]
    context: Literal[
        "initial_panel",
        "same_session_reversal",
        "later_diligence",
        "off_panel",
        "unclear",
    ]
    check_tier: str | None
    conditions: tuple[str, ...]
    evidence: tuple[DecisionEvidence, ...]
    audit_source: str = Field(min_length=1)
    audit_notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence_presence(self) -> PrecedentDecision:
        if self.status != "unobserved" and not self.evidence:
            raise ValueError("observed decision requires exact decision evidence")
        if self.status == "unobserved" and self.evidence:
            raise ValueError("unobserved decision cannot contain decision evidence")
        return self


class PrecedentEpisode(FrozenModel):
    episode_slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    episode_number: int | None = Field(ge=0)
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    investor_aliases: tuple[str, ...]
    investor_present: bool
    turns: tuple[TranscriptTurn, ...]
    decision: PrecedentDecision

    @model_validator(mode="after")
    def bind_to_source_turns(self) -> PrecedentEpisode:
        prefix = self.episode_slug.split("-", 1)[0]
        slug_number = int(prefix) if prefix.isdecimal() else None
        if self.episode_number != slug_number:
            raise ValueError(
                "episode number must equal the numeric episode slug prefix"
            )
        expected_indices = tuple(range(len(self.turns)))
        if tuple(turn.turn_index for turn in self.turns) != expected_indices:
            raise ValueError("transcript turn indices must be contiguous")
        for span in self.decision.evidence:
            if span.turn_start > span.turn_end or span.turn_end >= len(self.turns):
                raise ValueError("decision evidence turn range is invalid")
            exact = "\n".join(
                turn.text
                for turn in self.turns[span.turn_start : span.turn_end + 1]
            )
            if span.text != exact:
                raise ValueError("decision evidence does not equal source turns")
        return self


class EpisodeSummary(FrozenModel):
    episode_slug: str
    episode_number: int | None
    investor_present: bool
    decision_status: Literal["In", "Out", "unobserved"]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FilteredPrecedentManifest(FrozenModel):
    schema_version: Literal["precedent-filtered-manifest-v1"]
    target_episode_slug: str
    accessible_episodes: tuple[EpisodeSummary, ...]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HistoricalEvidence(FrozenModel):
    evidence_id: str = Field(pattern=r"^H-[0-9a-f]{20}$")
    episode_slug: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    turn_start: int = Field(ge=0)
    turn_end: int = Field(ge=0)
    text: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    citation_mode: Literal["exact"] = "exact"


class PrecedentRead(FrozenModel):
    episode_slug: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    turn_start: int = Field(ge=0)
    turn_end: int = Field(ge=0)
    turns: tuple[TranscriptTurn, ...]
    evidence: HistoricalEvidence | None = None


class DecisionRead(FrozenModel):
    episode_slug: str
    decision: PrecedentDecision
    evidence: tuple[HistoricalEvidence, ...]


class PrecedentSearchHit(FrozenModel):
    episode_slug: str
    chunk_id: str = Field(pattern=r"^H-[0-9a-f]{20}$")
    turn_start: int = Field(ge=0)
    turn_end: int = Field(ge=0)
    excerpt: str
    score: float
    retrieval_modes: tuple[str, ...] = ("lexical",)


class PrecedentSearchResult(FrozenModel):
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    hits: tuple[PrecedentSearchHit, ...]
    quality_findings: tuple[str, ...]


class _PrecedentChunk(FrozenModel):
    chunk_id: str = Field(pattern=r"^H-[0-9a-f]{20}$")
    episode_slug: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    turn_start: int = Field(ge=0)
    turn_end: int = Field(ge=0)
    text: str = Field(min_length=1)
    tokens: tuple[str, ...]
    embedding: tuple[float, ...] | None


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _identity(source_sha256: str, start: int, end: int, text: str) -> str:
    material = f"{source_sha256}\0{start}\0{end}\0{text}".encode("utf-8")
    return "H-" + sha256(material).hexdigest()[:20]


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


_MODEL_VISIBLE_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _normalized_model_visible_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(_MODEL_VISIBLE_TOKEN.findall(normalized))


def _model_visible_episode_text(row: PrecedentEpisode) -> tuple[str, ...]:
    """Enumerate every row-derived string exposed by search or exact opens."""
    payload = {
        "transcript": {
            "episode_slug": row.episode_slug,
            "source_sha256": row.source_sha256,
            "turns": [turn.model_dump(mode="json") for turn in row.turns],
        },
        "decision": {
            "episode_slug": row.episode_slug,
            "decision": row.decision.model_dump(mode="json"),
        },
    }

    def strings(value: object) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value,)
        if isinstance(value, dict):
            return tuple(
                item for nested in value.values() for item in strings(nested)
            )
        if isinstance(value, (list, tuple)):
            return tuple(item for nested in value for item in strings(nested))
        return ()

    return strings(payload)


def _contains_normalized_alias(
    value: str,
    alias_tokens: tuple[str, ...],
    alias_literal: str,
) -> bool:
    value_tokens = _normalized_model_visible_tokens(value)
    if alias_tokens:
        width = len(alias_tokens)
        return any(
            value_tokens[index : index + width] == alias_tokens
            for index in range(len(value_tokens) - width + 1)
        )
    normalized = " ".join(
        unicodedata.normalize("NFKC", value).casefold().split()
    )
    return bool(alias_literal and alias_literal in normalized)


def _episode_digest(row: PrecedentEpisode) -> str:
    return sha256(_canonical_bytes(row.model_dump(mode="json"))).hexdigest()


def _safe_corpus_file(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a safe corpus-relative file path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError(f"{label} must be a safe corpus-relative file path")
    try:
        candidate = (root / relative).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a safe corpus-relative file path"
        ) from exc
    if not candidate.is_file():
        raise ValueError(f"{label} must be a safe corpus-relative file path")
    return candidate


def _validated_vectors(
    vectors: object, expected_count: int
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(vectors, Sequence) or len(vectors) != expected_count:
        raise ValueError("embedding count is invalid")
    normalized: list[tuple[float, ...]] = []
    dimension: int | None = None
    for raw in vectors:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
            raise ValueError("embedding vector is invalid")
        vector = tuple(float(value) for value in raw)
        if not all(isfinite(value) for value in vector):
            raise ValueError("embedding vector is invalid")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError("embedding dimensions are inconsistent")
        normalized.append(vector)
    return tuple(normalized)


def _rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


class PrecedentCorpus:
    """A target-filtered corpus with immutable source-bound retrieval chunks."""

    def __init__(
        self,
        episodes: Sequence[PrecedentEpisode],
        target_slug: str | None = None,
        embedder: Embedder | None = None,
        *,
        _chunks: Sequence[_PrecedentChunk] | None = None,
        _embedding_fallback: bool = False,
        _corpus_root: Path | None = None,
        _manifest_sha256: str | None = None,
        _record_sha256s: dict[str, str] | None = None,
        _embedding_index: dict[str, object] | None = None,
        require_complete_embeddings: bool = False,
    ) -> None:
        rows = tuple(episodes)
        by_slug = {row.episode_slug: row for row in rows}
        if len(by_slug) != len(rows):
            raise ValueError("precedent episode slugs must be unique")
        self._episodes = {
            slug: row for slug, row in by_slug.items() if slug != target_slug
        }
        self.target_slug = target_slug
        self.embedder = embedder
        self.require_complete_embeddings = require_complete_embeddings
        self._corpus_root = _corpus_root.resolve() if _corpus_root else None
        self._manifest_sha256 = _manifest_sha256
        supplied_hashes = _record_sha256s or {}
        self._record_sha256s = {
            slug: supplied_hashes.get(slug, _episode_digest(row))
            for slug, row in self._episodes.items()
        }
        self._canonical_episode_sha256s = {
            slug: _episode_digest(row) for slug, row in self._episodes.items()
        }
        if _chunks is None:
            self._chunks, failed = self._build_chunks()
            self._embedding_fallback = _embedding_fallback or failed
            if failed and require_complete_embeddings:
                raise RuntimeError("cannot build complete precedent embedding index")
        else:
            self._chunks = tuple(_chunks)
            self._embedding_fallback = _embedding_fallback
            self._validate_chunk_consistency()
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        if len(self._chunks_by_id) != len(self._chunks):
            raise ValueError("precedent index chunk IDs are not unique")
        self.embedding_index = _embedding_index or self._embedding_metadata()
        self._validate_embedding_index()

    @classmethod
    def from_episodes(
        cls,
        episodes: Sequence[PrecedentEpisode],
        embedder: Embedder | None = None,
        *,
        target_slug: str | None = None,
        require_complete_embeddings: bool = False,
    ) -> PrecedentCorpus:
        return cls(
            episodes,
            target_slug=target_slug,
            embedder=embedder,
            require_complete_embeddings=require_complete_embeddings,
        )

    @classmethod
    def build(
        cls,
        corpus_root: str | Path,
        embedder: Embedder | None = None,
        *,
        target_slug: str | None = None,
        require_complete_embeddings: bool = False,
    ) -> PrecedentCorpus:
        """Build an index from a builder corpus after verifying every record."""
        root = Path(corpus_root).resolve()
        episodes, manifest_digest, record_hashes = cls._read_corpus(root)
        return cls(
            episodes,
            target_slug=target_slug,
            embedder=embedder,
            _corpus_root=root,
            _manifest_sha256=manifest_digest,
            _record_sha256s=record_hashes,
            require_complete_embeddings=require_complete_embeddings,
        )

    def for_target(
        self, target_slug: str, *, excluded_aliases: Sequence[str] = ()
    ) -> PrecedentCorpus:
        aliases = tuple(
            (
                _normalized_model_visible_tokens(alias),
                " ".join(unicodedata.normalize("NFKC", alias).casefold().split()),
            )
            for alias in excluded_aliases
            if alias.strip()
        )

        def is_safe(row: PrecedentEpisode) -> bool:
            if row.episode_slug == target_slug:
                return False
            if not aliases:
                return True
            return not any(
                _contains_normalized_alias(value, alias_tokens, alias_literal)
                for value in _model_visible_episode_text(row)
                for alias_tokens, alias_literal in aliases
            )

        accessible = tuple(
            row for row in self._episodes.values() if is_safe(row)
        )
        accessible_slugs = {row.episode_slug for row in accessible}
        return PrecedentCorpus(
            accessible,
            target_slug,
            self.embedder,
            _chunks=tuple(
                chunk
                for chunk in self._chunks
                if chunk.episode_slug in accessible_slugs
            ),
            _embedding_fallback=self._embedding_fallback,
            _corpus_root=self._corpus_root,
            _manifest_sha256=self._manifest_sha256,
            _record_sha256s=self._record_sha256s,
            require_complete_embeddings=self.require_complete_embeddings,
        )

    def _embedding_metadata(self) -> dict[str, object]:
        identity = embedding_identity(self.embedder)
        embedded = [chunk.embedding for chunk in self._chunks if chunk.embedding is not None]
        dimension = len(embedded[0]) if embedded else 0
        return {
            **identity,
            "dimension": dimension,
            "embedded_count": len(embedded),
            "total_count": len(self._chunks),
            "coverage": len(embedded) / len(self._chunks) if self._chunks else 0.0,
        }

    def _validate_embedding_index(self) -> None:
        actual = self._embedding_metadata()
        for field in ("dimension", "embedded_count", "total_count", "coverage"):
            if self.embedding_index.get(field) != actual[field]:
                raise ValueError("precedent embedding index metadata is inconsistent")
        if self.require_complete_embeddings:
            if actual["coverage"] != 1.0 or actual["dimension"] == 0:
                raise ValueError("complete precedent embedding index is required")
            identity_fields = (
                "backend", "model", "revision", "normalize",
                "document_prefix", "query_prefix",
            )
            if any(
                self.embedding_index.get(field) != actual[field]
                for field in identity_fields
            ):
                raise ValueError("precedent embedding configuration mismatch")

    def _get(self, slug: str) -> PrecedentEpisode:
        if slug == self.target_slug:
            raise PermissionError(f"target episode is inaccessible: {slug}")
        try:
            return self._episodes[slug]
        except KeyError as exc:
            raise KeyError(f"unknown precedent episode: {slug}") from exc

    def list_episodes(self) -> tuple[EpisodeSummary, ...]:
        return tuple(
            EpisodeSummary(
                episode_slug=row.episode_slug,
                episode_number=row.episode_number,
                investor_present=row.investor_present,
                decision_status=row.decision.status,
                source_sha256=row.source_sha256,
            )
            for row in self._ordered_episodes()
        )

    def question_archetypes(self, investor_aliases: Sequence[str]):
        """Return decision-excluded questions from this already-filtered corpus."""
        from .question_memory import extract_question_archetypes

        return extract_question_archetypes(
            self._ordered_episodes(), investor_aliases=set(investor_aliases)
        )

    def filtered_manifest(self) -> FilteredPrecedentManifest:
        if self.target_slug is None:
            raise ValueError("filtered manifest requires a target-filtered corpus")
        payload = {
            "schema_version": "precedent-filtered-manifest-v1",
            "target_episode_slug": self.target_slug,
            "accessible_episodes": [
                row.model_dump(mode="json") for row in self.list_episodes()
            ],
        }
        digest = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return FilteredPrecedentManifest(**payload, sha256=digest)

    def open_transcript(
        self,
        slug: str,
        start: int = 0,
        end: int | None = None,
        *,
        query_id: str | None = None,
    ) -> PrecedentRead:
        row = self._get(slug)
        self._verify_record(row)
        stop = len(row.turns) - 1 if end is None else end
        if start < 0 or stop < start or stop >= len(row.turns):
            raise IndexError("invalid precedent transcript turn range")
        turns = row.turns[start : stop + 1]
        evidence = None
        if query_id is not None:
            text = "\n".join(turn.text for turn in turns)
            evidence = self._evidence(row, start, stop, text, query_id)
        return PrecedentRead(
            episode_slug=slug,
            source_sha256=row.source_sha256,
            turn_start=start,
            turn_end=stop,
            turns=turns,
            evidence=evidence,
        )

    @overload
    def open_decision(self, slug: str) -> PrecedentDecision: ...

    @overload
    def open_decision(self, slug: str, *, query_id: str) -> DecisionRead: ...

    def open_decision(
        self, slug: str, *, query_id: str | None = None
    ) -> PrecedentDecision | DecisionRead:
        row = self._get(slug)
        self._verify_record(row)
        if query_id is None:
            return row.decision
        evidence = tuple(
            self._evidence(
                row,
                span.turn_start,
                span.turn_end,
                span.text,
                query_id,
            )
            for span in row.decision.evidence
        )
        return DecisionRead(
            episode_slug=slug,
            decision=row.decision,
            evidence=evidence,
        )

    def evidence_text(self, evidence_id: str) -> str:
        """Return canonical indexed text for internal evidence identity checks."""
        chunk = self._chunks_by_id.get(evidence_id)
        if chunk is None:
            raise KeyError(f"unknown precedent evidence: {evidence_id}")
        self._verify_record(self._episodes[chunk.episode_slug])
        return chunk.text

    @overload
    def search(
        self,
        query: str,
        limit: int,
        *,
        top_k: None = None,
        query_id: None = None,
        selection_policy: Literal["semantic", "contrastive"] = "semantic",
        candidate_pool_k: int | None = None,
        in_slots: int = 2,
        out_slots: int = 2,
        observed_only: bool = False,
        substantive_only: bool = False,
    ) -> tuple[PrecedentSearchHit, ...]: ...

    @overload
    def search(
        self,
        query: str,
        limit: int | None = None,
        *,
        top_k: int | None = None,
        query_id: str,
        selection_policy: Literal["semantic", "contrastive"] = "semantic",
        candidate_pool_k: int | None = None,
        in_slots: int = 2,
        out_slots: int = 2,
        observed_only: bool = False,
        substantive_only: bool = False,
    ) -> PrecedentSearchResult: ...

    def search(
        self,
        query: str,
        limit: int | None = None,
        *,
        top_k: int | None = None,
        query_id: str | None = None,
        selection_policy: Literal["semantic", "contrastive"] = "semantic",
        candidate_pool_k: int | None = None,
        in_slots: int = 2,
        out_slots: int = 2,
        observed_only: bool = False,
        substantive_only: bool = False,
    ) -> tuple[PrecedentSearchHit, ...] | PrecedentSearchResult:
        requested = top_k if top_k is not None else limit
        if requested is None or requested < 1:
            raise ValueError("query and positive limit are required")
        if limit is not None and top_k is not None and limit != top_k:
            raise ValueError("limit and top_k disagree")
        query_tokens = _tokens(query)
        if not query_tokens:
            raise ValueError("query must contain at least one lexical token")
        if selection_policy not in {"semantic", "contrastive"}:
            raise ValueError("unknown precedent selection policy")
        pool_size = requested
        if selection_policy == "contrastive":
            if in_slots < 0 or out_slots < 0 or in_slots + out_slots < 1:
                raise ValueError("contrastive slots are invalid")
            pool_size = max(requested, candidate_pool_k or requested)
        eligible_chunks = tuple(
            chunk
            for chunk in self._chunks
            if self._chunk_is_eligible(
                chunk,
                observed_only=observed_only,
                substantive_only=substantive_only,
            )
        )
        hits, query_failed = self._rank(
            query,
            query_tokens,
            pool_size,
            chunks=eligible_chunks,
        )
        if selection_policy == "contrastive":
            hits = self._select_contrastive(
                hits,
                limit=requested,
                in_slots=in_slots,
                out_slots=out_slots,
            )
        if query_id is None:
            return hits
        findings = (
            (_EMBEDDING_FALLBACK,)
            if self._embedding_fallback or query_failed
            else ()
        )
        return PrecedentSearchResult(
            query_id=query_id,
            query=query,
            hits=hits,
            quality_findings=findings,
        )

    def _chunk_is_eligible(
        self,
        chunk: _PrecedentChunk,
        *,
        observed_only: bool,
        substantive_only: bool,
    ) -> bool:
        episode = self._episodes[chunk.episode_slug]
        if observed_only and episode.decision.status == "unobserved":
            return False
        if not substantive_only:
            return True
        if episode.decision.status == "unobserved":
            return False
        if any(
            evidence.turn_start <= chunk.turn_end
            and evidence.turn_end >= chunk.turn_start
            for evidence in episode.decision.evidence
        ):
            return True
        aliases = {alias.casefold().strip() for alias in episode.investor_aliases}
        for turn in episode.turns[chunk.turn_start : chunk.turn_end + 1]:
            if turn.speaker.casefold().strip() not in aliases:
                continue
            normalized = " ".join(
                turn.text.casefold().replace("’", "'").replace("‘", "'").split()
            )
            if len(_tokens(normalized)) < 5:
                continue
            if (
                (normalized.startswith("i'm ") or normalized.startswith("i am "))
                and any(
                    role in normalized
                    for role in (
                        "managing partner",
                        "general partner",
                        "founder and",
                        "co-founder and",
                    )
                )
            ):
                continue
            return True
        return False

    def _select_contrastive(
        self,
        hits: Sequence[PrecedentSearchHit],
        *,
        limit: int,
        in_slots: int,
        out_slots: int,
    ) -> tuple[PrecedentSearchHit, ...]:
        unique: list[PrecedentSearchHit] = []
        seen_episodes: set[str] = set()
        for hit in hits:
            if hit.episode_slug in seen_episodes:
                continue
            seen_episodes.add(hit.episode_slug)
            if self._episodes[hit.episode_slug].decision.status != "unobserved":
                unique.append(hit)

        ins = [
            hit
            for hit in unique
            if self._episodes[hit.episode_slug].decision.status == "In"
        ]
        outs = [
            hit
            for hit in unique
            if self._episodes[hit.episode_slug].decision.status == "Out"
        ]
        selected = {hit.episode_slug for hit in ins[:in_slots]}
        selected.update(hit.episode_slug for hit in outs[:out_slots])
        capacity = min(limit, in_slots + out_slots, len(unique))
        for hit in unique:
            if len(selected) >= capacity:
                break
            selected.add(hit.episode_slug)
        return tuple(hit for hit in unique if hit.episode_slug in selected)[:capacity]

    def save(self, path: str | Path) -> None:
        """Persist a self-checking index without embedding provider credentials."""
        payload: dict[str, Any] = {
            "schema": _INDEX_SCHEMA,
            "target_episode_slug": self.target_slug,
            "corpus_manifest_sha256": self._manifest_sha256,
            "record_sha256s": dict(sorted(self._record_sha256s.items())),
            "embedding_fallback": self._embedding_fallback,
            "embedding_index": self.embedding_index,
            "episodes": [
                row.model_dump(mode="json") for row in self._ordered_episodes()
            ],
            "chunks": [chunk.model_dump(mode="json") for chunk in self._chunks],
        }
        payload["index_sha256"] = sha256(_canonical_bytes(payload)).hexdigest()
        candidate = Path(path)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", dir=candidate.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_canonical_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, candidate)

    @classmethod
    def load(
        cls,
        path: str | Path,
        corpus_root: str | Path | None = None,
        embedder: Embedder | None = None,
        *,
        require_complete_embeddings: bool = False,
    ) -> PrecedentCorpus:
        """Load an index and reject index, manifest, or record tampering."""
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("precedent index is invalid or tampered") from exc
        schema = raw.get("schema") if isinstance(raw, dict) else None
        if schema not in {_INDEX_SCHEMA, _LEGACY_INDEX_SCHEMA}:
            raise ValueError("precedent index schema is invalid")
        if require_complete_embeddings and schema != _INDEX_SCHEMA:
            raise ValueError("complete precedent embedding index lacks verified metadata")
        claimed_digest = raw.pop("index_sha256", None)
        actual_digest = sha256(_canonical_bytes(raw)).hexdigest()
        if claimed_digest != actual_digest:
            raise ValueError("precedent index digest mismatch; index was tampered")
        try:
            episodes = tuple(
                PrecedentEpisode.model_validate(item) for item in raw["episodes"]
            )
            chunks = tuple(
                _PrecedentChunk.model_validate(item) for item in raw["chunks"]
            )
            record_hashes = dict(raw["record_sha256s"])
            target_slug = raw["target_episode_slug"]
            manifest_digest = raw["corpus_manifest_sha256"]
            embedding_fallback = bool(raw["embedding_fallback"])
            embedding_index = raw.get("embedding_index")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("precedent index consistency is invalid") from exc
        if target_slug is not None and any(
            row.episode_slug == target_slug for row in episodes
        ):
            raise ValueError("precedent index consistency includes target episode")
        episode_slugs = {row.episode_slug for row in episodes}
        if (
            not isinstance(target_slug, (str, type(None)))
            or set(record_hashes) != episode_slugs
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in record_hashes.values()
            )
        ):
            raise ValueError("precedent index record hash consistency is invalid")

        root = Path(corpus_root).resolve() if corpus_root is not None else None
        if manifest_digest is not None and root is None:
            raise ValueError("corpus root is required to verify manifest binding")
        if root is not None:
            backing, backing_manifest, backing_hashes = cls._read_corpus(root)
            backing_by_slug = {row.episode_slug: row for row in backing}
            if manifest_digest != backing_manifest:
                raise ValueError("corpus manifest digest mismatch")
            expected_slugs = set(backing_by_slug)
            if target_slug is not None:
                expected_slugs.discard(target_slug)
            if episode_slugs != expected_slugs:
                raise ValueError("precedent index episode slug consistency mismatch")
            for row in episodes:
                if (
                    backing_by_slug.get(row.episode_slug) != row
                    or backing_hashes.get(row.episode_slug)
                    != record_hashes.get(row.episode_slug)
                ):
                    raise ValueError("corpus record hash or index consistency mismatch")
        index = cls(
            episodes,
            target_slug=target_slug,
            embedder=embedder,
            _chunks=chunks,
            _embedding_fallback=embedding_fallback,
            _corpus_root=root,
            _manifest_sha256=manifest_digest,
            _record_sha256s=record_hashes,
            _embedding_index=embedding_index,
            require_complete_embeddings=require_complete_embeddings,
        )
        return index

    @classmethod
    def _read_corpus(
        cls, root: Path
    ) -> tuple[tuple[PrecedentEpisode, ...], str, dict[str, str]]:
        manifest_path = _safe_corpus_file(
            root, "corpus-manifest.json", "corpus manifest path"
        )
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("corpus manifest is invalid") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != "precedent-corpus-v1"
            or not isinstance(manifest.get("records"), list)
        ):
            raise ValueError("corpus manifest schema is invalid")
        episodes: list[PrecedentEpisode] = []
        record_hashes: dict[str, str] = {}
        for entry in manifest["records"]:
            try:
                record_path = _safe_corpus_file(
                    root, entry["record_path"], "record path"
                )
                payload = record_path.read_bytes()
                digest = sha256(payload).hexdigest()
                if digest != entry["record_sha256"]:
                    raise ValueError("corpus record hash mismatch")
                row = PrecedentEpisode.model_validate_json(payload)
                source_path = _safe_corpus_file(
                    root, entry["source_path"], "source path"
                )
                source_digest = sha256(source_path.read_bytes()).hexdigest()
                if (
                    row.episode_slug != entry["episode_slug"]
                    or row.source_path != entry["source_path"]
                    or row.source_sha256 != entry["source_sha256"]
                    or source_digest != entry["source_sha256"]
                    or row.decision.status != entry["decision_status"]
                ):
                    if source_digest != entry["source_sha256"]:
                        raise ValueError("corpus source snapshot hash mismatch")
                    raise ValueError("corpus manifest record consistency mismatch")
            except (KeyError, OSError, TypeError, ValueError) as exc:
                if isinstance(exc, ValueError):
                    message = str(exc)
                    if (
                        message.startswith("corpus")
                        or "safe corpus-relative" in message
                    ):
                        raise
                raise ValueError(
                    "corpus manifest record consistency is invalid"
                ) from exc
            if row.episode_slug in record_hashes:
                raise ValueError("corpus manifest episode slugs are not unique")
            episodes.append(row)
            record_hashes[row.episode_slug] = digest
        return (
            tuple(episodes),
            sha256(manifest_bytes).hexdigest(),
            record_hashes,
        )

    def _build_chunks(self) -> tuple[tuple[_PrecedentChunk, ...], bool]:
        pending: list[tuple[PrecedentEpisode, int, int, str, tuple[str, ...]]] = []
        for row in self._ordered_episodes():
            for start in range(0, len(row.turns), _CHUNK_TURNS):
                turns = row.turns[start : start + _CHUNK_TURNS]
                text = "\n".join(f"{turn.speaker}: {turn.text}" for turn in turns)
                tokens = tuple(_tokens(text))
                if tokens:
                    pending.append((row, start, start + len(turns) - 1, text, tokens))
        identities = [
            _identity(row.source_sha256, start, end, text)
            for row, start, end, text, _ in pending
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("precedent chunk identity collision")
        vectors: tuple[tuple[float, ...], ...] | None = None
        failed = False
        if pending and self.embedder is not None:
            try:
                vectors = _validated_vectors(
                    embed_documents(self.embedder, [item[3] for item in pending]),
                    len(pending),
                )
            except Exception:
                failed = True
        chunks = tuple(
            _PrecedentChunk(
                chunk_id=identities[index],
                episode_slug=row.episode_slug,
                source_sha256=row.source_sha256,
                turn_start=start,
                turn_end=end,
                text=text,
                tokens=tokens,
                embedding=vectors[index] if vectors is not None else None,
            )
            for index, (row, start, end, text, tokens) in enumerate(pending)
        )
        return chunks, failed

    def _rank(
        self,
        query: str,
        query_tokens: list[str],
        top_k: int,
        *,
        chunks: Sequence[_PrecedentChunk] | None = None,
    ) -> tuple[tuple[PrecedentSearchHit, ...], bool]:
        ranked_chunks = tuple(chunks) if chunks is not None else self._chunks
        if not ranked_chunks:
            return (), False
        document_count = len(ranked_chunks)
        document_frequency = {
            token: sum(token in chunk.tokens for chunk in ranked_chunks)
            for token in set(query_tokens)
        }
        lexical_scores = []
        for chunk in ranked_chunks:
            score = sum(
                chunk.tokens.count(token)
                * (log((document_count + 1) / (document_frequency[token] + 1)) + 1)
                for token in query_tokens
            )
            lexical_scores.append(float(score))
        lexical_order = sorted(
            (
                index
                for index in range(document_count)
                if lexical_scores[index] > 0
            ),
            key=lambda index: (-lexical_scores[index], index),
        )
        lexical_ids = {
            ranked_chunks[index].chunk_id for index in lexical_order
        }
        rankings = []
        if lexical_order:
            rankings.append(
                [ranked_chunks[index].chunk_id for index in lexical_order]
            )
        dense_ids: set[str] = set()
        query_failed = False
        dense_chunks = [chunk for chunk in ranked_chunks if chunk.embedding is not None]
        if dense_chunks and self.embedder is not None:
            try:
                query_vector = _validated_vectors(
                    embed_queries(self.embedder, [query]), 1
                )[0]
                if len(query_vector) != len(dense_chunks[0].embedding or ()):
                    raise ValueError("query embedding dimension is invalid")
                query_norm = sqrt(sum(value * value for value in query_vector))
                scored: list[tuple[float, int, str]] = []
                for index, chunk in enumerate(dense_chunks):
                    vector = chunk.embedding or ()
                    vector_norm = sqrt(sum(value * value for value in vector))
                    denominator = query_norm * vector_norm
                    similarity = (
                        sum(left * right for left, right in zip(vector, query_vector))
                        / denominator
                        if denominator
                        else 0.0
                    )
                    scored.append((similarity, index, chunk.chunk_id))
                dense = [
                    item[2]
                    for item in sorted(
                        scored, key=lambda item: (-item[0], item[1])
                    )
                ]
                rankings.append(dense)
                dense_ids = set(dense)
            except Exception:
                query_failed = True
        fused = _rrf(rankings)
        ordered_ids = sorted(fused, key=lambda item: (-fused[item], item))[:top_k]
        if len(ordered_ids) < top_k:
            ordered_ids.extend(
                chunk.chunk_id
                for chunk in ranked_chunks
                if chunk.chunk_id not in ordered_ids
            )
            ordered_ids = ordered_ids[:top_k]
        hits = tuple(
            PrecedentSearchHit(
                episode_slug=self._chunks_by_id[chunk_id].episode_slug,
                chunk_id=chunk_id,
                turn_start=self._chunks_by_id[chunk_id].turn_start,
                turn_end=self._chunks_by_id[chunk_id].turn_end,
                excerpt=self._chunks_by_id[chunk_id].text[:800],
                score=fused.get(chunk_id, 0.0),
                retrieval_modes=(
                    tuple(
                        mode
                        for mode, members in (
                            ("lexical", lexical_ids),
                            ("dense", dense_ids),
                        )
                        if chunk_id in members
                    )
                ),
            )
            for chunk_id in ordered_ids
        )
        return hits, query_failed

    def _validate_chunk_consistency(self) -> None:
        expected, _ = self._build_chunks_without_embeddings()
        if len(expected) != len(self._chunks):
            raise ValueError("precedent index chunk consistency mismatch")
        for actual, canonical in zip(self._chunks, expected, strict=True):
            comparable = actual.model_copy(update={"embedding": None})
            if comparable != canonical:
                raise ValueError("precedent index chunk consistency mismatch")
        embedded = [
            chunk.embedding
            for chunk in self._chunks
            if chunk.embedding is not None
        ]
        if embedded:
            _validated_vectors(embedded, len(embedded))

    def _build_chunks_without_embeddings(
        self,
    ) -> tuple[tuple[_PrecedentChunk, ...], bool]:
        provider = self.embedder
        self.embedder = None
        try:
            return self._build_chunks()
        finally:
            self.embedder = provider

    def _verify_record(self, row: PrecedentEpisode) -> None:
        expected = self._record_sha256s[row.episode_slug]
        if self._corpus_root is None:
            canonical_digest = self._canonical_episode_sha256s[row.episode_slug]
            if _episode_digest(row) != canonical_digest:
                raise ValueError("canonical precedent record hash changed")
            return
        manifest_path = _safe_corpus_file(
            self._corpus_root,
            "corpus-manifest.json",
            "corpus manifest path",
        )
        if (
            self._manifest_sha256 is not None
            and sha256(manifest_path.read_bytes()).hexdigest() != self._manifest_sha256
        ):
            raise ValueError("corpus manifest digest changed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = {
            item["episode_slug"]: item for item in manifest.get("records", [])
        }
        entry = entries.get(row.episode_slug)
        if entry is None:
            raise ValueError("canonical precedent record binding changed")
        if (
            entry.get("source_path") != row.source_path
            or entry.get("source_sha256") != row.source_sha256
        ):
            raise ValueError("canonical precedent source binding changed")
        record_path = _safe_corpus_file(
            self._corpus_root, entry["record_path"], "record path"
        )
        if sha256(record_path.read_bytes()).hexdigest() != expected:
            raise ValueError("canonical precedent record hash changed")
        source_path = _safe_corpus_file(
            self._corpus_root, row.source_path, "source path"
        )
        if sha256(source_path.read_bytes()).hexdigest() != row.source_sha256:
            raise ValueError("corpus source snapshot hash changed")

    @staticmethod
    def _evidence(
        row: PrecedentEpisode,
        start: int,
        end: int,
        text: str,
        query_id: str,
    ) -> HistoricalEvidence:
        return HistoricalEvidence(
            evidence_id=_identity(row.source_sha256, start, end, text),
            episode_slug=row.episode_slug,
            source_sha256=row.source_sha256,
            turn_start=start,
            turn_end=end,
            text=text,
            query_id=query_id,
        )

    def _ordered_episodes(self) -> tuple[PrecedentEpisode, ...]:
        return tuple(
            sorted(
                self._episodes.values(),
                key=lambda row: (
                    row.episode_number is None,
                    row.episode_number if row.episode_number is not None else 0,
                    row.episode_slug,
                ),
            )
        )
