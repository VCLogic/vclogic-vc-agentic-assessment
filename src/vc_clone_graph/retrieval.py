"""Source-addressable hybrid retrieval over an investor wiki."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Protocol, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .firewall import sha256_file
from .providers.base import embed_documents, embed_queries, embedding_identity


_TOKEN = re.compile(r"[a-z0-9][a-z0-9_-]*")
_HEADING = re.compile(r"(?m)^#{1,6} .+$")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class WikiChunk:
    chunk_id: str
    source_path: str
    source_sha256: str
    heading: str
    text: str
    embedding: tuple[float, ...] | None


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    source_path: str
    heading: str
    excerpt: str
    score: float
    retrieval_modes: tuple[str, ...]


@dataclass(frozen=True)
class ExactEvidence:
    chunk_id: str
    source_path: str
    source_sha256: str
    heading: str
    text: str


@dataclass(frozen=True)
class SanitizationReport:
    alias_hashes: tuple[str, ...]
    changed_chunk_ids: tuple[str, ...]
    changed_chunk_count: int
    reused_embedding_count: int
    recomputed_embedding_count: int
    lexical_only_chunk_ids: tuple[str, ...]
    quality_findings: tuple[str, ...]


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _word_character(value: str) -> bool:
    return value == "_" or value.isalnum()


def _replace_casefold_alias(text: str, alias: str, replacement: str) -> str:
    """Replace a whole-word alias using full Unicode case-fold matching."""
    folded_parts: list[str] = []
    source_indexes: list[int] = []
    for index, character in enumerate(text):
        folded = character.casefold()
        folded_parts.append(folded)
        source_indexes.extend([index] * len(folded))
    folded_text = "".join(folded_parts)
    folded_alias = alias.casefold()
    spans: list[tuple[int, int]] = []
    start = 0
    while (match := folded_text.find(folded_alias, start)) >= 0:
        end = match + len(folded_alias)
        left_ok = match == 0 or not _word_character(folded_text[match - 1])
        right_ok = end == len(folded_text) or not _word_character(folded_text[end])
        if left_ok and right_ok:
            source_start = source_indexes[match]
            source_end = source_indexes[end - 1] + 1
            spans.append((source_start, source_end))
        start = max(end, match + 1)
    for source_start, source_end in reversed(spans):
        text = text[:source_start] + replacement + text[source_end:]
    return text


def _contains_casefold_alias(text: str, alias: str) -> bool:
    sentinel = "[__ALIAS_MATCH__]"
    return _replace_casefold_alias(text, alias, sentinel) != text


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def _sections(text: str) -> list[tuple[str, str]]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        stripped = text.strip()
        return [("Document", stripped)] if stripped else []
    rows: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[match.start():end].strip()
        if section:
            rows.append((match.group(0).lstrip("# ").strip(), section))
    return rows


class HybridWikiIndex:
    def __init__(
        self,
        wiki_root: Path,
        chunks: Sequence[WikiChunk],
        embedder: Embedder,
        *,
        sanitized_chunk_ids: frozenset[str] = frozenset(),
        canonical_source_paths: dict[str, str] | None = None,
        embedding_index: dict[str, object] | None = None,
        require_complete_embeddings: bool = False,
    ):
        self.wiki_root = Path(wiki_root).resolve()
        self.chunks = tuple(chunks)
        self.embedder = embedder
        self.require_complete_embeddings = require_complete_embeddings
        self._sanitized_chunk_ids = sanitized_chunk_ids
        self._canonical_source_paths = canonical_source_paths or {
            chunk.chunk_id: chunk.source_path for chunk in self.chunks
        }
        self._by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        if len(self._by_id) != len(self.chunks):
            raise ValueError("wiki chunk IDs are not unique")
        self.embedding_index = embedding_index or self._embedding_metadata()
        self._validate_embedding_index()
        self._bm25 = BM25Okapi([_tokens(chunk.text) for chunk in self.chunks])

    @classmethod
    def build(
        cls,
        wiki_root: Path,
        embedder: Embedder,
        *,
        require_complete_embeddings: bool = False,
    ) -> "HybridWikiIndex":
        root = Path(wiki_root).resolve()
        pending: list[tuple[str, str, str, str]] = []
        for path in sorted(root.rglob("*.md")):
            relative = path.relative_to(root).as_posix()
            digest = sha256_file(path)
            for heading, text in _sections(path.read_text(encoding="utf-8")):
                pending.append((relative, digest, heading, text))
        texts = [row[3] for row in pending]
        vectors: list[list[float]] | None
        try:
            vectors = embed_documents(embedder, texts)
            if len(vectors) != len(texts) or any(not vector for vector in vectors):
                raise ValueError("embedding count is invalid")
        except Exception as exc:
            if require_complete_embeddings:
                raise RuntimeError("cannot build complete wiki embedding index") from exc
            vectors = None
        chunks = []
        for index, (relative, digest, heading, text) in enumerate(pending):
            identity = sha256(f"{relative}\0{text}".encode()).hexdigest()[:20]
            chunks.append(
                WikiChunk(
                    chunk_id=f"W-{identity}",
                    source_path=relative,
                    source_sha256=digest,
                    heading=heading,
                    text=text,
                    embedding=(tuple(vectors[index]) if vectors is not None else None),
                )
            )
        if not chunks:
            raise ValueError("wiki contains no Markdown sections")
        return cls(
            root,
            chunks,
            embedder,
            require_complete_embeddings=require_complete_embeddings,
        )

    def _embedding_metadata(self) -> dict[str, object]:
        identity = embedding_identity(self.embedder)
        embedded = [chunk.embedding for chunk in self.chunks if chunk.embedding is not None]
        dimension = len(embedded[0]) if embedded else 0
        return {
            **identity,
            "dimension": dimension,
            "embedded_count": len(embedded),
            "total_count": len(self.chunks),
            "coverage": len(embedded) / len(self.chunks) if self.chunks else 0.0,
        }

    def _validate_embedding_index(self) -> None:
        actual = self._embedding_metadata()
        for field in ("dimension", "embedded_count", "total_count", "coverage"):
            if self.embedding_index.get(field) != actual[field]:
                raise ValueError("wiki embedding index metadata is inconsistent")
        if self.require_complete_embeddings:
            if actual["coverage"] != 1.0 or actual["dimension"] == 0:
                raise ValueError("complete wiki embedding index is required")
            identity_fields = (
                "backend", "model", "revision", "normalize",
                "document_prefix", "query_prefix",
            )
            if any(
                self.embedding_index.get(field) != actual[field]
                for field in identity_fields
            ):
                raise ValueError("wiki embedding configuration mismatch")

    def search(self, query: str, limit: int) -> list[SearchHit]:
        if not query.strip() or limit < 1:
            raise ValueError("query and positive limit are required")
        lexical_scores = self._bm25.get_scores(_tokens(query))
        lexical = [
            self.chunks[index].chunk_id
            for index in np.argsort(-lexical_scores, kind="stable")
        ]
        rankings: list[list[str]] = [lexical]
        dense: list[str] = []
        dense_chunks = [chunk for chunk in self.chunks if chunk.embedding is not None]
        if dense_chunks:
            try:
                query_vector = np.asarray(
                    embed_queries(self.embedder, [query])[0], dtype=float
                )
                matrix = np.asarray([chunk.embedding for chunk in dense_chunks], dtype=float)
                denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
                similarities = np.divide(
                    matrix @ query_vector,
                    denominator,
                    out=np.zeros(len(matrix)),
                    where=denominator != 0,
                )
                dense = [
                    dense_chunks[index].chunk_id
                    for index in np.argsort(-similarities, kind="stable")
                ]
                rankings.append(dense)
            except Exception:
                dense = []
        fused = reciprocal_rank_fusion(rankings)
        ordered = sorted(fused, key=lambda item: (-fused[item], item))[:limit]
        hits = []
        for chunk_id in ordered:
            chunk = self._by_id[chunk_id]
            modes = ["lexical"]
            if chunk_id in dense:
                modes.append("dense")
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    source_path=chunk.source_path,
                    heading=chunk.heading,
                    excerpt=chunk.text[:500],
                    score=fused[chunk_id],
                    retrieval_modes=tuple(modes),
                )
            )
        return hits

    def read(self, chunk_id: str) -> ExactEvidence:
        try:
            chunk = self._by_id[chunk_id]
        except KeyError as exc:
            raise KeyError(f"unknown wiki chunk: {chunk_id}") from exc
        canonical_source_path = self._canonical_source_paths[chunk_id]
        source = self.wiki_root / canonical_source_path
        if sha256_file(source) != chunk.source_sha256:
            raise ValueError(f"wiki source hash changed: {chunk.source_path}")
        if (
            chunk_id not in self._sanitized_chunk_ids
            and chunk.text not in source.read_text(encoding="utf-8")
        ):
            raise ValueError(f"wiki chunk text changed: {chunk_id}")
        return ExactEvidence(
            chunk.chunk_id,
            chunk.source_path,
            chunk.source_sha256,
            chunk.heading,
            chunk.text,
        )

    def for_pitch(
        self, aliases: Sequence[str]
    ) -> tuple["HybridWikiIndex", SanitizationReport]:
        """Return an ephemeral target-redacted view of this canonical index."""
        normalized = tuple(alias.strip() for alias in aliases if alias.strip())
        if not normalized:
            raise ValueError("at least one target company alias is required")
        ordered_aliases = sorted(normalized, key=lambda value: len(value.casefold()), reverse=True)

        sentinel = "[__PITCH_TARGET_ALIAS__]"
        changed_rows: list[tuple[WikiChunk, str, str, str]] = []
        unchanged: list[WikiChunk] = []
        for chunk in self.chunks:
            text = chunk.text
            heading = chunk.heading
            source_path = chunk.source_path
            for alias in ordered_aliases:
                text = _replace_casefold_alias(text, alias, sentinel)
                heading = _replace_casefold_alias(heading, alias, sentinel)
                source_path = _replace_casefold_alias(source_path, alias, sentinel)
            text = re.sub(rf",\s*{re.escape(sentinel)}", "", text)
            text = re.sub(rf"{re.escape(sentinel)}\s*,\s*", "", text)
            text = text.replace(sentinel, "")
            text = re.sub(r"(?m)^\s*[-*]\s*[.;:]?\s*$\n?", "", text)
            heading = heading.replace(sentinel, "").strip() or "Sanitized section"
            if sentinel in source_path:
                source_path = f"redacted/{chunk.chunk_id}.md"
            if any(
                _contains_casefold_alias(value, alias)
                for alias in normalized
                for value in (text, heading, source_path)
            ):
                raise ValueError("target alias sanitization failed")
            if (
                text == chunk.text
                and heading == chunk.heading
                and source_path == chunk.source_path
            ):
                unchanged.append(chunk)
            else:
                changed_rows.append((chunk, text, heading, source_path))

        vectors: list[list[float]] | None = []
        quality_findings: tuple[str, ...] = ()
        if changed_rows:
            try:
                vectors = embed_documents(
                    self.embedder, [text for _, text, _, _ in changed_rows]
                )
                if len(vectors) != len(changed_rows) or any(not vector for vector in vectors):
                    raise ValueError("embedding count is invalid")
            except Exception as exc:
                if self.require_complete_embeddings:
                    raise RuntimeError(
                        "cannot sanitize complete wiki embedding index"
                    ) from exc
                vectors = None
                quality_findings = ("SANITIZED_EMBEDDING_FALLBACK",)

        replacements: dict[str, WikiChunk] = {}
        lexical_only: list[str] = []
        for index, (chunk, text, heading, source_path) in enumerate(changed_rows):
            embedding = tuple(vectors[index]) if vectors is not None else None
            if embedding is None:
                lexical_only.append(chunk.chunk_id)
            replacements[chunk.chunk_id] = WikiChunk(
                chunk_id=chunk.chunk_id,
                source_path=source_path,
                source_sha256=chunk.source_sha256,
                heading=heading,
                text=text,
                embedding=embedding,
            )
        chunks = tuple(replacements.get(chunk.chunk_id, chunk) for chunk in self.chunks)
        changed_ids = tuple(chunk.chunk_id for chunk, _, _, _ in changed_rows)
        report = SanitizationReport(
            alias_hashes=tuple(
                sha256(alias.casefold().encode("utf-8")).hexdigest()
                for alias in normalized
            ),
            changed_chunk_ids=changed_ids,
            changed_chunk_count=len(changed_ids),
            reused_embedding_count=sum(
                chunk.embedding is not None for chunk in unchanged
            ),
            recomputed_embedding_count=(len(changed_ids) if vectors is not None else 0),
            lexical_only_chunk_ids=tuple(lexical_only),
            quality_findings=quality_findings,
        )
        return (
            HybridWikiIndex(
                self.wiki_root,
                chunks,
                self.embedder,
                sanitized_chunk_ids=frozenset(changed_ids),
                canonical_source_paths=dict(self._canonical_source_paths),
                require_complete_embeddings=self.require_complete_embeddings,
            ),
            report,
        )

    def save(self, path: Path) -> None:
        candidate = Path(path)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "schema": "hybrid-wiki-index-v2",
                "embedding_index": self.embedding_index,
                "chunks": [asdict(c) for c in self.chunks],
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=candidate.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, candidate)

    @classmethod
    def load(
        cls,
        path: Path,
        wiki_root: Path,
        embedder: Embedder,
        *,
        require_complete_embeddings: bool = False,
    ) -> "HybridWikiIndex":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = raw.get("schema")
        if schema not in {"hybrid-wiki-index-v1", "hybrid-wiki-index-v2"} or type(raw.get("chunks")) is not list:
            raise ValueError("wiki index schema is invalid")
        if require_complete_embeddings and schema != "hybrid-wiki-index-v2":
            raise ValueError("complete wiki embedding index lacks verified metadata")
        chunks = [
            WikiChunk(
                chunk_id=row["chunk_id"],
                source_path=row["source_path"],
                source_sha256=row["source_sha256"],
                heading=row["heading"],
                text=row["text"],
                embedding=(tuple(row["embedding"]) if row["embedding"] is not None else None),
            )
            for row in raw["chunks"]
        ]
        index = cls(
            wiki_root,
            chunks,
            embedder,
            embedding_index=raw.get("embedding_index"),
            require_complete_embeddings=require_complete_embeddings,
        )
        for chunk in index.chunks:
            index.read(chunk.chunk_id)
        return index
