"""Automatic, source-bound retrieval for compact v4 model prompts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, Sequence

from .precedents import PrecedentCorpus, PrecedentSearchHit
from .retrieval import ExactEvidence, SearchHit


class WikiRetriever(Protocol):
    def search(self, query: str, limit: int) -> list[SearchHit]: ...
    def read(self, chunk_id: str) -> ExactEvidence: ...


@dataclass(frozen=True)
class V4RetrievalResult:
    wiki_searches: tuple[dict[str, Any], ...]
    precedent_searches: tuple[dict[str, Any], ...]
    wiki_evidence: tuple[dict[str, Any], ...]
    historical_evidence: tuple[dict[str, Any], ...]
    precedent_reads: tuple[dict[str, Any], ...]
    opened_episode_slugs: tuple[str, ...]
    warnings: tuple[str, ...]


def _clean_queries(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(" ".join(value.split()) for value in values if value.strip()))


def retrieve_v4(
    *,
    wiki_index: WikiRetriever | None,
    precedent_corpus: PrecedentCorpus | None,
    wiki_queries: Sequence[str],
    precedent_queries: Sequence[str],
    top_k: int,
    max_wiki_reads: int,
    max_precedent_reads: int,
    phase: Literal["phase1", "phase2"],
    turn: int,
    selection_policy: Literal["semantic", "contrastive"] = "semantic",
    candidate_pool_k: int | None = None,
    in_slots: int = 2,
    out_slots: int = 2,
) -> V4RetrievalResult:
    """Search the full internal stores and auto-open only current-turn top evidence."""
    if top_k < 1 or max_wiki_reads < 0 or max_precedent_reads < 0 or turn < 1:
        raise ValueError("v4 retrieval budgets are invalid")

    wiki_searches: list[dict[str, Any]] = []
    wiki_by_id: dict[str, dict[str, Any]] = {}
    if wiki_index is not None:
        for position, query in enumerate(_clean_queries(wiki_queries), start=1):
            hits = wiki_index.search(query, top_k)
            wiki_searches.append(
                {"query_id": f"{phase}-t{turn:02d}-wiki-{position:02d}",
                 "query": query, "hits": [asdict(hit) for hit in hits]}
            )
            for hit in hits:
                if len(wiki_by_id) >= max_wiki_reads:
                    break
                if hit.chunk_id not in wiki_by_id:
                    exact = asdict(wiki_index.read(hit.chunk_id))
                    # The v4 contract uses one source-neutral citation field while
                    # preserving the legacy chunk identity for audit compatibility.
                    exact["evidence_id"] = exact["chunk_id"]
                    wiki_by_id[hit.chunk_id] = exact

    precedent_searches: list[dict[str, Any]] = []
    ranked_hits: list[tuple[str, PrecedentSearchHit]] = []
    warnings: list[str] = []
    if precedent_corpus is not None:
        for position, query in enumerate(_clean_queries(precedent_queries), start=1):
            query_id = f"{phase}-t{turn:02d}-precedent-{position:02d}"
            result = precedent_corpus.search(
                query,
                top_k=top_k,
                query_id=query_id,
                selection_policy=selection_policy,
                candidate_pool_k=candidate_pool_k,
                in_slots=in_slots,
                out_slots=out_slots,
                observed_only=True,
                substantive_only=True,
            )
            precedent_searches.append(result.model_dump(mode="json"))
            warnings.extend(result.quality_findings)
            ranked_hits.extend((query_id, hit) for hit in result.hits)

    selected: list[tuple[str, PrecedentSearchHit]] = []
    seen_episodes: set[str] = set()
    for query_id, hit in ranked_hits:
        if len(selected) >= max_precedent_reads:
            break
        if hit.episode_slug in seen_episodes:
            continue
        seen_episodes.add(hit.episode_slug)
        selected.append((query_id, hit))

    historical_by_id: dict[str, dict[str, Any]] = {}
    precedent_reads: list[dict[str, Any]] = []
    for position, (query_id, hit) in enumerate(selected, start=1):
        read_id = f"{phase}-t{turn:02d}-read-{position:02d}"
        transcript = precedent_corpus.open_transcript(
            hit.episode_slug,
            hit.turn_start,
            hit.turn_end,
            query_id=read_id,
        )
        decision = precedent_corpus.open_decision(hit.episode_slug, query_id=read_id)
        precedent_reads.append(
            {
                "query_id": query_id,
                "transcript": transcript.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
            }
        )
        if transcript.evidence is not None:
            row = transcript.evidence.model_dump(mode="json")
            row.update(
                episode_slug=hit.episode_slug,
                decision_status=decision.decision.status,
            )
            historical_by_id[row["evidence_id"]] = row
        for evidence in decision.evidence:
            row = evidence.model_dump(mode="json")
            row.update(
                episode_slug=hit.episode_slug,
                decision_status=decision.decision.status,
                evidence_role="observed_decision",
            )
            historical_by_id[row["evidence_id"]] = row

    return V4RetrievalResult(
        wiki_searches=tuple(wiki_searches),
        precedent_searches=tuple(precedent_searches),
        wiki_evidence=tuple(wiki_by_id.values()),
        historical_evidence=tuple(historical_by_id.values()),
        precedent_reads=tuple(precedent_reads),
        opened_episode_slugs=tuple(hit.episode_slug for _, hit in selected),
        warnings=tuple(dict.fromkeys(warnings)),
    )
