"""Claim-scoped evidence retrieval and local taxonomy neighborhoods for v4.4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from math import isfinite, sqrt
import re
from typing import Any

from rank_bm25 import BM25Okapi

from .phase1_v44 import (
    ClaimEvidenceBundleV44,
    ClaimMapV44,
    ClaimRetrievalManifestV44,
    EvidenceRegistryRecordV44,
    RetrievalActionV44,
    RetrievedEvidenceV44,
    TaxonomyCandidateV44,
    TaxonomyNeighborhoodManifestV44,
    TaxonomyNeighborhoodV44,
)
from .providers.base import embed_documents, embed_queries, embedding_identity
from .retrieval import reciprocal_rank_fusion
from .retrieval_v4 import retrieve_v4


_TOKEN = re.compile(r"[a-z0-9][a-z0-9_-]*")
_SOURCE_FIELDS = (
    "evidence_id",
    "source_kind",
    "text",
    "source_locator",
    "source_sha256",
    "episode_slug",
    "turn_start",
    "turn_end",
    "decision_status",
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _compact(*parts: str) -> str:
    return " | ".join(" ".join(part.split()) for part in parts)


def _source_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    return {
        field: getattr(value, field)
        for field in dir(value)
        if not field.startswith("_") and not callable(getattr(value, field))
    }


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _wiki_source_rows(wiki_index: Any) -> dict[str, dict[str, Any]]:
    if wiki_index is None:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for value in getattr(wiki_index, "chunks", ()):
        row = _source_payload(value)
        evidence_id = row.get("chunk_id")
        if isinstance(evidence_id, str):
            rows[evidence_id] = row
    return rows


def _precedent_summaries(precedent_corpus: Any) -> dict[str, dict[str, Any]]:
    if precedent_corpus is None:
        return {}
    return {
        row["episode_slug"]: row
        for value in precedent_corpus.list_episodes()
        for row in (_source_payload(value),)
    }


class _TargetSafePrecedentProxy:
    """Reject target leakage before retrieve_v4 can open an unsafe search hit."""

    def __init__(self, delegate: Any, target_slug: str) -> None:
        self._delegate = delegate
        self._target_slug = target_slug

    def search(self, *args: Any, **kwargs: Any) -> Any:
        result = self._delegate.search(*args, **kwargs)
        hits = result.hits if hasattr(result, "hits") else result
        for hit in hits:
            slug = (
                hit.get("episode_slug")
                if isinstance(hit, Mapping)
                else hit.episode_slug
            )
            if slug == self._target_slug:
                raise ValueError("target episode appeared in historical search")
        return result

    def open_transcript(self, slug: str, *args: Any, **kwargs: Any) -> Any:
        if slug == self._target_slug:
            raise ValueError("target episode cannot be opened as historical transcript")
        return self._delegate.open_transcript(slug, *args, **kwargs)

    def open_decision(self, slug: str, *args: Any, **kwargs: Any) -> Any:
        if slug == self._target_slug:
            raise ValueError("target episode cannot be opened as historical decision")
        return self._delegate.open_decision(slug, *args, **kwargs)

    def evidence_text(self, evidence_id: str) -> str | None:
        resolver = getattr(self._delegate, "evidence_text", None)
        if not callable(resolver):
            return None
        return resolver(evidence_id)


def _portfolio_disclosures(portfolio_index: Any) -> dict[str, Any]:
    if portfolio_index is None:
        return {}
    filtered = getattr(portfolio_index, "filtered", None)
    if filtered is None:
        raise ValueError("portfolio index must expose its filtered corpus")
    rows = tuple(getattr(filtered, "disclosures", ()))
    result = {row.disclosure_id: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("portfolio disclosures must have unique IDs")
    return result


def _portfolio_source(disclosure: Any, target_slug: str) -> dict[str, Any]:
    if disclosure.source_episode_slug == target_slug:
        raise ValueError("target episode cannot appear in portfolio disclosures")
    evidence = tuple(disclosure.evidence)
    hashes = {row.source_sha256 for row in evidence}
    if len(hashes) != 1:
        raise ValueError("portfolio disclosure evidence must bind to one source hash")
    turn_indexes = sorted({row.turn_index for row in evidence})
    text = "\n".join(row.text for row in evidence)
    contiguous = turn_indexes == list(range(turn_indexes[0], turn_indexes[-1] + 1))
    turn_locator = (
        f"{turn_indexes[0]}-{turn_indexes[-1]}"
        if contiguous and len(turn_indexes) > 1
        else ",".join(str(value) for value in turn_indexes)
    )
    locator = f"{disclosure.source_episode_slug}:turns-{turn_locator}"
    return {
        "evidence_id": disclosure.disclosure_id,
        "source_kind": "portfolio",
        "text": text,
        "source_locator": locator,
        "source_sha256": next(iter(hashes)),
        "episode_slug": None,
        "turn_start": None,
        "turn_end": None,
        "decision_status": None,
    }


def retrieve_claim_evidence_v44(
    *,
    claim_map: ClaimMapV44,
    wiki_index: Any,
    precedent_corpus: Any,
    portfolio_index: Any,
    target_episode_slug: str,
    top_k: int,
    max_wiki_reads: int,
    max_precedent_reads: int,
    only_ids: Sequence[str] | None = None,
) -> ClaimRetrievalManifestV44:
    """Retrieve a bounded, source-bound evidence graph for each claim target."""
    if top_k < 1 or max_wiki_reads < 1 or max_precedent_reads < 0:
        raise ValueError(
            "top_k and wiki reads must be positive; precedent reads must be nonnegative"
        )
    if claim_map.episode_slug != target_episode_slug:
        raise ValueError("claim map and target episode disagree")

    targets: list[tuple[str, str, str, int]] = []
    for ordinal, row in enumerate(
        [
            *(('claim', item) for item in claim_map.material_claims),
            *(('claim', item) for item in claim_map.adverse_claims),
            *(('question', item) for item in claim_map.unanswered_questions),
        ],
        start=1,
    ):
        kind, item = row
        if kind == "claim":
            target_id = item.claim_id
            query = _compact(item.statement, item.topic, item.decision_relevance)
        else:
            target_id = item.question_id
            query = _compact(item.question, item.why_material)
        targets.append((kind, target_id, query, ordinal))

    known_ids = {row[1] for row in targets}
    selected_ids: set[str] | None = None
    if only_ids is not None:
        if isinstance(only_ids, (str, bytes)) or not isinstance(only_ids, Sequence):
            raise ValueError("only_ids must be a sequence of target IDs")
        materialized_ids = tuple(only_ids)
        if any(not isinstance(value, str) for value in materialized_ids):
            raise ValueError("only_ids must be a sequence of target IDs")
        if len(materialized_ids) != len(set(materialized_ids)):
            raise ValueError("only_ids must be unique")
        unknown = set(materialized_ids) - known_ids
        if unknown:
            raise ValueError(f"unknown only_ids target: {sorted(unknown)[0]}")
        selected_ids = set(materialized_ids)
        targets = [row for row in targets if row[1] in selected_ids]

    if not targets:
        return ClaimRetrievalManifestV44(
            schema_version="claim-retrieval-manifest-v4.4",
            episode_slug=target_episode_slug,
            claim_map_sha256=_canonical_sha256(claim_map.model_dump(mode="json")),
            claim_bundles=(),
            target_episode_excluded=True,
            warnings=(),
            evidence_registry=(),
            retrieval_actions=(),
        )

    if portfolio_index is not None:
        filtered = getattr(portfolio_index, "filtered", None)
        if (
            filtered is None
            or getattr(filtered, "target_episode_slug", None)
            != target_episode_slug
        ):
            raise ValueError("portfolio index target is not verified")
        target_prefix = target_episode_slug.split("-", 1)[0]
        target_number = int(target_prefix) if target_prefix.isdecimal() else None
        if getattr(filtered, "target_episode_number", None) != target_number:
            raise ValueError("portfolio temporal filter marker is inconsistent")
        for disclosure in getattr(filtered, "disclosures", ()):
            source_slug = disclosure.source_episode_slug
            source_number = disclosure.source_episode_number
            if (
                source_slug == target_episode_slug
                or target_number is None
                or source_number is None
                or source_number >= target_number
            ):
                raise ValueError("portfolio temporal eligibility is inconsistent")

    if precedent_corpus is not None:
        if getattr(precedent_corpus, "target_slug", None) != target_episode_slug:
            raise ValueError("precedent target episode is not verified as excluded")
        if any(
            row["episode_slug"] == target_episode_slug
            for row in (_source_payload(value) for value in precedent_corpus.list_episodes())
        ):
            raise ValueError("precedent target episode is not excluded")
    wiki_sources = _wiki_source_rows(wiki_index)
    precedent_summaries = _precedent_summaries(precedent_corpus)
    portfolio_sources = _portfolio_disclosures(portfolio_index)
    safe_precedent = (
        _TargetSafePrecedentProxy(precedent_corpus, target_episode_slug)
        if precedent_corpus is not None
        else None
    )
    registry: dict[str, dict[str, Any]] = {}
    registry_exact: dict[str, bool] = {}
    actions: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    manifest_warnings: list[str] = []

    def register(source: dict[str, Any], *, exact: bool) -> None:
        evidence_id = source["evidence_id"]
        prior = registry.get(evidence_id)
        if prior is None:
            registry[evidence_id] = {field: source.get(field) for field in _SOURCE_FIELDS}
            registry_exact[evidence_id] = exact
            return
        stable_fields = tuple(field for field in _SOURCE_FIELDS if field != "text")
        if any(prior[field] != source.get(field) for field in stable_fields):
            raise ValueError(f"conflicting source identity for evidence {evidence_id}")
        if prior["text"] != source.get("text"):
            if exact and registry_exact[evidence_id]:
                raise ValueError(f"conflicting exact source text for evidence {evidence_id}")
            if exact:
                registry[evidence_id] = {field: source.get(field) for field in _SOURCE_FIELDS}
        if exact:
            registry_exact[evidence_id] = True

    def add_action(
        bundle_action_ids: list[str],
        *,
        target_id: str,
        source_kind: str,
        action_kind: str,
        query_id: str,
        query: str,
        result_ids: Sequence[str],
        opened_slugs: Sequence[str] = (),
        warnings: Sequence[str] = (),
    ) -> None:
        action_id = f"A{len(actions) + 1}"
        actions.append(
            {
                "action_id": action_id,
                "target_id": target_id,
                "source_kind": source_kind,
                "action_kind": action_kind,
                "query_id": query_id,
                "query": query,
                "result_evidence_ids": _unique(tuple(result_ids)),
                "opened_episode_slugs": _unique(tuple(opened_slugs)),
                "warnings": _unique(tuple(warnings)),
            }
        )
        bundle_action_ids.append(action_id)

    for kind, target_id, query, ordinal in targets:
        result = retrieve_v4(
            wiki_index=wiki_index,
            precedent_corpus=safe_precedent,
            wiki_queries=(query,),
            precedent_queries=(query,),
            top_k=top_k,
            max_wiki_reads=max_wiki_reads,
            max_precedent_reads=max_precedent_reads,
            phase="phase1",
            turn=ordinal,
            selection_policy="semantic",
        )
        reported_opened_slugs = tuple(result.opened_episode_slugs)
        read_slugs: list[str] = []
        for read in result.precedent_reads:
            transcript_slug = read["transcript"]["episode_slug"]
            decision_slug = read["decision"]["episode_slug"]
            if transcript_slug != decision_slug:
                raise ValueError(
                    "historical opened episode read is internally inconsistent"
                )
            read_slugs.append(transcript_slug)
        if (
            len(reported_opened_slugs) != len(set(reported_opened_slugs))
            or len(read_slugs) != len(set(read_slugs))
            or set(read_slugs) != set(reported_opened_slugs)
        ):
            raise ValueError("historical opened episode accounting is inconsistent")
        if target_episode_slug in reported_opened_slugs:
            raise ValueError("target episode appeared in historical openings")
        manifest_warnings.extend(result.warnings)
        occurrences: dict[str, list[dict[str, Any]]] = {
            "wiki": [],
            "historical": [],
            "portfolio": [],
        }
        occurrence_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        bundle_action_ids: list[str] = []

        def occurrence(
            *,
            source_kind: str,
            evidence_id: str,
            query_id: str,
            text: str,
            modes: Sequence[str],
            score: float,
            eligible: bool,
            warning: str | None = None,
        ) -> None:
            key = (source_kind, evidence_id, query_id)
            prior = occurrence_by_key.get(key)
            if prior is None:
                row = {
                    "evidence_id": evidence_id,
                    "query_id": query_id,
                    "text": text,
                    "retrieval_modes": _unique(tuple(modes)),
                    "score": float(score),
                    "eligible": eligible,
                    "warning": warning,
                }
                occurrences[source_kind].append(row)
                occurrence_by_key[key] = row
            else:
                prior["retrieval_modes"] = _unique(
                    (*prior["retrieval_modes"], *tuple(modes))
                )
                prior["score"] = max(prior["score"], float(score))
                if eligible:
                    prior["text"] = text
                prior["eligible"] = prior["eligible"] or eligible
                prior["warning"] = prior["warning"] or warning

        exact_wiki = {row["evidence_id"]: row for row in result.wiki_evidence}
        for search in result.wiki_searches:
            query_id = search["query_id"]
            hit_ids: list[str] = []
            for hit in search["hits"]:
                evidence_id = hit["chunk_id"]
                exact = exact_wiki.get(evidence_id)
                indexed = wiki_sources.get(evidence_id)
                if exact is not None:
                    source = {
                        "evidence_id": evidence_id,
                        "source_kind": "wiki",
                        "text": exact["text"],
                        "source_locator": exact["source_path"],
                        "source_sha256": exact["source_sha256"],
                        "episode_slug": None,
                        "turn_start": None,
                        "turn_end": None,
                        "decision_status": None,
                    }
                elif indexed is not None and indexed.get("source_sha256"):
                    source = {
                        "evidence_id": evidence_id,
                        "source_kind": "wiki",
                        "text": indexed["text"],
                        "source_locator": indexed.get("source_path") or hit["source_path"],
                        "source_sha256": indexed["source_sha256"],
                        "episode_slug": None,
                        "turn_start": None,
                        "turn_end": None,
                        "decision_status": None,
                    }
                else:
                    raise ValueError(f"wiki hit lacks source-bound metadata: {evidence_id}")
                register(source, exact=exact is not None)
                occurrence(
                    source_kind="wiki",
                    evidence_id=evidence_id,
                    query_id=query_id,
                    text=source["text"] if exact is not None else hit["excerpt"],
                    modes=hit.get("retrieval_modes") or ("search",),
                    score=hit["score"],
                    eligible=exact is not None,
                )
                hit_ids.append(evidence_id)
            add_action(
                bundle_action_ids,
                target_id=target_id,
                source_kind="wiki",
                action_kind="search",
                query_id=query_id,
                query=query,
                result_ids=hit_ids,
            )
            opened_ids = [evidence_id for evidence_id in hit_ids if evidence_id in exact_wiki]
            if opened_ids:
                add_action(
                    bundle_action_ids,
                    target_id=target_id,
                    source_kind="wiki",
                    action_kind="read",
                    query_id=query_id,
                    query=query,
                    result_ids=opened_ids,
                )

        exact_historical = {
            row["evidence_id"]: row for row in result.historical_evidence
        }
        historical_query_id = (
            result.precedent_searches[0]["query_id"]
            if result.precedent_searches
            else f"phase1-t{ordinal:02d}-precedent-01"
        )
        hit_by_episode: dict[str, dict[str, Any]] = {}
        for search in result.precedent_searches:
            query_id = search["query_id"]
            findings = tuple(search.get("quality_findings", ()))
            hit_ids: list[str] = []
            for hit in search["hits"]:
                slug = hit["episode_slug"]
                if slug == target_episode_slug:
                    raise ValueError("target episode appeared in historical search")
                summary = precedent_summaries.get(slug)
                if summary is None:
                    raise ValueError(f"historical hit has unknown episode: {slug}")
                exact = exact_historical.get(hit["chunk_id"])
                canonical_text = (
                    safe_precedent.evidence_text(hit["chunk_id"])
                    if safe_precedent is not None
                    else None
                )
                source = {
                    "evidence_id": hit["chunk_id"],
                    "source_kind": "historical",
                    "text": (
                        exact["text"]
                        if exact is not None
                        else canonical_text or hit["excerpt"]
                    ),
                    "source_locator": f"{slug}:turns-{hit['turn_start']}-{hit['turn_end']}",
                    "source_sha256": summary["source_sha256"],
                    "episode_slug": slug,
                    "turn_start": hit["turn_start"],
                    "turn_end": hit["turn_end"],
                    "decision_status": summary["decision_status"],
                }
                register(source, exact=exact is not None)
                occurrence(
                    source_kind="historical",
                    evidence_id=hit["chunk_id"],
                    query_id=query_id,
                    text=source["text"] if exact is not None else hit["excerpt"],
                    modes=hit.get("retrieval_modes") or ("search",),
                    score=hit["score"],
                    eligible=exact is not None,
                    warning=findings[0] if findings else None,
                )
                hit_ids.append(hit["chunk_id"])
                hit_by_episode.setdefault(slug, hit)
            add_action(
                bundle_action_ids,
                target_id=target_id,
                source_kind="historical",
                action_kind="search",
                query_id=query_id,
                query=query,
                result_ids=hit_ids,
                warnings=findings,
            )

        for read in result.precedent_reads:
            transcript = read["transcript"]
            decision = read["decision"]
            slug = transcript["episode_slug"]
            if slug == target_episode_slug:
                raise ValueError("target episode appeared in historical read")
            exact_rows: list[dict[str, Any]] = []
            if transcript.get("evidence") is not None:
                exact_rows.append(transcript["evidence"])
            exact_rows.extend(decision.get("evidence", ()))
            hit = hit_by_episode.get(slug, {})
            result_ids: list[str] = []
            for exact in exact_rows:
                evidence_id = exact["evidence_id"]
                source = {
                    "evidence_id": evidence_id,
                    "source_kind": "historical",
                    "text": exact["text"],
                    "source_locator": f"{slug}:turns-{exact['turn_start']}-{exact['turn_end']}",
                    "source_sha256": exact["source_sha256"],
                    "episode_slug": slug,
                    "turn_start": exact["turn_start"],
                    "turn_end": exact["turn_end"],
                    "decision_status": decision["decision"]["status"],
                }
                register(source, exact=True)
                occurrence(
                    source_kind="historical",
                    evidence_id=evidence_id,
                    query_id=historical_query_id,
                    text=source["text"],
                    modes=hit.get("retrieval_modes") or ("read",),
                    score=hit.get("score", 0.0),
                    eligible=True,
                )
                result_ids.append(evidence_id)
            if result_ids:
                add_action(
                    bundle_action_ids,
                    target_id=target_id,
                    source_kind="historical",
                    action_kind="read",
                    query_id=historical_query_id,
                    query=query,
                    result_ids=result_ids,
                    opened_slugs=(slug,),
                )

        if portfolio_index is not None:
            portfolio_query_id = f"phase1-t{ordinal:02d}-portfolio-01"
            candidates = portfolio_index.search(
                query,
                limit=top_k,
                candidate_pool_k=None,
            )
            disclosure_ids: list[str] = []
            for candidate_value in candidates:
                candidate = _source_payload(candidate_value)
                for evidence_id in candidate["disclosure_ids"]:
                    disclosure = portfolio_sources.get(evidence_id)
                    if disclosure is None:
                        raise ValueError(
                            f"portfolio hit has unknown disclosure: {evidence_id}"
                        )
                    source = _portfolio_source(disclosure, target_episode_slug)
                    register(source, exact=True)
                    occurrence(
                        source_kind="portfolio",
                        evidence_id=evidence_id,
                        query_id=portfolio_query_id,
                        text=source["text"],
                        modes=candidate.get("retrieval_modes") or ("search",),
                        score=candidate["score"],
                        eligible=True,
                    )
                    disclosure_ids.append(evidence_id)
            disclosure_ids = list(_unique(tuple(disclosure_ids)))
            add_action(
                bundle_action_ids,
                target_id=target_id,
                source_kind="portfolio",
                action_kind="search",
                query_id=portfolio_query_id,
                query=query,
                result_ids=disclosure_ids,
            )
            if disclosure_ids:
                add_action(
                    bundle_action_ids,
                    target_id=target_id,
                    source_kind="portfolio",
                    action_kind="read",
                    query_id=portfolio_query_id,
                    query=query,
                    result_ids=disclosure_ids,
                )

        bundles.append(
            {
                "target_kind": kind,
                "target_id": target_id,
                "query": query,
                "occurrences": occurrences,
                "warnings": _unique(result.warnings),
                "retrieval_action_ids": tuple(bundle_action_ids),
            }
        )

    bundle_models: list[ClaimEvidenceBundleV44] = []
    for bundle in bundles:
        evidence_groups: dict[str, tuple[RetrievedEvidenceV44, ...]] = {}
        for source_kind in ("wiki", "historical", "portfolio"):
            materialized: list[RetrievedEvidenceV44] = []
            for occurrence_row in bundle["occurrences"][source_kind]:
                source = registry[occurrence_row["evidence_id"]]
                materialized.append(
                    RetrievedEvidenceV44(
                        **{**source, "text": occurrence_row["text"]},
                        query_id=occurrence_row["query_id"],
                        retrieval_modes=occurrence_row["retrieval_modes"],
                        score=occurrence_row["score"],
                        eligible=occurrence_row["eligible"],
                        warning=occurrence_row["warning"],
                    )
                )
            evidence_groups[source_kind] = tuple(materialized)
        bundle_models.append(
            ClaimEvidenceBundleV44(
                target_kind=bundle["target_kind"],
                target_id=bundle["target_id"],
                query=bundle["query"],
                wiki_evidence=evidence_groups["wiki"],
                historical_evidence=evidence_groups["historical"],
                portfolio_disclosures=evidence_groups["portfolio"],
                warnings=bundle["warnings"],
                retrieval_action_ids=bundle["retrieval_action_ids"],
            )
        )

    return ClaimRetrievalManifestV44(
        schema_version="claim-retrieval-manifest-v4.4",
        episode_slug=target_episode_slug,
        claim_map_sha256=_canonical_sha256(claim_map.model_dump(mode="json")),
        claim_bundles=tuple(bundle_models),
        target_episode_excluded=True,
        warnings=_unique(tuple(manifest_warnings)),
        evidence_registry=tuple(
            EvidenceRegistryRecordV44(**source) for source in registry.values()
        ),
        retrieval_actions=tuple(RetrievalActionV44(**row) for row in actions),
    )


def _taxonomy_rows(
    taxonomy: Sequence[Any],
) -> tuple[tuple[dict[str, str], ...], tuple[dict[str, Any], ...]]:
    if isinstance(taxonomy, (str, bytes)) or not isinstance(taxonomy, Sequence):
        raise ValueError("taxonomy must be an ordered sequence")
    rows: list[dict[str, str]] = []
    supplied_rows: list[dict[str, Any]] = []
    for index, value in enumerate(taxonomy):
        raw = _source_payload(value)
        try:
            row = {
                field: raw[field]
                for field in ("label", "coarse_parent", "definition")
            }
        except KeyError as exc:
            raise ValueError(f"taxonomy row {index} is missing a required field") from exc
        if any(not isinstance(item, str) or not item.strip() for item in row.values()):
            raise ValueError(f"taxonomy row {index} has an empty field")
        if any(item != item.strip() for item in row.values()):
            raise ValueError(f"taxonomy row {index} fields must be normalized")
        rows.append(row)
        supplied_rows.append(raw)
    if not rows:
        raise ValueError("taxonomy must not be empty")
    labels = [row["label"] for row in rows]
    if len(labels) != len(set(labels)):
        raise ValueError("taxonomy labels must be unique")
    return tuple(rows), tuple(supplied_rows)


def _validated_vectors(
    vectors: Any,
    *,
    expected_count: int,
    expected_dimension: int | None = None,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(vectors, Sequence) or isinstance(vectors, (str, bytes)):
        raise ValueError("embedding provider returned an invalid collection")
    if len(vectors) != expected_count:
        raise ValueError("embedding coverage is incomplete")
    rows: list[tuple[float, ...]] = []
    dimension = expected_dimension
    for vector in vectors:
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) or not vector:
            raise ValueError("embedding vectors must be nonempty")
        try:
            row = tuple(float(value) for value in vector)
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding vectors must be numeric") from exc
        if any(not isfinite(value) for value in row):
            raise ValueError("embedding vectors must be finite")
        if dimension is None:
            dimension = len(row)
        if len(row) != dimension:
            raise ValueError("embedding dimensions must match")
        rows.append(row)
    return tuple(rows)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    denominator = left_norm * right_norm
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def retrieve_taxonomy_neighborhoods_v44(
    *,
    claim_bundles: Sequence[ClaimEvidenceBundleV44],
    taxonomy: Sequence[Any],
    embedder: Any,
    top_k: int,
) -> TaxonomyNeighborhoodManifestV44:
    """Rank a deterministic local taxonomy neighborhood for each claim bundle."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    bundles = tuple(claim_bundles)
    if not bundles:
        raise ValueError("claim bundles must not be empty")
    target_ids = [row.target_id for row in bundles]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("claim bundle targets must be unique")
    rows, supplied_rows = _taxonomy_rows(taxonomy)

    identity = embedding_identity(embedder)
    model = identity.get("model")
    revision = identity.get("revision")
    if not isinstance(model, str) or not model.strip() or not isinstance(revision, str) or not revision.strip():
        raise ValueError("embedding identity must include a pinned model and revision")

    documents = [
        f"{row['label']} {row['coarse_parent']} {row['definition']}" for row in rows
    ]
    queries: list[str] = []
    for bundle in bundles:
        eligible_text = _unique(
            tuple(
                evidence.text
                for evidence in (
                    *bundle.wiki_evidence,
                    *bundle.historical_evidence,
                    *bundle.portfolio_disclosures,
                )
                if evidence.eligible
            )
        )
        queries.append(_compact(bundle.query, *eligible_text))

    document_vectors = _validated_vectors(
        embed_documents(embedder, documents),
        expected_count=len(documents),
    )
    query_vectors = _validated_vectors(
        embed_queries(embedder, queries),
        expected_count=len(queries),
        expected_dimension=len(document_vectors[0]),
    )
    bm25 = BM25Okapi([_tokens(document) for document in documents])
    neighborhoods: list[TaxonomyNeighborhoodV44] = []
    ordered_labels: list[str] = []

    for bundle, query, query_vector in zip(bundles, queries, query_vectors, strict=True):
        dense_scores = [
            _cosine(vector, query_vector) for vector in document_vectors
        ]
        lexical_scores = [float(value) for value in bm25.get_scores(_tokens(query))]
        dense_order = sorted(
            range(len(rows)), key=lambda index: (-dense_scores[index], rows[index]["label"])
        )
        lexical_order = sorted(
            (index for index in range(len(rows)) if lexical_scores[index] > 0.0),
            key=lambda index: (-lexical_scores[index], rows[index]["label"]),
        )
        rankings = [[rows[index]["label"] for index in dense_order]]
        if lexical_order:
            rankings.append([rows[index]["label"] for index in lexical_order])
        fused = reciprocal_rank_fusion(rankings)
        ordered = sorted(
            range(len(rows)),
            key=lambda index: (-fused[rows[index]["label"]], rows[index]["label"]),
        )
        boundary = fused[rows[ordered[min(top_k, len(ordered)) - 1]]["label"]]
        retained = [
            index for index in ordered if fused[rows[index]["label"]] >= boundary
        ]
        candidates: list[TaxonomyCandidateV44] = []
        previous_score: float | None = None
        current_rank = 0
        for position, index in enumerate(retained, start=1):
            row = rows[index]
            fused_score = fused[row["label"]]
            if previous_score is None or fused_score != previous_score:
                current_rank = position
                previous_score = fused_score
            candidates.append(
                TaxonomyCandidateV44(
                    taxonomy_label=row["label"],
                    definition=row["definition"],
                    coarse_parent=row["coarse_parent"],
                    dense_score=dense_scores[index],
                    lexical_score=lexical_scores[index],
                    fused_score=fused_score,
                    rank=current_rank,
                    target_id=bundle.target_id,
                    query=query,
                    selection_reason=(
                        "Selected by deterministic dense and BM25 reciprocal-rank fusion; "
                        "kth-score ties are retained."
                    ),
                )
            )
            if row["label"] not in ordered_labels:
                ordered_labels.append(row["label"])
        neighborhoods.append(
            TaxonomyNeighborhoodV44(
                target_id=bundle.target_id,
                query=query,
                candidates=tuple(candidates),
            )
        )

    return TaxonomyNeighborhoodManifestV44(
        schema_version="taxonomy-neighborhood-manifest-v4.4",
        taxonomy_sha256=_canonical_sha256(supplied_rows),
        embedding_model=model,
        embedding_revision=revision,
        claim_neighborhoods=tuple(neighborhoods),
        ordered_labels=tuple(ordered_labels),
    )
