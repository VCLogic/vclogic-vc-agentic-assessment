from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

import vc_clone_graph.retrieval_v44 as retrieval_v44_module

from vc_clone_graph.phase1_v44 import (
    ClaimEvidenceBundleV44,
    ClaimMapV44,
    ClaimRetrievalManifestV44,
    TaxonomyNeighborhoodManifestV44,
)
from vc_clone_graph.portfolio_memory import (
    DisclosureEvidence,
    PortfolioCandidate,
    PortfolioDisclosure,
)
from vc_clone_graph.precedents import (
    DecisionEvidence,
    PrecedentCorpus,
    PrecedentDecision,
    PrecedentEpisode,
    PrecedentSearchHit,
    PrecedentSearchResult,
    TranscriptTurn,
)
from vc_clone_graph.retrieval import ExactEvidence, SearchHit, reciprocal_rank_fusion
from vc_clone_graph.retrieval_v44 import (
    retrieve_claim_evidence_v44,
    retrieve_taxonomy_neighborhoods_v44,
)


class StableEmbedder:
    metadata = {
        "backend": "test",
        "model": "stable-embedding",
        "revision": "rev-1",
        "normalize": True,
    }

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float("risk" in text.casefold())] for text in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)


@dataclass(frozen=True)
class WikiChunkFixture:
    chunk_id: str
    source_path: str
    source_sha256: str
    text: str


class FakeWiki:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []
        self.reads: list[str] = []
        self.chunks = (
            WikiChunkFixture("W-common", "wiki/common.md", "a" * 64, "Exact common investor principle."),
            WikiChunkFixture("W-secondary", "wiki/secondary.md", "b" * 64, "Exact secondary investor principle."),
        )
        self._by_id = {row.chunk_id: row for row in self.chunks}

    def search(self, query: str, limit: int) -> list[SearchHit]:
        self.queries.append((query, limit))
        return [
            SearchHit("W-common", "wiki/common.md", "Common", "common investor", 0.9, ("dense", "lexical")),
            SearchHit("W-secondary", "wiki/secondary.md", "Secondary", "secondary investor", 0.4, ("lexical",)),
        ][:limit]

    def read(self, chunk_id: str) -> ExactEvidence:
        self.reads.append(chunk_id)
        row = self._by_id[chunk_id]
        return ExactEvidence(row.chunk_id, row.source_path, row.source_sha256, "Heading", row.text)


def _episode(slug: str, investor_text: str, status: str) -> PrecedentEpisode:
    turns = (
        TranscriptTurn(turn_index=0, speaker="Founder", text="Founder context for the company."),
        TranscriptTurn(turn_index=1, speaker="Avery", text=investor_text),
    )
    return PrecedentEpisode(
        episode_slug=slug,
        episode_number=int(slug.split("-", 1)[0]),
        source_path=f"episodes/{slug}.json",
        source_sha256=sha256(slug.encode()).hexdigest(),
        investor_aliases=("Avery",),
        investor_present=True,
        turns=turns,
        decision=PrecedentDecision(
            status=status,
            context="initial_panel",
            check_tier=None,
            conditions=(),
            evidence=(DecisionEvidence(turn_start=1, turn_end=1, text=investor_text),),
            audit_source="deterministic fixture",
            audit_notes="Observed decision.",
        ),
    )


def _precedents(*, filtered: bool = True) -> PrecedentCorpus:
    rows = (
        _episode("17-earlier", "Customer concentration risk makes this difficult for me.", "Out"),
        _episode("18-target", "Target-only private evidence must never be retrieved.", "In"),
        _episode("99-later", "Quantum retention and repeat demand make this investable.", "In"),
    )
    return PrecedentCorpus.from_episodes(
        rows,
        StableEmbedder(),
        target_slug="18-target" if filtered else None,
    )


class CountingPrecedent:
    def __init__(self, corpus: PrecedentCorpus) -> None:
        self._corpus = corpus
        self.target_slug = corpus.target_slug
        self.retrieval_calls = 0

    def list_episodes(self):
        return self._corpus.list_episodes()

    def search(self, *args: object, **kwargs: object):
        self.retrieval_calls += 1
        return self._corpus.search(*args, **kwargs)

    def open_transcript(self, *args: object, **kwargs: object):
        self.retrieval_calls += 1
        return self._corpus.open_transcript(*args, **kwargs)

    def open_decision(self, *args: object, **kwargs: object):
        self.retrieval_calls += 1
        return self._corpus.open_decision(*args, **kwargs)


class FakePortfolioIndex:
    def __init__(
        self,
        target_slug: str = "18-target",
        *,
        extra_disclosures: tuple[PortfolioDisclosure, ...] = (),
    ) -> None:
        disclosure = PortfolioDisclosure(
            disclosure_id="PM-" + "c" * 20,
            vc_slug="avery",
            company_name="OverlapCo",
            aliases=("OverlapCo",),
            descriptor="Retention analytics platform",
            relationship="investment",
            observed_overlap="possible",
            observed_consequence="permission_or_check_required",
            source_episode_slug="17-earlier",
            source_episode_number=17,
            evidence=(
                DisclosureEvidence(
                    source_sha256="d" * 64,
                    turn_index=3,
                    speaker="Avery",
                    text="I invested in OverlapCo for retention analytics.",
                ),
            ),
            confidence=1.0,
            validation_status="human_validated",
        )
        self.filtered = SimpleNamespace(
            target_episode_slug=target_slug,
            target_episode_number=(
                int(target_slug.split("-", 1)[0])
                if target_slug.split("-", 1)[0].isdecimal()
                else None
            ),
            disclosures=(disclosure, *extra_disclosures),
        )
        self.queries: list[tuple[str, int, int | None]] = []

    def search(
        self, query: str, *, limit: int, candidate_pool_k: int | None = None
    ) -> list[PortfolioCandidate]:
        self.queries.append((query, limit, candidate_pool_k))
        return [
            PortfolioCandidate(
                entity_id="PE-" + "e" * 20,
                company_name="OverlapCo",
                aliases=("OverlapCo",),
                descriptor="Retention analytics platform",
                disclosure_ids=("PM-" + "c" * 20,),
                score=0.75,
                retrieval_modes=("lexical", "dense"),
            )
        ][:limit]


def _claim_map() -> ClaimMapV44:
    payload = {
        "schema_version": "claim-map-v4.4",
        "episode_slug": "18-target",
        "material_claims": [
            {
                "claim_id": "C1",
                "claim_type": "material",
                "statement": "Quantum retention is strong",
                "topic": "Revenue quality",
                "decision_relevance": "Validates repeat demand",
                "pitch_evidence_ids": ["P-001"],
            }
        ],
        "adverse_claims": [
            {
                "claim_id": "C2",
                "claim_type": "adverse",
                "statement": "Customer concentration is high",
                "topic": "Revenue risk",
                "decision_relevance": "Could undermine durability",
                "pitch_evidence_ids": ["P-002"],
            }
        ],
        "unanswered_questions": [
            {
                "question_id": "Q1",
                "question": "Will pilots renew?",
                "why_material": "Renewal determines durability",
                "anchor_pitch_evidence_ids": ["P-003"],
            }
        ],
        "claim_coverage": [
            {"pitch_evidence_id": "P-001", "claim_id": "C1"},
            {"pitch_evidence_id": "P-002", "claim_id": "C2"},
            {"pitch_evidence_id": "P-003", "question_id": "Q1"},
        ],
    }
    return ClaimMapV44.model_validate_json(json.dumps(payload))


def _retrieve(**overrides: object) -> tuple[ClaimRetrievalManifestV44, FakeWiki, FakePortfolioIndex]:
    wiki = FakeWiki()
    portfolio = FakePortfolioIndex()
    arguments = {
        "claim_map": _claim_map(),
        "wiki_index": wiki,
        "precedent_corpus": _precedents(),
        "portfolio_index": portfolio,
        "target_episode_slug": "18-target",
        "top_k": 2,
        "max_wiki_reads": 1,
        "max_precedent_reads": 2,
    }
    arguments.update(overrides)
    return retrieve_claim_evidence_v44(**arguments), wiki, portfolio


def test_claim_retrieval_is_distinct_source_bound_audited_and_deterministic() -> None:
    first, wiki, portfolio = _retrieve()
    second, _, _ = _retrieve()

    assert [row.target_id for row in first.claim_bundles] == ["C1", "C2", "Q1"]
    queries = [row.query for row in first.claim_bundles]
    assert len(set(queries)) == 3
    assert queries[0] == "Quantum retention is strong | Revenue quality | Validates repeat demand"
    assert queries[2] == "Will pilots renew? | Renewal determines durability"
    assert [query for query, _ in wiki.queries] == queries
    assert [query for query, _, _ in portfolio.queries] == queries
    assert all(limit == 2 for _, limit in wiki.queries)
    assert all(limit == 2 and pool is None for _, limit, pool in portfolio.queries)

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert first.target_episode_excluded is True
    assert all(
        row.episode_slug != "18-target"
        for bundle in first.claim_bundles
        for row in bundle.historical_evidence
    )
    assert any(
        row.episode_slug == "99-later"
        for bundle in first.claim_bundles
        for row in bundle.historical_evidence
    ), "later-numbered non-target precedents remain eligible"

    registry_ids = [row.evidence_id for row in first.evidence_registry]
    assert registry_ids.count("W-common") == 1
    assert "PM-" + "c" * 20 in registry_ids
    assert all(len(row.source_sha256) == 64 for row in first.evidence_registry)
    assert any(not row.eligible for row in first.claim_bundles[0].wiki_evidence)
    assert all(bundle.retrieval_action_ids for bundle in first.claim_bundles)
    assert {row.action_kind for row in first.retrieval_actions} == {"search", "read"}
    assert any(row.warnings for row in first.retrieval_actions) is False
    expected_claim_hash = sha256(
        json.dumps(
            _claim_map().model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert first.claim_map_sha256 == expected_claim_hash

    round_trip = ClaimRetrievalManifestV44.model_validate_json(first.model_dump_json())
    assert round_trip == first


def test_wiki_read_budget_preserves_query_specific_visible_text() -> None:
    class QueryOrderedWiki(FakeWiki):
        def search(self, query: str, limit: int) -> list[SearchHit]:
            self.queries.append((query, limit))
            common = SearchHit(
                "W-common",
                "wiki/common.md",
                "Common",
                "common investor",
                0.9,
                ("dense", "lexical"),
            )
            secondary = SearchHit(
                "W-secondary",
                "wiki/secondary.md",
                "Secondary",
                "secondary investor",
                0.8,
                ("lexical",),
            )
            ordered = (
                [secondary, common]
                if query.startswith("Quantum")
                else [common, secondary]
            )
            return ordered[:limit]

    wiki = QueryOrderedWiki()
    result, _, _ = _retrieve(
        wiki_index=wiki,
        only_ids=("C1", "C2"),
        top_k=2,
        max_wiki_reads=1,
    )

    assert wiki.reads == ["W-secondary", "W-common"]
    c1, c2 = result.claim_bundles
    c1_rows = {row.evidence_id: row for row in c1.wiki_evidence}
    c2_rows = {row.evidence_id: row for row in c2.wiki_evidence}
    assert c1_rows["W-secondary"].eligible is True
    assert c1_rows["W-secondary"].text == "Exact secondary investor principle."
    assert c1_rows["W-common"].eligible is False
    assert c1_rows["W-common"].text == "common investor"
    assert c2_rows["W-common"].eligible is True
    assert c2_rows["W-common"].text == "Exact common investor principle."
    assert c2_rows["W-secondary"].eligible is False
    assert c2_rows["W-secondary"].text == "secondary investor"

    read_actions = [
        action
        for action in result.retrieval_actions
        if action.source_kind == "wiki" and action.action_kind == "read"
    ]
    assert [action.result_evidence_ids for action in read_actions] == [
        ("W-secondary",),
        ("W-common",),
    ]
    registry = {row.evidence_id: row for row in result.evidence_registry}
    assert registry["W-common"].text == "Exact common investor principle."
    assert registry["W-secondary"].text == "Exact secondary investor principle."


def test_unopened_wiki_hit_registry_retains_owner_chunk_text() -> None:
    result, wiki, _ = _retrieve(
        only_ids=("C1",),
        top_k=2,
        max_wiki_reads=1,
    )

    registry = {row.evidence_id: row for row in result.evidence_registry}
    indexed = {row.chunk_id: row for row in wiki.chunks}
    occurrence = {
        row.evidence_id: row for row in result.claim_bundles[0].wiki_evidence
    }["W-secondary"]
    assert occurrence.eligible is False
    assert occurrence.text == "secondary investor"
    assert registry["W-secondary"].text == indexed["W-secondary"].text


def test_only_ids_preserve_original_order_and_validate_input() -> None:
    partial, _, _ = _retrieve(only_ids=("Q1", "C1"))
    assert [row.target_id for row in partial.claim_bundles] == ["C1", "Q1"]
    assert partial.retrieval_actions[0].query_id.startswith("phase1-t01-")
    assert any(row.query_id.startswith("phase1-t03-") for row in partial.retrieval_actions)

    with pytest.raises(ValueError, match="unique"):
        _retrieve(only_ids=("C1", "C1"))
    with pytest.raises(ValueError, match="unknown"):
        _retrieve(only_ids=("C9",))
    with pytest.raises(ValueError, match="sequence"):
        _retrieve(only_ids="C1")


@pytest.mark.parametrize(
    ("field", "value"),
    [("top_k", 0), ("max_wiki_reads", 0), ("max_precedent_reads", -1)],
)
def test_claim_retrieval_rejects_nonpositive_budgets(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        _retrieve(**{field: value})


def test_claim_retrieval_fails_closed_when_target_filter_is_not_verified() -> None:
    with pytest.raises(ValueError, match="target.*excluded"):
        _retrieve(precedent_corpus=_precedents(filtered=False))
    with pytest.raises(ValueError, match="portfolio.*target"):
        _retrieve(portfolio_index=FakePortfolioIndex("19-wrong"))


def test_target_safe_precedent_proxy_rejects_hit_before_open() -> None:
    class InconsistentPrecedentAdapter:
        target_slug = "18-target"

        def __init__(self) -> None:
            self.search_calls = 0
            self.transcript_open_calls = 0
            self.decision_open_calls = 0

        def list_episodes(self):
            return _precedents().list_episodes()

        def search(self, query: str, **kwargs: object) -> PrecedentSearchResult:
            self.search_calls += 1
            return PrecedentSearchResult(
                query_id=str(kwargs["query_id"]),
                query=query,
                hits=(
                    PrecedentSearchHit(
                        episode_slug="18-target",
                        chunk_id="H-" + "f" * 20,
                        turn_start=0,
                        turn_end=1,
                        excerpt="Target evidence",
                        score=1.0,
                        retrieval_modes=("lexical",),
                    ),
                ),
                quality_findings=(),
            )

        def open_transcript(self, *args: object, **kwargs: object):
            self.transcript_open_calls += 1
            raise AssertionError("target transcript must not be delegated")

        def open_decision(self, *args: object, **kwargs: object):
            self.decision_open_calls += 1
            raise AssertionError("target decision must not be delegated")

    precedent = InconsistentPrecedentAdapter()
    with pytest.raises(ValueError, match="target episode.*historical search"):
        retrieve_claim_evidence_v44(
            claim_map=_claim_map(),
            wiki_index=None,
            precedent_corpus=precedent,
            portfolio_index=None,
            target_episode_slug="18-target",
            top_k=1,
            max_wiki_reads=1,
            max_precedent_reads=1,
            only_ids=("C1",),
        )

    assert precedent.search_calls == 1
    assert precedent.transcript_open_calls == 0
    assert precedent.decision_open_calls == 0


def test_zero_precedent_read_budget_retains_only_ineligible_search_hits() -> None:
    result, _, _ = _retrieve(
        only_ids=("C1",),
        top_k=2,
        max_precedent_reads=0,
    )

    historical = result.claim_bundles[0].historical_evidence
    assert len(historical) == 2
    assert all(not row.eligible for row in historical)
    registry = {row.evidence_id: row for row in result.evidence_registry}
    for occurrence in historical:
        source = registry[occurrence.evidence_id]
        expected_id = "H-" + sha256(
            (
                f"{source.source_sha256}\0{source.turn_start}\0"
                f"{source.turn_end}\0{source.text}"
            ).encode("utf-8")
        ).hexdigest()[:20]
        assert source.evidence_id == expected_id
        assert occurrence.text in source.text
    historical_actions = [
        row for row in result.retrieval_actions if row.source_kind == "historical"
    ]
    assert [row.action_kind for row in historical_actions] == ["search"]
    assert historical_actions[0].opened_episode_slugs == ()


def test_unopened_historical_hit_registry_retains_canonical_chunk_text() -> None:
    long_text = " ".join(["customer concentration risk"] * 100)
    corpus = PrecedentCorpus.from_episodes(
        (
            _episode("17-earlier", long_text, "Out"),
            _episode("18-target", "Target evidence.", "In"),
        ),
        StableEmbedder(),
        target_slug="18-target",
    )

    result = retrieve_claim_evidence_v44(
        claim_map=_claim_map(),
        wiki_index=None,
        precedent_corpus=corpus,
        portfolio_index=None,
        target_episode_slug="18-target",
        top_k=1,
        max_wiki_reads=1,
        max_precedent_reads=0,
        only_ids=("C1",),
    )

    occurrence = result.claim_bundles[0].historical_evidence[0]
    source = result.evidence_registry[0]
    expected_id = "H-" + sha256(
        (
            f"{source.source_sha256}\0{source.turn_start}\0"
            f"{source.turn_end}\0{source.text}"
        ).encode("utf-8")
    ).hexdigest()[:20]
    assert occurrence.eligible is False
    assert len(occurrence.text) == 800
    assert len(source.text) > len(occurrence.text)
    assert occurrence.text in source.text
    assert source.evidence_id == expected_id


def test_historical_search_does_not_claim_unopened_top_k_hits() -> None:
    result, _, _ = _retrieve(
        only_ids=("C1",),
        top_k=2,
        max_precedent_reads=1,
    )

    actions = [
        row for row in result.retrieval_actions if row.source_kind == "historical"
    ]
    search = next(row for row in actions if row.action_kind == "search")
    reads = [row for row in actions if row.action_kind == "read"]
    assert len(search.result_evidence_ids) == 2
    assert search.opened_episode_slugs == ()
    assert len(reads) == 1
    assert len(reads[0].opened_episode_slugs) == 1

    registry = {row.evidence_id: row for row in result.evidence_registry}
    opened = set(reads[0].opened_episode_slugs)
    unopened_ids = [
        evidence_id
        for evidence_id in search.result_evidence_ids
        if registry[evidence_id].episode_slug not in opened
    ]
    assert len(unopened_ids) == 1
    bundle_rows = {
        row.evidence_id: row for row in result.claim_bundles[0].historical_evidence
    }
    assert bundle_rows[unopened_ids[0]].eligible is False


def test_inconsistent_reported_historical_openings_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = retrieval_v44_module.retrieve_v4

    def inconsistent(**kwargs: object):
        result = original(**kwargs)
        return replace(result, opened_episode_slugs=("17-earlier",))

    monkeypatch.setattr(retrieval_v44_module, "retrieve_v4", inconsistent)
    with pytest.raises(ValueError, match="opened episode"):
        _retrieve(only_ids=("C1",), top_k=1, max_precedent_reads=1)


@pytest.mark.parametrize("contaminated_slug", ["18-target", "19-later"])
def test_portfolio_contamination_fails_before_any_retrieval(
    contaminated_slug: str,
) -> None:
    episode_number = int(contaminated_slug.split("-", 1)[0])
    contaminated = PortfolioDisclosure(
        disclosure_id="PM-" + "f" * 20,
        vc_slug="avery",
        company_name="HiddenCo",
        aliases=("HiddenCo",),
        descriptor="This disclosure is never returned by search",
        relationship="investment",
        observed_overlap="possible",
        observed_consequence="unclear",
        source_episode_slug=contaminated_slug,
        source_episode_number=episode_number,
        evidence=(
            DisclosureEvidence(
                source_sha256="f" * 64,
                turn_index=1,
                speaker="Avery",
                text="Contaminated disclosure.",
            ),
        ),
        confidence=1.0,
        validation_status="human_validated",
    )
    wiki = FakeWiki()
    precedent = CountingPrecedent(_precedents())
    portfolio = FakePortfolioIndex(extra_disclosures=(contaminated,))

    with pytest.raises(ValueError, match="portfolio.*temporal"):
        retrieve_claim_evidence_v44(
            claim_map=_claim_map(),
            wiki_index=wiki,
            precedent_corpus=precedent,
            portfolio_index=portfolio,
            target_episode_slug="18-target",
            top_k=2,
            max_wiki_reads=1,
            max_precedent_reads=1,
        )

    assert wiki.queries == []
    assert wiki.reads == []
    assert precedent.retrieval_calls == 0
    assert portfolio.queries == []


def test_portfolio_noncontiguous_locator_enumerates_actual_turns() -> None:
    portfolio = FakePortfolioIndex()
    disclosure = portfolio.filtered.disclosures[0]
    portfolio.filtered.disclosures = (
        disclosure.model_copy(
            update={
                "evidence": (
                    DisclosureEvidence(
                        source_sha256="d" * 64,
                        turn_index=1,
                        speaker="Avery",
                        text="First exact disclosure span.",
                    ),
                    DisclosureEvidence(
                        source_sha256="d" * 64,
                        turn_index=3,
                        speaker="Avery",
                        text="Second exact disclosure span.",
                    ),
                )
            }
        ),
    )

    result, _, _ = _retrieve(
        portfolio_index=portfolio,
        only_ids=("C1",),
    )

    row = next(
        item for item in result.evidence_registry if item.source_kind == "portfolio"
    )
    assert row.source_locator == "17-earlier:turns-1,3"


def test_empty_sources_produce_deterministic_empty_bundles_and_actions() -> None:
    arguments = {
        "claim_map": _claim_map(),
        "wiki_index": None,
        "precedent_corpus": None,
        "portfolio_index": None,
        "target_episode_slug": "18-target",
        "top_k": 2,
        "max_wiki_reads": 1,
        "max_precedent_reads": 0,
    }
    first = retrieve_claim_evidence_v44(**arguments)
    second = retrieve_claim_evidence_v44(**arguments)

    assert first == second
    assert first.retrieval_actions == ()
    assert first.evidence_registry == ()
    assert all(
        not bundle.wiki_evidence
        and not bundle.historical_evidence
        and not bundle.portfolio_disclosures
        and not bundle.retrieval_action_ids
        for bundle in first.claim_bundles
    )


def test_empty_hits_preserve_search_audit_without_evidence() -> None:
    class EmptyWiki(FakeWiki):
        def __init__(self) -> None:
            super().__init__()
            self.chunks = ()

        def search(self, query: str, limit: int) -> list[SearchHit]:
            self.queries.append((query, limit))
            return []

    class EmptyPortfolio(FakePortfolioIndex):
        def __init__(self) -> None:
            super().__init__()
            self.filtered.disclosures = ()

        def search(
            self, query: str, *, limit: int, candidate_pool_k: int | None = None
        ) -> list[PortfolioCandidate]:
            self.queries.append((query, limit, candidate_pool_k))
            return []

    precedent = PrecedentCorpus.from_episodes(
        (), StableEmbedder(), target_slug="18-target"
    )
    result, _, _ = _retrieve(
        wiki_index=EmptyWiki(),
        precedent_corpus=precedent,
        portfolio_index=EmptyPortfolio(),
        only_ids=("C1",),
    )

    bundle = result.claim_bundles[0]
    assert bundle.wiki_evidence == ()
    assert bundle.historical_evidence == ()
    assert bundle.portfolio_disclosures == ()
    actions = [
        action
        for action in result.retrieval_actions
        if action.target_id == bundle.target_id
    ]
    assert [action.source_kind for action in actions] == [
        "wiki",
        "historical",
        "portfolio",
    ]
    assert all(
        action.action_kind == "search" and not action.result_evidence_ids
        for action in actions
    )


def test_empty_only_ids_returns_without_touching_sources() -> None:
    class ExplodingSource:
        def __init__(self) -> None:
            self.calls = 0

        def __getattr__(self, name: str):
            self.calls += 1
            raise AssertionError(f"source accessed through {name}")

    wiki = ExplodingSource()
    precedent = ExplodingSource()
    portfolio = ExplodingSource()
    result = retrieve_claim_evidence_v44(
        claim_map=_claim_map(),
        wiki_index=wiki,
        precedent_corpus=precedent,
        portfolio_index=portfolio,
        target_episode_slug="18-target",
        top_k=2,
        max_wiki_reads=1,
        max_precedent_reads=0,
        only_ids=(),
    )

    assert result.claim_bundles == ()
    assert result.evidence_registry == ()
    assert result.retrieval_actions == ()
    assert wiki.calls == precedent.calls == portfolio.calls == 0


def test_claim_retrieval_propagates_fallback_warnings_and_exact_investor_words() -> None:
    class QueryFailEmbedder(StableEmbedder):
        def embed_queries(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("deterministic query failure")

    rows = (
        _episode("18-target", "Target-only private evidence must never be retrieved.", "In"),
        _episode("99-later", "Quantum retention and repeat demand make this investable.", "In"),
    )
    corpus = PrecedentCorpus.from_episodes(
        rows,
        QueryFailEmbedder(),
        target_slug="18-target",
    )

    result, _, _ = _retrieve(
        precedent_corpus=corpus,
        only_ids=("C1",),
        top_k=1,
        max_precedent_reads=1,
    )

    assert result.warnings
    assert result.claim_bundles[0].warnings == result.warnings
    assert any(action.warnings for action in result.retrieval_actions)
    assert any(row.warning for row in result.claim_bundles[0].historical_evidence)
    assert any(
        row.text == "Quantum retention and repeat demand make this investable."
        and row.eligible
        and row.decision_status == "In"
        for row in result.claim_bundles[0].historical_evidence
    )


class RecordingEmbedder:
    metadata = {
        "backend": "test",
        "model": "taxonomy-embedder",
        "revision": "sha256:abc123",
        "normalize": False,
    }

    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("role-specific embedding methods must be used")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]][: len(texts)]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.query_calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


def _taxonomy() -> list[dict[str, str]]:
    return [
        {"label": "Alpha", "coarse_parent": "Traction", "definition": "investor principle"},
        {"label": "Beta", "coarse_parent": "Traction", "definition": "quantum retention"},
        {"label": "Gamma", "coarse_parent": "Team", "definition": "founder ability"},
    ]


def test_taxonomy_neighborhood_uses_dense_and_bm25_keeps_kth_ties() -> None:
    retrieval, _, _ = _retrieve(only_ids=("C1",))
    embedder = RecordingEmbedder()

    result = retrieve_taxonomy_neighborhoods_v44(
        claim_bundles=retrieval.claim_bundles,
        taxonomy=_taxonomy(),
        embedder=embedder,
        top_k=1,
    )
    repeated = retrieve_taxonomy_neighborhoods_v44(
        claim_bundles=retrieval.claim_bundles,
        taxonomy=_taxonomy(),
        embedder=RecordingEmbedder(),
        top_k=1,
    )

    assert embedder.document_calls == [[
        "Alpha Traction investor principle",
        "Beta Traction quantum retention",
        "Gamma Team founder ability",
    ]]
    assert len(embedder.query_calls) == 1
    assert embedder.query_calls[0][0].startswith(retrieval.claim_bundles[0].query)
    assert "Exact common investor principle." in embedder.query_calls[0][0]
    candidates = result.claim_neighborhoods[0].candidates
    assert [row.taxonomy_label for row in candidates] == ["Alpha", "Beta"]
    assert [row.rank for row in candidates] == [1, 1]
    assert all(row.coarse_parent == "Traction" for row in candidates)
    assert [row.definition for row in candidates] == ["investor principle", "quantum retention"]
    expected_fused = reciprocal_rank_fusion(
        [["Alpha", "Beta", "Gamma"], ["Beta", "Alpha", "Gamma"]]
    )
    assert [row.fused_score for row in candidates] == [
        expected_fused["Alpha"],
        expected_fused["Beta"],
    ]
    assert result.ordered_labels == ("Alpha", "Beta")
    assert result == repeated
    assert result.embedding_model == "taxonomy-embedder"
    assert result.embedding_revision == "sha256:abc123"
    assert len(result.taxonomy_sha256) == 64
    assert TaxonomyNeighborhoodManifestV44.model_validate_json(result.model_dump_json()) == result

    annotated = [dict(row, status="active") for row in _taxonomy()]
    changed = [dict(row, status="retired") for row in _taxonomy()]
    annotated_result = retrieve_taxonomy_neighborhoods_v44(
        claim_bundles=retrieval.claim_bundles,
        taxonomy=annotated,
        embedder=RecordingEmbedder(),
        top_k=1,
    )
    changed_result = retrieve_taxonomy_neighborhoods_v44(
        claim_bundles=retrieval.claim_bundles,
        taxonomy=changed,
        embedder=RecordingEmbedder(),
        top_k=1,
    )
    assert annotated_result.taxonomy_sha256 != changed_result.taxonomy_sha256


def test_all_zero_lexical_scores_use_dense_ranking_only() -> None:
    class DenseOnlyEmbedder:
        metadata = {
            "backend": "test",
            "model": "dense-only",
            "revision": "rev-1",
        }

        def embed(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("role-specific methods required")

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 1.0], [1.0, 0.0]]

        def embed_queries(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]]

    bundle = ClaimEvidenceBundleV44(
        target_kind="claim",
        target_id="C1",
        query="zzzz-no-lexical-match",
        wiki_evidence=(),
        historical_evidence=(),
        portfolio_disclosures=(),
        warnings=(),
        retrieval_action_ids=(),
    )
    taxonomy = [
        {"label": "Alpha", "coarse_parent": "One", "definition": "apple"},
        {"label": "Beta", "coarse_parent": "Two", "definition": "banana"},
    ]

    result = retrieve_taxonomy_neighborhoods_v44(
        claim_bundles=(bundle,),
        taxonomy=taxonomy,
        embedder=DenseOnlyEmbedder(),
        top_k=1,
    )

    assert [
        row.taxonomy_label for row in result.claim_neighborhoods[0].candidates
    ] == ["Beta"]
    assert result.claim_neighborhoods[0].candidates[0].lexical_score == 0.0


def test_raw_dense_score_ties_are_stable_by_label_without_lexical_hits() -> None:
    class TiedDenseEmbedder:
        metadata = {
            "backend": "test",
            "model": "tied-dense",
            "revision": "rev-1",
        }

        def embed(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("role-specific methods required")

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        def embed_queries(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    bundle = ClaimEvidenceBundleV44(
        target_kind="claim",
        target_id="C1",
        query="zzzz-no-lexical-match",
        wiki_evidence=(),
        historical_evidence=(),
        portfolio_disclosures=(),
        warnings=(),
        retrieval_action_ids=(),
    )
    result = retrieve_taxonomy_neighborhoods_v44(
        claim_bundles=(bundle,),
        taxonomy=[
            {"label": "Beta", "coarse_parent": "Two", "definition": "banana"},
            {"label": "Alpha", "coarse_parent": "One", "definition": "apple"},
        ],
        embedder=TiedDenseEmbedder(),
        top_k=1,
    )

    assert [
        row.taxonomy_label for row in result.claim_neighborhoods[0].candidates
    ] == ["Alpha"]


class InvalidVectorEmbedder(RecordingEmbedder):
    def __init__(self, case: str) -> None:
        super().__init__()
        self.case = case

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.case == "count":
            return [[1.0, 0.0]]
        if self.case == "empty":
            return [[] for _ in texts]
        if self.case == "ragged":
            return [[1.0], [1.0, 0.0], [1.0, 0.0]]
        if self.case == "nonfinite":
            return [[float("nan"), 0.0] for _ in texts]
        return super().embed_documents(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        if self.case == "dimension":
            return [[1.0] for _ in texts]
        return super().embed_queries(texts)


@pytest.mark.parametrize("case", ["count", "empty", "ragged", "nonfinite", "dimension"])
def test_taxonomy_neighborhood_rejects_incomplete_vectors_without_fallback(case: str) -> None:
    retrieval, _, _ = _retrieve(only_ids=("C1",))
    with pytest.raises(ValueError, match="embedding"):
        retrieve_taxonomy_neighborhoods_v44(
            claim_bundles=retrieval.claim_bundles,
            taxonomy=_taxonomy(),
            embedder=InvalidVectorEmbedder(case),
            top_k=1,
        )


@pytest.mark.parametrize(
    "taxonomy",
    [
        [{"label": "Alpha", "coarse_parent": "Traction"}],
        [{"label": "", "coarse_parent": "Traction", "definition": "signal"}],
        [
            {"label": "Alpha", "coarse_parent": "Traction", "definition": "one"},
            {"label": "Alpha", "coarse_parent": "Other", "definition": "two"},
        ],
    ],
)
def test_taxonomy_neighborhood_rejects_invalid_taxonomy(taxonomy: list[dict[str, str]]) -> None:
    retrieval, _, _ = _retrieve(only_ids=("C1",))
    with pytest.raises(ValueError, match="taxonomy"):
        retrieve_taxonomy_neighborhoods_v44(
            claim_bundles=retrieval.claim_bundles,
            taxonomy=taxonomy,
            embedder=RecordingEmbedder(),
            top_k=1,
        )


def test_taxonomy_neighborhood_rejects_unpinned_identity_and_invalid_targets() -> None:
    retrieval, _, _ = _retrieve(only_ids=("C1",))
    embedder = RecordingEmbedder()
    embedder.metadata = {"backend": "test", "model": "taxonomy-embedder", "revision": None}
    with pytest.raises(ValueError, match="identity"):
        retrieve_taxonomy_neighborhoods_v44(
            claim_bundles=retrieval.claim_bundles,
            taxonomy=_taxonomy(),
            embedder=embedder,
            top_k=1,
        )
    with pytest.raises(ValueError, match="top_k"):
        retrieve_taxonomy_neighborhoods_v44(
            claim_bundles=retrieval.claim_bundles,
            taxonomy=_taxonomy(),
            embedder=RecordingEmbedder(),
            top_k=0,
        )
    with pytest.raises(ValueError, match="unique"):
        retrieve_taxonomy_neighborhoods_v44(
            claim_bundles=(retrieval.claim_bundles[0], retrieval.claim_bundles[0]),
            taxonomy=_taxonomy(),
            embedder=RecordingEmbedder(),
            top_k=1,
        )
