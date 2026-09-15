from __future__ import annotations

from pathlib import Path

import pytest

from vc_clone_graph.portfolio_memory import (
    DisclosureEvidence,
    PortfolioDisclosure,
    PortfolioMemoryCorpus,
    PortfolioMemoryIndex,
    make_disclosure_id,
)


class FakeEmbedder:
    metadata = {"backend": "fake", "model": "fake", "revision": "one"}

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_queries(self, texts):
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text):
        lower = text.lower()
        return [float("calendar" in lower or "scheduling" in lower), float("food" in lower)]


def disclosure(
    episode: int | None,
    company: str | None,
    descriptor: str,
    *,
    turn: int = 1,
    overlap: str = "disclosure_only",
):
    slug = f"{episode}-episode" if episode is not None else "special-episode"
    return PortfolioDisclosure(
        disclosure_id=make_disclosure_id("vc", slug, turn, company or descriptor),
        vc_slug="vc",
        company_name=company,
        aliases=(company,) if company else (),
        descriptor=descriptor,
        relationship="investment",
        observed_overlap=overlap,
        observed_consequence="unclear",
        source_episode_slug=slug,
        source_episode_number=episode,
        evidence=(
            DisclosureEvidence(
                source_sha256="a" * 64,
                turn_index=turn,
                speaker="Investor",
                text=f"I invested in {company or descriptor}",
            ),
        ),
        confidence=0.9,
        validation_status="automated_candidate",
    )


def test_temporal_filter_is_strict_and_excludes_unordered():
    corpus = PortfolioMemoryCorpus(
        "vc",
        [
            disclosure(5, "OldCo", "calendar scheduling"),
            disclosure(10, "CurrentCo", "calendar scheduling"),
            disclosure(11, "FutureCo", "calendar scheduling"),
            disclosure(None, None, "anonymous calendar company"),
        ],
    )
    filtered = corpus.eligible_for_target("10-target")
    assert [row.company_name for row in filtered.disclosures] == ["OldCo"]
    assert filtered.excluded_current_or_later_count == 2
    assert filtered.excluded_unordered_count == 1
    unordered_target = corpus.eligible_for_target("special-target")
    assert unordered_target.target_episode_number is None
    assert unordered_target.disclosures == ()
    assert unordered_target.excluded_current_or_later_count == 3
    assert unordered_target.excluded_unordered_count == 1


def test_duplicate_source_span_is_rejected():
    row = disclosure(5, "OldCo", "calendar scheduling")
    with pytest.raises(ValueError, match="duplicate source span"):
        PortfolioMemoryCorpus("vc", [row, row.model_copy(update={"disclosure_id": "PM-" + "b" * 20})])


def test_named_events_aggregate_but_later_text_does_not_leak():
    corpus = PortfolioMemoryCorpus(
        "vc",
        [
            disclosure(5, "LetsMeet", "group scheduling"),
            disclosure(8, "letsmeet", "calendar coordination", turn=2),
            disclosure(12, "LetsMeet", "future secret product", turn=3),
            disclosure(6, None, "anonymous food company", turn=4),
        ],
    )
    index = PortfolioMemoryIndex.build(corpus, FakeEmbedder())
    filtered = index.for_target("10-target")
    assert len(filtered.entities) == 2
    named = next(row for row in filtered.entities if row.company_name)
    assert named.disclosure_ids == tuple(
        row.disclosure_id for row in corpus.disclosures
        if row.company_name and row.company_name.casefold() == "letsmeet"
        and row.source_episode_number < 10
    )
    assert "future secret" not in named.search_text
    assert next(row for row in filtered.entities if row.company_name is None).descriptor == "anonymous food company"


def test_hybrid_search_returns_company_level_candidates_with_provenance():
    corpus = PortfolioMemoryCorpus(
        "vc",
        [
            disclosure(5, "LetsMeet", "group scheduling"),
            disclosure(6, "SnackCo", "food snacks", turn=2),
        ],
    )
    filtered = PortfolioMemoryIndex.build(corpus, FakeEmbedder()).for_target("10-target")
    hits = filtered.search("calendar scheduling assistant", limit=2, candidate_pool_k=2)
    assert hits[0].company_name == "LetsMeet"
    assert hits[0].disclosure_ids == (corpus.disclosures[0].disclosure_id,)
    assert "dense" in hits[0].retrieval_modes


def test_index_round_trip_is_hash_and_embedding_bound(tmp_path: Path):
    corpus = PortfolioMemoryCorpus("vc", [disclosure(5, "LetsMeet", "group scheduling")])
    index = PortfolioMemoryIndex.build(corpus, FakeEmbedder())
    corpus_path = tmp_path / "disclosure-events.jsonl"
    index_path = tmp_path / "embedding-index.json"
    corpus.save_events(corpus_path)
    index.save(index_path, corpus_path)
    loaded_corpus = PortfolioMemoryCorpus.load_events("vc", corpus_path)
    loaded = PortfolioMemoryIndex.load(index_path, loaded_corpus, FakeEmbedder(), corpus_path)
    assert loaded.for_target("10-target").search("calendar", limit=1)[0].company_name == "LetsMeet"
    corpus_path.write_text(corpus_path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        PortfolioMemoryIndex.load(index_path, loaded_corpus, FakeEmbedder(), corpus_path)
