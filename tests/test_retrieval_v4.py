from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from vc_clone_graph.precedents import (
    DecisionEvidence,
    PrecedentCorpus,
    PrecedentDecision,
    PrecedentEpisode,
    TranscriptTurn,
)
from vc_clone_graph.retrieval import ExactEvidence, SearchHit
from vc_clone_graph.retrieval_v4 import retrieve_v4


class Embedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float("hardware" in text.lower()), float("market" in text.lower())]
            for text in texts
        ]


class Wiki:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, limit: int) -> list[SearchHit]:
        self.queries.append(query)
        return [
            SearchHit("W-one", "one.md", "One", "Founder evidence", 1.0, ("dense",)),
            SearchHit("W-two", "two.md", "Two", "Market evidence", 0.5, ("dense",)),
        ][:limit]

    def read(self, chunk_id: str) -> ExactEvidence:
        return ExactEvidence(
            chunk_id,
            f"{chunk_id}.md",
            "a" * 64,
            chunk_id,
            f"Exact text for {chunk_id}",
        )


def episode(slug: str, decision: str, founder_text: str) -> PrecedentEpisode:
    turns = (
        TranscriptTurn(turn_index=0, speaker="Founder", text=founder_text),
        TranscriptTurn(turn_index=1, speaker="Charles", text=f"My decision is {decision}."),
    )
    return PrecedentEpisode(
        episode_slug=slug,
        episode_number=int(slug.split("-", 1)[0]),
        source_path=f"source/{slug}.json",
        source_sha256=sha256(slug.encode()).hexdigest(),
        investor_aliases=("Charles",),
        investor_present=True,
        turns=turns,
        decision=PrecedentDecision(
            status=decision,
            context="initial_panel",
            check_tier=None,
            conditions=(),
            evidence=(DecisionEvidence(turn_start=1, turn_end=1, text=turns[1].text),),
            audit_source="test ledger",
            audit_notes="Observed.",
        ),
    )


def corpus() -> PrecedentCorpus:
    return PrecedentCorpus.from_episodes(
        (
            episode("18-rowvigor", "Out", "Target hardware pitch."),
            episode("20-harper-wilde", "Out", "A hardware consumer market."),
            episode("39-example", "In", "A software market wedge."),
        ),
        Embedder(),
        target_slug="18-rowvigor",
    )


def corpus_with_unobserved_intro_noise() -> PrecedentCorpus:
    observed_turns = tuple(
        [
            TranscriptTurn(
                turn_index=0,
                speaker="Narrator",
                text="Quasar quasar quasar. Meet today's investors.",
            )
        ]
        + [
            TranscriptTurn(
                turn_index=1,
                speaker="Charles",
                text="I’m Charles Hudson, managing partner at Precursor Ventures.",
            )
        ]
        + [
            TranscriptTurn(
                turn_index=index,
                speaker="Founder",
                text=f"Opening company statement {index}.",
            )
            for index in range(2, 9)
        ]
        + [
            TranscriptTurn(
                turn_index=9,
                speaker="Charles",
                text="The founder learned quickly, and I will invest.",
            )
        ]
    )
    observed = PrecedentEpisode(
        episode_slug="20-observed",
        episode_number=20,
        source_path="source/20-observed.json",
        source_sha256="c" * 64,
        investor_aliases=("Charles",),
        investor_present=True,
        turns=observed_turns,
        decision=PrecedentDecision(
            status="In",
            context="initial_panel",
            check_tier=None,
            conditions=(),
            evidence=(
                DecisionEvidence(
                    turn_start=9,
                    turn_end=9,
                    text=observed_turns[9].text,
                ),
            ),
            audit_source="test ledger",
            audit_notes="Observed.",
        ),
    )
    unobserved_turn = TranscriptTurn(
        turn_index=0,
        speaker="Narrator",
        text="quasar quasar quasar quasar quasar",
    )
    unobserved = PrecedentEpisode(
        episode_slug="22-unobserved",
        episode_number=22,
        source_path="source/22-unobserved.json",
        source_sha256="d" * 64,
        investor_aliases=("Charles",),
        investor_present=False,
        turns=(unobserved_turn,),
        decision=PrecedentDecision(
            status="unobserved",
            context="unclear",
            check_tier=None,
            conditions=(),
            evidence=(),
            audit_source="test ledger",
            audit_notes="No observed decision.",
        ),
    )
    return PrecedentCorpus.from_episodes((observed, unobserved))


def test_retrieval_automatically_opens_top_exact_evidence_and_decision() -> None:
    result = retrieve_v4(
        wiki_index=Wiki(),
        precedent_corpus=corpus(),
        wiki_queries=["founder"],
        precedent_queries=["hardware"],
        top_k=2,
        max_wiki_reads=1,
        max_precedent_reads=1,
        phase="phase1",
        turn=1,
    )

    assert [row["chunk_id"] for row in result.wiki_evidence] == ["W-one"]
    assert result.opened_episode_slugs == ("20-harper-wilde",)
    assert {row["episode_slug"] for row in result.historical_evidence} == {
        "20-harper-wilde"
    }
    assert any(row.get("decision_status") == "Out" for row in result.historical_evidence)
    assert all(row["episode_slug"] != "18-rowvigor" for row in result.historical_evidence)


def test_retrieval_deduplicates_hits_and_applies_limits_per_iteration() -> None:
    first = retrieve_v4(
        wiki_index=Wiki(),
        precedent_corpus=corpus(),
        wiki_queries=["founder", "founder"],
        precedent_queries=["hardware", "hardware"],
        top_k=2,
        max_wiki_reads=2,
        max_precedent_reads=1,
        phase="phase1",
        turn=1,
    )
    second = retrieve_v4(
        wiki_index=Wiki(),
        precedent_corpus=corpus(),
        wiki_queries=["market"],
        precedent_queries=["market"],
        top_k=2,
        max_wiki_reads=1,
        max_precedent_reads=2,
        phase="phase1",
        turn=2,
    )

    assert len(first.wiki_evidence) == 2
    assert len(first.opened_episode_slugs) == 1
    assert len(second.wiki_evidence) == 1
    assert len(second.opened_episode_slugs) == 2
    assert all(row["query_id"].startswith("phase1-t02") for row in second.historical_evidence)


def test_phase2_retrieval_can_request_balanced_decision_boundary() -> None:
    balanced = PrecedentCorpus.from_episodes(
        (
            episode("18-near-in", "In", "founder conviction execution"),
            episode("20-near-out", "Out", "founder conviction execution"),
            episode("22-other-in", "In", "founder execution"),
            episode("24-other-out", "Out", "founder execution"),
        )
    )
    result = retrieve_v4(
        wiki_index=None,
        precedent_corpus=balanced,
        wiki_queries=[],
        precedent_queries=["founder conviction execution"],
        top_k=4,
        max_wiki_reads=0,
        max_precedent_reads=4,
        phase="phase2",
        turn=1,
        selection_policy="contrastive",
        candidate_pool_k=4,
        in_slots=2,
        out_slots=2,
    )

    statuses = [row["decision"]["decision"]["status"] for row in result.precedent_reads]
    assert statuses.count("In") == 2
    assert statuses.count("Out") == 2


def test_v4_retrieval_opens_only_observed_substantive_precedents() -> None:
    result = retrieve_v4(
        wiki_index=None,
        precedent_corpus=corpus_with_unobserved_intro_noise(),
        wiki_queries=[],
        precedent_queries=["quasar"],
        top_k=3,
        max_wiki_reads=0,
        max_precedent_reads=3,
        phase="phase1",
        turn=1,
    )

    assert result.opened_episode_slugs == ("20-observed",)
    assert len(result.precedent_reads) == 1
    read = result.precedent_reads[0]
    assert read["decision"]["decision"]["status"] == "In"
    assert read["transcript"]["turn_start"] == 8
    assert read["transcript"]["turn_end"] == 9
