from hashlib import sha256
import json
from pathlib import Path

from pydantic import ValidationError
import pytest

from vc_clone_graph.precedents import (
    DecisionEvidence,
    PrecedentCorpus,
    PrecedentDecision,
    PrecedentEpisode,
    TranscriptTurn,
)
from vc_clone_graph.precedent_builder import build_precedent_corpus


class TinyEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float("quasar" in text.lower()), float("garden" in text.lower())]
            for text in texts
        ]


class FailingEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("local embedding failed")


class QueryFailingEmbedder(TinyEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        if len(texts) == 1 and texts[0] == "quasar":
            raise RuntimeError("local query embedding failed")
        return super().embed(texts)


class SemanticEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "semantic concept": [1.0, 0.0],
            "Narrator: alpha topic": [0.0, 1.0],
            "Narrator: beta topic": [1.0, 0.0],
        }
        return [vectors[text] for text in texts]


class CountingEmbedder(TinyEmbedder):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return super().embed(texts)


class MetadataEmbedder(TinyEmbedder):
    def __init__(self, model: str = "fixture-embed-v1") -> None:
        self._model = model

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "backend": "sentence_transformers",
            "model": self._model,
            "revision": "abc123",
            "normalize": True,
            "document_prefix": "search_document: ",
            "query_prefix": "search_query: ",
        }


def episode(
    slug: str,
    decision: str = "Out",
    *,
    conditions: tuple[str, ...] = (),
    turn_count: int = 2,
) -> PrecedentEpisode:
    turns = tuple(
        TranscriptTurn(
            turn_index=index,
            speaker="Charles" if index == turn_count - 1 else "Founder",
            text=(
                "This is not right for me."
                if index == turn_count - 1
                else f"We sell software for market {index}."
            ),
        )
        for index in range(turn_count)
    )
    return PrecedentEpisode(
        episode_slug=slug,
        episode_number=int(slug.split("-", 1)[0]),
        source_path=f"source/{slug}.json",
        source_sha256=sha256(slug.encode("utf-8")).hexdigest(),
        investor_aliases=("Charles", "Charles Hudson"),
        investor_present=True,
        turns=turns,
        decision=PrecedentDecision(
            status=decision,
            context="initial_panel",
            check_tier=None,
            conditions=conditions,
            evidence=(
                DecisionEvidence(
                    turn_start=turn_count - 1,
                    turn_end=turn_count - 1,
                    text=turns[-1].text,
                ),
            ),
            audit_source="evaluation/labels/charles_pitch_window_decisions.json",
            audit_notes="Observed on panel.",
        ),
    )


def unobserved_episode(
    slug: str, text: str, *, speaker: str = "Narrator"
) -> PrecedentEpisode:
    turn = TranscriptTurn(turn_index=0, speaker=speaker, text=text)
    return PrecedentEpisode(
        episode_slug=slug,
        episode_number=int(slug.split("-", 1)[0]),
        source_path=f"source/{slug}.json",
        source_sha256="b" * 64,
        investor_aliases=("Charles", "Charles Hudson"),
        investor_present=False,
        turns=(turn,),
        decision=PrecedentDecision(
            status="unobserved",
            context="unclear",
            check_tier=None,
            conditions=(),
            evidence=(),
            audit_source="source audit",
            audit_notes="No investment decision was observed.",
        ),
    )


def observed_episode(slug: str, decision: str, text: str) -> PrecedentEpisode:
    payload = episode(slug, decision).model_dump(mode="json")
    payload["turns"][0]["text"] = text
    return PrecedentEpisode.model_validate(payload)


def test_target_is_absent_from_every_operation() -> None:
    corpus = PrecedentCorpus.from_episodes(
        (episode("18-rowvigor"), episode("20-harper-wilde"))
    )
    view = corpus.for_target("20-harper-wilde")

    assert [row.episode_slug for row in view.list_episodes()] == ["18-rowvigor"]
    assert all(
        hit.episode_slug != "20-harper-wilde"
        for hit in view.search("software", 10)
    )
    with pytest.raises(PermissionError, match="target episode"):
        view.open_transcript("20-harper-wilde")
    with pytest.raises(PermissionError, match="target episode"):
        view.open_decision("20-harper-wilde")


def test_direct_target_construction_excludes_target_from_every_operation() -> None:
    view = PrecedentCorpus(
        (episode("18-rowvigor"), episode("20-harper-wilde")),
        target_slug="20-harper-wilde",
    )

    assert [row.episode_slug for row in view.list_episodes()] == ["18-rowvigor"]
    assert {
        hit.episode_slug for hit in view.search("software", 10)
    } == {"18-rowvigor"}
    with pytest.raises(PermissionError, match="target episode"):
        view.open_transcript("20-harper-wilde")
    with pytest.raises(PermissionError, match="target episode"):
        view.open_decision("20-harper-wilde")


def test_for_target_is_idempotent_and_manifest_stays_deterministic() -> None:
    once = PrecedentCorpus.from_episodes(
        (episode("20-harper-wilde"), episode("18-rowvigor"))
    ).for_target("20-harper-wilde")
    twice = once.for_target("20-harper-wilde")

    assert twice.list_episodes() == once.list_episodes()
    assert twice.filtered_manifest() == once.filtered_manifest()


def test_target_absent_from_source_is_still_inaccessible() -> None:
    view = PrecedentCorpus.from_episodes((episode("18-rowvigor"),)).for_target(
        "99-target"
    )

    assert [row.episode_slug for row in view.list_episodes()] == ["18-rowvigor"]
    assert {hit.episode_slug for hit in view.search("software", 10)} == {
        "18-rowvigor"
    }
    with pytest.raises(PermissionError, match="99-target"):
        view.open_transcript("99-target")
    with pytest.raises(PermissionError, match="99-target"):
        view.open_decision("99-target")


def test_non_top_k_episode_remains_directly_openable() -> None:
    corpus = PrecedentCorpus.from_episodes(
        (episode("18-rowvigor"), episode("20-harper-wilde"))
    )
    view = corpus.for_target("99-target")

    hits = view.search("software", 1)

    assert len(hits) == 1
    omitted = ({"18-rowvigor", "20-harper-wilde"} - {hits[0].episode_slug}).pop()
    assert view.open_transcript(omitted).turns[-1].text == "This is not right for me."


def test_decision_evidence_must_equal_source_turns() -> None:
    payload = episode("18-rowvigor").model_dump(mode="json")
    payload["decision"]["evidence"][0]["text"] = "changed words"

    with pytest.raises(ValidationError, match="decision evidence"):
        PrecedentEpisode.model_validate(payload)


def test_unobserved_requires_no_evidence_and_remains_unobserved() -> None:
    row = episode("18-rowvigor").model_copy(
        update={
            "decision": PrecedentDecision(
                status="unobserved",
                context="unclear",
                check_tier=None,
                conditions=(),
                evidence=(),
                audit_source="source audit",
                audit_notes="No investment decision was observed.",
            )
        }
    )
    corpus = PrecedentCorpus.from_episodes((row,))

    assert corpus.list_episodes()[0].decision_status == "unobserved"
    assert corpus.open_decision("18-rowvigor").status == "unobserved"


def test_observed_decision_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="observed decision"):
        PrecedentDecision(
            status="Out",
            context="initial_panel",
            check_tier=None,
            conditions=(),
            evidence=(),
            audit_source="source audit",
            audit_notes="Observed on panel.",
        )


def test_unobserved_decision_rejects_evidence() -> None:
    with pytest.raises(ValidationError, match="unobserved decision"):
        PrecedentDecision(
            status="unobserved",
            context="unclear",
            check_tier=None,
            conditions=(),
            evidence=(
                DecisionEvidence(turn_start=0, turn_end=0, text="Not a decision."),
            ),
            audit_source="source audit",
            audit_notes="No investment decision was observed.",
        )


def test_conditional_in_preserves_status_and_conditions() -> None:
    row = episode(
        "41-can-this-startup-help-retailers-take-on-amazon",
        "In",
        conditions=("Subject to customer references",),
    )

    decision = PrecedentCorpus.from_episodes((row,)).open_decision(
        row.episode_slug
    )

    assert decision.status == "In"
    assert decision.conditions == ("Subject to customer references",)


def test_filtered_manifest_is_deterministic_hash_bound_and_excludes_target() -> None:
    first = PrecedentCorpus.from_episodes(
        (episode("20-harper-wilde"), episode("18-rowvigor"))
    ).for_target("20-harper-wilde")
    second = PrecedentCorpus.from_episodes(
        (episode("18-rowvigor"), episode("20-harper-wilde"))
    ).for_target("20-harper-wilde")

    manifest = first.filtered_manifest()
    payload = manifest.model_dump(mode="json", exclude={"sha256"})
    expected = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert manifest == second.filtered_manifest()
    assert manifest.sha256 == expected
    assert manifest.target_episode_slug == "20-harper-wilde"
    assert [row.episode_slug for row in manifest.accessible_episodes] == [
        "18-rowvigor"
    ]


def test_duplicate_episode_slugs_are_rejected() -> None:
    row = episode("18-rowvigor")

    with pytest.raises(ValueError, match="unique"):
        PrecedentCorpus.from_episodes((row, row))


def test_nonnumeric_episode_slug_has_no_number_and_sorts_after_numeric() -> None:
    payload = episode("18-rowvigor").model_dump(mode="json")
    payload.update(
        {
            "episode_slug": "gnara-disrupting-your-pants",
            "episode_number": None,
            "source_path": "source/gnara-disrupting-your-pants.json",
            "source_sha256": sha256(
                b"gnara-disrupting-your-pants"
            ).hexdigest(),
        }
    )
    gnara = PrecedentEpisode.model_validate(payload)
    corpus = PrecedentCorpus.from_episodes(
        (gnara, episode("20-harper-wilde"), episode("18-rowvigor"))
    )

    summaries = corpus.list_episodes()

    assert [row.episode_slug for row in summaries] == [
        "18-rowvigor",
        "20-harper-wilde",
        "gnara-disrupting-your-pants",
    ]
    assert summaries[-1].episode_number is None


@pytest.mark.parametrize(
    ("start", "end"),
    ((-1, None), (1, 0), (0, 2), (2, None)),
)
def test_invalid_transcript_ranges_raise_index_error(
    start: int, end: int | None
) -> None:
    corpus = PrecedentCorpus.from_episodes((episode("18-rowvigor"),))

    with pytest.raises(IndexError, match="turn range"):
        corpus.open_transcript("18-rowvigor", start, end)


def test_omitted_end_opens_the_full_remaining_transcript() -> None:
    corpus = PrecedentCorpus.from_episodes((episode("18-rowvigor", turn_count=10),))

    read = corpus.open_transcript("18-rowvigor", start=1)

    assert read.turn_start == 1
    assert read.turn_end == 9
    assert tuple(turn.turn_index for turn in read.turns) == tuple(range(1, 10))


def test_unknown_non_target_episode_raises_key_error() -> None:
    view = PrecedentCorpus.from_episodes((episode("18-rowvigor"),)).for_target(
        "99-target"
    )

    with pytest.raises(KeyError, match="unknown precedent episode"):
        view.open_transcript("100-unknown")
    with pytest.raises(KeyError, match="unknown precedent episode"):
        view.open_decision("100-unknown")


def test_search_uses_stable_eight_turn_chunks() -> None:
    row = episode("18-rowvigor", turn_count=10)
    first = PrecedentCorpus.from_episodes((row,)).search("market", 10)
    second = PrecedentCorpus.from_episodes((row,)).search("market", 10)

    assert sorted((hit.turn_start, hit.turn_end) for hit in first) == [(0, 7), (8, 9)]
    assert [hit.chunk_id for hit in first] == [hit.chunk_id for hit in second]
    assert all(hit.chunk_id.startswith("H-") for hit in first)


def test_search_ranks_a_sole_matching_chunk_above_a_nonmatch() -> None:
    corpus = PrecedentCorpus.from_episodes(
        (
            unobserved_episode("18-rowvigor", "garden tools"),
            unobserved_episode("20-harper-wilde", "quasar analytics"),
        )
    )

    hits = corpus.search("quasar", 2)

    assert [hit.episode_slug for hit in hits] == [
        "20-harper-wilde",
        "18-rowvigor",
    ]
    assert hits[0].score > hits[1].score


def test_contrastive_search_selects_nearest_unique_ins_and_outs() -> None:
    corpus = PrecedentCorpus.from_episodes(
        (
            observed_episode("18-nearest-in", "In", "quasar quasar quasar"),
            observed_episode("20-nearest-out", "Out", "quasar quasar"),
            observed_episode("22-second-in", "In", "quasar market"),
            observed_episode("24-second-out", "Out", "quasar product"),
            unobserved_episode("26-unobserved", "quasar quasar quasar quasar"),
        )
    )

    result = corpus.search(
        "quasar",
        top_k=4,
        query_id="Q-contrastive",
        selection_policy="contrastive",
        candidate_pool_k=5,
        in_slots=2,
        out_slots=2,
    )

    assert len(result.hits) == 4
    assert {hit.episode_slug for hit in result.hits} == {
        "18-nearest-in",
        "20-nearest-out",
        "22-second-in",
        "24-second-out",
    }
    assert len({hit.episode_slug for hit in result.hits}) == len(result.hits)
    assert [
        corpus.open_decision(hit.episode_slug).status for hit in result.hits
    ].count("In") == 2
    assert [
        corpus.open_decision(hit.episode_slug).status for hit in result.hits
    ].count("Out") == 2


def test_semantic_search_can_require_observed_substantive_investor_chunks() -> None:
    turns = (
        TranscriptTurn(
            turn_index=0,
            speaker="Narrator",
            text="Quasar quasar quasar. Welcome to the show and meet the investors.",
        ),
        TranscriptTurn(
            turn_index=1,
            speaker="Charles",
            text="I’m Charles Hudson, managing partner at Precursor Ventures.",
        ),
        *(
            TranscriptTurn(
                turn_index=index,
                speaker="Founder",
                text=f"Opening pitch detail {index}.",
            )
            for index in range(2, 9)
        ),
        TranscriptTurn(
            turn_index=9,
            speaker="Charles",
            text="The customer evidence is compelling, so I am in.",
        ),
    )
    observed = PrecedentEpisode(
        episode_slug="20-observed",
        episode_number=20,
        source_path="source/20-observed.json",
        source_sha256="c" * 64,
        investor_aliases=("Charles", "Charles Hudson"),
        investor_present=True,
        turns=turns,
        decision=PrecedentDecision(
            status="In",
            context="initial_panel",
            check_tier=None,
            conditions=(),
            evidence=(
                DecisionEvidence(
                    turn_start=9,
                    turn_end=9,
                    text=turns[9].text,
                ),
            ),
            audit_source="test ledger",
            audit_notes="Observed on panel.",
        ),
    )
    corpus = PrecedentCorpus.from_episodes(
        (
            unobserved_episode(
                "18-unobserved",
                "quasar quasar quasar quasar quasar",
            ),
            observed,
        )
    )

    ordinary = corpus.search("quasar", top_k=2, query_id="Q-ordinary")
    filtered = corpus.search(
        "quasar",
        top_k=2,
        query_id="Q-filtered",
        observed_only=True,
        substantive_only=True,
    )

    assert ordinary.hits[0].episode_slug == "18-unobserved"
    assert [(hit.episode_slug, hit.turn_start, hit.turn_end) for hit in filtered.hits] == [
        ("20-observed", 8, 9)
    ]


def test_contrastive_search_fills_a_missing_class_by_similarity() -> None:
    corpus = PrecedentCorpus.from_episodes(
        (
            observed_episode("18-only-in", "In", "quasar quasar quasar"),
            observed_episode("20-first-out", "Out", "quasar quasar"),
            observed_episode("22-second-out", "Out", "quasar market"),
            observed_episode("24-third-out", "Out", "quasar product"),
        )
    )

    result = corpus.search(
        "quasar",
        top_k=4,
        query_id="Q-shortage",
        selection_policy="contrastive",
        candidate_pool_k=4,
        in_slots=2,
        out_slots=2,
    )

    assert len(result.hits) == 4
    assert [
        corpus.open_decision(hit.episode_slug).status for hit in result.hits
    ].count("In") == 1
    assert [
        corpus.open_decision(hit.episode_slug).status for hit in result.hits
    ].count("Out") == 3


def test_contrastive_search_preserves_target_exclusion() -> None:
    corpus = PrecedentCorpus.from_episodes(
        (
            observed_episode("18-visible-in", "In", "quasar"),
            observed_episode("20-target", "In", "quasar quasar quasar"),
            observed_episode("22-visible-out", "Out", "quasar"),
        )
    ).for_target("20-target")

    result = corpus.search(
        "quasar",
        top_k=2,
        query_id="Q-target",
        selection_policy="contrastive",
        candidate_pool_k=3,
        in_slots=1,
        out_slots=1,
    )

    assert {hit.episode_slug for hit in result.hits} == {
        "18-visible-in",
        "22-visible-out",
    }


def test_for_target_excludes_non_target_episode_that_mentions_target_entity() -> None:
    corpus = PrecedentCorpus.from_episodes(
        (
            observed_episode("18-visible-in", "In", "ordinary apparel company"),
            observed_episode("20-follow-up", "Out", "Bluffworks was an earlier pitch"),
            observed_episode("39-target", "In", "Bluffworks founder pitch"),
        )
    ).for_target("39-target", excluded_aliases=("Bluffworks",))

    assert [row.episode_slug for row in corpus.list_episodes()] == ["18-visible-in"]


def test_more_query_occurrences_do_not_rank_below_zero_occurrences() -> None:
    corpus = PrecedentCorpus.from_episodes(
        (
            unobserved_episode("18-rowvigor", "garden tools"),
            unobserved_episode("20-harper-wilde", "quasar quasar quasar"),
        )
    )

    hits = corpus.search("quasar", 2)

    assert hits[0].episode_slug == "20-harper-wilde"
    assert hits[0].score > hits[1].score


def test_search_rejects_a_query_without_lexical_tokens() -> None:
    corpus = PrecedentCorpus.from_episodes((episode("18-rowvigor"),))

    with pytest.raises(ValueError, match="token"):
        corpus.search("!!! 🚀", 5)


def test_search_omits_tokenless_chunks_but_direct_open_remains_available() -> None:
    tokenless = unobserved_episode("18-rowvigor", "!!!", speaker="🚀")
    searchable = unobserved_episode("20-harper-wilde", "quasar analytics")
    corpus = PrecedentCorpus.from_episodes((tokenless, searchable))

    hits = corpus.search("quasar", 10)

    assert [hit.episode_slug for hit in hits] == ["20-harper-wilde"]
    assert corpus.open_transcript("18-rowvigor").turns == tokenless.turns


def test_search_returns_empty_when_every_chunk_is_tokenless() -> None:
    tokenless = unobserved_episode("18-rowvigor", "!!!", speaker="🚀")
    corpus = PrecedentCorpus.from_episodes((tokenless,))

    assert corpus.search("quasar", 5) == ()
    assert corpus.open_decision("18-rowvigor").status == "unobserved"


def test_models_are_frozen_and_forbid_extra_fields() -> None:
    turn = TranscriptTurn(turn_index=0, speaker="Founder", text="Hello")

    with pytest.raises(ValidationError):
        turn.text = "Changed"
    with pytest.raises(ValidationError, match="Extra inputs"):
        TranscriptTurn(
            turn_index=0,
            speaker="Founder",
            text="Hello",
            unexpected=True,
        )


@pytest.mark.parametrize("embedder", [FailingEmbedder(), QueryFailingEmbedder()])
def test_embedding_failure_keeps_lexical_hits_and_reports_fallback(
    embedder: object,
) -> None:
    corpus = PrecedentCorpus.from_episodes(
        (
            unobserved_episode("18-rowvigor", "garden tools"),
            unobserved_episode("20-harper-wilde", "quasar analytics"),
        ),
        embedder=embedder,
    )

    result = corpus.search("quasar", top_k=2, query_id="Q-embedding")

    assert [hit.episode_slug for hit in result.hits] == [
        "20-harper-wilde",
        "18-rowvigor",
    ]
    assert result.hits[0].retrieval_modes == ("lexical",)
    assert result.hits[1].retrieval_modes == ()
    assert result.quality_findings == ("PRECEDENT_EMBEDDING_FALLBACK",)


def test_complete_precedent_index_fails_closed_on_embedding_error() -> None:
    with pytest.raises(RuntimeError, match="complete precedent embedding index"):
        PrecedentCorpus.from_episodes(
            (unobserved_episode("18-rowvigor", "garden tools"),),
            embedder=FailingEmbedder(),
            require_complete_embeddings=True,
        )


def test_complete_precedent_index_persists_metadata_and_rejects_model_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "precedents.json"
    PrecedentCorpus.from_episodes(
        (unobserved_episode("18-rowvigor", "garden tools"),),
        embedder=MetadataEmbedder(),
        require_complete_embeddings=True,
    ).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["embedding_index"]["coverage"] == 1.0
    assert payload["embedding_index"]["dimension"] == 2
    with pytest.raises(ValueError, match="embedding configuration mismatch"):
        PrecedentCorpus.load(
            path,
            embedder=MetadataEmbedder("different-model"),
            require_complete_embeddings=True,
        )


def test_hybrid_search_records_modes_and_ranking_is_deterministic() -> None:
    rows = (
        unobserved_episode("18-rowvigor", "garden tools"),
        unobserved_episode("20-harper-wilde", "quasar analytics"),
    )
    corpus = PrecedentCorpus.from_episodes(rows, embedder=TinyEmbedder())

    first = corpus.search("quasar", top_k=2, query_id="Q-first")
    second = corpus.search("quasar", top_k=2, query_id="Q-second")

    assert [hit.chunk_id for hit in first.hits] == [
        hit.chunk_id for hit in second.hits
    ]
    assert first.hits[0].episode_slug == "20-harper-wilde"
    assert first.hits[0].retrieval_modes == ("lexical", "dense")
    assert first.hits[1].retrieval_modes == ("dense",)
    assert first.quality_findings == ()


def test_semantic_only_query_is_not_boosted_by_zero_score_lexical_order() -> None:
    corpus = PrecedentCorpus.from_episodes(
        (
            unobserved_episode("18-rowvigor", "alpha topic"),
            unobserved_episode("20-harper-wilde", "beta topic"),
        ),
        embedder=SemanticEmbedder(),
    )

    result = corpus.search("semantic concept", top_k=2, query_id="Q-semantic")

    assert [hit.episode_slug for hit in result.hits] == [
        "20-harper-wilde",
        "18-rowvigor",
    ]
    assert all(hit.retrieval_modes == ("dense",) for hit in result.hits)


def test_for_target_reuses_non_target_chunks_without_embedding_calls() -> None:
    embedder = CountingEmbedder()
    corpus = PrecedentCorpus.from_episodes(
        (
            unobserved_episode("18-rowvigor", "garden tools"),
            unobserved_episode("20-harper-wilde", "quasar analytics"),
            unobserved_episode("99-target", "secret target"),
        ),
        embedder=embedder,
    )
    expected = {
        chunk.chunk_id: chunk.embedding
        for chunk in corpus._chunks
        if chunk.episode_slug != "99-target"
    }
    baseline = [
        hit.chunk_id
        for hit in corpus.search(
            "quasar", top_k=3, query_id="Q-before-filter"
        ).hits
        if hit.episode_slug != "99-target"
    ]
    embedder.calls.clear()

    view = corpus.for_target("99-target")

    assert embedder.calls == []
    assert {chunk.chunk_id: chunk.embedding for chunk in view._chunks} == expected
    assert view._embedding_fallback == corpus._embedding_fallback
    assert [
        hit.chunk_id
        for hit in view.search(
            "quasar", top_k=2, query_id="Q-after-filter"
        ).hits
    ] == baseline


def test_exact_transcript_read_has_stable_source_bound_evidence() -> None:
    corpus = PrecedentCorpus.from_episodes((episode("18-rowvigor"),))

    first = corpus.open_transcript(
        "18-rowvigor", start=0, end=1, query_id="Q-first"
    )
    second = corpus.open_transcript(
        "18-rowvigor", start=0, end=1, query_id="Q-second"
    )

    expected_text = "\n".join(turn.text for turn in first.turns)
    expected_id = "H-" + sha256(
        f"{first.source_sha256}\0{0}\0{1}\0{expected_text}".encode("utf-8")
    ).hexdigest()[:20]
    assert first.evidence.evidence_id == second.evidence.evidence_id == expected_id
    assert first.evidence.query_id == "Q-first"
    assert second.evidence.query_id == "Q-second"
    assert first.evidence.text == expected_text
    assert first.evidence.citation_mode == "exact"


def test_decision_read_returns_exact_historical_evidence_or_none() -> None:
    observed = PrecedentCorpus.from_episodes((episode("18-rowvigor"),))
    unobserved = PrecedentCorpus.from_episodes(
        (unobserved_episode("20-harper-wilde", "garden tools"),)
    )

    decision = observed.open_decision("18-rowvigor", query_id="Q-decision")
    missing = unobserved.open_decision("20-harper-wilde", query_id="Q-missing")

    assert decision.decision.status == "Out"
    assert len(decision.evidence) == 1
    assert decision.evidence[0].text == "This is not right for me."
    assert decision.evidence[0].query_id == "Q-decision"
    assert missing.decision.status == "unobserved"
    assert missing.evidence == ()


def test_search_chunk_id_uses_the_exact_source_bound_formula() -> None:
    row = unobserved_episode("18-rowvigor", "quasar analytics")
    hit = PrecedentCorpus.from_episodes((row,)).search("quasar", 1)[0]
    exact_text = "Narrator: quasar analytics"
    expected = "H-" + sha256(
        f"{row.source_sha256}\0{0}\0{0}\0{exact_text}".encode("utf-8")
    ).hexdigest()[:20]

    assert hit.chunk_id == expected


def test_duplicate_source_bound_chunk_identity_fails_closed() -> None:
    first = unobserved_episode("18-rowvigor", "identical transcript")
    payload = first.model_dump(mode="json")
    payload.update(
        {
            "episode_slug": "20-harper-wilde",
            "episode_number": 20,
            "source_path": "source/20-harper-wilde.json",
        }
    )
    second = PrecedentEpisode.model_validate(payload)

    with pytest.raises(ValueError, match="chunk identity collision"):
        PrecedentCorpus.from_episodes((first, second))


def _build_disk_corpus(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for slug, text, status in (
        ("18-rowvigor", "garden tools", "Unobserved"),
        ("20-harper-wilde", "quasar analytics", "Unobserved"),
        ("99-target", "quasar secret", "Unobserved"),
    ):
        (source / f"{slug}.json").write_text(
            json.dumps({"transcript": f"Narrator: {text}"}), encoding="utf-8"
        )
        rows.append(
            {
                "episode_slug": slug,
                "pitch_window_decision": status,
                "decision_context": "unclear",
                "audit_notes": "No observed decision.",
            }
        )
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "corpus"
    build_precedent_corpus(source, ledger, output, ("Charles",))
    return output


def test_save_load_roundtrip_preserves_ranking_ids_and_target_filter(
    tmp_path: Path,
) -> None:
    corpus_root = _build_disk_corpus(tmp_path)
    built = PrecedentCorpus.build(
        corpus_root, embedder=TinyEmbedder(), target_slug="99-target"
    )
    before = built.search("quasar", top_k=1, query_id="Q-before")
    index_path = tmp_path / "precedents-index.json"
    built.save(index_path)

    loaded = PrecedentCorpus.load(
        index_path, corpus_root, embedder=TinyEmbedder()
    )
    after = loaded.search("quasar", top_k=1, query_id="Q-after")

    assert [hit.chunk_id for hit in before.hits] == [
        hit.chunk_id for hit in after.hits
    ]
    assert [row.episode_slug for row in loaded.list_episodes()] == [
        "18-rowvigor",
        "20-harper-wilde",
    ]
    omitted = ({"18-rowvigor", "20-harper-wilde"} - {after.hits[0].episode_slug}).pop()
    assert loaded.open_transcript(omitted, query_id="Q-open").evidence.text
    with pytest.raises(PermissionError, match="target episode"):
        loaded.open_transcript("99-target", query_id="Q-target")


@pytest.mark.parametrize("tamper", ["record", "manifest", "index"])
def test_load_rejects_tampered_corpus_or_index(tmp_path: Path, tamper: str) -> None:
    corpus_root = _build_disk_corpus(tmp_path)
    index_path = tmp_path / "precedents-index.json"
    PrecedentCorpus.build(corpus_root, embedder=TinyEmbedder()).save(index_path)
    if tamper == "record":
        record = corpus_root / "records" / "18-rowvigor.json"
        record.write_text(record.read_text(encoding="utf-8") + " ", encoding="utf-8")
    elif tamper == "manifest":
        manifest = corpus_root / "corpus-manifest.json"
        manifest.write_text(
            manifest.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
    else:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        raw["chunks"][0]["text"] = "tampered"
        index_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="(hash|digest|consistency|tamper)"):
        PrecedentCorpus.load(index_path, corpus_root, embedder=TinyEmbedder())


def test_load_rejects_a_digest_valid_index_missing_a_non_target_episode(
    tmp_path: Path,
) -> None:
    corpus_root = _build_disk_corpus(tmp_path)
    index_path = tmp_path / "precedents-index.json"
    PrecedentCorpus.build(
        corpus_root, embedder=TinyEmbedder(), target_slug="99-target"
    ).save(index_path)
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    raw["episodes"] = [
        row for row in raw["episodes"] if row["episode_slug"] != "18-rowvigor"
    ]
    raw["chunks"] = [
        row for row in raw["chunks"] if row["episode_slug"] != "18-rowvigor"
    ]
    raw["record_sha256s"].pop("18-rowvigor")
    raw.pop("index_sha256")
    canonical = (
        json.dumps(
            raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        + "\n"
    ).encode("utf-8")
    raw["index_sha256"] = sha256(canonical).hexdigest()
    index_path.write_text(
        json.dumps(
            raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="episode slug consistency"):
        PrecedentCorpus.load(index_path, corpus_root, embedder=TinyEmbedder())


@pytest.mark.parametrize("operation", ["transcript", "decision"])
def test_exact_open_rejects_changed_source_snapshot(
    tmp_path: Path, operation: str
) -> None:
    corpus_root = _build_disk_corpus(tmp_path)
    index_path = tmp_path / "precedents-index.json"
    PrecedentCorpus.build(corpus_root).save(index_path)
    loaded = PrecedentCorpus.load(index_path, corpus_root)
    snapshot = corpus_root / "sources" / "18-rowvigor.json"
    snapshot.write_bytes(snapshot.read_bytes() + b" ")

    with pytest.raises(ValueError, match="source snapshot hash"):
        if operation == "transcript":
            loaded.open_transcript("18-rowvigor", query_id="Q-open")
        else:
            loaded.open_decision("18-rowvigor", query_id="Q-open")


def test_load_rejects_changed_source_snapshot(tmp_path: Path) -> None:
    corpus_root = _build_disk_corpus(tmp_path)
    index_path = tmp_path / "precedents-index.json"
    PrecedentCorpus.build(corpus_root).save(index_path)
    snapshot = corpus_root / "sources" / "18-rowvigor.json"
    snapshot.write_bytes(snapshot.read_bytes() + b" ")

    with pytest.raises(ValueError, match="source snapshot hash"):
        PrecedentCorpus.load(index_path, corpus_root)


@pytest.mark.parametrize("unsafe_path", ["../outside.json", "/tmp/outside.json"])
def test_build_rejects_unsafe_corpus_source_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    corpus_root = _build_disk_corpus(tmp_path)
    manifest_path = corpus_root / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["records"][0]
    record_path = corpus_root / entry["record_path"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["source_path"] = unsafe_path
    record_bytes = (
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    record_path.write_bytes(record_bytes)
    entry["source_path"] = unsafe_path
    entry["record_sha256"] = sha256(record_bytes).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="safe corpus-relative"):
        PrecedentCorpus.build(corpus_root)


def test_build_rejects_source_snapshot_symlink_escape(tmp_path: Path) -> None:
    corpus_root = _build_disk_corpus(tmp_path)
    snapshot = corpus_root / "sources" / "18-rowvigor.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(snapshot.read_bytes())
    snapshot.unlink()
    snapshot.symlink_to(outside)

    with pytest.raises(ValueError, match="safe corpus-relative"):
        PrecedentCorpus.build(corpus_root)


def test_build_rejects_unsafe_corpus_record_path(tmp_path: Path) -> None:
    corpus_root = _build_disk_corpus(tmp_path)
    manifest_path = corpus_root / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][0]["record_path"] = "../outside-record.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="safe corpus-relative"):
        PrecedentCorpus.build(corpus_root)


def _episode_with_model_visible_alias(
    field: str, alias: str
) -> PrecedentEpisode:
    payload = episode("18-visible").model_dump(mode="json")
    if field == "episode_slug":
        payload["episode_slug"] = f"18-{alias.casefold()}-company"
    elif field == "speaker":
        payload["turns"][0]["speaker"] = alias
    elif field == "turn_text":
        payload["turns"][0]["text"] = f"The {alias} company is pitching."
    elif field == "decision_evidence":
        payload["turns"][-1]["text"] = f"My decision references {alias}."
        payload["decision"]["evidence"][0]["text"] = payload["turns"][-1][
            "text"
        ]
    elif field == "condition":
        payload["decision"]["conditions"] = [f"Subject to {alias} verification"]
    elif field == "audit_notes":
        payload["decision"]["audit_notes"] = f"Observed after the {alias} pitch."
    elif field == "audit_source":
        payload["decision"]["audit_source"] = f"{alias} review ledger"
    elif field == "check_tier":
        payload["decision"]["check_tier"] = f"{alias} tier"
    elif field == "status":
        payload["decision"]["status"] = alias
    elif field == "context":
        payload["decision"]["context"] = "initial_panel"
    else:  # pragma: no cover - test construction guard
        raise AssertionError(f"unknown model-visible field: {field}")
    return PrecedentEpisode.model_validate(payload)


@pytest.mark.parametrize("alias", ["AI", "X", "Ox"])
@pytest.mark.parametrize(
    "field",
    [
        "episode_slug",
        "speaker",
        "turn_text",
        "decision_evidence",
        "condition",
        "audit_notes",
        "audit_source",
        "check_tier",
    ],
)
def test_for_target_filters_short_alias_from_every_model_visible_text_field(
    field: str, alias: str
) -> None:
    row = _episode_with_model_visible_alias(field, alias)
    corpus = PrecedentCorpus.from_episodes((row,))
    transcript = corpus.open_transcript(row.episode_slug, query_id="Q-transcript")
    decision = corpus.open_decision(row.episode_slug, query_id="Q-decision")
    serialized = (
        transcript.model_dump_json() + "\n" + decision.model_dump_json()
    ).casefold()
    assert alias.casefold() in serialized

    filtered = corpus.for_target("99-target", excluded_aliases=(alias,))

    assert filtered.list_episodes() == ()
    with pytest.raises(KeyError, match="unknown precedent episode"):
        filtered.open_transcript(row.episode_slug)
    with pytest.raises(KeyError, match="unknown precedent episode"):
        filtered.open_decision(row.episode_slug)


@pytest.mark.parametrize(
    ("field", "alias"),
    [("status", "Out"), ("context", "initial panel")],
)
def test_for_target_filters_alias_from_decision_enum_fields(
    field: str, alias: str
) -> None:
    row = _episode_with_model_visible_alias(field, alias)
    decision = PrecedentCorpus.from_episodes((row,)).open_decision(
        row.episode_slug, query_id="Q-decision"
    )
    assert alias.replace(" ", "_") in decision.model_dump_json()

    filtered = PrecedentCorpus.from_episodes((row,)).for_target(
        "99-target", excluded_aliases=(alias,)
    )

    assert filtered.list_episodes() == ()


def test_for_target_short_aliases_use_token_boundaries_not_substrings() -> None:
    rows = (
        unobserved_episode("18-chair", "The chair supports domain analytics."),
        unobserved_episode("20-xavier", "Xavier studies sandbox systems."),
        unobserved_episode("22-oxford", "Oxford provides proxy market data."),
    )

    filtered = PrecedentCorpus.from_episodes(rows).for_target(
        "99-target", excluded_aliases=("AI", "X", "Ox")
    )

    assert [row.episode_slug for row in filtered.list_episodes()] == [
        "18-chair",
        "20-xavier",
        "22-oxford",
    ]
