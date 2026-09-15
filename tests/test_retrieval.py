import json
from pathlib import Path

import pytest

from vc_clone_graph.retrieval import HybridWikiIndex, reciprocal_rank_fusion


class TinyEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float("founder" in text.lower()), float("market" in text.lower())]
            for text in texts
        ]


class FailingEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding backend unavailable")


class RecordingEmbedder(TinyEmbedder):
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


def make_wiki(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "persona.md").write_text(
        "# Persona\n\nFounder speed and customer learning matter.\n\n"
        "## Market\n\nA large reachable market matters.\n",
        encoding="utf-8",
    )
    (wiki / "constraints.md").write_text(
        "# Constraints\n\nSmall initial checks and pre-seed entry.\n",
        encoding="utf-8",
    )
    return wiki


def test_reciprocal_rank_fusion_rewards_results_in_both_lists() -> None:
    scores = reciprocal_rank_fusion([["A", "B"], ["B", "C"]])
    assert scores["B"] > scores["A"]
    assert scores["B"] > scores["C"]


def test_exact_read_returns_hash_bound_source(tmp_path: Path) -> None:
    wiki = make_wiki(tmp_path)
    index = HybridWikiIndex.build(wiki, TinyEmbedder())
    hit = index.search("founder customer learning", limit=2)[0]
    evidence = index.read(hit.chunk_id)
    assert evidence.text in (wiki / evidence.source_path).read_text()
    assert len(evidence.source_sha256) == 64
    assert evidence.chunk_id == hit.chunk_id


def test_fusion_keeps_lexical_results_when_embedding_fails(tmp_path: Path) -> None:
    index = HybridWikiIndex.build(make_wiki(tmp_path), FailingEmbedder())
    hit = index.search("customer learning", limit=1)[0]
    assert hit.source_path == "persona.md"
    assert hit.retrieval_modes == ("lexical",)


def test_complete_wiki_index_fails_closed_on_embedding_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="complete wiki embedding index"):
        HybridWikiIndex.build(
            make_wiki(tmp_path),
            FailingEmbedder(),
            require_complete_embeddings=True,
        )


def test_complete_wiki_index_persists_metadata_and_rejects_model_mismatch(
    tmp_path: Path,
) -> None:
    wiki = make_wiki(tmp_path)
    path = tmp_path / "index.json"
    HybridWikiIndex.build(
        wiki, MetadataEmbedder(), require_complete_embeddings=True
    ).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["embedding_index"]["coverage"] == 1.0
    assert payload["embedding_index"]["dimension"] == 2
    assert payload["embedding_index"]["model"] == "fixture-embed-v1"
    with pytest.raises(ValueError, match="embedding configuration mismatch"):
        HybridWikiIndex.load(
            path,
            wiki,
            MetadataEmbedder("different-model"),
            require_complete_embeddings=True,
        )


def test_index_round_trip_is_deterministic(tmp_path: Path) -> None:
    wiki = make_wiki(tmp_path)
    first = HybridWikiIndex.build(wiki, TinyEmbedder())
    path = tmp_path / "index.json"
    first.save(path)
    second = HybridWikiIndex.load(path, wiki, TinyEmbedder())
    assert [hit.chunk_id for hit in first.search("market", 3)] == [
        hit.chunk_id for hit in second.search("market", 3)
    ]


def test_read_rejects_unknown_chunk(tmp_path: Path) -> None:
    index = HybridWikiIndex.build(make_wiki(tmp_path), TinyEmbedder())
    try:
        index.read("W-does-not-exist")
    except KeyError as exc:
        assert "unknown wiki chunk" in str(exc)
    else:
        raise AssertionError("unknown chunk was accepted")


def test_pitch_view_redacts_target_but_preserves_other_holdings(tmp_path: Path) -> None:
    wiki = make_wiki(tmp_path)
    (wiki / "portfolio.md").write_text(
        "# Portfolio\n\nEdtech holdings: Campus, ClassDojo, Nectir.\n",
        encoding="utf-8",
    )
    embedder = RecordingEmbedder()
    base = HybridWikiIndex.build(wiki, embedder)
    original_vectors = {chunk.chunk_id: chunk.embedding for chunk in base.chunks}
    embedder.calls.clear()

    sanitized, report = base.for_pitch(["Nectir"])

    changed = sanitized.read(report.changed_chunk_ids[0])
    assert "Nectir" not in changed.text
    assert "TARGET COMPANY" not in changed.text
    assert "Campus" in changed.text
    assert report.changed_chunk_count == 1
    assert report.recomputed_embedding_count == 1
    assert report.reused_embedding_count == len(base.chunks) - 1
    assert report.alias_hashes and "Nectir" not in repr(report)
    assert len(embedder.calls) == 1
    unchanged = [c for c in sanitized.chunks if c.chunk_id not in report.changed_chunk_ids]
    assert all(c.embedding == original_vectors[c.chunk_id] for c in unchanged)


def test_pitch_view_alias_matching_is_case_insensitive(tmp_path: Path) -> None:
    wiki = make_wiki(tmp_path)
    (wiki / "portfolio.md").write_text(
        "# Portfolio\n\nNECTIR is an education holding.\n", encoding="utf-8"
    )
    sanitized, report = HybridWikiIndex.build(wiki, TinyEmbedder()).for_pitch(
        ["nectir"]
    )
    assert report.changed_chunk_count == 1
    assert "NECTIR" not in sanitized.read(report.changed_chunk_ids[0]).text


def test_pitch_view_sanitizes_headings_and_unicode_casefold_variants(
    tmp_path: Path,
) -> None:
    wiki = make_wiki(tmp_path)
    (wiki / "target.md").write_text(
        "# STRASSE\n\nSTRASSE builds education software.\n", encoding="utf-8"
    )

    sanitized, report = HybridWikiIndex.build(wiki, TinyEmbedder()).for_pitch(
        ["Straße"]
    )

    evidence = sanitized.read(report.changed_chunk_ids[0])
    assert "strasse" not in evidence.heading.casefold()
    assert "strasse" not in evidence.text.casefold()
    hit = next(
        hit
        for hit in sanitized.search("education", 10)
        if hit.chunk_id == evidence.chunk_id
    )
    assert "strasse" not in hit.heading.casefold()


def test_pitch_view_sanitizes_aliases_from_exposed_source_paths(
    tmp_path: Path,
) -> None:
    wiki = make_wiki(tmp_path)
    target = wiki / "Nectir.md"
    target.write_text(
        "# Portfolio note\n\nAn education company is in the portfolio.\n",
        encoding="utf-8",
    )

    sanitized, _ = HybridWikiIndex.build(wiki, TinyEmbedder()).for_pitch(
        ["Nectir"]
    )
    hit = next(
        hit for hit in sanitized.search("education", 10)
        if "education" in hit.excerpt.lower()
    )
    evidence = sanitized.read(hit.chunk_id)

    assert "nectir" not in hit.source_path.casefold()
    assert "nectir" not in evidence.source_path.casefold()
    assert evidence.text == "# Portfolio note\n\nAn education company is in the portfolio."


def test_pitch_view_rejects_canonical_source_mutation(tmp_path: Path) -> None:
    wiki = make_wiki(tmp_path)
    portfolio = wiki / "portfolio.md"
    portfolio.write_text("# Portfolio\n\nNectir and Campus.\n", encoding="utf-8")
    sanitized, report = HybridWikiIndex.build(wiki, TinyEmbedder()).for_pitch(
        ["Nectir"]
    )
    portfolio.write_text("# Portfolio\n\nChanged.\n", encoding="utf-8")
    try:
        sanitized.read(report.changed_chunk_ids[0])
    except ValueError as exc:
        assert "hash changed" in str(exc)
    else:
        raise AssertionError("mutated canonical source was accepted")


def test_pitch_view_embedding_failure_keeps_changed_chunk_lexical_only(
    tmp_path: Path,
) -> None:
    wiki = make_wiki(tmp_path)
    (wiki / "portfolio.md").write_text(
        "# Portfolio\n\nNectir and Campus.\n", encoding="utf-8"
    )
    base = HybridWikiIndex.build(wiki, TinyEmbedder())
    base.embedder = FailingEmbedder()

    sanitized, report = base.for_pitch(["Nectir"])

    assert report.lexical_only_chunk_ids == report.changed_chunk_ids
    assert report.quality_findings == ("SANITIZED_EMBEDDING_FALLBACK",)
    hit = next(
        hit for hit in sanitized.search("Campus", 10) if hit.chunk_id in report.changed_chunk_ids
    )
    assert hit.retrieval_modes == ("lexical",)
