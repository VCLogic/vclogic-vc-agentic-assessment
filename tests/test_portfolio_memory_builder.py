from __future__ import annotations

import json
from pathlib import Path

from vc_clone_graph.portfolio_memory import PortfolioMemoryCorpus
from vc_clone_graph.portfolio_memory_builder import (
    build_portfolio_memory,
    discover_candidate_turns,
)
from vc_clone_graph.precedents import (
    PrecedentDecision,
    PrecedentEpisode,
    TranscriptTurn,
)


class FakeEmbedder:
    metadata = {"backend": "fake", "model": "fake", "revision": "one"}

    def embed_documents(self, texts):
        return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

    def embed_queries(self, texts):
        return [[1.0, 1.0] for _ in texts]


def write_record(root: Path, slug: str, turns: list[tuple[str, str]]):
    record = PrecedentEpisode(
        episode_slug=slug,
        episode_number=int(slug.split("-", 1)[0]),
        source_path=f"episodes/{slug}.json",
        source_sha256="a" * 64,
        investor_aliases=("Charles",),
        investor_present=True,
        turns=tuple(
            TranscriptTurn(turn_index=index, speaker=speaker, text=text)
            for index, (speaker, text) in enumerate(turns)
        ),
        decision=PrecedentDecision(
            status="unobserved", context="initial_panel", check_tier=None,
            conditions=(), evidence=(), audit_source="audit", audit_notes="notes",
        ),
    )
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    (records / f"{slug}.json").write_text(record.model_dump_json(indent=2) + "\n")


def test_candidate_discovery_only_returns_investor_disclosure_cues():
    turns = (
        TranscriptTurn(turn_index=0, speaker="Founder", text="I have investors."),
        TranscriptTurn(turn_index=1, speaker="Charles", text="We have a portfolio company nearby."),
        TranscriptTurn(turn_index=2, speaker="Charles", text="I like the market."),
    )
    found = discover_candidate_turns(turns, ("Charles",))
    assert [row.turn_index for row in found] == [1]


def test_builder_rebinds_notes_rejects_self_entry_and_supplements_conflict(tmp_path: Path):
    precedents = tmp_path / "precedents"
    write_record(
        precedents,
        "10-calendar",
        [
            ("Founder", "We coordinate group calendars."),
            ("Charles", "I am an investor in a company called LetsMeet that does group scheduling."),
            ("Charles", "I have to be out because this is too close."),
        ],
    )
    write_record(
        precedents,
        "12-other",
        [("Charles", "Cyan wired the money and signed the SAFE.")],
    )
    notes = tmp_path / "notes.json"
    notes.write_text(
        json.dumps(
            {
                "vc_slug": "charles",
                "holdings": [
                    {
                        "company": "LetsMeet",
                        "descriptor": "group scheduling",
                        "evidence": "I am an investor in a company called LetsMeet that does group scheduling.",
                        "source_episode": "10-calendar",
                    },
                    {
                        "company": "Charles",
                        "descriptor": "not a company",
                        "evidence": "Cyan wired the money and signed the SAFE.",
                        "source_episode": "12-other",
                    },
                ],
            }
        )
    )
    references = tmp_path / "references"
    references.mkdir()
    (references / "10-calendar.json").write_text(
        json.dumps(
            {
                "episode_slug": "10-calendar",
                "vc_slug": "charles",
                "rationales": [
                    {
                        "rationale_label": "portfolio_conflict_constraint",
                        "evidence": ["I have to be out because this is too close."],
                        "decision_link": "explicit",
                    }
                ],
            }
        )
    )
    output = tmp_path / "portfolio-memory"
    report = build_portfolio_memory(
        vc_slug="charles",
        investor_aliases=("Charles",),
        precedent_root=precedents,
        notes_path=notes,
        reference_records_root=references,
        output_root=output,
        embedder=FakeEmbedder(),
    )
    corpus = PortfolioMemoryCorpus.load_events("charles", output / "disclosure-events.jsonl")
    assert report["accepted_count"] == 2
    assert {row.company_name for row in corpus.disclosures} == {"LetsMeet", None}
    assert any(row.observed_consequence == "out" for row in corpus.disclosures)
    audit = [json.loads(line) for line in (output / "extraction-audit.jsonl").read_text().splitlines()]
    assert any(row["status"] == "rejected" and "self" in row["reason"] for row in audit)
    assert (output / "corpus-manifest.json").is_file()
    assert (output / "embedding-index.json").is_file()
    assert (output / "company-index.json").is_file()


def test_builder_recovers_paraphrase_from_exact_company_turn(tmp_path: Path):
    precedents = tmp_path / "precedents"
    write_record(
        precedents,
        "10-clinic",
        [("Charles", "We invested in a business that is somewhat similar, called Ease.")],
    )
    notes = tmp_path / "notes.json"
    notes.write_text(json.dumps({
        "vc_slug": "charles",
        "holdings": [{
            "company": "Ease",
            "descriptor": "helps doctors launch clinics",
            "evidence": "We invested in a business called Ease.",
            "source_episode": "10-clinic",
        }],
    }))
    output = tmp_path / "portfolio-memory"
    build_portfolio_memory(
        vc_slug="charles", investor_aliases=("Charles",), precedent_root=precedents,
        notes_path=notes, reference_records_root=None, output_root=output,
        embedder=FakeEmbedder(),
    )
    corpus = PortfolioMemoryCorpus.load_events("charles", output / "disclosure-events.jsonl")
    assert corpus.disclosures[0].evidence[0].text == (
        "We invested in a business that is somewhat similar, called Ease."
    )


def test_builder_selects_investment_turn_when_company_is_mentioned_again(tmp_path: Path):
    precedents = tmp_path / "precedents"
    write_record(
        precedents,
        "10-restaurant",
        [
            ("Jesse", "I backed a company called Squire when they started."),
            ("Jesse", "I could ask the founders at Squire about international sales."),
        ],
    )
    notes = tmp_path / "notes.json"
    notes.write_text(json.dumps({
        "vc_slug": "jesse",
        "holdings": [{
            "company": "Squire", "descriptor": "barbershop software",
            "evidence": "I've backed Squire.", "source_episode": "10-restaurant",
        }],
    }))
    output = tmp_path / "portfolio-memory"
    build_portfolio_memory(
        vc_slug="jesse", investor_aliases=("Jesse",), precedent_root=precedents,
        notes_path=notes, reference_records_root=None, output_root=output,
        embedder=FakeEmbedder(),
    )
    corpus = PortfolioMemoryCorpus.load_events("jesse", output / "disclosure-events.jsonl")
    assert corpus.disclosures[0].evidence[0].turn_index == 0
    assert corpus.disclosures[0].relationship == "investment"


def test_builder_recovers_split_relationship_and_company_disclosure(tmp_path: Path):
    precedents = tmp_path / "precedents"
    write_record(
        precedents,
        "10-delivery",
        [
            ("Elizabeth", "I backed Anthony in his new company."),
            ("Founder", "Vinovest?"),
            ("Elizabeth", "Vinovest. Yes."),
        ],
    )
    notes = tmp_path / "notes.json"
    notes.write_text(json.dumps({
        "vc_slug": "elizabeth",
        "holdings": [{
            "company": "Vinovest",
            "descriptor": "Anthony's new company",
            "evidence": "I have backed Anthony in his new company. Vinovest.",
            "source_episode": "10-delivery",
        }],
    }))
    output = tmp_path / "portfolio-memory"
    build_portfolio_memory(
        vc_slug="elizabeth", investor_aliases=("Elizabeth",), precedent_root=precedents,
        notes_path=notes, reference_records_root=None, output_root=output,
        embedder=FakeEmbedder(),
    )
    corpus = PortfolioMemoryCorpus.load_events("elizabeth", output / "disclosure-events.jsonl")
    assert [item.turn_index for item in corpus.disclosures[0].evidence] == [0, 2]


def test_builder_does_not_accept_company_name_without_nearby_relationship(tmp_path: Path):
    precedents = tmp_path / "precedents"
    write_record(
        precedents,
        "10-alcohol",
        [
            ("Charles", "I like the market."),
            ("Founder", "We were like Drizly."),
            ("Charles", "Drizly."),
        ],
    )
    notes = tmp_path / "notes.json"
    notes.write_text(json.dumps({
        "vc_slug": "charles",
        "holdings": [{
            "company": "Drizly", "descriptor": "alcohol delivery",
            "evidence": "we backed Drizly", "source_episode": "10-alcohol",
        }],
    }))
    output = tmp_path / "portfolio-memory"
    report = build_portfolio_memory(
        vc_slug="charles", investor_aliases=("Charles",), precedent_root=precedents,
        notes_path=notes, reference_records_root=None, output_root=output,
        embedder=FakeEmbedder(),
    )
    assert report["accepted_count"] == 0
