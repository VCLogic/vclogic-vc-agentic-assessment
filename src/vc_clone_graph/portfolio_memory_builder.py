"""Build source-bound portfolio memories from episode-derived disclosures."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from .portfolio_memory import (
    DisclosureEvidence,
    PortfolioDisclosure,
    PortfolioMemoryCorpus,
    PortfolioMemoryIndex,
    make_disclosure_id,
)
from .precedents import PrecedentEpisode, TranscriptTurn


_CUE = re.compile(
    r"\b(portfolio|invest(?:ed|or|ment)|backed|board|eir|conflict|too close|"
    r"competitive|complementary|disclos|check with|ask (?:the|my|our) founder)\b",
    re.IGNORECASE,
)


def _normalize(value: str) -> str:
    return " ".join(value.replace("’", "'").replace("‘", "'").casefold().split())


def _speaker_matches(speaker: str, aliases: Sequence[str]) -> bool:
    normalized = _normalize(speaker)
    return any(normalized == _normalize(alias) for alias in aliases)


def discover_candidate_turns(
    turns: Sequence[TranscriptTurn], investor_aliases: Sequence[str]
) -> tuple[TranscriptTurn, ...]:
    return tuple(
        turn for turn in turns
        if _speaker_matches(turn.speaker, investor_aliases) and _CUE.search(turn.text)
    )


def _load_records(root: Path) -> dict[str, PrecedentEpisode]:
    result: dict[str, PrecedentEpisode] = {}
    for path in sorted((root / "records").glob("*.json")):
        record = PrecedentEpisode.model_validate_json(path.read_text(encoding="utf-8"))
        if record.episode_slug in result:
            raise ValueError(f"duplicate precedent episode: {record.episode_slug}")
        result[record.episode_slug] = record
    if not result:
        raise ValueError(f"precedent corpus has no records: {root}")
    return result


def _bind_quote(
    record: PrecedentEpisode, quote: str, aliases: Sequence[str]
) -> tuple[DisclosureEvidence, ...]:
    needle = _normalize(quote)
    if not needle:
        raise ValueError("portfolio evidence quote is empty")
    matches = [
        turn for turn in record.turns
        if _speaker_matches(turn.speaker, aliases) and needle in _normalize(turn.text)
    ]
    if len(matches) != 1:
        raise ValueError(
            "portfolio evidence quote does not bind uniquely to a target-investor turn"
        )
    turn = matches[0]
    return (
        DisclosureEvidence(
            source_sha256=record.source_sha256,
            turn_index=turn.turn_index,
            speaker=turn.speaker,
            text=quote.strip(),
        ),
    )


def _turn_evidence(record: PrecedentEpisode, turn: TranscriptTurn) -> DisclosureEvidence:
    return DisclosureEvidence(
        source_sha256=record.source_sha256,
        turn_index=turn.turn_index,
        speaker=turn.speaker,
        text=turn.text.strip(),
    )


def _bind_note_evidence(
    record: PrecedentEpisode,
    quote: str,
    company: str | None,
    aliases: Sequence[str],
) -> tuple[DisclosureEvidence, ...]:
    try:
        return _bind_quote(record, quote, aliases)
    except ValueError:
        if company is None:
            raise
    company_needle = _normalize(company)
    company_turns = [
        turn
        for turn in record.turns
        if _speaker_matches(turn.speaker, aliases)
        and company_needle in _normalize(turn.text)
    ]
    relationship_turns = [
        turn for turn in company_turns if _relationship(turn.text) != "uncertain"
    ]
    if len(relationship_turns) == 1:
        return (_turn_evidence(record, relationship_turns[0]),)
    if len(company_turns) != 1:
        raise ValueError(
            "portfolio evidence paraphrase does not bind to one exact company turn"
        )
    company_turn = company_turns[0]
    preceding = [
        turn
        for turn in record.turns
        if _speaker_matches(turn.speaker, aliases)
        and 0 < company_turn.turn_index - turn.turn_index <= 2
        and _relationship(turn.text) != "uncertain"
    ]
    if len(preceding) != 1:
        raise ValueError(
            "company mention has no uniquely bound nearby investment relationship"
        )
    return (
        _turn_evidence(record, preceding[0]),
        _turn_evidence(record, company_turn),
    )


def _relationship(text: str) -> str:
    value = _normalize(text)
    if "board" in value:
        return "board_role"
    if re.search(r"\beir\b", value):
        return "eir"
    if "acquisition" in value or "acquired" in value:
        return "indirect_holding"
    if "went out of business" in value or "former" in value:
        return "former_investment"
    if "invest" in value or "backed" in value or "portfolio" in value:
        return "investment"
    return "uncertain"


def _overlap(text: str) -> str:
    value = _normalize(text)
    if "complement" in value:
        return "complementary"
    if "too close" in value or "direct conflict" in value:
        return "direct"
    if "conflict" in value or "competitive" in value or "check with" in value:
        return "possible"
    return "disclosure_only"


def _consequence(text: str) -> str:
    value = _normalize(text)
    if re.search(r"\b(?:i am|i'm|i have to be|we are|we're) out\b", value):
        return "out"
    if "as long as" in value or "provided that" in value or "conditional" in value:
        return "conditional_in"
    if "check with" in value or "ask" in value or "permission" in value or "double check" in value:
        return "permission_or_check_required"
    if "complement" in value:
        return "no_effect"
    return "unclear"


def _episode_number(slug: str) -> int | None:
    prefix = slug.split("-", 1)[0]
    return int(prefix) if prefix.isdecimal() else None


def _safe_company(value: object, aliases: Sequence[str]) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    company = value.strip()
    normalized = _normalize(company)
    if any(normalized == _normalize(alias) for alias in aliases):
        raise ValueError("self entry is not a portfolio company")
    generic = {"company", "portfolio company", "hardware companies", "companies"}
    return None if normalized in generic else company


def _note_disclosure(
    vc_slug: str,
    aliases: Sequence[str],
    note: dict[str, Any],
    records: dict[str, PrecedentEpisode],
) -> PortfolioDisclosure:
    slug = note.get("source_episode")
    if not isinstance(slug, str) or slug not in records:
        raise ValueError("note source episode is unavailable")
    quote = note.get("evidence")
    if not isinstance(quote, str):
        raise ValueError("note evidence is missing")
    company = _safe_company(note.get("company"), aliases)
    evidence = _bind_note_evidence(records[slug], quote, company, aliases)
    descriptor = note.get("descriptor")
    if not isinstance(descriptor, str) or not descriptor.strip():
        raise ValueError("note descriptor is missing")
    identity = company or descriptor
    source_text = " ".join(item.text for item in evidence)
    return PortfolioDisclosure(
        disclosure_id=make_disclosure_id(vc_slug, slug, evidence[0].turn_index, identity),
        vc_slug=vc_slug,
        company_name=company,
        aliases=(company,) if company else (),
        descriptor=descriptor.strip(),
        relationship=_relationship(source_text),
        observed_overlap=_overlap(source_text),
        observed_consequence=_consequence(source_text),
        source_episode_slug=slug,
        source_episode_number=_episode_number(slug),
        evidence=evidence,
        confidence=0.95,
        validation_status="automated_candidate",
    )


def _reference_disclosures(
    vc_slug: str,
    reference_vc_slug: str,
    aliases: Sequence[str],
    records: dict[str, PrecedentEpisode],
    reference_root: Path | None,
) -> Iterable[PortfolioDisclosure]:
    if reference_root is None or not reference_root.is_dir():
        return ()
    result: list[PortfolioDisclosure] = []
    for path in sorted(reference_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        slug = payload.get("episode_slug")
        if slug not in records or payload.get("vc_slug") != reference_vc_slug:
            continue
        for rationale in payload.get("rationales", []):
            if rationale.get("rationale_label") != "portfolio_conflict_constraint":
                continue
            for quote in rationale.get("evidence", []):
                if not isinstance(quote, str) or not quote.strip():
                    continue
                evidence = _bind_quote(records[slug], quote, aliases)
                descriptor = quote.strip()
                result.append(
                    PortfolioDisclosure(
                        disclosure_id=make_disclosure_id(
                            vc_slug, slug, evidence[0].turn_index, descriptor
                        ),
                        vc_slug=vc_slug,
                        company_name=None,
                        aliases=(),
                        descriptor=descriptor,
                        relationship=_relationship(quote),
                        observed_overlap=_overlap(quote),
                        observed_consequence=_consequence(quote),
                        source_episode_slug=slug,
                        source_episode_number=_episode_number(slug),
                        evidence=evidence,
                        confidence=0.9,
                        validation_status="automated_candidate",
                    )
                )
    return result


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_portfolio_memory(
    *,
    vc_slug: str,
    investor_aliases: Sequence[str],
    precedent_root: Path,
    notes_path: Path,
    reference_records_root: Path | None,
    output_root: Path,
    embedder: Any,
    reference_vc_slug: str | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "corpus-manifest.json"
    manifest_path.unlink(missing_ok=True)
    records = _load_records(precedent_root)
    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    if notes.get("vc_slug") != vc_slug or not isinstance(notes.get("holdings"), list):
        raise ValueError("portfolio notes identity or holdings are invalid")
    accepted: list[PortfolioDisclosure] = []
    audit: list[dict[str, Any]] = []
    for index, note in enumerate(notes["holdings"]):
        try:
            row = _note_disclosure(vc_slug, investor_aliases, note, records)
            accepted.append(row)
            audit.append({"status": "accepted", "source": "portfolio_notes", "index": index, "disclosure_id": row.disclosure_id})
        except Exception as exc:
            audit.append({"status": "rejected", "source": "portfolio_notes", "index": index, "reason": str(exc)})
    existing_spans = {
        (row.source_episode_slug, item.turn_index, _normalize(item.text))
        for row in accepted for item in row.evidence
    }
    for row in _reference_disclosures(
        vc_slug,
        reference_vc_slug or vc_slug,
        investor_aliases,
        records,
        reference_records_root,
    ):
        span = (row.source_episode_slug, row.evidence[0].turn_index, _normalize(row.evidence[0].text))
        if span in existing_spans:
            continue
        accepted.append(row)
        existing_spans.add(span)
        audit.append({"status": "accepted", "source": "conflict_reference", "disclosure_id": row.disclosure_id})
    # Preserve cue coverage as auditable candidates without trusting them as facts.
    accepted_spans = {(row.source_episode_slug, item.turn_index) for row in accepted for item in row.evidence}
    for record in records.values():
        for turn in discover_candidate_turns(record.turns, investor_aliases):
            if (record.episode_slug, turn.turn_index) not in accepted_spans:
                audit.append({
                    "status": "candidate_unextracted", "source": "cue_scan",
                    "episode_slug": record.episode_slug, "turn_index": turn.turn_index,
                    "speaker": turn.speaker, "text": turn.text,
                })
    corpus = PortfolioMemoryCorpus(vc_slug, accepted)
    events_path = output_root / "disclosure-events.jsonl"
    corpus.save_events(events_path)
    index = PortfolioMemoryIndex.build(corpus, embedder)
    index_path = output_root / "embedding-index.json"
    index.save(index_path, events_path)
    all_entities = index.for_target("999999-all").entities
    company_path = output_root / "company-index.json"
    company_path.write_text(
        json.dumps([row.model_dump(mode="json") for row in all_entities], indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    audit_path = output_root / "extraction-audit.jsonl"
    _write_jsonl(audit_path, audit)
    files = [events_path, audit_path, company_path, index_path]
    report = {
        "schema": "portfolio-memory-corpus-manifest-v1",
        "vc_slug": vc_slug,
        "accepted_count": len(corpus.disclosures),
        "named_count": sum(row.company_name is not None for row in corpus.disclosures),
        "anonymous_count": sum(row.company_name is None for row in corpus.disclosures),
        "rejected_count": sum(row["status"] == "rejected" for row in audit),
        "candidate_unextracted_count": sum(row["status"] == "candidate_unextracted" for row in audit),
        "files": {
            path.name: sha256(path.read_bytes()).hexdigest() for path in files
        },
        "embedding": index.metadata,
    }
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
