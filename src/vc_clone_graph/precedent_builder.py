"""Build deterministic, source-bound precedent records from audited transcripts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .precedents import (
    DecisionEvidence,
    PrecedentDecision,
    PrecedentEpisode,
    TranscriptTurn,
)


_PERSON_NAME = r"[^\W\d_](?:(?:[^\W\d_])|[ .'-]){0,80}"
_SPEAKER = re.compile(rf"^({_PERSON_NAME}):( ?)(.*)$")
_SPLIT_SPEAKER = re.compile(rf"^{_PERSON_NAME}$")
_SPLIT_COLON = re.compile(r"^:\s?(.*)$")
_ALLOWED_CONTEXTS = {
    "initial_panel",
    "same_session_reversal",
    "later_diligence",
    "off_panel",
    "unclear",
}
_ALLOWED_STATUSES = ("In", "Out", "Unobserved")
_CHUNK_TURNS = 8
_MISSING = object()


def parse_speaker_turns(transcript: str) -> tuple[TranscriptTurn, ...]:
    """Parse speaker-prefixed lines while retaining source wording and newlines."""
    pending: list[tuple[str, list[str]]] = []
    lines = transcript.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        match = _SPEAKER.match(line)
        if match:
            initial = match.group(3)
            pending.append((match.group(1), [initial] if initial else []))
        elif (
            _SPLIT_SPEAKER.fullmatch(line)
            and index + 1 < len(lines)
            and (colon := _SPLIT_COLON.match(lines[index + 1]))
        ):
            initial = colon.group(1)
            pending.append((line, [initial] if initial else []))
            index += 1
        elif pending:
            pending[-1][1].append(line)
        elif line:
            pending.append(("Narrator", [line]))
        index += 1

    turns: list[TranscriptTurn] = []
    for speaker, lines in pending:
        text = "\n".join(lines)
        if not text:
            continue
        turns.append(
            TranscriptTurn(
                turn_index=len(turns),
                speaker=speaker,
                text=text,
            )
        )
    return tuple(turns)


def _locator_text(value: str) -> str:
    return " ".join(value.replace("’", "'").replace("‘", "'").split())


def _matches_investor_alias(speaker: str, investor_aliases: Sequence[str]) -> bool:
    normalized = _locator_text(speaker).casefold()
    return any(
        normalized == _locator_text(alias).casefold() for alias in investor_aliases
    )


def _match_exact_quote(
    turns: Sequence[TranscriptTurn],
    quote: str,
    investor_aliases: Sequence[str],
    evidence_turn_index: object = _MISSING,
) -> tuple[DecisionEvidence, ...]:
    needle = _locator_text(quote)
    if evidence_turn_index is not _MISSING:
        if type(evidence_turn_index) is not int:
            raise ValueError("evidence_turn_index must be an integer")
        if evidence_turn_index < 0 or evidence_turn_index >= len(turns):
            raise ValueError("evidence_turn_index is outside transcript bounds")
        selected = turns[evidence_turn_index]
        if selected.turn_index != evidence_turn_index:
            raise ValueError("evidence_turn_index does not bind a source turn")
        if not _matches_investor_alias(selected.speaker, investor_aliases):
            raise ValueError("evidence_turn_index does not identify the investor")
        exact_candidates = (
            _locator_text(selected.text),
            _locator_text(f"{selected.speaker}: {selected.text}"),
        )
        if not needle or not any(needle in candidate for candidate in exact_candidates):
            raise ValueError("evidence_turn_index quote does not match exact turn")
        return (
            DecisionEvidence(
                turn_start=selected.turn_index,
                turn_end=selected.turn_index,
                text=selected.text,
            ),
        )

    if not needle:
        return ()
    matches = tuple(
        turn
        for turn in turns
        if _matches_investor_alias(turn.speaker, investor_aliases)
        and (
            needle in _locator_text(turn.text)
            or needle in _locator_text(f"{turn.speaker}: {turn.text}")
        )
    )
    if len(matches) > 1:
        raise ValueError("ambiguous investor evidence quote")
    if not matches:
        return ()
    selected = matches[0]
    return (
        DecisionEvidence(
            turn_start=selected.turn_index,
            turn_end=selected.turn_index,
            text=selected.text,
        ),
    )


def _episode_record(
    source_path: str,
    source_hash: str,
    turns: tuple[TranscriptTurn, ...],
    audit: dict[str, Any],
    evidence: tuple[DecisionEvidence, ...],
    investor_aliases: tuple[str, ...],
    audit_source: str,
) -> PrecedentEpisode:
    raw_status = audit.get("pitch_window_decision")
    status = raw_status if raw_status in {"In", "Out"} else "unobserved"
    context = audit["decision_context"]
    aliases_casefolded = {alias.casefold() for alias in investor_aliases}
    condition = audit.get("condition")
    conditions = (condition,) if isinstance(condition, str) and condition else ()
    slug = audit["episode_slug"]
    prefix = slug.split("-", 1)[0]
    return PrecedentEpisode(
        episode_slug=slug,
        episode_number=int(prefix) if prefix.isdecimal() else None,
        source_path=source_path,
        source_sha256=source_hash,
        investor_aliases=investor_aliases,
        investor_present=any(
            turn.speaker.casefold() in aliases_casefolded for turn in turns
        ),
        turns=turns,
        decision=PrecedentDecision(
            status=status,
            context=context,
            check_tier=audit.get("check_tier"),
            conditions=conditions,
            evidence=evidence,
            audit_source=audit_source,
            audit_notes=audit["audit_notes"],
        ),
    )


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _safe_output_file(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("output path must remain within corpus output")
    parent = root / relative.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        parent.resolve(strict=True).relative_to(root)
        candidate = parent / relative.name
        candidate.resolve(strict=False).relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError("output path must remain within corpus output") from exc
    if candidate.exists() and not candidate.is_file():
        raise ValueError("output path must remain within corpus output")
    return candidate


def _chunk_rows(records: Sequence[PrecedentEpisode]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        for start in range(0, len(record.turns), _CHUNK_TURNS):
            turns = record.turns[start : start + _CHUNK_TURNS]
            identity = sha256(
                f"{record.episode_slug}:{start}".encode("utf-8")
            ).hexdigest()[:20]
            rows.append(
                {
                    "chunk_id": f"H-{identity}",
                    "episode_slug": record.episode_slug,
                    "source_sha256": record.source_sha256,
                    "text": "\n".join(
                        f"{turn.speaker}: {turn.text}" for turn in turns
                    ),
                    "turn_end": start + len(turns) - 1,
                    "turn_start": start,
                }
            )
    return rows


def build_precedent_corpus(
    transcript_root: str | Path,
    ledger_path: str | Path,
    output_root: str | Path,
    investor_aliases: Sequence[str],
) -> tuple[PrecedentEpisode, ...]:
    """Build records for every audited ledger row and publish the manifest last."""
    transcripts = Path(transcript_root)
    ledger_file = Path(ledger_path)
    output = Path(output_root)
    aliases = tuple(investor_aliases)
    manifest_path = output / "corpus-manifest.json"
    manifest_path.unlink(missing_ok=True)
    ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
    if not isinstance(ledger, list):
        raise ValueError("decision audit must be a JSON list")
    labels = {row["episode_slug"]: row for row in ledger}
    if len(labels) != len(ledger):
        raise ValueError("decision audit contains duplicate episode slugs")
    for slug, audit in labels.items():
        status = audit.get("pitch_window_decision")
        if status not in _ALLOWED_STATUSES:
            raise ValueError(
                f"invalid pitch_window_decision for {slug}: {status!r}"
            )
        context = audit.get("decision_context")
        if context not in _ALLOWED_CONTEXTS:
            raise ValueError(f"invalid decision_context for {slug}: {context!r}")
    records: list[PrecedentEpisode] = []
    record_payloads: list[tuple[str, bytes]] = []
    source_payloads: list[tuple[str, bytes, str]] = []
    for slug, audit in sorted(labels.items()):
        source = transcripts / f"{slug}.json"
        if not source.is_file():
            raise ValueError(f"source transcript record missing: {slug}")
        source_bytes = source.read_bytes()
        source_hash = sha256(source_bytes).hexdigest()
        raw = json.loads(source_bytes)
        transcript = raw.get("transcript")
        turns = parse_speaker_turns(transcript) if transcript else ()
        observed = audit.get("pitch_window_decision") in {"In", "Out"}
        evidence = (
            _match_exact_quote(
                turns,
                audit.get("evidence_quote", ""),
                aliases,
                (
                    audit["evidence_turn_index"]
                    if "evidence_turn_index" in audit
                    else _MISSING
                ),
            )
            if observed
            else ()
        )
        if observed and not evidence:
            raise ValueError(f"evidence quote not found in transcript: {slug}")
        source_relative = f"sources/{slug}.json"
        record = _episode_record(
            source_path=source_relative,
            source_hash=source_hash,
            turns=turns,
            audit=audit,
            evidence=evidence,
            investor_aliases=aliases,
            audit_source=ledger_file.as_posix(),
        )
        records.append(record)
        source_payloads.append((source_relative, source_bytes, source_hash))
        record_payloads.append(
            (
                f"records/{slug}.json",
                _canonical_json_bytes(record.model_dump(mode="json")),
            )
        )

    chunks = _chunk_rows(records)
    chunks_bytes = b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for row in chunks
    )
    manifest_rows = []
    for record, (relative_path, payload) in zip(records, record_payloads, strict=True):
        manifest_rows.append(
            {
                "decision_status": record.decision.status,
                "episode_slug": record.episode_slug,
                "record_path": relative_path,
                "record_sha256": sha256(payload).hexdigest(),
                "source_path": record.source_path,
                "source_sha256": record.source_sha256,
            }
        )
    manifest_bytes = _canonical_json_bytes(
        {"records": manifest_rows, "schema": "precedent-corpus-v1"}
    )

    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    for relative_path, payload, source_hash in source_payloads:
        target = _safe_output_file(output, relative_path)
        target.write_bytes(payload)
        if sha256(target.read_bytes()).hexdigest() != source_hash:
            raise ValueError(f"source snapshot hash mismatch: {relative_path}")
    for relative_path, payload in record_payloads:
        target = _safe_output_file(output, relative_path)
        target.write_bytes(payload)
    _safe_output_file(output, "chunks.jsonl").write_bytes(chunks_bytes)
    _safe_output_file(output, "corpus-manifest.json").write_bytes(manifest_bytes)
    return tuple(records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build source-bound historical precedent records."
    )
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--investor-alias", action="append", required=True, dest="investor_aliases"
    )
    args = parser.parse_args(argv)
    records = build_precedent_corpus(
        args.transcripts,
        args.ledger,
        args.output,
        args.investor_aliases,
    )
    print(f"built {len(records)} precedent records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
