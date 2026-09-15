"""Build leakage-safe pitch variants with neutral non-target panel questions."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any, Mapping, Sequence


def _norm(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _aliases(name: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", name.casefold())
    return {" ".join(tokens), *(tokens[:1]), *(tokens[-1:])} - {""}


def _speaker_matches(speaker: object, aliases: set[str]) -> bool:
    normalized = _norm(speaker)
    return normalized in aliases or bool(set(normalized.split()) & aliases)


def _is_question(text: str) -> bool:
    normalized = _norm(text).lstrip(". ,:-")
    if "?" in text:
        return True
    return bool(re.match(
        r"^(how|what|why|where|when|who|which|is|are|am|do|does|did|can|could|"
        r"would|will|have|has|tell me|walk me|help me understand)\b",
        normalized,
    ))


def _has_verdict_or_offer(text: str) -> bool:
    normalized = _norm(text)
    patterns = (
        r"\bi(?:'m| am|’m)? out\b", r"\bgoing to pass\b", r"\bi(?:'ll| will) pass\b",
        r"\bwhy i(?:'m| am) passing\b", r"\binvestment offer\b",
        r"\bpart of (?:the|your) round\b", r"\bwrite (?:a|the) check\b",
        r"\bi(?:'d| would) invest\b", r"\bi(?:'m| am) in\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


@dataclass(frozen=True)
class QuestionInsertion:
    source_turn_index: int
    answer_turn_index: int
    canonical_line_index: int
    question: str
    original_speaker: str


@dataclass(frozen=True)
class OmittedCandidate:
    source_turn_index: int
    original_speaker: str
    text: str
    reason: str


@dataclass(frozen=True)
class PanelQuestionArtifact:
    pitch_text: str
    insertions: tuple[QuestionInsertion, ...]
    omitted: tuple[OmittedCandidate, ...]
    boundary_turn_index: int


def _boundary_index(
    turns: Sequence[Mapping[str, Any]], audit: Mapping[str, Any]
) -> int:
    endpoint = audit.get("decision_window_endpoint")
    if isinstance(endpoint, int) and endpoint >= 0:
        return endpoint
    cleanup = audit.get("boundary_cleanup")
    event = cleanup.get("boundary_event") if isinstance(cleanup, Mapping) else None
    evidence = event.get("evidence") if isinstance(event, Mapping) else None
    if isinstance(evidence, str) and _norm(evidence):
        needle = _norm(evidence)
        matches = [
            int(turn.get("turn_index", turn.get("idx", -1)))
            for turn in turns
            if needle in _norm(turn.get("text")) or _norm(turn.get("text")) in needle
        ]
        if matches:
            return min(matches)
    segments = audit.get("segments")
    if isinstance(segments, Sequence):
        included = [
            int(row["source_turn_index"])
            for row in segments if isinstance(row, Mapping)
            and row.get("action") in {"retained", "neutralized"}
            and isinstance(row.get("source_turn_index"), int)
        ]
        if included:
            return max(included) + 1
    verdicts = [
        int(turn.get("turn_index", turn.get("idx", -1)))
        for turn in turns if _has_verdict_or_offer(str(turn.get("text", "")))
    ]
    if verdicts:
        return min(verdicts)
    indices = [int(turn.get("turn_index", turn.get("idx", -1))) for turn in turns]
    if indices:
        # The canonical pitch is already boundary-audited. Question insertion
        # still requires a mapped founder answer present in that canonical text,
        # so using the source terminus cannot introduce post-boundary founder data.
        return max(indices) + 1
    raise ValueError("audited decision boundary cannot be resolved in empty source turns")


def _canonical_lines(pitch: str) -> list[tuple[str, str, str]]:
    result = []
    for raw in pitch.splitlines():
        if not raw.strip():
            continue
        speaker, separator, text = raw.partition(":")
        if not separator or not speaker.strip() or not text.strip():
            if not result:
                raise ValueError("canonical pitch begins with an unstructured non-empty line")
            prior_raw, prior_speaker, prior_text = result[-1]
            result[-1] = (
                f"{prior_raw}\n{raw}", prior_speaker, f"{prior_text} {raw.strip()}"
            )
            continue
        result.append((raw, speaker.strip(), text.strip()))
    return result


def _founder_line_map(
    turns: Sequence[Mapping[str, Any]], lines: Sequence[tuple[str, str, str]],
    founder_aliases: set[str], boundary: int,
) -> dict[int, int]:
    candidates = [
        (int(turn.get("turn_index", turn.get("idx", -1))), _norm(turn.get("text")))
        for turn in turns
        if int(turn.get("turn_index", turn.get("idx", -1))) < boundary
        and _speaker_matches(turn.get("speaker"), founder_aliases)
    ]
    mapping: dict[int, int] = {}
    after = -1
    for line_index, (_, speaker, text) in enumerate(lines):
        if not _speaker_matches(speaker, founder_aliases):
            continue
        normalized = _norm(text)
        scored = [
            (SequenceMatcher(None, normalized, source).ratio(), turn_index)
            for turn_index, source in candidates if turn_index > after
        ]
        if not scored:
            continue
        score, turn_index = max(scored, key=lambda row: (row[0], -row[1]))
        if score >= 0.72 or normalized in dict(candidates).get(turn_index, ""):
            mapping[turn_index] = line_index
            after = turn_index
    return mapping


def build_panel_question_pitch(
    *, canonical_pitch: str, source_record: Mapping[str, object],
    episode: Mapping[str, object], audit: Mapping[str, object],
    target_vc_slug: str,
) -> PanelQuestionArtifact:
    turns_raw = source_record.get("turns")
    if not isinstance(turns_raw, Sequence):
        raise ValueError("source precedent record has no structured turns")
    turns = [turn for turn in turns_raw if isinstance(turn, Mapping)]
    founders = episode.get("founders")
    panel = episode.get("panel")
    if not isinstance(founders, Sequence) or not isinstance(panel, Sequence):
        raise ValueError("episode metadata lacks founders or panel")
    founder_aliases = set().union(*(
        _aliases(str(row.get("name", ""))) for row in founders if isinstance(row, Mapping)
    ))
    target_rows = [
        row for row in panel if isinstance(row, Mapping) and row.get("slug") == target_vc_slug
    ]
    if len(target_rows) != 1:
        raise ValueError("target VC is not uniquely present in episode panel")
    target_aliases = _aliases(str(target_rows[0].get("name", "")))
    target_aliases.update(_norm(value) for value in source_record.get("investor_aliases", [])
                          if isinstance(value, str))
    other_aliases: dict[str, set[str]] = {
        str(row.get("name")): _aliases(str(row.get("name", "")))
        for row in panel if isinstance(row, Mapping) and row.get("slug") != target_vc_slug
    }
    boundary = _boundary_index(turns, audit)
    lines = _canonical_lines(canonical_pitch)
    canonical_normalized = _norm(canonical_pitch)
    founder_map = _founder_line_map(turns, lines, founder_aliases, boundary)
    insertions: list[QuestionInsertion] = []
    omitted: list[OmittedCandidate] = []
    for position, turn in enumerate(turns):
        turn_index = int(turn.get("turn_index", turn.get("idx", -1)))
        if turn_index >= boundary:
            continue
        speaker = str(turn.get("speaker", ""))
        text = str(turn.get("text", "")).strip()
        if _speaker_matches(speaker, target_aliases):
            continue
        original = next((name for name, aliases in other_aliases.items()
                         if _speaker_matches(speaker, aliases)), None)
        if original is None or not _is_question(text):
            continue
        if _norm(text) in canonical_normalized:
            omitted.append(OmittedCandidate(turn_index, speaker, text, "question_already_present"))
            continue
        if _has_verdict_or_offer(text):
            omitted.append(OmittedCandidate(turn_index, speaker, text, "verdict_or_offer_language"))
            continue
        answer_turn = next((
            candidate for candidate in turns[position + 1:]
            if int(candidate.get("turn_index", candidate.get("idx", -1))) < boundary
            and _speaker_matches(candidate.get("speaker"), founder_aliases)
        ), None)
        if answer_turn is None:
            omitted.append(OmittedCandidate(turn_index, speaker, text, "no_pre_boundary_founder_answer"))
            continue
        answer_index = int(answer_turn.get("turn_index", answer_turn.get("idx", -1)))
        line_index = founder_map.get(answer_index)
        if line_index is None:
            omitted.append(OmittedCandidate(
                turn_index, speaker, text, "founder_answer_not_in_canonical_pitch"))
            continue
        insertions.append(QuestionInsertion(
            source_turn_index=turn_index, answer_turn_index=answer_index,
            canonical_line_index=line_index, question=text, original_speaker=speaker,
        ))
    before: dict[int, list[QuestionInsertion]] = {}
    for insertion in insertions:
        before.setdefault(insertion.canonical_line_index, []).append(insertion)
    output: list[str] = []
    for index, (raw, _, _) in enumerate(lines):
        for insertion in sorted(before.get(index, []), key=lambda row: row.source_turn_index):
            output.append(f"Panel Investor: {insertion.question}")
        output.append(raw)
    return PanelQuestionArtifact(
        pitch_text="\n".join(output) + ("\n" if output else ""),
        insertions=tuple(insertions), omitted=tuple(omitted),
        boundary_turn_index=boundary,
    )
