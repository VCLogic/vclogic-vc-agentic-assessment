"""Source-bound historical dialogue, distinct from decision-blind evaluation pitches.

Review pitch ends mark the first decision, not the end of the conversation.
Extend through reviewed in-room decision evidence, never through later evidence.
Unattributed source lines are not safe continuations: the scraper also puts host
narration there. Only explicitly speaker-labelled statements are published.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any

from .precedent_builder import (
    _SPEAKER, _SPLIT_COLON, _SPLIT_SPEAKER, _locator_text, parse_speaker_turns,
)


_SHOW_COMMENTARY = re.compile(
    r"\b(?:(?:last|previous|next|this|today[’']s) (?:episode|pitch on the show)"
    r"|previous pitch|(?:next|last) week on (?:the|this) show"
    r"|on (?:the|today[’']s|this) show|(?:last|previous|earlier) season"
    r"|(?:hears?|hear|listen(?:s|ing)? to) the episode)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HistoricalConversation:
    transcript: str
    original_to_clean: dict[int, int]
    audit: dict[str, Any]
    exclusion_reason: str | None = None

    @property
    def included(self) -> bool:
        return self.exclusion_reason is None


def _explicit_statements(transcript: str) -> dict[int, str]:
    """Match the legacy parser's indexes but retain only explicitly labelled text."""
    pending: list[tuple[str, list[str], str]] = []
    lines = transcript.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        match = _SPEAKER.match(line)
        if match:
            initial = match.group(3)
            pending.append((match.group(1), [initial] if initial else [], initial))
        elif (_SPLIT_SPEAKER.fullmatch(line) and index + 1 < len(lines)
              and (colon := _SPLIT_COLON.match(lines[index + 1]))):
            initial = colon.group(1)
            pending.append((line, [initial] if initial else [], initial))
            index += 1
        elif pending:
            speaker, body, initial = pending[-1]
            # An empty explicit header labels the immediately following statement.
            # Later lines remain untrusted, even when the legacy parser merges them.
            if not body and line and speaker != "Narrator":
                pending[-1] = (speaker, body, line)
            pending[-1][1].append(line)
        elif line:
            pending.append(("Narrator", [line], ""))
        index += 1
    return {i: initial for i, (_, _, initial) in enumerate(
        item for item in pending if "\n".join(item[1])
    )}


def clean_historical_conversation(
    episode: Mapping[str, Any],
    review_document: Mapping[str, Any] | None,
    *,
    participant_aliases: Sequence[str] = (),
) -> HistoricalConversation:
    """Return clean dialogue and original-to-clean indexes, or an explicit exclusion.

    ``review_document`` is the complete reviews/<slug>.json document. The A/B
    candidate pitch, excluded_turns, and verification.remove_lines are deliberately
    not applied: they suppress decisions which belong in historical dialogue.
    """
    transcript = episode.get("transcript")
    audit: dict[str, Any] = {"schema": "historical-conversation-v1", "removed_turns": [],
                            "trimmed_turns": []}

    def exclude(reason: str) -> HistoricalConversation:
        return HistoricalConversation("", {}, {**audit, "exclusion_reason": reason}, reason)

    if not isinstance(transcript, str) or not transcript.strip():
        return exclude("missing_transcript")
    if not isinstance(review_document, Mapping):
        return exclude("missing_review")
    review = review_document.get("review")
    if not isinstance(review, Mapping) or review.get("episode_kind") != "pitch":
        return exclude("missing_or_non_pitch_review")
    slug = episode.get("slug") or episode.get("episode_slug")
    if slug and review_document.get("episode_slug") != slug:
        return exclude("review_episode_mismatch")
    turns = parse_speaker_turns(transcript)
    start, pitch_end = review.get("pitch_start_turn"), review.get("pitch_end_turn")
    if (type(start) is not int or type(pitch_end) is not int
            or not 0 <= start < pitch_end <= len(turns)):
        return exclude("invalid_review_boundaries")
    founders = review.get("founder_speakers")
    if not isinstance(founders, list) or not founders or not all(isinstance(x, str) for x in founders):
        return exclude("missing_founder_speakers")
    names = set(founders)
    for key in ("founders", "panel"):
        for person in episode.get(key, []) or []:
            if isinstance(person, Mapping) and isinstance(person.get("name"), str):
                names.add(person["name"])
    names.update(alias for alias in participant_aliases if isinstance(alias, str))
    full_names = {name.strip() for name in names if name.strip()}
    first_counts = Counter(name.split()[0].casefold() for name in full_names if " " in name)
    names.update(name.split()[0] for name in full_names
                 if first_counts[name.split()[0].casefold()] <= 1)
    allowed = {_locator_text(name).casefold() for name in names}
    evidence: list[tuple[int, str]] = []
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        return exclude("missing_decision_review")
    for decision in decisions:
        if not isinstance(decision, Mapping):
            return exclude("invalid_decision_review")
        if decision.get("decision_context") not in {"initial_panel", "same_session_reversal"}:
            continue
        receipts = decision.get("pitch_evidence", [])
        if not isinstance(receipts, list):
            return exclude("invalid_decision_evidence")
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                return exclude("invalid_decision_evidence")
            index, quote = receipt.get("turn_index"), receipt.get("quote")
            if type(index) is not int or not start <= index < len(turns) or not isinstance(quote, str) or not quote.strip():
                return exclude("invalid_decision_evidence")
            evidence.append((index, quote))
    if not evidence:
        return exclude("missing_in_room_decision_evidence")
    end = max(pitch_end - 1, *(index for index, _ in evidence))
    explicit = _explicit_statements(transcript)
    kept: list[str] = []
    mapping: dict[int, int] = {}
    for turn in turns:
        index = turn.turn_index
        reason = None
        if not start <= index <= end:
            reason = "outside_reviewed_conversation"
        elif _locator_text(turn.speaker).casefold() not in allowed:
            reason = "nonparticipant_speaker"
        elif not explicit[index].strip():
            reason = "no_explicit_statement"
        elif _SHOW_COMMENTARY.search(explicit[index]):
            reason = "episode_commentary"
        if reason:
            audit["removed_turns"].append({"original_turn_index": index, "reason": reason})
            continue
        statement = explicit[index]
        mapping[index] = len(kept)
        kept.append(f"{turn.speaker}: {statement}")
        if statement != turn.text:
            audit["trimmed_turns"].append({"original_turn_index": index,
                                           "reason": "unattributed_continuation"})
    for index, quote in evidence:
        if index not in mapping or _locator_text(quote) not in _locator_text(kept[mapping[index]]):
            return exclude("decision_evidence_not_preserved")
    clean = "\n".join(kept)
    if not kept or len(parse_speaker_turns(clean)) != len(kept):
        return exclude("invalid_clean_turn_mapping")
    audit.update({"source_turn_count": len(turns), "clean_turn_count": len(kept),
                  "original_start_turn": start, "original_end_turn_inclusive": end,
                  "participants": sorted(allowed),
                  "original_to_clean": {str(k): v for k, v in mapping.items()}})
    return HistoricalConversation(clean, mapping, audit)
