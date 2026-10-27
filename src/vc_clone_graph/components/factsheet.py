"""Neutral company facts from the founders' in-room answers after the pitch cutoff.

The pitch ends before the first panel decision. Founders keep answering
questions while the panel decides, and those answers carry facts the pitch
lacks, but also reactions that reveal decisions. Only founder turns between the
pitch cutoff and the last in-room panel decision are read; the model may return
only factual claims about the company, each with a verbatim quote, and any
claim that refers to the panel, feedback, offers or thanks is rejected.
One fact sheet per episode, shared by every investor on the panel.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Sequence

from ..dataset_prep import _sanitize_pitch_text
from ..precedents import TranscriptTurn
from ..providers.base import GenerationProvider, GenerationRequest
from .config import FactsheetSettings
from .coverage import _normalize

SCHEMA_VERSION = "founder-factsheet-v1"
IN_ROOM_CONTEXTS = frozenset({"initial_panel", "same_session_reversal"})
CATEGORIES = (
    "founder_team",
    "product_solution",
    "market_opportunity",
    "traction_growth",
    "business_model_economics",
    "competition_defensibility",
    "deal_terms_valuation",
    "timing",
)
# Reactions, references to the panel and to the outcome. Matched on quote and fact.
_REACTION = re.compile(
    r"(?i)\b(?:thanks?|thank\s+you|appreciat\w*|feedback|offer(?:s|ed|ing)?|"
    r"great\s+(?:question|point)|good\s+(?:question|point)|fair\s+(?:question|point)|"
    r"(?:would|i'd|we'd)\s+love\s+to|look(?:ing)?\s+forward|follow[\s-]*up|next\s+steps?|"
    r"pass(?:ed|ing)?\s+on|(?:i'm|we're|you're|they're)\s+(?:in|out)\b|in\s+for\b|"
    r"invest(?:ing|ed|ment)?\s+in\s+us|your\s+(?:fund|firm|check|portfolio|money)|"
    r"(?:you|y'all|you\s+all|you\s+guys)\s+(?:said|asked|mentioned|think|thought)|"
    r"we\s+(?:heard|hear)|agree|understand|totally|completely|absolutely|disappoint\w*|excited)"
)
_SECOND_PERSON = re.compile(r"(?i)\b(?:you|your|yours|y'all)\b")


def in_room_window(review: dict[str, Any]) -> tuple[int, int] | None:
    """(first, last) turn indices of the in-room Q&A after the pitch cutoff."""
    start = review["review"].get("pitch_end_turn")
    if review["review"].get("episode_kind") != "pitch" or not isinstance(start, int):
        return None
    ends = [
        evidence["turn_index"]
        for row in review.get("annotations", [])
        if row.get("decision_context") in IN_ROOM_CONTEXTS
        for evidence in row.get("pitch_evidence") or []
    ]
    end = max(ends, default=start - 1)
    return (start, end) if end >= start else None


def founder_turns(
    turns: Sequence[TranscriptTurn], review: dict[str, Any], panel_names: Sequence[str]
) -> list[dict[str, Any]]:
    window = in_room_window(review)
    if window is None:
        return []
    founders = tuple(review["review"]["founder_speakers"])
    rows = []
    for turn in turns:
        if window[0] <= turn.turn_index <= window[1] and turn.speaker in founders:
            text = _sanitize_pitch_text(turn.text, panel_names, founders)
            if len(text.split()) >= 4:
                rows.append({"turn_index": turn.turn_index, "speaker": turn.speaker, "text": text})
    return rows


def check_transcript(turns: Sequence[TranscriptTurn], review: dict[str, Any]) -> None:
    """The reviewed decision quotes must sit at their turn indices in this transcript."""
    by_index = {turn.turn_index: turn for turn in turns}
    for row in review.get("annotations", []):
        for evidence in row.get("pitch_evidence") or []:
            turn = by_index.get(evidence["turn_index"])
            if turn is None or evidence["quote"] not in turn.text:
                raise ValueError(f"transcript does not match review: {review['episode_slug']}")


def build_prompt(rows: Sequence[dict[str, Any]], settings: FactsheetSettings) -> str:
    lines = "\n".join(f"[T{r['turn_index']}] {r['speaker']}: {r['text']}" for r in rows)
    return f"""Below are statements a startup founder made while answering investors' questions. The questions are not shown.

Task: extract factual claims about the company that a reader could not learn without these statements.

Allowed: numbers and facts about the team, product, market, customers, traction, revenue, pricing, unit economics, competition, fundraising and deal terms, and the company's plans.
Not allowed, leave these out entirely:
- reactions, thanks, agreement, enthusiasm or disappointment;
- anything about the investors, their questions, their feedback, offers, interest or decisions;
- anything the founder says they heard or learned from the panel;
- opinions about how the conversation went.

Rules:
- Write each fact as a neutral third-person statement about the company ("The company ..."), at most 30 words, with no "you" or "your".
- quote: one contiguous excerpt, at most 40 words, copied exactly from a single statement; turn: its [T..] number.
- category: one of {", ".join(CATEGORIES)}.
- Return at most {settings.max_facts} facts. An empty list is correct when nothing qualifies.

Statements:
{lines}"""


def output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": list(CATEGORIES)},
                        "fact": {"type": "string"},
                        "quote": {"type": "string"},
                        "turn": {"type": "integer"},
                    },
                },
            },
        },
    }


def validate_facts(
    raw: dict[str, Any] | None,
    rows: Sequence[dict[str, Any]],
    panel_names: Sequence[str],
    settings: FactsheetSettings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("facts"), list):
        raise ValueError("fact sheet output is not a valid structured object")
    text_by_turn = {r["turn_index"]: _normalize(r["text"]) for r in rows}
    names = [n for name in panel_names for n in (name, name.split()[0]) if n]
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in raw["facts"]:
        fact, quote = str(item.get("fact", "")), str(item.get("quote", ""))
        turn = item.get("turn")
        reason = None
        if item.get("category") not in CATEGORIES:
            reason = "invalid_category"
        elif turn not in text_by_turn or len(_normalize(quote)) < 8 or _normalize(quote) not in text_by_turn[turn]:
            reason = "quote_not_found_in_turn"
        elif _REACTION.search(fact) or _REACTION.search(quote):
            reason = "reaction_or_panel_reference"
        elif _SECOND_PERSON.search(fact):
            reason = "second_person_in_fact"
        elif any(re.search(rf"(?i)\b{re.escape(n)}\b", fact + " " + quote) for n in names):
            reason = "panel_name"
        elif len(fact.split()) > 40 or len(quote.split()) > 50:
            reason = "too_long"
        elif len(kept) >= settings.max_facts:
            reason = "over_max_facts"
        if reason:
            rejected.append({**item, "rejection_reason": reason})
            continue
        kept.append({"fact_id": f"F{len(kept) + 1}", "category": item["category"],
                     "fact": fact, "quote": quote, "turn_index": turn})
    return kept, rejected


def run_factsheet(
    review: dict[str, Any],
    turns: Sequence[TranscriptTurn],
    panel_names: Sequence[str],
    settings: FactsheetSettings,
    provider: GenerationProvider,
) -> dict[str, Any]:
    check_transcript(turns, review)
    rows = founder_turns(turns, review, panel_names)
    base = {
        "schema": SCHEMA_VERSION,
        "episode_slug": review["episode_slug"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.model,
        "window": in_room_window(review),
        "source_turns": [r["turn_index"] for r in rows],
        "source_words": sum(len(r["text"].split()) for r in rows),
    }
    if not rows:
        return {**base, "status": "skipped_no_in_room_answers", "facts": [], "rejected": []}
    prompt = build_prompt(rows, settings)
    result = provider.generate(GenerationRequest(
        phase="founder_factsheet",
        prompt=prompt,
        schema=output_schema(),
        max_output_tokens=settings.max_output_tokens,
        reasoning_effort=settings.reasoning_effort,
    ))
    facts, rejected = validate_facts(result.parsed, rows, panel_names, settings)
    return {
        **base,
        "status": "completed",
        "prompt_chars": len(prompt),
        "facts": facts,
        "rejected": rejected,
        "usage": result.usage.model_dump(),
        "provider_metadata": result.raw_metadata,
    }
