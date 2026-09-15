"""Deterministic preparation of leakage-reduced pitch-window inputs."""

from __future__ import annotations

import re
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .precedents import TranscriptTurn
from .precedent_builder import parse_speaker_turns


_SENTENCE = re.compile(r"[^.!?]+[.!?]?")
_QUESTION_START = re.compile(
    r"(?is)^\s*(?:(?:and|but|so|okay|well|like|i mean)\s+)*"
    r"(?:what|why|how|when|where|who|which|can|could|would|will|should|"
    r"do|does|did|are|is|was|were|have|has|had|tell (?:us|me)|"
    r"walk (?:us|me)|help (?:us|me) understand|explain|describe|remind me|"
    r"give (?:us|me))\b"
)
_DECISION_SIGNAL = re.compile(
    r"(?is)\b(?:i(?:'m| am) (?:in|out|a pass|passing)|i(?:'ll| will) pass|"
    r"i(?:'d| would) (?:like|love) to invest|i(?:'m| am) going to invest|"
    r"i(?:'m| am) in for|we(?:'re| are) out|sit out|conflicted out|"
    r"offer(?:ing)? \$?[0-9]|my offer)\b"
)
_ACK = re.compile(
    r"(?is)^\s*(?:thank you|thanks|awesome|amazing|great|okay|ok|got it|"
    r"appreciate it|sounds good|fantastic|perfect)"
    r"(?:\s*,?\s*(?:i\s+)?appreciate it)?[.!\s\[\]-]*$"
)
_STAGE_DIRECTION = re.compile(
    r"(?is)^\s*\[(?:[A-Za-z .'-]+\s+)?(?:laughter|laughs?|hellos?|crosstalk(?: yes)?|"
    r"mornings?|pause|demo continues|video plays|wows?|thank yous?|coughs?)\]\s*$"
)
_EDITORIAL_LINE = re.compile(
    r"(?is)^\s*(?:"
    r"i(?:'|’)m josh muccio\b|today on the show\b|this is the pitch\b|"
    r"the pitch for\b|our food critics today\b|and you,? our listeners\b|"
    r"so that(?:'|’)s one pass\b|can this .*\binvestors?\b|oof\b|"
    r"alright,?\s+time to\b|this ex-trucker\b|so far this pitch\b|"
    r"but even the finest\b|well.*\bthat was fast\b|the vcs? are looking\b|"
    r"cold cases\?*\b|can a media company\b|eaps?, by the way\b|oo+h\b|"
    r"a world without\b|those darn humans\b|ju+st kidding\b|it works[.!]|"
    r"we(?:'|’)ll be right back\b|the pitch .*\bright back\b|"
    r".*\bmakes an offer\b|okay i(?:'|’)ll bite\b|"
    r"i just want to recognize how\b|"
    r"so [A-Z][A-Za-z0-9 .'-]+ is doing well\b|it(?:'|’)s pretty clear\b|"
    r"lending\. from the\b|\[\d{2}:\d{2}:\d{2}\]|"
    r"\.\s*(?:yep|yeah|wow)\b|"
    r"so \d+(?:\.\d+)?% of\b|"
    r".*\bis out early\b|.*\bis in on\b|.*\bright after this\b"
    r")"
)
_OUTCOME_NARRATION_LINE = re.compile(
    r"(?is)^\s*(?:"
    r"[A-Z][A-Za-z'-]+ needs another million\b|"
    r"has [A-Za-z .'-]+ convinced (?:the )?investors?\b|"
    r"(?:so )?(?!We\b|Her\b|Our\b|They\b)[A-Z][A-Za-z'-]+ "
    r"(?:just )?passed(?: on the deal)?(?:\s*[,?.]|\s*$)|"
    r"everyone is out\b|"
    r"[A-Za-z .'-]+ is out(?:\s*[,!.?]|\s*$)|"
    r"(?:one|two|three|four|five|\d+) out of "
    r"(?:one|two|three|four|five|\d+) investors? have passed\b|"
    r"all (?:of )?the investors? said no\b|"
    r"only [A-Za-z .'-]+ remains\b|"
    r"last investor (?:left|remaining)\b|"
    r"(?:so|both) [A-Za-z .'-]*\b(?:is|are) in\b.*\bhere(?:'|’)s\b|"
    r"so [A-Za-z .'-]*\bis in(?:\s*[,!.?]|\s*$)|"
    r"both [A-Za-z .'-]*\bare in\b|"
    r"out[,.]?\s+here(?:'|’)s\b|"
    r"[A-Za-z .'-]+(?:'|’)s out[,.]?\s+here(?:'|’)s\b|"
    r"[A-Za-z .'-]+(?:'|’)s out(?:\s*[,!.?]|\s*$)|"
    r"[A-Z][A-Za-z'-]+ is the (?:only|last) investor (?:left|remaining)\b|"
    r"it(?:'|’)s decision time\b|"
    r"let(?:'|’)s see (?:if|how)\b|"
    r"have the founders done enough to get (?:the )?investors? on board\b|"
    r"(?:so )?investors? are thinking\b|"
    r"will (?:he|she|they) pass the test with (?:the|our) investors?\b|"
    r"will investors? decide\b|"
    r"(?:now )?can (?:they|the founders?) get investors? to\b|"
    r"you in or you out\?"
    r")"
)
_STRUCTURAL_NARRATION_START = re.compile(
    r"(?is)^\s*(?:\.\.\.)?(?:"
    r"from gimlet\b|this is the pitch\b|the pitch\b|i(?:'|’)m josh muccio\b|"
    r"today(?:,|(?:'|’)s\b|\s+(?:on|we|the|investors?)\b)|this week\b|"
    r"welcome back\b|when we (?:come|get) back\b|after the break\b|"
    r"coming up\b|stay tuned\b|alright,? on with (?:the )?pitch\b|"
    r"and now it(?:'|’)s time to join the investors\b|at last--?it(?:'|’)s showtime\b|"
    r"right at the end of the pitch\b|did you catch that\b|"
    r"well\.*\s+that was fast\b|"
    r"a declining .+\bnot an encouraging sign\b|"
    r"i gotta say\b|if you are thinking this pitch\b|still confused\b|"
    r"it sounds like .+\binvestors? (?:are|were|seem|look|sound)?\s*nervous\b|"
    r"pretty sure .+\b(?:doesn(?:'|’)t|does not) want .+ money\b|"
    r"but [A-Z][A-Za-z'-]+ will have to convince investors\b|"
    r"[A-Z][A-Za-z'-]+ is getting into the weeds\b|"
    r"[A-Z][A-Za-z'-]+ confidently strides .+\binvestors? hands\b|"
    r"the investors? (?:are|were|seem|look|sound) .+\b(?:leaning in|nervous)\b|"
    r"the investors? (?:are|were) trying to decide\b|"
    r"the investors? sound\b|the founder (?:has|is|was|seems|looks)\b|"
    r"you can almost see .+\binvestors?\b|"
    r"the mention of .+ gets everyone(?:'|’)s attention\b|"
    r"everyone agrees that\b|bringing up a big issue\b|"
    r"for right now,? .+\b(?:wants|needs) to\b|in other words\b|"
    r"traditionally there(?:'|’)s only one option\b|don(?:'|’)t ever underestimate\b|"
    r"[A-Z][A-Za-z'-]+ first created (?:her|his|their) company\b|"
    r"[A-Z][A-Za-z'-]+ pulls out\b|"
    r"what [A-Z][A-Za-z'-]+ means is\b|"
    r"once [A-Z][A-Za-z'-]+ (?:had|has|developed|created|found)\b|"
    r"that(?:'|’)s [A-Z][A-Za-z'-]+,? the co-founder\b|"
    r"basically,? we(?:'|’)ve all gotten accustomed\b|"
    r"so [A-Z][A-Za-z'-]+(?:'|’)s saying\b|"
    r"for those wondering\b|while [A-Z][A-Za-z'-]+ and [A-Z][A-Za-z'-]+\b|"
    r"the first incarnation of\b|after spending .+\bthe pitch\b|"
    r"(?:but )?what happens when someone tries to reinvent\b|"
    r"in a manner familiar to\b|"
    r"so,? [A-Z][A-Za-z'-]+ explains\b|"
    r"[A-Z][A-Za-z'-]+ says that\b|"
    r"eventually [A-Z][A-Za-z'-]+ moved on\b|"
    r"and this is when [A-Z][A-Za-z'-]+ pulls out\b|"
    r"it(?:'|’)s a clear sign that\b|"
    r"so [A-Z][A-Za-z'-]+(?:'|’)s plan is\b|"
    r"when [A-Z][A-Za-z'-]+ says\b|"
    r"[A-Z][A-Za-z'-]+(?:'|’)s trying to say\b|"
    r"after seeing these\b|for a lot of venture capitalists\b|"
    r"let(?:'|’)s find out if (?:the|our) investors?\b|"
    r"basically,? what [A-Z][A-Za-z'-]+ is saying\b|"
    r"but [A-Z][A-Za-z'-]+ actually sees\b|"
    r"it(?:'|’)s one thing when investors?\b|"
    r"[A-Z][A-Za-z'-]+(?:'|’)s first bare-bones product\b|"
    r"[A-Z][A-Za-z'-]+ has already proven that\b|"
    r"[A-Z][A-Za-z'-]+ got the title\b|"
    r"but,? for [A-Z][A-Za-z'-]+,|"
    r"[A-Z][A-Za-z'-]+ is not just any\b|"
    r"so (?:he|she|they) had an idea\b|"
    r"[A-Z][A-Za-z'-]+ [A-Z][A-Za-z'-]+ is the business side\b|"
    r"[A-Z][A-Za-z'-]+ explains that\b|"
    r"\d+-\d+(?:,\s*\d+-\d+)+.*\bthe point is\b|"
    r"[A-Z][A-Za-z'-]+ says the game\b|"
    r"once [A-Z][A-Za-z'-]+ began to dig into\b|"
    r"all right,? so the way .+ works is\b|"
    r"in short,|so if you actually know\b|in this case,|"
    r"[A-Z][A-Za-z'-]+(?:'|’)s pushing the investors?\b|"
    r"[A-Z][A-Za-z'-]+(?:'|’)s vision for\b|"
    r"with both .+\bin[!.]\s+the last up is\b|"
    r"and while that might be an honorable mission\b|"
    r"what (?:he|she|they)(?:'|’)s built with\b|"
    r"when a newsroom is using\b|this is where companies are targeting\b|"
    r"here(?:'|’)s\s*\.\s*$"
    r")"
)
_PANEL_SENTIMENT = re.compile(
    r"(?is)\b(?:investors? (?:are|were|seem|look|sound)\s+"
    r"(?:leaning in|nervous)|good sign for|not an encouraging sign|"
    r"founder(?:'|’)s job .+ project confidence|"
    r"whether .+ should even be\s+leading\s+the company)\b"
)
_ALLOWED_DECISION_CONTEXTS = {
    "initial_panel",
    "same_session_reversal",
    "later_diligence",
    "off_panel",
    "unclear",
}


@dataclass(frozen=True)
class PitchPackagePaths:
    pitch: Path
    audit: Path


def package_relative_path(path: str | Path, package_root: str | Path) -> str:
    """Return a normalized provenance path inside the package root."""
    return Path(path).resolve().relative_to(Path(package_root).resolve()).as_posix()


def _name_keys(name: str) -> set[str]:
    normalized = " ".join(name.casefold().split())
    if not normalized:
        return set()
    return {normalized, normalized.split()[0]}


def _matches_name(speaker: str, names: Sequence[str]) -> bool:
    normalized_speaker = " ".join(speaker.casefold().split())
    return any(normalized_speaker in _name_keys(name) for name in names)


def _is_structural_narration(text: str) -> bool:
    candidate = text.strip()
    return bool(
        _STAGE_DIRECTION.fullmatch(candidate)
        or _STRUCTURAL_NARRATION_START.search(candidate)
        or _OUTCOME_NARRATION_LINE.search(candidate)
        or _PANEL_SENTIMENT.search(candidate)
    )


def _is_editorial_question(text: str) -> bool:
    lowered = " ".join(text.casefold().split())
    return bool(
        _is_structural_narration(text)
        or
        re.search(
            r"^(?:the|our) investors?\s+"
            r"(?:sound|want|need|have|think|decide|see|know|wonder|look|seem|"
            r"get|write|slam)\b|"
            r"^(?:can|will|would|could|do|did|have) (?:the|our) investors?\b|"
            r"^what [a-z .'-]+ is offering the founder\b|"
            r"^the pitch room\b|^before\b|^you in or you out\??$",
            lowered,
        )
    )


def _is_panel_verdict_or_evaluation_question(
    text: str, panel_names: Sequence[str] = ()
) -> bool:
    """Reject questions that expose a panelist's view of the current deal."""
    candidate = " ".join(text.replace("’", "'").split())
    exact_panel_names = tuple(
        re.escape(" ".join(name.split())).replace(r"\ ", r"\s+")
        for name in panel_names
        if name.strip()
    )
    actor = (
        "(?:"
        + "|".join(("i", "you", "we", "they", "he", "she", *exact_panel_names))
        + ")"
    )
    current_deal = r"(?:this|it|the (?:company|business|deal|pitch))"
    terminal = r"(?=\s*(?:[?!.;,]|$))"
    patterns = (
        # Direct in/out/pass/done checks, including named third parties.
        rf"(?i)\b(?:am|are|is|was|were)\s+{actor}\s+(?:still\s+)?"
        rf"(?:in|out|passing){terminal}",
        rf"(?i)\b{actor}(?:'m|'re|'s|\s+(?:am|are|is|was|were))\s+"
        rf"(?:still\s+)?(?:in|out|passing){terminal}",
        rf"(?i)\b(?:am|are|is|was|were)\s+{actor}\s+(?:still\s+)?done"
        rf"(?:\s+with\s+{current_deal})?{terminal}",
        rf"(?i)\b{actor}(?:'m|'re|'s|\s+(?:am|are|is|was|were))\s+"
        rf"(?:still\s+)?done(?:\s+with\s+{current_deal})?{terminal}",
        rf"(?i)\b(?:did|do|does|will|would|can|could|should|has|have)\s+"
        rf"{actor}\s+(?:still\s+)?pass(?:ed|ing)?\b(?!\s+through\b)",
        # A current-deal investment intention or allocation.
        rf"(?i)\b{actor}(?:(?:'m|'re|'s)|\s+(?:am|are|is))?\s+"
        rf"(?:gonna|(?:going|want(?:s)?|ready)\s+to|would|will)\s+"
        rf"invest(?:ing)?\s+(?:in\s+)?{current_deal}\b",
        rf"(?i)\b(?:are|is|was|were|do|does|did|will|would|can|could|should)\s+"
        rf"{actor}\s+(?:(?:gonna|going)\s+to\s+)?invest(?:ing)?\s+"
        rf"(?:in\s+)?{current_deal}\b",
        rf"(?i)\bhow much\s+(?:are|is|do|does|will|would)\s+{actor}\s+"
        rf"(?:(?:gonna|going)\s+to\s+)?(?:invest|put\s+in|come\s+in\s+at|"
        rf"go\s+in\s+for){terminal}",
        rf"(?i)\bhow much\s+did\s+(?:you|we)\s+"
        rf"(?:come\s+in\s+at|go\s+in\s+for){terminal}",
        rf"(?i)\b{actor}(?:(?:'m|'re|'s)|\s+(?:am|are|is|was|were))?\s+"
        rf"(?:gonna\s+|(?:going|want(?:s)?)\s+to\s+)?go\s+in"
        rf"(?:\s+for\s+(?:\$?\d[\d,.]*|how much))?{terminal}",
        rf"(?i)\b(?:am|are|is|was|were)\s+{actor}\s+(?:still\s+)?in\s+for\b",
        rf"(?i)\b{actor}(?:'m|'re|'s|\s+(?:am|are|is|was|were))\s+"
        rf"(?:still\s+)?in\s+for\b",
        # Offers and current-pitch affect are themselves panel evaluations.
        r"(?i)\bwhat(?:'s| is)\s+(?:my|your|our|his|her|their)\s+offer\b",
        rf"(?i)\b(?:have|has)\s+{actor}\s+(?:ever\s+)?invested{terminal}",
        rf"(?i)\b(?:you know\s+)?what\s+(?:do\s+)?{actor}\s+"
        rf"(?:love|hate|like)\s+about\s+{current_deal}\b",
        rf"(?i)\b(?:do|does|did|would)\s+{actor}\s+"
        rf"(?:love|hate|like)\s+{current_deal}\b",
        # Whether a panelist has been persuaded is also verdict state.
        rf"(?i)\b(?:am|are|is|was|were|have|has|had)\s+{actor}\s+"
        rf"(?:been\s+)?convinced(?:\s+by\s+{current_deal})?{terminal}",
        rf"(?i)\bdid\s+{actor}\s+convince\s+{actor}{terminal}",
        # Panel-name redaction must not turn a verdict into an actorless question.
        rf"(?i)\b(?:am|are|is)\s+(?:still\s+)?(?:in|out|passing){terminal}",
        rf"(?i)\b(?:did|do|does|should)\s+(?:still\s+)?pass(?:ed|ing)?"
        rf"(?:\s+on\s+{current_deal})?{terminal}",
        rf"(?i)\b(?:will|would|can|could|should)\s+invest(?:ing)?\s+"
        rf"(?:in\s+)?{current_deal}{terminal}",
        rf"(?i)\bhow much\s+(?:are|is|will|would)\s+"
        rf"(?:(?:gonna|going)\s+to\s+)?(?:invest|put\s+in|come\s+in\s+at|"
        rf"go\s+in\s+for){terminal}",
        rf"(?i)\bdid\s+convince\s+(?:me|you|us|them|him|her){terminal}",
    )
    return any(re.search(pattern, candidate) for pattern in patterns)


def _is_investor_question(
    candidate: str, panel_names: Sequence[str] = ()
) -> bool:
    if (
        "?" not in candidate
        or _is_editorial_question(candidate)
        or _is_panel_verdict_or_evaluation_question(candidate, panel_names)
    ):
        return False
    if _QUESTION_START.search(candidate):
        return True
    words = re.findall(r"[A-Za-z0-9$]+", candidate)
    lowered = " ".join(candidate.casefold().split())
    return bool(
        len(words) <= 12
        and (
            lowered.endswith("right?")
            or lowered.startswith(("$", "for ", "both ", "two time "))
            or re.search(r"\b(?:you|your)\b", lowered)
        )
    )


def _audit_question_reveals_panel_stance(
    text: str, panel_names: Sequence[str] = ()
) -> bool:
    """Independently detect verdict/evaluation semantics in rendered questions."""
    lowered = " ".join(text.casefold().replace("’", "'").split())
    deal = r"(?:this|it|the (?:company|business|deal|pitch))"
    actor = r"(?:i|you|we|they|he|she)"
    exact_panel_names = tuple(
        re.escape(" ".join(name.casefold().split())).replace(r"\ ", r"\s+")
        for name in panel_names
        if name.strip()
    )
    person = "(?:" + "|".join((actor, *exact_panel_names)) + ")"
    checks = (
        rf"\b(?:am|are|is|was|were)\s+(?:{person}\s+)?(?:still\s+)?"
        r"(?:in|out|passing)(?=\s*[?!.;,])",
        rf"\b{actor}(?:'m|'re|'s|\s+(?:am|are|is|was|were))\s+"
        r"(?:still\s+)?(?:in|out|passing)(?=\s*[?!.;,])",
        rf"\b(?:did|do|does|will|would|should|have|has)\s+{person}\s+"
        r"(?:still\s+)?pass(?:ed|ing)?\b(?!\s+through\b)",
        rf"\b{actor}\b[^?!.;]*\b(?:gonna|going to|want to|would|will)\s+"
        rf"(?:invest\s+(?:in\s+)?{deal}|go\s+in\b)",
        rf"\b(?:are|is|do|does|did|will|would)\s+{person}\b[^?!.;]*"
        rf"\binvest\s+(?:in\s+)?{deal}\b",
        rf"\bhow much\s+(?:are|is|do|does|will|would)\s+{person}\b[^?!.;]*"
        r"\b(?:invest|put in|come in at|go in for)(?=\s*[?!.;,])",
        r"\bhow much\s+did\s+(?:you|we)\s+(?:come in at|go in for)"
        r"(?=\s*[?!.;,])",
        rf"\b(?:are|is)\s+{person}\s+(?:still\s+)?in\s+for\b",
        rf"\b{actor}(?:'re|'s|\s+(?:are|is))\s+(?:still\s+)?in\s+for\b",
        r"\b(?:my|your|our|his|her|their)\s+offer\b",
        rf"\b(?:have|has)\s+{actor}\s+(?:ever\s+)?invested(?=\s*[?!.;,])",
        rf"\b{actor}\b\s+(?:really\s+)?(?:love|hate|like)\s+"
        rf"(?:about\s+)?{deal}\b",
        r"\b(?:been\s+)?convinced\s+by\s+the\s+pitch\b",
        rf"\bdid\s+(?:(?:the\s+pitch|{person})\s+)?convince\s+{actor}"
        r"(?=\s*[?!.;,])",
        rf"\b{actor}(?:'m|'re|'s|\s+(?:am|are|is))\s+"
        r"(?:still\s+)?done(?=\s*[?!.;,])",
        r"^(?:am|are|is)\s+(?:still\s+)?done(?=\s*[?!.;,])",
        r"\b(?:am|are|is)\s+(?:still\s+)?(?:in|out|passing)(?=\s*[?!.;,])",
        rf"\b(?:did|do|does|should)\s+(?:still\s+)?pass(?:ed|ing)?"
        rf"(?:\s+on\s+{deal})?(?=\s*[?!.;,])",
        rf"\b(?:will|would|can|could|should)\s+invest(?:ing)?\s+"
        rf"(?:in\s+)?{deal}(?=\s*[?!.;,])",
        r"\bhow much\s+(?:are|is|will|would)\s+"
        r"(?:(?:gonna|going)\s+to\s+)?(?:invest|put in|come in at|go in for)"
        r"(?=\s*[?!.;,])",
    )
    return any(re.search(check, lowered) for check in checks)


def neutral_question_text(
    text: str, panel_names: Sequence[str] = ()
) -> str:
    """Keep question clauses while dropping surrounding investor evaluation."""
    kept: list[str] = []
    for piece in _SENTENCE.findall(text or ""):
        candidate = piece.strip()
        if not candidate:
            continue
        if _is_editorial_question(candidate):
            break
        if _is_investor_question(candidate, panel_names):
            kept.append(candidate)
        elif kept:
            break
    return " ".join(kept).strip()


def _remove_panel_names(text: str, panel_names: Sequence[str]) -> str:
    output = text
    variants: set[str] = set()
    for name in panel_names:
        parts = name.split()
        if parts:
            variants.add(parts[0])
        if name:
            variants.add(name)
    for variant in sorted(variants, key=len, reverse=True):
        escaped = re.escape(variant)
        output = re.sub(
            rf"(?is)\b(?:hi|hey|hello|good to see you|nice to meet you)\s*,?\s*"
            rf"{escaped}\b[.!?]?\s*",
            "",
            output,
        )
        output = re.sub(rf"(?is)\b{escaped}(?:'s|’s)?\b\s*[,?:]?\s*", "", output)
    output = re.sub(r"[ \t]{2,}", " ", output).strip()
    return output


def _strip_editorial_lines(text: str, subject_names: Sequence[str]) -> str:
    kept: list[str] = []
    lines = text.splitlines()
    for line_index, line in enumerate(lines):
        if "[BREAK]" in line:
            prefix = line.split("[BREAK]", 1)[0].rstrip()
            if prefix:
                kept.append(prefix)
            break
        candidate = line.strip()
        if not candidate:
            continue
        if candidate.casefold() in {"and", "but", "c"}:
            continue
        following_lines = " ".join(
            part.strip() for part in lines[line_index + 1 : line_index + 4]
        )
        if (
            candidate.casefold() == "at home."
            and following_lines
            and _is_structural_narration(following_lines)
        ):
            break
        if _EDITORIAL_LINE.search(candidate) or _is_structural_narration(candidate):
            break
        boundary_start: int | None = None
        for sentence in _SENTENCE.finditer(line):
            sentence_text = sentence.group().strip()
            if _EDITORIAL_LINE.search(sentence_text) or _is_structural_narration(
                sentence_text
            ):
                boundary_start = sentence.start()
                break
        if boundary_start is not None:
            prefix = line[:boundary_start].rstrip()
            if prefix:
                kept.append(prefix)
            break
        kept.append(line)
    return "\n".join(kept)


def _sanitize_pitch_text(
    text: str,
    panel_names: Sequence[str],
    subject_names: Sequence[str] = (),
) -> str:
    without_editorial = _strip_editorial_lines(text, subject_names)
    without_names = _remove_panel_names(without_editorial, panel_names)
    sanitized = _strip_editorial_lines(without_names, subject_names).strip()
    return sanitized if re.search(r"\w", sanitized) else ""


def _attributed_line(text: str) -> tuple[str, int | None, int]:
    """Return the one explicit content line and count later unlabeled lines."""
    nonempty = tuple(
        (line_index, line)
        for line_index, line in enumerate(text.splitlines())
        if line.strip()
    )
    if not nonempty:
        return "", None, 0
    line_index, line = nonempty[0]
    return line, line_index, len(nonempty) - 1


def _ordered_token_subset(candidate: str, source: str) -> bool:
    candidate_tokens = re.findall(r"\w+", candidate.casefold())
    source_tokens = iter(re.findall(r"\w+", source.casefold()))
    return bool(candidate_tokens) and all(
        any(source_token == token for source_token in source_tokens)
        for token in candidate_tokens
    )


def _structural_provenance_findings(
    *,
    pitch: str,
    segments: Sequence[dict[str, Any]],
    turns: Sequence[TranscriptTurn],
    founder_names: Sequence[str],
    panel_names: Sequence[str],
) -> list[str]:
    """Independently verify every rendered line came from one attributed line."""
    findings: list[str] = []
    rendered_segments = [segment for segment in segments if segment.get("rendered_text")]
    if [segment["rendered_text"] for segment in rendered_segments] != pitch.splitlines():
        findings.append("rendered pitch lines do not match classified segments")

    for segment in rendered_segments:
        source_index = segment.get("source_turn_index")
        if type(source_index) is not int or not 0 <= source_index < len(turns):
            findings.append("rendered segment has invalid source turn")
            continue
        turn = turns[source_index]
        nonempty = tuple(
            (line_index, line)
            for line_index, line in enumerate(turn.text.splitlines())
            if line.strip()
        )
        if not nonempty:
            findings.append(f"turn {source_index} has no attributed content")
            continue
        attributed_index, attributed_source = nonempty[0]
        if segment.get("attributed_source_line_index") != attributed_index:
            findings.append(f"turn {source_index} has incorrect attributed line index")
        if segment.get("unattributed_continuation_count") != len(nonempty) - 1:
            findings.append(f"turn {source_index} has incorrect continuation count")
        if segment.get("attributed_source_line_sha256") != sha256(
            attributed_source.encode("utf-8")
        ).hexdigest():
            findings.append(f"turn {source_index} has incorrect attributed line hash")

        output = segment.get("rendered_text")
        if not isinstance(output, str) or "\n" in output:
            findings.append(f"turn {source_index} rendered multiple lines")
            continue
        content_class = segment.get("retained_content_class")
        if content_class == "explicit_founder":
            prefix = f"{turn.speaker}: "
            if not _matches_name(turn.speaker, founder_names) or not output.startswith(prefix):
                findings.append(f"turn {source_index} is not an explicit founder line")
                continue
            rendered_content = output.removeprefix(prefix)
        elif content_class == "explicit_investor_question":
            prefix = "Investor question: "
            if not _matches_name(turn.speaker, panel_names) or not output.startswith(prefix):
                findings.append(f"turn {source_index} is not an explicit investor question")
                continue
            rendered_content = output.removeprefix(prefix)
            if "?" not in rendered_content or not re.search(r"\w", rendered_content):
                findings.append(f"turn {source_index} is not a structurally valid question")
            attributed_questions = tuple(
                piece.strip()
                for piece in _SENTENCE.findall(attributed_source)
                if "?" in piece
            )
            normalized_rendered = " ".join(rendered_content.casefold().split())
            relevant_attributed_questions = tuple(
                piece
                for piece in attributed_questions
                if " ".join(piece.casefold().split()) in normalized_rendered
                or " ".join(
                    _remove_panel_names(piece, panel_names).casefold().split()
                )
                in normalized_rendered
            )
            if any(
                _audit_question_reveals_panel_stance(piece, panel_names)
                for piece in relevant_attributed_questions
            ) or _audit_question_reveals_panel_stance(rendered_content):
                findings.append(
                    f"turn {source_index} reveals panel verdict or evaluation state"
                )
        else:
            findings.append(f"turn {source_index} has unclassified retained content")
            continue
        if not _ordered_token_subset(rendered_content, attributed_source):
            findings.append(f"turn {source_index} output is not derived from attributed content")
    return findings


def locate_investor_turn(
    turns: Sequence[TranscriptTurn],
    investor_aliases: Sequence[str],
    evidence_contains: str,
    occurrence: int | None = None,
) -> tuple[int, str]:
    """Locate one reviewed investor statement and return exact source wording."""
    needle = " ".join(evidence_contains.casefold().split())
    matches = [
        turn
        for turn in turns
        if _matches_name(turn.speaker, investor_aliases)
        and needle in " ".join(turn.text.casefold().split())
    ]
    if occurrence is not None:
        if occurrence < 1 or occurrence > len(matches):
            raise ValueError(
                f"reviewed decision occurrence {occurrence} is unavailable"
            )
        selected = matches[occurrence - 1]
        return selected.turn_index, selected.text
    if len(matches) != 1:
        raise ValueError(
            f"reviewed decision locator matched {len(matches)} investor turns"
        )
    return matches[0].turn_index, matches[0].text


def build_pitch_only_document(
    *,
    turns: Sequence[TranscriptTurn],
    founder_names: Sequence[str],
    panel_names: Sequence[str],
    endpoint: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Render founder evidence and neutral questions before a reviewed endpoint."""
    rendered: list[str] = []
    segments: list[dict[str, Any]] = []
    previous_investor_decision = False
    for turn in turns:
        if turn.turn_index >= endpoint:
            break
        action = "removed"
        reason = "removed non-founder, evaluative, or unnecessary context"
        output = ""
        retained_content_class: str | None = None
        preceded_by_investor_decision = previous_investor_decision
        attributed_text, attributed_line_index, continuation_count = _attributed_line(
            turn.text
        )
        attributed_line_hash = (
            sha256(attributed_text.encode("utf-8")).hexdigest()
            if attributed_line_index is not None
            else None
        )
        if _matches_name(turn.speaker, founder_names):
            founder_text = _sanitize_pitch_text(
                attributed_text, panel_names, founder_names
            )
            if founder_text and not (
                previous_investor_decision and _ACK.fullmatch(founder_text)
            ):
                action = "retained"
                reason = "founder pitch or pre-decision factual answer"
                retained_content_class = "explicit_founder"
                output = f"{turn.speaker}: {founder_text}"
                rendered.append(output)
            previous_investor_decision = False
        elif _matches_name(turn.speaker, panel_names):
            question = neutral_question_text(attributed_text, panel_names)
            question = _sanitize_pitch_text(question, panel_names)
            if question and _is_investor_question(question, panel_names):
                action = "neutralized"
                reason = "neutralized investor question"
                retained_content_class = "explicit_investor_question"
                output = f"Investor question: {question}"
                rendered.append(output)
            previous_investor_decision = bool(
                _DECISION_SIGNAL.search(attributed_text)
            )
        else:
            previous_investor_decision = False
        if continuation_count:
            reason += f"; removed {continuation_count} unattributed continuation line(s)"
        segments.append(
            {
                "action": action,
                "attributed_source_line_index": attributed_line_index,
                "attributed_source_line_sha256": attributed_line_hash,
                "reason": reason,
                "rendered_text": output,
                "retained_content_class": retained_content_class,
                "preceded_by_investor_decision": preceded_by_investor_decision,
                "source_turn_index": turn.turn_index,
                "speaker": turn.speaker,
                "unattributed_continuation_count": continuation_count,
            }
        )
    text = "\n".join(rendered)
    return (text + "\n" if text else "", segments)


def _contains_alias(text: str, aliases: Sequence[str]) -> bool:
    return any(
        re.search(rf"(?i)\b{re.escape(alias.strip())}\b", text)
        for alias in aliases
        if alias.strip()
    )


def _derive_leakage_checklist(
    *,
    pitch: str,
    segments: Sequence[dict[str, Any]],
    turns: Sequence[TranscriptTurn],
    endpoint: int,
    target_investor_aliases: Sequence[str],
    structural_findings: Sequence[str],
) -> dict[str, bool]:
    """Derive leakage findings from rendered text and recorded transformations."""
    lines = tuple(line.strip() for line in pitch.splitlines() if line.strip())
    investor_lines = tuple(
        line.removeprefix("Investor question:").strip()
        for line in lines
        if line.startswith("Investor question:")
    )
    rendered_segments = tuple(
        segment
        for segment in segments
        if isinstance(segment.get("rendered_text"), str)
        and segment["rendered_text"].strip()
    )
    return {
        "actual_label_in_package": any(
            _DECISION_SIGNAL.search(line) for line in investor_lines
        ),
        "complete_transcript_in_package": bool(
            turns
            and endpoint >= len(turns)
            and len(rendered_segments) == len(turns)
            and all(segment.get("action") == "retained" for segment in segments)
        ),
        "founder_decision_acknowledgement_retained": any(
            segment.get("action") == "retained"
            and segment.get("preceded_by_investor_decision") is True
            and isinstance(segment.get("rendered_text"), str)
            and _ACK.fullmatch(segment["rendered_text"].split(":", 1)[-1].strip())
            for segment in segments
        ),
        "investor_evaluation_retained": any(
            _DECISION_SIGNAL.search(line) or _is_editorial_question(line)
            for line in investor_lines
        ),
        "narrator_evaluation_retained": bool(structural_findings),
        "post_decision_material_retained": any(
            int(segment.get("source_turn_index", -1)) >= endpoint
            for segment in rendered_segments
        ),
        "prior_episode_inputs_in_package": bool(
            re.search(
                r"(?im)^\s*(?:next|last|previous) (?:week|episode)\b",
                pitch,
            )
        ),
        "target_investor_identity_in_pitch": _contains_alias(
            pitch, target_investor_aliases
        ),
        "unattributed_continuation_retained": bool(structural_findings),
    }


def compile_review_rows(
    *,
    review_rows: Sequence[dict[str, Any]],
    transcript_root: str | Path,
    vc_slug: str,
    investor_aliases: Sequence[str],
) -> list[dict[str, Any]]:
    """Expand compact human review rows into a source-linked decision ledger."""
    root = Path(transcript_root)
    compiled: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reviewed in review_rows:
        slug = reviewed["episode_slug"]
        if slug in seen:
            raise ValueError(f"duplicate review row: {slug}")
        seen.add(slug)
        status = reviewed["pitch_window_decision"]
        if status not in {"In", "Out", "Unobserved"}:
            raise ValueError(f"invalid pitch-window decision: {status!r}")
        decision_context = reviewed["decision_context"]
        if decision_context not in _ALLOWED_DECISION_CONTEXTS:
            raise ValueError(f"invalid decision context: {decision_context!r}")
        founder_aliases = reviewed.get("founder_aliases", [])
        if not isinstance(founder_aliases, list) or not all(
            isinstance(alias, str) and alias.strip() for alias in founder_aliases
        ):
            raise ValueError(f"invalid founder aliases: {slug}")
        source = root / f"{slug}.json"
        raw = json.loads(source.read_text(encoding="utf-8"))
        panel = raw.get("panel") or []
        metadata_on_panel = any(row.get("slug") == vc_slug for row in panel)
        on_panel = reviewed.get("on_panel", metadata_on_panel)
        turns = parse_speaker_turns(raw.get("transcript") or "")
        evidence_quote = ""
        evidence_index: int | None = None
        locator = reviewed.get("evidence_contains")
        if status in {"In", "Out"}:
            if not isinstance(locator, str) or not locator.strip():
                raise ValueError(f"observed review lacks evidence locator: {slug}")
            evidence_index, evidence_quote = locate_investor_turn(
                turns,
                investor_aliases,
                locator,
                reviewed.get("evidence_occurrence"),
            )
        elif locator:
            raise ValueError(f"unobserved review cannot cite a decision: {slug}")
        output: dict[str, Any] = {
            "episode_slug": slug,
            "final_decision": reviewed["final_decision"],
            "on_panel": bool(on_panel),
            "pitch_window_decision": status,
            "initial_response": reviewed["initial_response"],
            "decision_context": decision_context,
            "evaluation_eligible": status in {"In", "Out"},
            "label_basis": reviewed["label_basis"],
            "evidence_quote": evidence_quote,
            "evidence_source": f"data/episodes/{slug}.json transcript",
            "audit_notes": reviewed["audit_notes"],
        }
        if evidence_index is not None:
            output["evidence_turn_index"] = evidence_index
        if founder_aliases:
            output["founder_aliases"] = founder_aliases
        for optional in ("condition", "check_tier"):
            if optional in reviewed:
                output[optional] = reviewed[optional]
        compiled.append(output)
    return compiled


def write_pitch_package(
    *,
    episode_path: str | Path,
    compiled_review: dict[str, Any],
    investor_root: str | Path,
    vc_slug: str,
    audit_source: str,
) -> PitchPackagePaths | None:
    """Write one deterministic pitch-only document and its leakage audit."""
    if compiled_review.get("pitch_window_decision") not in {"In", "Out"}:
        return None
    endpoint = compiled_review.get("evidence_turn_index")
    if not isinstance(endpoint, int):
        raise ValueError("eligible review lacks a decision endpoint")
    source = Path(episode_path)
    source_bytes = source.read_bytes()
    episode = json.loads(source_bytes)
    metadata_founders = tuple(
        row["name"] for row in episode.get("founders", []) if row.get("name")
    )
    reviewed_founders = tuple(compiled_review.get("founder_aliases", ()))
    founders = tuple(dict.fromkeys((*metadata_founders, *reviewed_founders)))
    panel = tuple(
        row["name"] for row in episode.get("panel", []) if row.get("name")
    )
    target_investor_aliases = tuple(
        row["name"]
        for row in episode.get("panel", [])
        if row.get("name") and row.get("slug") == vc_slug
    )
    if not founders:
        raise ValueError("eligible episode has no founder identities")
    turns = parse_speaker_turns(episode.get("transcript") or "")
    pitch, segments = build_pitch_only_document(
        turns=turns,
        founder_names=founders,
        panel_names=panel,
        endpoint=endpoint,
    )
    if not pitch.strip():
        raise ValueError("eligible episode produced an empty pitch")
    if not any(
        segment["action"] == "retained"
        and _matches_name(segment["speaker"], founders)
        for segment in segments
    ):
        raise ValueError("eligible pitch must retain at least one founder turn")
    structural_findings = _structural_provenance_findings(
        pitch=pitch,
        segments=segments,
        turns=turns,
        founder_names=founders,
        panel_names=panel,
    )
    if structural_findings:
        raise ValueError(
            "structural provenance audit failed: " + "; ".join(structural_findings)
        )
    output = Path(investor_root)
    slug = compiled_review["episode_slug"]
    pitch_path = output / "pitches" / f"{slug}.txt"
    audit_path = output / "audits" / f"{slug}.json"
    pitch_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    pitch_path.write_text(pitch, encoding="utf-8")
    pitch_digest = sha256(pitch.encode("utf-8")).hexdigest()
    company_alias = " ".join(slug.split("-", 1)[-1].split("-")).title()
    checklist = _derive_leakage_checklist(
        pitch=pitch,
        segments=segments,
        turns=turns,
        endpoint=endpoint,
        target_investor_aliases=target_investor_aliases,
        structural_findings=structural_findings,
    )
    action_counts = Counter(segment["action"] for segment in segments)
    removed_continuations = sum(
        int(segment.get("unattributed_continuation_count", 0))
        for segment in segments
    )
    retained_founder_lines = sum(
        segment.get("retained_content_class") == "explicit_founder"
        for segment in segments
    )
    retained_questions = sum(
        segment.get("retained_content_class") == "explicit_investor_question"
        for segment in segments
    )
    audit = {
        "target_company_aliases": [company_alias],
        "audit_date": date.today().isoformat(),
        "auditor": "reviewed-ledger-plus-deterministic-renderer",
        "decision_window_endpoint": endpoint,
        "episode_slug": slug,
        "leakage_checklist": checklist,
        "pitch_path": (
            f"data/investors/{vc_slug}/pitches/{slug}.txt"
        ),
        "pitch_sha256": pitch_digest,
        "schema": "pitch-leakage-audit-v1",
        "segments": segments,
        "source_audit": audit_source,
        "source_transcript_sha256": sha256(source_bytes).hexdigest(),
        "status": "audited",
        "structural_provenance": {
            "findings": structural_findings,
            "removed_unattributed_continuations": removed_continuations,
            "retained_explicit_founder_lines": retained_founder_lines,
            "retained_explicit_investor_questions": retained_questions,
            "status": "verified",
        },
        "transformation_counts": {
            action: action_counts.get(action, 0)
            for action in ("neutralized", "removed", "retained")
        },
        "vc_slug": vc_slug,
    }
    audit["transformation_counts"]["removed_unattributed_continuations"] = (
        removed_continuations
    )
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return PitchPackagePaths(pitch=pitch_path, audit=audit_path)
