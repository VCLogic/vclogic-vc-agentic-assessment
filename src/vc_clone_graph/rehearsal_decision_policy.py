"""Deterministic founder-rehearsal decision-language safeguards."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .rehearsal_schemas import FinalAssessment


_AMOUNT = re.compile(
    r"(?:(?:approximately|around|about)\s+)?(?:US\s*)?\$\s*"
    r"(?P<number>\d[\d,]*(?:\.\d+)?)\s*(?P<suffix>[kKmM])?"
)


def _amount_value(match: re.Match[str]) -> int:
    number = float(match.group("number").replace(",", ""))
    suffix = (match.group("suffix") or "").casefold()
    multiplier = 1_000 if suffix == "k" else 1_000_000 if suffix == "m" else 1
    return round(number * multiplier)


def _evidence_amounts(
    registry: Mapping[str, Mapping[str, Any]], source_kind: str
) -> set[int]:
    amounts: set[int] = set()
    for row in registry.values():
        if row.get("source_kind") != source_kind:
            continue
        text = str(row.get("excerpt") or row.get("text") or "")
        amounts.update(_amount_value(match) for match in _AMOUNT.finditer(text))
    return amounts


def calibrate_commitment_amounts(
    assessment: FinalAssessment,
    *,
    evidence_registry: Mapping[str, Mapping[str, Any]],
) -> tuple[FinalAssessment, tuple[str, ...]]:
    """Remove exact checks unless pitch facts and investor policy both support them."""
    text = assessment.decision_justification
    matches = tuple(_AMOUNT.finditer(text))
    if assessment.decision != "In" or not matches:
        return assessment, ()
    pitch_amounts = _evidence_amounts(evidence_registry, "pitch")
    policy_amounts = _evidence_amounts(evidence_registry, "wiki")
    unsupported = {
        _amount_value(match)
        for match in matches
        if _amount_value(match) not in pitch_amounts
        or _amount_value(match) not in policy_amounts
    }
    if not unsupported:
        return assessment, ()
    calibrated = _AMOUNT.sub(
        lambda match: (
            "a small first check"
            if _amount_value(match) in unsupported
            else match.group(0)
        ),
        text,
    )
    return (
        assessment.model_copy(update={"decision_justification": calibrated}),
        ("unsupported_exact_check_amount_removed",),
    )
