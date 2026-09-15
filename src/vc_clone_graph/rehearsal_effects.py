"""Deterministic safeguards for evidence-responsive rehearsal updates."""

from dataclasses import dataclass
from typing import Literal


EvidenceEffect = Literal[
    "new_positive",
    "new_negative",
    "clarification",
    "unresolved",
    "contradiction",
]

_MATERIAL_EFFECTS = {"new_positive", "new_negative", "contradiction"}


@dataclass(frozen=True)
class EvidenceEffectDelta:
    before: float
    after: float
    findings: tuple[str, ...]


def _probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def enforce_evidence_effect_delta(
    *,
    effect: EvidenceEffect,
    before: float,
    proposed_after: float,
    supporting_excerpt: str | None,
) -> EvidenceEffectDelta:
    """Allow movement only for a supported, materially new evidence effect."""
    baseline = _probability(before)
    proposed = _probability(proposed_after)
    findings: list[str] = []
    if proposed != float(proposed_after):
        findings.append("proposed_likelihood_clamped")
    if effect in {"clarification", "unresolved"}:
        if proposed != baseline:
            findings = ["non_material_effect_preserved_assessment"]
        return EvidenceEffectDelta(baseline, baseline, tuple(findings))
    if effect in _MATERIAL_EFFECTS and not (supporting_excerpt or "").strip():
        return EvidenceEffectDelta(
            baseline,
            baseline,
            ("material_effect_missing_answer_excerpt",),
        )
    return EvidenceEffectDelta(baseline, proposed, tuple(findings))
