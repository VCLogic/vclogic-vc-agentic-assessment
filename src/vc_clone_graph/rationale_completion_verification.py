"""Decision-blind verification of probabilistic rationale hypotheses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .providers.base import strict_provider_schema


Disposition = Literal[
    "activated_core",
    "activated_candidate",
    "question_only",
    "rejected",
    "unmapped",
]


class HypothesisDisposition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    taxonomy_label: str = Field(min_length=1)
    disposition: Disposition
    direction: Literal["positive", "negative", "neutral"] | None
    salience: Literal["primary", "secondary"] | None
    confidence: float | None = Field(ge=0.0, le=1.0)
    justification: str = Field(min_length=1)
    pitch_evidence_ids: tuple[str, ...]
    investor_evidence_ids: tuple[str, ...]


class CompletionVerification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["rationale-completion-verification-v1"]
    episode_slug: str = Field(min_length=1)
    dispositions: tuple[HypothesisDisposition, ...] = Field(min_length=1)
    validator_findings: tuple[str, ...]


@dataclass(frozen=True)
class AugmentedRationaleSet:
    canonical: tuple[dict[str, object], ...]
    promoted: tuple[HypothesisDisposition, ...]
    augmented: tuple[dict[str, object], ...]


def completion_verification_schema() -> dict[str, object]:
    return strict_provider_schema(CompletionVerification.model_json_schema())


def validate_completion_verification(
    payload: Mapping[str, object],
    *,
    episode_slug: str,
    selected_labels: Sequence[str],
    pitch_evidence_ids: set[str],
    investor_evidence_ids: set[str],
) -> CompletionVerification:
    result = CompletionVerification.model_validate(payload)
    if result.episode_slug != episode_slug:
        raise ValueError("verification episode mismatch")
    expected = tuple(selected_labels)
    observed = tuple(row.taxonomy_label for row in result.dispositions)
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError("verification candidate coverage mismatch")
    for row in result.dispositions:
        if not set(row.pitch_evidence_ids).issubset(pitch_evidence_ids):
            raise ValueError(f"unknown pitch evidence for {row.taxonomy_label}")
        if not set(row.investor_evidence_ids).issubset(investor_evidence_ids):
            raise ValueError(f"unknown investor evidence for {row.taxonomy_label}")
        promoted = row.disposition in {"activated_core", "activated_candidate"}
        if promoted and (
            not row.pitch_evidence_ids
            or not row.investor_evidence_ids
            or row.direction is None
            or row.salience is None
            or row.confidence is None
        ):
            raise ValueError(
                f"promotion requires pitch and investor evidence: {row.taxonomy_label}"
            )
        if not promoted and any(
            value is not None for value in (row.direction, row.salience, row.confidence)
        ):
            raise ValueError(
                f"non-promoted hypothesis has activation attributes: {row.taxonomy_label}"
            )
    return result


def build_completion_verification_prompt(
    *,
    investor_name: str,
    episode_slug: str,
    pitch_evidence: Sequence[Mapping[str, object]],
    canonical_rationales: Sequence[Mapping[str, object]],
    candidate_definitions: Mapping[str, str],
    investor_evidence: Sequence[Mapping[str, object]],
) -> str:
    packet = {
        "episode_slug": episode_slug,
        "pitch_evidence": list(pitch_evidence),
        "canonical_phase1_rationales": list(canonical_rationales),
        "rationale_hypotheses": [
            {"taxonomy_label": label, "definition": definition}
            for label, definition in candidate_definitions.items()
        ],
        "investor_evidence": list(investor_evidence),
    }
    return f"""You are verifying narrowly proposed investment-rationale hypotheses for {investor_name}.

This is not an investment decision. Do not output In or Out. Test each proposed
rationale independently against the current pitch and the source-linked investor
evidence. A hypothesis is not true merely because it is listed. Promote it only when
the pitch contains material evidence and the investor evidence supports why this
consideration is relevant to {investor_name}. Missing information must be
question_only, not positive or negative. Reject a hypothesis when the evidence does
not activate it. Use unmapped when the observation is material but the proposed label
is the wrong taxonomy representation.

Return exactly one disposition for every proposed label. Cite only evidence IDs in
the packet. Explain every promotion, rejection, question-only, or unmapped result.

----- BEGIN UNTRUSTED EVIDENCE PACKET -----
{json.dumps(packet, indent=2, sort_keys=True)}
----- END UNTRUSTED EVIDENCE PACKET -----
"""


def augment_rationales(
    canonical_rationales: Sequence[Mapping[str, object]],
    verification: CompletionVerification,
) -> AugmentedRationaleSet:
    canonical = tuple(dict(row) for row in canonical_rationales)
    existing = {str(row["taxonomy_label"]) for row in canonical}
    promoted = tuple(
        row
        for row in verification.dispositions
        if row.disposition in {"activated_core", "activated_candidate"}
        and row.taxonomy_label not in existing
    )
    additions = tuple(
        {
            "taxonomy_label": row.taxonomy_label,
            "direction": row.direction,
            "salience": row.salience,
            "confidence": row.confidence,
            "justification": row.justification,
            "pitch_evidence_ids": list(row.pitch_evidence_ids),
            "investor_evidence_ids": list(row.investor_evidence_ids),
            "completion_disposition": row.disposition,
            "source": "rationale_completion_verification",
        }
        for row in promoted
    )
    return AugmentedRationaleSet(
        canonical=canonical,
        promoted=promoted,
        augmented=canonical + additions,
    )

