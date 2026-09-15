"""Feature parity between interactive rehearsal and offline calibration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .phase2_calibration_evaluation import extract_taxonomy_features
from .rehearsal_schemas import RationaleState


def _rationale_payload(rationale: RationaleState) -> dict[str, Any]:
    pitch_ids: list[str] = []
    wiki_ids: list[str] = []
    historical_ids: list[str] = []
    for evidence in rationale.evidence_refs:
        if evidence.source_kind in {"pitch", "founder_answer"}:
            pitch_ids.append(evidence.evidence_id)
        elif evidence.source_kind in {"wiki", "portfolio"}:
            wiki_ids.append(evidence.evidence_id)
        elif evidence.source_kind == "precedent":
            historical_ids.append(evidence.evidence_id)
    return {
        "rationale_id": rationale.rationale_id,
        "taxonomy_label": rationale.taxonomy_label,
        "direction": rationale.direction,
        "salience": rationale.salience,
        "confidence": rationale.confidence,
        "pitch_evidence_ids": pitch_ids,
        "wiki_evidence_ids": wiki_ids,
        "historical_evidence_ids": historical_ids,
    }


def rehearsal_feature_map(
    *,
    rationales: Sequence[RationaleState | Mapping[str, Any]],
    investment_likelihood: float,
    decision_confidence: float,
    direct_decision: Literal["In", "Out"],
    constraint_assessments: Sequence[Mapping[str, Any]] = (),
    controlling_rationale_ids: Sequence[str] = (),
) -> dict[str, float]:
    """Encode validated rehearsal state using canonical calibration names."""
    models = tuple(
        rationale
        if isinstance(rationale, RationaleState)
        else RationaleState.model_validate(rationale)
        for rationale in rationales
    )
    semantic = extract_taxonomy_features(
        {
            "rationales": [_rationale_payload(rationale) for rationale in models],
            "constraint_assessments": list(constraint_assessments),
        },
        {"controlling_rationale_ids": list(controlling_rationale_ids)},
        vc_slug="runtime",
    )
    features = {
        name: float(value)
        for name, value in semantic.items()
        if name.startswith(("rationale__", "constraint__"))
    }
    decision_in = float(direct_decision == "In")
    features.update(
        {
            "decision_in": decision_in,
            "investment_likelihood": float(investment_likelihood),
            "decision_confidence": float(decision_confidence),
            "signed_decision_confidence": float(decision_confidence)
            * (1.0 if decision_in else -1.0),
            "controlling_rationale_count": float(len(controlling_rationale_ids)),
        }
    )
    return features


def counterfactual_feature_maps(
    base_features: Mapping[str, float],
    rationale_label: str,
    *,
    confidence: float = 0.75,
    salience: Literal["primary", "secondary"] = "primary",
) -> tuple[dict[str, float], dict[str, float]]:
    """Create bounded positive and negative states for one rationale."""
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between zero and one")
    prefix = f"rationale__{rationale_label}__"
    cleaned = {
        name: float(value)
        for name, value in base_features.items()
        if not name.startswith(prefix)
    }
    salience_weight = 1.0 if salience == "primary" else 0.5

    def candidate(direction: Literal["positive", "negative"]) -> dict[str, float]:
        sign = 1.0 if direction == "positive" else -1.0
        values = dict(cleaned)
        values.update(
            {
                f"{prefix}count": 1.0,
                f"{prefix}signed_confidence": sign * confidence,
                f"{prefix}salience_confidence": sign
                * confidence
                * salience_weight,
                f"{prefix}direction__{direction}": 1.0,
                f"{prefix}salience__{salience}": 1.0,
                f"{prefix}pitch_evidence_count": 1.0,
                f"{prefix}wiki_evidence_count": 0.0,
                f"{prefix}historical_evidence_count": 0.0,
            }
        )
        return values

    return candidate("positive"), candidate("negative")
