"""Deployable feature assembly for per-VC structured fusion models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .phase2_calibration_evaluation import Phase2CalibrationRecord
from .schemas_v5 import InvestigationV5
from .v5_rationale_classifier import association_feature_map


FEATURE_CONDITIONS = (
    "phase1",
    "phase2",
    "phase1_assoc",
    "phase1_phase2",
    "full",
)


@dataclass(frozen=True)
class StructuredFusionCase:
    vc_slug: str
    vc_name: str
    episode_slug: str
    group: str
    target: int
    phase1_features: Mapping[str, float]
    association_features: Mapping[str, float]
    phase2_features: Mapping[str, float]
    interaction_features: Mapping[str, float]
    association_training_episode_slugs: tuple[str, ...]
    phase1_sha256: str
    phase2_sha256: str
    source_investigation_sha256: str


def _safe(features: Mapping[str, float], prefix: str) -> dict[str, float]:
    result = {}
    for name, value in features.items():
        lowered = name.lower()
        if (
            "target" in lowered
            or "actual" in lowered
            or name.startswith("vc__")
            or name.startswith("vc_rationale__")
            or name.startswith("vc_constraint__")
        ):
            continue
        result[f"{prefix}{name}"] = float(value)
    return result


def _phase1_features(
    record: Phase2CalibrationRecord, investigation: InvestigationV5
) -> dict[str, float]:
    allowed = {
        name: value
        for name, value in record.semantic_features.items()
        if name.startswith(("rationale__", "constraint__"))
        or name == "conflict_count"
    }
    result = _safe(allowed, "p1__")
    directions = ("positive", "negative", "neutral")
    saliences = ("primary", "secondary")
    for direction in directions:
        selected = [row for row in investigation.rationales if row.direction == direction]
        result[f"p1_aggregate__{direction}_count"] = float(len(selected))
        result[f"p1_aggregate__{direction}_confidence_sum"] = sum(
            row.confidence for row in selected
        )
    for salience in saliences:
        result[f"p1_aggregate__{salience}_count"] = float(sum(
            row.salience == salience for row in investigation.rationales
        ))
    result.update({
        "p1_aggregate__rationale_count": float(len(investigation.rationales)),
        "p1_aggregate__conflict_count": float(len(investigation.conflicts)),
        "p1_aggregate__unmapped_count": float(len(investigation.unmapped_observations)),
        "p1_aggregate__material_statement_count": float(
            len(investigation.material_statement_coverage)
        ),
        "p1_aggregate__constraint_count": float(
            len(investigation.constraint_assessments)
        ),
        "p1_aggregate__portfolio_overlap_count": float(
            len(investigation.portfolio_overlap_assessments)
        ),
        "p1_aggregate__information_sufficient": float(
            investigation.information_sufficient
        ),
        "p1_aggregate__searchable_question_count": float(
            len(investigation.searchable_questions)
        ),
        "p1_aggregate__diligence_question_count": float(
            len(investigation.diligence_questions)
        ),
    })
    return result


def _interactions(
    phase1: Mapping[str, float], phase2: Mapping[str, float]
) -> dict[str, float]:
    positive = float(phase1.get("p1_aggregate__positive_confidence_sum", 0.0))
    negative = float(phase1.get("p1_aggregate__negative_confidence_sum", 0.0))
    signed = positive - negative
    likelihood = float(phase2.get("p2__investment_likelihood", 0.0))
    confidence = float(phase2.get("p2__decision_confidence", 0.0))
    decision_in = float(phase2.get("p2__decision_in", 0.0))
    return {
        "interaction__signed_rationale_x_likelihood": signed * likelihood,
        "interaction__signed_rationale_x_signed_decision": signed * (
            1.0 if decision_in else -1.0
        ) * confidence,
        "interaction__negative_confidence_x_review_priority": negative * float(
            phase2.get("p2__review_priority_score", 0.0)
        ),
        "interaction__phase1_phase2_direction_agreement": float(
            (signed >= 0.0) == bool(decision_in)
        ),
    }


def build_structured_fusion_cases(
    records: Sequence[Phase2CalibrationRecord],
    v5_root: Path,
    *,
    families: Mapping[str, str],
) -> list[StructuredFusionCase]:
    result: list[StructuredFusionCase] = []
    for record in records:
        path = (
            Path(v5_root) / "investors" / record.vc_slug / record.episode_slug
            / "phase1/investigation.json"
        )
        investigation = InvestigationV5.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if investigation.episode_slug != record.episode_slug:
            raise ValueError("v5 investigation episode does not match Phase 2 record")
        if record.episode_slug in investigation.association_training_episode_slugs:
            raise ValueError("target episode appears in association training provenance")
        phase1 = _phase1_features(record, investigation)
        association = association_feature_map(investigation, families=families)
        phase2 = _safe(record.phase2_features, "p2__")
        result.append(StructuredFusionCase(
            vc_slug=record.vc_slug,
            vc_name=record.vc_name,
            episode_slug=record.episode_slug,
            group=record.group,
            target=record.target,
            phase1_features=phase1,
            association_features=association,
            phase2_features=phase2,
            interaction_features=_interactions(phase1, phase2),
            association_training_episode_slugs=tuple(
                investigation.association_training_episode_slugs
            ),
            phase1_sha256=record.phase1_sha256,
            phase2_sha256=record.phase2_sha256,
            source_investigation_sha256=investigation.source_investigation_sha256,
        ))
    return result


def feature_view(case: StructuredFusionCase, condition: str) -> dict[str, float]:
    if condition not in FEATURE_CONDITIONS:
        raise ValueError(f"unknown structured fusion feature condition: {condition}")
    if condition == "phase1":
        return dict(case.phase1_features)
    if condition == "phase2":
        return dict(case.phase2_features)
    if condition == "phase1_assoc":
        return {**case.phase1_features, **case.association_features}
    if condition == "phase1_phase2":
        return {
            **case.phase1_features,
            **case.phase2_features,
            **case.interaction_features,
        }
    return {
        **case.phase1_features,
        **case.association_features,
        **case.phase2_features,
        **case.interaction_features,
    }
