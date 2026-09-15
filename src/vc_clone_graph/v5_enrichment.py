"""Deterministic transformation from canonical Phase 1 to investigation-v5."""

from __future__ import annotations

from collections.abc import Mapping, Set
from copy import deepcopy

from .rationale_association_annotations import build_rationale_association_annotations
from .rationale_completion import CompletionPrediction
from .schemas_v5 import InvestigationV5, validate_v5_investigation


def enrich_investigation_v5(
    source: Mapping[str, object],
    source_sha256: str,
    prediction: CompletionPrediction,
    *,
    taxonomy_labels: Set[str],
    min_posterior_probability: float = 0.60,
    min_support: int = 3,
    min_lift: float = 1.0,
    max_per_source: int = 5,
) -> InvestigationV5:
    annotations = build_rationale_association_annotations(
        source,
        source_sha256,
        prediction,
        taxonomy_labels=taxonomy_labels,
        min_posterior_probability=min_posterior_probability,
        min_support=min_support,
        min_lift=min_lift,
        max_per_source=max_per_source,
    )
    payload = deepcopy(dict(source))
    source_version = str(payload.pop("schema_version"))
    claim_annotations = {
        row.rationale_id: row.associated_rationales
        for row in annotations.claim_annotations
    }
    for row in payload.get("rationales", []):
        row["associated_rationales"] = [
            item.model_dump(mode="json")
            for item in claim_annotations[str(row["rationale_id"])]
        ]
    statement_annotations = {
        row.pitch_evidence_id: row.associated_rationales
        for row in annotations.material_statement_annotations
    }
    for row in payload.get("material_statement_coverage", []):
        row["associated_rationales"] = [
            item.model_dump(mode="json")
            for item in statement_annotations[str(row["pitch_evidence_id"])]
        ]
    for row in payload.get("unmapped_observations", []):
        row["association_status"] = "unavailable_unmapped"
        row["associated_rationales"] = []
    payload.update(
        schema_version="investigation-v5",
        source_schema_version=source_version,
        source_investigation_sha256=source_sha256,
        association_training_episode_slugs=list(annotations.training_episode_slugs),
        association_thresholds=annotations.thresholds.model_dump(mode="json"),
        episode_level_associations=[
            row.model_dump(mode="json")
            for row in annotations.episode_level_associations
        ],
    )
    return validate_v5_investigation(payload, taxonomy_labels)
