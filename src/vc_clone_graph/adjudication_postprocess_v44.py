"""Deterministic local post-processing for Phase 1 v4.4 adjudications."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

if TYPE_CHECKING:
    from .phase1_v44 import (
        RationaleAdjudicationV44,
        RationaleDispositionV44,
        TaxonomyNeighborhoodManifestV44,
    )


@dataclass(frozen=True)
class AdjudicationNormalizationAuditV44:
    """Mechanical edits applied after a provider response is recorded."""

    cleared_nonactivating_dispositions: int
    exact_duplicate_dispositions_removed: int
    removed_disposition_positions: tuple[int, ...]


@dataclass(frozen=True)
class ConstraintMappingRevisionV44:
    """One provider mapping and its deterministic source-bound replacement."""

    constraint_id: str
    original_mapped_ids: tuple[str, ...]
    recomputed_mapped_ids: tuple[str, ...]


@dataclass(frozen=True)
class ConstraintRemappingAuditV44:
    """Complete deterministic constraint-remapping audit."""

    constraint_mappings: tuple[ConstraintMappingRevisionV44, ...]
    unresolved_constraint_ids: tuple[str, ...]


@dataclass(frozen=True)
class CrossTargetEvidenceReuseV44:
    """Eligible evidence used outside the target bundle that discovered it."""

    disposition_position: int
    taxonomy_label: str
    target_ids: tuple[str, ...]
    evidence_kind: Literal["wiki", "historical", "portfolio"]
    evidence_id: str


def canonical_disposition_key_v44(row: Mapping[str, object]) -> str:
    """Return the stable full-payload identity for one normalized disposition."""
    return json.dumps(
        row,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def normalize_adjudication_payload_v44(
    value: object,
) -> tuple[object, AdjudicationNormalizationAuditV44]:
    """Copy, null nonactivating fields, and remove only exact duplicate rows."""
    empty_audit = AdjudicationNormalizationAuditV44(0, 0, ())
    if type(value) is not dict or type(value.get("dispositions")) is not list:
        return value, empty_audit

    normalized = deepcopy(value)
    retained: list[object] = []
    seen: set[str] = set()
    cleared = 0
    removed: list[int] = []

    for position, candidate in enumerate(normalized["dispositions"]):
        if type(candidate) is not dict:
            retained.append(candidate)
            continue
        if candidate.get("disposition") in {"question_only", "rejected"}:
            if any(
                candidate.get(field) is not None
                for field in ("direction", "salience", "confidence")
            ):
                cleared += 1
            for field in ("direction", "salience", "confidence"):
                candidate[field] = None
        key = canonical_disposition_key_v44(candidate)
        if key in seen:
            removed.append(position)
            continue
        seen.add(key)
        retained.append(candidate)

    normalized["dispositions"] = retained
    return normalized, AdjudicationNormalizationAuditV44(
        cleared_nonactivating_dispositions=cleared,
        exact_duplicate_dispositions_removed=len(removed),
        removed_disposition_positions=tuple(removed),
    )


def ordered_activated_dispositions_v44(
    dispositions: Sequence[RationaleDispositionV44],
    neighborhood: TaxonomyNeighborhoodManifestV44 | Any,
) -> tuple[RationaleDispositionV44, ...]:
    """Return activated claim-level instances in the canonical local order."""
    order = {
        label: index for index, label in enumerate(neighborhood.ordered_labels)
    }
    activated = [
        row for row in dispositions if row.disposition in {"core", "candidate"}
    ]
    return tuple(
        sorted(
            activated,
            key=lambda row: (
                order.get(row.taxonomy_label, len(order)),
                row.taxonomy_label,
                canonical_disposition_key_v44(
                    row.model_dump(mode="json", warnings=False)
                ),
            ),
        )
    )


def rationale_id_bindings_v44(
    dispositions: Sequence[RationaleDispositionV44],
    neighborhood: TaxonomyNeighborhoodManifestV44 | Any,
) -> tuple[tuple[str, RationaleDispositionV44], ...]:
    """Bind deterministic local R IDs to activated rationale instances."""
    return tuple(
        (f"R{index}", row)
        for index, row in enumerate(
            ordered_activated_dispositions_v44(dispositions, neighborhood),
            start=1,
        )
    )


def _mapped_id_order(identifier: str) -> tuple[int, int]:
    prefix_order = {"R": 0, "U": 1, "Q": 2}[identifier[0]]
    return prefix_order, int(identifier[1:])


def remap_constraint_assessments_v44(
    adjudication: RationaleAdjudicationV44 | Any,
    neighborhood: TaxonomyNeighborhoodManifestV44 | Any,
) -> tuple[RationaleAdjudicationV44, ConstraintRemappingAuditV44]:
    """Rebind constraints to every R/U object sharing their pitch evidence."""
    bindings: dict[str, frozenset[str]] = {
        rationale_id: frozenset(row.pitch_evidence_ids)
        for rationale_id, row in rationale_id_bindings_v44(
            adjudication.dispositions, neighborhood
        )
    }
    bindings.update(
        {
            row.observation_id: frozenset(row.pitch_evidence_ids)
            for row in adjudication.unmapped_observations
        }
    )
    question_bindings: dict[str, set[str]] = {}
    for row in adjudication.dispositions:
        if row.disposition != "question_only":
            continue
        for question_id in row.question_ids:
            question_bindings.setdefault(question_id, set()).update(
                row.pitch_evidence_ids
            )
    bindings.update(
        {
            question_id: frozenset(pitch_evidence_ids)
            for question_id, pitch_evidence_ids in question_bindings.items()
        }
    )

    payload = adjudication.model_dump(mode="json", warnings=False)
    revisions: list[ConstraintMappingRevisionV44] = []
    unresolved: list[str] = []
    for position, constraint in enumerate(adjudication.constraint_assessments):
        pitch_ids = frozenset(constraint.pitch_evidence_ids)
        recomputed = tuple(
            sorted(
                (
                    identifier
                    for identifier, source_ids in bindings.items()
                    if pitch_ids & source_ids
                ),
                key=_mapped_id_order,
            )
        )
        revisions.append(
            ConstraintMappingRevisionV44(
                constraint_id=constraint.constraint_id,
                original_mapped_ids=tuple(constraint.mapped_ids),
                recomputed_mapped_ids=recomputed,
            )
        )
        if recomputed:
            payload["constraint_assessments"][position]["mapped_ids"] = list(
                recomputed
            )
        else:
            unresolved.append(constraint.constraint_id)

    remapped = type(adjudication).model_validate_json(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    )
    return remapped, ConstraintRemappingAuditV44(
        constraint_mappings=tuple(revisions),
        unresolved_constraint_ids=tuple(unresolved),
    )


def cross_target_evidence_reuse_v44(
    adjudication: RationaleAdjudicationV44 | Any,
    retrieval: Any,
) -> tuple[CrossTargetEvidenceReuseV44, ...]:
    """Report globally eligible evidence cited outside its retrieval targets."""
    attributes = {
        "wiki": "wiki_evidence",
        "historical": "historical_evidence",
        "portfolio": "portfolio_disclosures",
    }
    citation_attributes = {
        "wiki": "wiki_evidence_ids",
        "historical": "historical_evidence_ids",
        "portfolio": "portfolio_disclosure_ids",
    }
    global_ids: dict[str, set[str]] = {kind: set() for kind in attributes}
    target_ids: dict[str, dict[str, set[str]]] = {}
    for bundle in retrieval.claim_bundles:
        by_kind = target_ids.setdefault(
            bundle.target_id, {kind: set() for kind in attributes}
        )
        for kind, attribute in attributes.items():
            for record in getattr(bundle, attribute):
                if record.eligible:
                    global_ids[kind].add(record.evidence_id)
                    by_kind[kind].add(record.evidence_id)

    rows: list[CrossTargetEvidenceReuseV44] = []
    for position, disposition in enumerate(adjudication.dispositions):
        bound_targets = tuple(
            sorted({*disposition.claim_ids, *disposition.question_ids})
        )
        for kind, attribute in citation_attributes.items():
            retrieved_for_targets = set().union(
                *(
                    target_ids.get(target, {}).get(kind, set())
                    for target in bound_targets
                )
            )
            for evidence_id in sorted(set(getattr(disposition, attribute))):
                if (
                    evidence_id in global_ids[kind]
                    and evidence_id not in retrieved_for_targets
                ):
                    rows.append(
                        CrossTargetEvidenceReuseV44(
                            disposition_position=position,
                            taxonomy_label=disposition.taxonomy_label,
                            target_ids=bound_targets,
                            evidence_kind=kind,
                            evidence_id=evidence_id,
                        )
                    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.disposition_position,
                row.evidence_kind,
                row.evidence_id,
            ),
        )
    )


def finalize_adjudication_v44(
    adjudication: RationaleAdjudicationV44 | Any,
    neighborhood: TaxonomyNeighborhoodManifestV44 | Any,
    retrieval: Any,
) -> tuple[
    RationaleAdjudicationV44,
    tuple[CrossTargetEvidenceReuseV44, ...],
    ConstraintRemappingAuditV44,
]:
    """Apply all model-level deterministic finalization after JSON validation."""
    remapped, mapping_audit = remap_constraint_assessments_v44(
        adjudication, neighborhood
    )
    reuse_audit = cross_target_evidence_reuse_v44(remapped, retrieval)
    return remapped, reuse_audit, mapping_audit
