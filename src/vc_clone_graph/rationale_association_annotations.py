"""Fold-safe advisory annotations derived from Phase 1 rationale associations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from .rationale_completion import AssociationRuleEstimate, CompletionPrediction
from .schemas import StrictModel


class AssociationAnnotationThresholds(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_posterior_probability: float = Field(default=0.60, ge=0.0, le=1.0)
    min_support: int = Field(default=3, ge=1)
    min_lift: float = Field(default=1.0, ge=0.0)
    max_per_source: int = Field(default=5, ge=1)


class AssociatedRationale(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    taxonomy_label: str = Field(min_length=1)
    posterior_probability: float = Field(ge=0.0, le=1.0)
    support: int = Field(ge=1)
    lift: float = Field(gt=0.0)
    antecedent_labels: tuple[str, ...] = Field(min_length=1, max_length=2)
    status: Literal["hypothesis_only"] = "hypothesis_only"


class ClaimAssociationAnnotation(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rationale_id: str
    source_taxonomy_label: str
    associated_rationales: tuple[AssociatedRationale, ...] = ()


class MaterialStatementAssociationAnnotation(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pitch_evidence_id: str
    mapped_ids: tuple[str, ...]
    source_taxonomy_labels: tuple[str, ...]
    associated_rationales: tuple[AssociatedRationale, ...] = ()


class UnmappedObservationAssociationAnnotation(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: str
    status: Literal["unavailable_unmapped"] = "unavailable_unmapped"
    associated_rationales: tuple[AssociatedRationale, ...] = ()


class RationaleAssociationAnnotations(StrictModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, serialize_by_alias=True
    )

    schema_name: Literal["rationale-association-annotations-v1"] = Field(
        default="rationale-association-annotations-v1", alias="schema"
    )
    scientific_status: Literal["fold_safe_advisory_hypotheses"] = (
        "fold_safe_advisory_hypotheses"
    )
    vc_slug: str
    episode_slug: str
    investigation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    thresholds: AssociationAnnotationThresholds
    training_episode_slugs: tuple[str, ...]
    claim_annotations: tuple[ClaimAssociationAnnotation, ...]
    material_statement_annotations: tuple[MaterialStatementAssociationAnnotation, ...]
    unmapped_observation_annotations: tuple[
        UnmappedObservationAssociationAnnotation, ...
    ]
    episode_level_associations: tuple[AssociatedRationale, ...]


@dataclass(frozen=True)
class AssociationAnnotationEvaluationCase:
    annotations: RationaleAssociationAnnotations
    observed_labels: tuple[str, ...]
    reference_labels: tuple[str, ...]


def _eligible(
    rule: AssociationRuleEstimate, thresholds: AssociationAnnotationThresholds
) -> bool:
    return (
        rule.posterior_probability >= thresholds.min_posterior_probability
        and rule.support >= thresholds.min_support
        and rule.lift > thresholds.min_lift
    )


def _associations_for(
    prediction: CompletionPrediction,
    available_labels: Set[str],
    active_labels: Set[str],
    thresholds: AssociationAnnotationThresholds,
    *,
    required_antecedent_labels: Set[str] | None = None,
) -> tuple[AssociatedRationale, ...]:
    best_by_target: dict[str, AssociationRuleEstimate] = {}
    for hypothesis in prediction.candidates:
        if hypothesis.label in active_labels:
            continue
        candidates = [
            rule
            for rule in hypothesis.association_rules
            if set(rule.antecedent).issubset(available_labels)
            and _eligible(rule, thresholds)
            and (
                required_antecedent_labels is None
                or bool(set(rule.antecedent) & required_antecedent_labels)
            )
        ]
        if not candidates:
            continue
        best_by_target[hypothesis.label] = min(
            candidates,
            key=lambda row: (
                -row.posterior_probability,
                -row.support,
                -row.lift,
                row.antecedent,
            ),
        )
    rows = [
        AssociatedRationale(
            taxonomy_label=label,
            posterior_probability=rule.posterior_probability,
            support=rule.support,
            lift=rule.lift,
            antecedent_labels=rule.antecedent,
        )
        for label, rule in best_by_target.items()
    ]
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                -row.posterior_probability,
                -row.support,
                -row.lift,
                row.taxonomy_label,
                row.antecedent_labels,
            ),
        )[: thresholds.max_per_source]
    )


def _object_rows(value: object, context: str) -> list[Mapping[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    if not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{context} must contain objects")
    return value


def build_rationale_association_annotations(
    investigation: Mapping[str, object],
    investigation_sha256: str,
    prediction: CompletionPrediction,
    *,
    taxonomy_labels: Set[str],
    min_posterior_probability: float = 0.60,
    min_support: int = 3,
    min_lift: float = 1.0,
    max_per_source: int = 5,
) -> RationaleAssociationAnnotations:
    """Annotate a canonical investigation with non-controlling LOEO hypotheses."""
    if len(investigation_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in investigation_sha256
    ):
        raise ValueError("investigation_sha256 must be 64 lowercase hexadecimal characters")
    if max_per_source < 1:
        raise ValueError("max_per_source must be positive")
    episode_slug = investigation.get("episode_slug")
    if episode_slug != prediction.episode_slug:
        raise ValueError(
            f"episode mismatch: investigation={episode_slug!r} prediction={prediction.episode_slug!r}"
        )
    if prediction.episode_slug in prediction.training_slugs:
        raise ValueError("held episode appears in association training provenance")

    model_labels = {
        *prediction.observed_labels,
        *(candidate.label for candidate in prediction.candidates),
    }
    unknown = sorted(model_labels - set(taxonomy_labels))
    if unknown:
        raise ValueError(f"association labels outside taxonomy: {unknown}")

    thresholds = AssociationAnnotationThresholds(
        min_posterior_probability=min_posterior_probability,
        min_support=min_support,
        min_lift=min_lift,
        max_per_source=max_per_source,
    )
    rationale_rows = _object_rows(investigation.get("rationales"), "rationales")
    rationale_labels: dict[str, str] = {}
    for row in rationale_rows:
        rationale_id = row.get("rationale_id")
        label = row.get("taxonomy_label")
        if not isinstance(rationale_id, str) or not isinstance(label, str):
            raise ValueError("rationale rows require rationale_id and taxonomy_label")
        if label not in taxonomy_labels:
            raise ValueError(f"rationale label outside taxonomy: {label}")
        if rationale_id in rationale_labels:
            raise ValueError(f"duplicate rationale ID: {rationale_id}")
        rationale_labels[rationale_id] = label
    active_labels = set(rationale_labels.values())

    claims = tuple(
        ClaimAssociationAnnotation(
            rationale_id=rationale_id,
            source_taxonomy_label=label,
            associated_rationales=_associations_for(
                prediction,
                active_labels,
                active_labels,
                thresholds,
                required_antecedent_labels={label},
            ),
        )
        for rationale_id, label in rationale_labels.items()
    )

    statement_annotations: list[MaterialStatementAssociationAnnotation] = []
    for row in _object_rows(
        investigation.get("material_statement_coverage"),
        "material_statement_coverage",
    ):
        evidence_id = row.get("pitch_evidence_id")
        mapped = row.get("mapped_ids", [])
        if not isinstance(evidence_id, str) or not isinstance(mapped, list) or not all(
            isinstance(item, str) for item in mapped
        ):
            raise ValueError("material statement requires pitch_evidence_id and mapped_ids")
        mapped_ids = tuple(mapped)
        source_labels = tuple(
            sorted({rationale_labels[item] for item in mapped_ids if item in rationale_labels})
        )
        if row.get("mapping_type") == "taxonomy_rationale" and len(source_labels) != len(
            set(mapped_ids)
        ):
            unresolved = sorted(set(mapped_ids) - set(rationale_labels))
            raise ValueError(f"unresolved rationale IDs in material statement: {unresolved}")
        statement_annotations.append(
            MaterialStatementAssociationAnnotation(
                pitch_evidence_id=evidence_id,
                mapped_ids=mapped_ids,
                source_taxonomy_labels=source_labels,
                associated_rationales=(
                    _associations_for(
                        prediction,
                        active_labels,
                        active_labels,
                        thresholds,
                        required_antecedent_labels=set(source_labels),
                    )
                    if source_labels
                    else ()
                ),
            )
        )

    observations = tuple(
        UnmappedObservationAssociationAnnotation(
            observation_id=str(row.get("observation_id"))
        )
        for row in _object_rows(
            investigation.get("unmapped_observations"), "unmapped_observations"
        )
    )
    return RationaleAssociationAnnotations(
        vc_slug=prediction.vc_slug,
        episode_slug=prediction.episode_slug,
        investigation_sha256=investigation_sha256,
        thresholds=thresholds,
        training_episode_slugs=prediction.training_slugs,
        claim_annotations=claims,
        material_statement_annotations=tuple(statement_annotations),
        unmapped_observation_annotations=observations,
        episode_level_associations=_associations_for(
            prediction, active_labels, active_labels, thresholds
        ),
    )


def _ratio(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
    return numerator / denominator if denominator else empty


def write_rationale_association_analysis(
    cases: Sequence[AssociationAnnotationEvaluationCase],
    output: Path,
    *,
    input_paths: Mapping[str, Path],
) -> dict[str, Path]:
    """Write immutable sidecars and development metrics for advisory annotations."""
    if not cases:
        raise ValueError("association annotation analysis requires cases")
    keys = [
        (case.annotations.vc_slug, case.annotations.episode_slug) for case in cases
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate association annotation evaluation case")

    output.mkdir(parents=True, exist_ok=True)
    record_paths: list[Path] = []
    rows: list[dict[str, object]] = []
    missing_total = proposals_total = hits_total = 0
    missing_cases = recovered_cases = 0
    claim_total = claim_with = statement_total = statement_with = 0
    for case in sorted(
        cases, key=lambda row: (row.annotations.vc_slug, row.annotations.episode_slug)
    ):
        annotation = case.annotations
        if annotation.episode_slug in annotation.training_episode_slugs:
            raise ValueError(
                f"held episode appears in training provenance: {annotation.episode_slug}"
            )
        record_path = (
            output
            / "records"
            / annotation.vc_slug
            / f"{annotation.episode_slug}.json"
        )
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            annotation.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        record_paths.append(record_path)

        missing = set(case.reference_labels) - set(case.observed_labels)
        proposed = {
            row.taxonomy_label for row in annotation.episode_level_associations
        }
        hits = missing & proposed
        missing_total += len(missing)
        proposals_total += len(proposed)
        hits_total += len(hits)
        if missing:
            missing_cases += 1
            recovered_cases += int(bool(hits))
        claim_count = len(annotation.claim_annotations)
        claim_associated = sum(
            bool(row.associated_rationales) for row in annotation.claim_annotations
        )
        statement_count = len(annotation.material_statement_annotations)
        statement_associated = sum(
            bool(row.associated_rationales)
            for row in annotation.material_statement_annotations
        )
        claim_total += claim_count
        claim_with += claim_associated
        statement_total += statement_count
        statement_with += statement_associated
        rows.append(
            {
                "vc_slug": annotation.vc_slug,
                "episode_slug": annotation.episode_slug,
                "training_count": len(annotation.training_episode_slugs),
                "observed_label_count": len(set(case.observed_labels)),
                "missing_reference_count": len(missing),
                "episode_hypothesis_count": len(proposed),
                "missing_reference_hits": len(hits),
                "claim_count": claim_count,
                "claims_with_associations": claim_associated,
                "material_statement_count": statement_count,
                "material_statements_with_associations": statement_associated,
            }
        )

    metrics: dict[str, object] = {
        "schema": "rationale-association-annotation-analysis-v1",
        "scientific_status": (
            "nested_development_automated_transcript_derived_references"
        ),
        "api_cost_usd": 0.0,
        "case_count": len(cases),
        "missing_reference_count": missing_total,
        "episode_hypothesis_count": proposals_total,
        "missing_reference_hits": hits_total,
        "missing_reference_recall": _ratio(hits_total, missing_total, empty=1.0),
        "hypothesis_precision": _ratio(hits_total, proposals_total, empty=1.0),
        "missing_reference_case_count": missing_cases,
        "recovered_missing_reference_cases": recovered_cases,
        "missing_case_recovery_rate": _ratio(
            recovered_cases, missing_cases, empty=1.0
        ),
        "claim_count": claim_total,
        "claims_with_associations": claim_with,
        "claim_association_coverage": _ratio(claim_with, claim_total, empty=1.0),
        "material_statement_count": statement_total,
        "material_statements_with_associations": statement_with,
        "material_statement_association_coverage": _ratio(
            statement_with, statement_total, empty=1.0
        ),
    }
    paths = {
        "annotations": output / "annotations.csv",
        "metrics": output / "metrics.json",
        "manifest": output / "manifest.json",
        "evaluation": output / "evaluation.md",
    }
    with paths["annotations"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths["metrics"].write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        **metrics,
        "record_count": len(record_paths),
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sorted(input_paths.items())
        },
        "records": {
            str(path.relative_to(output)): sha256(path.read_bytes()).hexdigest()
            for path in record_paths
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["evaluation"].write_text(
        "# Phase 1 Associated-Rationale Advisory Layer\n\n"
        "This is a fold-safe development evaluation against automated transcript-derived "
        "rationale references, not a human-validated held-out result. Associations are "
        "hypotheses only and do not alter canonical Phase 1.\n\n"
        "## Results\n\n"
        f"- Cases: {len(cases)}\n"
        f"- Missing-reference recall: {float(metrics['missing_reference_recall']):.3f} "
        f"({hits_total}/{missing_total})\n"
        f"- Hypothesis precision: {float(metrics['hypothesis_precision']):.3f} "
        f"({hits_total}/{proposals_total})\n"
        f"- Missing-case recovery: {float(metrics['missing_case_recovery_rate']):.3f} "
        f"({recovered_cases}/{missing_cases})\n"
        f"- Claim association coverage: {float(metrics['claim_association_coverage']):.3f} "
        f"({claim_with}/{claim_total})\n"
        f"- Material-statement association coverage: "
        f"{float(metrics['material_statement_association_coverage']):.3f} "
        f"({statement_with}/{statement_total})\n"
        "- API cost: $0.00\n",
        encoding="utf-8",
    )
    return paths
