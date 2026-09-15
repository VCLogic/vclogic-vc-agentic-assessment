"""Self-contained v5 investigation contract with advisory rationale associations."""

from __future__ import annotations

from typing import Annotated, Literal, Mapping, Sequence

from pydantic import Field, model_validator

from .rationale_association_annotations import (
    AssociatedRationale,
    AssociationAnnotationThresholds,
)
from .schemas import NonEmptyText, SafeSlug, StrictModel
from .schemas_v4 import (
    ConstraintAssessmentV41,
    EvidenceId,
    PitchEvidenceId,
    PortfolioOverlapAssessmentV41,
    QuestionAssessmentV4,
)


class RationaleV5(StrictModel):
    rationale_id: Annotated[str, Field(pattern=r"^R[1-9][0-9]*$")]
    taxonomy_label: NonEmptyText
    direction: Literal["positive", "negative", "neutral"]
    salience: Literal["primary", "secondary"]
    confidence: float = Field(ge=0, le=1)
    pitch_evidence: list[NonEmptyText] | None = None
    pitch_evidence_ids: list[PitchEvidenceId] = Field(min_length=1)
    wiki_evidence_ids: list[EvidenceId]
    historical_evidence_ids: list[EvidenceId]
    justification: NonEmptyText
    associated_rationales: list[AssociatedRationale]


class UnmappedObservationV5(StrictModel):
    observation_id: Annotated[str, Field(pattern=r"^U[1-9][0-9]*$")]
    description: NonEmptyText
    pitch_evidence: list[NonEmptyText] = Field(min_length=1)
    evidence_ids: list[EvidenceId] = Field(min_length=1)
    decision_relevance: NonEmptyText
    association_status: Literal["unavailable_unmapped"]
    associated_rationales: list[AssociatedRationale] = Field(max_length=0)


class MaterialStatementCoverageV5(StrictModel):
    pitch_evidence_id: PitchEvidenceId
    decision_dimension: str
    direction: Literal["positive", "negative", "neutral"]
    constraint_signal: Literal["none", "possible", "triggered"]
    mapping_type: Literal["taxonomy_rationale", "unmapped_observation"]
    mapped_ids: list[str] = Field(min_length=1)
    assessment: NonEmptyText
    associated_rationales: list[AssociatedRationale]


class InvestigationV5(StrictModel):
    schema_version: Literal["investigation-v5"]
    source_schema_version: Literal["investigation-v4", "investigation-v4.1"]
    source_investigation_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    episode_slug: SafeSlug
    association_training_episode_slugs: list[SafeSlug]
    association_thresholds: AssociationAnnotationThresholds
    questions: list[QuestionAssessmentV4]
    rationales: list[RationaleV5] = Field(min_length=1)
    conflicts: list[NonEmptyText]
    unmapped_observations: list[UnmappedObservationV5]
    information_sufficient: bool
    sufficiency_assessment: NonEmptyText
    searchable_questions: list[NonEmptyText]
    diligence_questions: list[NonEmptyText]
    next_search_objectives: list[NonEmptyText]
    summary: NonEmptyText
    reviewed_pitch_evidence_ids: list[PitchEvidenceId] | None = None
    material_statement_coverage: list[MaterialStatementCoverageV5] = []
    constraint_assessments: list[ConstraintAssessmentV41] = []
    portfolio_overlap_assessments: list[PortfolioOverlapAssessmentV41] = []
    episode_level_associations: list[AssociatedRationale]

    @model_validator(mode="after")
    def validate_provenance_and_mappings(self) -> "InvestigationV5":
        if self.episode_slug in self.association_training_episode_slugs:
            raise ValueError("target episode appears in association training provenance")
        identifiers = {row.rationale_id for row in self.rationales} | {
            row.observation_id for row in self.unmapped_observations
        }
        unknown = sorted(
            value
            for row in self.material_statement_coverage
            for value in row.mapped_ids
            if value not in identifiers
        )
        if unknown:
            raise ValueError(f"unknown mapped IDs: {unknown}")
        return self


def validate_v5_investigation(
    payload: Mapping[str, object], taxonomy_labels: Sequence[str] | set[str]
) -> InvestigationV5:
    model = InvestigationV5.model_validate(payload)
    taxonomy = set(taxonomy_labels)
    active = {row.taxonomy_label for row in model.rationales}
    labels = {
        association.taxonomy_label
        for row in model.rationales
        for association in row.associated_rationales
    } | {
        association.taxonomy_label
        for row in model.material_statement_coverage
        for association in row.associated_rationales
    } | {row.taxonomy_label for row in model.episode_level_associations}
    unknown = sorted((active | labels) - taxonomy)
    if unknown:
        raise ValueError(f"v5 rationale labels outside taxonomy: {unknown}")
    duplicated = sorted(active & labels)
    if duplicated:
        raise ValueError(f"associated rationale already active: {duplicated}")
    return model
