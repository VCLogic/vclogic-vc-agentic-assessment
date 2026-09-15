"""Evidence-first Phase 1 v4.4 contracts and deterministic finalization."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Annotated, get_origin, Literal, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationInfo,
    field_serializer,
    model_validator,
)

from .adjudication_postprocess_v44 import rationale_id_bindings_v44
from .schemas import NonEmptyText, SafeSlug
from .schemas_v4 import (
    ConstraintAssessmentV41,
    EvidenceId,
    PitchEvidenceId,
    PortfolioDisclosureId,
    PortfolioOverlapAssessmentV41,
    RationaleId,
)


ClaimId = Annotated[str, Field(pattern=r"^C[1-9][0-9]*$")]
QuestionId = Annotated[str, Field(pattern=r"^Q[1-9][0-9]*$")]
PitchEvidenceIdV44 = Annotated[str, Field(pattern=r"^P-[0-9]{3,4}$")]
WikiEvidenceIdV44 = Annotated[str, Field(pattern=r"^W-[A-Za-z0-9_-]+$")]
HistoricalEvidenceIdV44 = Annotated[str, Field(pattern=r"^H-[A-Za-z0-9_-]+$")]
PortfolioDisclosureIdV44 = Annotated[
    str, Field(pattern=r"^PM-[0-9a-f]{20}$")
]
Sha256V44 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
RationaleIdV44 = Annotated[str, Field(pattern=r"^R[1-9][0-9]*$")]
ObservationIdV44 = Annotated[str, Field(pattern=r"^U[1-9][0-9]*$")]
ConstraintLinkIdV44 = Annotated[str, Field(pattern=r"^[RUQ][1-9][0-9]*$")]
TargetIdV44 = Annotated[str, Field(pattern=r"^[CQ][1-9][0-9]*$")]
RetrievalActionIdV44 = Annotated[str, Field(pattern=r"^A[1-9][0-9]*$")]


class FrozenStrictModelV44(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="before")
    @classmethod
    def convert_declared_json_arrays(
        cls, value: object, info: ValidationInfo
    ) -> object:
        """Convert JSON arrays only for fields explicitly declared as tuples."""
        if info.mode != "json" or type(value) is not dict:
            return value
        converted = dict(value)
        for field_name, field in cls.model_fields.items():
            if (
                get_origin(field.annotation) is tuple
                and type(converted.get(field_name)) is list
            ):
                converted[field_name] = tuple(converted[field_name])
        return converted


class PitchClaimV44(FrozenStrictModelV44):
    claim_id: ClaimId
    claim_type: Literal["material", "adverse"]
    statement: NonEmptyText
    topic: NonEmptyText
    decision_relevance: NonEmptyText
    pitch_evidence_ids: tuple[PitchEvidenceIdV44, ...] = Field(min_length=1)


class UnansweredQuestionV44(FrozenStrictModelV44):
    question_id: QuestionId
    question: NonEmptyText
    why_material: NonEmptyText
    anchor_pitch_evidence_ids: tuple[PitchEvidenceIdV44, ...] = Field(min_length=1)


class ClaimCoverageV44(FrozenStrictModelV44):
    """Exactly one disposition for an important immutable pitch evidence item."""

    pitch_evidence_id: PitchEvidenceIdV44
    claim_id: ClaimId | None = None
    question_id: QuestionId | None = None
    non_material_justification: NonEmptyText | None = None

    @model_validator(mode="after")
    def require_exactly_one_handling(self) -> "ClaimCoverageV44":
        handling_count = sum(
            value is not None
            for value in (
                self.claim_id,
                self.question_id,
                self.non_material_justification,
            )
        )
        if handling_count != 1:
            raise ValueError("claim coverage requires exactly one handling")
        return self


class ClaimMapV44(FrozenStrictModelV44):
    schema_version: Literal["claim-map-v4.4"]
    episode_slug: SafeSlug
    material_claims: tuple[PitchClaimV44, ...]
    adverse_claims: tuple[PitchClaimV44, ...]
    unanswered_questions: tuple[UnansweredQuestionV44, ...]
    claim_coverage: tuple[ClaimCoverageV44, ...]

    @model_validator(mode="after")
    def validate_claim_map(self) -> "ClaimMapV44":
        if any(row.claim_type != "material" for row in self.material_claims):
            raise ValueError("material_claims must contain material claims")
        if any(row.claim_type != "adverse" for row in self.adverse_claims):
            raise ValueError("adverse_claims must contain adverse claims")

        claims = [*self.material_claims, *self.adverse_claims]
        claim_ids = [row.claim_id for row in claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim IDs must be unique across claim lists")
        question_ids = [row.question_id for row in self.unanswered_questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("question IDs must be unique")

        coverage_evidence = [row.pitch_evidence_id for row in self.claim_coverage]
        if len(coverage_evidence) != len(set(coverage_evidence)):
            raise ValueError("claim coverage pitch evidence IDs must be unique")

        claim_by_id = {row.claim_id: row for row in claims}
        question_by_id = {row.question_id: row for row in self.unanswered_questions}
        for row in self.claim_coverage:
            if row.claim_id is not None:
                target = claim_by_id.get(row.claim_id)
                if target is None:
                    raise ValueError(f"unknown claim coverage target: {row.claim_id}")
                if row.pitch_evidence_id not in target.pitch_evidence_ids:
                    raise ValueError(
                        f"claim {row.claim_id} does not cite pitch evidence "
                        f"{row.pitch_evidence_id}"
                    )
            elif row.question_id is not None:
                target = question_by_id.get(row.question_id)
                if target is None:
                    raise ValueError(
                        f"unknown question coverage target: {row.question_id}"
                    )
                if row.pitch_evidence_id not in target.anchor_pitch_evidence_ids:
                    raise ValueError(
                        f"question {row.question_id} does not cite pitch evidence "
                        f"{row.pitch_evidence_id}"
                    )

        covered = set(coverage_evidence)
        cited = {
            evidence_id for row in claims for evidence_id in row.pitch_evidence_ids
        } | {
            evidence_id
            for row in self.unanswered_questions
            for evidence_id in row.anchor_pitch_evidence_ids
        }
        missing = sorted(cited - covered)
        if missing:
            raise ValueError(
                "pitch evidence citations missing claim coverage: " + ", ".join(missing)
            )
        return self


class EvidenceRegistryRecordV44(FrozenStrictModelV44):
    evidence_id: NonEmptyText
    source_kind: Literal["wiki", "historical", "portfolio"]
    text: NonEmptyText
    source_locator: NonEmptyText
    source_sha256: Sha256V44
    episode_slug: SafeSlug | None = None
    turn_start: int | None = Field(default=None, ge=0, strict=True)
    turn_end: int | None = Field(default=None, ge=0, strict=True)
    decision_status: Literal["In", "Out", "unobserved"] | None = None

    @model_validator(mode="after")
    def validate_source_identity(self) -> "EvidenceRegistryRecordV44":
        identifier_types = {
            "wiki": WikiEvidenceIdV44,
            "historical": HistoricalEvidenceIdV44,
            "portfolio": PortfolioDisclosureIdV44,
        }
        TypeAdapter(identifier_types[self.source_kind]).validate_python(
            self.evidence_id
        )
        if (self.turn_start is None) != (self.turn_end is None):
            raise ValueError("turn bounds must be both present or both absent")
        if (
            self.turn_start is not None
            and self.turn_end is not None
            and self.turn_start > self.turn_end
        ):
            raise ValueError("turn start cannot exceed turn end")
        if self.source_kind == "historical":
            if self.episode_slug is None:
                raise ValueError("historical evidence requires an episode slug")
            if self.turn_start is None:
                raise ValueError("historical evidence requires a turn span")
            if self.decision_status is None:
                raise ValueError("historical evidence requires a decision status")
        elif (
            self.episode_slug is not None
            or self.turn_start is not None
            or self.decision_status is not None
        ):
            raise ValueError("only historical evidence can carry episode metadata")
        return self


class RetrievedEvidenceV44(EvidenceRegistryRecordV44):
    query_id: NonEmptyText
    retrieval_modes: tuple[NonEmptyText, ...] = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)
    eligible: bool
    warning: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_retrieval_modes(self) -> "RetrievedEvidenceV44":
        if len(self.retrieval_modes) != len(set(self.retrieval_modes)):
            raise ValueError("retrieval modes must be unique")
        return self


class RetrievalActionV44(FrozenStrictModelV44):
    action_id: RetrievalActionIdV44
    target_id: TargetIdV44
    source_kind: Literal["wiki", "historical", "portfolio"]
    action_kind: Literal["search", "read"]
    query_id: NonEmptyText
    query: NonEmptyText
    result_evidence_ids: tuple[NonEmptyText, ...]
    opened_episode_slugs: tuple[SafeSlug, ...]
    warnings: tuple[NonEmptyText, ...]

    @model_validator(mode="after")
    def validate_action_sources(self) -> "RetrievalActionV44":
        adapters = {
            "wiki": TypeAdapter(WikiEvidenceIdV44),
            "historical": TypeAdapter(HistoricalEvidenceIdV44),
            "portfolio": TypeAdapter(PortfolioDisclosureIdV44),
        }
        for evidence_id in self.result_evidence_ids:
            try:
                adapters[self.source_kind].validate_python(evidence_id)
            except ValueError as exc:
                raise ValueError("action result ID has wrong source kind") from exc
        if len(self.result_evidence_ids) != len(set(self.result_evidence_ids)):
            raise ValueError("result evidence IDs must be unique within action")
        if len(self.opened_episode_slugs) != len(set(self.opened_episode_slugs)):
            raise ValueError("opened episode slugs must be unique within action")
        if self.action_kind == "search" and self.opened_episode_slugs:
            raise ValueError("search actions cannot open episodes")
        if self.source_kind != "historical" and self.opened_episode_slugs:
            raise ValueError("opened episodes are only valid for historical actions")
        return self


class ClaimEvidenceBundleV44(FrozenStrictModelV44):
    target_kind: Literal["claim", "question"]
    target_id: TargetIdV44
    query: NonEmptyText
    wiki_evidence: tuple[RetrievedEvidenceV44, ...]
    historical_evidence: tuple[RetrievedEvidenceV44, ...]
    portfolio_disclosures: tuple[RetrievedEvidenceV44, ...]
    warnings: tuple[NonEmptyText, ...]
    retrieval_action_ids: tuple[RetrievalActionIdV44, ...]

    @model_validator(mode="after")
    def validate_bundle(self) -> "ClaimEvidenceBundleV44":
        expected_prefix = "C" if self.target_kind == "claim" else "Q"
        if not self.target_id.startswith(expected_prefix):
            raise ValueError("retrieval target kind and ID disagree")
        groups = (
            (self.wiki_evidence, "wiki"),
            (self.historical_evidence, "historical"),
            (self.portfolio_disclosures, "portfolio"),
        )
        evidence_query_ids: list[tuple[str, str]] = []
        for records, expected_kind in groups:
            if any(row.source_kind != expected_kind for row in records):
                raise ValueError("retrieval evidence appears in the wrong source group")
            evidence_query_ids.extend(
                (row.evidence_id, row.query_id) for row in records
            )
        if len(evidence_query_ids) != len(set(evidence_query_ids)):
            raise ValueError(
                "retrieved evidence and query IDs must be unique within a bundle"
            )
        return self


class ClaimRetrievalManifestV44(FrozenStrictModelV44):
    """Ordered audit record; producer order is canonical and is never score-sorted here."""

    schema_version: Literal["claim-retrieval-manifest-v4.4"]
    episode_slug: SafeSlug
    claim_map_sha256: Sha256V44
    claim_bundles: tuple[ClaimEvidenceBundleV44, ...]
    target_episode_excluded: bool
    warnings: tuple[NonEmptyText, ...]
    evidence_registry: tuple[EvidenceRegistryRecordV44, ...]
    retrieval_actions: tuple[RetrievalActionV44, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> "ClaimRetrievalManifestV44":
        targets = [row.target_id for row in self.claim_bundles]
        if len(targets) != len(set(targets)):
            raise ValueError("claim retrieval targets must be unique")
        if not self.target_episode_excluded:
            raise ValueError("claim retrieval must fail closed on target exclusion")
        registry_ids = [row.evidence_id for row in self.evidence_registry]
        if len(registry_ids) != len(set(registry_ids)):
            raise ValueError("registry IDs must be unique")
        action_ids = [row.action_id for row in self.retrieval_actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("retrieval action IDs must be unique")
        registry = {row.evidence_id: row for row in self.evidence_registry}
        actions = {row.action_id: row for row in self.retrieval_actions}
        action_ownership = {action_id: 0 for action_id in actions}
        bundle_relations: dict[str, set[tuple[str, str, str]]] = {}
        source_identity_fields = (
            "source_kind",
            "source_locator",
            "source_sha256",
            "episode_slug",
            "turn_start",
            "turn_end",
            "decision_status",
        )
        for bundle in self.claim_bundles:
            if len(bundle.retrieval_action_ids) != len(
                set(bundle.retrieval_action_ids)
            ):
                raise ValueError("action IDs must be unique within bundle")
            owned_actions: list[RetrievalActionV44] = []
            bundle_relations[bundle.target_id] = {
                (record.evidence_id, record.source_kind, record.query_id)
                for record in [
                    *bundle.wiki_evidence,
                    *bundle.historical_evidence,
                    *bundle.portfolio_disclosures,
                ]
            }
            for record in [
                *bundle.wiki_evidence,
                *bundle.historical_evidence,
                *bundle.portfolio_disclosures,
            ]:
                registered = registry.get(record.evidence_id)
                if registered is None or any(
                    getattr(record, field) != getattr(registered, field)
                    for field in source_identity_fields
                ):
                    raise ValueError(
                        f"bundle evidence {record.evidence_id} does not match registry"
                    )
                if record.eligible and record.text != registered.text:
                    raise ValueError("eligible evidence text does not match registry")
                if not record.eligible and record.text not in registered.text:
                    raise ValueError(
                        "ineligible evidence text must be registry substring"
                    )
            for action_id in bundle.retrieval_action_ids:
                action = actions.get(action_id)
                if action is None:
                    raise ValueError(f"unknown retrieval action: {action_id}")
                if action.target_id != bundle.target_id:
                    raise ValueError(f"retrieval action {action_id} has wrong target")
                action_ownership[action_id] += 1
                owned_actions.append(action)
            for record in [
                *bundle.wiki_evidence,
                *bundle.historical_evidence,
                *bundle.portfolio_disclosures,
            ]:
                if not any(
                    action.source_kind == record.source_kind
                    and action.query_id == record.query_id
                    and record.evidence_id in action.result_evidence_ids
                    for action in owned_actions
                ):
                    raise ValueError(
                        f"bundle evidence {record.evidence_id} is not connected to owned action"
                    )
            for record in bundle.historical_evidence:
                if record.episode_slug == self.episode_slug:
                    raise ValueError("target episode cannot appear in historical evidence")
        query_text_by_id: dict[str, str] = {}
        action_result_ids: set[str] = set()
        for action in self.retrieval_actions:
            prior_query = query_text_by_id.setdefault(action.query_id, action.query)
            if prior_query != action.query:
                raise ValueError("query ID maps to conflicting query strings")
            historical_result_episodes: set[str] = set()
            for evidence_id in action.result_evidence_ids:
                record = registry.get(evidence_id)
                if record is None:
                    raise ValueError(f"action result absent from registry: {evidence_id}")
                if record.source_kind != action.source_kind:
                    raise ValueError("action result has wrong source kind")
                if (
                    evidence_id,
                    action.source_kind,
                    action.query_id,
                ) not in bundle_relations.get(action.target_id, set()):
                    raise ValueError(
                        f"action result {evidence_id} is not materialized in owning bundle"
                    )
                if action.source_kind == "historical":
                    historical_result_episodes.add(record.episode_slug)
                action_result_ids.add(evidence_id)
            if self.episode_slug in action.opened_episode_slugs:
                raise ValueError("target episode cannot appear in retrieval actions")
            if (
                action.source_kind == "historical"
                and action.action_kind == "read"
                and historical_result_episodes != set(action.opened_episode_slugs)
            ):
                raise ValueError(
                    "historical read result episodes must match opened episodes"
                )
            if action_ownership[action.action_id] == 0:
                raise ValueError(f"orphan retrieval action: {action.action_id}")
            if action_ownership[action.action_id] != 1:
                raise ValueError(
                    f"retrieval action must be referenced exactly once: {action.action_id}"
                )
        for evidence_id in registry.keys() - action_result_ids:
            raise ValueError(f"orphan registry record: {evidence_id}")
        for record in self.evidence_registry:
            if record.episode_slug == self.episode_slug:
                raise ValueError("target episode cannot appear in evidence registry")
        return self


class TaxonomyCandidateV44(FrozenStrictModelV44):
    taxonomy_label: NonEmptyText
    definition: NonEmptyText
    coarse_parent: NonEmptyText
    dense_score: float = Field(allow_inf_nan=False)
    lexical_score: float = Field(allow_inf_nan=False)
    fused_score: float = Field(allow_inf_nan=False)
    rank: int = Field(ge=1, strict=True)
    target_id: TargetIdV44
    query: NonEmptyText
    selection_reason: NonEmptyText


class TaxonomyNeighborhoodV44(FrozenStrictModelV44):
    target_id: TargetIdV44
    query: NonEmptyText
    candidates: tuple[TaxonomyCandidateV44, ...]

    @model_validator(mode="after")
    def validate_candidates(self) -> "TaxonomyNeighborhoodV44":
        labels = [row.taxonomy_label for row in self.candidates]
        if len(labels) != len(set(labels)):
            raise ValueError("taxonomy candidate labels must be unique per target")
        ranks = [row.rank for row in self.candidates]
        if ranks != sorted(ranks):
            raise ValueError("taxonomy candidate ranks must be ordered")
        if any(
            row.target_id != self.target_id or row.query != self.query
            for row in self.candidates
        ):
            raise ValueError("taxonomy candidate claim/query binding disagrees")
        return self


class TaxonomyNeighborhoodManifestV44(FrozenStrictModelV44):
    schema_version: Literal["taxonomy-neighborhood-manifest-v4.4"]
    taxonomy_sha256: Sha256V44
    embedding_model: NonEmptyText
    embedding_revision: NonEmptyText
    claim_neighborhoods: tuple[TaxonomyNeighborhoodV44, ...]
    ordered_labels: tuple[NonEmptyText, ...]

    @model_validator(mode="after")
    def validate_ordered_labels(self) -> "TaxonomyNeighborhoodManifestV44":
        targets = [row.target_id for row in self.claim_neighborhoods]
        if len(targets) != len(set(targets)):
            raise ValueError("taxonomy neighborhood targets must be unique")
        expected: list[str] = []
        for neighborhood in self.claim_neighborhoods:
            for candidate in neighborhood.candidates:
                if candidate.taxonomy_label not in expected:
                    expected.append(candidate.taxonomy_label)
        if self.ordered_labels != tuple(expected):
            raise ValueError("ordered_labels must match first-seen neighborhood order")
        return self


class RationaleDispositionV44(FrozenStrictModelV44):
    taxonomy_label: NonEmptyText
    disposition: Literal["core", "candidate", "question_only", "rejected"]
    claim_ids: tuple[ClaimId, ...]
    question_ids: tuple[QuestionId, ...]
    pitch_evidence_ids: tuple[PitchEvidenceIdV44, ...]
    wiki_evidence_ids: tuple[WikiEvidenceIdV44, ...]
    historical_evidence_ids: tuple[HistoricalEvidenceIdV44, ...]
    portfolio_disclosure_ids: tuple[PortfolioDisclosureIdV44, ...]
    direction: Literal["positive", "negative", "neutral"] | None = None
    salience: Literal["primary", "secondary"] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    justification: NonEmptyText

    @model_validator(mode="after")
    def validate_disposition(self) -> "RationaleDispositionV44":
        activation_fields = (self.direction, self.salience, self.confidence)
        if self.disposition in {"core", "candidate"}:
            if not self.claim_ids:
                raise ValueError("activated rationale requires at least one claim ID")
            if not self.pitch_evidence_ids:
                raise ValueError("activated rationale requires pitch evidence")
            if not (
                self.wiki_evidence_ids
                or self.historical_evidence_ids
                or self.portfolio_disclosure_ids
            ):
                raise ValueError("activated rationale requires investor evidence")
            if any(value is None for value in activation_fields):
                raise ValueError(
                    "activated rationale requires direction, salience, and confidence"
                )
        else:
            if any(value is not None for value in activation_fields):
                if self.disposition == "question_only":
                    raise ValueError("question_only cannot activate")
                raise ValueError("rejected rationale cannot retain activation fields")
            if self.disposition == "question_only" and not self.question_ids:
                raise ValueError("question_only requires at least one question ID")
        return self


class UnmappedObservationV44(FrozenStrictModelV44):
    observation_id: ObservationIdV44
    statement: NonEmptyText
    claim_ids: tuple[ClaimId, ...]
    question_ids: tuple[QuestionId, ...]
    pitch_evidence_ids: tuple[PitchEvidenceIdV44, ...]
    wiki_evidence_ids: tuple[WikiEvidenceIdV44, ...]
    historical_evidence_ids: tuple[HistoricalEvidenceIdV44, ...]
    portfolio_disclosure_ids: tuple[PortfolioDisclosureIdV44, ...]
    justification: NonEmptyText


class _FrozenConstraintAssessmentV44(ConstraintAssessmentV41):
    """Immutable snapshot retaining all v4.1 constraint semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    pitch_evidence_ids: tuple[PitchEvidenceId, ...]
    wiki_evidence_ids: tuple[EvidenceId, ...]
    historical_evidence_ids: tuple[EvidenceId, ...]
    mapped_ids: tuple[ConstraintLinkIdV44, ...]

    @model_validator(mode="before")
    @classmethod
    def convert_json_arrays(cls, value: object, info: ValidationInfo) -> object:
        if info.mode != "json" or type(value) is not dict:
            return value
        converted = dict(value)
        for field_name in (
            "pitch_evidence_ids",
            "wiki_evidence_ids",
            "historical_evidence_ids",
            "mapped_ids",
        ):
            if type(converted.get(field_name)) is list:
                converted[field_name] = tuple(converted[field_name])
        return converted


class _FrozenPortfolioOverlapAssessmentV44(PortfolioOverlapAssessmentV41):
    """Immutable snapshot retaining all v4.1 portfolio-overlap semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    disclosure_ids: tuple[PortfolioDisclosureId, ...]
    pitch_evidence_ids: tuple[PitchEvidenceId, ...]


def _validate_raw_assessment_collection(
    payload: dict,
    *,
    field_name: str,
    model_type: type[BaseModel],
    string_fields: Sequence[str],
    list_fields: Sequence[str],
    numeric_fields: Sequence[str] = (),
) -> None:
    if field_name not in payload:
        return
    records = payload[field_name]
    if type(records) is not list:
        raise ValueError(f"{field_name} must be a list")
    for index, record in enumerate(records):
        if isinstance(record, model_type):
            continue
        if type(record) is not dict:
            raise ValueError(f"{field_name}[{index}] must be an object")
        for nested_field in string_fields:
            if nested_field in record and type(record[nested_field]) is not str:
                raise ValueError(
                    f"{field_name}[{index}].{nested_field} must be a string"
                )
        for nested_field in list_fields:
            if nested_field not in record:
                continue
            nested_value = record[nested_field]
            if type(nested_value) is not list:
                raise ValueError(
                    f"{field_name}[{index}].{nested_field} must be a list"
                )
            if any(type(item) is not str for item in nested_value):
                raise ValueError(
                    f"{field_name}[{index}].{nested_field} items must be strings"
                )
        for nested_field in numeric_fields:
            if nested_field in record and type(record[nested_field]) not in (int, float):
                raise ValueError(
                    f"{field_name}[{index}].{nested_field} must be a number"
                )


def _immutable_constraint_assessment(
    assessment: ConstraintAssessmentV41,
) -> ConstraintAssessmentV41:
    payload = assessment.model_dump(mode="python", warnings=False)
    for field_name in (
        "pitch_evidence_ids",
        "wiki_evidence_ids",
        "historical_evidence_ids",
        "mapped_ids",
    ):
        payload[field_name] = tuple(payload[field_name])
    return _FrozenConstraintAssessmentV44.model_validate(payload)


def _immutable_portfolio_assessment(
    assessment: PortfolioOverlapAssessmentV41,
) -> PortfolioOverlapAssessmentV41:
    payload = assessment.model_dump(mode="python", warnings=False)
    for field_name in ("disclosure_ids", "pitch_evidence_ids"):
        payload[field_name] = tuple(payload[field_name])
    return _FrozenPortfolioOverlapAssessmentV44.model_validate(payload)


class RationaleAdjudicationV44(FrozenStrictModelV44):
    schema_version: Literal["rationale-adjudication-v4.4"]
    episode_slug: SafeSlug
    dispositions: tuple[RationaleDispositionV44, ...]
    unmapped_observations: tuple[UnmappedObservationV44, ...]
    constraint_assessments: tuple[_FrozenConstraintAssessmentV44, ...]
    portfolio_overlap_assessments: tuple[PortfolioOverlapAssessmentV41, ...]
    adjudication_status: Literal["valid", "provisional"]
    validator_findings: tuple[NonEmptyText, ...]
    requested_retrieval_ids: tuple[TargetIdV44, ...]

    @field_serializer(
        "constraint_assessments",
        "portfolio_overlap_assessments",
        when_used="json",
    )
    def serialize_reused_assessments(
        self,
        values: tuple[
            _FrozenConstraintAssessmentV44 | PortfolioOverlapAssessmentV41,
            ...,
        ],
    ) -> tuple[dict, ...]:
        return tuple(row.model_dump(mode="json", warnings=False) for row in values)

    @model_validator(mode="before")
    @classmethod
    def enforce_strict_reused_assessment_inputs(cls, value: object) -> object:
        if isinstance(value, cls) or type(value) is not dict:
            return value
        value = dict(value)
        constraints = value.get("constraint_assessments")
        if type(constraints) is list:
            value["constraint_assessments"] = [
                _immutable_constraint_assessment(row)
                if isinstance(row, ConstraintAssessmentV41)
                and not isinstance(row, _FrozenConstraintAssessmentV44)
                else row
                for row in constraints
            ]
        _validate_raw_assessment_collection(
            value,
            field_name="constraint_assessments",
            model_type=_FrozenConstraintAssessmentV44,
            string_fields=(
                "constraint_id",
                "constraint_kind",
                "policy_statement",
                "status",
                "severity",
                "assessment",
            ),
            list_fields=(
                "pitch_evidence_ids",
                "wiki_evidence_ids",
                "historical_evidence_ids",
                "mapped_ids",
            ),
        )
        _validate_raw_assessment_collection(
            value,
            field_name="portfolio_overlap_assessments",
            model_type=PortfolioOverlapAssessmentV41,
            string_fields=(
                "portfolio_entity_id",
                "overlap_status",
                "decision_consequence",
                "assessment",
            ),
            list_fields=("disclosure_ids", "pitch_evidence_ids"),
            numeric_fields=("confidence",),
        )
        for field_name in (
            "constraint_assessments",
            "portfolio_overlap_assessments",
        ):
            if field_name in value:
                value[field_name] = tuple(value[field_name])
        return value

    @model_validator(mode="after")
    def validate_record_ids(self) -> "RationaleAdjudicationV44":
        observation_ids = [row.observation_id for row in self.unmapped_observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("unmapped observation IDs must be unique")
        if len(self.requested_retrieval_ids) != len(set(self.requested_retrieval_ids)):
            raise ValueError("requested retrieval IDs must be unique")
        object.__setattr__(
            self,
            "constraint_assessments",
            tuple(
                _immutable_constraint_assessment(row)
                for row in self.constraint_assessments
            ),
        )
        object.__setattr__(
            self,
            "portfolio_overlap_assessments",
            tuple(
                _immutable_portfolio_assessment(row)
                for row in self.portfolio_overlap_assessments
            ),
        )
        return self


class RationaleV44(FrozenStrictModelV44):
    rationale_id: RationaleIdV44
    taxonomy_label: NonEmptyText
    disposition: Literal["core", "candidate"]
    direction: Literal["positive", "negative", "neutral"]
    salience: Literal["primary", "secondary"]
    confidence: float = Field(ge=0, le=1)
    justification: NonEmptyText
    claim_ids: tuple[ClaimId, ...] = Field(min_length=1)
    question_ids: tuple[QuestionId, ...]
    pitch_evidence_ids: tuple[PitchEvidenceIdV44, ...] = Field(min_length=1)
    wiki_evidence_ids: tuple[WikiEvidenceIdV44, ...]
    historical_evidence_ids: tuple[HistoricalEvidenceIdV44, ...]
    portfolio_disclosure_ids: tuple[PortfolioDisclosureIdV44, ...]

    @model_validator(mode="after")
    def require_investor_evidence(self) -> "RationaleV44":
        if not (
            self.wiki_evidence_ids
            or self.historical_evidence_ids
            or self.portfolio_disclosure_ids
        ):
            raise ValueError("rationale requires investor evidence")
        return self


class InvestigationV44(FrozenStrictModelV44):
    schema_version: Literal["investigation-v4.4"]
    episode_slug: SafeSlug
    claim_map_sha256: Sha256V44
    claim_retrieval_sha256: Sha256V44
    taxonomy_neighborhood_sha256: Sha256V44
    adjudication_sha256: Sha256V44
    material_claims: tuple[PitchClaimV44, ...]
    adverse_claims: tuple[PitchClaimV44, ...]
    claim_coverage: tuple[ClaimCoverageV44, ...]
    candidate_rationales: tuple[RationaleV44, ...]
    core_rationales: tuple[RationaleV44, ...]
    rationales: tuple[RationaleV44, ...]
    unanswered_questions: tuple[UnansweredQuestionV44, ...]
    question_only_dispositions: tuple[RationaleDispositionV44, ...]
    rejected_taxonomy_candidates: tuple[RationaleDispositionV44, ...]
    unmapped_observations: tuple[UnmappedObservationV44, ...]
    constraint_assessments: tuple[_FrozenConstraintAssessmentV44, ...]
    portfolio_overlap_assessments: tuple[PortfolioOverlapAssessmentV41, ...]
    claim_retrieval_manifest: ClaimRetrievalManifestV44
    taxonomy_neighborhood_manifest: TaxonomyNeighborhoodManifestV44
    investigation_status: Literal["accepted", "provisional"]
    validator_findings: tuple[NonEmptyText, ...]

    @field_serializer(
        "constraint_assessments",
        "portfolio_overlap_assessments",
        when_used="json",
    )
    def serialize_reused_assessments(
        self,
        values: tuple[
            _FrozenConstraintAssessmentV44 | PortfolioOverlapAssessmentV41,
            ...,
        ],
    ) -> tuple[dict, ...]:
        return tuple(row.model_dump(mode="json", warnings=False) for row in values)

    @model_validator(mode="after")
    def validate_dual_views(self) -> "InvestigationV44":
        candidate_ids = [row.rationale_id for row in self.candidate_rationales]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate rationale IDs must be unique")
        candidate_by_id = {
            row.rationale_id: row for row in self.candidate_rationales
        }
        for row in self.core_rationales:
            candidate = candidate_by_id.get(row.rationale_id)
            if candidate is None or candidate != row:
                raise ValueError("core rationales must be an exact candidate subset")
            if row.disposition != "core":
                raise ValueError("core rationale must carry the core disposition")
        if self.rationales != self.core_rationales:
            raise ValueError("compatibility rationales must exactly equal core_rationales")
        if any(row.disposition != "question_only" for row in self.question_only_dispositions):
            raise ValueError("question_only_dispositions contains another disposition")
        if any(row.disposition != "rejected" for row in self.rejected_taxonomy_candidates):
            raise ValueError("rejected_taxonomy_candidates contains another disposition")
        if self.claim_retrieval_manifest.episode_slug != self.episode_slug:
            raise ValueError("retrieval manifest episode does not match investigation")
        object.__setattr__(
            self,
            "constraint_assessments",
            tuple(
                _immutable_constraint_assessment(row)
                for row in self.constraint_assessments
            ),
        )
        object.__setattr__(
            self,
            "portfolio_overlap_assessments",
            tuple(
                _immutable_portfolio_assessment(row)
                for row in self.portfolio_overlap_assessments
            ),
        )
        return self


def _bind_array_ids(schema: dict, values: Sequence[str]) -> None:
    allowed = sorted(set(values))
    if allowed:
        schema["items"]["enum"] = allowed
    else:
        schema["maxItems"] = 0


def _bind_scalar_ids(schema: dict, values: Sequence[str]) -> None:
    schema["enum"] = sorted(set(values))


def claim_map_v44_json_schema(
    *, episode_slug: str, pitch_evidence_ids: Sequence[str]
) -> dict:
    """Build a per-call claim schema without mutating Pydantic's model schema."""
    allowed_pitch_ids = _validated_runtime_pitch_ids(pitch_evidence_ids)
    schema = ClaimMapV44.model_json_schema()
    schema["properties"]["episode_slug"]["const"] = episode_slug
    definitions = schema["$defs"]
    _bind_array_ids(
        definitions["PitchClaimV44"]["properties"]["pitch_evidence_ids"],
        allowed_pitch_ids,
    )
    _bind_array_ids(
        definitions["UnansweredQuestionV44"]["properties"][
            "anchor_pitch_evidence_ids"
        ],
        allowed_pitch_ids,
    )
    _bind_scalar_ids(
        definitions["ClaimCoverageV44"]["properties"]["pitch_evidence_id"],
        allowed_pitch_ids,
    )
    coverage = schema["properties"]["claim_coverage"]
    coverage["minItems"] = len(allowed_pitch_ids)
    coverage["maxItems"] = len(allowed_pitch_ids)
    return schema


def _validated_runtime_pitch_ids(
    pitch_evidence_ids: Sequence[str],
) -> tuple[str, ...]:
    adapter = TypeAdapter(PitchEvidenceIdV44, config=ConfigDict(strict=True))
    validated = tuple(adapter.validate_python(value) for value in pitch_evidence_ids)
    if len(validated) != len(set(validated)):
        raise ValueError("runtime pitch evidence IDs must be unique")
    return validated


def validate_claim_map_runtime_v44(
    claim_map: ClaimMapV44, pitch_evidence_ids: Sequence[str]
) -> ClaimMapV44:
    """Require exactly one coverage row for every runtime-accessible pitch item."""
    allowed = set(_validated_runtime_pitch_ids(pitch_evidence_ids))
    covered = {row.pitch_evidence_id for row in claim_map.claim_coverage}
    missing = sorted(allowed - covered)
    unexpected = sorted(covered - allowed)
    messages: list[str] = []
    if missing:
        messages.append("missing claim coverage: " + ", ".join(missing))
    if unexpected:
        messages.append(
            "claim coverage outside runtime inventory: " + ", ".join(unexpected)
        )
    if messages:
        raise ValueError("; ".join(messages))
    return claim_map


def adjudication_v44_json_schema(
    *,
    episode_slug: str,
    claim_ids: Sequence[str],
    question_ids: Sequence[str],
    pitch_evidence_ids: Sequence[str],
    investor_evidence_ids: Sequence[str],
    portfolio_disclosure_ids: Sequence[str],
    taxonomy_labels: Sequence[str],
) -> dict:
    """Build a runtime-bound evidence and taxonomy adjudication schema."""
    schema = RationaleAdjudicationV44.model_json_schema()
    schema["properties"]["episode_slug"]["const"] = episode_slug
    definitions = schema["$defs"]
    disposition = definitions["RationaleDispositionV44"]["properties"]
    observation = definitions["UnmappedObservationV44"]["properties"]
    wiki_ids = [value for value in investor_evidence_ids if value.startswith("W-")]
    historical_ids = [
        value for value in investor_evidence_ids if value.startswith("H-")
    ]
    for properties in (disposition, observation):
        _bind_array_ids(properties["claim_ids"], claim_ids)
        _bind_array_ids(properties["question_ids"], question_ids)
        _bind_array_ids(properties["pitch_evidence_ids"], pitch_evidence_ids)
        _bind_array_ids(properties["wiki_evidence_ids"], wiki_ids)
        _bind_array_ids(properties["historical_evidence_ids"], historical_ids)
        _bind_array_ids(
            properties["portfolio_disclosure_ids"], portfolio_disclosure_ids
        )
    _bind_scalar_ids(disposition["taxonomy_label"], taxonomy_labels)
    _bind_array_ids(schema["properties"]["requested_retrieval_ids"], [*claim_ids, *question_ids])

    constraint = definitions["_FrozenConstraintAssessmentV44"]["properties"]
    _bind_array_ids(constraint["pitch_evidence_ids"], pitch_evidence_ids)
    _bind_array_ids(constraint["wiki_evidence_ids"], wiki_ids)
    _bind_array_ids(constraint["historical_evidence_ids"], historical_ids)
    overlap = definitions["PortfolioOverlapAssessmentV41"]["properties"]
    _bind_array_ids(overlap["pitch_evidence_ids"], pitch_evidence_ids)
    _bind_array_ids(overlap["disclosure_ids"], portfolio_disclosure_ids)
    return schema


rationale_adjudication_v44_json_schema = adjudication_v44_json_schema


def adjudication_findings_v44(
    claim_map: ClaimMapV44,
    adjudication: RationaleAdjudicationV44,
    neighborhood: TaxonomyNeighborhoodManifestV44,
    *,
    pitch_evidence_ids: Sequence[str],
    investor_evidence_ids: Sequence[str],
    portfolio_disclosure_ids: Sequence[str],
    retrieval_manifest: ClaimRetrievalManifestV44 | dict | None = None,
) -> list[str]:
    """Return stable structural findings; unanswered questions are not blockers."""
    findings: set[str] = set()
    claim_ids = {
        row.claim_id for row in [*claim_map.material_claims, *claim_map.adverse_claims]
    }
    question_ids = {row.question_id for row in claim_map.unanswered_questions}
    allowed_pitch = set(pitch_evidence_ids)
    allowed_investor = set(investor_evidence_ids)
    allowed_portfolio = set(portfolio_disclosure_ids)
    allowed_labels = set(neighborhood.ordered_labels)

    coverage_ids = [row.pitch_evidence_id for row in claim_map.claim_coverage]
    for duplicate in _duplicates(coverage_ids):
        findings.add(f"DUPLICATE_COVERAGE:{duplicate}")
    for evidence_id in coverage_ids:
        if evidence_id not in allowed_pitch:
            findings.add(f"INACCESSIBLE_PITCH_EVIDENCE:{evidence_id}")

    for claim in [*claim_map.material_claims, *claim_map.adverse_claims]:
        for evidence_id in claim.pitch_evidence_ids:
            if evidence_id not in allowed_pitch:
                findings.add(f"INACCESSIBLE_PITCH_EVIDENCE:{evidence_id}")
    for question in claim_map.unanswered_questions:
        for evidence_id in question.anchor_pitch_evidence_ids:
            if evidence_id not in allowed_pitch:
                findings.add(f"INACCESSIBLE_PITCH_EVIDENCE:{evidence_id}")

    handled_claims: set[str] = set()
    rows: list[RationaleDispositionV44 | UnmappedObservationV44] = [
        *adjudication.dispositions,
        *adjudication.unmapped_observations,
    ]
    for row in rows:
        for claim_id in row.claim_ids:
            if claim_id not in claim_ids:
                findings.add(f"UNKNOWN_CLAIM:{claim_id}")
            else:
                handled_claims.add(claim_id)
        for question_id in row.question_ids:
            if question_id not in question_ids:
                findings.add(f"UNKNOWN_QUESTION:{question_id}")
        for evidence_id in row.pitch_evidence_ids:
            if evidence_id not in allowed_pitch:
                findings.add(f"INACCESSIBLE_PITCH_EVIDENCE:{evidence_id}")
        for evidence_id in [*row.wiki_evidence_ids, *row.historical_evidence_ids]:
            if evidence_id not in allowed_investor:
                findings.add(f"INACCESSIBLE_INVESTOR_EVIDENCE:{evidence_id}")
        for evidence_id in row.portfolio_disclosure_ids:
            if evidence_id not in allowed_portfolio:
                findings.add(f"INACCESSIBLE_PORTFOLIO_DISCLOSURE:{evidence_id}")

    for row in adjudication.dispositions:
        if row.taxonomy_label not in allowed_labels:
            findings.add(f"LABEL_OUTSIDE_NEIGHBORHOOD:{row.taxonomy_label}")
    if retrieval_manifest is not None:
        retrieval = _validate_retrieval_manifest_v44(retrieval_manifest)
        pitch_by_target: dict[str, set[str]] = {
            row.claim_id: set(row.pitch_evidence_ids)
            for row in [*claim_map.material_claims, *claim_map.adverse_claims]
        }
        pitch_by_target.update(
            {
                row.question_id: set(row.anchor_pitch_evidence_ids)
                for row in claim_map.unanswered_questions
            }
        )
        for row in rows:
            target_ids = {*row.claim_ids, *row.question_ids}
            if not target_ids:
                continue
            target_pitch = set().union(
                *(pitch_by_target.get(target_id, set()) for target_id in target_ids)
            )
            for evidence_id in row.pitch_evidence_ids:
                if evidence_id in allowed_pitch and evidence_id not in target_pitch:
                    findings.add(
                        f"PITCH_EVIDENCE_NOT_BOUND_TO_TARGET:{evidence_id}"
                    )
    for constraint in adjudication.constraint_assessments:
        for evidence_id in constraint.pitch_evidence_ids:
            if evidence_id not in allowed_pitch:
                findings.add(f"INACCESSIBLE_PITCH_EVIDENCE:{evidence_id}")
        for evidence_id in [
            *constraint.wiki_evidence_ids,
            *constraint.historical_evidence_ids,
        ]:
            if evidence_id not in allowed_investor:
                findings.add(f"INACCESSIBLE_INVESTOR_EVIDENCE:{evidence_id}")
    for overlap in adjudication.portfolio_overlap_assessments:
        for evidence_id in overlap.pitch_evidence_ids:
            if evidence_id not in allowed_pitch:
                findings.add(f"INACCESSIBLE_PITCH_EVIDENCE:{evidence_id}")
        for evidence_id in overlap.disclosure_ids:
            if evidence_id not in allowed_portfolio:
                findings.add(f"INACCESSIBLE_PORTFOLIO_DISCLOSURE:{evidence_id}")
    for claim_id in sorted(claim_ids - handled_claims):
        findings.add(f"UNHANDLED_CLAIM:{claim_id}")

    requested = set(adjudication.requested_retrieval_ids)
    for target_id in requested - claim_ids - question_ids:
        findings.add(f"UNKNOWN_RETRIEVAL_TARGET:{target_id}")
    return sorted(findings)


def _duplicates(values: Sequence[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _model_sha256(value: BaseModel) -> str:
    payload = json.dumps(
        value.model_dump(mode="json", warnings=False),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _eligible_evidence_ids(
    manifest: ClaimRetrievalManifestV44,
) -> tuple[set[str], set[str]]:
    investor: set[str] = set()
    portfolio: set[str] = set()
    for bundle in manifest.claim_bundles:
        for record in [*bundle.wiki_evidence, *bundle.historical_evidence]:
            if record.eligible:
                investor.add(record.evidence_id)
        for record in bundle.portfolio_disclosures:
            if record.eligible:
                portfolio.add(record.evidence_id)
    return investor, portfolio


def _validate_retrieval_manifest_v44(
    value: ClaimRetrievalManifestV44 | dict,
) -> ClaimRetrievalManifestV44:
    """Accept a validated manifest or an ordinary JSON-decoded manifest object."""
    if isinstance(value, ClaimRetrievalManifestV44):
        return value
    if type(value) is dict:
        return ClaimRetrievalManifestV44.model_validate_json(
            json.dumps(value, sort_keys=True, separators=(",", ":"))
        )
    raise TypeError("retrieval manifest must be a model or JSON object")


def build_investigation_v44(
    *,
    episode_slug: str,
    claim_map: ClaimMapV44,
    adjudication: RationaleAdjudicationV44,
    retrieval_manifest: ClaimRetrievalManifestV44 | dict,
    neighborhood: TaxonomyNeighborhoodManifestV44,
    claim_map_sha256: str,
    adjudication_sha256: str,
    pitch_evidence_ids: Sequence[str],
) -> InvestigationV44:
    """Freeze stable broad/core views from a validated evidence-first record."""
    validated_claim_hash = TypeAdapter(Sha256V44).validate_python(claim_map_sha256)
    validated_adjudication_hash = TypeAdapter(Sha256V44).validate_python(
        adjudication_sha256
    )
    actual_claim_hash = _model_sha256(claim_map)
    if validated_claim_hash != actual_claim_hash:
        raise ValueError("claim map hash does not match canonical claim map")
    actual_adjudication_hash = _model_sha256(adjudication)
    if validated_adjudication_hash != actual_adjudication_hash:
        raise ValueError("adjudication hash does not match canonical adjudication")
    allowed_pitch_ids = _validated_runtime_pitch_ids(pitch_evidence_ids)
    validate_claim_map_runtime_v44(claim_map, allowed_pitch_ids)
    retrieval = _validate_retrieval_manifest_v44(retrieval_manifest)
    if claim_map.episode_slug != episode_slug:
        raise ValueError("claim map episode does not match final episode")
    if adjudication.episode_slug != episode_slug:
        raise ValueError("adjudication episode does not match final episode")
    if retrieval.episode_slug != episode_slug:
        raise ValueError("retrieval episode does not match final episode")
    if retrieval.claim_map_sha256 != validated_claim_hash:
        raise ValueError("retrieval manifest does not bind the supplied claim map hash")
    expected_targets = {
        row.claim_id
        for row in [*claim_map.material_claims, *claim_map.adverse_claims]
    } | {row.question_id for row in claim_map.unanswered_questions}
    retrieval_targets = {row.target_id for row in retrieval.claim_bundles}
    if retrieval_targets != expected_targets:
        raise ValueError("retrieval targets do not match claim map targets")
    neighborhood_targets = {row.target_id for row in neighborhood.claim_neighborhoods}
    if neighborhood_targets != expected_targets:
        raise ValueError("taxonomy neighborhood targets do not match claim map targets")

    candidate_rationales: list[RationaleV44] = []
    for rationale_id, row in rationale_id_bindings_v44(
        adjudication.dispositions, neighborhood
    ):
        assert row.direction is not None
        assert row.salience is not None
        assert row.confidence is not None
        candidate_rationales.append(
            RationaleV44(
                rationale_id=rationale_id,
                taxonomy_label=row.taxonomy_label,
                disposition=row.disposition,
                direction=row.direction,
                salience=row.salience,
                confidence=row.confidence,
                justification=row.justification,
                claim_ids=row.claim_ids,
                question_ids=row.question_ids,
                pitch_evidence_ids=row.pitch_evidence_ids,
                wiki_evidence_ids=row.wiki_evidence_ids,
                historical_evidence_ids=row.historical_evidence_ids,
                portfolio_disclosure_ids=row.portfolio_disclosure_ids,
            )
        )
    core_rationales = [
        row for row in candidate_rationales if row.disposition == "core"
    ]

    investor_ids, portfolio_ids = _eligible_evidence_ids(retrieval)
    structural_findings = adjudication_findings_v44(
        claim_map,
        adjudication,
        neighborhood,
        pitch_evidence_ids=allowed_pitch_ids,
        investor_evidence_ids=sorted(investor_ids),
        portfolio_disclosure_ids=sorted(portfolio_ids),
        retrieval_manifest=retrieval,
    )
    findings = sorted({*adjudication.validator_findings, *structural_findings})
    allowed_mapped_ids = {
        row.rationale_id for row in candidate_rationales
    } | {row.observation_id for row in adjudication.unmapped_observations} | {
        question_id
        for row in adjudication.dispositions
        if row.disposition == "question_only"
        for question_id in row.question_ids
    }
    for constraint in adjudication.constraint_assessments:
        for mapped_id in constraint.mapped_ids:
            if mapped_id not in allowed_mapped_ids:
                findings.append(
                    "DANGLING_CONSTRAINT_MAPPED_ID:"
                    f"{constraint.constraint_id}:{mapped_id}"
                )
    findings = sorted(set(findings))
    status = (
        "accepted"
        if adjudication.adjudication_status == "valid" and not findings
        else "provisional"
    )

    return InvestigationV44(
        schema_version="investigation-v4.4",
        episode_slug=episode_slug,
        claim_map_sha256=validated_claim_hash,
        claim_retrieval_sha256=_model_sha256(retrieval),
        taxonomy_neighborhood_sha256=_model_sha256(neighborhood),
        adjudication_sha256=validated_adjudication_hash,
        material_claims=claim_map.material_claims,
        adverse_claims=claim_map.adverse_claims,
        claim_coverage=claim_map.claim_coverage,
        candidate_rationales=tuple(candidate_rationales),
        core_rationales=tuple(core_rationales),
        rationales=tuple(core_rationales),
        unanswered_questions=claim_map.unanswered_questions,
        question_only_dispositions=tuple(
            row
            for row in adjudication.dispositions
            if row.disposition == "question_only"
        ),
        rejected_taxonomy_candidates=tuple(
            row for row in adjudication.dispositions if row.disposition == "rejected"
        ),
        unmapped_observations=adjudication.unmapped_observations,
        constraint_assessments=adjudication.constraint_assessments,
        portfolio_overlap_assessments=adjudication.portfolio_overlap_assessments,
        claim_retrieval_manifest=retrieval,
        taxonomy_neighborhood_manifest=neighborhood,
        investigation_status=status,
        validator_findings=tuple(findings),
    )
