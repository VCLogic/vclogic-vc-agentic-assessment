"""Natural, evidence-linked contracts for the adaptive v4 investor workflow."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Annotated, Literal, Sequence

from pydantic import Field, model_validator

from .schemas import NonEmptyText, SafeSlug, StrictModel


EvidenceId = Annotated[str, Field(pattern=r"^[WH]-[A-Za-z0-9_-]+$")]
PitchEvidenceId = Annotated[str, Field(pattern=r"^P-[0-9]{3,4}$")]
SourceEvidenceId = Annotated[str, Field(pattern=r"^[WHP]-[A-Za-z0-9_-]+$")]
RationaleId = Annotated[str, Field(pattern=r"^[RU][1-9][0-9]*$")]
ConstraintId = Annotated[str, Field(pattern=r"^C[1-9][0-9]*$")]
PortfolioEntityId = Annotated[str, Field(pattern=r"^PE-[0-9a-f]{20}$")]
PortfolioDisclosureId = Annotated[str, Field(pattern=r"^PM-[0-9a-f]{20}$")]


_GENERATED_ID = re.compile(r"^([RCU])[-_]?0*([1-9][0-9]*)$")
_OBSERVATION_ALIAS_ID = re.compile(r"^(?:UO|O)[-_]?0*([1-9][0-9]*)$")
_INVESTOR_EVIDENCE_ID = re.compile(r"^[WH]-[A-Za-z0-9_-]+$")
_DECISION_DIMENSIONS = {
    "founder_ambition",
    "founder_commitment",
    "founder_execution",
    "investor_fit",
    "conflict",
    "stage_or_check_fit",
    "ownership_or_governance",
    "market",
    "product",
    "traction",
    "business_model",
    "venture_economics",
    "competition",
    "timing",
    "other",
}
_CONSTRAINT_KINDS = {
    "category_or_expertise_fit",
    "portfolio_conflict",
    "stage_or_check_fit",
    "founder_ambition",
    "ownership_or_governance",
    "venture_economics",
    "other",
}


def _canonical_generated_id(value: object) -> object:
    if type(value) is not str:
        return value
    match = _GENERATED_ID.fullmatch(value)
    if match is None:
        return value
    return f"{match.group(1)}{int(match.group(2))}"


def _canonical_observation_id(value: object) -> object:
    canonical = _canonical_generated_id(value)
    if canonical != value or type(value) is not str:
        return canonical
    match = _OBSERVATION_ALIAS_ID.fullmatch(value)
    if match is None:
        return value
    return f"U{int(match.group(1))}"


def normalize_investigation_v41_payload(
    payload: object,
) -> tuple[object, list[str]]:
    """Repair mechanical v4.1 variants without changing investment semantics."""
    normalized = deepcopy(payload)
    findings: list[str] = []
    if type(normalized) is not dict or type(normalized.get("rationales")) is not list:
        return normalized, findings

    labels: dict[str, list[str]] = {}
    rationale_ids: set[str] = set()
    for row in normalized["rationales"]:
        if type(row) is not dict:
            continue
        original = row.get("rationale_id")
        canonical = _canonical_generated_id(original)
        if canonical != original:
            row["rationale_id"] = canonical
            findings.append(f"NORMALIZED_GENERATED_ID:{original}->{canonical}")
        if type(canonical) is str and canonical.startswith("R"):
            rationale_ids.add(canonical)
        label = row.get("taxonomy_label")
        if type(label) is str and type(canonical) is str:
            labels.setdefault(label, []).append(canonical)

    unique_labels = {
        label: ids[0] for label, ids in labels.items() if len(ids) == 1
    }

    observation_aliases: dict[str, str] = {}

    def normalize_mapped_ids(
        row: dict,
        *,
        context: str,
        allowed_prefix: str | None = None,
        drop_unknown: bool = False,
    ) -> None:
        values = row.get("mapped_ids")
        if type(values) is not list:
            return
        repaired: list[object] = []
        for value in values:
            canonical = _canonical_generated_id(value)
            if canonical == value and type(value) is str:
                canonical = observation_aliases.get(
                    value, unique_labels.get(value, value)
                )
            if canonical != value:
                findings.append(f"NORMALIZED_MAPPED_ID:{value}->{canonical}")
            repaired.append(canonical)

        known_ids = {
            *rationale_ids,
            *unique_labels.values(),
            *observation_aliases.values(),
        }
        valid = [
            value
            for value in repaired
            if type(value) is str
            and value in known_ids
            and (allowed_prefix is None or value.startswith(allowed_prefix))
        ]
        if valid and valid != repaired:
            row["mapped_ids"] = valid
            findings.append(
                f"NORMALIZED_FILTERED_MAPPED_IDS:{context}:"
                f"removed={len(repaired) - len(valid)}"
            )
        elif drop_unknown and not valid and repaired:
            row["mapped_ids"] = []
            findings.append(
                f"NORMALIZED_FILTERED_MAPPED_IDS:{context}:"
                f"removed={len(repaired)}"
            )
        else:
            row["mapped_ids"] = repaired

    observations = normalized.get("unmapped_observations")
    if type(observations) is list:
        for row in observations:
            if type(row) is not dict:
                continue
            original = row.get("observation_id")
            canonical = _canonical_observation_id(original)
            if canonical != original:
                row["observation_id"] = canonical
                findings.append(f"NORMALIZED_GENERATED_ID:{original}->{canonical}")
            if type(original) is str and type(canonical) is str:
                observation_aliases[original] = canonical
                observation_aliases[canonical] = canonical

    questions = normalized.get("questions")
    if type(questions) is list:
        for position, row in enumerate(questions, start=1):
            if type(row) is not dict or type(row.get("evidence_ids")) is not list:
                continue
            values = row["evidence_ids"]
            repaired = [
                value
                for value in values
                if type(value) is str and _INVESTOR_EVIDENCE_ID.fullmatch(value)
            ]
            if repaired != values:
                row["evidence_ids"] = repaired
                findings.append(
                    f"NORMALIZED_QUESTION_EVIDENCE:{position}:"
                    f"removed={len(values) - len(repaired)}"
                )

    statements = normalized.get("material_statement_coverage")
    if type(statements) is list:
        for row in statements:
            if type(row) is not dict:
                continue
            mapping_type = row.get("mapping_type")
            allowed_prefix = (
                "R" if mapping_type == "taxonomy_rationale" else
                "U" if mapping_type == "unmapped_observation" else None
            )
            normalize_mapped_ids(
                row,
                context="material_statement",
                allowed_prefix=allowed_prefix,
            )
            dimension = row.get("decision_dimension")
            if type(dimension) is str and dimension not in _DECISION_DIMENSIONS:
                row["decision_dimension"] = "other"
                findings.append(
                    f"NORMALIZED_DECISION_DIMENSION:{dimension}->other"
                )

    constraints = normalized.get("constraint_assessments")
    if type(constraints) is list:
        repaired_constraints: list[object] = []
        for row in constraints:
            if type(row) is not dict:
                repaired_constraints.append(row)
                continue
            original = row.get("constraint_id")
            canonical = _canonical_generated_id(original)
            if canonical != original:
                row["constraint_id"] = canonical
                findings.append(f"NORMALIZED_GENERATED_ID:{original}->{canonical}")
            kind = row.get("constraint_kind")
            if type(kind) is str and kind not in _CONSTRAINT_KINDS:
                row["constraint_kind"] = "other"
                findings.append(f"NORMALIZED_CONSTRAINT_KIND:{kind}->other")
            normalize_mapped_ids(
                row,
                context="constraint",
                drop_unknown=True,
            )
            missing: list[str] = []
            if type(row.get("pitch_evidence_ids")) is not list or not row[
                "pitch_evidence_ids"
            ]:
                missing.append("pitch_evidence")
            wiki = row.get("wiki_evidence_ids")
            history = row.get("historical_evidence_ids")
            if not (type(wiki) is list and wiki) and not (
                type(history) is list and history
            ):
                missing.append("investor_evidence")
            if type(row.get("mapped_ids")) is not list or not row["mapped_ids"]:
                missing.append("mapping")
            if missing:
                findings.append(
                    "NORMALIZED_DROPPED_CONSTRAINT:"
                    f"{row.get('constraint_id', 'unknown')}:"
                    + ",".join(missing)
                )
                continue
            repaired_constraints.append(row)
        normalized["constraint_assessments"] = repaired_constraints

    return normalized, findings


def normalize_rationale_mapping_v43_payload(
    payload: object,
) -> tuple[object, list[str]]:
    """Repair structured-output branch fill without changing mapping semantics."""
    normalized = deepcopy(payload)
    findings: list[str] = []
    if type(normalized) is not dict or type(normalized.get("dispositions")) is not list:
        return normalized, findings
    for row in normalized["dispositions"]:
        if type(row) is not dict or row.get("action") == "merge":
            continue
        if row.get("merge_into_rationale_id") is not None:
            findings.append(f"NORMALIZED_NONMERGE_TARGET:{row.get('rationale_id')}")
            row["merge_into_rationale_id"] = None
    return normalized, findings


def normalize_rationale_lock_v42_payload(
    payload: object,
) -> tuple[object, list[str]]:
    """Clear fields that structured output populated on the inapplicable branch."""
    normalized = deepcopy(payload)
    findings: list[str] = []
    if type(normalized) is not dict or type(normalized.get("dispositions")) is not list:
        return normalized, findings
    for row in normalized["dispositions"]:
        if type(row) is not dict:
            continue
        identifier = row.get("rationale_id")
        if row.get("decision") == "locked" and row.get("rejection_reason") is not None:
            row["rejection_reason"] = None
            findings.append(f"NORMALIZED_LOCKED_REJECTION_REASON:{identifier}")
        elif row.get("decision") == "rejected" and any(
            row.get(field) is not None
            for field in ("direction", "salience", "confidence")
        ):
            row["direction"] = None
            row["salience"] = None
            row["confidence"] = None
            findings.append(f"NORMALIZED_REJECTED_LOCK_ATTRIBUTES:{identifier}")
    return normalized, findings


def normalize_investigation_v41_evidence_payload(
    payload: object,
    *,
    wiki_evidence_ids: set[str],
    historical_evidence_ids: set[str],
) -> tuple[object, list[str]]:
    """Remove inaccessible citations without inventing replacement evidence."""
    normalized = deepcopy(payload)
    findings: list[str] = []
    if type(normalized) is not dict:
        return normalized, findings

    def filter_field(
        row: dict,
        field: str,
        allowed: set[str],
        kind: str,
        identifier: str,
    ) -> None:
        values = row.get(field)
        if type(values) is not list:
            return
        retained = [value for value in values if value in allowed]
        for value in values:
            if value not in allowed:
                findings.append(
                    f"NORMALIZED_DROPPED_INACCESSIBLE_{kind}_EVIDENCE:{identifier}:{value}"
                )
        row[field] = retained

    for row in normalized.get("rationales", []):
        if type(row) is not dict:
            continue
        identifier = str(row.get("rationale_id", "unknown"))
        filter_field(
            row,
            "wiki_evidence_ids",
            wiki_evidence_ids,
            "RATIONALE",
            identifier,
        )
        filter_field(
            row,
            "historical_evidence_ids",
            historical_evidence_ids,
            "RATIONALE",
            identifier,
        )
    combined = wiki_evidence_ids | historical_evidence_ids
    for index, row in enumerate(normalized.get("questions", []), start=1):
        if type(row) is dict:
            filter_field(row, "evidence_ids", combined, "QUESTION", str(index))
    for row in normalized.get("unmapped_observations", []):
        if type(row) is dict:
            filter_field(
                row,
                "evidence_ids",
                combined,
                "OBSERVATION",
                str(row.get("observation_id", "unknown")),
            )
    for row in normalized.get("constraint_assessments", []):
        if type(row) is not dict:
            continue
        identifier = str(row.get("constraint_id", "unknown"))
        filter_field(
            row,
            "wiki_evidence_ids",
            wiki_evidence_ids,
            "CONSTRAINT",
            identifier,
        )
        filter_field(
            row,
            "historical_evidence_ids",
            historical_evidence_ids,
            "CONSTRAINT",
            identifier,
        )
    return normalized, findings


def model_facing_taxonomy(
    rows: Sequence[dict[str, object]],
) -> tuple[dict[str, str], ...]:
    """Return every rationale's decision-relevant fields and no build metadata."""
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        compact: dict[str, str] = {}
        for key in ("label", "definition", "coarse_parent"):
            value = row.get(key)
            if type(value) is not str or not value.strip():
                raise ValueError(f"taxonomy row requires nonempty {key}")
            compact[key] = value
        if compact["label"] in seen:
            raise ValueError("taxonomy labels must be unique")
        seen.add(compact["label"])
        result.append(compact)
    if not result:
        raise ValueError("taxonomy must not be empty")
    return tuple(result)


def _bind_array_ids(schema: dict, values: Sequence[str]) -> None:
    allowed = sorted(set(values))
    if allowed:
        schema["items"]["enum"] = allowed
    else:
        schema["maxItems"] = 0


def investigation_v4_json_schema(
    *,
    episode_slug: str,
    pitch_evidence_ids: Sequence[str],
    wiki_evidence_ids: Sequence[str],
    historical_evidence_ids: Sequence[str],
) -> dict:
    schema = InvestigationV4.model_json_schema()
    schema["properties"]["episode_slug"]["const"] = episode_slug
    definitions = schema["$defs"]
    rationale = definitions["RationaleV4"]["properties"]
    question = definitions["QuestionAssessmentV4"]["properties"]
    observation = definitions["UnmappedObservationV4"]["properties"]
    _bind_array_ids(rationale["pitch_evidence_ids"], pitch_evidence_ids)
    _bind_array_ids(rationale["wiki_evidence_ids"], wiki_evidence_ids)
    _bind_array_ids(rationale["historical_evidence_ids"], historical_evidence_ids)
    investor_evidence_ids = (*wiki_evidence_ids, *historical_evidence_ids)
    _bind_array_ids(question["evidence_ids"], investor_evidence_ids)
    _bind_array_ids(observation["evidence_ids"], investor_evidence_ids)
    return schema


def decision_v4_json_schema(
    *,
    episode_slug: str,
    investigation_sha256: str,
    pitch_evidence_ids: Sequence[str],
    wiki_evidence_ids: Sequence[str],
    historical_evidence_ids: Sequence[str],
    rationale_ids: Sequence[str],
) -> dict:
    schema = DecisionV4.model_json_schema()
    properties = schema["properties"]
    properties["episode_slug"]["const"] = episode_slug
    properties["investigation_sha256"]["const"] = investigation_sha256
    evidence_ids = (*pitch_evidence_ids, *wiki_evidence_ids, *historical_evidence_ids)
    basis = schema["$defs"]["EvidenceBasisV4"]["properties"]
    _bind_array_ids(basis["evidence_ids"], evidence_ids)
    _bind_array_ids(properties["controlling_rationale_ids"], rationale_ids)
    _bind_array_ids(properties["founder_exception_rationale_ids"], rationale_ids)
    _bind_array_ids(
        properties["founder_exception_precedent_ids"], historical_evidence_ids
    )
    return schema


def investigation_v41_json_schema(
    *,
    episode_slug: str,
    pitch_evidence_ids: Sequence[str],
    wiki_evidence_ids: Sequence[str],
    historical_evidence_ids: Sequence[str],
    portfolio_entity_ids: Sequence[str] = (),
    portfolio_disclosure_ids: Sequence[str] = (),
) -> dict:
    schema = InvestigationV41.model_json_schema()
    schema["properties"]["episode_slug"]["const"] = episode_slug
    definitions = schema["$defs"]
    rationale = definitions["RationaleV41"]["properties"]
    question = definitions["QuestionAssessmentV4"]["properties"]
    observation = definitions["UnmappedObservationV4"]["properties"]
    statement = definitions["MaterialStatementCoverageV41"]["properties"]
    constraint = definitions["ConstraintAssessmentV41"]["properties"]
    overlap = definitions["PortfolioOverlapAssessmentV41"]["properties"]
    _bind_array_ids(
        schema["properties"]["reviewed_pitch_evidence_ids"], pitch_evidence_ids
    )
    statement["pitch_evidence_id"]["enum"] = sorted(set(pitch_evidence_ids))
    _bind_array_ids(rationale["pitch_evidence_ids"], pitch_evidence_ids)
    _bind_array_ids(rationale["wiki_evidence_ids"], wiki_evidence_ids)
    _bind_array_ids(rationale["historical_evidence_ids"], historical_evidence_ids)
    investor_evidence_ids = (*wiki_evidence_ids, *historical_evidence_ids)
    _bind_array_ids(question["evidence_ids"], investor_evidence_ids)
    _bind_array_ids(observation["evidence_ids"], investor_evidence_ids)
    _bind_array_ids(constraint["pitch_evidence_ids"], pitch_evidence_ids)
    _bind_array_ids(constraint["wiki_evidence_ids"], wiki_evidence_ids)
    _bind_array_ids(
        constraint["historical_evidence_ids"], historical_evidence_ids
    )
    if portfolio_entity_ids:
        overlap["portfolio_entity_id"]["enum"] = sorted(set(portfolio_entity_ids))
    else:
        schema["properties"]["portfolio_overlap_assessments"]["maxItems"] = 0
    _bind_array_ids(overlap["disclosure_ids"], portfolio_disclosure_ids)
    return schema


def rationale_lock_v42_json_schema(
    *,
    episode_slug: str,
    candidate_investigation_sha256: str,
    candidate_rationale_ids: Sequence[str],
) -> dict:
    schema = RationaleLockV42.model_json_schema()
    properties = schema["properties"]
    properties["episode_slug"]["const"] = episode_slug
    properties["candidate_investigation_sha256"]["const"] = (
        candidate_investigation_sha256
    )
    dispositions = properties["dispositions"]
    dispositions["minItems"] = len(candidate_rationale_ids)
    dispositions["maxItems"] = len(candidate_rationale_ids)
    disposition = schema["$defs"]["RationaleLockDispositionV42"]["properties"]
    disposition["rationale_id"]["enum"] = sorted(set(candidate_rationale_ids))
    return schema


def rationale_mapping_v43_json_schema(
    *,
    episode_slug: str,
    candidate_investigation_sha256: str,
    candidate_rationale_ids: Sequence[str],
    taxonomy_labels: Sequence[str],
) -> dict:
    schema = RationaleMappingV43.model_json_schema()
    properties = schema["properties"]
    properties["episode_slug"]["const"] = episode_slug
    properties["candidate_investigation_sha256"]["const"] = (
        candidate_investigation_sha256
    )
    dispositions = properties["dispositions"]
    dispositions["minItems"] = len(candidate_rationale_ids)
    dispositions["maxItems"] = len(candidate_rationale_ids)
    disposition = schema["$defs"]["RationaleMappingDispositionV43"]["properties"]
    disposition["rationale_id"]["enum"] = sorted(set(candidate_rationale_ids))
    disposition["target_taxonomy_label"]["enum"] = sorted(set(taxonomy_labels))
    disposition["merge_into_rationale_id"] = {
        "type": "string",
        "enum": sorted({"none", *candidate_rationale_ids}),
    }
    return schema


def decision_v41_json_schema(
    *,
    episode_slug: str,
    investigation_sha256: str,
    pitch_evidence_ids: Sequence[str],
    wiki_evidence_ids: Sequence[str],
    historical_evidence_ids: Sequence[str],
    rationale_ids: Sequence[str],
    constraint_ids: Sequence[str],
) -> dict:
    schema = DecisionV41.model_json_schema()
    properties = schema["properties"]
    properties["episode_slug"]["const"] = episode_slug
    properties["investigation_sha256"]["const"] = investigation_sha256
    evidence_ids = (*pitch_evidence_ids, *wiki_evidence_ids, *historical_evidence_ids)
    basis = schema["$defs"]["EvidenceBasisV4"]["properties"]
    _bind_array_ids(basis["evidence_ids"], evidence_ids)
    _bind_array_ids(properties["controlling_rationale_ids"], rationale_ids)
    _bind_array_ids(properties["founder_exception_rationale_ids"], rationale_ids)
    _bind_array_ids(
        properties["founder_exception_precedent_ids"], historical_evidence_ids
    )
    _bind_array_ids(properties["blocking_constraint_ids"], constraint_ids)
    return schema


class RetrievalPlanV4(StrictModel):
    questions: list[NonEmptyText] = Field(min_length=1, max_length=6)
    wiki_queries: list[NonEmptyText] = Field(max_length=6)
    precedent_queries: list[NonEmptyText] = Field(max_length=6)
    continuation_focus: NonEmptyText


class QuestionAssessmentV4(StrictModel):
    question: NonEmptyText
    status: Literal["answered", "partial", "unanswered"]
    answer: NonEmptyText
    evidence_ids: list[EvidenceId]


class RationaleV4(StrictModel):
    rationale_id: Annotated[str, Field(pattern=r"^R[1-9][0-9]*$")]
    taxonomy_label: NonEmptyText
    direction: Literal["positive", "negative", "neutral"]
    salience: Literal["primary", "secondary"]
    confidence: float = Field(ge=0, le=1)
    pitch_evidence: list[NonEmptyText] = Field(min_length=1)
    pitch_evidence_ids: list[PitchEvidenceId] = Field(min_length=1)
    wiki_evidence_ids: list[EvidenceId]
    historical_evidence_ids: list[EvidenceId]
    justification: NonEmptyText

    @model_validator(mode="after")
    def require_investor_evidence(self) -> "RationaleV4":
        if not self.wiki_evidence_ids and not self.historical_evidence_ids:
            raise ValueError("rationale requires wiki or historical investor evidence")
        return self


class UnmappedObservationV4(StrictModel):
    observation_id: Annotated[str, Field(pattern=r"^U[1-9][0-9]*$")]
    description: NonEmptyText
    pitch_evidence: list[NonEmptyText] = Field(min_length=1)
    evidence_ids: list[EvidenceId] = Field(min_length=1)
    decision_relevance: NonEmptyText


class InvestigationV4(StrictModel):
    schema_version: Literal["investigation-v4"]
    episode_slug: SafeSlug
    questions: list[QuestionAssessmentV4] = Field(min_length=1, max_length=12)
    rationales: list[RationaleV4] = Field(min_length=1)
    conflicts: list[NonEmptyText]
    unmapped_observations: list[UnmappedObservationV4]
    information_sufficient: bool
    sufficiency_assessment: NonEmptyText
    searchable_questions: list[NonEmptyText]
    diligence_questions: list[NonEmptyText]
    next_search_objectives: list[NonEmptyText]
    summary: NonEmptyText

    @model_validator(mode="after")
    def validate_sufficiency(self) -> "InvestigationV4":
        # Sufficiency/search-field inconsistencies are audit findings, not reasons to
        # throw away an otherwise usable investment assessment.
        return self


class MaterialStatementCoverageV41(StrictModel):
    pitch_evidence_id: PitchEvidenceId
    decision_dimension: Literal[
        "founder_ambition",
        "founder_commitment",
        "founder_execution",
        "investor_fit",
        "conflict",
        "stage_or_check_fit",
        "ownership_or_governance",
        "market",
        "product",
        "traction",
        "business_model",
        "venture_economics",
        "competition",
        "timing",
        "other",
    ]
    direction: Literal["positive", "negative", "neutral"]
    constraint_signal: Literal["none", "possible", "triggered"]
    mapping_type: Literal["taxonomy_rationale", "unmapped_observation"]
    mapped_ids: list[RationaleId] = Field(min_length=1)
    assessment: NonEmptyText

    @model_validator(mode="after")
    def validate_mapping_prefix(self) -> "MaterialStatementCoverageV41":
        expected = "R" if self.mapping_type == "taxonomy_rationale" else "U"
        if any(not value.startswith(expected) for value in self.mapped_ids):
            raise ValueError("material statement mapping type and IDs disagree")
        return self


class ConstraintAssessmentV41(StrictModel):
    constraint_id: ConstraintId
    constraint_kind: Literal[
        "category_or_expertise_fit",
        "portfolio_conflict",
        "stage_or_check_fit",
        "founder_ambition",
        "ownership_or_governance",
        "venture_economics",
        "other",
    ]
    policy_statement: NonEmptyText
    status: Literal["triggered", "possible", "not_triggered"]
    severity: Literal["hard", "material", "ordinary"]
    pitch_evidence_ids: list[PitchEvidenceId] = Field(min_length=1)
    wiki_evidence_ids: list[EvidenceId]
    historical_evidence_ids: list[EvidenceId]
    mapped_ids: list[RationaleId] = Field(min_length=1)
    assessment: NonEmptyText

    @model_validator(mode="after")
    def require_investor_evidence(self) -> "ConstraintAssessmentV41":
        if not self.wiki_evidence_ids and not self.historical_evidence_ids:
            raise ValueError("constraint requires wiki or historical investor evidence")
        return self


class PortfolioOverlapAssessmentV41(StrictModel):
    portfolio_entity_id: PortfolioEntityId
    disclosure_ids: list[PortfolioDisclosureId] = Field(min_length=1)
    pitch_evidence_ids: list[PitchEvidenceId] = Field(min_length=1)
    overlap_status: Literal[
        "direct_conflict",
        "possible_conflict",
        "complementary",
        "no_material_overlap",
        "insufficient_information",
    ]
    decision_consequence: Literal[
        "blocking", "permission_required", "diligence_only", "none"
    ]
    confidence: float = Field(ge=0, le=1)
    assessment: NonEmptyText


class RationaleV41(StrictModel):
    """Compact v4.1 rationale: exact pitch wording is resolved through immutable P IDs."""

    rationale_id: Annotated[str, Field(pattern=r"^R[1-9][0-9]*$")]
    taxonomy_label: NonEmptyText
    direction: Literal["positive", "negative", "neutral"]
    salience: Literal["primary", "secondary"]
    confidence: float = Field(ge=0, le=1)
    pitch_evidence_ids: list[PitchEvidenceId] = Field(min_length=1)
    wiki_evidence_ids: list[EvidenceId]
    historical_evidence_ids: list[EvidenceId]
    justification: NonEmptyText

    @model_validator(mode="after")
    def require_investor_evidence(self) -> "RationaleV41":
        if not self.wiki_evidence_ids and not self.historical_evidence_ids:
            raise ValueError("rationale requires wiki or historical investor evidence")
        return self


class InvestigationV41(InvestigationV4):
    schema_version: Literal["investigation-v4.1"]
    questions: list[QuestionAssessmentV4] = Field(min_length=1, max_length=8)
    rationales: list[RationaleV41] = Field(min_length=1, max_length=16)
    reviewed_pitch_evidence_ids: list[PitchEvidenceId] = Field(min_length=1)
    material_statement_coverage: list[MaterialStatementCoverageV41] = Field(
        min_length=1, max_length=12
    )
    constraint_assessments: list[ConstraintAssessmentV41] = Field(max_length=8)
    portfolio_overlap_assessments: list[PortfolioOverlapAssessmentV41] = Field(
        default_factory=list, max_length=8
    )

    @model_validator(mode="after")
    def validate_coverage_mappings(self) -> "InvestigationV41":
        allowed = {row.rationale_id for row in self.rationales} | {
            row.observation_id for row in self.unmapped_observations
        }
        referenced = {
            value
            for row in self.material_statement_coverage
            for value in row.mapped_ids
        } | {
            value for row in self.constraint_assessments for value in row.mapped_ids
        }
        unknown = sorted(referenced - allowed)
        if unknown:
            raise ValueError(
                "unknown frozen rationale or observation: " + ", ".join(unknown)
            )
        coverage_ids = [row.pitch_evidence_id for row in self.material_statement_coverage]
        if len(coverage_ids) != len(set(coverage_ids)):
            raise ValueError("material statement pitch evidence IDs must be unique")
        constraint_ids = [row.constraint_id for row in self.constraint_assessments]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("constraint IDs must be unique")
        return self


RationaleLockRejectionReason = Literal[
    "generic_checklist",
    "insufficient_pitch_evidence",
    "insufficient_investor_specificity",
    "diligence_only",
    "duplicate",
    "non_material",
    "direction_unresolved",
]


class RationaleLockDispositionV42(StrictModel):
    rationale_id: Annotated[str, Field(pattern=r"^R[1-9][0-9]*$")]
    decision: Literal["locked", "rejected"]
    rejection_reason: RationaleLockRejectionReason | None
    direction: Literal["positive", "negative", "neutral"] | None
    salience: Literal["primary", "secondary"] | None
    confidence: float | None = Field(default=None, ge=0, le=1)
    materiality_justification: NonEmptyText

    @model_validator(mode="after")
    def validate_disposition(self) -> "RationaleLockDispositionV42":
        if self.decision == "locked":
            if self.rejection_reason is not None:
                raise ValueError("locked rationale cannot have a rejection reason")
            if self.direction is None or self.salience is None or self.confidence is None:
                raise ValueError("locked rationale requires direction, salience, and confidence")
        else:
            if self.rejection_reason is None:
                raise ValueError("rejected rationale requires a rejection reason")
            if self.direction is not None or self.salience is not None or self.confidence is not None:
                raise ValueError("rejected rationale cannot retain lock attributes")
        return self


class RationaleLockV42(StrictModel):
    schema_version: Literal["rationale-lock-v4.2"]
    episode_slug: SafeSlug
    candidate_investigation_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    dispositions: list[RationaleLockDispositionV42]
    lock_summary: NonEmptyText

    @model_validator(mode="after")
    def require_unique_dispositions(self) -> "RationaleLockV42":
        identifiers = [row.rationale_id for row in self.dispositions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate rationales must be dispositioned exactly once")
        return self


class InvestigationV42(InvestigationV41):
    schema_version: Literal["investigation-v4.2"]
    rationales: list[RationaleV41] = Field(max_length=16)
    candidate_rationale_ids: list[Annotated[str, Field(pattern=r"^R[1-9][0-9]*$")]]
    candidate_investigation_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    rationale_lock_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    rejected_candidate_count: int = Field(ge=0, le=16)

    @model_validator(mode="after")
    def validate_coverage_mappings(self) -> "InvestigationV42":
        locked = {row.rationale_id for row in self.rationales}
        candidate = set(self.candidate_rationale_ids)
        if not locked <= candidate:
            raise ValueError("locked rationale is absent from candidate rationale IDs")
        if self.rejected_candidate_count != len(candidate - locked):
            raise ValueError("rejected candidate count does not match the locked subset")
        allowed = candidate | {row.observation_id for row in self.unmapped_observations}
        referenced = {
            value
            for row in self.material_statement_coverage
            for value in row.mapped_ids
        } | {
            value for row in self.constraint_assessments for value in row.mapped_ids
        }
        unknown = sorted(referenced - allowed)
        if unknown:
            raise ValueError(
                "unknown candidate rationale or observation: " + ", ".join(unknown)
            )
        return self


class RationaleMappingDispositionV43(StrictModel):
    rationale_id: Annotated[str, Field(pattern=r"^R[1-9][0-9]*$")]
    action: Literal["keep", "relabel", "merge"]
    target_taxonomy_label: NonEmptyText
    merge_into_rationale_id: (
        Annotated[str, Field(pattern=r"^R[1-9][0-9]*$")] | None
    ) = None
    winning_definition: NonEmptyText
    rejected_alternatives: list[NonEmptyText]
    mapping_justification: NonEmptyText

    @model_validator(mode="after")
    def validate_action(self) -> "RationaleMappingDispositionV43":
        if self.action == "merge":
            if self.merge_into_rationale_id is None:
                raise ValueError("merge action requires merge_into_rationale_id")
            if self.merge_into_rationale_id == self.rationale_id:
                raise ValueError("merge target must differ from rationale ID")
        elif self.merge_into_rationale_id is not None:
            raise ValueError("only merge action may specify merge_into_rationale_id")
        return self


class RationaleMappingV43(StrictModel):
    schema_version: Literal["rationale-mapping-v4.3"]
    episode_slug: SafeSlug
    candidate_investigation_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    dispositions: list[RationaleMappingDispositionV43]
    mapping_summary: NonEmptyText

    @model_validator(mode="after")
    def require_unique_dispositions(self) -> "RationaleMappingV43":
        identifiers = [row.rationale_id for row in self.dispositions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate rationales must be dispositioned exactly once")
        return self


class InvestigationV43(InvestigationV41):
    schema_version: Literal["investigation-v4.3"]
    rationales: list[RationaleV41] = Field(max_length=16)
    candidate_rationale_ids: list[Annotated[str, Field(pattern=r"^R[1-9][0-9]*$")]]
    candidate_investigation_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    rationale_mapping_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    relabeled_candidate_count: int = Field(ge=0, le=16)
    merged_candidate_count: int = Field(ge=0, le=16)
    mapping_status: Literal["accepted", "fallback_original"]

    @model_validator(mode="after")
    def validate_mapping_counts(self) -> "InvestigationV43":
        candidate = set(self.candidate_rationale_ids)
        final = {row.rationale_id for row in self.rationales}
        if len(candidate) != len(self.candidate_rationale_ids):
            raise ValueError("candidate rationale IDs must be unique")
        if not final <= candidate:
            raise ValueError("mapped rationale is absent from candidate rationale IDs")
        if self.relabeled_candidate_count + self.merged_candidate_count > len(candidate):
            raise ValueError("mapping counts exceed candidate rationale count")
        if len(candidate) - len(final) != self.merged_candidate_count:
            raise ValueError("merged candidate count does not match final rationale set")
        return self


class EvidenceBasisV4(StrictModel):
    source_type: Literal["pitch", "wiki", "precedent"]
    source_reference: NonEmptyText
    evidence_ids: list[SourceEvidenceId] = Field(min_length=1)
    interpretation: NonEmptyText
    effect_on_decision: Literal["supports", "opposes", "context"]


class OpposingCaseV4(StrictModel):
    argument: NonEmptyText
    response: NonEmptyText


class DecisionV4(StrictModel):
    schema_version: Literal["decision-v4"]
    episode_slug: SafeSlug
    investigation_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    decision: Literal["In", "Out"]
    decision_path: Literal["conventional_fit", "founder_conviction_exception", "out"]
    investment_likelihood: float = Field(ge=0, le=1)
    decision_confidence: float = Field(ge=0, le=1)
    decision_justification: NonEmptyText
    recommended_check_tier: Literal["none", "exploratory_lt_100k", "standard"]
    review_priority_score: float = Field(ge=0, le=1)
    review_priority_reason: NonEmptyText
    controlling_rationale_ids: list[RationaleId] = Field(min_length=1)
    evidence_basis: list[EvidenceBasisV4] = Field(min_length=1)
    strongest_opposing_case: OpposingCaseV4
    searchable_questions: list[NonEmptyText]
    diligence_questions: list[NonEmptyText]
    reversal_conditions: list[NonEmptyText]
    information_sufficient: bool
    sufficiency_assessment: NonEmptyText
    next_search_objectives: list[NonEmptyText]
    phase1_reopen_recommended: bool
    missing_considerations: list[NonEmptyText]
    founder_exception_considered: bool
    founder_exception_rationale_ids: list[RationaleId]
    founder_exception_precedent_ids: list[EvidenceId]
    founder_exception_assessment: NonEmptyText

    @model_validator(mode="after")
    def validate_decision_semantics(self) -> "DecisionV4":
        if (self.decision == "In") != (self.investment_likelihood >= 0.5):
            raise ValueError("decision and investment likelihood disagree")
        if (self.decision == "Out") != (self.decision_path == "out"):
            raise ValueError("decision path and final decision disagree")
        if self.decision_path == "founder_conviction_exception":
            if not self.founder_exception_considered or not self.founder_exception_rationale_ids:
                raise ValueError("founder exception path requires supporting rationales")
        if self.decision == "Out" and self.recommended_check_tier != "none":
            raise ValueError("Out decision requires check tier none")
        if self.decision == "In" and self.recommended_check_tier == "none":
            raise ValueError("In decision requires a funded check tier")
        # Search/sufficiency inconsistencies are retained as quality findings by the
        # workflow so a valid final decision is never lost over minor bookkeeping.
        if self.phase1_reopen_recommended and not self.missing_considerations:
            raise ValueError("Phase 1 reopening requires a missing consideration")
        return self


class DecisionV41(DecisionV4):
    schema_version: Literal["decision-v4.1"]
    blocking_constraint_ids: list[ConstraintId]
    constraint_assessment_summary: NonEmptyText
    founder_exception_precedent_match: NonEmptyText

    @model_validator(mode="after")
    def validate_v41_exception(self) -> "DecisionV41":
        if self.decision_path == "founder_conviction_exception":
            if not self.founder_exception_precedent_ids:
                raise ValueError(
                    "founder exception requires an observed In precedent"
                )
            if self.blocking_constraint_ids:
                raise ValueError(
                    "founder exception cannot override a blocking constraint"
                )
        if self.decision == "In" and self.blocking_constraint_ids:
            raise ValueError("In decision cannot retain a blocking constraint")
        return self


def validate_decision_v41_against_investigation(
    decision: DecisionV41, investigation: InvestigationV41
) -> None:
    known = {row.constraint_id for row in investigation.constraint_assessments}
    unknown = sorted(set(decision.blocking_constraint_ids) - known)
    if unknown:
        raise ValueError("unknown blocking constraint: " + ", ".join(unknown))
    triggered_hard = {
        row.constraint_id
        for row in investigation.constraint_assessments
        if row.status == "triggered" and row.severity == "hard"
    }
    if decision.decision == "In" and triggered_hard:
        raise ValueError(
            "In decision conflicts with triggered hard constraint: "
            + ", ".join(sorted(triggered_hard))
        )
    missing = sorted(triggered_hard - set(decision.blocking_constraint_ids))
    if decision.decision == "Out" and missing:
        raise ValueError(
            "triggered hard constraint missing from blocking constraints: "
            + ", ".join(missing)
        )


class ProposedRationaleV4(StrictModel):
    provisional_id: Annotated[str, Field(pattern=r"^P[1-9][0-9]*$")]
    proposed_label: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    proposed_definition: NonEmptyText
    confidence: float = Field(ge=0.9, le=1)
    material_to_frozen_decision: Literal[True]
    reusable_in_future_decisions: Literal[True]
    pitch_evidence: list[NonEmptyText] = Field(min_length=1)
    investor_evidence_ids: list[EvidenceId] = Field(min_length=1)
    nearest_existing_labels: list[NonEmptyText] = Field(min_length=1)
    why_existing_taxonomy_is_insufficient: NonEmptyText
    necessity_justification: NonEmptyText


class TaxonomyReflectionV4(StrictModel):
    schema_version: Literal["taxonomy-reflection-v4"]
    episode_slug: SafeSlug
    decision_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    proposals: list[ProposedRationaleV4]
    review_summary: NonEmptyText
