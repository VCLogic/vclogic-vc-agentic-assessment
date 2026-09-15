"""Compact provider-neutral contracts for both decision phases."""

from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryPlan(StrictModel):
    questions: list[str] = Field(min_length=2, max_length=6)
    search_queries: list[str] = Field(min_length=2, max_length=8)
    reconsideration_focus: str


SafeSlug = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
NonEmptyText = Annotated[str, Field(min_length=1)]
HistoricalEvidenceId = Annotated[str, Field(pattern=r"^H-[0-9a-f]{20}$")]


class TranscriptReadRequest(StrictModel):
    """An exact precedent transcript read, optionally bounded to a turn range."""

    episode_slug: SafeSlug
    turn_start: int | None = Field(ge=0, strict=True)
    turn_end: int | None = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_turn_range(self) -> TranscriptReadRequest:
        if (self.turn_start is None) != (self.turn_end is None):
            raise ValueError(
                "transcript turn bounds must be both JSON null or both nonnegative integers"
            )
        if (
            self.turn_start is not None
            and self.turn_end is not None
            and self.turn_start > self.turn_end
        ):
            raise ValueError("transcript turn start cannot exceed turn end")
        return self


def normalize_retrieval_plan_payload(payload: object) -> tuple[object, list[str]]:
    """Repair unambiguous full reads and discard only invalid read children."""
    normalized = deepcopy(payload)
    findings: list[str] = []
    if type(normalized) is not dict or type(normalized.get("transcript_reads")) is not list:
        return normalized, findings
    retained: list[dict] = []
    for index, row in enumerate(normalized["transcript_reads"]):
        if (
            type(row) is dict
            and row.get("turn_start") == 0
            and row.get("turn_end") is None
        ):
            row["turn_start"] = None
            findings.append(
                f"NORMALIZED_FULL_TRANSCRIPT_RANGE:{row.get('episode_slug', index)}"
            )
        try:
            retained.append(
                TranscriptReadRequest.model_validate(row).model_dump(mode="json")
            )
        except Exception:
            slug = row.get("episode_slug") if type(row) is dict else None
            findings.append(
                f"DROPPED_INVALID_TRANSCRIPT_READ:{index}:{slug or 'unknown'}"
            )
    normalized["transcript_reads"] = retained
    return normalized, findings


class Phase1RetrievalPlan(StrictModel):
    """A v3 plan spanning investor memory and historical precedent stores."""

    questions: list[NonEmptyText] = Field(min_length=2, max_length=6)
    wiki_queries: list[NonEmptyText] = Field(min_length=1, max_length=8)
    precedent_queries: list[NonEmptyText] = Field(max_length=8)
    transcript_reads: list[TranscriptReadRequest] = Field(max_length=8)
    decision_reads: list[SafeSlug] = Field(max_length=8)
    reconsideration_focus: str = Field(min_length=1)


class Phase2RetrievalPlan(StrictModel):
    """A provider-portable decision plan over the independently opened corpus."""

    decision_questions: list[NonEmptyText] = Field(min_length=2, max_length=6)
    wiki_queries: list[NonEmptyText] = Field(max_length=8)
    precedent_queries: list[NonEmptyText] = Field(max_length=8)
    transcript_reads: list[TranscriptReadRequest] = Field(max_length=8)
    decision_reads: list[SafeSlug] = Field(max_length=8)
    opposing_case_to_test: str = Field(min_length=1)


class Rationale(StrictModel):
    rationale_id: str = Field(pattern=r"^R[1-9][0-9]*$")
    label: str = Field(min_length=1)
    direction: Literal["positive", "negative", "neutral"]
    salience: Literal["primary", "secondary"]
    confidence: float = Field(ge=0, le=1)
    pitch_evidence: list[str] = Field(min_length=1)
    wiki_evidence_ids: list[str] = Field(min_length=1)
    interpretation: str = Field(min_length=1)


class Investigation(StrictModel):
    episode_slug: str
    questions: list[str] = Field(min_length=1)
    rationales: list[Rationale] = Field(min_length=1)
    conflicts: list[str]
    unanswered_questions: list[str]
    saturated: bool
    summary: str = Field(min_length=1)


class DealFact(StrictModel):
    """A pitch-grounded deal term, kept separate from investment rationales."""

    status: Literal["observed", "unknown", "not_applicable"]
    value: str | None
    pitch_evidence: list[str]

    @model_validator(mode="after")
    def validate_status_payload(self) -> DealFact:
        if self.status == "observed":
            if not self.value or not self.pitch_evidence:
                raise ValueError("observed deal fact requires a value and pitch evidence")
        elif self.status == "unknown":
            if self.value is not None:
                raise ValueError("unknown deal fact cannot contain a value")
        elif self.value is not None or self.pitch_evidence:
            raise ValueError(
                "not-applicable deal fact cannot contain a value or pitch evidence"
            )
        return self


class DealContext(StrictModel):
    """Deal mechanics that must not be conflated with one another."""

    total_round_size: DealFact
    company_stage: DealFact
    entry_valuation: DealFact
    possible_investor_check: DealFact
    lead_required: DealFact
    ownership_feasibility: DealFact


class RationaleV2(Rationale):
    """A rationale whose evidence state and constraint severity are explicit."""

    evidence_status: Literal[
        "affirmative_positive", "affirmative_adverse", "unresolved"
    ]
    constraint_kind: Literal[
        "none",
        "portfolio_overlap",
        "total_round_size",
        "company_stage",
        "entry_valuation",
        "investor_check_size",
        "lead_requirement",
        "ownership_feasibility",
        "vision_alignment",
        "other",
    ]
    constraint_severity: Literal[
        "none", "routine", "size_limiting", "material", "fatal"
    ]
    severity_basis: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_provenance_and_severity(self) -> RationaleV2:
        if self.evidence_status == "unresolved" and self.constraint_severity == "fatal":
            raise ValueError("unresolved evidence cannot be fatal")
        if self.constraint_kind == "total_round_size" and self.constraint_severity == "fatal":
            raise ValueError("total round size cannot be fatal")
        if self.constraint_kind == "none" and self.constraint_severity != "none":
            raise ValueError("constraint kind none requires severity none")
        return self


class InvestigationV2(Investigation):
    schema_version: Literal["investigation-v2"]
    rationales: list[RationaleV2] = Field(min_length=1)
    deal_context: DealContext


class RationaleV3(RationaleV2):
    """A v2 rationale augmented by source-bound historical precedent evidence."""

    wiki_evidence_ids: list[str]
    historical_evidence_ids: list[HistoricalEvidenceId]
    precedent_interpretation: str = Field(min_length=1)


class RationaleDisposition(StrictModel):
    """The explicit Phase 1 disposition of one configured taxonomy rationale."""

    label: NonEmptyText
    disposition: Literal["activated", "rejected"]
    basis: NonEmptyText


class InvestigationV3(InvestigationV2):
    """Phase 1 evidence plus an auditable taxonomy-candidate funnel."""

    schema_version: Literal["investigation-v3"]
    rationales: list[RationaleV3] = Field(min_length=1)
    activated_candidates: list[NonEmptyText]
    queried_candidates: list[NonEmptyText]
    rejected_candidates: list[NonEmptyText]
    unmapped_candidates: list[NonEmptyText]
    taxonomy_dispositions: list[RationaleDisposition] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_candidate_deduplication(self) -> InvestigationV3:
        for field_name in (
            "activated_candidates",
            "queried_candidates",
            "rejected_candidates",
            "unmapped_candidates",
        ):
            candidates = getattr(self, field_name)
            if len(candidates) != len(set(candidates)):
                raise ValueError(f"{field_name} must be deduplicated")
        disposition_labels = [row.label for row in self.taxonomy_dispositions]
        if len(disposition_labels) != len(set(disposition_labels)):
            raise ValueError("taxonomy disposition labels must be deduplicated")
        return self


def normalize_investigation_v2_payload(
    payload: object,
) -> tuple[object, list[str]]:
    """Canonicalize repairable cross-field constraint typing mismatches."""
    normalized = deepcopy(payload)
    findings: list[str] = []
    if type(normalized) is not dict or type(normalized.get("rationales")) is not list:
        return normalized, findings
    for rationale in normalized["rationales"]:
        if type(rationale) is not dict:
            continue
        if (
            rationale.get("constraint_kind") == "none"
            and rationale.get("constraint_severity") == "none"
            and rationale.get("severity_basis") == ""
        ):
            rationale["severity_basis"] = "No constraint identified."
            findings.append(
                "NORMALIZED_EMPTY_NO_CONSTRAINT_BASIS:"
                f"{rationale.get('rationale_id', 'unknown')}"
            )
        elif rationale.get("severity_basis") == "":
            rationale["severity_basis"] = (
                "Model supplied no severity basis; review required."
            )
            findings.append(
                "NORMALIZED_EMPTY_CONSTRAINT_BASIS:"
                f"{rationale.get('rationale_id', 'unknown')}"
            )
        if (
            rationale.get("constraint_kind") == "none"
            and rationale.get("constraint_severity") not in {None, "none"}
        ):
            rationale["constraint_kind"] = "other"
            findings.append(
                "NORMALIZED_UNTYPED_CONSTRAINT:"
                f"{rationale.get('rationale_id', 'unknown')}:none->other"
            )
    return normalized, findings


def normalize_investigation_v3_payload(
    payload: object,
) -> tuple[object, list[str]]:
    """Apply the same repairable constraint normalization to a v3 payload."""
    return normalize_investigation_v2_payload(payload)


class RationaleAssessment(StrictModel):
    rationale_id: str = Field(pattern=r"^R[1-9][0-9]*$")
    effective_direction: Literal["positive", "negative", "neutral"]
    decision_weight: Literal["decisive", "supporting", "contextual"]
    assessment: str = Field(min_length=1)
    counterevidence: list[str]


class Counterargument(StrictModel):
    argument: str = Field(min_length=1)
    rationale_ids: list[str] = Field(min_length=1)
    response: str = Field(min_length=1)


class AnyCheckDecision(StrictModel):
    decision: Literal["In", "Out"]
    likelihood: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    supporting_rationale_ids: list[str] = Field(min_length=1)
    opposing_rationale_ids: list[str]
    fatal_constraint_present: bool
    optionality_explanation: str = Field(min_length=1)
    strongest_counterargument: Counterargument
    reversal_conditions: list[str]


class MarketGate(StrictModel):
    status: Literal["positive", "negative", "unresolved"]
    controlling_rationale_ids: list[str]
    explanation: str = Field(min_length=1)


class StandardCheckDecision(StrictModel):
    decision: Literal["In", "Out"]
    likelihood: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    supporting_rationale_ids: list[str] = Field(min_length=1)
    opposing_rationale_ids: list[str]
    failure_rationale_ids: list[str]
    market_gate: MarketGate
    strongest_counterargument: Counterargument
    upgrade_conditions: list[str]


class RiskLedgerEntry(StrictModel):
    rationale_id: str = Field(pattern=r"^R[1-9][0-9]*$")
    risk_type: Literal[
        "fatal_constraint",
        "affirmative_adverse",
        "missing_evidence",
        "size_limiting",
        "rebutted",
    ]
    controlling_for_any_check: bool
    counterevidence: list[str]
    explanation: str = Field(min_length=1)


class DeliberationStep(StrictModel):
    step_id: str = Field(pattern=r"^D[1-9][0-9]*$")
    endpoint: Literal["any_check", "standard_check"]
    stage: Literal["assessment", "opposing_case", "decision", "consistency"]
    question: str = Field(min_length=1)
    rationale_ids: list[str] = Field(min_length=1)
    evidence_assessment: str = Field(min_length=1)
    likelihood_before: float = Field(ge=0, le=1)
    likelihood_after: float = Field(ge=0, le=1)
    effect: Literal["raises", "lowers", "no_change"]
    check_tier_implication: str = Field(min_length=1)
    decision_update: str = Field(min_length=1)


class Decision(StrictModel):
    episode_slug: str
    investigation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["In", "Out"]
    investment_likelihood: float = Field(ge=0, le=1)
    decision_confidence: float = Field(ge=0, le=1)
    ranking_score: float = Field(ge=0, le=1)
    check_tier: str = Field(min_length=1)
    decision_endpoint: Literal["any_check"]
    recommended_check_tier: str = Field(min_length=1)
    controlling_rationale_ids: list[str] = Field(min_length=1)
    any_check: AnyCheckDecision
    standard_check: StandardCheckDecision
    risk_ledger: list[RiskLedgerEntry]
    rationale_assessments: list[RationaleAssessment] = Field(min_length=1)
    deliberation_steps: list[DeliberationStep] = Field(min_length=3, max_length=8)
    strongest_counterargument: str = Field(min_length=1)
    unresolved_questions: list[str]
    reversal_conditions: list[str]
    feedback: str = Field(min_length=1)


class DecisionV2(Decision):
    schema_version: Literal["decision-v2"]


class DecisivePrecedent(StrictModel):
    episode_slug: SafeSlug
    historical_evidence_ids: list[HistoricalEvidenceId] = Field(min_length=1)
    observed_decision: Literal["In", "Out", "unobserved"]
    comparison: Literal["supports", "opposes", "exception", "context"]
    explanation: str = Field(min_length=1)


class DecisionV3(DecisionV2):
    schema_version: Literal["decision-v3"]
    stable: bool
    decisive_precedents: list[DecisivePrecedent]
    exception_analogies: list[NonEmptyText]


def _normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _pitch_supports(evidence: str, pitch: str) -> bool:
    """Accept exact spans or conservative close paraphrases of a local pitch span."""
    if evidence.lower() in pitch.lower():
        return True
    evidence_tokens = _normalized_tokens(evidence)
    pitch_tokens = _normalized_tokens(pitch)
    if len(evidence_tokens) < 4 or not pitch_tokens:
        return False
    shared = set(evidence_tokens) & set(pitch_tokens)
    if len(shared) / len(set(evidence_tokens)) < 0.7:
        return False
    target = " ".join(evidence_tokens)
    lower = max(1, len(evidence_tokens) - 3)
    upper = min(len(pitch_tokens), len(evidence_tokens) + 3)
    for width in range(lower, upper + 1):
        for start in range(0, len(pitch_tokens) - width + 1):
            window = " ".join(pitch_tokens[start : start + width])
            if SequenceMatcher(None, target, window).ratio() >= 0.86:
                return True
    return False


def investigation_json_schema(
    *, episode_slug: str, taxonomy_labels: set[str], exact_wiki_ids: set[str]
) -> dict:
    """Return a strict schema with run-specific identifiers constrained at decoding."""
    schema = deepcopy(Investigation.model_json_schema())
    schema["properties"]["episode_slug"]["const"] = episode_slug
    rationale = schema["$defs"]["Rationale"]["properties"]
    rationale["label"]["enum"] = sorted(taxonomy_labels)
    rationale["wiki_evidence_ids"]["items"] = {
        "type": "string",
        "enum": sorted(exact_wiki_ids),
    }
    return schema


def investigation_v2_json_schema(
    *, episode_slug: str, taxonomy_labels: set[str], exact_wiki_ids: set[str]
) -> dict:
    """Return the strict v2 schema bound to this episode and retrieved evidence."""
    schema = deepcopy(InvestigationV2.model_json_schema())
    schema["properties"]["episode_slug"]["const"] = episode_slug
    rationale = schema["$defs"]["RationaleV2"]["properties"]
    rationale["label"]["enum"] = sorted(taxonomy_labels)
    rationale["wiki_evidence_ids"]["items"] = {
        "type": "string",
        "enum": sorted(exact_wiki_ids),
    }
    return schema


def investigation_v3_json_schema(
    *,
    episode_slug: str,
    taxonomy_labels: set[str],
    exact_wiki_ids: set[str],
    exact_historical_ids: set[str],
) -> dict:
    """Return v3 decoding constraints bound to evidence opened in this run."""
    schema = deepcopy(InvestigationV3.model_json_schema())
    schema["properties"]["episode_slug"]["const"] = episode_slug
    rationale = schema["$defs"]["RationaleV3"]["properties"]
    rationale["label"]["enum"] = sorted(taxonomy_labels)
    wiki = rationale["wiki_evidence_ids"]
    wiki.pop("minItems", None)
    if exact_wiki_ids:
        wiki["items"] = {
            "type": "string",
            "enum": sorted(exact_wiki_ids),
        }
    else:
        wiki["maxItems"] = 0
    historical = rationale["historical_evidence_ids"]
    historical.pop("minItems", None)
    if exact_historical_ids:
        historical["items"] = {
            "type": "string",
            "enum": sorted(exact_historical_ids),
        }
    else:
        # An empty enum is not accepted uniformly by structured-output providers.
        # maxItems=0 expresses the same closed-world constraint portably.
        historical["maxItems"] = 0
    for field_name in (
        "activated_candidates",
        "queried_candidates",
        "rejected_candidates",
    ):
        schema["properties"][field_name]["uniqueItems"] = True
        schema["properties"][field_name]["items"] = {
            "type": "string",
            "enum": sorted(taxonomy_labels),
        }
    schema["properties"]["unmapped_candidates"]["uniqueItems"] = True
    disposition = schema["$defs"]["RationaleDisposition"]["properties"]
    disposition["label"]["enum"] = sorted(taxonomy_labels)
    schema["properties"]["taxonomy_dispositions"]["minItems"] = len(
        taxonomy_labels
    )
    schema["properties"]["taxonomy_dispositions"]["maxItems"] = len(
        taxonomy_labels
    )
    return schema


def decision_json_schema(
    *,
    episode_slug: str,
    investigation_sha256: str,
    rationale_ids: set[str],
    check_tiers: set[str],
) -> dict:
    """Return a strict schema bound to the frozen Phase 1 artifact."""
    return _bound_decision_json_schema(
        Decision,
        episode_slug=episode_slug,
        investigation_sha256=investigation_sha256,
        rationale_ids=rationale_ids,
        check_tiers=check_tiers,
    )


def decision_v2_json_schema(
    *,
    episode_slug: str,
    investigation_sha256: str,
    rationale_ids: set[str],
    check_tiers: set[str],
) -> dict:
    """Return the v2 decision schema bound to the complete frozen investigation."""
    return _bound_decision_json_schema(
        DecisionV2,
        episode_slug=episode_slug,
        investigation_sha256=investigation_sha256,
        rationale_ids=rationale_ids,
        check_tiers=check_tiers,
    )


def decision_v3_json_schema(
    *,
    episode_slug: str,
    investigation_sha256: str,
    rationale_ids: set[str],
    check_tiers: set[str],
    exact_historical_ids: set[str],
    available_precedent_slugs: set[str],
) -> dict:
    """Bind v3 synthesis to the frozen rationale record and opened precedents."""
    schema = _bound_decision_json_schema(
        DecisionV3,
        episode_slug=episode_slug,
        investigation_sha256=investigation_sha256,
        rationale_ids=rationale_ids,
        check_tiers=check_tiers,
    )
    precedent = schema["$defs"]["DecisivePrecedent"]["properties"]
    slugs = available_precedent_slugs - {episode_slug}
    if not slugs or not exact_historical_ids:
        schema["properties"]["decisive_precedents"]["maxItems"] = 0
    else:
        precedent["episode_slug"]["enum"] = sorted(slugs)
        precedent["historical_evidence_ids"]["items"] = {
            "type": "string",
            "enum": sorted(exact_historical_ids),
        }
    return schema


def _bound_decision_json_schema(
    model: type[Decision],
    *,
    episode_slug: str,
    investigation_sha256: str,
    rationale_ids: set[str],
    check_tiers: set[str],
) -> dict:
    schema = deepcopy(model.model_json_schema())
    properties = schema["properties"]
    properties["episode_slug"]["const"] = episode_slug
    properties["investigation_sha256"]["const"] = investigation_sha256
    properties["check_tier"]["enum"] = sorted(check_tiers)
    properties["recommended_check_tier"]["enum"] = sorted(check_tiers)
    properties["controlling_rationale_ids"]["items"] = {
        "type": "string",
        "enum": sorted(rationale_ids),
    }
    assessment = schema["$defs"]["RationaleAssessment"]["properties"]
    assessment["rationale_id"]["enum"] = sorted(rationale_ids)
    step = schema["$defs"]["DeliberationStep"]["properties"]
    step["rationale_ids"]["items"] = {
        "type": "string",
        "enum": sorted(rationale_ids),
    }
    step["check_tier_implication"]["enum"] = sorted(check_tiers)
    for definition, fields in {
        "AnyCheckDecision": ("supporting_rationale_ids", "opposing_rationale_ids"),
        "StandardCheckDecision": (
            "supporting_rationale_ids",
            "opposing_rationale_ids",
            "failure_rationale_ids",
        ),
    }.items():
        nested = schema["$defs"][definition]["properties"]
        for field in fields:
            nested[field]["items"] = {
                "type": "string",
                "enum": sorted(rationale_ids),
            }
    schema["$defs"]["Counterargument"]["properties"]["rationale_ids"]["items"] = {
        "type": "string",
        "enum": sorted(rationale_ids),
    }
    schema["$defs"]["MarketGate"]["properties"]["controlling_rationale_ids"][
        "items"
    ] = {"type": "string", "enum": sorted(rationale_ids)}
    schema["$defs"]["RiskLedgerEntry"]["properties"]["rationale_id"]["enum"] = sorted(
        rationale_ids
    )
    return schema


def validate_investigation(
    candidate: Investigation,
    *,
    episode_slug: str,
    pitch: str,
    taxonomy_labels: set[str],
    exact_wiki_ids: set[str],
) -> None:
    if candidate.episode_slug != episode_slug:
        raise ValueError("investigation episode slug does not match")
    rationale_ids = [row.rationale_id for row in candidate.rationales]
    if len(rationale_ids) != len(set(rationale_ids)):
        raise ValueError("rationale IDs must be unique")
    for rationale in candidate.rationales:
        if rationale.label not in taxonomy_labels:
            raise ValueError(f"unknown taxonomy label: {rationale.label}")
        if any(not _pitch_supports(evidence, pitch) for evidence in rationale.pitch_evidence):
            raise ValueError(f"pitch evidence is not present: {rationale.rationale_id}")
        if not set(rationale.wiki_evidence_ids) <= exact_wiki_ids:
            raise ValueError(f"wiki evidence was not read exactly: {rationale.rationale_id}")


def validate_investigation_v2(
    candidate: InvestigationV2,
    *,
    episode_slug: str,
    pitch: str,
    taxonomy_labels: set[str],
    exact_wiki_ids: set[str],
) -> None:
    """Validate the v2 investigation and every pitch-derived deal fact."""
    validate_investigation(
        candidate,
        episode_slug=episode_slug,
        pitch=pitch,
        taxonomy_labels=taxonomy_labels,
        exact_wiki_ids=exact_wiki_ids,
    )
    for field_name in DealContext.model_fields:
        fact = getattr(candidate.deal_context, field_name)
        if any(
            not _pitch_supports(evidence, pitch) for evidence in fact.pitch_evidence
        ):
            raise ValueError(f"deal fact evidence is not present: {field_name}")


def validate_investigation_v3(
    candidate: InvestigationV3,
    *,
    episode_slug: str,
    pitch: str,
    taxonomy_labels: set[str],
    exact_wiki_ids: set[str],
    exact_historical_ids: set[str],
) -> None:
    """Validate v3 evidence provenance and the finalized candidate funnel."""
    validate_investigation_v2(
        candidate,
        episode_slug=episode_slug,
        pitch=pitch,
        taxonomy_labels=taxonomy_labels,
        exact_wiki_ids=exact_wiki_ids,
    )
    for rationale in candidate.rationales:
        if not rationale.wiki_evidence_ids and not rationale.historical_evidence_ids:
            raise ValueError(
                f"rationale requires wiki or historical evidence: {rationale.rationale_id}"
            )
        if not set(rationale.historical_evidence_ids) <= exact_historical_ids:
            raise ValueError(
                f"historical evidence was not opened exactly: {rationale.rationale_id}"
            )

    for field_name in (
        "activated_candidates",
        "queried_candidates",
        "rejected_candidates",
    ):
        unknown = set(getattr(candidate, field_name)) - taxonomy_labels
        if unknown:
            raise ValueError(
                f"unknown taxonomy candidate in {field_name}: {sorted(unknown)[0]}"
            )

    invalid_unmapped = set(candidate.unmapped_candidates) & taxonomy_labels
    if invalid_unmapped:
        raise ValueError(
            "unmapped candidate matches a taxonomy label: "
            f"{sorted(invalid_unmapped)[0]}"
        )

    rationale_labels = {rationale.label for rationale in candidate.rationales}
    activated = set(candidate.activated_candidates)
    queried = set(candidate.queried_candidates)
    rejected = set(candidate.rejected_candidates)
    if activated != rationale_labels:
        raise ValueError("activated candidates must equal rationale labels")
    if not activated <= queried:
        raise ValueError("activated candidates must be queried")
    if not rejected <= queried:
        raise ValueError("rejected candidates must be queried")

    finalized = {
        "activated_candidates": activated,
        "rejected_candidates": rejected,
        "unmapped_candidates": set(candidate.unmapped_candidates),
    }
    finalized_names = tuple(finalized)
    for index, left_name in enumerate(finalized_names):
        for right_name in finalized_names[index + 1 :]:
            overlap = finalized[left_name] & finalized[right_name]
            if overlap:
                raise ValueError(
                    "candidate appears in mutually exclusive finalized buckets: "
                    f"{sorted(overlap)[0]}"
                )
    if rejected != queried - activated:
        raise ValueError(
            "rejected candidates must equal queried candidates minus activated candidates"
        )
    dispositions = {row.label: row.disposition for row in candidate.taxonomy_dispositions}
    if set(dispositions) != taxonomy_labels:
        raise ValueError("taxonomy dispositions must cover all taxonomy labels exactly once")
    if queried != taxonomy_labels:
        raise ValueError("queried candidates must include all taxonomy labels")
    disposition_activated = {
        label for label, disposition in dispositions.items() if disposition == "activated"
    }
    disposition_rejected = {
        label for label, disposition in dispositions.items() if disposition == "rejected"
    }
    if disposition_activated != activated or disposition_rejected != rejected:
        raise ValueError(
            "taxonomy dispositions must match activated and rejected candidates"
        )


def validate_decision(
    candidate: Decision,
    episode_slug: str,
    investigation_sha256: str,
    rationale_ids: set[str],
    check_tiers: set[str],
    *,
    allow_missing_final_consistency: bool = False,
) -> None:
    if candidate.episode_slug != episode_slug:
        raise ValueError("decision episode slug does not match")
    if candidate.investigation_sha256 != investigation_sha256:
        raise ValueError("decision investigation hash does not match")
    reference_lists = [
        candidate.controlling_rationale_ids,
        candidate.any_check.supporting_rationale_ids,
        candidate.any_check.opposing_rationale_ids,
        candidate.any_check.strongest_counterargument.rationale_ids,
        candidate.standard_check.supporting_rationale_ids,
        candidate.standard_check.opposing_rationale_ids,
        candidate.standard_check.failure_rationale_ids,
        candidate.standard_check.market_gate.controlling_rationale_ids,
        candidate.standard_check.strongest_counterargument.rationale_ids,
        [row.rationale_id for row in candidate.risk_ledger],
        *[step.rationale_ids for step in candidate.deliberation_steps],
    ]
    if any(not set(references) <= rationale_ids for references in reference_lists):
        raise ValueError("decision references an unknown rationale")
    if any(len(references) != len(set(references)) for references in reference_lists):
        raise ValueError("decision rationale references must not contain duplicates")
    if candidate.check_tier not in check_tiers or candidate.recommended_check_tier not in check_tiers:
        raise ValueError("decision check tier is invalid")
    assessed_ids = [row.rationale_id for row in candidate.rationale_assessments]
    if len(assessed_ids) != len(set(assessed_ids)) or set(assessed_ids) != rationale_ids:
        raise ValueError("every frozen rationale must be assessed exactly once")
    if candidate.standard_check.decision == "In" and candidate.any_check.decision == "Out":
        raise ValueError("standard-check In requires any-check In")
    if (
        candidate.standard_check.decision == "In"
        and candidate.standard_check.market_gate.status != "positive"
    ):
        raise ValueError("standard-check In requires a positive market gate")
    if candidate.standard_check.likelihood > candidate.any_check.likelihood:
        raise ValueError("standard-check likelihood cannot exceed any-check likelihood")
    tracks: dict[str, list[DeliberationStep]] = {"any_check": [], "standard_check": []}
    for index, step in enumerate(candidate.deliberation_steps, start=1):
        if step.step_id != f"D{index}":
            raise ValueError("deliberation step IDs must be sequential")
        if step.check_tier_implication not in check_tiers:
            raise ValueError("deliberation step check tier is invalid")
        movement = step.likelihood_after - step.likelihood_before
        expected = "raises" if movement > 1e-9 else "lowers" if movement < -1e-9 else "no_change"
        if step.effect != expected:
            raise ValueError("deliberation effect conflicts with likelihood movement")
        tracks[step.endpoint].append(step)
    endpoint_likelihoods = {
        "any_check": candidate.any_check.likelihood,
        "standard_check": candidate.standard_check.likelihood,
    }
    for endpoint, steps in tracks.items():
        if not any(step.stage == "opposing_case" for step in steps):
            raise ValueError(f"{endpoint} deliberation requires an opposing-case step")
        if (
            not allow_missing_final_consistency
            and (not steps or steps[-1].stage != "consistency")
        ):
            raise ValueError(f"{endpoint} deliberation must finish with a consistency step")
        for previous, current in zip(steps, steps[1:]):
            if abs(current.likelihood_before - previous.likelihood_after) > 1e-9:
                raise ValueError("deliberation likelihood continuity is broken")
        if abs(steps[-1].likelihood_after - endpoint_likelihoods[endpoint]) > 1e-9:
            raise ValueError(f"final {endpoint} likelihood does not match its endpoint")
    any_track_ids = {
        rationale_id for step in tracks["any_check"] for rationale_id in step.rationale_ids
    }
    controlling_risk_ids = {
        row.rationale_id for row in candidate.risk_ledger if row.controlling_for_any_check
    }
    if not controlling_risk_ids <= any_track_ids:
        raise ValueError("any-check controlling risks must appear in the any-check track")
    standard_track_ids = {
        rationale_id
        for step in tracks["standard_check"]
        for rationale_id in step.rationale_ids
    }
    if not set(candidate.standard_check.market_gate.controlling_rationale_ids) <= standard_track_ids:
        raise ValueError(
            "market-gate controlling rationales must appear in the standard-check track"
        )
    used_ids = {rationale_id for step in candidate.deliberation_steps for rationale_id in step.rationale_ids}
    if not set(candidate.controlling_rationale_ids) <= used_ids:
        raise ValueError("controlling rationales must appear in deliberation steps")
    for name, endpoint in (("any-check", candidate.any_check), ("standard-check", candidate.standard_check)):
        if endpoint.decision == "In" and endpoint.likelihood < 0.5:
            raise ValueError(f"{name} In conflicts with likelihood")
        if endpoint.decision == "Out" and endpoint.likelihood >= 0.5:
            raise ValueError(f"{name} Out conflicts with likelihood")
    fatal_present = any(row.risk_type == "fatal_constraint" for row in candidate.risk_ledger)
    if candidate.any_check.fatal_constraint_present != fatal_present:
        raise ValueError("fatal constraint flag must match the risk ledger")
    exploratory_tiers = {tier for tier in check_tiers if "exploratory" in tier}
    standard_tiers = check_tiers - {"no_check_tier"} - exploratory_tiers
    if "no_check_tier" not in check_tiers or len(exploratory_tiers) != 1 or not standard_tiers:
        raise ValueError("check-tier vocabulary must define no-check, exploratory, and standard tiers")
    if candidate.any_check.decision == "Out":
        expected_tiers = {"no_check_tier"}
    elif candidate.standard_check.decision == "Out":
        expected_tiers = exploratory_tiers
    else:
        expected_tiers = standard_tiers
    if candidate.recommended_check_tier not in expected_tiers:
        raise ValueError("recommended check tier conflicts with dual-check decisions")
    if (
        candidate.decision != candidate.any_check.decision
        or abs(candidate.investment_likelihood - candidate.any_check.likelihood) > 1e-9
        or abs(candidate.decision_confidence - candidate.any_check.confidence) > 1e-9
        or candidate.check_tier != candidate.recommended_check_tier
    ):
        raise ValueError("compatibility summary must match any-check and recommended tier")


def validate_decision_v2(
    candidate: DecisionV2,
    investigation: InvestigationV2,
    investigation_sha256: str,
    check_tiers: set[str],
    *,
    allow_missing_final_consistency: bool = False,
) -> None:
    """Validate a decision without allowing Phase 2 to invent stronger evidence."""
    rationale_by_id = {
        rationale.rationale_id: rationale for rationale in investigation.rationales
    }
    validate_decision(
        candidate,
        investigation.episode_slug,
        investigation_sha256,
        set(rationale_by_id),
        check_tiers,
        allow_missing_final_consistency=allow_missing_final_consistency,
    )
    for risk in candidate.risk_ledger:
        source = rationale_by_id[risk.rationale_id]
        if risk.risk_type == "fatal_constraint":
            if source.constraint_kind == "total_round_size":
                raise ValueError("total round size cannot become fatal in Phase 2")
            if not (
                source.evidence_status == "affirmative_adverse"
                and source.constraint_severity == "fatal"
            ):
                raise ValueError(
                    "fatal risk requires affirmative adverse fatal provenance"
                )
        if (
            risk.risk_type == "affirmative_adverse"
            and source.evidence_status != "affirmative_adverse"
        ):
            raise ValueError(
                "affirmative adverse risk requires affirmative adverse evidence"
            )
    for assessment in candidate.rationale_assessments:
        source = rationale_by_id[assessment.rationale_id]
        if (
            assessment.effective_direction == "negative"
            and source.evidence_status != "affirmative_adverse"
        ):
            raise ValueError(
                "negative assessment requires affirmative adverse Phase 1 evidence"
            )


def validate_decision_v3(
    candidate: DecisionV3,
    investigation: InvestigationV3,
    investigation_sha256: str,
    check_tiers: set[str],
    *,
    historical_evidence: list[dict],
    precedent_reads: list[dict],
    target_episode_slug: str,
    allow_missing_final_consistency: bool = False,
) -> None:
    """Validate source-local precedent citations and retain all v2 endpoint rules."""
    validate_decision_v2(
        candidate,
        investigation,
        investigation_sha256,
        check_tiers,
        allow_missing_final_consistency=allow_missing_final_consistency,
    )
    evidence_by_id = {
        row.get("evidence_id"): row
        for row in historical_evidence
        if isinstance(row, dict) and row.get("evidence_id")
    }
    observed_by_slug: dict[str, str] = {}
    for read in precedent_reads:
        if not isinstance(read, dict) or read.get("status") != "ok":
            continue
        decision = read.get("decision")
        if read.get("read_type") == "decision" and isinstance(decision, dict):
            status = decision.get("status")
            if status in {"In", "Out", "unobserved"}:
                observed_by_slug[str(read.get("episode_slug"))] = status
    for precedent in candidate.decisive_precedents:
        if precedent.episode_slug == target_episode_slug:
            raise ValueError("target episode cannot be a decisive precedent")
        actual = observed_by_slug.get(precedent.episode_slug)
        if actual is None:
            raise ValueError("decisive precedent requires an opened decision read")
        if precedent.observed_decision != actual:
            raise ValueError("decisive precedent observed decision does not match opened read")
        for evidence_id in precedent.historical_evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                raise ValueError("decisive precedent cites unopened historical evidence")
            if evidence.get("episode_slug") != precedent.episode_slug:
                raise ValueError("decisive precedent evidence must come from the same episode")
def decision_quality_findings(candidate: Decision) -> list[str]:
    """Return semantic concerns that warrant reconsideration without discarding output."""
    findings: list[str] = []
    ledger_ids = [row.rationale_id for row in candidate.risk_ledger]
    if len(ledger_ids) != len(set(ledger_ids)) or set(ledger_ids) != set(
        candidate.any_check.opposing_rationale_ids
    ):
        findings.append("RISK_LEDGER_COVERAGE_MISMATCH")
    fatal_present = any(
        row.risk_type == "fatal_constraint" for row in candidate.risk_ledger
    )
    if fatal_present and candidate.any_check.decision == "In":
        findings.append("ANY_CHECK_IN_WITH_MATERIAL_CONSTRAINT")
    return findings
