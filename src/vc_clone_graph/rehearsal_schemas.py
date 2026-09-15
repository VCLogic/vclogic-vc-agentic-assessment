"""Structured contracts for interactive founder rehearsal sessions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .rehearsal_effects import EvidenceEffect


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceRef(FrozenModel):
    evidence_id: str = Field(min_length=1)
    source_kind: Literal[
        "pitch", "founder_answer", "wiki", "precedent", "portfolio"
    ]
    source_path: str | None = None
    excerpt: str = Field(min_length=1)


class RationaleState(FrozenModel):
    rationale_id: str = Field(min_length=1)
    taxonomy_label: str = Field(min_length=1)
    direction: Literal["positive", "negative", "neutral"]
    salience: Literal["primary", "secondary"]
    confidence: float = Field(ge=0, le=1)
    assessment: str = Field(min_length=1)
    evidence_refs: tuple[EvidenceRef, ...] = Field(min_length=1)


class RehearsalQuestion(FrozenModel):
    question_id: str = Field(pattern=r"^Q-[0-9]{3}$")
    text: str = Field(min_length=1)
    dimension: str = Field(min_length=1)
    rationale_labels: tuple[str, ...] = Field(min_length=1)
    expected_decision_value: float = Field(ge=0, le=1)
    why_now: str = Field(min_length=1)
    response_comment: str | None = Field(default=None, min_length=1)

    def public_payload(self) -> dict[str, str]:
        """Return the interview-safe question without private assessment fields."""
        payload = {
            "question_id": self.question_id,
            "question": self.text,
            "dimension": self.dimension,
        }
        if self.response_comment:
            payload["response_comment"] = self.response_comment
        return payload


class InitialAssessment(FrozenModel):
    decision: Literal["In", "Out"]
    investment_likelihood: float = Field(ge=0, le=1)
    decision_confidence: float = Field(ge=0, le=1)
    rationales: tuple[RationaleState, ...]
    unresolved_questions: tuple[str, ...]
    candidate_questions: tuple[RehearsalQuestion, ...]


class FounderAnswerEvidence(FrozenModel):
    answer_id: str = Field(pattern=r"^A-[0-9]{3}$")
    question_id: str = Field(pattern=r"^Q-[0-9]{3}$")
    text_verbatim: str = Field(min_length=1)
    received_at: str = Field(min_length=1)
    submission_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class AnswerUpdate(FrozenModel):
    answer_id: str = Field(pattern=r"^A-[0-9]{3}$")
    affected_rationale_labels: tuple[str, ...]
    direction: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1)
    resolved_question: bool
    materially_changed_assessment: bool
    investment_likelihood_before: float = Field(ge=0, le=1)
    investment_likelihood_after: float = Field(ge=0, le=1)
    decision_confidence_after: float = Field(ge=0, le=1)
    rationale_state_after: tuple[RationaleState, ...]
    unresolved_questions_after: tuple[str, ...]
    update_summary: str = Field(min_length=1)


class ContinuationDecision(FrozenModel):
    continue_interview: bool
    reason: str = Field(min_length=1)
    next_question_focus: str | None = None

    @model_validator(mode="after")
    def require_focus_when_continuing(self) -> "ContinuationDecision":
        if self.continue_interview and not self.next_question_focus:
            raise ValueError("continuing requires next_question_focus")
        return self


class ScoreReconciliation(FrozenModel):
    before: float = Field(ge=0, le=1)
    after: float = Field(ge=0, le=1)
    direction: Literal["unchanged", "increased", "decreased"]
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def direction_matches_values(self) -> "ScoreReconciliation":
        delta = self.after - self.before
        expected = (
            "unchanged"
            if abs(delta) < 1e-9
            else "increased"
            if delta > 0
            else "decreased"
        )
        if self.direction != expected:
            raise ValueError("reconciliation direction conflicts with its values")
        return self


class FinalAssessment(FrozenModel):
    decision: Literal["In", "Out"]
    investment_likelihood: float = Field(ge=0, le=1)
    decision_confidence: float = Field(ge=0, le=1)
    strongest_positive_rationale_ids: tuple[str, ...]
    decisive_concern_rationale_ids: tuple[str, ...]
    unresolved_uncertainties: tuple[str, ...]
    reversal_conditions: tuple[str, ...]
    rationale_state: tuple[RationaleState, ...]
    decision_justification: str = Field(min_length=1)
    score_reconciliation: ScoreReconciliation

    @model_validator(mode="before")
    @classmethod
    def add_legacy_reconciliation(cls, value: object) -> object:
        if not isinstance(value, dict) or value.get("score_reconciliation") is not None:
            return value
        likelihood = float(value.get("investment_likelihood", 0.0))
        return {
            **value,
            "score_reconciliation": {
                "before": likelihood,
                "after": likelihood,
                "direction": "unchanged",
                "explanation": (
                    "Score reconciliation was unavailable in this legacy assessment."
                ),
            },
        }

    @model_validator(mode="after")
    def reconciliation_matches_final_likelihood(self) -> "FinalAssessment":
        if (
            self.score_reconciliation is not None
            and abs(
                self.score_reconciliation.after - self.investment_likelihood
            ) >= 1e-9
        ):
            raise ValueError("score reconciliation must end at final likelihood")
        return self


def normalize_final_score_reconciliation(
    assessment: FinalAssessment,
    *,
    conversational_likelihood: float,
) -> tuple[FinalAssessment, tuple[str, ...]]:
    """Ensure every new final assessment explains its actual score transition."""
    current = min(1.0, max(0.0, float(conversational_likelihood)))
    existing = assessment.score_reconciliation
    if existing is not None and abs(existing.before - current) < 1e-9:
        return assessment, ()
    final = assessment.investment_likelihood
    delta_points = round(abs(final - current) * 100)
    if abs(final - current) < 1e-9:
        direction: Literal["unchanged", "increased", "decreased"] = "unchanged"
        explanation = (
            "The final synthesis preserved the last conversational likelihood after "
            "balancing the complete rationale state."
        )
    elif final > current:
        direction = "increased"
        explanation = (
            f"The final synthesis increased the likelihood by {delta_points} percentage "
            "points after balancing all answer-linked rationale changes."
        )
    else:
        direction = "decreased"
        explanation = (
            f"The final synthesis decreased the likelihood by {delta_points} percentage "
            "points after balancing all answer-linked rationale changes against the "
            "remaining concerns."
        )
    reconciliation = ScoreReconciliation(
        before=current,
        after=final,
        direction=direction,
        explanation=explanation,
    )
    return (
        assessment.model_copy(update={"score_reconciliation": reconciliation}),
        ("score_reconciliation_normalized",),
    )


class CoachingOutput(FrozenModel):
    material_answer_ids: tuple[str, ...]
    pitch_improvement_suggestions: tuple[str, ...]
    founder_reflection_prompts: tuple[str, ...]


class ClassificationSnapshot(FrozenModel):
    stage: Literal["initial", "updated", "final"]
    probability_in: float = Field(ge=0, le=1)
    predicted_decision: Literal["In", "Out"]


class ClassificationSummary(FrozenModel):
    status: Literal["available", "fallback"]
    model_version: str | None = None
    artifact_id: str | None = None
    snapshots: tuple[ClassificationSnapshot, ...] = ()
    fallback_reason: str | None = None

    @model_validator(mode="after")
    def validate_status_fields(self) -> "ClassificationSummary":
        if self.status == "available" and not self.artifact_id:
            raise ValueError("available classification requires artifact_id")
        if self.status == "fallback" and not self.fallback_reason:
            raise ValueError("fallback classification requires fallback_reason")
        return self


class GroundedArtifactRef(FrozenModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["accepted", "provisional"]


class GroundedBaselineSummary(FrozenModel):
    contract_version: Literal["v4", "v4.1"]
    episode_slug: str = Field(min_length=1)
    phase1: GroundedArtifactRef
    phase2: GroundedArtifactRef
    decision: Literal["In", "Out"]
    investment_likelihood: float = Field(ge=0, le=1)
    decision_confidence: float = Field(ge=0, le=1)
    controlling_rationale_ids: tuple[str, ...] = Field(min_length=1)


class GroundedRationaleChange(FrozenModel):
    rationale_id: str = Field(min_length=1)
    taxonomy_label: str = Field(min_length=1)
    change: Literal["added", "removed", "strengthened", "weakened", "redirected"]
    before_direction: Literal["positive", "negative", "neutral"] | None = None
    after_direction: Literal["positive", "negative", "neutral"] | None = None
    answer_ids: tuple[str, ...] = Field(min_length=1)
    justification: str = Field(min_length=1)


class GroundedAnswerUpdate(AnswerUpdate):
    grounded_changes: tuple[GroundedRationaleChange, ...]
    evidence_effect: EvidenceEffect
    supporting_answer_excerpt: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_current_answer_provenance(self) -> "GroundedAnswerUpdate":
        if self.evidence_effect in {
            "new_positive",
            "new_negative",
            "contradiction",
        } and not self.supporting_answer_excerpt:
            raise ValueError(
                "a material evidence effect requires supporting_answer_excerpt"
            )
        for change in self.grounded_changes:
            if self.answer_id not in change.answer_ids:
                raise ValueError("every grounded change must cite the current answer")
        if self.materially_changed_assessment and not self.grounded_changes:
            raise ValueError("a material grounded update requires a rationale change")
        changed_ids = {change.rationale_id for change in self.grounded_changes}
        updated_ids = {row.rationale_id for row in self.rationale_state_after}
        if changed_ids != updated_ids:
            raise ValueError(
                "grounded rationale_state_after must contain exactly the changed rationales"
            )
        updated_by_id = {row.rationale_id: row for row in self.rationale_state_after}
        for change in self.grounded_changes:
            if change.change == "added":
                expected_id = (
                    f"AR-{self.answer_id}-{change.taxonomy_label}"
                )
                if change.rationale_id != expected_id:
                    raise ValueError(
                        "an answer-activated rationale must use the deterministic "
                        "rationale_id AR-{answer_id}-{taxonomy_label}"
                    )
            rationale = updated_by_id[change.rationale_id]
            cites_current_answer = any(
                evidence.source_kind == "founder_answer"
                and evidence.evidence_id == self.answer_id
                for evidence in rationale.evidence_refs
            )
            if not cites_current_answer:
                raise ValueError(
                    "every changed rationale must cite current founder-answer evidence"
                )
        return self


def normalize_grounded_no_change_update(
    update: GroundedAnswerUpdate,
    *,
    current_likelihood: float,
    current_confidence: float,
) -> tuple[GroundedAnswerUpdate, tuple[str, ...]]:
    """Make an empty answer-linked overlay a deterministic zero-delta update."""
    if update.grounded_changes:
        return update, ()
    differs = (
        update.materially_changed_assessment
        or update.investment_likelihood_before != current_likelihood
        or update.investment_likelihood_after != current_likelihood
        or update.decision_confidence_after != current_confidence
    )
    if not differs:
        return update, ()
    return (
        update.model_copy(
            update={
                "materially_changed_assessment": False,
                "investment_likelihood_before": current_likelihood,
                "investment_likelihood_after": current_likelihood,
                "decision_confidence_after": current_confidence,
            }
        ),
        ("no_grounded_changes_preserved_assessment",),
    )


def validate_grounded_update_context(
    update: GroundedAnswerUpdate,
    previous_rationale_ids: Iterable[str],
) -> None:
    """Ensure an answer-linked overlay changes only an existing rationale."""
    allowed = set(previous_rationale_ids) | {
        rationale.rationale_id for rationale in update.rationale_state_after
    }
    unknown = sorted(
        change.rationale_id
        for change in update.grounded_changes
        if change.rationale_id not in allowed
    )
    if unknown:
        raise ValueError("grounded changes cite unknown rationales: " + ", ".join(unknown))


class GroundedDecisionComparison(FrozenModel):
    baseline: GroundedBaselineSummary
    final_contract_version: Literal["v4", "v4.1"]
    final_decision: Literal["In", "Out"]
    final_investment_likelihood: float = Field(ge=0, le=1)
    final_decision_confidence: float = Field(ge=0, le=1)
    decision_changed: bool
    rationale_changes: tuple[GroundedRationaleChange, ...]
    final_artifact_path: str = Field(min_length=1)
    final_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FounderReport(FrozenModel):
    session_id: str = Field(min_length=1)
    vc_slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    disclosure: str = Field(min_length=1)
    initial_assessment: InitialAssessment
    final_assessment: FinalAssessment
    material_answer_ids: tuple[str, ...]
    pitch_improvement_suggestions: tuple[str, ...]
    founder_reflection_prompts: tuple[str, ...] = ()
    usage: dict[str, int | float]
    classification: ClassificationSummary | None = None
    grounded: GroundedDecisionComparison | None = None

    @model_validator(mode="after")
    def require_simulation_disclosure(self) -> "FounderReport":
        if "simulation" not in self.disclosure.casefold():
            raise ValueError("report disclosure must identify the output as a simulation")
        return self


def validate_taxonomy_labels(labels: Iterable[str], allowed: set[str]) -> None:
    """Reject model-produced rationale labels outside the configured taxonomy."""
    unknown = sorted(set(labels) - allowed)
    if unknown:
        raise ValueError("unknown rationale labels: " + ", ".join(unknown))


def validate_rationale_state(
    rationales: Iterable[RationaleState], allowed: set[str]
) -> None:
    validate_taxonomy_labels(
        (rationale.taxonomy_label for rationale in rationales), allowed
    )
