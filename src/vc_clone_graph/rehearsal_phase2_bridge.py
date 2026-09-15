"""Bridge founder-answer overlays into one canonical Phase 2 synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol, Sequence

from .rehearsal_artifacts import RehearsalArtifactStore
from .rehearsal_grounding import CanonicalRehearsalBaseline
from .rehearsal_schemas import (
    FinalAssessment,
    FounderAnswerEvidence,
    GroundedRationaleChange,
    RationaleState,
    normalize_final_score_reconciliation,
)
from .schemas_v4 import DecisionV4, DecisionV41
from .workflow_v4 import Phase2RehearsalContext


@dataclass(frozen=True)
class GroundedSynthesisResult:
    status: str
    decision: DecisionV4 | DecisionV41 | None
    final_assessment: FinalAssessment | None
    artifact_path: str | None
    artifact_sha256: str | None
    usage: dict[str, int | float]
    failure_reason: str | None = None


class GroundedPhase2Synthesizer(Protocol):
    def synthesize(
        self,
        *,
        baseline: CanonicalRehearsalBaseline,
        answers: Sequence[FounderAnswerEvidence],
        changes: Sequence[GroundedRationaleChange],
        current_rationales: Sequence[RationaleState],
        conversational_likelihood: float,
    ) -> GroundedSynthesisResult: ...


class _CanonicalWorkflow(Protocol):
    def invoke_phase2(self, thread_id, investigation, investigation_sha256, **kwargs): ...


def _final_assessment(
    decision: DecisionV4 | DecisionV41,
    rationales: Sequence[RationaleState],
    *,
    conversational_likelihood: float,
) -> FinalAssessment:
    by_id = {row.rationale_id: row for row in rationales}
    positive = tuple(
        identifier
        for identifier in decision.controlling_rationale_ids
        if identifier in by_id and by_id[identifier].direction == "positive"
    )
    concerns = tuple(
        identifier
        for identifier in decision.controlling_rationale_ids
        if identifier not in positive
    )
    assessment = FinalAssessment(
        decision=decision.decision,
        investment_likelihood=decision.investment_likelihood,
        decision_confidence=decision.decision_confidence,
        strongest_positive_rationale_ids=positive,
        decisive_concern_rationale_ids=concerns,
        unresolved_uncertainties=tuple(
            dict.fromkeys((*decision.searchable_questions, *decision.diligence_questions))
        ),
        reversal_conditions=tuple(decision.reversal_conditions),
        rationale_state=tuple(rationales),
        decision_justification=decision.decision_justification,
    )
    return normalize_final_score_reconciliation(
        assessment,
        conversational_likelihood=conversational_likelihood,
    )[0]


class CanonicalPhase2Bridge:
    def __init__(
        self,
        *,
        workflow: _CanonicalWorkflow,
        store: RehearsalArtifactStore,
    ) -> None:
        self.workflow = workflow
        self.store = store

    def synthesize(
        self,
        *,
        baseline: CanonicalRehearsalBaseline,
        answers: Sequence[FounderAnswerEvidence],
        changes: Sequence[GroundedRationaleChange],
        current_rationales: Sequence[RationaleState],
        conversational_likelihood: float,
    ) -> GroundedSynthesisResult:
        if not changes:
            decision = baseline.decision
            relative = "grounding/final-phase2/decision.json"
            path = self.store.write_accepted(
                relative, decision.model_dump(mode="json")
            )
            return GroundedSynthesisResult(
                status=baseline.phase2_status,
                decision=decision,
                final_assessment=_final_assessment(
                    decision,
                    current_rationales,
                    conversational_likelihood=conversational_likelihood,
                ),
                artifact_path=relative,
                artifact_sha256=sha256(path.read_bytes()).hexdigest(),
                usage={},
            )
        context = Phase2RehearsalContext(
            baseline_decision_sha256=baseline.decision_sha256,
            baseline_decision=baseline.decision.decision,
            baseline_investment_likelihood=baseline.decision.investment_likelihood,
            baseline_decision_confidence=baseline.decision.decision_confidence,
            baseline_decision_justification=baseline.decision.decision_justification,
            founder_answers=[
                row.model_dump(mode="json", exclude_none=True) for row in answers
            ],
            rationale_changes=[row.model_dump(mode="json") for row in changes],
            current_rationale_state=[
                row.model_dump(mode="json") for row in current_rationales
            ],
        )
        try:
            state = self.workflow.invoke_phase2(
                f"rehearsal-{self.store.session_root.name}",
                baseline.investigation,
                baseline.investigation_sha256,
                phase1_status=baseline.phase1_status,
                phase1_state=baseline.phase1_state,
                rehearsal_context=context,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self.store.write_accepted(
                "grounding/final-phase2/failure.json", {"status": "failed", "reason": reason}
            )
            return GroundedSynthesisResult(
                status="failed",
                decision=None,
                final_assessment=None,
                artifact_path=None,
                artifact_sha256=None,
                usage={},
                failure_reason=reason,
            )
        status = str(state.get("phase2_status", "failed"))
        decision_payload = state.get("decision")
        if status not in {"accepted", "provisional"} or not decision_payload:
            return GroundedSynthesisResult(
                status="failed",
                decision=None,
                final_assessment=None,
                artifact_path=None,
                artifact_sha256=None,
                usage=dict(state.get("usage_by_phase", {}).get("phase2", {})),
                failure_reason="canonical Phase 2 did not return a usable decision",
            )
        model = DecisionV41 if baseline.contract_version == "v4.1" else DecisionV4
        decision = model.model_validate(decision_payload)
        favorable_only = bool(changes) and all(
            (
                change.change == "strengthened"
                and change.after_direction == "positive"
            )
            or (
                change.change == "weakened"
                and change.before_direction == "negative"
            )
            or (
                change.change == "redirected"
                and change.before_direction == "negative"
                and change.after_direction in {"neutral", "positive"}
            )
            for change in changes
        )
        unfavorable_only = bool(changes) and all(
            (
                change.change == "strengthened"
                and change.after_direction == "negative"
            )
            or (
                change.change == "weakened"
                and change.before_direction == "positive"
            )
            or (
                change.change == "redirected"
                and change.before_direction == "positive"
                and change.after_direction in {"neutral", "negative"}
            )
            for change in changes
        )
        inconsistent = (
            favorable_only
            and decision.investment_likelihood < baseline.decision.investment_likelihood
        ) or (
            unfavorable_only
            and decision.investment_likelihood > baseline.decision.investment_likelihood
        )
        if inconsistent:
            self.store.write_accepted(
                "grounding/final-phase2/inconsistent-candidate.json",
                {
                    "reason": "likelihood movement contradicts answer-linked changes",
                    "candidate": decision.model_dump(mode="json"),
                },
            )
            decision = baseline.decision
            status = "provisional"
        relative = "grounding/final-phase2/decision.json"
        path = self.store.write_accepted(relative, decision.model_dump(mode="json"))
        digest = sha256(path.read_bytes()).hexdigest()
        final = _final_assessment(
            decision,
            current_rationales,
            conversational_likelihood=conversational_likelihood,
        )
        return GroundedSynthesisResult(
            status=status,
            decision=decision,
            final_assessment=final,
            artifact_path=relative,
            artifact_sha256=digest,
            usage=dict(state.get("usage_by_phase", {}).get("phase2", state.get("usage", {}))),
        )
