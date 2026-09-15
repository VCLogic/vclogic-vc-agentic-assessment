from pathlib import Path

import pytest

from vc_clone_graph.rehearsal_artifacts import RehearsalArtifactStore
from vc_clone_graph.rehearsal_grounding import (
    baseline_initial_assessment,
    load_canonical_baseline,
)
from vc_clone_graph.rehearsal_phase2_bridge import CanonicalPhase2Bridge
from vc_clone_graph.rehearsal_schemas import (
    FounderAnswerEvidence,
    GroundedRationaleChange,
)


class FakeCanonicalWorkflow:
    def __init__(self, decision: dict, *, status: str = "accepted") -> None:
        self.decision = decision
        self.status = status
        self.invoke_count = 0
        self.rehearsal_context = None

    def invoke_phase2(self, thread_id, investigation, investigation_sha256, **kwargs):
        self.invoke_count += 1
        self.rehearsal_context = kwargs["rehearsal_context"]
        return {
            "phase2_status": self.status,
            "decision": self.decision,
            "usage": {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01},
        }


def _baseline():
    workspace = Path(__file__).resolve().parents[1]
    if not (workspace / "outputs/canonical-v4-v41-portfolio-2026-08-15/investors/charles-hudson/39-this-pitch-is-damn-near-perfect" / "summary.json").is_file():
        pytest.skip("requires archived canonical assessment outputs")
    return load_canonical_baseline(
        workspace=workspace,
        registry_path=workspace
        / "evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json",
        canonical_vc_slug="charles-hudson",
        episode_slug="39-this-pitch-is-damn-near-perfect",
    )


def test_bridge_invokes_phase2_once_with_grounded_overlay(tmp_path: Path) -> None:
    baseline = _baseline()
    decision = baseline.decision.model_dump(mode="json")
    workflow = FakeCanonicalWorkflow(decision)
    store = RehearsalArtifactStore.create(
        tmp_path, "charles-hudson", "bridge-test", "Founder pitch."
    )
    bridge = CanonicalPhase2Bridge(workflow=workflow, store=store)
    answer = FounderAnswerEvidence(
        answer_id="A-001",
        question_id="Q-001",
        text_verbatim="Retention is 72%.",
        received_at="2026-08-24T00:00:00Z",
        submission_sha256="a" * 64,
    )
    change = GroundedRationaleChange(
        rationale_id="R1",
        taxonomy_label=baseline.investigation.rationales[0].taxonomy_label,
        change="strengthened",
        before_direction="positive",
        after_direction="positive",
        answer_ids=("A-001",),
        justification="The founder supplied retention evidence.",
    )

    result = bridge.synthesize(
        baseline=baseline,
        answers=(answer,),
        changes=(change,),
        current_rationales=baseline_initial_assessment(baseline).rationales,
        conversational_likelihood=0.68,
    )

    assert workflow.invoke_count == 1
    assert workflow.rehearsal_context.baseline_decision_sha256 == baseline.decision_sha256
    assert workflow.rehearsal_context.founder_answers[0]["submission_sha256"] == "a" * 64
    assert result.status == "accepted"
    assert result.final_assessment.decision == baseline.decision.decision
    assert result.final_assessment.score_reconciliation.before == 0.68
    assert result.final_assessment.score_reconciliation.after == baseline.decision.investment_likelihood
    assert store.read_json("grounding/final-phase2/decision.json")["schema_version"] == "decision-v4"


def test_unrecoverable_phase2_failure_never_returns_decision(tmp_path: Path) -> None:
    baseline = _baseline()
    workflow = FakeCanonicalWorkflow({}, status="failed")
    store = RehearsalArtifactStore.create(
        tmp_path, "charles-hudson", "bridge-failure", "Founder pitch."
    )
    result = CanonicalPhase2Bridge(workflow=workflow, store=store).synthesize(
        baseline=baseline,
        answers=(FounderAnswerEvidence(
            answer_id="A-001",
            question_id="Q-001",
            text_verbatim="Retention is unknown.",
            received_at="2026-08-24T00:00:00Z",
        ),),
        changes=(GroundedRationaleChange(
            rationale_id="R1",
            taxonomy_label=baseline.investigation.rationales[0].taxonomy_label,
            change="weakened",
            before_direction="positive",
            after_direction="neutral",
            answer_ids=("A-001",),
            justification="The answer did not substantiate retention.",
        ),),
        current_rationales=baseline_initial_assessment(baseline).rationales,
        conversational_likelihood=0.63,
    )

    assert workflow.invoke_count == 1
    assert result.status == "failed"
    assert result.decision is None
    assert result.final_assessment is None


def test_no_material_change_preserves_baseline_without_phase2_call(
    tmp_path: Path,
) -> None:
    baseline = _baseline()
    workflow = FakeCanonicalWorkflow({}, status="failed")
    store = RehearsalArtifactStore.create(
        tmp_path, "charles-hudson", "bridge-noop", "Founder pitch."
    )

    result = CanonicalPhase2Bridge(workflow=workflow, store=store).synthesize(
        baseline=baseline,
        answers=(),
        changes=(),
        current_rationales=baseline_initial_assessment(baseline).rationales,
        conversational_likelihood=baseline.decision.investment_likelihood,
    )

    assert workflow.invoke_count == 0
    assert result.decision == baseline.decision
    assert result.final_assessment.decision == baseline.decision.decision
    assert result.final_assessment.score_reconciliation.direction == "unchanged"
