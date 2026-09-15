import pytest
from pydantic import ValidationError

from vc_clone_graph.rehearsal_schemas import (
    ClassificationSnapshot,
    ClassificationSummary,
    EvidenceRef,
    FinalAssessment,
    FounderReport,
    GroundedArtifactRef,
    GroundedAnswerUpdate,
    GroundedBaselineSummary,
    GroundedDecisionComparison,
    GroundedRationaleChange,
    InitialAssessment,
    RehearsalQuestion,
    RationaleState,
    ScoreReconciliation,
    normalize_final_score_reconciliation,
    normalize_grounded_no_change_update,
    validate_taxonomy_labels,
)


def _evidence() -> EvidenceRef:
    return EvidenceRef(
        evidence_id="P-001",
        source_kind="pitch",
        excerpt="Five paid pilots.",
    )


def _rationale() -> RationaleState:
    return RationaleState(
        rationale_id="R-001",
        taxonomy_label="founder_execution",
        direction="positive",
        salience="primary",
        confidence=0.8,
        assessment="The pilots show execution.",
        evidence_refs=(_evidence(),),
    )


def _initial() -> InitialAssessment:
    return InitialAssessment(
        decision="In",
        investment_likelihood=0.65,
        decision_confidence=0.6,
        rationales=(_rationale(),),
        unresolved_questions=("Retention is unknown.",),
        candidate_questions=(),
    )


def _final() -> FinalAssessment:
    return FinalAssessment(
        decision="In",
        investment_likelihood=0.72,
        decision_confidence=0.7,
        strongest_positive_rationale_ids=("R-001",),
        decisive_concern_rationale_ids=(),
        unresolved_uncertainties=(),
        reversal_conditions=("Pilot demand fails to convert.",),
        rationale_state=(_rationale(),),
        decision_justification="Execution evidence clears the simulated pitch-stage bar.",
    )


def test_score_reconciliation_rejects_direction_that_conflicts_with_values() -> None:
    with pytest.raises(ValidationError, match="reconciliation direction"):
        ScoreReconciliation(
            before=0.61,
            after=0.64,
            direction="decreased",
            explanation="The final synthesis reduced the score.",
        )


def test_missing_score_reconciliation_is_normalized_from_actual_delta() -> None:
    assessment, findings = normalize_final_score_reconciliation(
        _final(), conversational_likelihood=0.68
    )

    assert assessment.score_reconciliation is not None
    assert assessment.score_reconciliation.before == 0.68
    assert assessment.score_reconciliation.after == 0.72
    assert assessment.score_reconciliation.direction == "increased"
    assert "4 percentage points" in assessment.score_reconciliation.explanation
    assert findings == ("score_reconciliation_normalized",)


def test_matching_score_reconciliation_is_preserved() -> None:
    reconciliation = ScoreReconciliation(
        before=0.68,
        after=0.72,
        direction="increased",
        explanation="New retention evidence outweighed the remaining diligence gaps.",
    )
    final = _final().model_copy(update={"score_reconciliation": reconciliation})

    assessment, findings = normalize_final_score_reconciliation(
        final, conversational_likelihood=0.68
    )

    assert assessment.score_reconciliation == reconciliation
    assert findings == ()


def test_no_grounded_changes_cannot_move_likelihood_or_confidence() -> None:
    update = GroundedAnswerUpdate(
        answer_id="A-001",
        affected_rationale_labels=(),
        direction="neutral",
        confidence=0.9,
        resolved_question=False,
        materially_changed_assessment=False,
        investment_likelihood_before=0.28,
        investment_likelihood_after=0.18,
        decision_confidence_after=0.87,
        rationale_state_after=(),
        unresolved_questions_after=("ROI remains unverified.",),
        update_summary="The founder confirmed that ROI is not yet verified.",
        grounded_changes=(),
        evidence_effect="unresolved",
        supporting_answer_excerpt=None,
    )

    normalized, findings = normalize_grounded_no_change_update(
        update,
        current_likelihood=0.28,
        current_confidence=0.8,
    )

    assert normalized.investment_likelihood_before == 0.28
    assert normalized.investment_likelihood_after == 0.28
    assert normalized.decision_confidence_after == 0.8
    assert findings == ("no_grounded_changes_preserved_assessment",)


def test_material_evidence_effect_requires_an_answer_excerpt() -> None:
    payload = {
        "answer_id": "A-001",
        "affected_rationale_labels": ["founder_execution"],
        "direction": "negative",
        "confidence": 0.8,
        "resolved_question": True,
        "materially_changed_assessment": True,
        "investment_likelihood_before": 0.4,
        "investment_likelihood_after": 0.3,
        "decision_confidence_after": 0.7,
        "rationale_state_after": [_rationale().model_dump(mode="json")],
        "unresolved_questions_after": [],
        "update_summary": "New adverse evidence changed the assessment.",
        "evidence_effect": "new_negative",
        "grounded_changes": [
            {
                "rationale_id": "R-001",
                "taxonomy_label": "founder_execution",
                "change": "weakened",
                "before_direction": "positive",
                "after_direction": "negative",
                "answer_ids": ["A-001"],
                "justification": "The answer added adverse execution evidence.",
            }
        ],
    }

    with pytest.raises(ValueError, match="supporting_answer_excerpt"):
        GroundedAnswerUpdate.model_validate(payload)


def test_question_exposes_dimension_but_not_private_assessment() -> None:
    question = RehearsalQuestion(
        question_id="Q-001",
        text="How many users return after the first month?",
        dimension="traction",
        rationale_labels=("traction_customer_validation",),
        expected_decision_value=0.8,
        why_now="Retention is unresolved.",
        response_comment="That retention evidence is useful, but I still need to understand its durability.",
    )

    assert question.public_payload() == {
        "question_id": "Q-001",
        "question": "How many users return after the first month?",
        "dimension": "traction",
        "response_comment": "That retention evidence is useful, but I still need to understand its durability.",
    }


def test_legacy_question_without_comment_remains_valid() -> None:
    question = RehearsalQuestion(
        question_id="Q-001",
        text="How many users return after the first month?",
        dimension="traction",
        rationale_labels=("traction_customer_validation",),
        expected_decision_value=0.8,
        why_now="Retention is unresolved.",
    )

    assert question.response_comment is None
    assert "response_comment" not in question.public_payload()


def test_grounded_update_requires_every_change_to_cite_current_answer() -> None:
    payload = {
        "answer_id": "A-001",
        "affected_rationale_labels": ["founder_execution"],
        "direction": "positive",
        "confidence": 0.8,
        "resolved_question": True,
        "materially_changed_assessment": True,
        "investment_likelihood_before": 0.3,
        "investment_likelihood_after": 0.6,
        "decision_confidence_after": 0.7,
        "rationale_state_after": [_rationale().model_dump(mode="json")],
        "unresolved_questions_after": [],
        "update_summary": "The answer strengthens execution.",
        "evidence_effect": "new_positive",
        "supporting_answer_excerpt": "retention",
        "grounded_changes": [{
            "rationale_id": "R-001",
            "taxonomy_label": "founder_execution",
            "change": "strengthened",
            "before_direction": "negative",
            "after_direction": "positive",
            "answer_ids": ["A-999"],
            "justification": "The answer reports retention.",
        }],
    }

    with pytest.raises(ValueError, match="current answer"):
        GroundedAnswerUpdate.model_validate(payload)


def test_grounded_update_requires_changed_rationale_to_cite_answer_evidence() -> None:
    payload = {
        "answer_id": "A-001",
        "affected_rationale_labels": ["founder_execution"],
        "direction": "positive",
        "confidence": 0.8,
        "resolved_question": True,
        "materially_changed_assessment": True,
        "investment_likelihood_before": 0.3,
        "investment_likelihood_after": 0.6,
        "decision_confidence_after": 0.7,
        "rationale_state_after": [_rationale().model_dump(mode="json")],
        "unresolved_questions_after": [],
        "update_summary": "The answer strengthens execution.",
        "evidence_effect": "new_positive",
        "supporting_answer_excerpt": "Five contracts renewed.",
        "grounded_changes": [{
            "rationale_id": "R-001",
            "taxonomy_label": "founder_execution",
            "change": "strengthened",
            "before_direction": "positive",
            "after_direction": "positive",
            "answer_ids": ["A-001"],
            "justification": "The answer reports completed renewals.",
        }],
    }

    with pytest.raises(ValueError, match="founder-answer evidence"):
        GroundedAnswerUpdate.model_validate(payload)


def test_new_answer_activated_rationale_uses_deterministic_id() -> None:
    added = _rationale().model_copy(
        update={
            "rationale_id": "invented-id",
            "taxonomy_label": "market_size_assessment",
            "evidence_refs": (
                EvidenceRef(
                    evidence_id="A-001",
                    source_kind="founder_answer",
                    excerpt="Three adjacent segments have signed pilots.",
                ),
            ),
        }
    )
    payload = {
        "answer_id": "A-001",
        "affected_rationale_labels": ["market_size_assessment"],
        "direction": "positive",
        "confidence": 0.8,
        "resolved_question": False,
        "materially_changed_assessment": True,
        "investment_likelihood_before": 0.3,
        "investment_likelihood_after": 0.4,
        "decision_confidence_after": 0.7,
        "rationale_state_after": [added.model_dump(mode="json")],
        "unresolved_questions_after": ["The original question remains unresolved."],
        "update_summary": "The answer activates adjacent-market evidence.",
        "evidence_effect": "new_positive",
        "supporting_answer_excerpt": "Three adjacent segments have signed pilots.",
        "grounded_changes": [{
            "rationale_id": "invented-id",
            "taxonomy_label": "market_size_assessment",
            "change": "added",
            "before_direction": None,
            "after_direction": "positive",
            "answer_ids": ["A-001"],
            "justification": "The answer identifies validated adjacent demand.",
        }],
    }

    with pytest.raises(ValueError, match="deterministic rationale_id"):
        GroundedAnswerUpdate.model_validate(payload)


def test_material_rationale_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence_refs"):
        RationaleState(
            rationale_id="R-001",
            taxonomy_label="market_size_assessment",
            direction="negative",
            salience="primary",
            confidence=0.8,
            assessment="The market appears bounded.",
            evidence_refs=(),
        )


def test_taxonomy_labels_are_checked() -> None:
    with pytest.raises(ValueError, match="unknown rationale"):
        validate_taxonomy_labels(
            ["founder_execution", "invented_label"], {"founder_execution"}
        )


def test_final_assessment_forces_in_or_out() -> None:
    payload = _final().model_dump(mode="json")
    payload["decision"] = "Maybe"

    with pytest.raises(ValidationError, match="decision"):
        FinalAssessment.model_validate(payload)


def test_report_requires_simulation_disclosure() -> None:
    with pytest.raises(ValidationError, match="simulation"):
        FounderReport(
            session_id="session-1",
            vc_slug="charles-hudson-precursor-ventures",
            disclosure="Charles Hudson assessment.",
            initial_assessment=_initial(),
            final_assessment=_final(),
            material_answer_ids=(),
            pitch_improvement_suggestions=("Quantify retention.",),
            usage={"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01},
        )


def test_report_accepts_optional_classification_trajectory() -> None:
    report = FounderReport(
        session_id="session-1",
        vc_slug="charles-hudson-precursor-ventures",
        disclosure="This is a simulation, not Charles Hudson's advice.",
        initial_assessment=_initial(),
        final_assessment=_final(),
        material_answer_ids=(),
        pitch_improvement_suggestions=("Quantify retention.",),
        usage={"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01},
        classification=ClassificationSummary(
            status="available",
            model_version="v1",
            artifact_id="charles-production-v1",
            snapshots=(
                ClassificationSnapshot(
                    stage="initial",
                    probability_in=0.41,
                    predicted_decision="Out",
                ),
                ClassificationSnapshot(
                    stage="final",
                    probability_in=0.68,
                    predicted_decision="In",
                ),
            ),
        ),
    )

    assert report.classification is not None
    assert report.classification.snapshots[-1].probability_in == 0.68


def test_existing_report_remains_backwards_compatible() -> None:
    report = FounderReport(
        session_id="session-1",
        vc_slug="charles-hudson-precursor-ventures",
        disclosure="This is a simulation, not Charles Hudson's advice.",
        initial_assessment=_initial(),
        final_assessment=_final(),
        material_answer_ids=(),
        pitch_improvement_suggestions=(),
        usage={"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
    )

    assert report.classification is None
    assert report.grounded is None


def test_report_accepts_grounded_decision_comparison() -> None:
    digest = "a" * 64
    baseline = GroundedBaselineSummary(
        contract_version="v4",
        episode_slug="39-this-pitch-is-damn-near-perfect",
        phase1=GroundedArtifactRef(
            path="outputs/canonical/phase1/investigation.json",
            sha256=digest,
            status="accepted",
        ),
        phase2=GroundedArtifactRef(
            path="outputs/canonical/phase2/decision.json",
            sha256=digest,
            status="accepted",
        ),
        decision="In",
        investment_likelihood=0.63,
        decision_confidence=0.72,
        controlling_rationale_ids=("R1",),
    )
    comparison = GroundedDecisionComparison(
        baseline=baseline,
        final_contract_version="v4",
        final_decision="Out",
        final_investment_likelihood=0.42,
        final_decision_confidence=0.7,
        decision_changed=True,
        rationale_changes=(
            GroundedRationaleChange(
                rationale_id="R1",
                taxonomy_label="founder_execution",
                change="weakened",
                before_direction="positive",
                after_direction="neutral",
                answer_ids=("A-001",),
                justification="The founder clarified that the pilots are unpaid.",
            ),
        ),
        final_artifact_path="grounded/final-phase2/decision.json",
        final_artifact_sha256=digest,
    )

    report = FounderReport(
        session_id="session-1",
        vc_slug="charles-hudson-precursor-ventures",
        disclosure="This is a simulation, not Charles Hudson's advice.",
        initial_assessment=_initial(),
        final_assessment=_final(),
        material_answer_ids=("A-001",),
        pitch_improvement_suggestions=(),
        usage={"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        grounded=comparison,
    )

    assert report.grounded is not None
    assert report.grounded.baseline.investment_likelihood == 0.63
    assert report.grounded.decision_changed is True
