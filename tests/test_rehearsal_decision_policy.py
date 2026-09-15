from vc_clone_graph.rehearsal_decision_policy import calibrate_commitment_amounts
from vc_clone_graph.rehearsal_schemas import FinalAssessment


def _assessment(text: str) -> FinalAssessment:
    return FinalAssessment(
        decision="In",
        investment_likelihood=0.64,
        decision_confidence=0.66,
        strongest_positive_rationale_ids=(),
        decisive_concern_rationale_ids=(),
        unresolved_uncertainties=(),
        reversal_conditions=(),
        rationale_state=(),
        decision_justification=text,
    )


def test_precedent_only_check_amount_is_replaced_with_bounded_language() -> None:
    result, findings = calibrate_commitment_amounts(
        _assessment("I would commit approximately $50,000 subject to verification."),
        evidence_registry={
            "H-001": {
                "source_kind": "precedent",
                "excerpt": "The investor committed $50,000 in that episode.",
            }
        },
    )

    assert "$50,000" not in result.decision_justification
    assert "a small first check" in result.decision_justification
    assert findings == ("unsupported_exact_check_amount_removed",)


def test_amount_is_preserved_when_pitch_and_investor_policy_both_support_it() -> None:
    assessment = _assessment("I would commit $25,000 subject to verification.")
    result, findings = calibrate_commitment_amounts(
        assessment,
        evidence_registry={
            "P-010": {
                "source_kind": "pitch",
                "excerpt": "The company is seeking a $25,000 allocation.",
            },
            "W-010": {
                "source_kind": "wiki",
                "excerpt": "A typical first check is $25,000.",
            },
        },
    )

    assert result == assessment
    assert findings == ()
