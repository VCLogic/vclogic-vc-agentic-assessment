from vc_clone_graph.rehearsal_question_coverage import (
    EvidenceStatement,
    assess_question_coverage,
)


def test_marks_binary_question_redundant_when_pitch_already_contains_answer() -> None:
    result = assess_question_coverage(
        question=(
            "Have any current agencies agreed to pay for a workforce-operations "
            "module beyond replacement coordination?"
        ),
        rationale_labels=("product_adoption",),
        evidence=(
            EvidenceStatement(
                evidence_id="P-050",
                text="Expansion-module willingness to pay has not been validated.",
                rationale_labels=("product_adoption",),
            ),
        ),
        prior_answers=(),
    )

    assert result.redundant is True
    assert result.supporting_evidence_ids == ("P-050",)
    assert result.reason == "answer_already_present"


def test_unknown_metric_does_not_block_question_seeking_actual_value() -> None:
    result = assess_question_coverage(
        question="What is your gross margin after implementation work?",
        rationale_labels=("unit_economics_assessment",),
        evidence=(
            EvidenceStatement(
                evidence_id="P-049",
                text="Gross margin has not yet been measured.",
                rationale_labels=("unit_economics_assessment",),
            ),
        ),
        prior_answers=(),
    )

    assert result.redundant is False
    assert result.reason == "unresolved_information_request"
    assert result.supporting_evidence_ids == ()


def test_marks_question_redundant_when_prior_answer_already_supplied_fact() -> None:
    result = assess_question_coverage(
        question="How many customers renewed after their first contract?",
        rationale_labels=("traction_repeatability_concern",),
        evidence=(),
        prior_answers=(
            EvidenceStatement(
                evidence_id="A-002",
                text=(
                    "Question: How many customers renewed after their first contract? "
                    "Answer: None have reached renewal yet."
                ),
                rationale_labels=("traction_repeatability_concern",),
            ),
        ),
    )

    assert result.redundant is True
    assert result.supporting_evidence_ids == ("A-002",)
    assert result.reason == "prior_answer_already_present"


def test_unrelated_pitch_fact_cannot_make_question_redundant() -> None:
    result = assess_question_coverage(
        question="What is your gross margin after implementation work?",
        rationale_labels=("unit_economics_assessment",),
        evidence=(
            EvidenceStatement(
                evidence_id="P-020",
                text="The company has eight paying agencies.",
                rationale_labels=("product_adoption",),
            ),
        ),
        prior_answers=(),
    )

    assert result.redundant is False
    assert result.reason == "unresolved_information_request"
