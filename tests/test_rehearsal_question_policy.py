from vc_clone_graph.rehearsal_classifier import LinearClassifierArtifact
from vc_clone_graph.rehearsal_question_policy import (
    QuestionCandidate,
    rank_counterfactual_questions,
    should_continue_interview,
)


def _classifier() -> LinearClassifierArtifact:
    return LinearClassifierArtifact.model_validate(
        {
            "schema": "rehearsal-linear-classifier-v1",
            "artifact_id": "test-v1",
            "model_version": "v1",
            "vc_slug": "charles-hudson-precursor-ventures",
            "training_context": "production",
            "excluded_episode_slug": None,
            "feature_names": [
                "rationale__market_size_assessment__signed_confidence",
                "rationale__founder_execution__signed_confidence",
            ],
            "scales": [1.0, 1.0],
            "coefficients": [2.0, 0.1],
            "intercept": 0.0,
            "decision_threshold": 0.5,
            "training_episode_slugs": ["18-rowvigor", "20-harper-wilde"],
        }
    )


def test_counterfactual_policy_prioritizes_largest_weighted_movement() -> None:
    priorities = rank_counterfactual_questions(
        _classifier(),
        {},
        (
            QuestionCandidate(
                rationale_label="founder_execution",
                evidence_gap="Execution proof is missing.",
                question_archetype="Ask for a shipped-product example.",
                answerability=1.0,
                investor_relevance=1.0,
                novelty=1.0,
            ),
            QuestionCandidate(
                rationale_label="market_size_assessment",
                evidence_gap="The reachable market is unresolved.",
                question_archetype="Ask for a bottom-up market estimate.",
                answerability=0.8,
                investor_relevance=1.0,
                novelty=1.0,
            ),
        ),
    )

    assert priorities[0].rationale_label == "market_size_assessment"
    assert priorities[0].probability_impact > priorities[1].probability_impact
    assert "probability" not in priorities[0].public_guidance()
    assert "desired_direction" not in priorities[0].public_guidance()


def test_policy_is_deterministic_and_marks_below_threshold() -> None:
    priorities = rank_counterfactual_questions(
        _classifier(),
        {},
        (
            QuestionCandidate(
                rationale_label="unknown_z",
                evidence_gap="Unknown Z.",
                question_archetype="Ask Z.",
            ),
            QuestionCandidate(
                rationale_label="unknown_a",
                evidence_gap="Unknown A.",
                question_archetype="Ask A.",
            ),
        ),
        minimum_probability_impact=0.05,
    )

    assert [row.rationale_label for row in priorities] == ["unknown_a", "unknown_z"]
    assert all(row.below_impact_threshold for row in priorities)


def test_minimum_question_overrides_low_impact_once() -> None:
    decision = should_continue_interview(
        question_count=0,
        minimum_questions=1,
        maximum_questions=4,
        founder_requested_finish=False,
        information_sufficient=True,
        highest_probability_impact=0.0,
        minimum_probability_impact=0.05,
    )

    assert decision.continue_interview
    assert decision.reason == "minimum_diagnostic_question"


def test_stopping_rules_are_explicit() -> None:
    founder_stop = should_continue_interview(
        question_count=1,
        minimum_questions=1,
        maximum_questions=4,
        founder_requested_finish=True,
        information_sufficient=False,
        highest_probability_impact=0.9,
        minimum_probability_impact=0.05,
    )
    capped = should_continue_interview(
        question_count=4,
        minimum_questions=1,
        maximum_questions=4,
        founder_requested_finish=False,
        information_sufficient=False,
        highest_probability_impact=0.9,
        minimum_probability_impact=0.05,
    )
    sufficient = should_continue_interview(
        question_count=1,
        minimum_questions=1,
        maximum_questions=4,
        founder_requested_finish=False,
        information_sufficient=True,
        highest_probability_impact=0.2,
        minimum_probability_impact=0.05,
    )
    low_impact = should_continue_interview(
        question_count=1,
        minimum_questions=1,
        maximum_questions=4,
        founder_requested_finish=False,
        information_sufficient=False,
        highest_probability_impact=0.01,
        minimum_probability_impact=0.05,
    )

    assert founder_stop.reason == "founder_requested"
    assert capped.reason == "question_cap"
    assert sufficient.reason == "information_sufficient"
    assert low_impact.reason == "low_expected_impact"
    assert not any(
        row.continue_interview
        for row in (founder_stop, capped, sufficient, low_impact)
    )
