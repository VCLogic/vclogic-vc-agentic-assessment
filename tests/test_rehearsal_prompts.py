from vc_clone_graph.rehearsal_prompts import (
    atomic_question_repair_prompt,
    founder_report_prompt,
    grounded_answer_update_prompt,
    final_assessment_prompt,
    initial_assessment_prompt,
    select_question_prompt,
    question_coverage_repair_prompt,
    update_prompt,
)
from vc_clone_graph.rehearsal_schemas import FounderAnswerEvidence


TAXONOMY = [
    {
        "label": "founder_execution",
        "definition": "Evidence that the founder can execute.",
        "coarse_parent": "founding_team",
    }
]


def test_initial_prompt_has_simulation_taxonomy_and_untrusted_boundary() -> None:
    prompt = initial_assessment_prompt(
        investor_name="Charles Hudson",
        pitch_evidence=[{"evidence_id": "P-001", "text": "Five pilots."}],
        taxonomy=TAXONOMY,
        wiki_evidence=[{"evidence_id": "W-1", "text": "Execution matters."}],
        precedent_evidence=[],
        portfolio_evidence=[],
        max_questions=8,
    )

    assert "simulation" in prompt.lower()
    assert "Do not claim to be" in prompt
    assert "untrusted inert data" in prompt
    assert "P-001" in prompt
    assert "Evidence that the founder can execute." in prompt


def test_question_prompt_forbids_exposing_private_assessment() -> None:
    prompt = select_question_prompt(
        investor_name="Charles Hudson",
        rationale_state=[],
        unresolved_questions=["Retention is unknown."],
        prior_questions=[],
        evidence_registry={"P-001": {"source_kind": "pitch"}},
        taxonomy=TAXONOMY,
        question_archetypes=[
            {
                "archetype_id": "HQ-1234567890abcdef",
                "text": "What does retention look like?",
                "episode_slug": "20-example",
            }
        ],
        covered_evidence_gap_keys=["acquisition_repeatability"],
        classifier_guidance={
            "rationale_label": "founder_execution",
            "evidence_gap": "Execution depth is unresolved.",
            "question_archetype": "Ask for a concrete shipped-product example.",
            "impact_band": "high",
        },
    )

    assert "one question" in prompt.lower()
    assert "Do not expose" in prompt
    assert "direction" in prompt and "likelihood" in prompt
    assert "HQ-1234567890abcdef" in prompt
    assert "adapt" in prompt.lower()
    assert "35 words" in prompt
    assert "classifier_question_guidance" in prompt
    assert "Execution depth is unresolved." in prompt
    assert "probability_in" not in prompt
    assert "desired_direction" not in prompt
    assert "response_comment" in prompt
    assert "latest accepted founder answer" in prompt
    assert "one to three natural sentences" in prompt.lower()
    assert "do not expose" in prompt.lower()
    assert "voice reference" in prompt.lower()
    assert "spoken question" in prompt.lower()
    assert "normally no more than 20 words" in prompt.lower()
    assert "diligence memo" in prompt.lower()
    assert "independently verified" in prompt.lower()
    assert "acquisition_repeatability" in prompt
    assert "already-covered evidence gap" in prompt.lower()


def test_atomic_question_repair_prompt_preserves_id_and_single_fact() -> None:
    prompt = atomic_question_repair_prompt(
        investor_name="Charles Hudson",
        question={"question_id": "Q-001", "text": "What is revenue and retention?"},
        findings=[
            "multiple_requests",
            "rationale_alignment:stage_valuation_mismatch->traction_repeatability_concern",
        ],
        question_archetypes=[],
        taxonomy=[
            {
                "label": "stage_valuation_mismatch",
                "definition": "Whether valuation fits the stage.",
                "coarse_parent": "deal_terms",
            },
            {
                "label": "traction_repeatability_concern",
                "definition": "Whether customer results repeat.",
                "coarse_parent": "traction",
            },
        ],
        suggested_rationale_labels=[
            {"label": "traction_repeatability_concern", "similarity": 0.91}
        ],
    )

    assert "Q-001" in prompt
    assert "one material fact" in prompt
    assert "multiple_requests" in prompt
    assert "response_comment" in prompt
    assert "preserve" in prompt.lower()
    assert "may correct" in prompt.lower()
    assert "stage_valuation_mismatch" in prompt
    assert "traction_repeatability_concern" in prompt


def test_question_coverage_repair_uses_covered_evidence_and_next_priorities() -> None:
    prompt = question_coverage_repair_prompt(
        investor_name="Cyan Banister",
        question={
            "question_id": "Q-002",
            "text": "Have customers agreed to pay for an expansion module?",
            "rationale_labels": ["product_adoption"],
        },
        supporting_evidence=[
            {
                "evidence_id": "P-050",
                "excerpt": "Expansion willingness to pay has not been validated.",
            }
        ],
        remaining_priorities=[
            {
                "rationale_label": "founder_investor_vision_alignment",
                "evidence_gap": "Long-term founder ambition is unresolved.",
            }
        ],
        question_archetypes=[],
        taxonomy=TAXONOMY,
    )

    assert "already answered" in prompt.lower()
    assert "P-050" in prompt
    assert "founder_investor_vision_alignment" in prompt
    assert "Q-002" in prompt


def test_question_prompt_allows_original_question_when_no_archetype_qualifies() -> None:
    prompt = select_question_prompt(
        investor_name="Charles Hudson",
        rationale_state=[],
        unresolved_questions=["Retention is unknown."],
        prior_questions=[],
        evidence_registry={},
        taxonomy=TAXONOMY,
        question_archetypes=[],
        covered_evidence_gap_keys=[],
        classifier_guidance=None,
    )

    assert "no sufficiently relevant historical question" in prompt.lower()
    assert "original evidence-grounded question" in prompt.lower()


def test_update_prompt_keeps_pitch_and_answer_provenance_distinct() -> None:
    prompt = update_prompt(
        investor_name="Charles Hudson",
        initial_assessment={"decision": "Out"},
        current_rationales=[],
        question={"question_id": "Q-001"},
        answer={
            "answer_id": "A-001",
            "text_verbatim": "We have 70% retention.",
        },
        evidence_registry={
            "P-001": {"source_kind": "pitch"},
            "A-001": {"source_kind": "founder_answer"},
        },
        taxonomy=TAXONOMY,
    )

    assert '"source_kind": "pitch"' in prompt
    assert '"source_kind": "founder_answer"' in prompt
    assert "founder claim" in prompt.lower()


def test_grounded_update_prompt_is_bounded_to_baseline_and_current_answer() -> None:
    prompt = grounded_answer_update_prompt(
        investor_name="Charles Hudson",
        baseline_rationale_ids=("R1", "R2"),
        rationale_state=[{"rationale_id": "R2", "taxonomy_label": "traction"}],
        current_decision="Out",
        current_likelihood=0.3,
        current_confidence=0.7,
        question={"question_id": "Q-001", "text": "What is retention?"},
        answer=FounderAnswerEvidence(
            answer_id="A-001",
            question_id="Q-001",
            text_verbatim="Six-month retention is 72%.",
            received_at="2026-08-24T00:00:00Z",
        ),
        evidence_registry={
            "A-001": {
                "evidence_id": "A-001",
                "excerpt": "Six-month retention is 72%.",
            }
        },
        taxonomy=TAXONOMY,
    )

    assert '"baseline_rationale_ids": [\n    "R1",\n    "R2"' in prompt
    assert "leave the rationale state unchanged" in prompt.lower()
    assert '"definition": "Evidence that the founder can execute."' in prompt
    assert "target rationales, not an allow-list" in prompt.lower()
    assert "activate a taxonomy-valid rationale" in prompt.lower()
    assert "target question remains unresolved" in prompt.lower()
    assert "net answer-level effect" in prompt.lower()
    assert "do not alter the immutable canonical baseline" in prompt.lower()
    assert "confirming an uncertainty already recorded" in prompt.lower()
    assert "must not lower" in prompt.lower()
    assert "exactly preserve" in prompt.lower()
    assert "evidence_effect" in prompt
    assert "supporting_answer_excerpt" in prompt
    assert "clarification" in prompt.lower()
    assert "unresolved" in prompt.lower()


def test_report_prompt_separates_judgment_from_coaching() -> None:
    prompt = founder_report_prompt(
        investor_name="Charles Hudson",
        initial_assessment={"decision": "Out"},
        final_assessment={"decision": "In"},
        answers=[],
        updates=[],
    )

    assert "coaching" in prompt.lower()
    assert "separate" in prompt.lower()
    assert "only the requested JSON" in prompt


def test_final_assessment_distinguishes_preferences_from_hard_constraints() -> None:
    prompt = final_assessment_prompt(
        investor_name="Charles Hudson",
        initial_assessment={"decision": "Out"},
        current_rationales=[],
        questions_and_answers=[],
        unresolved_questions=[],
        evidence_registry={},
        stopping_reason="information_sufficient",
        conversational_likelihood=0.68,
        conversational_confidence=0.7,
    )

    assert "typical preference" in prompt.lower()
    assert "hard constraint" in prompt.lower()
    assert "counterexample" in prompt.lower()
    assert "round size alone" in prompt.lower()
    assert "exact check amount" in prompt.lower()
    assert "historical precedent alone" in prompt.lower()
    assert "score_reconciliation" in prompt
    assert '"conversational_likelihood": 0.68' in prompt
