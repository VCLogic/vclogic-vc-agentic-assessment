from __future__ import annotations

from vc_clone_graph.prompts_v4 import (
    phase1_investigation_v4_prompt,
    phase1_plan_v4_prompt,
    phase2_decision_v4_prompt,
    phase2_plan_v4_prompt,
    taxonomy_reflection_v4_prompt,
    phase1_investigation_v41_prompt,
    phase1_plan_v41_prompt,
    phase2_decision_v41_prompt,
    phase2_plan_v41_prompt,
    phase1_rationale_lock_v42_prompt,
)


PITCH = "Founder: We built the product and signed ten customers."
TAXONOMY = (
    {
        "label": "founder_execution",
        "definition": "Evidence that the founder can convert plans into completed work.",
        "coarse_parent": "founding_team",
    },
    {
        "label": "market_size_assessment",
        "definition": "Assessment of whether the reachable market supports venture outcomes.",
        "coarse_parent": "market_opportunity",
    },
)
WIKI = [{"chunk_id": "W-founder", "text": "Charles values learning velocity."}]
HISTORY = [
    {
        "evidence_id": "H-example",
        "episode_slug": "39-example",
        "text": "Charles: I am investing because the founder learned quickly.",
        "decision_status": "In",
    }
]
INVESTIGATION = {
    "schema_version": "investigation-v4",
    "episode_slug": "18-rowvigor",
    "rationales": [{"rationale_id": "R1", "taxonomy_label": "founder_execution"}],
    "unmapped_observations": [],
    "information_sufficient": False,
}
DECISION = {
    "schema_version": "decision-v4",
    "decision": "Out",
    "decision_justification": "The execution evidence does not resolve scale.",
}
ASSOCIATIONS = {
    "schema": "rationale-association-annotations-v1",
    "scientific_status": "fold_safe_advisory_hypotheses",
    "episode_slug": "18-rowvigor",
    "investigation_sha256": "a" * 64,
    "episode_level_associations": [
        {
            "taxonomy_label": "market_size_assessment",
            "posterior_probability": 0.72,
            "support": 4,
            "lift": 1.8,
            "antecedent_labels": ["founder_execution"],
            "status": "hypothesis_only",
        }
    ],
}


def assert_noise_absent(prompt: str) -> None:
    assert "accessible_episode_inventory" not in prompt
    assert "source_sha256" not in prompt
    assert "precedent_searches" not in prompt
    assert "retrieval_findings" not in prompt


def test_phase1_plan_contains_exact_pitch_and_full_taxonomy_definitions() -> None:
    prompt = phase1_plan_v4_prompt(
        "Charles Hudson", "18-rowvigor", PITCH, TAXONOMY, None, [], []
    )

    assert PITCH in prompt
    for row in TAXONOMY:
        assert row["label"] in prompt
        assert row["definition"] in prompt
        assert row["coarse_parent"] in prompt
    assert_noise_absent(prompt)


def test_phase1_investigation_contains_only_current_exact_evidence() -> None:
    prompt = phase1_investigation_v4_prompt(
        "Charles Hudson",
        "18-rowvigor",
        PITCH,
        TAXONOMY,
        WIKI,
        HISTORY,
        None,
        [],
    )

    assert "W-founder" in prompt
    assert "H-example" in prompt
    assert "unrelated-old-evidence" not in prompt
    assert_noise_absent(prompt)


def test_phase2_prompts_receive_pitch_but_no_registry() -> None:
    plan = phase2_plan_v4_prompt(
        "Charles Hudson", "18-rowvigor", PITCH, INVESTIGATION, None, [], []
    )
    decision = phase2_decision_v4_prompt(
        "Charles Hudson",
        "18-rowvigor",
        PITCH,
        INVESTIGATION,
        "a" * 64,
        WIKI,
        HISTORY,
        None,
        [],
    )

    for prompt in (plan, decision):
        assert PITCH in prompt
        assert_noise_absent(prompt)
    assert "cheap optionality" in decision
    assert "review_priority_score" in decision
    assert "plain-language" in decision
    assert "exact check amount" in decision.lower()
    assert "historical precedent alone" in decision.lower()


def test_phase2_prompts_receive_associations_only_as_non_controlling_hypotheses() -> None:
    plan = phase2_plan_v4_prompt(
        "Charles Hudson",
        "18-rowvigor",
        PITCH,
        INVESTIGATION,
        None,
        [],
        [],
        ASSOCIATIONS,
    )
    decision = phase2_decision_v4_prompt(
        "Charles Hudson",
        "18-rowvigor",
        PITCH,
        INVESTIGATION,
        "a" * 64,
        WIKI,
        HISTORY,
        None,
        [],
        ASSOCIATIONS,
    )

    for prompt in (plan, decision):
        assert "market_size_assessment" in prompt
        assert "hypothesis_only" in prompt
        assert "not activated rationales" in prompt
        assert "controlling rationale IDs" in prompt


def test_phase2_plan_is_decision_neutral_and_contrastive() -> None:
    prompt = phase2_plan_v4_prompt(
        "Elizabeth Yin", "89-a-big-bet-on-an-ex-trucker", PITCH,
        INVESTIGATION, None, [], []
    )

    assert "Do not assume or name a current In or Out case" in prompt
    assert "at least one observed In and one observed Out" in prompt


def test_phase2_decision_uses_pitch_window_not_post_diligence_standard() -> None:
    prompt = phase2_decision_v4_prompt(
        "Elizabeth Yin",
        "89-a-big-bet-on-an-ex-trucker",
        PITCH,
        INVESTIGATION,
        "a" * 64,
        WIKI,
        HISTORY,
        None,
        [],
    )

    assert "observable pitch-room commitment" in prompt
    assert "subject to ordinary verification" in prompt
    assert "Information absent from the pitch is uncertainty, not adverse evidence" in prompt


def test_phase2_never_uses_withheld_target_decision_as_a_feature() -> None:
    prompt = phase2_decision_v4_prompt(
        "Elizabeth Yin",
        "89-a-big-bet-on-an-ex-trucker",
        PITCH,
        INVESTIGATION,
        "a" * 64,
        WIKI,
        HISTORY,
        None,
        [],
    )

    assert "intentionally contains no target-investor decision" in prompt
    assert "absence of commitment language cannot support Out" in prompt
    assert "underlying founder and venture evidence before the historical decision" in prompt


def test_taxonomy_reflection_is_explicitly_after_frozen_decision() -> None:
    prompt = taxonomy_reflection_v4_prompt(
        "Charles Hudson",
        "18-rowvigor",
        PITCH,
        TAXONOMY,
        INVESTIGATION,
        DECISION,
        "b" * 64,
    )

    assert "already frozen" in prompt
    assert "0.90" in prompt
    assert "human review" in prompt
    assert_noise_absent(prompt)


def test_phase1_v41_requires_complete_pitch_review_and_dispositive_dimensions() -> None:
    plan_prompt = phase1_plan_v41_prompt(
        "Cyan Banister", "142-kinometrix", PITCH, TAXONOMY, None, [], []
    )
    investigation_prompt = phase1_investigation_v41_prompt(
        "Cyan Banister", "142-kinometrix", PITCH, TAXONOMY,
        WIKI, HISTORY, None, [],
    )

    for prompt in (plan_prompt, investigation_prompt):
        assert "founder ambition" in prompt.lower()
        assert "exit" in prompt.lower()
        assert "category" in prompt.lower()
        assert "expertise" in prompt.lower()
        assert "conflict" in prompt.lower()
        assert "ownership" in prompt.lower()
        assert "venture economics" in prompt.lower()
    assert "reviewed_pitch_evidence_ids" in investigation_prompt
    assert "every p-xxx" in investigation_prompt.lower()
    assert "material_statement_coverage" in investigation_prompt
    assert "constraint_assessments" in investigation_prompt


def test_phase2_v41_checks_constraints_before_founder_exception() -> None:
    plan_prompt = phase2_plan_v41_prompt(
        "Jesse Middleton", "149-dopl", PITCH, INVESTIGATION, None, [], []
    )
    decision_prompt = phase2_decision_v41_prompt(
        "Jesse Middleton", "149-dopl", PITCH, INVESTIGATION, "a" * 64,
        WIKI, HISTORY, None, [],
    )

    assert "hard constraints" in plan_prompt.lower()
    assert "before" in decision_prompt.lower()
    assert "founder-conviction exception" in decision_prompt.lower()
    assert "genuinely analogous observed in precedent" in decision_prompt.lower()
    assert "ordinary diligence" in decision_prompt.lower()
    assert "category" in decision_prompt.lower()
    assert "investor expertise" in decision_prompt.lower()
    assert "triggering pitch or founder-answer fact" in decision_prompt.lower()
    assert "explicit investor rule or recurring observed pattern" in decision_prompt.lower()
    assert "counterexample" in decision_prompt.lower()
    assert "large financing round" in decision_prompt.lower()


def test_phase1_v42_lock_prompt_is_independent_materiality_review() -> None:
    candidate = {
        **INVESTIGATION,
        "schema_version": "investigation-v4.1",
        "rationales": [
            {
                "rationale_id": "R1",
                "taxonomy_label": "founder_execution",
                "pitch_evidence_ids": ["P-001"],
                "wiki_evidence_ids": ["W-founder"],
                "historical_evidence_ids": ["H-example"],
            }
        ],
    }
    prompt = phase1_rationale_lock_v42_prompt(
        "Charles Hudson",
        "18-rowvigor",
        PITCH,
        TAXONOMY,
        candidate,
        "b" * 64,
        {"W-founder": WIKI[0], "H-example": HISTORY[0]},
    )

    assert PITCH in prompt
    assert "founder_execution" in prompt
    assert TAXONOMY[0]["definition"] in prompt
    assert "W-founder" in prompt and "H-example" in prompt
    assert "directly activated" in prompt
    assert "investor-specific" in prompt
    assert "material" in prompt
    assert "distinct" in prompt
    assert "generic_checklist" in prompt
    assert "Do not retrieve" in prompt
    assert "Do not make an In or Out" in prompt
    assert "actual_decision" not in prompt
    assert_noise_absent(prompt)
