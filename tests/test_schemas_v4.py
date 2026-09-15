from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vc_clone_graph.schemas_v4 import (
    DecisionV4,
    DecisionV41,
    InvestigationV4,
    InvestigationV41,
    InvestigationV42,
    RationaleLockV42,
    ProposedRationaleV4,
    TaxonomyReflectionV4,
    decision_v4_json_schema,
    decision_v41_json_schema,
    investigation_v4_json_schema,
    investigation_v41_json_schema,
    rationale_lock_v42_json_schema,
    model_facing_taxonomy,
    normalize_investigation_v41_payload,
    normalize_investigation_v41_evidence_payload,
    normalize_rationale_lock_v42_payload,
    validate_decision_v41_against_investigation,
)


ROOT = Path(__file__).parents[1]


def rationale() -> dict:
    return {
        "rationale_id": "R1",
        "taxonomy_label": "founder_execution",
        "direction": "positive",
        "salience": "primary",
        "confidence": 0.82,
        "pitch_evidence": ["We built and launched the product."],
        "pitch_evidence_ids": ["P-001"],
        "wiki_evidence_ids": ["W-founder"],
        "historical_evidence_ids": [],
        "justification": "Execution is material to this investor.",
    }


def investigation() -> dict:
    return {
        "schema_version": "investigation-v4",
        "episode_slug": "18-rowvigor",
        "questions": [
            {
                "question": "Can this team execute?",
                "status": "answered",
                "answer": "The pitch reports a completed launch.",
                "evidence_ids": ["W-founder"],
            }
        ],
        "rationales": [rationale()],
        "conflicts": [],
        "unmapped_observations": [],
        "information_sufficient": True,
        "sufficiency_assessment": "The material founder question is answered.",
        "searchable_questions": [],
        "diligence_questions": [],
        "next_search_objectives": [],
        "summary": "Strong execution evidence.",
    }


def decision() -> dict:
    return {
        "schema_version": "decision-v4",
        "episode_slug": "18-rowvigor",
        "investigation_sha256": "a" * 64,
        "decision": "In",
        "decision_path": "founder_conviction_exception",
        "investment_likelihood": 0.62,
        "decision_confidence": 0.74,
        "decision_justification": "I would commit a small check because the team has executed.",
        "recommended_check_tier": "exploratory_lt_100k",
        "review_priority_score": 0.81,
        "review_priority_reason": "The pitch deserves direct review despite unresolved scale.",
        "controlling_rationale_ids": ["R1"],
        "evidence_basis": [
            {
                "source_type": "pitch",
                "source_reference": "We built and launched the product.",
                "evidence_ids": ["P-001"],
                "interpretation": "The team has shipped.",
                "effect_on_decision": "supports",
            }
        ],
        "strongest_opposing_case": {
            "argument": "Scale is unresolved.",
            "response": "That limits check size rather than eliminating conviction.",
        },
        "searchable_questions": [],
        "diligence_questions": ["Can acquisition scale?"],
        "reversal_conditions": ["Launch retention is weak."],
        "information_sufficient": True,
        "sufficiency_assessment": "Enough evidence exists for a bounded decision.",
        "next_search_objectives": [],
        "phase1_reopen_recommended": False,
        "missing_considerations": [],
        "founder_exception_considered": True,
        "founder_exception_rationale_ids": ["R1"],
        "founder_exception_precedent_ids": [],
        "founder_exception_assessment": "Exceptional execution supports a deliberately small first check.",
    }


def investigation_v41() -> dict:
    payload = investigation()
    for row in payload["rationales"]:
        row.pop("pitch_evidence", None)
    payload.update(
        schema_version="investigation-v4.1",
        reviewed_pitch_evidence_ids=["P-001"],
        material_statement_coverage=[
            {
                "pitch_evidence_id": "P-001",
                "decision_dimension": "founder_execution",
                "direction": "positive",
                "constraint_signal": "none",
                "mapping_type": "taxonomy_rationale",
                "mapped_ids": ["R1"],
                "assessment": "The launch is material execution evidence.",
            }
        ],
        constraint_assessments=[
            {
                "constraint_id": "C1",
                "constraint_kind": "category_or_expertise_fit",
                "policy_statement": "The investor must be able to evaluate and help the company.",
                "status": "not_triggered",
                "severity": "material",
                "pitch_evidence_ids": ["P-001"],
                "wiki_evidence_ids": ["W-founder"],
                "historical_evidence_ids": [],
                "mapped_ids": ["R1"],
                "assessment": "No category mismatch is established by the supplied record.",
            }
        ],
    )
    return payload


def decision_v41() -> dict:
    payload = decision()
    payload.update(
        schema_version="decision-v4.1",
        blocking_constraint_ids=[],
        constraint_assessment_summary="No supplied constraint blocks a small check.",
        founder_exception_precedent_match=(
            "The cited observed In has comparable founder execution and ordinary open diligence."
        ),
    )
    payload["founder_exception_precedent_ids"] = ["H-example"]
    return payload


def rationale_lock_v42() -> dict:
    return {
        "schema_version": "rationale-lock-v4.2",
        "episode_slug": "18-rowvigor",
        "candidate_investigation_sha256": "b" * 64,
        "dispositions": [
            {
                "rationale_id": "R1",
                "decision": "locked",
                "rejection_reason": None,
                "direction": "positive",
                "salience": "primary",
                "confidence": 0.81,
                "materiality_justification": "Direct execution evidence is investor-specific and material.",
            },
            {
                "rationale_id": "R2",
                "decision": "rejected",
                "rejection_reason": "generic_checklist",
                "direction": None,
                "salience": None,
                "confidence": None,
                "materiality_justification": "The candidate is only a generic diligence topic.",
            },
        ],
        "lock_summary": "One of two candidates is materially activated.",
    }


def test_model_facing_taxonomy_keeps_all_definitions_and_parents() -> None:
    rows = json.loads(
        (ROOT / "inputs/taxonomy/codebook_v_final.json").read_text(encoding="utf-8")
    )
    view = model_facing_taxonomy(rows)

    assert len(view) == 44
    assert {row["label"] for row in view} == {row["label"] for row in rows}
    assert all(row["definition"] for row in view)
    assert all(row["coarse_parent"] for row in view)
    assert all(set(row) == {"label", "definition", "coarse_parent"} for row in view)


def test_investigation_schema_binds_only_runtime_accessible_evidence_ids() -> None:
    schema = investigation_v4_json_schema(
        episode_slug="18-rowvigor",
        pitch_evidence_ids=("P-001", "P-002"),
        wiki_evidence_ids=("W-founder",),
        historical_evidence_ids=("H-example",),
    )

    rationale = schema["$defs"]["RationaleV4"]["properties"]
    question = schema["$defs"]["QuestionAssessmentV4"]["properties"]
    observation = schema["$defs"]["UnmappedObservationV4"]["properties"]

    assert rationale["pitch_evidence_ids"]["items"]["enum"] == ["P-001", "P-002"]
    assert rationale["wiki_evidence_ids"]["items"]["enum"] == ["W-founder"]
    assert rationale["historical_evidence_ids"]["items"]["enum"] == ["H-example"]
    assert question["evidence_ids"]["items"]["enum"] == ["H-example", "W-founder"]
    assert observation["evidence_ids"]["items"]["enum"] == ["H-example", "W-founder"]

    no_history = investigation_v4_json_schema(
        episode_slug="18-rowvigor",
        pitch_evidence_ids=("P-001",),
        wiki_evidence_ids=("W-founder",),
        historical_evidence_ids=(),
    )
    assert no_history["$defs"]["RationaleV4"]["properties"][
        "historical_evidence_ids"
    ]["maxItems"] == 0


def test_decision_schema_binds_evidence_and_frozen_rationale_ids() -> None:
    schema = decision_v4_json_schema(
        episode_slug="18-rowvigor",
        investigation_sha256="a" * 64,
        pitch_evidence_ids=("P-001",),
        wiki_evidence_ids=("W-founder",),
        historical_evidence_ids=("H-example",),
        rationale_ids=("R1", "U1"),
    )

    properties = schema["properties"]
    basis = schema["$defs"]["EvidenceBasisV4"]["properties"]

    assert basis["evidence_ids"]["items"]["enum"] == [
        "H-example",
        "P-001",
        "W-founder",
    ]
    assert properties["controlling_rationale_ids"]["items"]["enum"] == ["R1", "U1"]
    assert properties["founder_exception_rationale_ids"]["items"]["enum"] == [
        "R1",
        "U1",
    ]
    assert properties["founder_exception_precedent_ids"]["items"]["enum"] == [
        "H-example"
    ]


def test_investigation_v41_schema_binds_pitch_review_and_constraint_evidence() -> None:
    schema = investigation_v41_json_schema(
        episode_slug="18-rowvigor",
        pitch_evidence_ids=("P-001", "P-002"),
        wiki_evidence_ids=("W-founder",),
        historical_evidence_ids=("H-example",),
    )

    properties = schema["properties"]
    constraint = schema["$defs"]["ConstraintAssessmentV41"]["properties"]

    assert properties["reviewed_pitch_evidence_ids"]["items"]["enum"] == [
        "P-001",
        "P-002",
    ]
    assert constraint["wiki_evidence_ids"]["items"]["enum"] == ["W-founder"]
    assert constraint["historical_evidence_ids"]["items"]["enum"] == ["H-example"]


def test_rationale_lock_v42_requires_exact_candidate_disposition_coverage() -> None:
    parsed = RationaleLockV42.model_validate(rationale_lock_v42())
    assert [row.rationale_id for row in parsed.dispositions] == ["R1", "R2"]

    duplicate = rationale_lock_v42()
    duplicate["dispositions"][1]["rationale_id"] = "R1"
    with pytest.raises(ValidationError, match="exactly once"):
        RationaleLockV42.model_validate(duplicate)


def test_normalize_rationale_lock_v42_clears_only_branch_inapplicable_fields() -> None:
    payload = rationale_lock_v42()
    payload["dispositions"][0]["rejection_reason"] = "generic_checklist"
    payload["dispositions"][1].update(
        direction="neutral", salience="secondary", confidence=0.45
    )

    normalized, findings = normalize_rationale_lock_v42_payload(payload)
    parsed = RationaleLockV42.model_validate(normalized)

    assert parsed.dispositions[0].rejection_reason is None
    assert parsed.dispositions[0].direction == "positive"
    assert parsed.dispositions[1].rejection_reason == "generic_checklist"
    assert parsed.dispositions[1].direction is None
    assert parsed.dispositions[1].salience is None
    assert parsed.dispositions[1].confidence is None
    assert findings == [
        "NORMALIZED_LOCKED_REJECTION_REASON:R1",
        "NORMALIZED_REJECTED_LOCK_ATTRIBUTES:R2",
    ]


@pytest.mark.parametrize(
    "reason",
    [
        "generic_checklist",
        "insufficient_pitch_evidence",
        "insufficient_investor_specificity",
        "diligence_only",
        "duplicate",
        "non_material",
        "direction_unresolved",
    ],
)
def test_rationale_lock_v42_accepts_auditable_rejection_reasons(reason: str) -> None:
    payload = rationale_lock_v42()
    payload["dispositions"][1]["rejection_reason"] = reason
    assert RationaleLockV42.model_validate(payload).dispositions[1].rejection_reason == reason


def test_rationale_lock_v42_schema_binds_runtime_candidate_ids() -> None:
    schema = rationale_lock_v42_json_schema(
        episode_slug="18-rowvigor",
        candidate_investigation_sha256="b" * 64,
        candidate_rationale_ids=("R1", "R2"),
    )
    disposition = schema["$defs"]["RationaleLockDispositionV42"]["properties"]
    assert disposition["rationale_id"]["enum"] == ["R1", "R2"]
    assert schema["properties"]["episode_slug"]["const"] == "18-rowvigor"
    assert schema["properties"]["candidate_investigation_sha256"]["const"] == "b" * 64


def test_investigation_v42_allows_zero_locked_rationales_and_preserves_candidate_mappings() -> None:
    payload = investigation_v41()
    payload.update(
        schema_version="investigation-v4.2",
        rationales=[],
        candidate_rationale_ids=["R1"],
        candidate_investigation_sha256="b" * 64,
        rationale_lock_sha256="c" * 64,
        rejected_candidate_count=1,
    )
    parsed = InvestigationV42.model_validate(payload)
    assert parsed.rationales == []
    assert parsed.material_statement_coverage[0].mapped_ids == ["R1"]


def test_investigation_v41_requires_material_statement_mappings_to_exist() -> None:
    payload = investigation_v41()
    parsed = InvestigationV41.model_validate(payload)
    assert parsed.material_statement_coverage[0].mapped_ids == ["R1"]

    payload["material_statement_coverage"][0]["mapped_ids"] = ["R99"]
    with pytest.raises(ValidationError, match="unknown frozen rationale or observation"):
        InvestigationV41.model_validate(payload)


def test_normalize_investigation_v41_repairs_only_mechanical_contract_variants() -> None:
    payload = investigation_v41()
    payload["questions"][0]["evidence_ids"] = ["P-001", "W-founder"]
    payload["rationales"][0]["rationale_id"] = "R-001"
    payload["material_statement_coverage"][0].update(
        decision_dimension="founder_market_fit",
        mapped_ids=["founder_execution"],
    )
    payload["constraint_assessments"][0].update(
        constraint_id="C-001",
        constraint_kind="fund_economics_constraint",
        mapped_ids=["R-001"],
    )
    payload["constraint_assessments"].append(
        {
            "constraint_id": "C-002",
            "constraint_kind": "portfolio_conflict",
            "policy_statement": "A supported conflict can block an investment.",
            "status": "possible",
            "severity": "hard",
            "pitch_evidence_ids": [],
            "wiki_evidence_ids": ["W-founder"],
            "historical_evidence_ids": [],
            "mapped_ids": [],
            "assessment": "No eligible portfolio candidate was supplied.",
        }
    )
    original_semantics = {
        key: payload["rationales"][0][key]
        for key in (
            "taxonomy_label",
            "direction",
            "salience",
            "confidence",
            "pitch_evidence_ids",
            "wiki_evidence_ids",
            "historical_evidence_ids",
            "justification",
        )
    }

    normalized, findings = normalize_investigation_v41_payload(payload)
    parsed = InvestigationV41.model_validate(normalized)

    assert payload["rationales"][0]["rationale_id"] == "R-001"
    assert parsed.rationales[0].rationale_id == "R1"
    assert {
        key: normalized["rationales"][0][key] for key in original_semantics
    } == original_semantics
    assert parsed.questions[0].evidence_ids == ["W-founder"]
    assert parsed.material_statement_coverage[0].mapped_ids == ["R1"]
    assert parsed.material_statement_coverage[0].decision_dimension == "other"
    assert parsed.constraint_assessments[0].constraint_id == "C1"
    assert parsed.constraint_assessments[0].constraint_kind == "other"
    assert [row.constraint_id for row in parsed.constraint_assessments] == ["C1"]
    assert any(item.startswith("NORMALIZED_GENERATED_ID:R-001->R1") for item in findings)
    assert any(item.startswith("NORMALIZED_DROPPED_CONSTRAINT:C2") for item in findings)


def test_normalize_investigation_v41_does_not_invent_unknown_mappings() -> None:
    payload = investigation_v41()
    payload["material_statement_coverage"][0]["mapped_ids"] = [
        "unknown_rationale_label"
    ]

    normalized, findings = normalize_investigation_v41_payload(payload)

    assert findings == []
    with pytest.raises(ValidationError):
        InvestigationV41.model_validate(normalized)


def test_normalize_investigation_v41_drops_only_inaccessible_investor_citations() -> None:
    payload = investigation_v41()
    payload["rationales"][0]["wiki_evidence_ids"].append("W-typo")
    payload["questions"][0]["evidence_ids"].append("H-unknown")

    normalized, findings = normalize_investigation_v41_evidence_payload(
        payload,
        wiki_evidence_ids={"W-founder"},
        historical_evidence_ids=set(),
    )
    parsed = InvestigationV41.model_validate(normalized)

    assert parsed.rationales[0].wiki_evidence_ids == ["W-founder"]
    assert parsed.questions[0].evidence_ids == ["W-founder"]
    assert findings == [
        "NORMALIZED_DROPPED_INACCESSIBLE_RATIONALE_EVIDENCE:R1:W-typo",
        "NORMALIZED_DROPPED_INACCESSIBLE_QUESTION_EVIDENCE:1:H-unknown",
    ]


def test_normalize_investigation_v41_repairs_observation_aliases_and_filters_auxiliary_mappings() -> None:
    payload = investigation_v41()
    payload["unmapped_observations"] = [
        {
            "observation_id": "UO-001",
            "description": "The two-sided activation dependency remains unresolved.",
            "pitch_evidence": ["P-001"],
            "evidence_ids": ["W-founder"],
            "decision_relevance": "It may affect adoption and economics.",
        }
    ]
    payload["material_statement_coverage"][0].update(
        mapping_type="unmapped_observation",
        mapped_ids=["UO-001", "R-001"],
    )
    payload["constraint_assessments"][0]["mapped_ids"] = [
        "founder_execution",
        "unsupported_constraint_label",
    ]

    normalized, findings = normalize_investigation_v41_payload(payload)
    parsed = InvestigationV41.model_validate(normalized)

    assert parsed.unmapped_observations[0].observation_id == "U1"
    assert parsed.material_statement_coverage[0].mapped_ids == ["U1"]
    assert parsed.constraint_assessments[0].mapped_ids == ["R1"]
    assert "NORMALIZED_GENERATED_ID:UO-001->U1" in findings
    assert any(item.startswith("NORMALIZED_FILTERED_MAPPED_IDS:material_statement") for item in findings)
    assert any(item.startswith("NORMALIZED_FILTERED_MAPPED_IDS:constraint") for item in findings)


def test_investigation_v41_schema_uses_compact_evidence_references() -> None:
    schema = investigation_v41_json_schema(
        episode_slug="18-rowvigor",
        pitch_evidence_ids=["P-001"],
        wiki_evidence_ids=["W-founder"],
        historical_evidence_ids=["H-example"],
    )

    assert "pitch_evidence" not in schema["$defs"]["RationaleV41"]["properties"]
    assert (
        "statement"
        not in schema["$defs"]["MaterialStatementCoverageV41"]["properties"]
    )
    assert schema["properties"]["material_statement_coverage"]["maxItems"] == 12


def test_decision_v41_exception_requires_in_precedent_and_respects_hard_constraint() -> None:
    investigation_payload = investigation_v41()
    parsed_investigation = InvestigationV41.model_validate(investigation_payload)
    parsed_decision = DecisionV41.model_validate(decision_v41())
    validate_decision_v41_against_investigation(parsed_decision, parsed_investigation)

    without_precedent = decision_v41()
    without_precedent["founder_exception_precedent_ids"] = []
    with pytest.raises(ValidationError, match="observed In precedent"):
        DecisionV41.model_validate(without_precedent)

    investigation_payload["constraint_assessments"][0].update(
        status="triggered", severity="hard"
    )
    blocked_investigation = InvestigationV41.model_validate(investigation_payload)
    with pytest.raises(ValueError, match="triggered hard constraint"):
        validate_decision_v41_against_investigation(
            parsed_decision, blocked_investigation
        )


def test_decision_v41_schema_binds_frozen_constraint_ids() -> None:
    schema = decision_v41_json_schema(
        episode_slug="18-rowvigor",
        investigation_sha256="a" * 64,
        pitch_evidence_ids=("P-001",),
        wiki_evidence_ids=("W-founder",),
        historical_evidence_ids=("H-example",),
        rationale_ids=("R1",),
        constraint_ids=("C1", "C2"),
    )

    assert schema["properties"]["blocking_constraint_ids"]["items"]["enum"] == [
        "C1",
        "C2",
    ]


def test_investigation_v4_contains_only_material_rationales_and_observations() -> None:
    payload = investigation()
    payload["unmapped_observations"] = [
        {
            "observation_id": "U1",
            "description": "The founder independently narrows the financing strategy.",
            "pitch_evidence": ["We will stage the financing."],
            "evidence_ids": ["H-example"],
            "decision_relevance": "May affect confidence in capital discipline.",
        }
    ]
    parsed = InvestigationV4.model_validate(payload)

    assert parsed.rationales[0].taxonomy_label == "founder_execution"
    assert parsed.unmapped_observations[0].observation_id == "U1"
    assert not hasattr(parsed, "taxonomy_dispositions")
    assert not hasattr(parsed, "activated_candidates")


def test_decision_v4_has_one_classification_and_independent_priority() -> None:
    parsed = DecisionV4.model_validate(decision())

    assert parsed.decision == "In"
    assert parsed.review_priority_score == 0.81
    assert not hasattr(parsed, "any_check")
    assert not hasattr(parsed, "standard_check")


def test_decision_path_must_match_final_decision() -> None:
    payload = decision()
    payload["decision_path"] = "out"
    with pytest.raises(ValidationError, match="decision path"):
        DecisionV4.model_validate(payload)


def test_diligence_questions_do_not_force_another_search() -> None:
    parsed = DecisionV4.model_validate(decision())
    assert parsed.information_sufficient is True
    assert parsed.diligence_questions == ["Can acquisition scale?"]


def test_out_decision_requires_none_check_tier() -> None:
    payload = decision()
    payload.update(
        decision="Out", decision_path="out", investment_likelihood=0.41,
        recommended_check_tier="standard"
    )
    with pytest.raises(ValidationError, match="Out decision requires check tier none"):
        DecisionV4.model_validate(payload)


def test_decision_and_likelihood_must_agree() -> None:
    payload = decision()
    payload["investment_likelihood"] = 0.49
    with pytest.raises(ValidationError, match="decision and investment likelihood disagree"):
        DecisionV4.model_validate(payload)


def test_proposed_rationale_requires_extreme_confidence() -> None:
    proposal = {
        "provisional_id": "P1",
        "proposed_label": "founder_self_awareness",
        "proposed_definition": "The founder independently recognizes material limitations.",
        "confidence": 0.89,
        "material_to_frozen_decision": True,
        "reusable_in_future_decisions": True,
        "pitch_evidence": ["We need to narrow our strategy."],
        "investor_evidence_ids": ["W-example"],
        "nearest_existing_labels": ["founder_coachability"],
        "why_existing_taxonomy_is_insufficient": "Coachability requires external feedback.",
        "necessity_justification": "The frozen explanation otherwise loses a material distinction.",
    }
    with pytest.raises(ValidationError):
        ProposedRationaleV4.model_validate(proposal)

    proposal["confidence"] = 0.95
    parsed = ProposedRationaleV4.model_validate(proposal)
    reflection = TaxonomyReflectionV4.model_validate(
        {
            "schema_version": "taxonomy-reflection-v4",
            "episode_slug": "18-rowvigor",
            "decision_sha256": "b" * 64,
            "proposals": [parsed.model_dump(mode="json")],
            "review_summary": "One proposal requires human review.",
        }
    )
    assert reflection.proposals[0].confidence == 0.95
