from __future__ import annotations

from hashlib import sha256

import pytest
from pydantic import ValidationError

from vc_clone_graph.prompts_v4 import phase1_rationale_mapping_v43_prompt
from vc_clone_graph.schemas_v4 import (
    InvestigationV41,
    InvestigationV43,
    RationaleMappingV43,
    normalize_rationale_mapping_v43_payload,
    rationale_mapping_v43_json_schema,
)
from vc_clone_graph.workflow_v4 import (
    apply_rationale_mapping_v43,
    constrain_rationale_mapping_v43,
)


DIGEST = sha256(b"candidate").hexdigest()


def mapping_payload(*, action: str = "relabel", target: str = "founder_execution") -> dict:
    return {
        "schema_version": "rationale-mapping-v4.3",
        "episode_slug": "18-rowvigor",
        "candidate_investigation_sha256": DIGEST,
        "dispositions": [
            {
                "rationale_id": "R1",
                "action": action,
                "target_taxonomy_label": target,
                "merge_into_rationale_id": None,
                "winning_definition": "Evidence that the founders can execute.",
                "rejected_alternatives": ["founder_qualities"],
                "mapping_justification": "Paid pilots are direct execution evidence, not a generic trait.",
            }
        ],
        "mapping_summary": "One candidate was mapped contrastively.",
    }


def investigation_payload() -> dict:
    return {
        "schema_version": "investigation-v4.3",
        "episode_slug": "18-rowvigor",
        "questions": [{
            "question": "Can the founders execute?",
            "status": "answered",
            "answer": "Five paid pilots support execution.",
            "evidence_ids": ["W-abc"],
        }],
        "rationales": [{
            "rationale_id": "R1",
            "taxonomy_label": "founder_execution",
            "direction": "positive",
            "salience": "primary",
            "confidence": 0.8,
            "pitch_evidence_ids": ["P-001"],
            "wiki_evidence_ids": ["W-abc"],
            "historical_evidence_ids": [],
            "justification": "Paid pilots match the investor's execution principle.",
        }],
        "conflicts": [],
        "unmapped_observations": [],
        "information_sufficient": True,
        "sufficiency_assessment": "The record is sufficient.",
        "searchable_questions": [],
        "diligence_questions": [],
        "next_search_objectives": [],
        "summary": "Execution is material.",
        "reviewed_pitch_evidence_ids": ["P-001"],
        "material_statement_coverage": [{
            "pitch_evidence_id": "P-001",
            "decision_dimension": "founder_execution",
            "direction": "positive",
            "constraint_signal": "none",
            "mapping_type": "taxonomy_rationale",
            "mapped_ids": ["R1"],
            "assessment": "Paid pilots are material execution evidence.",
        }],
        "constraint_assessments": [],
        "portfolio_overlap_assessments": [],
        "candidate_rationale_ids": ["R1"],
        "candidate_investigation_sha256": DIGEST,
        "rationale_mapping_sha256": "b" * 64,
        "relabeled_candidate_count": 1,
        "merged_candidate_count": 0,
        "mapping_status": "accepted",
    }


def candidate_payload() -> dict:
    payload = investigation_payload()
    payload["schema_version"] = "investigation-v4.1"
    payload["rationales"][0]["taxonomy_label"] = "founder_qualities"
    for key in (
        "candidate_rationale_ids",
        "candidate_investigation_sha256",
        "rationale_mapping_sha256",
        "relabeled_candidate_count",
        "merged_candidate_count",
        "mapping_status",
    ):
        payload.pop(key)
    return payload


def test_v43_mapping_contract_and_bound_schema() -> None:
    parsed = RationaleMappingV43.model_validate(mapping_payload())
    assert parsed.dispositions[0].action == "relabel"

    schema = rationale_mapping_v43_json_schema(
        episode_slug="18-rowvigor",
        candidate_investigation_sha256=DIGEST,
        candidate_rationale_ids=("R1",),
        taxonomy_labels=("founder_execution", "founder_qualities"),
    )
    assert schema["properties"]["episode_slug"]["const"] == "18-rowvigor"
    disposition = schema["$defs"]["RationaleMappingDispositionV43"]["properties"]
    assert disposition["rationale_id"]["enum"] == ["R1"]
    assert disposition["target_taxonomy_label"]["enum"] == [
        "founder_execution", "founder_qualities"
    ]
    assert disposition["merge_into_rationale_id"] == {
        "type": "string",
        "enum": ["R1", "none"],
    }

    duplicate = mapping_payload()
    duplicate["dispositions"].append(dict(duplicate["dispositions"][0]))
    with pytest.raises(ValidationError, match="exactly once"):
        RationaleMappingV43.model_validate(duplicate)


def test_v43_mapping_contract_requires_valid_merge_target() -> None:
    payload = mapping_payload(action="merge")
    with pytest.raises(ValidationError, match="merge action requires"):
        RationaleMappingV43.model_validate(payload)


def test_v43_mapping_normalizes_nonmerge_schema_branch_fill() -> None:
    payload = mapping_payload(action="keep", target="founder_qualities")
    payload["dispositions"][0]["merge_into_rationale_id"] = "R1"
    normalized, findings = normalize_rationale_mapping_v43_payload(payload)
    assert normalized["dispositions"][0]["merge_into_rationale_id"] is None
    assert findings == ["NORMALIZED_NONMERGE_TARGET:R1"]
    assert RationaleMappingV43.model_validate(normalized)


def test_v43_final_investigation_records_mapping_binding() -> None:
    parsed = InvestigationV43.model_validate(investigation_payload())
    assert parsed.mapping_status == "accepted"
    invalid = investigation_payload()
    invalid["relabeled_candidate_count"] = 2
    with pytest.raises(ValidationError, match="mapping counts"):
        InvestigationV43.model_validate(invalid)


def test_v43_mapping_prompt_is_compact_contrastive_and_leakage_safe() -> None:
    prompt = phase1_rationale_mapping_v43_prompt(
        "Charles Hudson",
        "18-rowvigor",
        [{
            "label": "founder_execution",
            "definition": "Evidence the founders can execute.",
            "coarse_parent": "founding_team",
        }, {
            "label": "founder_qualities",
            "definition": "General founder traits.",
            "coarse_parent": "founding_team",
        }, {
            "label": "market_size_assessment",
            "definition": "Market scale.",
            "coarse_parent": "market",
        }],
        {
            "schema_version": "investigation-v4.1",
            "episode_slug": "18-rowvigor",
            "rationales": [{"rationale_id": "R1", "taxonomy_label": "founder_qualities"}],
        },
        DIGEST,
        {"W-abc": {"evidence_id": "W-abc", "text": "Execution matters."}},
    )
    assert "founder_execution" in prompt
    assert "founder_qualities" in prompt
    assert "market_size_assessment" not in prompt
    assert "allowed_target_labels_by_rationale_id" in prompt
    assert "Do not decide In or Out" in prompt
    assert "accessible_episode_inventory" not in prompt
    assert "actual_decision" not in prompt


def test_apply_v43_mapping_relabels_without_changing_evidence() -> None:
    candidate = InvestigationV41.model_validate(candidate_payload())
    mapping = RationaleMappingV43.model_validate(mapping_payload())
    final = apply_rationale_mapping_v43(
        candidate,
        mapping,
        taxonomy={
            "founder_execution": "founding_team",
            "founder_qualities": "founding_team",
        },
        mapping_sha256="b" * 64,
    )
    assert final.rationales[0].taxonomy_label == "founder_execution"
    assert final.rationales[0].pitch_evidence_ids == candidate.rationales[0].pitch_evidence_ids
    assert final.rationales[0].wiki_evidence_ids == candidate.rationales[0].wiki_evidence_ids
    assert final.relabeled_candidate_count == 1


def test_apply_v43_mapping_rejects_cross_family_relabel() -> None:
    candidate = InvestigationV41.model_validate(candidate_payload())
    mapping = RationaleMappingV43.model_validate(
        mapping_payload(target="market_size_assessment")
    )
    with pytest.raises(ValueError, match="same taxonomy family"):
        apply_rationale_mapping_v43(
            candidate,
            mapping,
            taxonomy={
                "founder_qualities": "founding_team",
                "market_size_assessment": "market",
            },
            mapping_sha256="b" * 64,
        )


def test_constrain_v43_mapping_retains_original_for_cross_family_action() -> None:
    candidate = InvestigationV41.model_validate(candidate_payload())
    mapping = RationaleMappingV43.model_validate(
        mapping_payload(target="market_size_assessment")
    )
    constrained, findings = constrain_rationale_mapping_v43(
        candidate,
        mapping,
        taxonomy={
            "founder_qualities": "founding_team",
            "market_size_assessment": "market",
        },
        definitions={"founder_qualities": "General founder traits."},
    )
    row = constrained.dispositions[0]
    assert row.action == "keep"
    assert row.target_taxonomy_label == "founder_qualities"
    assert findings == ["MAPPING_CROSS_FAMILY_RETAINED_ORIGINAL:R1"]
