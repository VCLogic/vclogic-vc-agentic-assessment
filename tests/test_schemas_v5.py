from __future__ import annotations

import pytest
from pydantic import ValidationError

from vc_clone_graph.schemas_v5 import InvestigationV5, validate_v5_investigation


def payload() -> dict:
    return {
        "schema_version": "investigation-v5",
        "source_schema_version": "investigation-v4.1",
        "source_investigation_sha256": "a" * 64,
        "episode_slug": "18-rowvigor",
        "association_training_episode_slugs": ["20-harper-wilde", "22-lunar"],
        "association_thresholds": {
            "min_posterior_probability": 0.6,
            "min_support": 3,
            "min_lift": 1.0,
            "max_per_source": 5,
        },
        "questions": [{"question": "Can they execute?", "status": "answered", "answer": "Pilots", "evidence_ids": ["W-a"]}],
        "rationales": [{
            "rationale_id": "R1", "taxonomy_label": "founder_execution",
            "direction": "positive", "salience": "primary", "confidence": 0.8,
            "pitch_evidence_ids": ["P-001"], "wiki_evidence_ids": ["W-a"],
            "historical_evidence_ids": [], "justification": "Pilots support execution.",
            "associated_rationales": [{
                "taxonomy_label": "market_size_assessment", "posterior_probability": 0.75,
                "support": 4, "lift": 2.0, "antecedent_labels": ["founder_execution"],
                "status": "hypothesis_only",
            }],
        }],
        "conflicts": [],
        "unmapped_observations": [{
            "observation_id": "U1", "description": "Other issue",
            "pitch_evidence": ["Other"], "evidence_ids": ["W-a"],
            "decision_relevance": "Potentially relevant",
            "association_status": "unavailable_unmapped", "associated_rationales": [],
        }],
        "information_sufficient": True, "sufficiency_assessment": "Enough",
        "searchable_questions": [], "diligence_questions": [], "next_search_objectives": [],
        "summary": "Summary", "reviewed_pitch_evidence_ids": ["P-001"],
        "material_statement_coverage": [{
            "pitch_evidence_id": "P-001", "decision_dimension": "founder_execution",
            "direction": "positive", "constraint_signal": "none",
            "mapping_type": "taxonomy_rationale", "mapped_ids": ["R1"],
            "assessment": "Execution", "associated_rationales": [],
        }],
        "constraint_assessments": [], "portfolio_overlap_assessments": [],
        "episode_level_associations": [],
    }


def test_v5_round_trips_nested_associations() -> None:
    model = validate_v5_investigation(payload(), {"founder_execution", "market_size_assessment"})
    assert isinstance(model, InvestigationV5)
    assert model.rationales[0].associated_rationales[0].status == "hypothesis_only"
    assert model.unmapped_observations[0].associated_rationales == []


def test_v5_rejects_target_training_leakage() -> None:
    candidate = payload()
    candidate["association_training_episode_slugs"].append("18-rowvigor")
    with pytest.raises(ValidationError, match="target episode"):
        InvestigationV5.model_validate(candidate)


def test_v5_rejects_unknown_or_already_active_hypothesis() -> None:
    with pytest.raises(ValueError, match="outside taxonomy"):
        validate_v5_investigation(payload(), {"founder_execution"})
    candidate = payload()
    candidate["rationales"][0]["associated_rationales"][0]["taxonomy_label"] = "founder_execution"
    with pytest.raises(ValueError, match="already active"):
        validate_v5_investigation(candidate, {"founder_execution"})
