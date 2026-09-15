from __future__ import annotations

import pytest

from vc_clone_graph.rationale_association_annotations import (
    AssociationAnnotationThresholds,
    build_rationale_association_annotations,
)
from vc_clone_graph.rationale_completion import (
    AssociationRuleEstimate,
    CompletionHypothesis,
    CompletionPrediction,
)


def _hypothesis(
    label: str,
    *rules: AssociationRuleEstimate,
) -> CompletionHypothesis:
    selected = rules[0]
    return CompletionHypothesis(
        label=label,
        rank=1,
        logistic_probability=0.0,
        logistic_fallback=True,
        association_probability=selected.posterior_probability,
        association_support=selected.support,
        association_antecedent=selected.antecedent,
        association_lift=selected.lift,
        completion_probability=selected.posterior_probability,
        training_slugs=("train-a", "train-b", "train-c"),
        association_rules=tuple(rules),
    )


def _prediction() -> CompletionPrediction:
    market_rules = (
        AssociationRuleEstimate(("founder_execution",), 4, 0.75, 0.30, 2.50),
        AssociationRuleEstimate(
            ("founder_execution", "traction_assessment"), 3, 0.80, 0.30, 2.67
        ),
    )
    weak_rules = (
        AssociationRuleEstimate(("founder_execution",), 8, 0.59, 0.40, 1.48),
        AssociationRuleEstimate(("traction_assessment",), 2, 0.80, 0.40, 2.00),
        AssociationRuleEstimate(("founder_execution",), 6, 0.70, 0.75, 0.93),
    )
    candidates = (
        _hypothesis("market_size_assessment", *market_rules),
        _hypothesis("business_model_viability", *weak_rules),
    )
    return CompletionPrediction(
        vc_slug="charles-hudson",
        episode_slug="20-harper-wilde",
        actual_decision="Out",
        observed_labels=("founder_execution", "traction_assessment"),
        reference_labels=("founder_execution", "market_size_assessment"),
        training_slugs=("train-a", "train-b", "train-c"),
        candidates=candidates,
        association=candidates,
        logistic=candidates,
        ensemble=candidates,
    )


def _investigation() -> dict:
    return {
        "schema_version": "investigation-v4.1",
        "episode_slug": "20-harper-wilde",
        "rationales": [
            {"rationale_id": "R1", "taxonomy_label": "founder_execution"},
            {"rationale_id": "R2", "taxonomy_label": "traction_assessment"},
        ],
        "unmapped_observations": [
            {"observation_id": "U1", "description": "Unclassified founder claim."}
        ],
        "material_statement_coverage": [
            {
                "pitch_evidence_id": "P-001",
                "mapping_type": "taxonomy_rationale",
                "mapped_ids": ["R1", "R2"],
            },
            {
                "pitch_evidence_id": "P-002",
                "mapping_type": "unmapped_observation",
                "mapped_ids": ["U1"],
            },
            {
                "pitch_evidence_id": "P-003",
                "mapping_type": "taxonomy_rationale",
                "mapped_ids": ["R2"],
            },
        ],
    }


def test_builds_gated_source_bound_annotations_for_claims_and_statements() -> None:
    annotations = build_rationale_association_annotations(
        _investigation(),
        "a" * 64,
        _prediction(),
        taxonomy_labels={
            "founder_execution",
            "traction_assessment",
            "market_size_assessment",
            "business_model_viability",
        },
    )

    assert annotations.schema_name == "rationale-association-annotations-v1"
    assert annotations.model_dump(mode="json")["schema"] == (
        "rationale-association-annotations-v1"
    )
    assert annotations.investigation_sha256 == "a" * 64
    assert annotations.thresholds == AssociationAnnotationThresholds()
    assert annotations.training_episode_slugs == ("train-a", "train-b", "train-c")
    assert annotations.claim_annotations[0].rationale_id == "R1"
    assert [row.taxonomy_label for row in annotations.claim_annotations[0].associated_rationales] == [
        "market_size_assessment"
    ]
    assert annotations.claim_annotations[1].associated_rationales[0].antecedent_labels == (
        "founder_execution",
        "traction_assessment",
    )
    assert annotations.material_statement_annotations[0].associated_rationales[0].antecedent_labels == (
        "founder_execution",
        "traction_assessment",
    )
    assert annotations.material_statement_annotations[1].associated_rationales == ()
    assert annotations.material_statement_annotations[2].associated_rationales[0].antecedent_labels == (
        "founder_execution",
        "traction_assessment",
    )
    assert annotations.unmapped_observation_annotations[0].status == "unavailable_unmapped"
    assert annotations.unmapped_observation_annotations[0].associated_rationales == ()
    assert "business_model_viability" not in {
        row.taxonomy_label for row in annotations.episode_level_associations
    }


def test_excludes_already_active_labels_and_orders_deterministically() -> None:
    prediction = _prediction()
    extra = _hypothesis(
        "founder_execution",
        AssociationRuleEstimate(("traction_assessment",), 5, 0.90, 0.40, 2.25),
    )
    prediction = CompletionPrediction(
        **{**prediction.__dict__, "candidates": (*prediction.candidates, extra)}
    )

    annotations = build_rationale_association_annotations(
        _investigation(),
        "a" * 64,
        prediction,
        taxonomy_labels={
            "founder_execution",
            "traction_assessment",
            "market_size_assessment",
            "business_model_viability",
        },
    )

    assert "founder_execution" not in {
        row.taxonomy_label for row in annotations.episode_level_associations
    }
    payload = annotations.model_dump_json()
    repeated = build_rationale_association_annotations(
        _investigation(),
        "a" * 64,
        prediction,
        taxonomy_labels={
            "business_model_viability",
            "market_size_assessment",
            "traction_assessment",
            "founder_execution",
        },
    ).model_dump_json()
    assert repeated == payload


def test_rejects_training_leakage_hash_mismatch_and_unknown_labels() -> None:
    leaking = _prediction()
    leaking = CompletionPrediction(
        **{
            **leaking.__dict__,
            "training_slugs": (*leaking.training_slugs, leaking.episode_slug),
        }
    )
    with pytest.raises(ValueError, match="held episode appears in association training"):
        build_rationale_association_annotations(
            _investigation(), "a" * 64, leaking, taxonomy_labels=set(_prediction().observed_labels)
        )

    changed = _investigation() | {"episode_slug": "21-other"}
    with pytest.raises(ValueError, match="episode mismatch"):
        build_rationale_association_annotations(
            changed, "a" * 64, _prediction(), taxonomy_labels={"founder_execution"}
        )

    with pytest.raises(ValueError, match="outside taxonomy"):
        build_rationale_association_annotations(
            _investigation(),
            "a" * 64,
            _prediction(),
            taxonomy_labels={"founder_execution", "traction_assessment"},
        )


def test_requires_valid_sha256_and_positive_candidate_limit() -> None:
    with pytest.raises(ValueError, match="investigation_sha256"):
        build_rationale_association_annotations(
            _investigation(), "bad", _prediction(), taxonomy_labels={"founder_execution"}
        )
    with pytest.raises(ValueError, match="max_per_source"):
        build_rationale_association_annotations(
            _investigation(),
            "a" * 64,
            _prediction(),
            taxonomy_labels={
                "founder_execution",
                "traction_assessment",
                "market_size_assessment",
                "business_model_viability",
            },
            max_per_source=0,
        )
