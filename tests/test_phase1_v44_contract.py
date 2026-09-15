from __future__ import annotations

from hashlib import sha256
import json

import pytest
from pydantic import BaseModel, ValidationError

from vc_clone_graph.adjudication_postprocess_v44 import (
    cross_target_evidence_reuse_v44,
)
from vc_clone_graph.phase1_v44 import (
    ClaimMapV44,
    ClaimRetrievalManifestV44,
    RetrievedEvidenceV44,
    RationaleAdjudicationV44,
    RationaleDispositionV44,
    TaxonomyCandidateV44,
    TaxonomyNeighborhoodManifestV44,
    adjudication_findings_v44,
    adjudication_v44_json_schema,
    build_investigation_v44,
    claim_map_v44_json_schema,
    validate_claim_map_runtime_v44,
)
from vc_clone_graph.schemas_v4 import (
    ConstraintAssessmentV41,
    PortfolioOverlapAssessmentV41,
)


def claim_map_payload() -> dict:
    return {
        "schema_version": "claim-map-v4.4",
        "episode_slug": "18-rowvigor",
        "material_claims": [
            {
                "claim_id": "C1",
                "claim_type": "material",
                "statement": "The company has five paid pilots.",
                "topic": "customer traction",
                "decision_relevance": "Paid pilots test execution and demand.",
                "pitch_evidence_ids": ["P-001"],
            },
            {
                "claim_id": "C3",
                "claim_type": "material",
                "statement": "The founders are raising a seed round.",
                "topic": "funding stage",
                "decision_relevance": "Stage affects investor fit.",
                "pitch_evidence_ids": ["P-003"],
            },
        ],
        "adverse_claims": [
            {
                "claim_id": "C2",
                "claim_type": "adverse",
                "statement": "No pilot has renewed yet.",
                "topic": "retention uncertainty",
                "decision_relevance": "Renewal risk weakens traction evidence.",
                "pitch_evidence_ids": ["P-002"],
            }
        ],
        "unanswered_questions": [
            {
                "question_id": "Q1",
                "question": "Will pilots renew?",
                "why_material": "Renewals distinguish trials from repeat demand.",
                "anchor_pitch_evidence_ids": ["P-002"],
            }
        ],
        "claim_coverage": [
            {"pitch_evidence_id": "P-001", "claim_id": "C1"},
            {"pitch_evidence_id": "P-002", "claim_id": "C2"},
            {"pitch_evidence_id": "P-003", "claim_id": "C3"},
            {
                "pitch_evidence_id": "P-004",
                "non_material_justification": "Background product detail only.",
            },
        ],
    }


def retrieval_payload() -> dict:
    wiki_record = {
        "evidence_id": "W-execution",
        "source_kind": "wiki",
        "text": "Execution evidence matters to this investor.",
        "source_locator": "wiki/principles.md#execution",
        "query_id": "query-c1",
        "retrieval_modes": ["dense", "lexical"],
        "turn_start": None,
        "turn_end": None,
        "decision_status": None,
        "score": 0.9,
        "source_sha256": "1" * 64,
        "eligible": True,
        "episode_slug": None,
        "warning": None,
    }
    historical_record = {
        "evidence_id": "H-renewal",
        "source_kind": "historical",
        "text": "I need to see customers renew after the pilot.",
        "source_locator": "17-other-company:turns-12-14",
        "query_id": "query-q1",
        "retrieval_modes": ["dense"],
        "turn_start": 12,
        "turn_end": 14,
        "decision_status": "Out",
        "score": 0.7,
        "source_sha256": "2" * 64,
        "eligible": True,
        "episode_slug": "17-other-company",
        "warning": None,
    }
    source_fields = (
        "evidence_id",
        "source_kind",
        "text",
        "source_locator",
        "source_sha256",
        "episode_slug",
        "turn_start",
        "turn_end",
        "decision_status",
    )
    return {
        "schema_version": "claim-retrieval-manifest-v4.4",
        "episode_slug": "18-rowvigor",
        "claim_map_sha256": DIGEST_A,
        "claim_bundles": [
            {
                "target_kind": "claim",
                "target_id": "C1",
                "query": "paid pilots execution demand",
                "wiki_evidence": [dict(wiki_record)],
                "historical_evidence": [],
                "portfolio_disclosures": [],
                "warnings": [],
                "retrieval_action_ids": ["A1"],
            },
            {
                "target_kind": "question",
                "target_id": "Q1",
                "query": "pilot renewal repeat demand",
                "wiki_evidence": [],
                "historical_evidence": [dict(historical_record)],
                "portfolio_disclosures": [],
                "warnings": [],
                "retrieval_action_ids": ["A2"],
            },
            {
                "target_kind": "claim",
                "target_id": "C2",
                "query": "renewal uncertainty",
                "wiki_evidence": [],
                "historical_evidence": [],
                "portfolio_disclosures": [],
                "warnings": ["No investor evidence retrieved."],
                "retrieval_action_ids": [],
            },
            {
                "target_kind": "claim",
                "target_id": "C3",
                "query": "seed stage fit",
                "wiki_evidence": [],
                "historical_evidence": [],
                "portfolio_disclosures": [],
                "warnings": ["No investor evidence retrieved."],
                "retrieval_action_ids": [],
            },
        ],
        "target_episode_excluded": True,
        "warnings": [],
        "evidence_registry": [
            {key: wiki_record[key] for key in source_fields},
            {key: historical_record[key] for key in source_fields},
        ],
        "retrieval_actions": [
            {
                "action_id": "A1",
                "target_id": "C1",
                "source_kind": "wiki",
                "action_kind": "read",
                "query_id": "query-c1",
                "query": "paid pilots execution demand",
                "result_evidence_ids": ["W-execution"],
                "opened_episode_slugs": [],
                "warnings": [],
            },
            {
                "action_id": "A2",
                "target_id": "Q1",
                "source_kind": "historical",
                "action_kind": "read",
                "query_id": "query-q1",
                "query": "pilot renewal repeat demand",
                "result_evidence_ids": ["H-renewal"],
                "opened_episode_slugs": ["17-other-company"],
                "warnings": [],
            },
        ],
    }


def neighborhood_payload() -> dict:
    return {
        "schema_version": "taxonomy-neighborhood-manifest-v4.4",
        "taxonomy_sha256": "3" * 64,
        "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
        "embedding_revision": "abc123",
        "claim_neighborhoods": [
            {
                "target_id": "C2",
                "query": "retention uncertainty",
                "candidates": [
                    {
                        "taxonomy_label": "traction_repeatability_concern",
                        "definition": "Concern that traction will not repeat.",
                        "coarse_parent": "traction",
                        "dense_score": 0.7,
                        "lexical_score": 0.5,
                        "fused_score": 0.8,
                        "rank": 1,
                        "target_id": "C2",
                        "query": "retention uncertainty",
                        "selection_reason": "Top fused match.",
                    }
                ],
            },
            {
                "target_id": "C1",
                "query": "paid pilots execution demand",
                "candidates": [
                    {
                        "taxonomy_label": "founder_execution",
                        "definition": "Evidence that founders can execute.",
                        "coarse_parent": "founding_team",
                        "dense_score": 0.9,
                        "lexical_score": 0.8,
                        "fused_score": 0.95,
                        "rank": 1,
                        "target_id": "C1",
                        "query": "paid pilots execution demand",
                        "selection_reason": "Top fused match.",
                    },
                    {
                        "taxonomy_label": "product_adoption",
                        "definition": "Evidence of customer adoption.",
                        "coarse_parent": "traction",
                        "dense_score": 0.8,
                        "lexical_score": 0.7,
                        "fused_score": 0.9,
                        "rank": 2,
                        "target_id": "C1",
                        "query": "paid pilots execution demand",
                        "selection_reason": "Within the top-k neighborhood.",
                    },
                ],
            },
            {"target_id": "C3", "query": "seed stage fit", "candidates": []},
            {
                "target_id": "Q1",
                "query": "pilot renewal repeat demand",
                "candidates": [],
            },
        ],
        "ordered_labels": [
            "traction_repeatability_concern",
            "founder_execution",
            "product_adoption",
        ],
    }


def disposition(
    label: str,
    kind: str,
    *,
    claim_ids: list[str] | None = None,
    question_ids: list[str] | None = None,
) -> dict:
    activated = kind in {"core", "candidate"}
    return {
        "taxonomy_label": label,
        "disposition": kind,
        "claim_ids": claim_ids if claim_ids is not None else (["C1"] if activated else []),
        "question_ids": question_ids or [],
        "pitch_evidence_ids": ["P-001"] if activated else [],
        "wiki_evidence_ids": ["W-execution"] if activated else [],
        "historical_evidence_ids": [],
        "portfolio_disclosure_ids": [],
        "direction": "positive" if activated else None,
        "salience": "primary" if activated else None,
        "confidence": 0.8 if activated else None,
        "justification": "The evidence supports this disposition.",
    }


def adjudication_payload() -> dict:
    return {
        "schema_version": "rationale-adjudication-v4.4",
        "episode_slug": "18-rowvigor",
        "dispositions": [
            disposition("founder_execution", "core"),
            disposition("product_adoption", "candidate"),
            disposition(
                "traction_repeatability_concern",
                "question_only",
                question_ids=["Q1"],
            ),
        ],
        "unmapped_observations": [],
        "constraint_assessments": [],
        "portfolio_overlap_assessments": [],
        "adjudication_status": "provisional",
        "validator_findings": [],
        "requested_retrieval_ids": [],
    }


def constraint_assessment_payload() -> dict:
    return {
        "constraint_id": "C1",
        "constraint_kind": "stage_or_check_fit",
        "policy_statement": "The fund invests at seed.",
        "status": "possible",
        "severity": "material",
        "pitch_evidence_ids": ["P-003"],
        "wiki_evidence_ids": ["W-execution"],
        "historical_evidence_ids": [],
        "mapped_ids": ["R1"],
        "assessment": "The round may be outside the normal entry point.",
    }


def portfolio_assessment_payload() -> dict:
    return {
        "portfolio_entity_id": "PE-" + "b" * 20,
        "disclosure_ids": ["PM-" + "a" * 20],
        "pitch_evidence_ids": ["P-003"],
        "overlap_status": "possible_conflict",
        "decision_consequence": "diligence_only",
        "confidence": 0.5,
        "assessment": "The products may overlap.",
    }


def claim_map_model(payload: dict | None = None) -> ClaimMapV44:
    return ClaimMapV44.model_validate_json(json.dumps(payload or claim_map_payload()))


def adjudication_model(payload: dict | None = None) -> RationaleAdjudicationV44:
    return RationaleAdjudicationV44.model_validate_json(
        json.dumps(payload or adjudication_payload())
    )


def neighborhood_model(
    payload: dict | None = None,
) -> TaxonomyNeighborhoodManifestV44:
    return TaxonomyNeighborhoodManifestV44.model_validate_json(
        json.dumps(payload or neighborhood_payload())
    )


def retrieval_model(payload: dict | None = None) -> ClaimRetrievalManifestV44:
    return ClaimRetrievalManifestV44.model_validate_json(
        json.dumps(payload or retrieval_payload())
    )


def canonical_model_sha256(value: BaseModel) -> str:
    payload = json.dumps(
        value.model_dump(mode="json", warnings=False),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


DIGEST_A = canonical_model_sha256(claim_map_model())
DIGEST_B = canonical_model_sha256(adjudication_model())


def test_claim_map_enforces_unique_ids_list_types_and_coverage_bindings() -> None:
    assert claim_map_model().material_claims[0].claim_id == "C1"

    duplicate = claim_map_payload()
    duplicate["adverse_claims"][0]["claim_id"] = "C1"
    with pytest.raises(ValidationError, match="claim IDs must be unique"):
        claim_map_model(duplicate)

    wrong_type = claim_map_payload()
    wrong_type["material_claims"][0]["claim_type"] = "adverse"
    with pytest.raises(ValidationError, match="material_claims must contain material"):
        claim_map_model(wrong_type)

    duplicate_question = claim_map_payload()
    duplicate_question["unanswered_questions"].append(
        dict(duplicate_question["unanswered_questions"][0])
    )
    with pytest.raises(ValidationError, match="question IDs must be unique"):
        claim_map_model(duplicate_question)

    bad_target = claim_map_payload()
    bad_target["claim_coverage"][0]["claim_id"] = "C9"
    with pytest.raises(ValidationError, match="unknown claim coverage target"):
        claim_map_model(bad_target)

    mismatched_citation = claim_map_payload()
    mismatched_citation["claim_coverage"][0]["claim_id"] = "C2"
    with pytest.raises(ValidationError, match="does not cite pitch evidence"):
        claim_map_model(mismatched_citation)


def test_claim_coverage_selects_exactly_one_handling() -> None:
    missing = claim_map_payload()
    missing["claim_coverage"][0].pop("claim_id")
    with pytest.raises(ValidationError, match="exactly one handling"):
        claim_map_model(missing)

    multiple = claim_map_payload()
    multiple["claim_coverage"][0]["question_id"] = "Q1"
    with pytest.raises(ValidationError, match="exactly one handling"):
        claim_map_model(multiple)

    duplicate = claim_map_payload()
    duplicate["claim_coverage"].append(dict(duplicate["claim_coverage"][0]))
    with pytest.raises(ValidationError, match="pitch evidence IDs must be unique"):
        claim_map_model(duplicate)


def test_claim_map_schema_binds_episode_and_all_pitch_citations() -> None:
    schema = claim_map_v44_json_schema(
        episode_slug="18-rowvigor", pitch_evidence_ids=("P-001", "P-002")
    )
    assert schema["properties"]["episode_slug"]["const"] == "18-rowvigor"
    definitions = schema["$defs"]
    assert definitions["PitchClaimV44"]["properties"]["pitch_evidence_ids"]["items"]["enum"] == [
        "P-001",
        "P-002",
    ]
    assert definitions["UnansweredQuestionV44"]["properties"]["anchor_pitch_evidence_ids"]["items"]["enum"] == [
        "P-001",
        "P-002",
    ]
    assert definitions["ClaimCoverageV44"]["properties"]["pitch_evidence_id"]["enum"] == [
        "P-001",
        "P-002",
    ]
    coverage = schema["properties"]["claim_coverage"]
    assert coverage["minItems"] == 2
    assert coverage["maxItems"] == 2


def test_runtime_claim_map_validation_requires_complete_allowed_pitch_coverage() -> None:
    claim_map = claim_map_model()
    assert (
        validate_claim_map_runtime_v44(
            claim_map, ("P-001", "P-002", "P-003", "P-004")
        )
        is claim_map
    )

    with pytest.raises(ValueError, match="missing claim coverage: P-005"):
        validate_claim_map_runtime_v44(
            claim_map, ("P-001", "P-002", "P-003", "P-004", "P-005")
        )


def test_v44_models_reject_coercive_python_inputs_but_accept_json_arrays() -> None:
    coercive_scalar = disposition("founder_execution", "core")
    coercive_scalar["confidence"] = "0.8"
    with pytest.raises(ValidationError):
        RationaleDispositionV44.model_validate(coercive_scalar)

    coercive_container = disposition("founder_execution", "core")
    coercive_container["claim_ids"] = ("C1",)
    with pytest.raises(ValidationError):
        RationaleDispositionV44.model_validate(coercive_container)

    encoded = json.dumps(disposition("founder_execution", "core"))
    assert RationaleDispositionV44.model_validate_json(encoded).claim_ids == ("C1",)


def test_adjudication_rejects_coercive_reused_assessment_inputs() -> None:
    string_confidence = adjudication_payload()
    portfolio = portfolio_assessment_payload()
    portfolio["confidence"] = "0.5"
    string_confidence["portfolio_overlap_assessments"] = [portfolio]
    with pytest.raises(ValidationError):
        RationaleAdjudicationV44.model_validate_json(json.dumps(string_confidence))

    tuple_nested_list = adjudication_model().model_dump(mode="python")
    constraint = constraint_assessment_payload()
    constraint["pitch_evidence_ids"] = ("P-003",)
    tuple_nested_list["constraint_assessments"] = [constraint]
    with pytest.raises(ValidationError):
        RationaleAdjudicationV44.model_validate(tuple_nested_list)

    tuple_outer_collection = adjudication_model().model_dump(mode="python")
    tuple_outer_collection["constraint_assessments"] = (
        constraint_assessment_payload(),
    )
    with pytest.raises(ValidationError):
        RationaleAdjudicationV44.model_validate(tuple_outer_collection)

    boolean_confidence = adjudication_payload()
    portfolio = portfolio_assessment_payload()
    portfolio["confidence"] = True
    boolean_confidence["portfolio_overlap_assessments"] = [portfolio]
    with pytest.raises(ValidationError):
        RationaleAdjudicationV44.model_validate_json(json.dumps(boolean_confidence))


def test_adjudication_strict_nested_assessments_accept_json_and_exact_models() -> None:
    payload = adjudication_payload()
    payload["constraint_assessments"] = [constraint_assessment_payload()]
    payload["portfolio_overlap_assessments"] = [portfolio_assessment_payload()]
    parsed = RationaleAdjudicationV44.model_validate_json(json.dumps(payload))
    assert isinstance(parsed.constraint_assessments[0], ConstraintAssessmentV41)
    assert isinstance(
        parsed.portfolio_overlap_assessments[0], PortfolioOverlapAssessmentV41
    )
    assert RationaleAdjudicationV44.model_validate_json(parsed.model_dump_json()) == parsed

    model_payload = adjudication_model().model_dump(mode="python")
    source_constraint = ConstraintAssessmentV41.model_validate(
        constraint_assessment_payload()
    )
    source_portfolio = PortfolioOverlapAssessmentV41.model_validate(
        portfolio_assessment_payload()
    )
    model_payload["constraint_assessments"] = [source_constraint]
    model_payload["portfolio_overlap_assessments"] = [source_portfolio]
    copied = RationaleAdjudicationV44.model_validate(model_payload)
    source_constraint.mapped_ids.append("R99")
    source_portfolio.disclosure_ids.append("PM-" + "c" * 20)
    assert copied.constraint_assessments[0].mapped_ids == ("R1",)
    assert copied.portfolio_overlap_assessments[0].disclosure_ids == (
        "PM-" + "a" * 20,
    )


def test_adjudication_constraint_accepts_open_question_mapping() -> None:
    payload = adjudication_payload()
    constraint = constraint_assessment_payload()
    constraint["mapped_ids"] = ["Q1"]
    payload["constraint_assessments"] = [constraint]

    parsed = RationaleAdjudicationV44.model_validate_json(json.dumps(payload))
    round_trip = RationaleAdjudicationV44.model_validate_json(
        parsed.model_dump_json()
    )

    assert parsed.constraint_assessments[0].mapped_ids == ("Q1",)
    assert round_trip == parsed


@pytest.mark.parametrize(
    ("source_kind", "evidence_id"),
    [
        ("wiki", "W-bad/id"),
        ("historical", "H-bad.id"),
        ("portfolio", "PM-" + "z" * 20),
    ],
)
def test_retrieved_evidence_uses_exact_identifier_pattern_for_source_kind(
    source_kind: str, evidence_id: str
) -> None:
    with pytest.raises(ValidationError):
        RetrievedEvidenceV44.model_validate(
            {
                "evidence_id": evidence_id,
                "source_kind": source_kind,
                "score": 0.5,
                "source_sha256": "1" * 64,
                "eligible": True,
                "episode_slug": "17-other-company"
                if source_kind == "historical"
                else None,
                "warning": None,
            }
        )


def test_disposition_activation_rules_and_investor_evidence_requirement() -> None:
    valid = RationaleDispositionV44.model_validate_json(
        json.dumps(disposition("founder_execution", "core"))
    )
    assert valid.confidence == 0.8

    no_investor_evidence = disposition("founder_execution", "candidate")
    no_investor_evidence["wiki_evidence_ids"] = []
    with pytest.raises(ValidationError, match="investor evidence"):
        RationaleDispositionV44.model_validate_json(json.dumps(no_investor_evidence))

    question_only = disposition(
        "traction_repeatability_concern", "question_only", question_ids=["Q1"]
    )
    question_only.update(direction="negative", salience="secondary", confidence=0.4)
    question_only["wiki_evidence_ids"] = ["W-execution"]
    with pytest.raises(ValidationError, match="question_only cannot activate"):
        RationaleDispositionV44.model_validate_json(json.dumps(question_only))

    no_question = disposition("traction_repeatability_concern", "question_only")
    with pytest.raises(ValidationError, match="question_only requires"):
        RationaleDispositionV44.model_validate_json(json.dumps(no_question))


def test_adjudication_schema_binds_all_runtime_identifiers() -> None:
    schema = adjudication_v44_json_schema(
        episode_slug="18-rowvigor",
        claim_ids=("C1", "C2"),
        question_ids=("Q1",),
        pitch_evidence_ids=("P-001",),
        investor_evidence_ids=("W-execution", "H-renewal"),
        portfolio_disclosure_ids=("PM-" + "a" * 20,),
        taxonomy_labels=("founder_execution", "product_adoption"),
    )
    row = schema["$defs"]["RationaleDispositionV44"]["properties"]
    assert schema["properties"]["episode_slug"]["const"] == "18-rowvigor"
    assert row["taxonomy_label"]["enum"] == ["founder_execution", "product_adoption"]
    assert row["claim_ids"]["items"]["enum"] == ["C1", "C2"]
    assert row["question_ids"]["items"]["enum"] == ["Q1"]
    assert row["wiki_evidence_ids"]["items"]["enum"] == ["W-execution"]
    assert row["historical_evidence_ids"]["items"]["enum"] == ["H-renewal"]
    assert row["portfolio_disclosure_ids"]["items"]["enum"] == ["PM-" + "a" * 20]


def test_structural_findings_are_stable_and_questions_are_not_failures() -> None:
    claim_map = claim_map_model()
    adjudication_data = adjudication_payload()
    adjudication_data["dispositions"].append(
        {
            **disposition("market_size_assessment", "rejected", claim_ids=["C2"]),
            "wiki_evidence_ids": ["W-x"],
        }
    )
    adjudication = adjudication_model(adjudication_data)
    neighborhood = neighborhood_model()

    findings = adjudication_findings_v44(
        claim_map,
        adjudication,
        neighborhood,
        pitch_evidence_ids=("P-001", "P-002", "P-003", "P-004"),
        investor_evidence_ids=("W-execution", "H-renewal"),
        portfolio_disclosure_ids=(),
    )

    assert "UNHANDLED_CLAIM:C3" in findings
    assert "INACCESSIBLE_INVESTOR_EVIDENCE:W-x" in findings
    assert "LABEL_OUTSIDE_NEIGHBORHOOD:market_size_assessment" in findings
    assert not any("UNHANDLED_QUESTION" in finding for finding in findings)
    assert findings == sorted(set(findings))


def test_neighborhood_manifest_requires_digestable_ordered_records() -> None:
    parsed = neighborhood_model()
    assert parsed.ordered_labels[0] == "traction_repeatability_concern"

    wrong_order = neighborhood_payload()
    wrong_order["ordered_labels"] = list(reversed(wrong_order["ordered_labels"]))
    with pytest.raises(ValidationError, match="ordered_labels"):
        neighborhood_model(wrong_order)


def test_neighborhood_manifest_allows_candidates_tied_at_the_cutoff() -> None:
    tied = neighborhood_payload()
    tied["claim_neighborhoods"][1]["candidates"][1]["rank"] = 1
    parsed = neighborhood_model(tied)
    assert [row.rank for row in parsed.claim_neighborhoods[1].candidates] == [1, 1]


def test_structural_findings_validate_claim_map_constraint_and_portfolio_citations() -> None:
    data = adjudication_payload()
    data["constraint_assessments"] = [
        {
            "constraint_id": "C1",
            "constraint_kind": "stage_or_check_fit",
            "policy_statement": "The fund invests at seed.",
            "status": "possible",
            "severity": "material",
            "pitch_evidence_ids": ["P-003"],
            "wiki_evidence_ids": ["W-missing"],
            "historical_evidence_ids": [],
            "mapped_ids": ["R1"],
            "assessment": "The round may be outside the normal entry point.",
        }
    ]
    data["portfolio_overlap_assessments"] = [
        {
            "portfolio_entity_id": "PE-" + "b" * 20,
            "disclosure_ids": ["PM-" + "a" * 20],
            "pitch_evidence_ids": ["P-003"],
            "overlap_status": "possible_conflict",
            "decision_consequence": "diligence_only",
            "confidence": 0.5,
            "assessment": "The products may overlap.",
        }
    ]
    findings = adjudication_findings_v44(
        claim_map_model(),
        adjudication_model(data),
        neighborhood_model(),
        pitch_evidence_ids=("P-001", "P-002"),
        investor_evidence_ids=("W-execution",),
        portfolio_disclosure_ids=(),
    )
    assert "INACCESSIBLE_PITCH_EVIDENCE:P-003" in findings
    assert "INACCESSIBLE_PITCH_EVIDENCE:P-004" in findings
    assert "INACCESSIBLE_INVESTOR_EVIDENCE:W-missing" in findings
    assert "INACCESSIBLE_PORTFOLIO_DISCLOSURE:PM-" + "a" * 20 in findings


def test_structural_findings_audit_globally_eligible_cross_target_evidence() -> None:
    retrieval = retrieval_payload()
    retrieval["claim_bundles"][2]["portfolio_disclosures"] = [
        {
            "evidence_id": "PM-" + "a" * 20,
            "source_kind": "portfolio",
            "text": "The portfolio includes an adjacent product.",
            "source_locator": "portfolio/disclosures.json#adjacent",
            "query_id": "query-c2",
            "retrieval_modes": ["lexical"],
            "turn_start": None,
            "turn_end": None,
            "decision_status": None,
            "score": 0.6,
            "source_sha256": "4" * 64,
            "eligible": True,
            "episode_slug": None,
            "warning": None,
        }
    ]
    retrieval["evidence_registry"].append(
        {
            "evidence_id": "PM-" + "a" * 20,
            "source_kind": "portfolio",
            "text": "The portfolio includes an adjacent product.",
            "source_locator": "portfolio/disclosures.json#adjacent",
            "source_sha256": "4" * 64,
            "episode_slug": None,
            "turn_start": None,
            "turn_end": None,
            "decision_status": None,
        }
    )
    retrieval["retrieval_actions"].append(
        {
            "action_id": "A3",
            "target_id": "C2",
            "source_kind": "portfolio",
            "action_kind": "read",
            "query_id": "query-c2",
            "query": "renewal uncertainty",
            "result_evidence_ids": ["PM-" + "a" * 20],
            "opened_episode_slugs": [],
            "warnings": [],
        }
    )
    retrieval["claim_bundles"][2]["retrieval_action_ids"] = ["A3"]
    data = adjudication_payload()
    data["dispositions"][0].update(
        pitch_evidence_ids=["P-001"],
        wiki_evidence_ids=[],
        historical_evidence_ids=["H-renewal"],
        portfolio_disclosure_ids=["PM-" + "a" * 20],
    )
    data["dispositions"][1].update(
        claim_ids=["C2"],
        pitch_evidence_ids=["P-002"],
        wiki_evidence_ids=["W-execution"],
    )
    adjudication = adjudication_model(data)
    manifest = retrieval_model(retrieval)
    findings = adjudication_findings_v44(
        claim_map_model(),
        adjudication,
        neighborhood_model(),
        pitch_evidence_ids=("P-001", "P-002", "P-003", "P-004"),
        investor_evidence_ids=("W-execution", "H-renewal"),
        portfolio_disclosure_ids=("PM-" + "a" * 20,),
        retrieval_manifest=manifest,
    )
    assert not any("NOT_RETRIEVED_FOR_TARGET" in row for row in findings)
    reuse = cross_target_evidence_reuse_v44(adjudication, manifest)
    assert [
        (
            row.disposition_position,
            row.taxonomy_label,
            row.target_ids,
            row.evidence_kind,
            row.evidence_id,
        )
        for row in reuse
    ] == [
        (0, "founder_execution", ("C1",), "historical", "H-renewal"),
        (
            0,
            "founder_execution",
            ("C1",),
            "portfolio",
            "PM-" + "a" * 20,
        ),
        (1, "product_adoption", ("C2",), "wiki", "W-execution"),
    ]


def test_target_evidence_check_uses_union_of_all_cited_targets() -> None:
    data = adjudication_payload()
    data["dispositions"][0].update(
        claim_ids=["C1"],
        question_ids=["Q1"],
        pitch_evidence_ids=["P-001", "P-002"],
        wiki_evidence_ids=["W-execution"],
        historical_evidence_ids=["H-renewal"],
    )
    findings = adjudication_findings_v44(
        claim_map_model(),
        adjudication_model(data),
        neighborhood_model(),
        pitch_evidence_ids=("P-001", "P-002", "P-003", "P-004"),
        investor_evidence_ids=("W-execution", "H-renewal"),
        portfolio_disclosure_ids=(),
        retrieval_manifest=retrieval_model(),
    )
    assert not any("NOT_RETRIEVED_FOR_TARGET" in row for row in findings)
    assert not any("PITCH_EVIDENCE_NOT_BOUND_TO_TARGET" in row for row in findings)


def test_build_investigation_assigns_stable_neighborhood_ordered_dual_views() -> None:
    claim_map = claim_map_model()
    adjudication = adjudication_model()
    neighborhood = neighborhood_model()

    kwargs = {
        "episode_slug": "18-rowvigor",
        "claim_map": claim_map,
        "adjudication": adjudication,
        "retrieval_manifest": retrieval_model(),
        "neighborhood": neighborhood,
        "claim_map_sha256": DIGEST_A,
        "adjudication_sha256": DIGEST_B,
        "pitch_evidence_ids": ("P-001", "P-002", "P-003", "P-004"),
    }
    first = build_investigation_v44(**kwargs)
    second = build_investigation_v44(**kwargs)

    assert first == second
    assert [(row.rationale_id, row.taxonomy_label) for row in first.candidate_rationales] == [
        ("R1", "founder_execution"),
        ("R2", "product_adoption"),
    ]
    assert [row.rationale_id for row in first.core_rationales] == ["R1"]
    assert first.rationales == first.core_rationales
    assert {row.rationale_id for row in first.core_rationales} <= {
        row.rationale_id for row in first.candidate_rationales
    }
    assert first.question_only_dispositions[0].taxonomy_label == "traction_repeatability_concern"
    assert first.model_validate_json(first.model_dump_json()) == first


def test_build_investigation_uses_deterministic_fallback_and_rejects_bad_hashes() -> None:
    data = adjudication_payload()
    data["dispositions"].append(
        {
            **disposition("unknown_zeta", "candidate", claim_ids=["C2"]),
            "pitch_evidence_ids": ["P-002"],
            "wiki_evidence_ids": [],
            "historical_evidence_ids": ["H-renewal"],
        }
    )
    final = build_investigation_v44(
        episode_slug="18-rowvigor",
        claim_map=claim_map_model(),
        adjudication=adjudication_model(data),
        retrieval_manifest=retrieval_model(),
        neighborhood=neighborhood_model(),
        claim_map_sha256=DIGEST_A,
        adjudication_sha256=canonical_model_sha256(adjudication_model(data)),
        pitch_evidence_ids=("P-001", "P-002", "P-003", "P-004"),
    )
    assert [row.taxonomy_label for row in final.candidate_rationales][-1] == "unknown_zeta"
    assert "LABEL_OUTSIDE_NEIGHBORHOOD:unknown_zeta" in final.validator_findings
    assert not any(
        value.startswith("INVESTOR_EVIDENCE_NOT_RETRIEVED_FOR_TARGET")
        for value in final.validator_findings
    )
    assert final.investigation_status == "provisional"

    with pytest.raises(ValidationError):
        build_investigation_v44(
            episode_slug="18-rowvigor",
            claim_map=claim_map_model(),
            adjudication=adjudication_model(),
            retrieval_manifest=retrieval_model(),
            neighborhood=neighborhood_model(),
            claim_map_sha256="A" * 64,
            adjudication_sha256=DIGEST_B,
            pitch_evidence_ids=("P-001", "P-002", "P-003", "P-004"),
        )


def test_build_investigation_reconciles_retrieval_targets_with_claim_map() -> None:
    retrieval = retrieval_payload()
    retrieval["claim_bundles"] = [
        row for row in retrieval["claim_bundles"] if row["target_id"] != "C3"
    ]
    with pytest.raises(ValueError, match="retrieval targets do not match claim map"):
        build_investigation_v44(
            episode_slug="18-rowvigor",
            claim_map=claim_map_model(),
            adjudication=adjudication_model(),
            retrieval_manifest=retrieval_model(retrieval),
            neighborhood=neighborhood_model(),
            claim_map_sha256=DIGEST_A,
            adjudication_sha256=DIGEST_B,
            pitch_evidence_ids=("P-001", "P-002", "P-003", "P-004"),
        )


def test_build_investigation_rejects_allowed_pitch_id_without_coverage() -> None:
    with pytest.raises(ValueError, match="missing claim coverage: P-005"):
        build_investigation_v44(
            episode_slug="18-rowvigor",
            claim_map=claim_map_model(),
            adjudication=adjudication_model(),
            retrieval_manifest=retrieval_model(),
            neighborhood=neighborhood_model(),
            claim_map_sha256=DIGEST_A,
            adjudication_sha256=DIGEST_B,
            pitch_evidence_ids=("P-001", "P-002", "P-003", "P-004", "P-005"),
        )


def test_v44_collections_cannot_be_mutated_after_validation() -> None:
    claim_map = claim_map_model()
    with pytest.raises(AttributeError):
        claim_map.material_claims.append(claim_map.material_claims[0])
    with pytest.raises(AttributeError):
        claim_map.material_claims[0].pitch_evidence_ids.append("P-004")

    payload = adjudication_payload()
    payload["constraint_assessments"] = [constraint_assessment_payload()]
    payload["portfolio_overlap_assessments"] = [portfolio_assessment_payload()]
    adjudication = RationaleAdjudicationV44.model_validate_json(json.dumps(payload))
    with pytest.raises(AttributeError):
        adjudication.dispositions.append(adjudication.dispositions[0])
    with pytest.raises(AttributeError):
        adjudication.constraint_assessments[0].mapped_ids.append("R99")
    with pytest.raises(AttributeError):
        adjudication.portfolio_overlap_assessments[0].disclosure_ids.append(
            "PM-" + "c" * 20
        )
    with pytest.raises(ValidationError):
        adjudication.constraint_assessments[0].status = "triggered"
    with pytest.raises(ValidationError):
        adjudication.portfolio_overlap_assessments[0].overlap_status = (
            "direct_conflict"
        )


def test_final_investigation_defensively_freezes_assessment_snapshots() -> None:
    payload = adjudication_payload()
    payload["constraint_assessments"] = [constraint_assessment_payload()]
    payload["portfolio_overlap_assessments"] = [portfolio_assessment_payload()]
    adjudication = adjudication_model(payload)
    final = build_investigation_v44(
        episode_slug="18-rowvigor",
        claim_map=claim_map_model(),
        adjudication=adjudication,
        retrieval_manifest=retrieval_model(),
        neighborhood=neighborhood_model(),
        claim_map_sha256=DIGEST_A,
        adjudication_sha256=canonical_model_sha256(adjudication),
        pitch_evidence_ids=("P-001", "P-002", "P-003", "P-004"),
    )
    assert final.constraint_assessments[0] is not adjudication.constraint_assessments[0]
    assert (
        final.portfolio_overlap_assessments[0]
        is not adjudication.portfolio_overlap_assessments[0]
    )
    object.__setattr__(
        adjudication.constraint_assessments[0], "status", "triggered"
    )
    assert final.constraint_assessments[0].status == "possible"
    with pytest.raises(ValidationError):
        final.constraint_assessments[0].status = "triggered"
    with pytest.raises(ValidationError):
        final.portfolio_overlap_assessments[0].confidence = 0.1

    round_trip = final.model_validate_json(final.model_dump_json())
    assert round_trip.constraint_assessments[0].mapped_ids == ("R1",)
    assert round_trip.portfolio_overlap_assessments[0].disclosure_ids == (
        "PM-" + "a" * 20,
    )
    with pytest.raises(ValidationError):
        round_trip.constraint_assessments[0].status = "triggered"


def test_public_apis_accept_raw_retrieval_manifest_dicts() -> None:
    claim_map = claim_map_model()
    adjudication = adjudication_model()
    neighborhood = neighborhood_model()
    findings = adjudication_findings_v44(
        claim_map,
        adjudication,
        neighborhood,
        pitch_evidence_ids=("P-001", "P-002", "P-003", "P-004"),
        investor_evidence_ids=("W-execution", "H-renewal"),
        portfolio_disclosure_ids=(),
        retrieval_manifest=retrieval_payload(),
    )
    assert findings == ["UNHANDLED_CLAIM:C2", "UNHANDLED_CLAIM:C3"]

    final = build_investigation_v44(
        episode_slug="18-rowvigor",
        claim_map=claim_map,
        adjudication=adjudication,
        retrieval_manifest=retrieval_payload(),
        neighborhood=neighborhood,
        claim_map_sha256=DIGEST_A,
        adjudication_sha256=DIGEST_B,
        pitch_evidence_ids=("P-001", "P-002", "P-003", "P-004"),
    )
    assert final.claim_retrieval_manifest.claim_map_sha256 == DIGEST_A


def test_repeated_label_instances_are_not_structural_duplicates() -> None:
    payload = adjudication_payload()
    repeated = disposition(
        "founder_execution",
        "candidate",
        claim_ids=["C2"],
    )
    repeated.update(
        {
            "pitch_evidence_ids": ["P-002"],
            "historical_evidence_ids": ["H-renewal"],
            "wiki_evidence_ids": [],
            "direction": "negative",
            "salience": "secondary",
            "confidence": 0.65,
            "justification": "Renewal uncertainty is a distinct execution concern.",
        }
    )
    payload["dispositions"].append(repeated)
    adjudication = adjudication_model(payload)

    findings = adjudication_findings_v44(
        claim_map_model(),
        adjudication,
        neighborhood_model(),
        pitch_evidence_ids=("P-001", "P-002", "P-003", "P-004"),
        investor_evidence_ids=("W-execution", "H-renewal"),
        portfolio_disclosure_ids=(),
        retrieval_manifest=retrieval_model(),
    )

    assert "DUPLICATE_DISPOSITION:founder_execution" not in findings
    final = build_investigation_v44(
        episode_slug="18-rowvigor",
        claim_map=claim_map_model(),
        adjudication=adjudication,
        retrieval_manifest=retrieval_model(),
        neighborhood=neighborhood_model(),
        claim_map_sha256=DIGEST_A,
        adjudication_sha256=canonical_model_sha256(adjudication),
        pitch_evidence_ids=("P-001", "P-002", "P-003", "P-004"),
    )
    founder_instances = [
        row
        for row in final.candidate_rationales
        if row.taxonomy_label == "founder_execution"
    ]
    assert [(row.rationale_id, row.direction) for row in founder_instances] == [
        ("R1", "positive"),
        ("R2", "negative"),
    ]


def test_build_investigation_rejects_well_formed_but_false_provenance_hashes() -> None:
    with pytest.raises(ValueError, match="claim map hash does not match"):
        build_investigation_v44(
            episode_slug="18-rowvigor",
            claim_map=claim_map_model(),
            adjudication=adjudication_model(),
            retrieval_manifest=retrieval_model(),
            neighborhood=neighborhood_model(),
            claim_map_sha256="f" * 64,
            adjudication_sha256=DIGEST_B,
            pitch_evidence_ids=("P-001", "P-002", "P-003", "P-004"),
        )

    with pytest.raises(ValueError, match="adjudication hash does not match"):
        build_investigation_v44(
            episode_slug="18-rowvigor",
            claim_map=claim_map_model(),
            adjudication=adjudication_model(),
            retrieval_manifest=retrieval_model(),
            neighborhood=neighborhood_model(),
            claim_map_sha256=DIGEST_A,
            adjudication_sha256="e" * 64,
            pitch_evidence_ids=("P-001", "P-002", "P-003", "P-004"),
        )


def test_build_investigation_finds_dangling_constraint_mapping_ids() -> None:
    payload = adjudication_payload()
    constraint = constraint_assessment_payload()
    constraint["mapped_ids"] = ["R99", "U9"]
    payload["constraint_assessments"] = [constraint]
    adjudication = adjudication_model(payload)
    final = build_investigation_v44(
        episode_slug="18-rowvigor",
        claim_map=claim_map_model(),
        adjudication=adjudication,
        retrieval_manifest=retrieval_model(),
        neighborhood=neighborhood_model(),
        claim_map_sha256=DIGEST_A,
        adjudication_sha256=canonical_model_sha256(adjudication),
        pitch_evidence_ids=("P-001", "P-002", "P-003", "P-004"),
    )
    assert "DANGLING_CONSTRAINT_MAPPED_ID:C1:R99" in final.validator_findings
    assert "DANGLING_CONSTRAINT_MAPPED_ID:C1:U9" in final.validator_findings
    assert final.investigation_status == "provisional"


def test_build_investigation_accepts_live_constraint_mapping_id() -> None:
    payload = adjudication_payload()
    payload["constraint_assessments"] = [constraint_assessment_payload()]
    adjudication = adjudication_model(payload)
    final = build_investigation_v44(
        episode_slug="18-rowvigor",
        claim_map=claim_map_model(),
        adjudication=adjudication,
        retrieval_manifest=retrieval_model(),
        neighborhood=neighborhood_model(),
        claim_map_sha256=DIGEST_A,
        adjudication_sha256=canonical_model_sha256(adjudication),
        pitch_evidence_ids=("P-001", "P-002", "P-003", "P-004"),
    )
    assert not any(
        finding.startswith("DANGLING_CONSTRAINT_MAPPED_ID")
        for finding in final.validator_findings
    )


def test_v44_retrieval_and_taxonomy_scores_must_be_finite() -> None:
    evidence = retrieval_payload()["claim_bundles"][0]["wiki_evidence"][0]
    evidence["score"] = float("nan")
    with pytest.raises(ValidationError):
        RetrievedEvidenceV44.model_validate(evidence)

    candidate = neighborhood_payload()["claim_neighborhoods"][0]["candidates"][0]
    candidate["fused_score"] = float("inf")
    with pytest.raises(ValidationError):
        TaxonomyCandidateV44.model_validate(candidate)


def test_retrieved_evidence_requires_text_valid_span_and_historical_episode() -> None:
    missing_text = retrieval_payload()["claim_bundles"][0]["wiki_evidence"][0]
    missing_text.pop("text")
    with pytest.raises(ValidationError):
        RetrievedEvidenceV44.model_validate_json(json.dumps(missing_text))

    mismatched_span = retrieval_payload()["claim_bundles"][1]["historical_evidence"][0]
    mismatched_span["turn_end"] = None
    with pytest.raises(ValidationError, match="turn bounds"):
        RetrievedEvidenceV44.model_validate_json(json.dumps(mismatched_span))

    missing_episode = retrieval_payload()["claim_bundles"][1]["historical_evidence"][0]
    missing_episode["episode_slug"] = None
    with pytest.raises(ValidationError, match="historical evidence requires"):
        RetrievedEvidenceV44.model_validate_json(json.dumps(missing_episode))

    missing_span = retrieval_payload()["claim_bundles"][1]["historical_evidence"][0]
    missing_span["turn_start"] = None
    missing_span["turn_end"] = None
    with pytest.raises(ValidationError, match="historical evidence requires.*span"):
        RetrievedEvidenceV44.model_validate_json(json.dumps(missing_span))

    missing_status = retrieval_payload()["claim_bundles"][1]["historical_evidence"][0]
    missing_status["decision_status"] = None
    with pytest.raises(ValidationError, match="historical evidence requires.*decision"):
        RetrievedEvidenceV44.model_validate_json(json.dumps(missing_status))


def test_manifest_rejects_duplicate_or_drifting_registry_records() -> None:
    duplicate = retrieval_payload()
    duplicate["evidence_registry"].append(dict(duplicate["evidence_registry"][0]))
    with pytest.raises(ValidationError, match="registry IDs must be unique"):
        retrieval_model(duplicate)

    drift = retrieval_payload()
    drift["evidence_registry"][0]["text"] = "Different source text."
    with pytest.raises(ValidationError, match="does not match registry"):
        retrieval_model(drift)


def test_manifest_rejects_dangling_wrong_target_and_wrong_kind_actions() -> None:
    dangling = retrieval_payload()
    dangling["claim_bundles"][0]["retrieval_action_ids"] = ["A99"]
    with pytest.raises(ValidationError, match="unknown retrieval action"):
        retrieval_model(dangling)

    wrong_target = retrieval_payload()
    wrong_target["retrieval_actions"][0]["target_id"] = "Q1"
    with pytest.raises(ValidationError, match="wrong target"):
        retrieval_model(wrong_target)

    wrong_kind = retrieval_payload()
    wrong_kind["retrieval_actions"][0]["source_kind"] = "historical"
    wrong_kind["retrieval_actions"][0]["opened_episode_slugs"] = [
        "17-other-company"
    ]
    with pytest.raises(ValidationError, match="source kind"):
        retrieval_model(wrong_kind)


def test_manifest_closes_action_ownership_and_bundle_result_chain() -> None:
    orphan = retrieval_payload()
    orphan["retrieval_actions"].append(
        {
            "action_id": "A3",
            "target_id": "C2",
            "source_kind": "wiki",
            "action_kind": "search",
            "query_id": "query-c2",
            "query": "renewal uncertainty",
            "result_evidence_ids": [],
            "opened_episode_slugs": [],
            "warnings": [],
        }
    )
    with pytest.raises(ValidationError, match="orphan retrieval action"):
        retrieval_model(orphan)

    empty_results = retrieval_payload()
    empty_results["retrieval_actions"][0]["result_evidence_ids"] = []
    with pytest.raises(ValidationError, match="not connected to owned action"):
        retrieval_model(empty_results)

    disconnected_query = retrieval_payload()
    disconnected_query["retrieval_actions"][0]["query_id"] = "another-query"
    with pytest.raises(ValidationError, match="not connected to owned action"):
        retrieval_model(disconnected_query)

    duplicate_reference = retrieval_payload()
    duplicate_reference["claim_bundles"][0]["retrieval_action_ids"] = ["A1", "A1"]
    with pytest.raises(ValidationError, match="action IDs must be unique within bundle"):
        retrieval_model(duplicate_reference)


def test_retrieval_audit_records_round_trip_with_text_and_are_immutable() -> None:
    manifest = retrieval_model()
    assert manifest.evidence_registry[0].text.startswith("Execution evidence")
    assert manifest.retrieval_actions[0].result_evidence_ids == ("W-execution",)
    assert retrieval_model(manifest.model_dump(mode="json")) == manifest
    with pytest.raises(AttributeError):
        manifest.evidence_registry.append(manifest.evidence_registry[0])
    with pytest.raises(AttributeError):
        manifest.retrieval_actions[0].result_evidence_ids.append("W-other")


def test_manifest_requires_inverse_action_results_and_registry_references() -> None:
    unmaterialized = retrieval_payload()
    extra = dict(unmaterialized["evidence_registry"][0])
    extra["evidence_id"] = "W-extra"
    unmaterialized["evidence_registry"].append(extra)
    unmaterialized["retrieval_actions"][0]["result_evidence_ids"].append("W-extra")
    with pytest.raises(ValidationError, match="not materialized in owning bundle"):
        retrieval_model(unmaterialized)

    orphan_registry = retrieval_payload()
    orphan = dict(orphan_registry["evidence_registry"][0])
    orphan["evidence_id"] = "W-orphan"
    orphan_registry["evidence_registry"].append(orphan)
    with pytest.raises(ValidationError, match="orphan registry record"):
        retrieval_model(orphan_registry)


def test_manifest_validates_historical_opened_episodes_and_action_uniqueness() -> None:
    wrong_episode = retrieval_payload()
    wrong_episode["retrieval_actions"][1]["opened_episode_slugs"] = [
        "16-different-company"
    ]
    with pytest.raises(ValidationError, match="opened episodes"):
        retrieval_model(wrong_episode)

    duplicate_results = retrieval_payload()
    duplicate_results["retrieval_actions"][0]["result_evidence_ids"] = [
        "W-execution",
        "W-execution",
    ]
    with pytest.raises(ValidationError, match="result evidence IDs must be unique"):
        retrieval_model(duplicate_results)

    duplicate_slugs = retrieval_payload()
    duplicate_slugs["retrieval_actions"][1]["opened_episode_slugs"] = [
        "17-other-company",
        "17-other-company",
    ]
    with pytest.raises(ValidationError, match="opened episode slugs must be unique"):
        retrieval_model(duplicate_slugs)


def test_manifest_rejects_conflicting_query_id_meanings() -> None:
    conflict = retrieval_payload()
    conflict["retrieval_actions"].append(
        {
            "action_id": "A3",
            "target_id": "C2",
            "source_kind": "wiki",
            "action_kind": "search",
            "query_id": "query-c1",
            "query": "a conflicting query string",
            "result_evidence_ids": [],
            "opened_episode_slugs": [],
            "warnings": [],
        }
    )
    conflict["claim_bundles"][2]["retrieval_action_ids"] = ["A3"]
    with pytest.raises(ValidationError, match="query ID maps to conflicting query"):
        retrieval_model(conflict)


def test_manifest_retains_same_evidence_for_distinct_queries_and_actions() -> None:
    payload = retrieval_payload()
    second_result = dict(payload["claim_bundles"][0]["wiki_evidence"][0])
    second_result["query_id"] = "query-c1-secondary"
    payload["claim_bundles"][0]["wiki_evidence"].append(second_result)
    payload["claim_bundles"][0]["retrieval_action_ids"].append("A3")
    payload["retrieval_actions"].append(
        {
            "action_id": "A3",
            "target_id": "C1",
            "source_kind": "wiki",
            "action_kind": "search",
            "query_id": "query-c1-secondary",
            "query": "founder execution evidence",
            "result_evidence_ids": ["W-execution"],
            "opened_episode_slugs": [],
            "warnings": [],
        }
    )

    manifest = retrieval_model(payload)

    assert [row.query_id for row in manifest.claim_bundles[0].wiki_evidence] == [
        "query-c1",
        "query-c1-secondary",
    ]
    assert manifest.claim_bundles[0].retrieval_action_ids == ("A1", "A3")
    assert manifest.retrieval_actions[0].result_evidence_ids == ("W-execution",)
    assert manifest.retrieval_actions[2].result_evidence_ids == ("W-execution",)


def test_bundle_rejects_duplicate_evidence_for_the_same_query() -> None:
    duplicate = retrieval_payload()
    duplicate["claim_bundles"][0]["wiki_evidence"].append(
        dict(duplicate["claim_bundles"][0]["wiki_evidence"][0])
    )

    with pytest.raises(ValidationError, match="evidence and query IDs must be unique"):
        retrieval_model(duplicate)


def test_retrieved_evidence_rejects_duplicate_retrieval_modes() -> None:
    duplicate_modes = retrieval_payload()["claim_bundles"][0]["wiki_evidence"][0]
    duplicate_modes["retrieval_modes"] = ["dense", "dense"]

    with pytest.raises(ValidationError, match="retrieval modes must be unique"):
        RetrievedEvidenceV44.model_validate_json(json.dumps(duplicate_modes))


def test_historical_search_results_do_not_require_opened_episodes() -> None:
    search = retrieval_payload()
    search["retrieval_actions"][1]["action_kind"] = "search"
    search["retrieval_actions"][1]["opened_episode_slugs"] = []

    manifest = retrieval_model(search)

    assert manifest.retrieval_actions[1].result_evidence_ids == ("H-renewal",)
    assert manifest.retrieval_actions[1].opened_episode_slugs == ()
    assert manifest.evidence_registry[1].episode_slug == "17-other-company"


def test_search_actions_reject_opened_episodes() -> None:
    search = retrieval_payload()
    search["retrieval_actions"][1]["action_kind"] = "search"

    with pytest.raises(ValidationError, match="search actions cannot open episodes"):
        retrieval_model(search)


def test_historical_read_rejects_missing_opened_result_episode() -> None:
    missing = retrieval_payload()
    missing["retrieval_actions"][1]["opened_episode_slugs"] = []
    with pytest.raises(ValidationError, match="result episodes must match opened episodes"):
        retrieval_model(missing)


def test_historical_read_rejects_opened_episode_without_result() -> None:
    extra = retrieval_payload()
    extra["retrieval_actions"][1]["opened_episode_slugs"].append(
        "16-unrelated-company"
    )
    with pytest.raises(ValidationError, match="result episodes must match opened episodes"):
        retrieval_model(extra)


def test_manifest_round_trips_shared_exact_and_query_excerpt_text() -> None:
    payload = retrieval_payload()
    excerpt = dict(payload["claim_bundles"][0]["wiki_evidence"][0])
    excerpt.update(
        {
            "query_id": "query-c2",
            "text": "evidence matters",
            "eligible": False,
        }
    )
    payload["claim_bundles"][2].update(
        {
            "wiki_evidence": [excerpt],
            "warnings": [],
            "retrieval_action_ids": ["A3"],
        }
    )
    payload["retrieval_actions"].append(
        {
            "action_id": "A3",
            "target_id": "C2",
            "source_kind": "wiki",
            "action_kind": "search",
            "query_id": "query-c2",
            "query": "renewal uncertainty",
            "result_evidence_ids": ["W-execution"],
            "opened_episode_slugs": [],
            "warnings": [],
        }
    )

    manifest = retrieval_model(payload)
    round_tripped = retrieval_model(manifest.model_dump(mode="json"))

    assert round_tripped == manifest
    assert manifest.claim_bundles[0].wiki_evidence[0].text == (
        "Execution evidence matters to this investor."
    )
    assert manifest.claim_bundles[2].wiki_evidence[0].text == "evidence matters"
    assert sum(
        row.evidence_id == "W-execution" for row in manifest.evidence_registry
    ) == 1


def test_manifest_rejects_truncated_eligible_evidence_text() -> None:
    truncated = retrieval_payload()
    truncated["claim_bundles"][0]["wiki_evidence"][0]["text"] = (
        "Execution evidence"
    )

    with pytest.raises(ValidationError, match="eligible evidence text.*match registry"):
        retrieval_model(truncated)


def test_manifest_rejects_ineligible_text_outside_registry_text() -> None:
    unrelated = retrieval_payload()
    unrelated["claim_bundles"][0]["wiki_evidence"][0].update(
        {"text": "Unrelated search result.", "eligible": False}
    )
    unrelated["retrieval_actions"][0]["action_kind"] = "search"

    with pytest.raises(
        ValidationError, match="ineligible evidence text must be registry substring"
    ):
        retrieval_model(unrelated)
