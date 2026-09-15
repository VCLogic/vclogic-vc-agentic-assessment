from __future__ import annotations

from copy import deepcopy
import json
from itertools import permutations
from types import SimpleNamespace

from vc_clone_graph.adjudication_postprocess_v44 import (
    canonical_disposition_key_v44,
    finalize_adjudication_v44,
    normalize_adjudication_payload_v44,
    rationale_id_bindings_v44,
    remap_constraint_assessments_v44,
)
from vc_clone_graph.phase1_v44 import (
    RationaleAdjudicationV44,
    RationaleDispositionV44,
)


def _disposition(
    *,
    label: str = "market_entry_path_assessment",
    disposition: str = "core",
    claim_ids: list[str] | None = None,
    question_ids: list[str] | None = None,
    pitch_evidence_ids: list[str] | None = None,
    direction: str | None = "positive",
    salience: str | None = "primary",
    confidence: float | None = 0.8,
) -> dict[str, object]:
    return {
        "taxonomy_label": label,
        "disposition": disposition,
        "claim_ids": ["C1"] if claim_ids is None else claim_ids,
        "question_ids": [] if question_ids is None else question_ids,
        "pitch_evidence_ids": (
            ["P-001"] if pitch_evidence_ids is None else pitch_evidence_ids
        ),
        "wiki_evidence_ids": ["W-1"],
        "historical_evidence_ids": [],
        "portfolio_disclosure_ids": [],
        "direction": direction,
        "salience": salience,
        "confidence": confidence,
        "justification": "The cited evidence supports this application.",
    }


def test_normalization_clears_nonactivating_fields_without_mutating_input() -> None:
    payload = {
        "dispositions": [
            _disposition(
                disposition="question_only",
                claim_ids=[],
                question_ids=["Q1"],
                direction="negative",
                salience="secondary",
                confidence=0.55,
            )
        ]
    }
    original = deepcopy(payload)

    normalized, audit = normalize_adjudication_payload_v44(payload)

    assert normalized is not payload
    assert payload == original
    assert normalized["dispositions"][0]["direction"] is None
    assert normalized["dispositions"][0]["salience"] is None
    assert normalized["dispositions"][0]["confidence"] is None
    assert audit.cleared_nonactivating_dispositions == 1
    assert audit.exact_duplicate_dispositions_removed == 0
    assert audit.removed_disposition_positions == ()


def test_normalization_removes_only_exact_canonical_duplicates() -> None:
    question_with_activation = _disposition(
        disposition="question_only",
        claim_ids=[],
        question_ids=["Q1"],
        direction="neutral",
        salience="secondary",
        confidence=0.4,
    )
    normalized_question = deepcopy(question_with_activation)
    normalized_question.update(
        {"direction": None, "salience": None, "confidence": None}
    )
    payload = {
        "dispositions": [
            _disposition(claim_ids=["C1"], direction="positive"),
            _disposition(
                claim_ids=["C2"],
                pitch_evidence_ids=["P-002"],
                direction="negative",
            ),
            question_with_activation,
            normalized_question,
            _disposition(
                disposition="candidate",
                claim_ids=["C3"],
                pitch_evidence_ids=["P-003"],
                direction="positive",
            ),
            _disposition(
                label="founder_execution",
                disposition="rejected",
                claim_ids=["C1"],
                direction=None,
                salience=None,
                confidence=None,
            ),
        ]
    }

    normalized, audit = normalize_adjudication_payload_v44(payload)

    assert audit.cleared_nonactivating_dispositions == 1
    assert audit.exact_duplicate_dispositions_removed == 1
    assert audit.removed_disposition_positions == (3,)
    assert len(normalized["dispositions"]) == 5
    market_rows = [
        row
        for row in normalized["dispositions"]
        if row["taxonomy_label"] == "market_entry_path_assessment"
    ]
    assert [row["direction"] for row in market_rows] == [
        "positive",
        "negative",
        None,
        "positive",
    ]
    assert [row["disposition"] for row in market_rows] == [
        "core",
        "core",
        "question_only",
        "candidate",
    ]


def test_canonical_key_is_independent_of_mapping_insertion_order() -> None:
    first = _disposition()
    second = dict(reversed(list(first.items())))

    assert canonical_disposition_key_v44(first) == canonical_disposition_key_v44(
        second
    )


def test_malformed_outer_payload_is_left_for_schema_validation() -> None:
    normalized, audit = normalize_adjudication_payload_v44(["not", "an", "object"])

    assert normalized == ["not", "an", "object"]
    assert audit.cleared_nonactivating_dispositions == 0
    assert audit.exact_duplicate_dispositions_removed == 0


def _disposition_model(**overrides: object) -> RationaleDispositionV44:
    payload = _disposition()
    payload.update(overrides)
    return RationaleDispositionV44.model_validate_json(json.dumps(payload))


def test_rationale_instance_ids_are_stable_across_provider_row_order() -> None:
    positive_market = _disposition_model()
    negative_market = _disposition_model(
        claim_ids=["C2"],
        pitch_evidence_ids=["P-002"],
        direction="negative",
    )
    positive_execution = _disposition_model(
        taxonomy_label="founder_execution",
        claim_ids=["C3"],
        pitch_evidence_ids=["P-003"],
    )
    neighborhood = SimpleNamespace(
        ordered_labels=("market_entry_path_assessment", "founder_execution")
    )

    results = [
        [
            (rationale_id, row.taxonomy_label, row.direction, row.claim_ids)
            for rationale_id, row in rationale_id_bindings_v44(order, neighborhood)
        ]
        for order in permutations(
            (positive_market, negative_market, positive_execution)
        )
    ]

    assert all(result == results[0] for result in results)
    assert results[0] == [
        ("R1", "market_entry_path_assessment", "positive", ("C1",)),
        ("R2", "market_entry_path_assessment", "negative", ("C2",)),
        ("R3", "founder_execution", "positive", ("C3",)),
    ]


def test_rationale_instance_order_excludes_nonactivating_dispositions() -> None:
    core = _disposition_model()
    candidate = _disposition_model(
        disposition="candidate",
        claim_ids=["C2"],
        pitch_evidence_ids=["P-002"],
    )
    question_only = _disposition_model(
        disposition="question_only",
        claim_ids=[],
        question_ids=["Q1"],
        pitch_evidence_ids=[],
        direction=None,
        salience=None,
        confidence=None,
    )
    neighborhood = SimpleNamespace(
        ordered_labels=("market_entry_path_assessment",)
    )

    bindings = rationale_id_bindings_v44(
        (question_only, candidate, core), neighborhood
    )

    assert [(identifier, row.disposition) for identifier, row in bindings] == [
        ("R1", "core"),
        ("R2", "candidate"),
    ]


def _constraint(
    *,
    constraint_id: str,
    pitch_evidence_ids: list[str],
    mapped_ids: list[str],
) -> dict[str, object]:
    return {
        "constraint_id": constraint_id,
        "constraint_kind": "stage_or_check_fit",
        "policy_statement": "The investor requires this fit condition.",
        "status": "possible",
        "severity": "material",
        "pitch_evidence_ids": pitch_evidence_ids,
        "wiki_evidence_ids": ["W-1"],
        "historical_evidence_ids": [],
        "mapped_ids": mapped_ids,
        "assessment": "The pitch evidence raises this constraint.",
    }


def _adjudication_for_constraint_remap() -> RationaleAdjudicationV44:
    payload = {
        "schema_version": "rationale-adjudication-v4.4",
        "episode_slug": "18-rowvigor",
        "dispositions": [
            _disposition(claim_ids=["C1"], pitch_evidence_ids=["P-001"]),
            _disposition(
                claim_ids=["C2"],
                pitch_evidence_ids=["P-002"],
                direction="negative",
            ),
            _disposition(
                label="founder_execution",
                disposition="candidate",
                claim_ids=["C3"],
                pitch_evidence_ids=["P-002", "P-003"],
            ),
        ],
        "unmapped_observations": [
            {
                "observation_id": "U10",
                "statement": "A late unmapped observation.",
                "claim_ids": ["C2"],
                "question_ids": [],
                "pitch_evidence_ids": ["P-002"],
                "wiki_evidence_ids": ["W-1"],
                "historical_evidence_ids": [],
                "portfolio_disclosure_ids": [],
                "justification": "The taxonomy has no precise category.",
            },
            {
                "observation_id": "U1",
                "statement": "An early unmapped observation.",
                "claim_ids": ["C2"],
                "question_ids": [],
                "pitch_evidence_ids": ["P-002"],
                "wiki_evidence_ids": ["W-1"],
                "historical_evidence_ids": [],
                "portfolio_disclosure_ids": [],
                "justification": "The taxonomy has no precise category.",
            },
        ],
        "constraint_assessments": [
            _constraint(
                constraint_id="C1",
                pitch_evidence_ids=["P-002"],
                mapped_ids=["R1"],
            ),
            _constraint(
                constraint_id="C2",
                pitch_evidence_ids=["P-099"],
                mapped_ids=["R1"],
            ),
        ],
        "portfolio_overlap_assessments": [],
        "adjudication_status": "provisional",
        "validator_findings": [],
        "requested_retrieval_ids": [],
    }
    return RationaleAdjudicationV44.model_validate_json(json.dumps(payload))


def test_constraint_mappings_are_recomputed_from_all_pitch_evidence_overlap() -> None:
    adjudication = _adjudication_for_constraint_remap()
    neighborhood = SimpleNamespace(
        ordered_labels=("market_entry_path_assessment", "founder_execution")
    )

    remapped, audit = remap_constraint_assessments_v44(
        adjudication, neighborhood
    )

    assert remapped is not adjudication
    assert remapped.constraint_assessments[0].mapped_ids == (
        "R2",
        "R3",
        "U1",
        "U10",
    )
    assert adjudication.constraint_assessments[0].mapped_ids == ("R1",)
    assert audit.constraint_mappings[0].constraint_id == "C1"
    assert audit.constraint_mappings[0].original_mapped_ids == ("R1",)
    assert audit.constraint_mappings[0].recomputed_mapped_ids == (
        "R2",
        "R3",
        "U1",
        "U10",
    )


def test_constraint_without_overlap_is_reported_and_not_fabricated() -> None:
    adjudication = _adjudication_for_constraint_remap()
    neighborhood = SimpleNamespace(
        ordered_labels=("market_entry_path_assessment", "founder_execution")
    )

    remapped, audit = remap_constraint_assessments_v44(
        adjudication, neighborhood
    )

    assert audit.unresolved_constraint_ids == ("C2",)
    unresolved = audit.constraint_mappings[1]
    assert unresolved.original_mapped_ids == ("R1",)
    assert unresolved.recomputed_mapped_ids == ()
    assert remapped.constraint_assessments[1].mapped_ids == ("R1",)


def test_possible_constraint_maps_to_evidence_linked_open_question() -> None:
    payload = {
        "schema_version": "rationale-adjudication-v4.4",
        "episode_slug": "39-this-pitch-is-damn-near-perfect",
        "dispositions": [
            _disposition(claim_ids=["C1"], pitch_evidence_ids=["P-001"]),
            _disposition(
                disposition="question_only",
                claim_ids=["C10"],
                question_ids=["Q4"],
                pitch_evidence_ids=["P-039"],
                direction=None,
                salience=None,
                confidence=None,
            ),
        ],
        "unmapped_observations": [],
        "constraint_assessments": [
            _constraint(
                constraint_id="C10",
                pitch_evidence_ids=["P-039"],
                mapped_ids=["R1"],
            )
        ],
        "portfolio_overlap_assessments": [],
        "adjudication_status": "valid",
        "validator_findings": [],
        "requested_retrieval_ids": ["Q4"],
    }
    adjudication = RationaleAdjudicationV44.model_validate_json(
        json.dumps(payload)
    )
    neighborhood = SimpleNamespace(
        ordered_labels=("market_entry_path_assessment",)
    )

    remapped, audit = remap_constraint_assessments_v44(
        adjudication, neighborhood
    )

    assert remapped.constraint_assessments[0].mapped_ids == ("Q4",)
    assert audit.unresolved_constraint_ids == ()
    assert audit.constraint_mappings[0].original_mapped_ids == ("R1",)
    assert audit.constraint_mappings[0].recomputed_mapped_ids == ("Q4",)
    second, second_audit = remap_constraint_assessments_v44(
        remapped, neighborhood
    )
    assert second == remapped
    assert second_audit.constraint_mappings[0].original_mapped_ids == ("Q4",)


def test_constraint_remapping_is_idempotent() -> None:
    adjudication = _adjudication_for_constraint_remap()
    neighborhood = SimpleNamespace(
        ordered_labels=("market_entry_path_assessment", "founder_execution")
    )

    first, _first_audit = remap_constraint_assessments_v44(
        adjudication, neighborhood
    )
    second, second_audit = remap_constraint_assessments_v44(first, neighborhood)

    assert second == first
    assert second_audit.constraint_mappings[0].original_mapped_ids == (
        "R2",
        "R3",
        "U1",
        "U10",
    )


def test_finalizer_returns_remapped_adjudication_and_reuse_audit() -> None:
    adjudication = _adjudication_for_constraint_remap()
    neighborhood = SimpleNamespace(
        ordered_labels=("market_entry_path_assessment", "founder_execution")
    )
    record = SimpleNamespace(evidence_id="W-1", eligible=True)
    retrieval = SimpleNamespace(
        claim_bundles=(
            SimpleNamespace(
                target_id="C1",
                wiki_evidence=(record,),
                historical_evidence=(),
                portfolio_disclosures=(),
            ),
        )
    )

    finalized, reuse, mapping = finalize_adjudication_v44(
        adjudication, neighborhood, retrieval
    )

    assert finalized.constraint_assessments[0].mapped_ids == (
        "R2",
        "R3",
        "U1",
        "U10",
    )
    assert mapping.unresolved_constraint_ids == ("C2",)
    assert [row.target_ids for row in reuse] == [("C2",), ("C3",)]
