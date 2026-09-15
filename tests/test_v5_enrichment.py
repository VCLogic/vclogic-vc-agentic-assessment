from __future__ import annotations

from tests.test_rationale_association_annotations import _investigation, _prediction
from vc_clone_graph.v5_enrichment import enrich_investigation_v5


def _full_source() -> dict:
    source = _investigation()
    source.update({
        "questions": [{"question": "Q", "status": "answered", "answer": "A", "evidence_ids": ["W-a"]}],
        "rationales": [
            {"rationale_id": "R1", "taxonomy_label": "founder_execution", "direction": "positive", "salience": "primary", "confidence": .8, "pitch_evidence_ids": ["P-001"], "wiki_evidence_ids": ["W-a"], "historical_evidence_ids": [], "justification": "Execution"},
            {"rationale_id": "R2", "taxonomy_label": "traction_assessment", "direction": "positive", "salience": "secondary", "confidence": .7, "pitch_evidence_ids": ["P-001"], "wiki_evidence_ids": ["W-a"], "historical_evidence_ids": [], "justification": "Traction"},
        ],
        "unmapped_observations": [{"observation_id": "U1", "description": "Other", "pitch_evidence": ["Other"], "evidence_ids": ["W-a"], "decision_relevance": "Relevant"}],
        "conflicts": [], "information_sufficient": True, "sufficiency_assessment": "Enough",
        "searchable_questions": [], "diligence_questions": [], "next_search_objectives": [], "summary": "Summary",
        "reviewed_pitch_evidence_ids": ["P-001", "P-002", "P-003"],
        "constraint_assessments": [], "portfolio_overlap_assessments": [],
    })
    for row in source["material_statement_coverage"]:
        row.update(
            decision_dimension="founder_execution",
            direction="positive",
            constraint_signal="none",
            assessment="Mapped statement",
        )
    return source


def test_enrichment_preserves_source_semantics_and_nests_associations() -> None:
    source = _full_source()
    result = enrich_investigation_v5(
        source, "a" * 64, _prediction(),
        taxonomy_labels={"founder_execution", "traction_assessment", "market_size_assessment", "business_model_viability"},
    )

    assert result.schema_version == "investigation-v5"
    assert result.source_schema_version == "investigation-v4.1"
    assert result.source_investigation_sha256 == "a" * 64
    assert result.rationales[0].model_dump(
        exclude={"associated_rationales"}, exclude_none=True
    ) == source["rationales"][0]
    assert result.rationales[1].associated_rationales[0].taxonomy_label == "market_size_assessment"
    assert result.material_statement_coverage[2].associated_rationales[0].taxonomy_label == "market_size_assessment"
    assert result.unmapped_observations[0].association_status == "unavailable_unmapped"
    assert result.episode_slug not in result.association_training_episode_slugs
