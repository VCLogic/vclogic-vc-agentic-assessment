from vc_clone_graph.phase2_calibration_evaluation import extract_taxonomy_features
from vc_clone_graph.rehearsal_features import (
    counterfactual_feature_maps,
    rehearsal_feature_map,
)
from vc_clone_graph.rehearsal_schemas import EvidenceRef, RationaleState


def _rationale() -> RationaleState:
    return RationaleState(
        rationale_id="R-001",
        taxonomy_label="founder_execution",
        direction="positive",
        salience="primary",
        confidence=0.8,
        assessment="The founder has repeatedly shipped.",
        evidence_refs=(
            EvidenceRef(
                evidence_id="P-001",
                source_kind="pitch",
                excerpt="Three products shipped.",
            ),
            EvidenceRef(
                evidence_id="A-001",
                source_kind="founder_answer",
                excerpt="I led each launch.",
            ),
            EvidenceRef(
                evidence_id="W-001",
                source_kind="wiki",
                excerpt="Charles values evidence of building.",
            ),
            EvidenceRef(
                evidence_id="H-001",
                source_kind="precedent",
                excerpt="A prior founder showed the same behavior.",
            ),
        ),
    )


def test_rehearsal_features_match_canonical_taxonomy_encoding() -> None:
    expected = extract_taxonomy_features(
        {
            "rationales": [
                {
                    "rationale_id": "R-001",
                    "taxonomy_label": "founder_execution",
                    "direction": "positive",
                    "salience": "primary",
                    "confidence": 0.8,
                    "pitch_evidence_ids": ["P-001", "A-001"],
                    "wiki_evidence_ids": ["W-001"],
                    "historical_evidence_ids": ["H-001"],
                }
            ],
            "constraint_assessments": [],
        },
        {"controlling_rationale_ids": []},
        vc_slug="charles-hudson-precursor-ventures",
    )
    expected = {
        key: value
        for key, value in expected.items()
        if key.startswith(("rationale__", "constraint__"))
    }

    actual = rehearsal_feature_map(
        rationales=(_rationale(),),
        investment_likelihood=0.6,
        decision_confidence=0.7,
        direct_decision="In",
    )

    assert {key: actual[key] for key in expected} == expected
    assert actual["investment_likelihood"] == 0.6
    assert actual["decision_confidence"] == 0.7
    assert actual["decision_in"] == 1.0
    assert not any("Three products" in name for name in actual)


def test_founder_answer_text_never_becomes_a_feature_name() -> None:
    features = rehearsal_feature_map(
        rationales=(_rationale(),),
        investment_likelihood=0.5,
        decision_confidence=0.5,
        direct_decision="Out",
    )

    assert all("I led each launch" not in name for name in features)


def test_counterfactuals_replace_only_selected_rationale_state() -> None:
    base = {
        "rationale__market_size_assessment__count": 1.0,
        "rationale__market_size_assessment__signed_confidence": 0.2,
        "rationale__founder_execution__count": 1.0,
    }

    positive, negative = counterfactual_feature_maps(
        base, "market_size_assessment", confidence=0.75
    )

    assert positive["rationale__market_size_assessment__signed_confidence"] == 0.75
    assert negative["rationale__market_size_assessment__signed_confidence"] == -0.75
    assert positive["rationale__market_size_assessment__direction__positive"] == 1.0
    assert negative["rationale__market_size_assessment__direction__negative"] == 1.0
    assert positive["rationale__founder_execution__count"] == 1.0
