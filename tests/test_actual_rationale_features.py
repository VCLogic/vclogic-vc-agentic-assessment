from __future__ import annotations

import pytest

from vc_clone_graph.actual_rationale_features import (
    actual_rationale_feature_dictionary,
    structured_rationale_features,
)


LABELS = ("founder_execution", *(f"label_{index}" for index in range(43)))


def test_equivalent_actual_and_predicted_rationales_have_identical_features() -> None:
    actual = [{
        "rationale_label": "founder_execution",
        "direction": "positive",
        "salience": "primary",
        "confidence": 0.99,
        "activation": "evaluated",
        "decision_link": "supports",
        "evidence": ["VC transcript evidence"],
    }]
    predicted = [{
        "taxonomy_label": "founder_execution",
        "direction": "positive",
        "salience": "primary",
        "confidence": 0.51,
        "justification": "Model prose",
        "pitch_evidence": ["Pitch text"],
    }]

    actual_features = structured_rationale_features(
        actual, LABELS, label_field="rationale_label"
    )
    predicted_features = structured_rationale_features(
        predicted, LABELS, label_field="taxonomy_label"
    )

    assert actual_features == predicted_features
    forbidden = ("confidence", "evidence", "decision", "episode", "source", "vc")
    assert not any(token in key for key in actual_features for token in forbidden)
    assert actual_features["rationale__founder_execution__present"] == 1.0
    assert actual_features["rationale__founder_execution__signed_salience"] == 1.0


def test_duplicate_rationales_are_aggregated_deterministically() -> None:
    rationales = [
        {"taxonomy_label": "founder_execution", "direction": "positive", "salience": "primary"},
        {"taxonomy_label": "founder_execution", "direction": "negative", "salience": "secondary"},
    ]

    result = structured_rationale_features(
        rationales, LABELS, label_field="taxonomy_label"
    )

    assert result["rationale__founder_execution__count"] == 2.0
    assert result["rationale__founder_execution__direction__positive"] == 1.0
    assert result["rationale__founder_execution__direction__negative"] == 1.0
    assert result["rationale__founder_execution__signed_salience"] == 0.5
    assert result["aggregate__distinct_label_count"] == 1.0
    assert result["aggregate__instance_count"] == 2.0


def test_invalid_structured_rationale_is_rejected() -> None:
    with pytest.raises(ValueError, match="direction"):
        structured_rationale_features(
            [{"taxonomy_label": "founder_execution", "direction": "maybe", "salience": "primary"}],
            LABELS,
            label_field="taxonomy_label",
        )


def test_feature_dictionary_matches_emitted_contract() -> None:
    dictionary = actual_rationale_feature_dictionary(LABELS)
    emitted = structured_rationale_features([], LABELS, label_field="taxonomy_label")

    assert {row["feature"] for row in dictionary} == set(emitted)
    assert len(dictionary) == len(emitted)
