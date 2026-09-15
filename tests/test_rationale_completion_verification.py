from __future__ import annotations

import pytest

from vc_clone_graph.rationale_completion_verification import (
    augment_rationales,
    build_completion_verification_prompt,
    completion_verification_schema,
    validate_completion_verification,
)


CANDIDATES = {
    "founder_skillset": "Whether the founder has the capabilities required.",
    "market_size_assessment": "Whether the addressable market is venture scale.",
}


def _payload():
    return {
        "schema_version": "rationale-completion-verification-v1",
        "episode_slug": "held",
        "dispositions": [
            {
                "taxonomy_label": "founder_skillset",
                "disposition": "activated_core",
                "direction": "positive",
                "salience": "primary",
                "confidence": 0.85,
                "justification": "The founder describes directly relevant operating experience.",
                "pitch_evidence_ids": ["P1"],
                "investor_evidence_ids": ["W1"],
            },
            {
                "taxonomy_label": "market_size_assessment",
                "disposition": "question_only",
                "direction": None,
                "salience": None,
                "confidence": None,
                "justification": "The pitch does not quantify the market.",
                "pitch_evidence_ids": ["P2"],
                "investor_evidence_ids": ["H1"],
            },
        ],
        "validator_findings": [],
    }


def test_schema_is_strict_and_covers_all_dispositions():
    schema = completion_verification_schema()

    assert schema["additionalProperties"] is False
    disposition = schema["$defs"]["HypothesisDisposition"]
    values = disposition["properties"]["disposition"]["enum"]
    assert set(values) == {
        "activated_core", "activated_candidate", "question_only", "rejected", "unmapped"
    }


def test_verification_requires_exact_candidate_coverage_and_promotion_evidence():
    result = validate_completion_verification(
        _payload(),
        episode_slug="held",
        selected_labels=tuple(CANDIDATES),
        pitch_evidence_ids={"P1", "P2"},
        investor_evidence_ids={"W1", "H1"},
    )
    assert len(result.dispositions) == 2

    missing = _payload()
    missing["dispositions"] = missing["dispositions"][:1]
    with pytest.raises(ValueError, match="candidate coverage"):
        validate_completion_verification(
            missing,
            episode_slug="held",
            selected_labels=tuple(CANDIDATES),
            pitch_evidence_ids={"P1", "P2"},
            investor_evidence_ids={"W1", "H1"},
        )

    unsupported = _payload()
    unsupported["dispositions"][0]["investor_evidence_ids"] = []
    with pytest.raises(ValueError, match="promotion requires"):
        validate_completion_verification(
            unsupported,
            episode_slug="held",
            selected_labels=tuple(CANDIDATES),
            pitch_evidence_ids={"P1", "P2"},
            investor_evidence_ids={"W1", "H1"},
        )


def test_prompt_contains_pitch_definitions_and_evidence_but_no_probabilities_or_decision():
    prompt = build_completion_verification_prompt(
        investor_name="Charles Hudson",
        episode_slug="held",
        pitch_evidence=[{"evidence_id": "P1", "text": "Founder operating experience."}],
        canonical_rationales=[{"taxonomy_label": "founder_market_fit"}],
        candidate_definitions=CANDIDATES,
        investor_evidence=[{"evidence_id": "W1", "text": "Charles values founder capability."}],
    )

    assert "Founder operating experience" in prompt
    assert "Whether the founder has" in prompt
    assert "Charles values founder capability" in prompt
    assert "completion_probability" not in prompt
    assert "actual_decision" not in prompt
    assert "reference_rationales" not in prompt


def test_only_evidence_supported_promotions_enter_augmented_set():
    verification = validate_completion_verification(
        _payload(),
        episode_slug="held",
        selected_labels=tuple(CANDIDATES),
        pitch_evidence_ids={"P1", "P2"},
        investor_evidence_ids={"W1", "H1"},
    )
    result = augment_rationales(
        ({"taxonomy_label": "founder_market_fit", "direction": "positive"},),
        verification,
    )

    assert {row.taxonomy_label for row in result.promoted} == {"founder_skillset"}
    assert {row["taxonomy_label"] for row in result.augmented} == {
        "founder_market_fit", "founder_skillset"
    }
    assert "market_size_assessment" not in {
        row["taxonomy_label"] for row in result.augmented
    }

