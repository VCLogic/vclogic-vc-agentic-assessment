import pytest

from vc_clone_graph.override_calibration import (
    exclude_changed_pitch_records,
    extract_override_features,
)


def test_extract_override_features_captures_exception_and_blocker_structure():
    investigation = {
        "rationales": [
            {
                "label": "founder_market_fit",
                "direction": "positive",
                "salience": "primary",
                "confidence": 0.9,
            },
            {
                "label": "founder_execution",
                "direction": "positive",
                "salience": "secondary",
                "confidence": 0.7,
            },
            {
                "label": "market_size_assessment",
                "direction": "negative",
                "salience": "primary",
                "confidence": 0.8,
            },
        ]
    }
    decision = {
        "decision": "Out",
        "any_check": {
            "decision": "Out",
            "likelihood": 0.3,
            "fatal_constraint_present": False,
            "supporting_rationale_ids": ["R1", "R2"],
            "opposing_rationale_ids": ["R3"],
            "strongest_counterargument": {"rationale_ids": ["R1", "R2"]},
        },
        "standard_check": {
            "likelihood": 0.1,
            "strongest_counterargument": {"rationale_ids": ["R1"]},
        },
        "risk_ledger": [
            {
                "risk_type": "affirmative_adverse",
                "controlling_for_any_check": True,
            },
            {
                "risk_type": "missing_evidence",
                "controlling_for_any_check": False,
            },
        ],
        "exception_analogies": ["one", "two"],
        "recommended_check_tier": "no_check_tier",
    }

    features = extract_override_features(investigation, decision)

    assert features["override__likelihood_gap"] == pytest.approx(0.2)
    assert features["override__fatal_constraint_present"] == 0.0
    assert features["override__out_without_fatal"] == 1.0
    assert features["override__risk__affirmative_adverse__controlling_count"] == 1.0
    assert features["override__risk__missing_evidence__count"] == 1.0
    assert features["override__exception_analogy_count"] == 2.0
    assert features["override__any_counterargument_rationale_count"] == 2.0
    assert features["override__standard_counterargument_rationale_count"] == 1.0
    assert features["override__support_minus_opposition"] == 1.0
    assert features["override__founder_positive_primary_count"] == 1.0
    assert features["override__founder_positive_confidence_sum"] == pytest.approx(1.6)
    assert features["override__founder_negative_count"] == 0.0
    assert features["override__check_tier__no_check_tier"] == 1.0


def test_extract_override_features_is_label_blind():
    features = extract_override_features(
        {"rationales": []},
        {
            "decision": "In",
            "any_check": {
                "decision": "In",
                "likelihood": 0.7,
                "fatal_constraint_present": False,
                "supporting_rationale_ids": ["R1"],
                "opposing_rationale_ids": [],
                "strongest_counterargument": {"rationale_ids": ["R2"]},
            },
            "standard_check": {
                "likelihood": 0.4,
                "strongest_counterargument": {"rationale_ids": ["R2"]},
            },
            "risk_ledger": [],
            "exception_analogies": [],
            "recommended_check_tier": "exploratory_lt_100k",
        },
    )

    assert all("actual" not in name and "target" not in name for name in features)


def test_exclude_changed_pitch_records_filters_before_snapshot():
    status = {
        "episode_count": 3,
        "in_count": 2,
        "out_count": 1,
        "completed_records": [
            {"episode_slug": "clean-in", "actual_decision": "In", "status": "completed"},
            {"episode_slug": "leaky-in", "actual_decision": "In", "status": "completed"},
            {"episode_slug": "clean-out", "actual_decision": "Out", "status": "completed"},
        ],
        "failed_records": [{"episode_slug": "failed"}],
    }

    filtered = exclude_changed_pitch_records(status, {"leaky-in"})

    assert [row["episode_slug"] for row in filtered["completed_records"]] == [
        "clean-in",
        "clean-out",
    ]
    assert filtered["episode_count"] == 2
    assert filtered["in_count"] == 1
    assert filtered["out_count"] == 1
    assert filtered["failed_records"] == []
    assert filtered["leakage_exclusions"] == ["leaky-in"]
