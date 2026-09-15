from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

from vc_clone_graph.phase1_calibration_cases import (
    CalibrationCase,
    RawLabelSignal,
    ReferenceAttribute,
)
from vc_clone_graph.phase1_calibration_models import CalibrationPrediction
from vc_clone_graph.phase1_evaluation import (
    Phase1Case,
    PredictedRationale,
    ReferenceRationale,
    TaxonomyLabel,
)
from vc_clone_graph.phase1_v43_diagnostics import (
    build_error_ledger,
    diagnose_intervention,
    select_canary_cases,
    summarize_diagnostics,
)


LABELS = (
    "market_a",
    "market_b",
    "market_c",
    "founder_a",
    "kept_true",
    "removed_true",
    "false_label",
)
FAMILIES = {
    "market_a": "market",
    "market_b": "market",
    "market_c": "market",
    "founder_a": "founder",
    "kept_true": "product",
    "removed_true": "economics",
    "false_label": "competition",
}


def _case() -> CalibrationCase:
    predicted = {
        "market_b": ("positive", "secondary", 0.7),
        "kept_true": ("positive", "secondary", 0.8),
        "removed_true": ("negative", "primary", 0.9),
        "false_label": ("negative", "secondary", 0.6),
    }
    references = {
        "market_a": ("positive", "primary"),
        "market_c": ("negative", "primary"),
        "founder_a": ("positive", "primary"),
        "kept_true": ("negative", "primary"),
        "removed_true": ("negative", "primary"),
    }
    phase1 = Phase1Case(
        vc_slug="vc-one",
        vc_name="VC One",
        episode_slug="1-example",
        actual_decision="In",
        source_tier="newly_extracted",
        source_format="transcript-observed-rationales-v1",
        predicted=tuple(
            PredictedRationale(label, direction, salience, confidence, "basis")
            for label, (direction, salience, confidence) in predicted.items()
        ),
        reference=tuple(
            ReferenceRationale(
                label,
                direction,
                salience,
                0.9,
                "evaluated",
                "decision_reason",
                "explicit",
                ("evidence",),
            )
            for label, (direction, salience) in references.items()
        ),
        artifact_path=Path("artifact.json"),
        reference_path=Path("reference.json"),
    )
    return CalibrationCase(
        vc_slug=phase1.vc_slug,
        vc_name=phase1.vc_name,
        episode_slug=phase1.episode_slug,
        actual_decision=phase1.actual_decision,
        source_tier=phase1.source_tier,
        source_format=phase1.source_format,
        pitch_text="pitch",
        pitch_path=Path("pitch.txt"),
        pitch_sha256="hash",
        labels=LABELS,
        families=FAMILIES,
        raw_signals={
            label: RawLabelSignal(
                predicted=label in predicted,
                confidence=predicted.get(label, ("", "", 0.0))[2],
                direction=predicted.get(label, ("absent", "", 0.0))[0],
                salience=predicted.get(label, ("", "absent", 0.0))[1],
                signed_salience=0.0,
            )
            for label in LABELS
        },
        reference_targets={label: int(label in references) for label in LABELS},
        reference_attributes={
            label: ReferenceAttribute(
                "evaluated", "decision_reason", "explicit", salience
            )
            for label, (_, salience) in references.items()
        },
        phase1_case=phase1,
    )


def _taxonomy() -> dict[str, TaxonomyLabel]:
    definitions = {
        "market_a": "Assessment of market demand and addressable market size.",
        "market_b": "Assessment of market demand and addressable market size.",
        "market_c": "Regulatory compliance requirements for manufacturing licenses.",
    }
    return {
        label: TaxonomyLabel(
            label, definitions.get(label, f"Definition for {label}"), FAMILIES[label]
        )
        for label in LABELS
    }


def test_error_ledger_classifies_observable_failure_types() -> None:
    case = _case()
    filtering = [
        CalibrationPrediction(
            case.vc_slug,
            case.episode_slug,
            label,
            "filtering_only",
            0.8,
            label in {"market_b", "kept_true"},
            case.raw_signals[label].predicted,
            bool(case.reference_targets[label]),
        )
        for label in LABELS
    ]

    rows = build_error_ledger(
        [case], _taxonomy(), filtering, semantic_neighbor_threshold=0.50
    )
    row_by_label = {row.label: row for row in rows}

    assert row_by_label["market_a"].error_type == "taxonomy_neighbor_confusion"
    assert row_by_label["market_a"].neighbor_label == "market_b"
    assert row_by_label["market_c"].error_type == "family_covered_exact_miss"
    assert row_by_label["founder_a"].error_type == "discovery_omission"
    assert row_by_label["kept_true"].filtering_effect == "retained_true"
    assert row_by_label["kept_true"].direction_disagreement is True
    assert row_by_label["kept_true"].salience_disagreement is True
    assert row_by_label["removed_true"].filtering_effect == "harmful_removal"
    assert row_by_label["false_label"].filtering_effect == "correct_removal"


def _diagnostic_population():
    case = _case()
    filtering = [
        CalibrationPrediction(
            case.vc_slug,
            case.episode_slug,
            label,
            "filtering_only",
            0.8,
            label in {"market_b", "kept_true"},
            case.raw_signals[label].predicted,
            bool(case.reference_targets[label]),
        )
        for label in LABELS
    ]
    base = build_error_ledger(
        [case], _taxonomy(), filtering, semantic_neighbor_threshold=0.99
    )
    rows = []
    for vc_index in range(6):
        for case_index in range(4):
            for label_index, row in enumerate(base):
                updates = {
                    "vc_slug": f"vc-{vc_index}",
                    "vc_name": f"VC {vc_index}",
                    "episode_slug": f"{case_index + 1}-case-{vc_index}",
                    "actual_decision": "In" if case_index % 2 == 0 else "Out",
                }
                if case_index == 1:
                    updates |= {
                        "outcome": "false_positive",
                        "error_type": "unsupported_prediction",
                        "raw_predicted": True,
                        "reference": False,
                    }
                elif case_index == 2:
                    is_true = label_index < 2
                    updates |= {
                        "outcome": "true_positive" if is_true else "true_negative",
                        "error_type": "none",
                        "raw_predicted": is_true,
                        "reference": is_true,
                        "direction_disagreement": False,
                        "salience_disagreement": False,
                    }
                elif case_index == 3 and row.outcome == "false_negative":
                    updates |= {
                        "reference_salience": "secondary",
                        "reference_decision_link": "none",
                        "reference_utterance_type": "assessment",
                    }
                rows.append(replace(row, **updates))
    return rows


def test_summary_contains_required_scopes_and_confusion_pairs() -> None:
    result = summarize_diagnostics(_diagnostic_population())

    assert len(result["case_rows"]) == 24
    assert len(result["vc_rows"]) == 6
    assert {row["label"] for row in result["label_rows"]} == set(LABELS)
    assert {row["family"] for row in result["family_rows"]} == set(FAMILIES.values())
    assert any(
        row["reference_label"] == "market_a"
        and row["predicted_neighbor_label"] == "market_b"
        for row in result["confusion_rows"]
    )
    assert result["missed_salient_rows"]


def test_canary_selection_is_three_roles_per_vc_and_order_invariant() -> None:
    ledger = _diagnostic_population()
    case_rows = summarize_diagnostics(ledger)["case_rows"]

    selection = select_canary_cases(case_rows, ledger, per_vc=3)
    reversed_selection = select_canary_cases(
        list(reversed(case_rows)), list(reversed(ledger)), per_vc=3
    )

    assert len(selection) == 18
    assert {row.role for row in selection} == {"coverage", "precision", "control"}
    assert Counter(row.vc_slug for row in selection) == {
        f"vc-{index}": 3 for index in range(6)
    }
    assert selection == reversed_selection
    assert all(row.selection_reason for row in selection)


def test_intervention_diagnosis_combines_close_and_broad_family_coverage() -> None:
    ledger = _diagnostic_population()

    result = diagnose_intervention(ledger)

    assert result["selected_intervention"] == "contrastive_within_family_mapping"
    assert result["family_covered_weighted_share"] >= 0.60
    assert result["supporting_vc_count"] >= 4
    assert result["api_cost_usd"] == 0.0
