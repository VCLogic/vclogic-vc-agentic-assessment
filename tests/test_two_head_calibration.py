from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from vc_clone_graph.artifacts import freeze_model
from vc_clone_graph.schemas_v4 import DecisionV4, InvestigationV4
from vc_clone_graph.two_head_calibration import (
    V4CalibrationRecord,
    conditional_head_targets,
    evaluate_two_head_predictions,
    extract_compact_v4_features,
    load_v4_calibration_population,
    nested_two_head_leave_one_out,
    select_capped_threshold,
    write_two_head_analysis,
)


def _investigation(slug: str = "18-rowvigor") -> dict:
    return {
        "schema_version": "investigation-v4",
        "episode_slug": slug,
        "questions": [
            {
                "question": "Can this team execute?",
                "status": "partial",
                "answer": "The team launched, while retention remains unknown.",
                "evidence_ids": ["W-founder"],
            }
        ],
        "rationales": [
            {
                "rationale_id": "R1",
                "taxonomy_label": "founder_execution",
                "direction": "positive",
                "salience": "primary",
                "confidence": 0.82,
                "pitch_evidence": ["We built and launched the product."],
                "pitch_evidence_ids": ["P-001"],
                "wiki_evidence_ids": ["W-founder"],
                "historical_evidence_ids": ["H-prior"],
                "justification": "Execution is material.",
            },
            {
                "rationale_id": "R2",
                "taxonomy_label": "market_size_assessment",
                "direction": "negative",
                "salience": "secondary",
                "confidence": 0.61,
                "pitch_evidence": ["The initial segment is narrow."],
                "pitch_evidence_ids": ["P-002"],
                "wiki_evidence_ids": ["W-market"],
                "historical_evidence_ids": [],
                "justification": "Scale remains uncertain.",
            },
        ],
        "conflicts": ["Execution is positive but scale is uncertain."],
        "unmapped_observations": [],
        "information_sufficient": True,
        "sufficiency_assessment": "Enough information exists for a decision.",
        "searchable_questions": [],
        "diligence_questions": ["What is retention?"],
        "next_search_objectives": [],
        "summary": "Execution is positive and scale is uncertain.",
    }


def _decision(slug: str = "18-rowvigor", digest: str = "a" * 64) -> dict:
    return {
        "schema_version": "decision-v4",
        "episode_slug": slug,
        "investigation_sha256": digest,
        "decision": "In",
        "decision_path": "founder_conviction_exception",
        "investment_likelihood": 0.62,
        "decision_confidence": 0.74,
        "decision_justification": "Execution supports a deliberately small check.",
        "recommended_check_tier": "exploratory_lt_100k",
        "review_priority_score": 0.81,
        "review_priority_reason": "The founder warrants direct review.",
        "controlling_rationale_ids": ["R1", "R2"],
        "evidence_basis": [
            {
                "source_type": "pitch",
                "source_reference": "The team launched.",
                "evidence_ids": ["P-001"],
                "interpretation": "The team has shipped.",
                "effect_on_decision": "supports",
            },
            {
                "source_type": "wiki",
                "source_reference": "Founder policy.",
                "evidence_ids": ["W-founder"],
                "interpretation": "Founder quality can support a small check.",
                "effect_on_decision": "context",
            },
        ],
        "strongest_opposing_case": {
            "argument": "Scale is unresolved.",
            "response": "That limits check size.",
        },
        "searchable_questions": [],
        "diligence_questions": ["Can acquisition scale?"],
        "reversal_conditions": ["Retention is weak."],
        "information_sufficient": True,
        "sufficiency_assessment": "Enough evidence exists for a bounded decision.",
        "next_search_objectives": [],
        "phase1_reopen_recommended": False,
        "missing_considerations": [],
        "founder_exception_considered": True,
        "founder_exception_rationale_ids": ["R1"],
        "founder_exception_precedent_ids": ["H-prior"],
        "founder_exception_assessment": "Execution supports a small first check.",
    }


def test_extract_compact_v4_features_is_aggregate_and_label_blind() -> None:
    features = extract_compact_v4_features(
        _investigation(),
        _decision(),
        {
            "phase1_status": "accepted",
            "phase2_status": "provisional",
            "phase1_findings": [],
            "phase2_findings": ["QUALITY_NOTE"],
        },
    )

    assert features["positive_primary_count"] == 1.0
    assert features["negative_secondary_confidence_sum"] == pytest.approx(0.61)
    assert features["historical_evidence_count"] == 1.0
    assert features["decision_in"] == 1.0
    assert features["decision_path__founder_conviction_exception"] == 1.0
    assert features["phase2_provisional"] == 1.0
    assert not any("target" in name or "actual" in name for name in features)
    assert not any("founder_execution" in name for name in features)
    assert all(isinstance(value, float) for value in features.values())


def test_load_v4_population_verifies_frozen_artifacts(tmp_path: Path) -> None:
    slug = "18-rowvigor"
    artifact_root = tmp_path / "run" / slug
    investigation = InvestigationV4.model_validate(_investigation(slug))
    _, investigation_digest = freeze_model(
        artifact_root / "phase1", "investigation", investigation
    )
    decision = DecisionV4.model_validate(_decision(slug, investigation_digest))
    freeze_model(artifact_root / "phase2", "decision", decision)
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "completed_records": [
                    {
                        "episode_slug": slug,
                        "artifact_root": str(artifact_root),
                        "status": "completed",
                        "phase1_status": "accepted",
                        "phase2_status": "accepted",
                        "phase1_findings": [],
                        "phase2_findings": [],
                    }
                ],
                "failed_records": [],
            }
        ),
        encoding="utf-8",
    )
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            [
                {
                    "episode_slug": slug,
                    "evaluation_eligible": True,
                    "pitch_window_decision": "Out",
                }
            ]
        ),
        encoding="utf-8",
    )

    records = load_v4_calibration_population(status_path, labels_path)

    assert len(records) == 1
    assert records[0].target == 0
    assert records[0].original_decision == "In"
    assert records[0].phase1_sha256 == investigation_digest


def test_select_capped_threshold_maximizes_recall_within_cap() -> None:
    selection = select_capped_threshold(
        targets=[1, 1, 0, 0],
        scores=[0.9, 0.6, 0.8, 0.4],
        false_positive_rate_cap=0.5,
    )

    assert selection.threshold == pytest.approx(0.6)
    assert selection.true_positives == 2
    assert selection.false_positives == 1
    assert selection.false_positive_rate == pytest.approx(0.5)


def test_confirmation_models_false_ins_while_rescue_models_missed_ins() -> None:
    records = _model_records()
    confirmation = [record for record in records if record.original_decision == "In"]
    rescue = [record for record in records if record.original_decision == "Out"]

    assert conditional_head_targets(confirmation, "confirmation") == tuple(
        1 - record.target for record in confirmation
    )
    assert conditional_head_targets(rescue, "rescue") == tuple(
        record.target for record in rescue
    )


def _model_records(*, flip_first: bool = False) -> list[V4CalibrationRecord]:
    records: list[V4CalibrationRecord] = []
    # Each original-decision branch contains three actual Ins and three actual Outs.
    for index in range(12):
        original = "In" if index < 6 else "Out"
        target = int(index % 2 == 0)
        if flip_first and index == 0:
            target = 1 - target
        strength = (index % 6) / 5
        records.append(
            V4CalibrationRecord(
                episode_slug=f"{index + 1}-example",
                target=target,
                target_decision="In" if target else "Out",
                original_decision=original,
                original_likelihood=0.35 + 0.3 * strength,
                review_priority_score=0.4 + 0.4 * strength,
                features={
                    "positive_primary_confidence_sum": strength,
                    "negative_primary_confidence_sum": 1.0 - strength,
                    "decision_in": float(original == "In"),
                    "founder_exception_considered": float(index % 3 == 0),
                },
                artifact_root=f"runs/{index + 1}-example",
                phase1_sha256=str(index).zfill(64),
                phase2_sha256=str(index + 1).zfill(64),
            )
        )
    return records


def test_nested_two_head_routes_by_original_decision_and_excludes_held_out() -> None:
    records = _model_records()

    predictions = nested_two_head_leave_one_out(
        records, false_positive_rate_cap=0.5
    )

    assert len(predictions) == len(records)
    for record, prediction in zip(records, predictions, strict=True):
        expected_head = "confirmation" if record.original_decision == "In" else "rescue"
        assert prediction.head == expected_head
        assert prediction.episode_slug not in prediction.training_slugs
        assert len(prediction.training_slugs) == len(records) - 1
        assert all(
            next(item for item in records if item.episode_slug == slug).original_decision
            == record.original_decision
            for slug in prediction.head_training_slugs
        )


def test_held_out_label_cannot_change_two_head_score_or_threshold() -> None:
    original = nested_two_head_leave_one_out(
        _model_records(), false_positive_rate_cap=0.5
    )[0]
    flipped = nested_two_head_leave_one_out(
        _model_records(flip_first=True), false_positive_rate_cap=0.5
    )[0]

    assert flipped.score == pytest.approx(original.score)
    assert flipped.threshold == pytest.approx(original.threshold)
    assert flipped.chosen_c == original.chosen_c


def test_two_head_falls_back_when_conditional_branch_has_one_class() -> None:
    records = _model_records()
    records = [
        record
        if record.original_decision == "Out"
        else V4CalibrationRecord(
            **{
                **record.__dict__,
                "target": 1,
                "target_decision": "In",
            }
        )
        for record in records
    ]

    predictions = nested_two_head_leave_one_out(
        records, false_positive_rate_cap=0.5
    )

    confirmation = [item for item in predictions if item.head == "confirmation"]
    assert all(item.fallback_reason == "insufficient_conditional_classes" for item in confirmation)
    assert all(item.predicted == 1 for item in confirmation)


def test_evaluation_accounts_for_every_decision_transition() -> None:
    records = _model_records()
    predictions = nested_two_head_leave_one_out(
        records, false_positive_rate_cap=0.5
    )

    result = evaluate_two_head_predictions(records, predictions)

    transitions = result["transitions"]
    assert sum(transitions.values()) == len(records)
    assert sum(result["confusion_matrix"].values()) == len(records)
    assert 0.0 <= result["balanced_accuracy"] <= 1.0
    assert result["head_diagnostics"]["confirmation"]["positive_event"] == "false_in"
    assert result["head_diagnostics"]["rescue"]["positive_event"] == "missed_in"


def test_write_two_head_analysis_outputs_reproducibility_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "analysis"

    result = write_two_head_analysis(
        _model_records(),
        output,
        false_positive_rate_caps=(0.25, 0.5),
        source_manifest={"status_path": "status.json", "labels_path": "labels.json"},
    )

    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "metrics.json",
        "predictions.csv",
        "report.md",
    }
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["api_cost_usd"] == 0.0
    assert manifest["population"] == {"episodes": 12, "ins": 6, "outs": 6}
    assert set(result["methods"]) == {"raw_v4", "two_head_fpr_0.25", "two_head_fpr_0.50"}
