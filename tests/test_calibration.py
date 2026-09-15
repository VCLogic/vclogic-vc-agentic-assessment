from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from vc_clone_graph.artifacts import freeze_model
from vc_clone_graph.calibration import (
    CalibrationRecord,
    evaluate_predictions,
    extract_legacy_phase1,
    extract_phase2,
    extract_semantic_phase1,
    load_frozen_population,
    nested_leave_one_out,
    write_analysis,
)
from vc_clone_graph.schemas import (
    DecisionV2,
    DecisionV3,
    InvestigationV2,
    InvestigationV3,
)


def investigation() -> dict:
    return {
        "episode_slug": "1-example",
        "questions": ["Can the founders execute?"],
        "conflicts": ["Market evidence conflicts."],
        "unanswered_questions": ["What is retention?"],
        "saturated": True,
        "summary": "Example",
        "schema_version": "investigation-v2",
        "rationales": [
            {
                "rationale_id": "R1",
                "label": "founder_execution",
                "direction": "positive",
                "salience": "primary",
                "confidence": 0.8,
                "pitch_evidence": ["built the product"],
                "wiki_evidence_ids": ["W-1"],
                "interpretation": "Execution evidence.",
                "evidence_status": "affirmative_positive",
                "constraint_kind": "none",
                "constraint_severity": "none",
                "severity_basis": "No constraint.",
            },
            {
                "rationale_id": "R2",
                "label": "portfolio_fit",
                "direction": "negative",
                "salience": "secondary",
                "confidence": 0.7,
                "pitch_evidence": ["same customers"],
                "wiki_evidence_ids": ["W-2"],
                "interpretation": "Possible overlap.",
                "evidence_status": "affirmative_adverse",
                "constraint_kind": "portfolio_overlap",
                "constraint_severity": "material",
                "severity_basis": "Customer overlap.",
            },
        ],
        "deal_context": {
            "total_round_size": {
                "status": "observed",
                "value": "$1m",
                "pitch_evidence": ["raising $1m"],
            },
            "company_stage": {
                "status": "observed",
                "value": "pre-seed",
                "pitch_evidence": ["pre-seed"],
            },
            "entry_valuation": {"status": "unknown", "value": None, "pitch_evidence": []},
            "possible_investor_check": {"status": "unknown", "value": None, "pitch_evidence": []},
            "lead_required": {"status": "unknown", "value": None, "pitch_evidence": []},
            "ownership_feasibility": {"status": "unknown", "value": None, "pitch_evidence": []},
        },
    }


def decision() -> dict:
    return {
        "schema_version": "decision-v2",
        "episode_slug": "1-example",
        "investigation_sha256": "0" * 64,
        "decision": "In",
        "decision_confidence": 0.66,
        "investment_likelihood": 0.61,
        "ranking_score": 0.72,
        "check_tier": "exploratory_lt_100k",
        "decision_endpoint": "any_check",
        "recommended_check_tier": "exploratory_lt_100k",
        "controlling_rationale_ids": ["R1", "R2"],
        "reversal_conditions": ["Retention fails."],
        "any_check": {
            "decision": "In",
            "likelihood": 0.61,
            "confidence": 0.66,
            "supporting_rationale_ids": ["R1"],
            "opposing_rationale_ids": ["R2"],
            "fatal_constraint_present": False,
            "optionality_explanation": "A small check buys information.",
            "strongest_counterargument": {
                "argument": "The overlap may matter.",
                "rationale_ids": ["R2"],
                "response": "Diligence can resolve it.",
            },
            "reversal_conditions": ["Conflict is confirmed."],
        },
        "standard_check": {
            "decision": "Out",
            "likelihood": 0.27,
            "confidence": 0.82,
            "supporting_rationale_ids": ["R1"],
            "opposing_rationale_ids": ["R2"],
            "failure_rationale_ids": ["R2"],
            "market_gate": {
                "status": "unresolved",
                "controlling_rationale_ids": ["R2"],
                "explanation": "Scale is unresolved.",
            },
            "strongest_counterargument": {
                "argument": "Execution may support a larger check.",
                "rationale_ids": ["R1"],
                "response": "Scale remains unresolved.",
            },
            "upgrade_conditions": ["Show repeatable scale."],
        },
        "risk_ledger": [
            {
                "rationale_id": "R2",
                "risk_type": "missing_evidence",
                "controlling_for_any_check": True,
                "counterevidence": ["The founders executed."],
                "explanation": "Overlap remains unresolved.",
            }
        ],
        "rationale_assessments": [
            {
                "rationale_id": "R1",
                "effective_direction": "positive",
                "decision_weight": "supporting",
                "assessment": "Execution supports a check.",
                "counterevidence": [],
            },
            {
                "rationale_id": "R2",
                "effective_direction": "negative",
                "decision_weight": "decisive",
                "assessment": "Overlap limits conviction.",
                "counterevidence": ["It may be complementary."],
            },
        ],
        "deliberation_steps": [
            {
                "step_id": "D1",
                "endpoint": "any_check",
                "stage": "assessment",
                "question": "Does execution support a check?",
                "rationale_ids": ["R1"],
                "evidence_assessment": "Execution is positive.",
                "likelihood_before": 0.4,
                "likelihood_after": 0.65,
                "effect": "raises",
                "check_tier_implication": "exploratory_lt_100k",
                "decision_update": "A small check is plausible.",
            },
            {
                "step_id": "D2",
                "endpoint": "any_check",
                "stage": "consistency",
                "question": "Does risk defeat the check?",
                "rationale_ids": ["R2"],
                "evidence_assessment": "The risk is not fatal.",
                "likelihood_before": 0.65,
                "likelihood_after": 0.61,
                "effect": "lowers",
                "check_tier_implication": "exploratory_lt_100k",
                "decision_update": "Retain a bounded In.",
            },
            {
                "step_id": "D3",
                "endpoint": "standard_check",
                "stage": "consistency",
                "question": "Does it clear the standard bar?",
                "rationale_ids": ["R1", "R2"],
                "evidence_assessment": "Scale is unresolved.",
                "likelihood_before": 0.4,
                "likelihood_after": 0.27,
                "effect": "lowers",
                "check_tier_implication": "no_check_tier",
                "decision_update": "Standard check remains Out.",
            },
        ],
        "strongest_counterargument": "The overlap may matter.",
        "unresolved_questions": ["Is there a conflict?"],
        "feedback": "Proceed only with a small check.",
    }


def investigation_v3() -> dict:
    candidate = deepcopy(investigation())
    candidate["schema_version"] = "investigation-v3"
    for rationale in candidate["rationales"]:
        rationale["historical_evidence_ids"] = []
        rationale["precedent_interpretation"] = "No precedent is needed."
    labels = [rationale["label"] for rationale in candidate["rationales"]]
    candidate.update(
        {
            "activated_candidates": labels,
            "queried_candidates": labels,
            "rejected_candidates": [],
            "unmapped_candidates": [],
            "taxonomy_dispositions": [
                {
                    "label": label,
                    "disposition": "activated",
                    "basis": "Pitch evidence activates the rationale.",
                }
                for label in labels
            ],
        }
    )
    return candidate


def decision_v3() -> dict:
    candidate = deepcopy(decision())
    candidate["schema_version"] = "decision-v3"
    candidate["stable"] = True
    candidate["decisive_precedents"] = []
    candidate["exception_analogies"] = []
    return candidate


def test_extracts_all_five_model_feature_families() -> None:
    legacy = extract_legacy_phase1(investigation())
    semantic = extract_semantic_phase1(investigation())
    phase2 = extract_phase2(
        decision(),
        phase1_status="provisional",
        phase2_status="accepted",
        phase1_findings=["PLANNER_FALLBACK_USED"],
        phase2_findings=["RISK_LEDGER_COVERAGE_MISMATCH"],
    )

    assert legacy["positive_primary_count"] == 1.0
    assert legacy["negative_secondary_confidence_sum"] == 0.7
    assert legacy["primary_signed_confidence_sum"] == 0.8
    assert semantic["label__founder_execution__signed_confidence"] == 0.8
    assert semantic["severity__material__count"] == 1.0
    assert semantic["severity_salience__material__secondary__count"] == 1.0
    assert semantic["deal__entry_valuation__unknown"] == 1.0
    assert phase2["any_check_likelihood"] == 0.61
    assert phase2["likelihood_gap"] == pytest.approx(0.34)
    assert phase2["risk__missing_evidence__controlling_count"] == 1.0
    assert phase2["assessment__negative__decisive_count"] == 1.0
    assert phase2["ranking_score"] == 0.72
    assert phase2["phase1_provisional"] == 1.0


def _write_population(tmp_path: Path) -> tuple[list[Path], Path]:
    records = []
    labels = []
    for number, target in ((1, "In"), (2, "Out"), (3, "Out")):
        slug = f"{number}-example"
        candidate = deepcopy(investigation())
        candidate["episode_slug"] = slug
        investigation_model = InvestigationV2.model_validate(candidate)
        run_root = tmp_path / "runs" / slug
        _, digest = freeze_model(run_root / "phase1", "investigation", investigation_model)

        decision_candidate = deepcopy(decision())
        decision_candidate["episode_slug"] = slug
        decision_candidate["investigation_sha256"] = digest
        decision_model = DecisionV2.model_validate(decision_candidate)
        freeze_model(run_root / "phase2", "decision", decision_model)
        records.append(
            {
                "episode_slug": slug,
                "status": "completed",
                "artifact_root": str(run_root),
                "phase1_status": "accepted",
                "phase1_findings": [],
                "phase2_status": "accepted",
                "phase2_findings": [],
            }
        )
        labels.append(
            {
                "episode_slug": slug,
                "evaluation_eligible": True,
                "pitch_window_decision": target,
            }
        )
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "progress": {"processed": 3, "total": 3},
                "completed_records": records,
            }
        ),
        encoding="utf-8",
    )
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    return [status_path], labels_path


def test_load_frozen_population_joins_verified_artifacts(tmp_path: Path) -> None:
    status_paths, labels_path = _write_population(tmp_path)

    records = load_frozen_population(status_paths, labels_path)

    assert [record.episode_slug for record in records] == [
        "1-example",
        "2-example",
        "3-example",
    ]
    assert records[0].target == 1
    assert records[0].features["semantic_phase1"]["positive_primary_count"] == 1.0
    assert set(records[0].features) == {
        "legacy_phase1",
        "semantic_phase1",
        "phase2",
        "legacy_plus_phase2",
        "semantic_plus_phase2",
    }


@pytest.mark.parametrize("include_taxonomy_dispositions", [True, False])
def test_load_frozen_population_loads_v3_artifacts(
    tmp_path: Path, include_taxonomy_dispositions: bool
) -> None:
    slug = "1-example"
    run_root = tmp_path / "runs" / slug
    investigation_candidate = investigation_v3()
    investigation_model = InvestigationV3.model_validate(investigation_candidate)
    investigation_path, digest = freeze_model(
        run_root / "phase1", "investigation", investigation_model
    )
    if not include_taxonomy_dispositions:
        legacy_payload = json.loads(investigation_path.read_text(encoding="utf-8"))
        legacy_payload.pop("taxonomy_dispositions")
        investigation_path.write_text(
            json.dumps(
                legacy_payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        digest = sha256(investigation_path.read_bytes()).hexdigest()
        (run_root / "phase1/investigation.sha256").write_text(
            digest + "\n", encoding="ascii"
        )
    frozen_investigation = investigation_path.read_bytes()
    decision_candidate = decision_v3()
    decision_candidate["investigation_sha256"] = digest
    decision_model = DecisionV3.model_validate(decision_candidate)
    freeze_model(run_root / "phase2", "decision", decision_model)
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "progress": {"processed": 1, "total": 1},
                "completed_records": [
                    {
                        "episode_slug": slug,
                        "status": "completed",
                        "artifact_root": str(run_root),
                        "phase1_status": "provisional",
                        "phase1_findings": [],
                        "phase2_status": "provisional",
                        "phase2_findings": [],
                    }
                ],
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
                    "pitch_window_decision": "In",
                }
            ]
        ),
        encoding="utf-8",
    )

    records = load_frozen_population([status_path], labels_path)

    assert records[0].episode_slug == slug
    assert records[0].target == 1
    assert records[0].features["legacy_phase1"]["positive_primary_count"] == 1.0
    assert records[0].features["phase2"]["any_check_likelihood"] == 0.61
    assert "legacy_plus_phase2" in records[0].features
    assert investigation_path.read_bytes() == frozen_investigation


def test_load_frozen_population_rejects_hash_mismatch(tmp_path: Path) -> None:
    status_paths, labels_path = _write_population(tmp_path)
    status = json.loads(status_paths[0].read_text(encoding="utf-8"))
    root = Path(status["completed_records"][0]["artifact_root"])
    (root / "phase1/investigation.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_population(status_paths, labels_path)


def _model_records(*, flip_first: bool = False) -> list[CalibrationRecord]:
    records = []
    for index in range(12):
        target = int(index % 3 == 0)
        if flip_first and index == 0:
            target = 1 - target
        core = {
            "positive_primary_count": float(index % 4),
            "negative_primary_count": float((index + 1) % 3),
            "primary_signed_confidence_sum": float(index - 5) / 5,
        }
        semantic = {
            **core,
            "label__founder_execution__signed_confidence": float(index % 5) / 4,
            "severity__material__count": float(index % 2),
        }
        phase2 = {
            "any_check_in": float(index % 2 == 0),
            "any_check_likelihood": 0.15 + index * 0.05,
            "standard_check_in": 0.0,
            "standard_check_likelihood": 0.05 + index * 0.02,
            "ranking_score": 0.9 - index * 0.04,
        }
        records.append(
            CalibrationRecord(
                episode_slug=f"{index + 1}-example",
                target=target,
                target_decision="In" if target else "Out",
                artifact_root=f"runs/{index + 1}-example",
                phase1_sha256=str(index).zfill(64),
                phase2_sha256=str(index + 1).zfill(64),
                features={
                    "legacy_phase1": core,
                    "semantic_phase1": semantic,
                    "phase2": phase2,
                    "legacy_plus_phase2": {
                        **{f"p1__{key}": value for key, value in core.items()},
                        **{f"p2__{key}": value for key, value in phase2.items()},
                    },
                    "semantic_plus_phase2": {
                        **{f"p1__{key}": value for key, value in semantic.items()},
                        **{f"p2__{key}": value for key, value in phase2.items()},
                    },
                },
            )
        )
    return records


def test_nested_leave_one_out_excludes_held_episode() -> None:
    records = _model_records()

    predictions = nested_leave_one_out(records, "semantic_plus_phase2")

    assert len(predictions) == 12
    for prediction in predictions:
        assert prediction.episode_slug not in prediction.training_slugs
        assert len(prediction.training_slugs) == 11
        assert 0.0 <= prediction.score <= 1.0


def test_held_out_label_cannot_change_its_model_output() -> None:
    original = nested_leave_one_out(_model_records(), "semantic_plus_phase2")[0]
    flipped = nested_leave_one_out(
        _model_records(flip_first=True), "semantic_plus_phase2"
    )[0]

    assert flipped.score == pytest.approx(original.score)
    assert flipped.chosen_c == original.chosen_c
    assert flipped.threshold == pytest.approx(original.threshold)


def test_evaluate_predictions_reports_classification_and_ranking() -> None:
    predictions = nested_leave_one_out(_model_records(), "legacy_phase1")

    metrics = evaluate_predictions(predictions, review_budgets=(5, 10))

    assert set(metrics["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}
    assert set(metrics["ranking"]["precision_at_k"]) == {"5", "10"}
    assert set(metrics["ranking"]["recall_at_k"]) == {"5", "10"}
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["average_precision"] <= 1.0


def test_write_analysis_outputs_all_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "analysis"

    result = write_analysis(_model_records(), output_root)

    expected = {
        "metrics.json",
        "predictions.csv",
        "rankings.csv",
        "feature-manifest.json",
        "report.md",
    }
    assert {path.name for path in output_root.iterdir()} == expected
    metrics = json.loads((output_root / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["methods"]) >= {
        "legacy_phase1",
        "semantic_phase1",
        "phase2",
        "legacy_plus_phase2",
        "semantic_plus_phase2",
    }
    assert result["dataset"]["episodes"] == 12
