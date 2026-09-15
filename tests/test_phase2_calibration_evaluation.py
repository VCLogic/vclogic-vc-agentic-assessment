from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vc_clone_graph.artifacts import freeze_model
from vc_clone_graph.phase2_calibration_evaluation import (
    Phase2CalibrationRecord,
    MethodPrediction,
    cluster_bootstrap_intervals,
    cluster_permutation_test,
    cluster_resample_indices,
    classification_metric_rows,
    extract_taxonomy_features,
    holm_adjust,
    load_phase2_records,
    make_repeated_grouped_folds,
    mcnemar_exact,
    nested_grouped_predictions,
    ranking_metric_rows,
    transition_metric_rows,
    validate_phase2_payloads,
)
from vc_clone_graph.schemas_v4 import DecisionV4, InvestigationV4


def _investigation(slug: str) -> dict:
    return {
        "schema_version": "investigation-v4",
        "episode_slug": slug,
        "questions": [{
            "question": "Can this team execute?",
            "status": "answered",
            "answer": "The team launched.",
            "evidence_ids": ["W-001"],
        }],
        "rationales": [{
            "rationale_id": "R1",
            "taxonomy_label": "founder_execution",
            "direction": "positive",
            "salience": "primary",
            "confidence": 0.82,
            "pitch_evidence": ["We launched."],
            "pitch_evidence_ids": ["P-001"],
            "wiki_evidence_ids": ["W-001"],
            "historical_evidence_ids": [],
            "justification": "Execution is material.",
        }],
        "conflicts": [],
        "unmapped_observations": [],
        "information_sufficient": True,
        "sufficiency_assessment": "Enough information exists.",
        "searchable_questions": [],
        "diligence_questions": [],
        "next_search_objectives": [],
        "summary": "The team has executed.",
    }


def _decision(slug: str, digest: str, decision: str, likelihood: float) -> dict:
    return {
        "schema_version": "decision-v4",
        "episode_slug": slug,
        "investigation_sha256": digest,
        "decision": decision,
        "decision_path": "conventional_fit" if decision == "In" else "out",
        "investment_likelihood": likelihood,
        "decision_confidence": 0.74,
        "decision_justification": "The evidence supports the decision.",
        "recommended_check_tier": "exploratory_lt_100k" if decision == "In" else "none",
        "review_priority_score": 0.81,
        "review_priority_reason": "Review priority follows the evidence.",
        "controlling_rationale_ids": ["R1"],
        "evidence_basis": [{
            "source_type": "pitch",
            "source_reference": "The team launched.",
            "evidence_ids": ["P-001"],
            "interpretation": "The team has shipped.",
            "effect_on_decision": "supports",
        }],
        "strongest_opposing_case": {
            "argument": "Scale is unresolved.",
            "response": "That can be tested.",
        },
        "searchable_questions": [],
        "diligence_questions": [],
        "reversal_conditions": [],
        "information_sufficient": True,
        "sufficiency_assessment": "Enough evidence exists.",
        "next_search_objectives": [],
        "phase1_reopen_recommended": False,
        "missing_considerations": [],
        "founder_exception_considered": False,
        "founder_exception_rationale_ids": [],
        "founder_exception_precedent_ids": [],
        "founder_exception_assessment": "No exception was needed.",
    }


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _registry_fixture(tmp_path: Path) -> Path:
    investors: dict[str, object] = {}
    for vc_slug, vc_name, actual, predicted, likelihood in (
        ("alpha", "Alpha", "In", "In", 0.7),
        ("beta", "Beta", "Out", "In", 0.6),
    ):
        slug = "1-shared"
        root = tmp_path / "runs" / vc_slug / slug
        investigation = InvestigationV4.model_validate(_investigation(slug))
        _, digest = freeze_model(root / "phase1", "investigation", investigation)
        decision = DecisionV4.model_validate(
            _decision(slug, digest, predicted, likelihood)
        )
        freeze_model(root / "phase2", "decision", decision)
        _write_json(root / "summary.json", {
            "episode_slug": slug,
            "phase1_status": "accepted",
            "phase1_findings": [],
            "phase2_status": "provisional",
            "phase2_findings": ["QUALITY_NOTE"],
            "decision": decision.model_dump(mode="json"),
        })
        label_path = _write_json(tmp_path / f"{vc_slug}-labels.json", [{
            "episode_slug": slug,
            "pitch_window_decision": actual,
            "evaluation_eligible": True,
        }])
        investors[vc_slug] = {
            "display_name": vc_name,
            "label_file": str(label_path.relative_to(tmp_path)),
            "eligible_count": 1,
            "sources": [{
                "kind": "summary_files",
                "paths": [str((root / "summary.json").relative_to(tmp_path))],
            }],
        }
    return _write_json(tmp_path / "registry.json", {
        "schema": "canonical-vc-evaluation-registry-v1",
        "investors": investors,
    })


def test_load_phase2_records_verifies_and_extracts_both_feature_families(
    tmp_path: Path,
) -> None:
    records = load_phase2_records(_registry_fixture(tmp_path))

    assert [(row.vc_slug, row.episode_slug) for row in records] == [
        ("alpha", "1-shared"),
        ("beta", "1-shared"),
    ]
    assert records[0].group == "1-shared"
    assert records[0].target == 1
    assert records[0].raw_decision == 1
    assert records[0].raw_likelihood == pytest.approx(0.7)
    assert records[0].phase2_features["investment_likelihood"] == pytest.approx(0.7)
    assert "positive_primary_count" not in records[0].phase2_features
    assert records[0].combined_features["positive_primary_count"] == 1.0
    assert records[0].combined_features["vc__alpha"] == 1.0
    assert not any("actual" in name or "target" in name for name in records[0].combined_features)


def test_taxonomy_features_preserve_label_meaning_and_vc_specific_effects() -> None:
    investigation = _investigation("1-shared")
    investigation["rationales"][0]["historical_evidence_ids"] = ["H-001"]
    investigation["constraint_assessments"] = [{
        "constraint_kind": "portfolio_conflict",
        "status": "triggered",
        "severity": "blocking",
    }]
    decision = _decision("1-shared", "a" * 64, "In", 0.7)

    features = extract_taxonomy_features(
        investigation,
        decision,
        vc_slug="alpha",
    )

    assert features["rationale__founder_execution__signed_confidence"] == pytest.approx(0.82)
    assert features["rationale__founder_execution__salience_confidence"] == pytest.approx(0.82)
    assert features["rationale__founder_execution__pitch_evidence_count"] == 1.0
    assert features["rationale__founder_execution__wiki_evidence_count"] == 1.0
    assert features["rationale__founder_execution__historical_evidence_count"] == 1.0
    assert features["rationale__founder_execution__controlling"] == 1.0
    assert features["vc_rationale__alpha__founder_execution__signed_confidence"] == pytest.approx(0.82)
    assert features["constraint__portfolio_conflict__triggered__blocking"] == 1.0
    assert not any("target" in name or "actual" in name for name in features)


def test_load_phase2_records_rejects_corrupt_frozen_artifact(tmp_path: Path) -> None:
    registry = _registry_fixture(tmp_path)
    sidecar = tmp_path / "runs/alpha/1-shared/phase2/decision.sha256"
    sidecar.write_text("0" * 64 + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_phase2_records(registry)


def test_validate_phase2_payloads_accepts_matching_v41_contract() -> None:
    investigation = _investigation("1-shared")
    for rationale in investigation["rationales"]:
        rationale.pop("pitch_evidence")
    investigation.update(
        schema_version="investigation-v4.1",
        reviewed_pitch_evidence_ids=["P-001"],
        material_statement_coverage=[{
            "pitch_evidence_id": "P-001",
            "decision_dimension": "founder_execution",
            "direction": "positive",
            "constraint_signal": "none",
            "mapping_type": "taxonomy_rationale",
            "mapped_ids": ["R1"],
            "assessment": "The launch is material execution evidence.",
        }],
        constraint_assessments=[{
            "constraint_id": "C1",
            "constraint_kind": "category_or_expertise_fit",
            "policy_statement": "The investor must be able to evaluate the company.",
            "status": "not_triggered",
            "severity": "material",
            "pitch_evidence_ids": ["P-001"],
            "wiki_evidence_ids": ["W-001"],
            "historical_evidence_ids": [],
            "mapped_ids": ["R1"],
            "assessment": "No mismatch is established.",
        }],
        portfolio_overlap_assessments=[],
    )
    decision = _decision("1-shared", "a" * 64, "In", 0.7)
    decision.update(
        schema_version="decision-v4.1",
        blocking_constraint_ids=[],
        constraint_assessment_summary="No supplied constraint blocks a check.",
        founder_exception_precedent_match="No exception is used.",
    )

    parsed_investigation, parsed_decision = validate_phase2_payloads(
        json.dumps(investigation), json.dumps(decision)
    )

    assert parsed_investigation["schema_version"] == "investigation-v4.1"
    assert parsed_decision["schema_version"] == "decision-v4.1"


def _model_records() -> list[Phase2CalibrationRecord]:
    records: list[Phase2CalibrationRecord] = []
    for episode in range(1, 31):
        for vc_index, vc_slug in enumerate(("alpha", "beta")):
            target = int((episode + vc_index) % 4 == 0)
            signal = 0.8 if target else 0.2
            raw = int(signal >= 0.5)
            shared = {
                "investment_likelihood": signal,
                "decision_in": float(raw),
                f"vc__{vc_slug}": 1.0,
            }
            records.append(
                Phase2CalibrationRecord(
                    vc_slug=vc_slug,
                    vc_name=vc_slug.title(),
                    episode_slug=f"{episode}-example",
                    group=f"{episode}-example",
                    target=target,
                    raw_decision=raw,
                    raw_likelihood=signal,
                    phase2_features=shared,
                    combined_features={
                        **shared,
                        "positive_primary_count": float(target),
                        "negative_primary_count": float(1 - target),
                    },
                    artifact_root=f"runs/{vc_slug}/{episode}-example",
                    phase1_sha256=str(episode).zfill(64),
                    phase2_sha256=str(episode + 1).zfill(64),
                )
            )
    return records


def test_repeated_grouped_folds_hold_shared_episodes_out_together() -> None:
    records = _model_records()
    folds = make_repeated_grouped_folds(
        records, repeats=2, splits=3, seed=17
    )

    assert len(folds) == 2
    for repetition in folds:
        held_indices: set[int] = set()
        for train, held in repetition:
            train_groups = {records[index].group for index in train}
            held_groups = {records[index].group for index in held}
            assert train_groups.isdisjoint(held_groups)
            assert held_indices.isdisjoint(held)
            held_indices.update(held)
        assert held_indices == set(range(len(records)))


def test_nested_grouped_predictions_are_deterministic_and_leakage_safe() -> None:
    records = _model_records()
    folds = make_repeated_grouped_folds(
        records, repeats=2, splits=3, seed=23
    )

    first = nested_grouped_predictions(
        records,
        "phase2",
        folds=folds,
        inner_splits=3,
        seed=23,
    )
    second = nested_grouped_predictions(
        records,
        "phase2",
        folds=folds,
        inner_splits=3,
        seed=23,
    )

    assert first == second
    assert len(first) == len(records)
    for record, prediction in zip(records, first, strict=True):
        assert prediction.outer_repeats == 2
        assert len(prediction.outer_training_groups) == 2
        assert all(
            record.group not in training
            for training in prediction.outer_training_groups
        )
        assert 0.0 <= prediction.score <= 1.0
        assert prediction.predicted in {0, 1}


def _metric_predictions() -> list[MethodPrediction]:
    rows: list[MethodPrediction] = []
    patterns = {
        "alpha": (
            (1, 1, 0.90),
            (1, 0, 0.40),
            (0, 1, 0.80),
            (0, 0, 0.10),
        ),
        "beta": (
            (1, 1, 0.80),
            (1, 1, 0.70),
            (0, 0, 0.30),
            (0, 0, 0.20),
        ),
    }
    for vc_slug, values in patterns.items():
        for index, (target, predicted, score) in enumerate(values, start=1):
            rows.append(
                MethodPrediction(
                    vc_slug=vc_slug,
                    vc_name=vc_slug.title(),
                    episode_slug=f"{index}-{vc_slug}",
                    group=f"{index}-{vc_slug}",
                    target=target,
                    method="example",
                    score=score,
                    predicted=predicted,
                )
            )
    return rows


def test_classification_metrics_include_per_vc_macro_and_pooled_calibration() -> None:
    rows = classification_metric_rows(_metric_predictions())
    by_scope = {(row["scope"], row["vc_slug"]): row for row in rows}

    alpha = by_scope[("investor", "alpha")]
    assert (alpha["tp"], alpha["fp"], alpha["tn"], alpha["fn"]) == (1, 1, 1, 1)
    assert alpha["balanced_accuracy"] == pytest.approx(0.5)
    assert alpha["in_f1"] == pytest.approx(0.5)
    assert alpha["specificity"] == pytest.approx(0.5)
    assert alpha["brier_score"] == pytest.approx(
        ((0.9 - 1) ** 2 + (0.4 - 1) ** 2 + (0.8 - 0) ** 2 + 0.1**2) / 4
    )
    macro = by_scope[("macro", "")]
    assert macro["balanced_accuracy"] == pytest.approx(0.75)
    assert macro["in_f1"] == pytest.approx(0.75)
    assert by_scope[("pooled_diagnostic", "")]["n"] == 8


def test_ranking_metrics_rank_within_each_investor() -> None:
    rows = ranking_metric_rows(_metric_predictions(), review_budgets=(1, 2))
    by_key = {(row["scope"], row["vc_slug"], row["review_budget"]): row for row in rows}

    assert by_key[("investor", "alpha", 1)]["hits"] == 1
    assert by_key[("investor", "beta", 1)]["hits"] == 1
    assert by_key[("investor", "alpha", 2)]["hits"] == 1
    assert by_key[("investor", "beta", 2)]["hits"] == 2
    macro = by_key[("macro", "", 1)]
    assert macro["precision_at_k"] == pytest.approx(1.0)
    assert macro["recall_at_k"] == pytest.approx(0.5)
    assert macro["average_precision"] == pytest.approx((5 / 6 + 1.0) / 2)


def test_transition_metrics_account_for_each_raw_decision_change() -> None:
    base = _model_records()[:4]
    records = [
        replace(base[0], target=1, raw_decision=1),
        replace(base[1], target=0, raw_decision=1),
        replace(base[2], target=1, raw_decision=0),
        replace(base[3], target=0, raw_decision=0),
    ]
    learned = [
        MethodPrediction(
            vc_slug=row.vc_slug,
            vc_name=row.vc_name,
            episode_slug=row.episode_slug,
            group=row.group,
            target=row.target,
            method="learned",
            score=0.8,
            predicted=prediction,
        )
        for row, prediction in zip(records, (1, 0, 1, 1), strict=True)
    ]

    transitions = transition_metric_rows(records, learned)

    assert transitions[0]["preserved_correct_in"] == 1
    assert transitions[0]["corrected_false_in"] == 1
    assert transitions[0]["rescued_in"] == 1
    assert transitions[0]["introduced_false_in"] == 1
    assert sum(
        transitions[0][name]
        for name in (
            "preserved_correct_in",
            "lost_correct_in",
            "corrected_false_in",
            "remaining_false_in",
            "rescued_in",
            "remaining_missed_in",
            "preserved_correct_out",
            "introduced_false_in",
        )
    ) == 4


def test_cluster_resampling_keeps_all_shared_episode_rows_together() -> None:
    rows = [
        replace(row, group=f"shared-{index // 2}")
        for index, row in enumerate(_metric_predictions())
    ]
    indices = cluster_resample_indices(rows, np.random.default_rng(9))

    for group in {row.group for row in rows}:
        source_count = sum(row.group == group for row in rows)
        sampled_count = sum(rows[index].group == group for index in indices)
        assert sampled_count % source_count == 0


def test_identical_methods_have_zero_bootstrap_delta_and_permutation_p_one() -> None:
    raw = [replace(row, method="raw") for row in _metric_predictions()]
    same = [replace(row, method="same") for row in raw]

    intervals = cluster_bootstrap_intervals(
        {"raw": raw, "same": same},
        raw_method="raw",
        metrics=("balanced_accuracy", "average_precision"),
        iterations=200,
        seed=31,
    )
    deltas = [row for row in intervals if row["kind"] == "paired_delta"]
    assert all(row["estimate"] == pytest.approx(0.0) for row in deltas)
    assert all(row["lower_95"] == pytest.approx(0.0) for row in deltas)
    assert all(row["upper_95"] == pytest.approx(0.0) for row in deltas)
    test = cluster_permutation_test(
        raw,
        same,
        metric="balanced_accuracy",
        iterations=200,
        seed=31,
    )
    assert test["observed_delta"] == pytest.approx(0.0)
    assert test["p_value"] == pytest.approx(1.0)


def test_cluster_permutation_detects_uniformly_better_paired_method() -> None:
    raw: list[MethodPrediction] = []
    learned: list[MethodPrediction] = []
    for vc_slug in ("alpha", "beta"):
        for index in range(20):
            target = index % 2
            common = dict(
                vc_slug=vc_slug,
                vc_name=vc_slug.title(),
                episode_slug=f"{index}-shared",
                group=f"{index}-shared",
                target=target,
            )
            raw.append(MethodPrediction(
                **common,
                method="raw",
                score=float(1 - target),
                predicted=1 - target,
            ))
            learned.append(MethodPrediction(
                **common,
                method="learned",
                score=float(target),
                predicted=target,
            ))

    result = cluster_permutation_test(
        raw,
        learned,
        metric="balanced_accuracy",
        iterations=999,
        seed=13,
    )

    assert result["observed_delta"] == pytest.approx(1.0)
    assert result["p_value"] < 0.05


def test_cluster_permutation_average_precision_matches_report_metric_with_ties() -> None:
    raw = [
        replace(
            row,
            method="raw",
            score=0.6 if row.target or row.predicted else 0.4,
        )
        for row in _metric_predictions()
    ]
    learned = [
        replace(row, method="learned", score=0.9 if row.target else 0.1)
        for row in raw
    ]
    raw_macro = next(
        row
        for row in ranking_metric_rows(raw, review_budgets=(1,))
        if row["scope"] == "macro"
    )["average_precision"]
    learned_macro = next(
        row
        for row in ranking_metric_rows(learned, review_budgets=(1,))
        if row["scope"] == "macro"
    )["average_precision"]

    result = cluster_permutation_test(
        raw,
        learned,
        metric="average_precision",
        iterations=20,
        seed=41,
    )

    assert result["observed_delta"] == pytest.approx(learned_macro - raw_macro)


def test_holm_adjustment_and_exact_mcnemar_are_well_formed() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.20])

    assert all(value >= raw for value, raw in zip(adjusted, (0.01, 0.04, 0.03, 0.20)))
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    result = mcnemar_exact(
        [replace(row, method="raw") for row in _metric_predictions()],
        [
            replace(row, method="learned", predicted=row.target)
            for row in _metric_predictions()
        ],
    )
    assert result["raw_correct_learned_wrong"] == 0
    assert result["raw_wrong_learned_correct"] == 2
    assert 0.0 <= result["p_value"] <= 1.0
