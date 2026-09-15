from __future__ import annotations

import json

from vc_clone_graph.sequential_evaluation import (
    build_benchmark_policy,
    evaluate_task_endpoints,
    evaluate_uncertainty_and_significance,
    phase1_label_error_rows,
    phase1_summary_rows,
    write_sequential_analysis,
)
from vc_clone_graph.partial_pooling_evaluation import CompactPrediction
from vc_clone_graph.phase2_calibration_evaluation import Phase2CalibrationRecord


def phase1_result() -> dict[str, object]:
    return {
        "overall": {
            "n": 2,
            "micro_precision": 0.25,
            "micro_recall": 0.75,
            "micro_f1": 0.375,
            "macro_jaccard": 0.2,
        },
        "attribute_metrics": {
            "direction": {"conditional_agreement": 0.6},
            "salience": {"conditional_agreement": 0.5},
        },
        "scope_rows": [
            {
                "dimension": "vc",
                "value": "alpha",
                "n": 2,
                "micro_precision": 0.25,
                "micro_recall": 0.75,
                "micro_f1": 0.375,
                "macro_jaccard": 0.2,
            }
        ],
        "per_label_rows": [
            {
                "label": "founder_execution",
                "coarse_parent": "founder_team",
                "tp": 1,
                "fp": 2,
                "fn": 3,
                "precision": 1 / 3,
                "recall": 0.25,
                "f1": 2 / 7,
            }
        ],
    }


def test_phase1_summary_preserves_coverage_precision_and_reference_status() -> None:
    rows = phase1_summary_rows(phase1_result())

    overall = next(row for row in rows if row["scope"] == "all")
    assert overall["micro_precision"] == 0.25
    assert overall["micro_recall"] == 0.75
    assert overall["reference_status"] == "automated_candidate_not_human_validated"
    assert overall["direction_conditional_agreement"] == 0.6
    assert overall["salience_conditional_agreement"] == 0.5
    vc = next(row for row in rows if row["scope"] == "vc")
    assert vc["vc_slug"] == "alpha"


def test_phase1_label_error_adapter_is_non_lossy() -> None:
    rows = phase1_label_error_rows(phase1_result())

    assert rows == phase1_result()["per_label_rows"]


def endpoint_records() -> list[Phase2CalibrationRecord]:
    rows = []
    for index, (target, raw, likelihood) in enumerate(
        ((0, 0, 0.1), (1, 1, 0.8), (0, 1, 0.6), (1, 0, 0.4)), start=1
    ):
        rows.append(Phase2CalibrationRecord(
            vc_slug="alpha", vc_name="Alpha", episode_slug=f"{index}-example",
            group=f"{index}-example", target=target, raw_decision=raw,
            raw_likelihood=likelihood,
            phase2_features={"investment_likelihood": likelihood},
            combined_features={}, semantic_features={}, artifact_root="runs/alpha",
            phase1_sha256="a" * 64, phase2_sha256="b" * 64,
        ))
    return rows


def endpoint_predictions(records: list[Phase2CalibrationRecord]) -> list[CompactPrediction]:
    result = []
    for index, (record, score) in enumerate(zip(records, (0.1, 0.8, 0.6, 0.4), strict=True)):
        training = tuple(value for value in range(len(records)) if value != index)
        result.append(CompactPrediction(
            vc_slug=record.vc_slug, vc_name=record.vc_name,
            episode_slug=record.episode_slug, held_group=record.group,
            target=record.target, method="partial_pooling_decision_ranking",
            score=score, balanced_decision=int(score >= 0.5),
            precision_decision=int(score >= 0.7), balanced_threshold=0.5,
            precision_threshold=0.7, selected_c=1.0, deviation_scale=0.5,
            training_episode_slugs=tuple(records[i].episode_slug for i in training),
            training_groups=tuple(records[i].group for i in training),
            training_vcs=("alpha",), training_indices=training,
        ))
    return result


def test_task_metrics_keep_thresholded_classification_and_continuous_ranking_separate() -> None:
    records = endpoint_records()
    result = evaluate_task_endpoints(
        records,
        {"partial_pooling": endpoint_predictions(records)},
        review_budgets=(1, 3, 5),
    )

    methods = {row["method"] for row in result["classification"]}
    assert methods == {
        "raw_phase2", "partial_pooling__balanced", "partial_pooling__precision"
    }
    ranking = [
        row for row in result["ranking"]
        if row["method"] == "partial_pooling" and row["scope"] == "investor"
    ]
    assert {row["review_budget"] for row in ranking} == {1, 3, 5}
    assert all("threshold" not in row for row in ranking)
    tradeoff = result["threshold_tradeoffs"][0]
    assert "delta_false_ins" in tradeoff
    assert "delta_missed_ins" in tradeoff


def test_uncertainty_and_significance_cover_both_prespecified_endpoints() -> None:
    records = endpoint_records()
    endpoints = evaluate_task_endpoints(
        records, {"partial_pooling": endpoint_predictions(records)},
        review_budgets=(1, 3),
    )

    result = evaluate_uncertainty_and_significance(
        endpoints, bootstrap_iterations=20, permutation_iterations=20, seed=37
    )

    assert {row["endpoint"] for row in result["significance"]} == {
        "classification", "prioritization"
    }
    assert all("adjusted_p_value" in row for row in result["significance"])
    assert all(row["significant_0_05"] in {True, False} for row in result["significance"])
    assert {row["metric"] for row in result["uncertainty"]} == {
        "balanced_accuracy", "average_precision"
    }


def test_benchmark_policy_replaces_raw_only_for_positive_corrected_result() -> None:
    significance = [
        {
            "endpoint": "classification", "learned_method": "better",
            "observed_delta": 0.05, "adjusted_p_value": 0.04,
        },
        {
            "endpoint": "prioritization", "learned_method": "looks_better",
            "observed_delta": 0.07, "adjusted_p_value": 0.20,
        },
    ]

    policy = build_benchmark_policy(significance)

    assert policy["primary_methods"]["classification"] == "better"
    assert policy["primary_methods"]["prioritization"] == "raw_phase2"
    assert policy["replacement_rule"]["adjusted_alpha"] == 0.05


def test_writer_emits_complete_sequential_artifacts(tmp_path) -> None:
    records = endpoint_records()
    endpoints = evaluate_task_endpoints(
        records, {"partial_pooling": endpoint_predictions(records)},
        review_budgets=(1, 3),
    )
    stats = evaluate_uncertainty_and_significance(
        endpoints, bootstrap_iterations=10, permutation_iterations=10, seed=41
    )
    source = tmp_path / "registry.json"
    source.write_text('{"schema": "test"}\n', encoding="utf-8")
    analysis = {
        "schema": "sequential-rationale-decision-evaluation-v1",
        "records": records,
        "phase1_summary": [{
            "scope": "all", "micro_precision": 0.2,
            "micro_recall": 0.5, "micro_f1": 2 / 7,
        }],
        "phase1_label_errors": [{"label": "founder_execution", "fp": 1}],
        "compact_features": [
            {"vc_slug": row.vc_slug, "episode_slug": row.episode_slug,
             "rationale_core__founder_execution": 0.5}
            for row in records
        ],
        "feature_dictionary": [{"feature": "rationale_core__founder_execution"}],
        "prediction_rows": [
            {"method": "raw_phase2", "vc_slug": row.vc_slug,
             "episode_slug": row.episode_slug, "target": row.target}
            for row in records
        ],
        "classification": endpoints["classification"],
        "ranking": endpoints["ranking"],
        "threshold_tradeoffs": endpoints["threshold_tradeoffs"],
        "uncertainty": stats["uncertainty"],
        "significance": stats["significance"],
        "coefficients": [{"effect_scope": "global", "feature": "founder_execution"}],
        "error_cases": [],
        "error_summary": [],
        "provenance": {"partial_pooling_folds": []},
        "model_bundle": {"schema": "test-model"},
        "api_cost_usd": 0.0,
    }

    paths = write_sequential_analysis(
        analysis, tmp_path / "report", input_paths={"registry": source}
    )

    required = {
        "evaluation", "phase1_summary", "phase1_label_errors",
        "compact_features", "feature_dictionary", "predictions",
        "classification", "ranking", "threshold_tradeoffs",
        "uncertainty", "significance", "coefficients", "error_cases",
        "error_summary", "benchmark_policy", "provenance", "manifest", "model",
    }
    assert required <= paths.keys()
    assert all(paths[name].is_file() for name in required)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["population"] == {
        "cases": 4, "investors": 1, "ins": 2, "outs": 2,
    }
    assert manifest["api_cost_usd"] == 0.0
