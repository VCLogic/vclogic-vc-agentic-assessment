"""Integrated rationale and decision evaluation adapters and reporting."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

import joblib

from .partial_pooling_evaluation import CompactPrediction
from .partial_pooling_evaluation import (
    fit_partial_pooling_explanatory_model,
    nested_partial_pooling_predictions,
    nested_per_vc_compact_predictions,
)
from .compact_rationale_features import (
    compact_decision_view,
    compact_feature_dictionary,
)
from .phase1_evaluation import Phase1Case, TaxonomyLabel, evaluate_cases
from .phase2_calibration_evaluation import (
    MethodPrediction,
    Phase2CalibrationRecord,
    cluster_bootstrap_intervals,
    cluster_permutation_test,
    classification_metric_rows,
    holm_adjust,
    ranking_metric_rows,
    raw_method_predictions,
)
from .sequential_error_analysis import analyze_error_cases, error_summary_rows


REFERENCE_STATUS = "automated_candidate_not_human_validated"


def phase1_summary_rows(result: Mapping[str, object]) -> list[dict[str, object]]:
    """Adapt canonical Phase 1 metrics without recomputing them."""
    overall = dict(result["overall"])
    attributes = result["attribute_metrics"]
    rows: list[dict[str, object]] = [{
        "scope": "all",
        "vc_slug": "",
        **overall,
        "direction_conditional_agreement": attributes["direction"][
            "conditional_agreement"
        ],
        "salience_conditional_agreement": attributes["salience"][
            "conditional_agreement"
        ],
        "reference_status": REFERENCE_STATUS,
    }]
    for source in result["scope_rows"]:
        if source["dimension"] != "vc":
            continue
        rows.append({
            "scope": "vc",
            "vc_slug": source["value"],
            **dict(source),
            "reference_status": REFERENCE_STATUS,
        })
    return rows


def phase1_label_error_rows(
    result: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return a copy of the canonical per-label Phase 1 diagnostics."""
    return [dict(row) for row in result["per_label_rows"]]


def _learned_method_rows(
    records: Sequence[Phase2CalibrationRecord],
    predictions: Sequence[CompactPrediction],
    *,
    method: str,
    decision: str,
) -> list[MethodPrediction]:
    if len(records) != len(predictions):
        raise ValueError("learned predictions must align with canonical records")
    rows: list[MethodPrediction] = []
    for record, prediction in zip(records, predictions, strict=True):
        if (
            record.vc_slug != prediction.vc_slug
            or record.episode_slug != prediction.episode_slug
            or record.target != prediction.target
        ):
            raise ValueError("learned prediction alignment failure")
        if decision == "balanced":
            predicted = prediction.balanced_decision
        elif decision == "precision":
            predicted = prediction.precision_decision
        elif decision == "score_only":
            predicted = prediction.balanced_decision
        else:
            raise ValueError(f"unknown learned decision selector: {decision}")
        rows.append(MethodPrediction(
            vc_slug=record.vc_slug,
            vc_name=record.vc_name,
            episode_slug=record.episode_slug,
            group=record.group,
            target=record.target,
            method=method,
            score=prediction.score,
            predicted=predicted,
        ))
    return rows


def evaluate_task_endpoints(
    records: Sequence[Phase2CalibrationRecord],
    learned: Mapping[str, Sequence[CompactPrediction]],
    *,
    review_budgets: Sequence[int] = (1, 3, 5, 10, 20),
) -> dict[str, object]:
    """Evaluate thresholded classification separately from continuous ranking."""
    raw = raw_method_predictions(records, "raw_phase2")
    classification_methods: dict[str, list[MethodPrediction]] = {
        "raw_phase2": raw
    }
    ranking_methods: dict[str, list[MethodPrediction]] = {
        "raw_phase2": raw
    }
    for name, predictions in learned.items():
        classification_methods[f"{name}__balanced"] = _learned_method_rows(
            records,
            predictions,
            method=f"{name}__balanced",
            decision="balanced",
        )
        classification_methods[f"{name}__precision"] = _learned_method_rows(
            records,
            predictions,
            method=f"{name}__precision",
            decision="precision",
        )
        ranking_methods[name] = _learned_method_rows(
            records, predictions, method=name, decision="score_only"
        )
    classification = [
        row
        for predictions in classification_methods.values()
        for row in classification_metric_rows(predictions)
    ]
    ranking = [
        row
        for predictions in ranking_methods.values()
        for row in ranking_metric_rows(
            predictions, review_budgets=review_budgets
        )
    ]
    tradeoffs: list[dict[str, object]] = []
    for name in learned:
        balanced = {
            (row["scope"], row["vc_slug"]): row
            for row in classification
            if row["method"] == f"{name}__balanced"
            and row["scope"] != "macro"
        }
        precision = {
            (row["scope"], row["vc_slug"]): row
            for row in classification
            if row["method"] == f"{name}__precision"
            and row["scope"] != "macro"
        }
        for key in sorted(balanced):
            left, right = balanced[key], precision[key]
            tradeoffs.append({
                "method": name,
                "scope": key[0],
                "vc_slug": key[1],
                "balanced_in_precision": left["in_precision"],
                "balanced_in_recall": left["in_recall"],
                "precision_in_precision": right["in_precision"],
                "precision_in_recall": right["in_recall"],
                "delta_false_ins": int(right["fp"]) - int(left["fp"]),
                "delta_missed_ins": int(right["fn"]) - int(left["fn"]),
            })
    return {
        "classification": classification,
        "ranking": ranking,
        "threshold_tradeoffs": tradeoffs,
        "classification_methods": classification_methods,
        "ranking_methods": ranking_methods,
    }


def evaluate_uncertainty_and_significance(
    endpoints: Mapping[str, object],
    *,
    bootstrap_iterations: int = 2_000,
    permutation_iterations: int = 5_000,
    seed: int = 20260816,
) -> dict[str, list[dict[str, object]]]:
    """Evaluate planned classification and prioritization comparisons."""
    classification_methods = endpoints["classification_methods"]
    ranking_methods = endpoints["ranking_methods"]
    uncertainty = []
    for endpoint, methods, metric in (
        ("classification", classification_methods, "balanced_accuracy"),
        ("prioritization", ranking_methods, "average_precision"),
    ):
        rows = cluster_bootstrap_intervals(
            methods,
            raw_method="raw_phase2",
            metrics=(metric,),
            iterations=bootstrap_iterations,
            seed=seed + len(uncertainty),
        )
        uncertainty.extend({"endpoint": endpoint, **row} for row in rows)
    significance: list[dict[str, object]] = []
    comparison_index = 0
    for endpoint, methods, metric in (
        ("classification", classification_methods, "balanced_accuracy"),
        ("prioritization", ranking_methods, "average_precision"),
    ):
        raw = methods["raw_phase2"]
        for name, learned in methods.items():
            if name == "raw_phase2":
                continue
            row = cluster_permutation_test(
                raw,
                learned,
                metric=metric,
                iterations=permutation_iterations,
                seed=seed + 10_000 + comparison_index,
            )
            significance.append({
                "endpoint": endpoint,
                "test": "episode_cluster_randomization",
                **row,
            })
            comparison_index += 1
    adjusted = holm_adjust([float(row["p_value"]) for row in significance])
    for row, value in zip(significance, adjusted, strict=True):
        row["adjusted_p_value"] = value
        row["significant_0_05"] = value < 0.05
    return {"uncertainty": uncertainty, "significance": significance}


def build_benchmark_policy(
    significance: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Select a challenger only after positive Holm-corrected improvement."""
    primary = {
        "classification": "raw_phase2",
        "prioritization": "raw_phase2",
    }
    qualifying: dict[str, list[Mapping[str, object]]] = {
        "classification": [],
        "prioritization": [],
    }
    for row in significance:
        endpoint = str(row["endpoint"])
        if (
            endpoint in qualifying
            and float(row["observed_delta"]) > 0
            and float(row["adjusted_p_value"]) < 0.05
        ):
            qualifying[endpoint].append(row)
    for endpoint, rows in qualifying.items():
        if rows:
            winner = max(rows, key=lambda row: float(row["observed_delta"]))
            primary[endpoint] = str(winner["learned_method"])
    return {
        "schema": "vc-clone-benchmark-policy-v1",
        "scientific_status": "development_benchmark_not_untouched_holdout",
        "primary_methods": primary,
        "replacement_rule": {
            "required_delta": "positive",
            "adjusted_alpha": 0.05,
            "correction": "Holm",
            "classification_metric": "macro_balanced_accuracy",
            "prioritization_metric": "macro_average_precision",
        },
        "reference_status": REFERENCE_STATUS,
    }


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
        temporary.replace(path)
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(name for name in row if name not in fields)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _render_report(analysis: Mapping[str, object], policy: Mapping[str, object]) -> str:
    records = analysis["records"]
    classification = analysis["classification"]
    ranking = analysis["ranking"]
    phase1 = next(
        row for row in analysis["phase1_summary"] if row["scope"] == "all"
    )
    macro_class = [row for row in classification if row["scope"] == "macro"]
    macro_rank = [
        row for row in ranking
        if row["scope"] == "macro" and int(row["review_budget"]) == 5
    ]
    lines = [
        "# Sequential Rationale and Decision Evaluation",
        "",
        "> Development analysis using automated candidate rationale references; not an untouched holdout or human-validated ground truth.",
        "",
        f"Population: **{len(records)} cases**, **{sum(row.target for row in records)} Ins**, "
        f"**{len(records)-sum(row.target for row in records)} Outs**, and "
        f"**{len(set(row.vc_slug for row in records))} VCs**.",
        "",
        "## 1. Phase 1 rationale recovery",
        "",
        f"Exact-label precision is **{float(phase1['micro_precision']):.3f}**, recall "
        f"**{float(phase1['micro_recall']):.3f}**, and F1 "
        f"**{float(phase1['micro_f1']):.3f}**.",
        "",
        "## 2–3. Compact and partial-pooling models",
        "",
        "| Method | Balanced accuracy | In precision | In recall |",
        "|---|---:|---:|---:|",
    ]
    for row in macro_class:
        lines.append(
            f"| {row['method']} | {float(row['balanced_accuracy']):.3f} | "
            f"{float(row['in_precision']):.3f} | {float(row['in_recall']):.3f} |"
        )
    lines.extend([
        "",
        "## 4. Opportunity prioritization",
        "",
        "| Method | AP | ROC AUC | Precision@5 | Recall@5 |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in macro_rank:
        lines.append(
            f"| {row['method']} | {float(row['average_precision']):.3f} | "
            f"{float(row['roc_auc']):.3f} | {float(row['precision_at_k']):.3f} | "
            f"{float(row['recall_at_k']):.3f} |"
        )
    lines.extend([
        "",
        "## 5. Error analysis",
        "",
        f"The case-level file contains **{len(analysis['error_cases'])}** false-In or missed-In rows with observable rationale diagnostics.",
        "",
        "## 6. Frozen development benchmark",
        "",
        f"Classification primary: **{policy['primary_methods']['classification']}**. "
        f"Prioritization primary: **{policy['primary_methods']['prioritization']}**.",
        "",
        "Rationale coefficients and VC deviations are predictive associations, not causal estimates of investor policy.",
        "",
    ])
    return "\n".join(lines)


def write_sequential_analysis(
    analysis: Mapping[str, object],
    output_root: Path,
    *,
    input_paths: Mapping[str, Path],
) -> dict[str, Path]:
    """Write complete sequential artifacts without mutating earlier reports."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "evaluation": output / "evaluation.md",
        "phase1_summary": output / "phase1_summary.csv",
        "phase1_label_errors": output / "phase1_label_errors.csv",
        "compact_features": output / "compact_features.csv",
        "feature_dictionary": output / "feature_dictionary.csv",
        "predictions": output / "predictions.csv",
        "classification": output / "classification_metrics.csv",
        "ranking": output / "ranking_metrics.csv",
        "threshold_tradeoffs": output / "threshold_tradeoffs.csv",
        "uncertainty": output / "uncertainty.csv",
        "significance": output / "significance.csv",
        "coefficients": output / "partial_pooling_coefficients.csv",
        "error_cases": output / "error_cases.csv",
        "error_summary": output / "error_summary.csv",
        "benchmark_policy": output / "benchmark_policy.json",
        "provenance": output / "fold_provenance.json",
        "manifest": output / "manifest.json",
        "model": output / "models" / "partial_pooling.joblib",
    }
    table_keys = (
        "phase1_summary", "phase1_label_errors", "compact_features",
        "feature_dictionary", "predictions", "classification", "ranking",
        "threshold_tradeoffs", "uncertainty", "significance", "coefficients",
        "error_cases", "error_summary",
    )
    for key in table_keys:
        source_key = "prediction_rows" if key == "predictions" else key
        _write_csv(paths[key], analysis[source_key])
    policy = build_benchmark_policy(analysis["significance"])
    _write_json(paths["benchmark_policy"], policy)
    _write_json(paths["provenance"], analysis["provenance"])
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    temporary_model = paths["model"].with_name(".partial_pooling.joblib.tmp")
    joblib.dump(analysis["model_bundle"], temporary_model)
    temporary_model.replace(paths["model"])
    _atomic_text(paths["evaluation"], _render_report(analysis, policy))
    records = analysis["records"]
    inputs = {}
    for name, raw_path in input_paths.items():
        path = Path(raw_path)
        inputs[name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema": analysis["schema"],
        "scientific_status": "development_not_untouched_holdout",
        "population": {
            "cases": len(records),
            "investors": len(set(row.vc_slug for row in records)),
            "ins": sum(row.target for row in records),
            "outs": len(records) - sum(row.target for row in records),
        },
        "api_cost_usd": float(analysis.get("api_cost_usd", 0.0)),
        "inputs": inputs,
        "outputs": sorted(
            str(path.relative_to(output)) for path in paths.values()
        ),
        "preserved_comparators": [
            "raw_phase2", "pooled_phase2", "advanced_phase2",
            "independent_per_vc_models",
        ],
    }
    _write_json(paths["manifest"], manifest)
    return paths


def _prediction_output_rows(
    records: Sequence[Phase2CalibrationRecord],
    learned: Mapping[str, Sequence[CompactPrediction]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append({
            "method": "raw_phase2",
            "vc_slug": record.vc_slug,
            "vc_name": record.vc_name,
            "episode_slug": record.episode_slug,
            "group": record.group,
            "actual_decision": "In" if record.target else "Out",
            "score": record.raw_likelihood,
            "balanced_decision": "In" if record.raw_decision else "Out",
            "precision_decision": "In" if record.raw_decision else "Out",
            "balanced_threshold": "fixed_agent_decision",
            "precision_threshold": "fixed_agent_decision",
            "selected_c": "",
            "deviation_scale": "",
        })
    for method, predictions in learned.items():
        for prediction in predictions:
            rows.append({
                "method": method,
                "vc_slug": prediction.vc_slug,
                "vc_name": prediction.vc_name,
                "episode_slug": prediction.episode_slug,
                "group": prediction.held_group,
                "actual_decision": "In" if prediction.target else "Out",
                "score": prediction.score,
                "balanced_decision": "In" if prediction.balanced_decision else "Out",
                "precision_decision": "In" if prediction.precision_decision else "Out",
                "balanced_threshold": prediction.balanced_threshold,
                "precision_threshold": prediction.precision_threshold,
                "selected_c": prediction.selected_c,
                "deviation_scale": prediction.deviation_scale,
            })
    return rows


def _fold_provenance(
    learned: Mapping[str, Sequence[CompactPrediction]],
) -> dict[str, object]:
    partial_rows: list[dict[str, object]] = []
    per_vc_rows: list[dict[str, object]] = []
    for method, predictions in learned.items():
        if method.startswith("partial_pooling"):
            seen: set[str] = set()
            for prediction in predictions:
                if prediction.held_group in seen:
                    continue
                seen.add(prediction.held_group)
                partial_rows.append({
                    "method": method,
                    "held_group": prediction.held_group,
                    "training_groups": prediction.training_groups,
                    "training_indices": prediction.training_indices,
                })
        else:
            for prediction in predictions:
                per_vc_rows.append({
                    "method": method,
                    "vc_slug": prediction.vc_slug,
                    "held_episode_slug": prediction.episode_slug,
                    "training_episode_slugs": prediction.training_episode_slugs,
                    "training_indices": prediction.training_indices,
                })
    return {
        "schema": "sequential-fold-provenance-v1",
        "partial_pooling_folds": partial_rows,
        "per_vc_folds": per_vc_rows,
    }


def run_sequential_analysis(
    records: Sequence[Phase2CalibrationRecord],
    cases: Sequence[Phase1Case],
    taxonomy: Mapping[str, TaxonomyLabel],
    *,
    inner_splits: int = 3,
    bootstrap_iterations: int = 2_000,
    permutation_iterations: int = 5_000,
    seed: int = 20260816,
    review_budgets: Sequence[int] = (1, 3, 5, 10, 20),
    n_jobs: int = 1,
) -> dict[str, object]:
    """Execute all six local evaluation stages in their prespecified order."""
    if len(records) != len(cases):
        raise ValueError("Phase 1 and Phase 2 populations must have equal size")
    case_keys = {(case.vc_slug, case.episode_slug) for case in cases}
    record_keys = {(record.vc_slug, record.episode_slug) for record in records}
    if case_keys != record_keys:
        raise ValueError("Phase 1 and Phase 2 populations do not align")
    phase1 = evaluate_cases(cases, taxonomy, top_ks=(1, 3, 5, 10))
    learned = {
        "per_vc_compact_decision": nested_per_vc_compact_predictions(
            records, taxonomy, family="decision", inner_splits=inner_splits,
            seed=seed, n_jobs=n_jobs,
        ),
        "per_vc_compact_rationale": nested_per_vc_compact_predictions(
            records, taxonomy, family="rationale", inner_splits=inner_splits,
            seed=seed + 1, n_jobs=n_jobs,
        ),
        "partial_pooling_decision": nested_partial_pooling_predictions(
            records, taxonomy, family="decision", objective="balanced_accuracy",
            inner_splits=inner_splits, seed=seed + 2,
            n_jobs=n_jobs,
        ),
        "partial_pooling_rationale": nested_partial_pooling_predictions(
            records, taxonomy, family="rationale", objective="average_precision",
            inner_splits=inner_splits, seed=seed + 3,
            n_jobs=n_jobs,
        ),
    }
    endpoints = evaluate_task_endpoints(
        records, learned, review_budgets=review_budgets
    )
    statistics = evaluate_uncertainty_and_significance(
        endpoints,
        bootstrap_iterations=bootstrap_iterations,
        permutation_iterations=permutation_iterations,
        seed=seed + 4,
    )
    explanatory = fit_partial_pooling_explanatory_model(
        records, taxonomy, family="decision", inner_splits=inner_splits,
        seed=seed + 5,
    )
    error_cases = analyze_error_cases(
        cases, endpoints["classification_methods"]
    )
    compact_rows = []
    for record in records:
        compact_rows.append({
            "vc_slug": record.vc_slug,
            "episode_slug": record.episode_slug,
            **compact_decision_view(record, taxonomy),
        })
    return {
        "schema": "sequential-rationale-decision-evaluation-v1",
        "records": list(records),
        "phase1_summary": phase1_summary_rows(phase1),
        "phase1_label_errors": phase1_label_error_rows(phase1),
        "compact_features": compact_rows,
        "feature_dictionary": compact_feature_dictionary(taxonomy),
        "prediction_rows": _prediction_output_rows(records, learned),
        "classification": endpoints["classification"],
        "ranking": endpoints["ranking"],
        "threshold_tradeoffs": endpoints["threshold_tradeoffs"],
        "uncertainty": statistics["uncertainty"],
        "significance": statistics["significance"],
        "coefficients": explanatory["coefficient_rows"],
        "error_cases": error_cases,
        "error_summary": error_summary_rows(error_cases),
        "provenance": _fold_provenance(learned),
        "model_bundle": explanatory,
        "api_cost_usd": 0.0,
    }
