"""Evaluation and diagnostics for models learned from observed VC rationales."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .actual_rationale_cases import RationaleModelCase
from .actual_rationale_cases import build_rationale_model_cases
from .actual_rationale_features import actual_rationale_feature_dictionary
from .actual_rationale_models import RationalePrediction, run_all_rationale_models
from .phase1_evaluation import Phase1Case, TaxonomyLabel
from .phase2_calibration_evaluation import (
    MethodPrediction,
    Phase2CalibrationRecord,
    cluster_permutation_test,
    holm_adjust,
)
from .sequential_evaluation import (
    evaluate_task_endpoints,
    evaluate_uncertainty_and_significance,
)


PRIMARY_CLASSIFICATION = "per_vc_logistic__actual_to_predicted__balanced"
PRIMARY_RANKING = "per_vc_logistic__actual_to_predicted"


def align_rationale_cases(
    records: Sequence[Phase2CalibrationRecord],
    cases: Sequence[RationaleModelCase],
) -> list[RationaleModelCase]:
    """Align cases to canonical Phase 2 order using explicit identity keys."""
    by_key: dict[tuple[str, str], RationaleModelCase] = {}
    for case in cases:
        key = (case.vc_slug, case.episode_slug)
        if key in by_key:
            raise ValueError(f"duplicate rationale case during alignment: {key}")
        by_key[key] = case
    expected = {(record.vc_slug, record.episode_slug) for record in records}
    if set(by_key) != expected:
        raise ValueError(
            f"rationale alignment coverage mismatch; missing={sorted(expected-set(by_key))} "
            f"unexpected={sorted(set(by_key)-expected)}"
        )
    aligned = [by_key[(record.vc_slug, record.episode_slug)] for record in records]
    for record, case in zip(records, aligned, strict=True):
        if record.target != case.target:
            raise ValueError(
                f"rationale alignment outcome mismatch: {record.vc_slug}/{record.episode_slug}"
            )
    return aligned


def _aligned(
    records: Sequence[Phase2CalibrationRecord], cases: Sequence[RationaleModelCase]
) -> None:
    if len(records) != len(cases):
        raise ValueError("rationale cases and Phase 2 records must have equal size")
    for record, case in zip(records, cases, strict=True):
        if (
            record.vc_slug,
            record.episode_slug,
            record.target,
        ) != (case.vc_slug, case.episode_slug, case.target):
            raise ValueError("rationale case and Phase 2 record alignment failure")


def rationale_difference_rows(
    cases: Sequence[RationaleModelCase],
) -> list[dict[str, object]]:
    """Describe observable Phase 1 omissions, additions, and attribute conflicts."""
    rows: list[dict[str, object]] = []
    for case in cases:
        actual_by_label: dict[str, set[tuple[str, str]]] = {}
        predicted_by_label: dict[str, set[tuple[str, str]]] = {}
        for label, direction, salience in case.actual_items:
            actual_by_label.setdefault(label, set()).add((direction, salience))
        for label, direction, salience in case.predicted_items:
            predicted_by_label.setdefault(label, set()).add((direction, salience))
        actual_labels, predicted_labels = set(actual_by_label), set(predicted_by_label)
        shared = actual_labels & predicted_labels
        direction_conflicts = sorted(
            label for label in shared
            if {item[0] for item in actual_by_label[label]}
            != {item[0] for item in predicted_by_label[label]}
        )
        salience_conflicts = sorted(
            label for label in shared
            if {item[1] for item in actual_by_label[label]}
            != {item[1] for item in predicted_by_label[label]}
        )
        rows.append({
            "vc_slug": case.vc_slug,
            "vc_name": case.vc_name,
            "episode_slug": case.episode_slug,
            "actual_decision": "In" if case.target else "Out",
            "source_tier": case.source_tier,
            "source_format": case.source_format,
            "actual_label_count": len(actual_labels),
            "predicted_label_count": len(predicted_labels),
            "shared_label_count": len(shared),
            "actual_only_count": len(actual_labels - predicted_labels),
            "predicted_only_count": len(predicted_labels - actual_labels),
            "actual_only_labels": "|".join(sorted(actual_labels - predicted_labels)),
            "predicted_only_labels": "|".join(sorted(predicted_labels - actual_labels)),
            "direction_conflict_count": len(direction_conflicts),
            "direction_conflict_labels": "|".join(direction_conflicts),
            "salience_conflict_count": len(salience_conflicts),
            "salience_conflict_labels": "|".join(salience_conflicts),
        })
    return rows


def _provenance(
    learned: Mapping[str, Sequence[RationalePrediction]],
) -> dict[str, object]:
    per_vc: list[dict[str, object]] = []
    hierarchical: list[dict[str, object]] = []
    for method, predictions in learned.items():
        if method.startswith("hierarchical"):
            seen: set[str] = set()
            for prediction in predictions:
                if prediction.held_group in seen:
                    continue
                seen.add(prediction.held_group)
                hierarchical.append({
                    "method": method,
                    "source_condition": prediction.source_condition,
                    "held_group": prediction.held_group,
                    "training_groups": prediction.training_groups,
                    "training_indices": prediction.training_indices,
                })
        else:
            for prediction in predictions:
                per_vc.append({
                    "method": method,
                    "source_condition": prediction.source_condition,
                    "vc_slug": prediction.vc_slug,
                    "held_episode_slug": prediction.episode_slug,
                    "training_episode_slugs": prediction.training_episode_slugs,
                    "training_indices": prediction.training_indices,
                })
    return {
        "schema": "actual-rationale-fold-provenance-v1",
        "per_vc_folds": per_vc,
        "hierarchical_folds": hierarchical,
    }


def _oracle_gaps(
    classification: Sequence[Mapping[str, object]],
    ranking: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    class_macro = {
        str(row["method"]): row for row in classification if row["scope"] == "macro"
    }
    rank_macro = {
        str(row["method"]): row
        for row in ranking
        if row["scope"] == "macro" and int(row["review_budget"]) == 5
    }
    rows: list[dict[str, object]] = []
    for family in ("per_vc_logistic", "per_vc_tree", "hierarchical_logistic"):
        oracle_class = class_macro[f"{family}__actual_to_actual__balanced"]
        deployment_class = class_macro[f"{family}__actual_to_predicted__balanced"]
        noise_class = class_macro[f"{family}__predicted_to_predicted__balanced"]
        oracle_rank = rank_macro[f"{family}__actual_to_actual"]
        deployment_rank = rank_macro[f"{family}__actual_to_predicted"]
        noise_rank = rank_macro[f"{family}__predicted_to_predicted"]
        rows.append({
            "model_family": family,
            "oracle_balanced_accuracy": oracle_class["balanced_accuracy"],
            "deployment_balanced_accuracy": deployment_class["balanced_accuracy"],
            "predicted_control_balanced_accuracy": noise_class["balanced_accuracy"],
            "oracle_to_deployment_balanced_accuracy_gap": float(oracle_class["balanced_accuracy"]) - float(deployment_class["balanced_accuracy"]),
            "deployment_minus_predicted_control_balanced_accuracy": float(deployment_class["balanced_accuracy"]) - float(noise_class["balanced_accuracy"]),
            "oracle_average_precision": oracle_rank["average_precision"],
            "deployment_average_precision": deployment_rank["average_precision"],
            "predicted_control_average_precision": noise_rank["average_precision"],
            "oracle_to_deployment_average_precision_gap": float(oracle_rank["average_precision"]) - float(deployment_rank["average_precision"]),
            "deployment_minus_predicted_control_average_precision": float(deployment_rank["average_precision"]) - float(noise_rank["average_precision"]),
        })
    return rows


def _sensitivity_rows(
    cases: Sequence[RationaleModelCase],
    classification_methods: Mapping[str, Sequence[MethodPrediction]],
    ranking_methods: Mapping[str, Sequence[MethodPrediction]],
) -> list[dict[str, object]]:
    key_to_case = {(case.vc_slug, case.episode_slug): case for case in cases}
    output: list[dict[str, object]] = []
    for dimension in ("source_tier", "source_format"):
        values = sorted({str(getattr(case, dimension)) for case in cases})
        for value in values:
            allowed = {
                (case.vc_slug, case.episode_slug)
                for case in cases if str(getattr(case, dimension)) == value
            }
            for method, predictions in classification_methods.items():
                selected = [
                    row for row in predictions
                    if (row.vc_slug, row.episode_slug) in allowed
                ]
                targets = np.asarray([row.target for row in selected], dtype=int)
                predicted = np.asarray([row.predicted for row in selected], dtype=int)
                output.append({
                    "endpoint": "classification",
                    "dimension": dimension,
                    "value": value,
                    "method": method,
                    "n": len(selected),
                    "in_count": int(targets.sum()),
                    "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)) if len(set(targets)) == 2 else "",
                    "in_precision": float(precision_score(targets, predicted, zero_division=0)),
                    "in_recall": float(recall_score(targets, predicted, zero_division=0)),
                })
            for method, predictions in ranking_methods.items():
                selected = [
                    row for row in predictions
                    if (row.vc_slug, row.episode_slug) in allowed
                ]
                targets = np.asarray([row.target for row in selected], dtype=int)
                scores = np.asarray([row.score for row in selected], dtype=float)
                output.append({
                    "endpoint": "prioritization",
                    "dimension": dimension,
                    "value": value,
                    "method": method,
                    "n": len(selected),
                    "in_count": int(targets.sum()),
                    "average_precision": float(average_precision_score(targets, scores)) if targets.sum() else "",
                    "roc_auc": float(roc_auc_score(targets, scores)) if len(set(targets)) == 2 else "",
                })
    return output


def _primary_comparisons(
    endpoints: Mapping[str, object], *, iterations: int, seed: int
) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    for endpoint, methods, primary, metric in (
        ("classification", endpoints["classification_methods"], PRIMARY_CLASSIFICATION, "balanced_accuracy"),
        ("prioritization", endpoints["ranking_methods"], PRIMARY_RANKING, "average_precision"),
    ):
        for index, (method, predictions) in enumerate(methods.items()):
            if method == primary:
                continue
            row = cluster_permutation_test(
                methods[primary], predictions, metric=metric,
                iterations=iterations, seed=seed + index,
            )
            comparisons.append({
                "endpoint": endpoint,
                "reference_kind": "prespecified_primary",
                **row,
            })
    adjusted = holm_adjust([float(row["p_value"]) for row in comparisons])
    for row, value in zip(comparisons, adjusted, strict=True):
        row["adjusted_p_value"] = value
        row["significant_0_05"] = value < 0.05
    return comparisons


def explanatory_importance_rows(
    cases: Sequence[RationaleModelCase],
) -> list[dict[str, object]]:
    """Fit full-data per-VC explanatory models; never use these as performance."""
    rows: list[dict[str, object]] = []
    for vc_slug in sorted({case.vc_slug for case in cases}):
        selected = [case for case in cases if case.vc_slug == vc_slug]
        maps = [case.actual_features for case in selected]
        targets = np.asarray([case.target for case in selected], dtype=int)
        logistic = make_pipeline(
            DictVectorizer(sparse=True), StandardScaler(with_mean=False),
            LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=2_000, random_state=20260817),
        )
        logistic.fit(maps, targets)
        vectorizer = logistic.named_steps["dictvectorizer"]
        coefficients = logistic.named_steps["logisticregression"].coef_[0]
        rows.extend({
            "scientific_status": "full_data_explanatory_not_performance_estimate",
            "model_family": "per_vc_logistic",
            "vc_slug": vc_slug,
            "feature": feature,
            "importance": float(value),
        } for feature, value in zip(vectorizer.get_feature_names_out(), coefficients, strict=True))
        tree_vectorizer = DictVectorizer(sparse=False)
        matrix = tree_vectorizer.fit_transform(maps)
        tree = RandomForestClassifier(
            n_estimators=128, max_depth=3, min_samples_leaf=2,
            class_weight="balanced", random_state=20260817, n_jobs=1,
        ).fit(matrix, targets)
        rows.extend({
            "scientific_status": "full_data_explanatory_not_performance_estimate",
            "model_family": "per_vc_tree",
            "vc_slug": vc_slug,
            "feature": feature,
            "importance": float(value),
        } for feature, value in zip(tree_vectorizer.get_feature_names_out(), tree.feature_importances_, strict=True))
    return rows


def evaluate_actual_rationale_results(
    records: Sequence[Phase2CalibrationRecord],
    cases: Sequence[RationaleModelCase],
    learned: Mapping[str, Sequence[RationalePrediction]],
    *,
    review_budgets: Sequence[int] = (1, 3, 5, 10, 20),
    bootstrap_iterations: int = 2_000,
    permutation_iterations: int = 5_000,
    seed: int = 20260817,
) -> dict[str, object]:
    """Evaluate the nine source/model conditions against raw Phase 2."""
    _aligned(records, cases)
    endpoints = evaluate_task_endpoints(
        records, learned, review_budgets=review_budgets
    )
    statistics = evaluate_uncertainty_and_significance(
        endpoints,
        bootstrap_iterations=bootstrap_iterations,
        permutation_iterations=permutation_iterations,
        seed=seed,
    )
    return {
        "classification": endpoints["classification"],
        "ranking": endpoints["ranking"],
        "threshold_tradeoffs": endpoints["threshold_tradeoffs"],
        "classification_methods": endpoints["classification_methods"],
        "ranking_methods": endpoints["ranking_methods"],
        "uncertainty": statistics["uncertainty"],
        "significance": statistics["significance"],
        "primary_significance": _primary_comparisons(
            endpoints, iterations=permutation_iterations, seed=seed + 50_000
        ),
        "oracle_gaps": _oracle_gaps(
            endpoints["classification"], endpoints["ranking"]
        ),
        "sensitivity": _sensitivity_rows(
            cases, endpoints["classification_methods"], endpoints["ranking_methods"]
        ),
        "rationale_differences": rationale_difference_rows(cases),
        "provenance": _provenance(learned),
    }


def _prediction_rows(
    records: Sequence[Phase2CalibrationRecord],
    learned: Mapping[str, Sequence[RationalePrediction]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append({
            "method": "raw_phase2",
            "model_family": "raw_phase2",
            "source_condition": "phase2",
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
            "selected_params": "",
        })
    for method, predictions in learned.items():
        for prediction in predictions:
            rows.append({
                "method": method,
                "model_family": prediction.model_family,
                "source_condition": prediction.source_condition,
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
                "selected_params": json.dumps(dict(prediction.selected_params), sort_keys=True),
            })
    return rows


def assemble_actual_rationale_analysis(
    records: Sequence[Phase2CalibrationRecord],
    cases: Sequence[RationaleModelCase],
    learned: Mapping[str, Sequence[RationalePrediction]],
    taxonomy: Mapping[str, object] | Sequence[str],
    *,
    review_budgets: Sequence[int] = (1, 3, 5, 10, 20),
    bootstrap_iterations: int = 2_000,
    permutation_iterations: int = 5_000,
    seed: int = 20260817,
) -> dict[str, object]:
    """Assemble metrics and auditable artifacts from completed OOF predictions."""
    result = evaluate_actual_rationale_results(
        records,
        cases,
        learned,
        review_budgets=review_budgets,
        bootstrap_iterations=bootstrap_iterations,
        permutation_iterations=permutation_iterations,
        seed=seed,
    )
    return {
        "schema": "actual-rationale-decision-evaluation-v1",
        "records": list(records),
        "cases": list(cases),
        "learned": dict(learned),
        "prediction_rows": _prediction_rows(records, learned),
        "feature_dictionary": actual_rationale_feature_dictionary(taxonomy),
        "importance": explanatory_importance_rows(cases),
        "api_cost_usd": 0.0,
        **result,
    }


def run_actual_rationale_analysis(
    records: Sequence[Phase2CalibrationRecord],
    phase1_cases: Sequence[Phase1Case],
    taxonomy: Mapping[str, TaxonomyLabel],
    *,
    review_budgets: Sequence[int] = (1, 3, 5, 10, 20),
    inner_splits: int = 3,
    bootstrap_iterations: int = 2_000,
    permutation_iterations: int = 5_000,
    seed: int = 20260817,
    n_jobs: int = 1,
) -> dict[str, object]:
    """Build aligned cases, run nine OOF conditions, and evaluate them."""
    cases = align_rationale_cases(
        records, build_rationale_model_cases(phase1_cases, taxonomy)
    )
    learned = run_all_rationale_models(
        cases, inner_splits=inner_splits, seed=seed, n_jobs=n_jobs
    )
    return assemble_actual_rationale_analysis(
        records,
        cases,
        learned,
        taxonomy,
        review_budgets=review_budgets,
        bootstrap_iterations=bootstrap_iterations,
        permutation_iterations=permutation_iterations,
        seed=seed + 100_000,
    )


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


def _benchmark_policy(analysis: Mapping[str, object]) -> dict[str, object]:
    classification = [
        row for row in analysis["classification"] if row["scope"] == "macro"
    ]
    ranking = [
        row for row in analysis["ranking"]
        if row["scope"] == "macro" and int(row["review_budget"]) == 5
    ]
    return {
        "schema": "actual-rationale-benchmark-policy-v1",
        "scientific_status": "development_not_untouched_holdout",
        "reference_status": "automated_candidate_not_human_validated",
        "prespecified_primary": {
            "classification": PRIMARY_CLASSIFICATION,
            "prioritization": PRIMARY_RANKING,
        },
        "descriptive_best": {
            "classification": max(classification, key=lambda row: float(row["balanced_accuracy"]))["method"],
            "prioritization": max(ranking, key=lambda row: float(row["average_precision"]))["method"],
        },
        "promotion_rule": "positive paired delta with Holm-adjusted p < 0.05",
    }


def _render_report(analysis: Mapping[str, object], policy: Mapping[str, object]) -> str:
    records = analysis["records"]
    macro_class = [
        row for row in analysis["classification"] if row["scope"] == "macro"
    ]
    macro_rank = [
        row for row in analysis["ranking"]
        if row["scope"] == "macro" and int(row["review_budget"]) == 5
    ]
    lines = [
        "# Actual-Rationale Decision-Model Evaluation",
        "",
        "> Development analysis. Transcript-observed rationales are automated candidate ground truth, not human-validated ground truth.",
        "",
        f"Population: **{len(records)} cases**, **{sum(row.target for row in records)} Ins**, "
        f"**{len(records)-sum(row.target for row in records)} Outs**, and "
        f"**{len(set(row.vc_slug for row in records))} VCs**.",
        "",
        "The main deployable condition trains on every other episode's observed VC rationales and predicts the held-out pitch from Phase 1 rationales.",
        "",
        "## Classification",
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
        "## Opportunity prioritization",
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
        "## Oracle-to-deployment gaps",
        "",
        "| Model | Oracle BA | Deployment BA | Oracle AP | Deployment AP |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in analysis["oracle_gaps"]:
        lines.append(
            f"| {row['model_family']} | {float(row['oracle_balanced_accuracy']):.3f} | "
            f"{float(row['deployment_balanced_accuracy']):.3f} | "
            f"{float(row['oracle_average_precision']):.3f} | "
            f"{float(row['deployment_average_precision']):.3f} |"
        )
    lines.extend([
        "",
        "## Prespecified status",
        "",
        f"Classification primary: **{policy['prespecified_primary']['classification']}**.  ",
        f"Prioritization primary: **{policy['prespecified_primary']['prioritization']}**.",
        "",
        "Full-data coefficients and tree importance are explanatory predictive associations, not causal estimates of investor policy.",
        "",
    ])
    return "\n".join(lines)


def write_actual_rationale_analysis(
    analysis: Mapping[str, object],
    output_root: Path,
    *,
    input_paths: Mapping[str, Path],
) -> dict[str, Path]:
    """Write the complete actual-rationale experiment atomically."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "evaluation": output / "evaluation.md",
        "predictions": output / "predictions.csv",
        "classification": output / "classification_metrics.csv",
        "ranking": output / "ranking_metrics.csv",
        "threshold_tradeoffs": output / "threshold_tradeoffs.csv",
        "uncertainty": output / "uncertainty.csv",
        "significance": output / "significance.csv",
        "primary_significance": output / "primary_significance.csv",
        "oracle_gaps": output / "oracle_gaps.csv",
        "sensitivity": output / "reference_sensitivity.csv",
        "rationale_differences": output / "rationale_differences.csv",
        "importance": output / "explanatory_importance.csv",
        "feature_dictionary": output / "feature_dictionary.csv",
        "provenance": output / "fold_provenance.json",
        "benchmark_policy": output / "benchmark_policy.json",
        "manifest": output / "manifest.json",
        "model_bundle": output / "models" / "explanatory.joblib",
    }
    table_sources = {
        "predictions": "prediction_rows",
        "classification": "classification",
        "ranking": "ranking",
        "threshold_tradeoffs": "threshold_tradeoffs",
        "uncertainty": "uncertainty",
        "significance": "significance",
        "primary_significance": "primary_significance",
        "oracle_gaps": "oracle_gaps",
        "sensitivity": "sensitivity",
        "rationale_differences": "rationale_differences",
        "importance": "importance",
        "feature_dictionary": "feature_dictionary",
    }
    for path_key, source_key in table_sources.items():
        _write_csv(paths[path_key], analysis[source_key])
    _write_json(paths["provenance"], analysis["provenance"])
    policy = _benchmark_policy(analysis)
    _write_json(paths["benchmark_policy"], policy)
    paths["model_bundle"].parent.mkdir(parents=True, exist_ok=True)
    temporary_model = paths["model_bundle"].with_name(".explanatory.joblib.tmp")
    joblib.dump({
        "scientific_status": "full_data_explanatory_not_performance_estimate",
        "importance": analysis["importance"],
    }, temporary_model)
    temporary_model.replace(paths["model_bundle"])
    _atomic_text(paths["evaluation"], _render_report(analysis, policy))
    inputs = {
        name: {"path": str(Path(path).resolve()), "sha256": sha256(Path(path).read_bytes()).hexdigest()}
        for name, path in input_paths.items()
    }
    records = analysis["records"]
    manifest = {
        "schema": analysis["schema"],
        "scientific_status": "development_not_untouched_holdout",
        "reference_status": "automated_candidate_not_human_validated",
        "population": {
            "cases": len(records),
            "investors": len(set(row.vc_slug for row in records)),
            "ins": sum(row.target for row in records),
            "outs": len(records) - sum(row.target for row in records),
        },
        "api_cost_usd": float(analysis["api_cost_usd"]),
        "inputs": inputs,
        "outputs": sorted(str(path.relative_to(output)) for path in paths.values()),
    }
    _write_json(paths["manifest"], manifest)
    return paths
