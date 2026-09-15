#!/usr/bin/env python3
"""Evaluate deployable Phase 1, associated-rationale, and Phase 2 fusion."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

import joblib
from scipy.stats import binomtest

from vc_clone_graph.phase1_evaluation import load_phase1_cases
from vc_clone_graph.phase2_calibration_evaluation import (
    MethodPrediction,
    classification_metric_rows,
    load_phase2_records,
    ranking_metric_rows,
)
from vc_clone_graph.structured_fusion_features import (
    FEATURE_CONDITIONS,
    build_structured_fusion_cases,
)
from vc_clone_graph.structured_fusion_models import nested_two_head_predictions


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _method_rows(predictions, *, head: str) -> list[MethodPrediction]:
    result = []
    for row in predictions:
        method = f"{row.model_family}__{row.condition}"
        score = row.classification_score if head == "classification" else row.ranking_score
        result.append(MethodPrediction(
            vc_slug=row.vc_slug, vc_name=row.vc_name,
            episode_slug=row.episode_slug, group=row.episode_slug,
            target=row.target, method=method, score=score,
            predicted=row.predicted,
        ))
    return result


def _mcnemar(primary, baseline) -> dict:
    left = {(r.vc_slug, r.episode_slug): r for r in primary}
    right = {(r.vc_slug, r.episode_slug): r for r in baseline}
    primary_only = baseline_only = 0
    for key in sorted(left.keys() & right.keys()):
        p_ok = left[key].predicted == left[key].target
        b_ok = right[key].predicted == right[key].target
        primary_only += int(p_ok and not b_ok)
        baseline_only += int(b_ok and not p_ok)
    discordant = primary_only + baseline_only
    return {
        "primary_only_correct": primary_only,
        "baseline_only_correct": baseline_only,
        "discordant": discordant,
        "exact_p": float(binomtest(
            min(primary_only, baseline_only), discordant, 0.5
        ).pvalue) if discordant else 1.0,
    }


def _selected_feature_rows(predictions) -> list[dict]:
    aggregate: dict[tuple[str, str, str, str, str], list[float]] = {}
    for row in predictions:
        for head, features in (
            ("classification", row.classification_selected_features),
            ("ranking", row.ranking_selected_features),
        ):
            for feature, value in features:
                key = (row.model_family, row.condition, row.vc_slug, head, feature)
                aggregate.setdefault(key, []).append(float(value))
    result = []
    for key, values in sorted(aggregate.items()):
        family, condition, vc, head, feature = key
        result.append({
            "model_family": family, "condition": condition, "vc_slug": vc,
            "head": head, "feature": feature,
            "selected_outer_folds": len(values),
            "mean_effect": sum(values) / len(values),
            "mean_absolute_effect": sum(abs(v) for v in values) / len(values),
        })
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", type=Path, default=Path(
        "evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json"))
    parser.add_argument("--references", type=Path, default=Path(
        "evaluation/phase1_ground_truth_rationales"))
    parser.add_argument("--taxonomy", type=Path, default=Path(
        "../agentic-vc-clone-framework/taxonomy/codebook_v_final.json"))
    parser.add_argument("--v5-root", type=Path, default=Path(
        "outputs/v5-associated-rationales/phase1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=FEATURE_CONDITIONS,
                        default=list(FEATURE_CONDITIONS))
    parser.add_argument("--families", nargs="+", choices=("elastic", "forest", "blend"),
                        default=("elastic", "forest", "blend"))
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--forest-trees", type=int, default=64)
    parser.add_argument("--inner-splits", type=int, default=3)
    args = parser.parse_args(argv)

    phase1_cases, taxonomy = load_phase1_cases(
        Path.cwd(), args.source_registry, args.references, args.taxonomy)
    del phase1_cases
    records = load_phase2_records(args.source_registry)
    cases = build_structured_fusion_cases(
        records, args.v5_root,
        families={key: value.coarse_parent for key, value in taxonomy.items()},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / ".prediction_checkpoint.joblib"
    checkpoint = joblib.load(checkpoint_path) if checkpoint_path.exists() else {}
    all_predictions = []
    for family in args.families:
        for condition in args.conditions:
            key = (family, condition, args.inner_splits, args.forest_trees)
            if key not in checkpoint:
                print(f"running {family}/{condition} on {len(cases)} cases", flush=True)
                checkpoint[key] = nested_two_head_predictions(
                    cases, condition=condition, model_family=family,
                    inner_splits=args.inner_splits, n_jobs=args.n_jobs,
                    forest_trees=args.forest_trees,
                )
                joblib.dump(checkpoint, checkpoint_path)
            else:
                print(f"reusing {family}/{condition} checkpoint", flush=True)
            all_predictions.extend(checkpoint[key])

    classification_methods = []
    ranking_methods = []
    for family in args.families:
        for condition in args.conditions:
            selected = [r for r in all_predictions
                        if r.model_family == family and r.condition == condition]
            classification_methods.extend(_method_rows(selected, head="classification"))
            ranking_methods.extend(_method_rows(selected, head="ranking"))
    classification = []
    ranking = []
    for method in sorted({row.method for row in classification_methods}):
        classification.extend(classification_metric_rows(
            [row for row in classification_methods if row.method == method]))
        ranking.extend(ranking_metric_rows(
            [row for row in ranking_methods if row.method == method]))

    _write_csv(args.output / "classification_metrics.csv", classification)
    _write_csv(args.output / "ranking_metrics.csv", ranking)
    _write_csv(args.output / "selected_features.csv", _selected_feature_rows(all_predictions))
    _write_csv(args.output / "predictions.csv", [asdict(row) for row in all_predictions])
    (args.output / "fold_provenance.json").write_text(json.dumps([
        {
            "model_family": row.model_family, "condition": row.condition,
            "vc_slug": row.vc_slug, "episode_slug": row.episode_slug,
            "training_episode_slugs": row.training_episode_slugs,
            "inner_validation_episode_slugs": row.inner_validation_episode_slugs,
        } for row in all_predictions
    ], indent=2) + "\n", encoding="utf-8")

    by_method = {}
    for row in classification_methods:
        by_method.setdefault(row.method, []).append(row)
    significance = {}
    if "elastic__full" in by_method:
        for baseline in ("elastic__phase1", "elastic__phase2"):
            if baseline in by_method:
                significance[f"elastic__full_vs_{baseline}"] = _mcnemar(
                    by_method["elastic__full"], by_method[baseline])
    macro_class = {row["method"]: row for row in classification if row["scope"] == "macro"}
    macro_rank = {}
    for row in ranking:
        if row["scope"] == "macro" and row["review_budget"] == 5:
            macro_rank[row["method"]] = row
    summary = {
        "schema": "structured-fusion-evaluation-v1",
        "case_count": len(cases), "vc_count": len({r.vc_slug for r in cases}),
        "classification_macro": macro_class,
        "ranking_macro_at_5": macro_rank, "paired_significance": significance,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
