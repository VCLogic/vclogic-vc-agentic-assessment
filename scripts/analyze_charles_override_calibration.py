#!/usr/bin/env python3
"""Evaluate a label-blind Charles override calibrator without new API calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vc_clone_graph.advanced_calibration import (
    bootstrap_intervals,
    make_repeated_stratified_folds,
    repeated_stratified_logistic_predictions,
    write_advanced_analysis,
)
from vc_clone_graph.artifacts import write_json
from vc_clone_graph.calibration import evaluate_predictions, load_frozen_population
from vc_clone_graph.interim_calibration import freeze_completed_snapshot
from vc_clone_graph.override_calibration import (
    augment_override_records,
    exclude_changed_pitch_records,
)


BASELINE_METHODS = (
    "semantic_phase1",
    "phase2",
    "semantic_plus_phase2",
    "majority_out",
    "raw_any_check",
    "raw_standard_check",
    "raw_ranking_score",
)
OVERRIDE_METHOD = "charles_override_logistic"
OVERRIDE_FAMILY = "semantic_plus_phase2_plus_override"


def _changed_slugs(cleanup_report: dict[str, object]) -> set[str]:
    episodes = cleanup_report.get("episodes", [])
    if not isinstance(episodes, list):
        raise ValueError("cleanup report episodes must be a list")
    return {
        str(row["episode_slug"])
        for row in episodes
        if isinstance(row, dict) and row.get("status") == "changed"
    }


def _report(summary: dict[str, object]) -> str:
    lines = [
        "# Charles override calibration",
        "",
        (
            f"Leakage-safe development population: {summary['dataset']['episodes']} "
            f"episodes ({summary['dataset']['ins']} Ins, "
            f"{summary['dataset']['outs']} Outs)."
        ),
        "",
        "| Method | Balanced accuracy | In precision | In recall | In F1 | AP | ROC AUC | Top 5 / 10 / 20 hits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in summary["methods"].items():
        hits = metrics["ranking"]["hits_at_k"]
        lines.append(
            f"| {method} | {metrics['balanced_accuracy']:.3f} | "
            f"{metrics['in_precision']:.3f} | {metrics['in_recall']:.3f} | "
            f"{metrics['in_f1']:.3f} | {metrics['average_precision']:.3f} | "
            f"{metrics['roc_auc']:.3f} | {hits['5']} / {hits['10']} / {hits['20']} |"
        )
    lines.extend(
        [
            "",
            "This is a repeated nested development estimate over legacy v3 outputs, not an untouched holdout result.",
            "No inference API was called for this analysis; incremental API cost was $0.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(
    *,
    status_path: Path,
    labels_path: Path,
    cleanup_report_path: Path,
    output_root: Path,
    repeats: int,
    splits: int,
    bootstrap_iterations: int,
) -> dict[str, object]:
    """Create the safe snapshot and compare existing and override calibrators."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    status = json.loads(Path(status_path).read_text(encoding="utf-8"))
    cleanup = json.loads(Path(cleanup_report_path).read_text(encoding="utf-8"))
    changed = _changed_slugs(cleanup)
    filtered = exclude_changed_pitch_records(status, changed)
    filtered_status_path = output / "filtered-source-status.json"
    write_json(filtered_status_path, filtered)
    snapshot = freeze_completed_snapshot(
        filtered_status_path, Path(labels_path), output / "snapshot"
    )
    if (snapshot.episode_count, snapshot.in_count, snapshot.out_count) != (63, 11, 52):
        raise ValueError(
            "unexpected safe population: "
            f"{snapshot.episode_count}/{snapshot.in_count}/{snapshot.out_count}"
        )
    records = augment_override_records(
        load_frozen_population([snapshot.status_path], snapshot.labels_path)
    )
    folds = make_repeated_stratified_folds(
        records, repeats=repeats, splits=splits
    )
    baseline = write_advanced_analysis(
        records,
        output / "baseline",
        methods=BASELINE_METHODS,
        bootstrap_iterations=bootstrap_iterations,
        fold_maps=folds,
    )
    override_predictions = repeated_stratified_logistic_predictions(
        records,
        method=OVERRIDE_METHOD,
        feature_family=OVERRIDE_FAMILY,
        fold_maps=folds,
    )
    override_metrics = evaluate_predictions(override_predictions)
    override_intervals = bootstrap_intervals(
        override_predictions,
        iterations=bootstrap_iterations,
        seed=20260803,
    )
    methods = {**baseline["methods"], OVERRIDE_METHOD: override_metrics}
    summary: dict[str, object] = {
        "schema": "charles-override-calibration-v1",
        "status": "nested_development_diagnostic_not_untouched_holdout",
        "dataset": baseline["dataset"],
        "pitch_cleanup": {
            "changed_in_report": len(changed),
            "completed_records_excluded": len(filtered["leakage_exclusions"]),
            "excluded_slugs": filtered["leakage_exclusions"],
        },
        "evaluation": baseline["evaluation"],
        "outer_protocol": baseline["outer_protocol"],
        "incremental_api_cost_usd": 0.0,
        "methods": methods,
        "override_intervals_95": override_intervals,
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "override-predictions.json",
        [
            {
                "episode_slug": row.episode_slug,
                "target": row.target,
                "score": row.score,
                "predicted": row.predicted,
                "selected_config": dict(row.selected_config),
                "training_slugs": list(row.training_slugs),
                "provenance": dict(row.provenance),
            }
            for row in override_predictions
        ],
    )
    (output / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--cleanup-report-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    args = parser.parse_args()
    result = run_analysis(
        status_path=args.status_path,
        labels_path=args.labels_path,
        cleanup_report_path=args.cleanup_report_path,
        output_root=args.output_root,
        repeats=args.repeats,
        splits=args.splits,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
