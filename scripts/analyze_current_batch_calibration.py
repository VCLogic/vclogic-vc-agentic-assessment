#!/usr/bin/env python3
"""Run leakage-controlled calibration on a frozen live-batch snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vc_clone_graph.advanced_calibration import (
    make_repeated_stratified_folds,
    write_advanced_analysis,
)
from vc_clone_graph.artifacts import write_json
from vc_clone_graph.calibration import load_frozen_population
from vc_clone_graph.interim_calibration import (
    InterimSnapshot,
    freeze_completed_snapshot,
)


METHODS = (
    "legacy_phase1",
    "phase2",
    "legacy_plus_phase2",
    "majority_out",
    "raw_any_check",
    "raw_standard_check",
    "raw_ranking_score",
)


def run_analysis(
    *,
    status_path: Path,
    labels_path: Path,
    output_root: Path,
    repeats: int,
    splits: int,
    bootstrap_iterations: int,
) -> dict[str, object]:
    """Freeze the current completed cohort and evaluate approved calibrators."""
    output = Path(output_root)
    snapshot = freeze_completed_snapshot(
        Path(status_path), Path(labels_path), output / "snapshot"
    )
    records = load_frozen_population([snapshot.status_path], snapshot.labels_path)
    fold_maps = make_repeated_stratified_folds(
        records, repeats=repeats, splits=splits
    )
    result = write_advanced_analysis(
        records,
        output,
        methods=METHODS,
        bootstrap_iterations=bootstrap_iterations,
        fold_maps=fold_maps,
    )
    summary = {
        "schema": "current-batch-interim-calibration-v1",
        "status": "interim_development_diagnostic",
        "snapshot": {
            "episodes": snapshot.episode_count,
            "ins": snapshot.in_count,
            "outs": snapshot.out_count,
            "status_path": str(snapshot.status_path),
            "labels_path": str(snapshot.labels_path),
        },
        "methods": list(METHODS),
        "evaluation": result["evaluation"],
        "outer_protocol": result["outer_protocol"],
    }
    write_json(output / "interim-summary.json", summary)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    args = parser.parse_args()
    result = run_analysis(
        status_path=args.status_path,
        labels_path=args.labels_path,
        output_root=args.output_root,
        repeats=args.repeats,
        splits=args.splits,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(json.dumps(result["dataset"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
