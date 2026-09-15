#!/usr/bin/env python3
"""Run the expanded local Charles calibration benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vc_clone_graph.advanced_calibration import (
    candidate_manifest,
    make_repeated_stratified_folds,
    write_advanced_analysis,
)
from vc_clone_graph.artifacts import write_json
from vc_clone_graph.calibration import load_frozen_population


STATUS_PATHS = (
    Path("outputs/openrouter-luna-charles-v2-codex36-2026-08-02-fix1/batch-status.json"),
    Path("outputs/openrouter-luna-charles-v2-remaining55-2026-08-02/batch-status.json"),
)
LABELS_PATH = Path("evaluation/labels/charles_pitch_window_decisions.json")
CONDITIONAL_OVERRIDES = {
    "41-can-this-startup-help-retailers-take-on-amazon": "In",
    "127-nectir-the-classroom-of-the-future": "In",
}


def _winner(result: dict, metric: str, methods: set[str] | None = None) -> dict:
    candidates = methods or set(result["methods"])
    method = max(candidates, key=lambda name: result["methods"][name][metric])
    return {"method": method, metric: result["methods"][method][metric]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/charles-expanded-calibration-2026-08-03"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    args = parser.parse_args()

    advanced_methods = set(candidate_manifest())
    strict_records = load_frozen_population(STATUS_PATHS, LABELS_PATH)
    fold_maps = make_repeated_stratified_folds(
        strict_records, repeats=5, splits=5
    )
    strict = write_advanced_analysis(
        strict_records,
        args.output_root,
        bootstrap_iterations=args.bootstrap_iterations,
        fold_maps=fold_maps,
    )
    conditional_records = load_frozen_population(
        STATUS_PATHS,
        LABELS_PATH,
        label_overrides=CONDITIONAL_OVERRIDES,
    )
    conditional = write_advanced_analysis(
        conditional_records,
        args.output_root / "conditional-interest",
        bootstrap_iterations=args.bootstrap_iterations,
        fold_maps=fold_maps,
    )
    summary = {
        "schema": "charles-expanded-calibration-comparison-v1",
        "strict": {
            "dataset": strict["dataset"],
            "best_advanced_classification": _winner(
                strict, "balanced_accuracy", advanced_methods
            ),
            "best_advanced_ranking": _winner(
                strict, "average_precision", advanced_methods
            ),
            "best_overall_classification": _winner(strict, "balanced_accuracy"),
            "best_overall_ranking": _winner(strict, "average_precision"),
        },
        "conditional_interest": {
            "label_overrides": CONDITIONAL_OVERRIDES,
            "dataset": conditional["dataset"],
            "best_advanced_classification": _winner(
                conditional, "balanced_accuracy", advanced_methods
            ),
            "best_advanced_ranking": _winner(
                conditional, "average_precision", advanced_methods
            ),
            "best_overall_classification": _winner(
                conditional, "balanced_accuracy"
            ),
            "best_overall_ranking": _winner(conditional, "average_precision"),
        },
    }
    write_json(args.output_root / "comparison-summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
