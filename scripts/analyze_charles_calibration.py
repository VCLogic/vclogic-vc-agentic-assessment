#!/usr/bin/env python3
"""Run the frozen Charles semantic-calibration comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vc_clone_graph.artifacts import write_json
from vc_clone_graph.calibration import (
    FEATURE_FAMILIES,
    load_frozen_population,
    write_analysis,
)


STATUS_PATHS = (
    Path("outputs/openrouter-luna-charles-v2-codex36-2026-08-02-fix1/batch-status.json"),
    Path("outputs/openrouter-luna-charles-v2-remaining55-2026-08-02/batch-status.json"),
)
LABELS_PATH = Path("evaluation/labels/charles_pitch_window_decisions.json")
CONDITIONAL_OVERRIDES = {
    "41-can-this-startup-help-retailers-take-on-amazon": "In",
    "127-nectir-the-classroom-of-the-future": "In",
}


def _winner(result: dict, metric: str) -> dict:
    name = max(
        FEATURE_FAMILIES,
        key=lambda method: result["methods"][method][metric],
    )
    return {"method": name, metric: result["methods"][name][metric]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/charles-semantic-calibration-2026-08-02"),
    )
    args = parser.parse_args()

    strict_records = load_frozen_population(STATUS_PATHS, LABELS_PATH)
    strict = write_analysis(strict_records, args.output_root)
    conditional_records = load_frozen_population(
        STATUS_PATHS,
        LABELS_PATH,
        label_overrides=CONDITIONAL_OVERRIDES,
    )
    conditional = write_analysis(
        conditional_records, args.output_root / "conditional-interest"
    )
    summary = {
        "schema": "charles-semantic-calibration-comparison-v1",
        "strict": {
            "dataset": strict["dataset"],
            "best_classification": _winner(strict, "balanced_accuracy"),
            "best_ranking": _winner(strict, "average_precision"),
        },
        "conditional_interest": {
            "label_overrides": CONDITIONAL_OVERRIDES,
            "dataset": conditional["dataset"],
            "best_classification": _winner(conditional, "balanced_accuracy"),
            "best_ranking": _winner(conditional, "average_precision"),
        },
    }
    write_json(args.output_root / "comparison-summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
