#!/usr/bin/env python3
"""Build label-blind batch shortlists from frozen post-hoc predictions."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil

from vc_clone_graph.posthoc_semantic_models import (
    PosthocPrediction,
    load_semantic_population,
    write_fixed_budget_shortlist_analysis,
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_predictions(path: Path) -> dict[str, list[PosthocPrediction]]:
    result: dict[str, list[PosthocPrediction]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            row = PosthocPrediction(
                method=raw["method"],
                episode_slug=raw["episode_slug"],
                target=int(raw["target"]),
                score=float(raw["score"]),
                predicted=int(raw["predicted"]),
                threshold=float(raw["threshold"]),
                selected_config=json.loads(raw["selected_config"]),
                training_slugs=tuple(json.loads(raw["training_slugs"])),
            )
            result.setdefault(row.method, []).append(row)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--review-budget", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_semantic_population(args.status, args.labels)
    predictions = _load_predictions(args.predictions)
    result = write_fixed_budget_shortlist_analysis(
        records,
        predictions,
        args.output,
        review_budget=args.review_budget,
        source_manifest={
            "status_path": str(args.status),
            "status_sha256": _digest(args.status),
            "labels_path": str(args.labels),
            "labels_sha256": _digest(args.labels),
            "predictions_path": str(args.predictions),
            "predictions_sha256": _digest(args.predictions),
        },
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.output / "report.md", args.report)
    for method, metrics in result["methods"].items():
        matrix = metrics["confusion_matrix"]
        print(
            f"{method}: budget={metrics['review_budget']} TP={matrix['tp']} "
            f"FP={matrix['fp']} TN={matrix['tn']} FN={matrix['fn']} "
            f"BA={metrics['balanced_accuracy']:.3f} "
            f"AP={metrics['average_precision']:.3f}"
        )


if __name__ == "__main__":
    main()
