#!/usr/bin/env python3
"""Run reusable v4 confirmation/rescue calibration without API calls."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import shutil

from vc_clone_graph.two_head_calibration import (
    load_v4_calibration_population,
    write_two_head_analysis,
)


def _digest(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--cap",
        type=float,
        action="append",
        dest="caps",
        help="Conditional false-positive-rate cap; repeat for multiple policies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    caps = tuple(args.caps or (0.15, 0.20, 0.25, 0.30))
    records = load_v4_calibration_population(args.status, args.labels)
    result = write_two_head_analysis(
        records,
        args.output,
        false_positive_rate_caps=caps,
        source_manifest={
            "status_path": str(args.status),
            "status_sha256": _digest(args.status),
            "labels_path": str(args.labels),
            "labels_sha256": _digest(args.labels),
        },
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.output / "report.md", args.report)
    population = result["population"]
    print(
        f"evaluated {population['episodes']} episodes: "
        f"{population['ins']} Ins, {population['outs']} Outs"
    )
    for name, metrics in result["methods"].items():
        matrix = metrics["confusion_matrix"]
        print(
            f"{name}: TP={matrix['tp']} FP={matrix['fp']} "
            f"TN={matrix['tn']} FN={matrix['fn']} "
            f"BA={metrics['balanced_accuracy']:.3f} "
            f"P={metrics['in_precision']:.3f} "
            f"R={metrics['in_recall']:.3f} F1={metrics['in_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
