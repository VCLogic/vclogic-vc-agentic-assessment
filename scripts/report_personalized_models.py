#!/usr/bin/env python3
"""Report personalized model classification, ranking, and uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from vc_clone_graph.personalized_config import load_personalized_config
from vc_clone_graph.personalized_evaluation import (
    PredictionRow,
    evaluate_personalized_predictions,
    write_personalized_evaluation,
)
from vc_clone_graph.phase2_calibration_evaluation import load_phase2_records


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument(
        "--families", nargs="+", default=("tabpfn", "setfit", "ensemble", "graph")
    )
    return result


def _checkpoint_predictions(output: Path, families: Sequence[str]) -> list[PredictionRow]:
    result = []
    for family in families:
        for path in sorted((output / family).glob("*/*.json")):
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            for row in checkpoint.get("predictions", []):
                result.append(PredictionRow(
                    method=f"{family}__{row['condition']}", family=family,
                    condition=str(row["condition"]), vc_slug=str(row["vc_slug"]),
                    vc_name=str(row["vc_name"]), episode_slug=str(row["episode_slug"]),
                    target=int(row["target"]),
                    classification_score=float(row["classification_score"]),
                    ranking_score=float(row["ranking_score"]), predicted=int(row["predicted"]),
                    status=str(row.get("status", checkpoint["status"])),
                    runtime_seconds=float(row.get("runtime_seconds", 0.0)),
                ))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_personalized_config(args.config)
    project = Path.cwd().resolve()
    registry = (project / config.data.source_registry).resolve()
    output = (project / config.execution.output).resolve()
    predictions = _checkpoint_predictions(output, args.families)
    for record in load_phase2_records(registry):
        predictions.append(PredictionRow(
            method="raw_phase2", family="raw_phase2", condition="raw",
            vc_slug=record.vc_slug, vc_name=record.vc_name,
            episode_slug=record.episode_slug, target=record.target,
            classification_score=record.raw_likelihood,
            ranking_score=record.raw_likelihood, predicted=record.raw_decision,
            status="complete",
        ))
    result = evaluate_personalized_predictions(
        predictions,
        expected_case_count=301,
        review_budgets=config.evaluation.review_budgets,
        seed=config.evaluation.seed,
    )
    manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8")) if (output / "run-manifest.json").is_file() else {}
    write_personalized_evaluation(
        output, result, source_manifest=manifest,
        automated_rationale_references=True,
    )
    print(f"promotion_status={result['promotion_status']} best={result['best_candidate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
