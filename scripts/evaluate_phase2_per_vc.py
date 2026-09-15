#!/usr/bin/env python3
"""Evaluate separate per-VC leave-one-pitch-out decision and rationale models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from vc_clone_graph.per_vc_rationale_models import (
    run_per_vc_analysis,
    write_per_vc_analysis,
)
from vc_clone_graph.phase2_calibration_evaluation import load_phase2_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = (
    PROJECT_ROOT / "evaluation" / "canonical_runs_v4_v41_portfolio_2026-08-15.json"
)


def _budgets(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("review budgets must be positive integers")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--forest-trees", type=int, default=100)
    parser.add_argument("--rule-trees", type=int, default=40)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--permutation-iterations", type=int, default=5_000)
    parser.add_argument("--importance-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--review-budgets", type=_budgets, default=(1, 3, 5, 10, 20))
    parser.add_argument(
        "--resume-analysis",
        action="store_true",
        help="reuse output/.analysis_checkpoint.joblib and only regenerate artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records = load_phase2_records(args.registry)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / ".analysis_checkpoint.joblib"
    if args.resume_analysis:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"analysis checkpoint does not exist: {checkpoint}")
        result = joblib.load(checkpoint)
    else:
        result = run_per_vc_analysis(
            records,
            inner_splits=args.inner_splits,
            forest_trees=args.forest_trees,
            rule_trees=args.rule_trees,
            bootstrap_iterations=args.bootstrap_iterations,
            permutation_iterations=args.permutation_iterations,
            seed=args.seed,
            review_budgets=args.review_budgets,
        )
        joblib.dump(result, checkpoint)
    paths = write_per_vc_analysis(
        records,
        result,
        args.output,
        registry_path=args.registry,
        importance_repeats=args.importance_repeats,
    )
    print(
        f"evaluated {len(records)} per-VC leave-one-pitch-out cases across "
        f"{len(set(record.vc_slug for record in records))} VCs; "
        f"report={paths['evaluation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
