#!/usr/bin/env python3
"""Compare canonical raw Phase 2 outputs with all-VC calibrated decisions."""

from __future__ import annotations

import argparse
from pathlib import Path

from vc_clone_graph.phase2_calibration_evaluation import (
    load_phase2_records,
    run_phase2_analysis,
    write_phase2_analysis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "evaluation"
    / "canonical_runs_v4_v41_portfolio_2026-08-15.json"
)


def _budgets(value: str) -> tuple[int, ...]:
    try:
        budgets = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("review budgets must be integers") from exc
    if not budgets or len(set(budgets)) != len(budgets) or any(item < 1 for item in budgets):
        raise argparse.ArgumentTypeError("review budgets must be unique positive integers")
    return budgets


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=5_000)
    parser.add_argument("--permutation-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--review-budgets", type=_budgets, default=(1, 3, 5, 10, 20))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records = load_phase2_records(args.registry)
    result = run_phase2_analysis(
        records,
        outer_repeats=args.outer_repeats,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        seed=args.seed,
        review_budgets=args.review_budgets,
    )
    paths = write_phase2_analysis(
        records,
        result,
        args.output,
        registry_path=args.registry,
    )
    print(
        f"evaluated {len(records)} cases across "
        f"{len(set(record.vc_slug for record in records))} VCs; "
        f"report={paths['evaluation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
