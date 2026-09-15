#!/usr/bin/env python3
"""Run actual-rationale versus Phase 1 decision-model evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import joblib

from vc_clone_graph.actual_rationale_evaluation import (
    run_actual_rationale_analysis,
    write_actual_rationale_analysis,
)
from vc_clone_graph.phase1_evaluation import load_phase1_cases
from vc_clone_graph.phase2_calibration_evaluation import load_phase2_records


DEFAULT_REGISTRY = Path("evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json")
DEFAULT_REFERENCES = Path("evaluation/phase1_ground_truth_rationales")
DEFAULT_TAXONOMY = Path("../agentic-vc-clone-framework/taxonomy/codebook_v_final.json")
DEFAULT_BENCHMARK = Path("evaluation/benchmarks/actual-rationale-primary-2026-08-17.json")


def _budgets(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("review budgets must be positive integers")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    result.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    result.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--benchmark-output", type=Path, default=DEFAULT_BENCHMARK)
    result.add_argument("--jobs", type=int, default=1)
    result.add_argument("--inner-splits", type=int, default=3)
    result.add_argument("--bootstrap-iterations", type=int, default=2_000)
    result.add_argument("--permutation-iterations", type=int, default=5_000)
    result.add_argument("--review-budgets", type=_budgets, default=(1, 3, 5, 10, 20))
    result.add_argument("--seed", type=int, default=20260817)
    result.add_argument("--resume-analysis", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / ".analysis_checkpoint.joblib"
    if args.resume_analysis:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"analysis checkpoint does not exist: {checkpoint}")
        analysis = joblib.load(checkpoint)
    else:
        records = load_phase2_records(args.registry)
        phase1_cases, taxonomy = load_phase1_cases(
            Path.cwd(), args.registry, args.references, args.taxonomy
        )
        analysis = run_actual_rationale_analysis(
            records,
            phase1_cases,
            taxonomy,
            review_budgets=args.review_budgets,
            inner_splits=args.inner_splits,
            bootstrap_iterations=args.bootstrap_iterations,
            permutation_iterations=args.permutation_iterations,
            seed=args.seed,
            n_jobs=args.jobs,
        )
        joblib.dump(analysis, checkpoint)
    paths = write_actual_rationale_analysis(
        analysis,
        args.output,
        input_paths={
            "registry": args.registry,
            "reference_manifest": args.references / "manifest.json",
            "taxonomy": args.taxonomy,
        },
    )
    args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths["benchmark_policy"], args.benchmark_output)
    records = analysis["records"]
    print(
        f"actual-rationale evaluation complete: cases={len(records)} "
        f"vcs={len(set(record.vc_slug for record in records))} "
        f"api_cost_usd={analysis['api_cost_usd']:.2f} "
        f"report={paths['evaluation']} checkpoint={checkpoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
