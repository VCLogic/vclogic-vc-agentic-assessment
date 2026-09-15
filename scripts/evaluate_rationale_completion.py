#!/usr/bin/env python3
"""Evaluate probabilistic rationale completion without external API calls."""

from __future__ import annotations

import argparse
from pathlib import Path

from vc_clone_graph.phase1_calibration_cases import build_calibration_cases
from vc_clone_graph.phase1_evaluation import load_phase1_cases
from vc_clone_graph.rationale_completion import (
    build_completion_cases,
    completion_analysis,
    nested_completion_predictions,
    write_completion_analysis,
)


DEFAULT_REGISTRY = Path("evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json")
DEFAULT_REFERENCES = Path("evaluation/phase1_ground_truth_rationales")
DEFAULT_TAXONOMY = Path("../agentic-vc-clone-framework/taxonomy/codebook_v_final.json")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    result.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    result.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--vc-slug", default="charles-hudson")
    result.add_argument("--max-hypotheses", type=int, default=3)
    result.add_argument("--seed", type=int, default=20260819)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    cases, taxonomy = load_phase1_cases(
        Path.cwd(),
        args.registry,
        args.references,
        args.taxonomy,
        selected_vcs=(args.vc_slug,),
    )
    calibration = build_calibration_cases(Path.cwd(), cases, taxonomy)
    completion_cases = build_completion_cases(calibration)
    predictions = nested_completion_predictions(
        completion_cases,
        max_hypotheses=args.max_hypotheses,
        seed=args.seed,
    )
    analysis = completion_analysis(predictions)
    source_paths = {
        f"canonical_investigation::{case.vc_slug}::{case.episode_slug}": case.phase1_case.artifact_path
        for case in calibration
        if case.phase1_case.artifact_path is not None
    }
    source_paths.update(
        {
            f"reference_record::{case.vc_slug}::{case.episode_slug}": case.phase1_case.reference_path
            for case in calibration
            if case.phase1_case.reference_path is not None
        }
    )
    source_paths.update(
        {
            f"audited_pitch::{case.vc_slug}::{case.episode_slug}": case.pitch_path
            for case in calibration
        }
    )
    paths = write_completion_analysis(
        predictions,
        analysis,
        args.output,
        input_paths={
            "registry": args.registry,
            "reference_manifest": args.references / "manifest.json",
            "taxonomy": args.taxonomy,
            **source_paths,
        },
    )
    print(
        f"rationale completion complete: cases={len(predictions)} "
        f"api_cost_usd=0.00 report={paths['evaluation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
