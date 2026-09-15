#!/usr/bin/env python3
"""Generate fold-safe post-Phase-1 associated-rationale advisory sidecars."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from vc_clone_graph.phase1_calibration_cases import build_calibration_cases
from vc_clone_graph.phase1_evaluation import load_phase1_cases
from vc_clone_graph.rationale_association_annotations import (
    AssociationAnnotationEvaluationCase,
    build_rationale_association_annotations,
    write_rationale_association_analysis,
)
from vc_clone_graph.rationale_completion import (
    build_completion_cases,
    nested_association_predictions,
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
    result.add_argument("--min-probability", type=float, default=0.60)
    result.add_argument("--min-support", type=int, default=3)
    result.add_argument("--min-lift", type=float, default=1.0)
    result.add_argument("--max-per-source", type=int, default=5)
    result.add_argument("--seed", type=int, default=20260819)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    project_root = Path.cwd()
    phase1_cases, taxonomy = load_phase1_cases(
        project_root,
        args.registry,
        args.references,
        args.taxonomy,
        selected_vcs=(args.vc_slug,),
    )
    calibration_cases = build_calibration_cases(project_root, phase1_cases, taxonomy)
    completion_cases = build_completion_cases(calibration_cases)
    predictions = nested_association_predictions(
        completion_cases,
        max_hypotheses=max(1, args.max_per_source),
    )
    prediction_by_key = {
        (row.vc_slug, row.episode_slug): row for row in predictions
    }
    evaluation_cases: list[AssociationAnnotationEvaluationCase] = []
    source_paths: dict[str, Path] = {
        "registry": args.registry,
        "reference_manifest": args.references / "manifest.json",
        "taxonomy": args.taxonomy,
    }
    for calibration in calibration_cases:
        phase1 = calibration.phase1_case
        if phase1.artifact_path is None:
            raise ValueError(f"missing Phase 1 artifact path: {phase1.episode_slug}")
        artifact_path = Path(phase1.artifact_path)
        if not artifact_path.is_absolute():
            artifact_path = project_root / artifact_path
        investigation = json.loads(artifact_path.read_text(encoding="utf-8"))
        digest = sha256(artifact_path.read_bytes()).hexdigest()
        prediction = prediction_by_key[(calibration.vc_slug, calibration.episode_slug)]
        annotations = build_rationale_association_annotations(
            investigation,
            digest,
            prediction,
            taxonomy_labels=set(taxonomy),
            min_posterior_probability=args.min_probability,
            min_support=args.min_support,
            min_lift=args.min_lift,
            max_per_source=args.max_per_source,
        )
        evaluation_cases.append(
            AssociationAnnotationEvaluationCase(
                annotations=annotations,
                observed_labels=prediction.observed_labels,
                reference_labels=prediction.reference_labels,
            )
        )
        key = f"{calibration.vc_slug}::{calibration.episode_slug}"
        source_paths[f"investigation::{key}"] = artifact_path
        source_paths[f"pitch::{key}"] = calibration.pitch_path
        if phase1.reference_path is not None:
            reference_path = Path(phase1.reference_path)
            if not reference_path.is_absolute():
                reference_path = project_root / reference_path
            source_paths[f"reference::{key}"] = reference_path

    paths = write_rationale_association_analysis(
        evaluation_cases,
        args.output,
        input_paths=source_paths,
    )
    print(
        f"rationale association annotations complete: cases={len(evaluation_cases)} "
        f"api_cost_usd=0.00 report={paths['evaluation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
