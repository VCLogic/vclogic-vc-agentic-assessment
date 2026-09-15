#!/usr/bin/env python3
"""Consolidate reusable transcript-observed Phase 1 rationale references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vc_clone_graph.evaluation import load_registry
from vc_clone_graph.phase1_references import ReferenceIdentity, build_reference_corpus


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "evaluation" / "canonical_runs.json"
DEFAULT_AGENTIC_ROOT = PROJECT_ROOT.parent / "agentic-vc-clone-framework"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "evaluation" / "phase1_ground_truth_rationales"

IDENTITIES = {
    "charles-hudson": ReferenceIdentity(
        "charles-hudson", "Charles Hudson", "Charles", "charles-hudson-precursor-ventures"
    ),
    "elizabeth-yin": ReferenceIdentity(
        "elizabeth-yin", "Elizabeth Yin", "Elizabeth", "elizabeth-yin-hustle-fund"
    ),
    "jillian-manus": ReferenceIdentity(
        "jillian-manus", "Jillian Manus", "Jillian", "jillian-manus-structure-capital"
    ),
    "phil-nadel": ReferenceIdentity(
        "phil-nadel", "Phil Nadel", "Phil", "phil-nadel"
    ),
    "jesse-middleton": ReferenceIdentity(
        "jesse-middleton", "Jesse Middleton", "Jesse", "jesse-middleton-flybridge"
    ),
    "cyan-banister": ReferenceIdentity(
        "cyan-banister", "Cyan Banister", "Cyan", "cyan-banister-long-journey-ventures"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy, normalize, and audit Phase 1 rationale references."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--agentic-root", type=Path, default=DEFAULT_AGENTIC_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    manifest = build_reference_corpus(
        registry=load_registry(args.registry),
        identities=IDENTITIES,
        agentic_root=args.agentic_root.resolve(),
        output_root=args.output_root.resolve(),
        allow_missing=args.allow_missing,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
