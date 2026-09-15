#!/usr/bin/env python3
"""Build the physical hybrid v4/v4.1 canonical evaluation snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from vc_clone_graph.canonical_snapshot import build_snapshot, write_snapshot_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "evaluation" / "canonical_runs.json"
DEFAULT_OVERRIDES = (
    PROJECT_ROOT
    / "outputs"
    / "openrouter-luna-portfolio-conflicts-v41-2026-08-14"
    / "batch-status.json"
)
DEFAULT_DESTINATION = (
    PROJECT_ROOT / "outputs" / "canonical-v4-v41-portfolio-2026-08-15"
)
DEFAULT_OUTPUT_REGISTRY = (
    PROJECT_ROOT
    / "evaluation"
    / "canonical_runs_v4_v41_portfolio_2026-08-15.json"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    result.add_argument("--override-status", type=Path, default=DEFAULT_OVERRIDES)
    result.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    result.add_argument("--output-registry", type=Path, default=DEFAULT_OUTPUT_REGISTRY)
    result.add_argument("--expected-total", type=int, default=301)
    result.add_argument("--expected-overrides", type=int, default=18)
    return result


def main() -> None:
    args = parser().parse_args()
    manifest = build_snapshot(
        args.registry,
        args.override_status,
        args.destination,
        expected_total=args.expected_total,
        expected_overrides=args.expected_overrides,
    )
    try:
        write_snapshot_registry(args.registry, args.destination, args.output_registry)
    except BaseException as exc:
        raise RuntimeError(
            f"snapshot was published at {args.destination}, but registry generation failed; "
            "the snapshot was preserved for forensic inspection"
        ) from exc
    tiers = manifest["counts"]["selection_tier"]
    print(
        f"Published {manifest['counts']['total']} artifacts at {args.destination}: "
        f"{tiers['portfolio_v41_override']} overrides, "
        f"{tiers['prior_canonical']} retained."
    )
    print(f"Registry: {args.output_registry}")


if __name__ == "__main__":
    main()
