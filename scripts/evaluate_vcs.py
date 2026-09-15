#!/usr/bin/env python3
"""Evaluate canonical VC Phase 2 classification and ranking outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from vc_clone_graph.evaluation import (
    build_evaluation,
    load_canonical_predictions,
    load_registry,
    render_markdown,
    write_evaluation_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "evaluation" / "canonical_runs.json"


def _top_ks(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top-k budgets must be comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("top-k budgets must be unique positive integers")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate canonical Phase 2 VC decisions and investment-likelihood ranking.",
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--vc", action="append", help="Canonical VC slug; repeatable.")
    selection.add_argument("--all", action="store_true", help="Evaluate every registered VC.")
    selection.add_argument("--list-vcs", action="store_true", help="List canonical VC slugs.")
    parser.add_argument("--top-k", type=_top_ks, default=(1, 3, 5, 10, 20))
    parser.add_argument("--format", choices=("markdown", "csv", "both"), default="markdown")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    registry = load_registry(args.registry)
    if args.list_vcs:
        for slug, spec in registry.investors.items():
            print(f"{slug}\t{spec.display_name}\t{spec.eligible_count}")
        return 0
    if args.format in {"csv", "both"} and args.output_dir is None:
        parser.error("--output-dir is required for CSV output")
    selected = list(registry.investors) if args.all else args.vc
    rows = load_canonical_predictions(registry, selected)
    result = build_evaluation(rows, top_ks=args.top_k)
    markdown = render_markdown(result)
    if args.format in {"markdown", "both"}:
        print(markdown, end="")
    if args.output_dir is not None:
        paths = write_evaluation_outputs(
            result,
            rows,
            args.output_dir,
            include_csv=args.format in {"csv", "both"},
            include_markdown=args.format in {"markdown", "both"},
        )
        if args.format == "csv":
            for name, path in paths.items():
                print(f"{name}\t{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
