#!/usr/bin/env python3
"""Build chronologically safe, episode-derived portfolio memory for one or all VCs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

from vc_clone_graph.portfolio_memory_builder import build_portfolio_memory
from vc_clone_graph.providers.sentence_transformers import (
    SentenceTransformerEmbeddingProvider,
)


DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
REFERENCE_SLUGS = {
    "charles-hudson-precursor-ventures": "charles-hudson",
    "cyan-banister-long-journey-ventures": "cyan-banister",
    "elizabeth-yin-hustle-fund": "elizabeth-yin",
    "jesse-middleton-flybridge": "jesse-middleton",
    "jillian-manus-structure-capital": "jillian-manus",
    "phil-nadel": "phil-nadel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vc", action="append", dest="vcs")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-revision", default=DEFAULT_REVISION)
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--notes-root", type=Path, default=Path("../taxonomy/portfolio_notes"))
    parser.add_argument(
        "--references-root",
        type=Path,
        default=Path("evaluation/phase1_ground_truth_rationales/records"),
    )
    return parser.parse_args()


def _registry(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def main() -> None:
    args = parse_args()
    if args.all == bool(args.vcs):
        raise SystemExit("select exactly one of --all or one-or-more --vc values")
    selected = sorted(REFERENCE_SLUGS) if args.all else list(dict.fromkeys(args.vcs))
    unknown = sorted(set(selected) - REFERENCE_SLUGS.keys())
    if unknown:
        raise SystemExit(f"unknown VC slug(s): {', '.join(unknown)}")
    embedder = SentenceTransformerEmbeddingProvider(
        args.embedding_model,
        revision=args.embedding_revision,
        device=args.embedding_device,
        batch_size=32,
    )
    reports: list[dict[str, object]] = []
    for vc_slug in selected:
        registry = _registry(Path("inputs/investors") / f"{vc_slug}.toml")
        report = build_portfolio_memory(
            vc_slug=vc_slug,
            investor_aliases=registry["speaker_names"],
            precedent_root=Path("inputs/data/investors") / vc_slug / "precedents",
            notes_path=args.notes_root / f"{vc_slug}.json",
            reference_records_root=args.references_root / REFERENCE_SLUGS[vc_slug],
            reference_vc_slug=REFERENCE_SLUGS[vc_slug],
            output_root=Path("inputs/data/investors") / vc_slug / "portfolio-memory",
            embedder=embedder,
        )
        print(
            f"{vc_slug}: accepted={report['accepted_count']} "
            f"named={report['named_count']} anonymous={report['anonymous_count']} "
            f"rejected={report['rejected_count']} candidates={report['candidate_unextracted_count']}"
        )
        reports.append(report)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "schema": "portfolio-memory-all-vc-build-report-v1",
                    "embedding_model": args.embedding_model,
                    "embedding_revision": args.embedding_revision,
                    "vc_count": len(reports),
                    "accepted_count": sum(int(row["accepted_count"]) for row in reports),
                    "named_count": sum(int(row["named_count"]) for row in reports),
                    "anonymous_count": sum(int(row["anonymous_count"]) for row in reports),
                    "rejected_count": sum(int(row["rejected_count"]) for row in reports),
                    "investors": reports,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
