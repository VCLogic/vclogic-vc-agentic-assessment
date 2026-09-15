#!/usr/bin/env python3
"""Materialize a reviewed pitch-window ledger, pitch packages, and precedents."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from vc_clone_graph.dataset_prep import (
    compile_review_rows,
    package_relative_path,
    write_pitch_package,
)
from vc_clone_graph.firewall import verify_package
from vc_clone_graph.manifest_builder import build_episode_manifest
from vc_clone_graph.precedent_builder import build_precedent_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, default=Path("."))
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--ledger-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--vc", required=True)
    parser.add_argument("--investor-alias", action="append", required=True)
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    package_root = args.package_root.resolve()
    transcript_root = source_root / "data" / "episodes"
    review_rows = json.loads(args.review.read_text(encoding="utf-8"))
    compiled = compile_review_rows(
        review_rows=review_rows,
        transcript_root=transcript_root,
        vc_slug=args.vc,
        investor_aliases=args.investor_alias,
    )
    args.ledger_output.parent.mkdir(parents=True, exist_ok=True)
    args.ledger_output.write_text(
        json.dumps(compiled, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    investor_root = package_root / "inputs" / "data" / "investors" / args.vc
    created: list[str] = []
    preserved: list[str] = []
    for row in compiled:
        if not row["evaluation_eligible"]:
            continue
        slug = row["episode_slug"]
        pitch = investor_root / "pitches" / f"{slug}.txt"
        audit = investor_root / "audits" / f"{slug}.json"
        if not args.replace_existing and pitch.is_file() and audit.is_file():
            preserved.append(slug)
            continue
        write_pitch_package(
            episode_path=transcript_root / f"{slug}.json",
            compiled_review=row,
            investor_root=investor_root,
            vc_slug=args.vc,
            audit_source=package_relative_path(args.ledger_output, package_root),
        )
        created.append(slug)

    precedent_root = investor_root / "precedents"
    records = build_precedent_corpus(
        transcript_root,
        args.ledger_output,
        precedent_root,
        args.investor_alias,
    )

    verified: list[str] = []
    for row in compiled:
        if not row["evaluation_eligible"]:
            continue
        slug = row["episode_slug"]
        build_episode_manifest(package_root / "inputs", args.vc, slug)
        verify_package(package_root / "inputs", args.vc, slug)
        verified.append(slug)

    counts = Counter(row["pitch_window_decision"] for row in compiled)
    report = {
        "schema": "investor-dataset-preparation-report-v1",
        "vc_slug": args.vc,
        "raw_review_count": len(review_rows),
        "compiled_ledger_count": len(compiled),
        "decision_counts": dict(sorted(counts.items())),
        "eligible_count": len(verified),
        "created_pitch_packages": created,
        "preserved_pitch_packages": preserved,
        "verified_pitch_packages": verified,
        "precedent_record_count": len(records),
        "precedent_observed_count": sum(
            record.decision.status in {"In", "Out"} for record in records
        ),
        "precedent_unobserved_count": sum(
            record.decision.status == "unobserved" for record in records
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
