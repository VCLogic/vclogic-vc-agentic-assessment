#!/usr/bin/env python3
"""Monotonically remove packaged pitch lines after the first decision boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from vc_clone_graph.manifest_builder import build_episode_manifest
from vc_clone_graph.pitch_cleanup import remove_post_boundary_lines


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--vc", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_root.resolve()))
    from inference.cuts import (  # noqa: PLC0415
        CUT_PROVENANCE,
        build_deployable_cuts,
        classify_turns,
    )

    package_root = args.package_root.resolve()
    investor_root = package_root / "data" / "investors" / args.vc
    pitches_root = investor_root / "pitches"
    audits_root = investor_root / "audits"
    episodes_root = args.source_root.resolve() / "data" / "episodes"
    results: list[dict[str, object]] = []

    for pitch_path in sorted(pitches_root.glob("*.txt")):
        episode_path = episodes_root / f"{pitch_path.stem}.json"
        if not episode_path.is_file():
            results.append({"episode_slug": pitch_path.stem, "status": "source_missing"})
            continue
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        generated = build_deployable_cuts(episode, args.vc)
        source_audit = generated["_meta"]["deployable_audit"]
        boundary = source_audit.get("boundary_event")
        boundary_index = boundary["index"] if boundary else None
        old_text = pitch_path.read_text(encoding="utf-8")
        new_text, removed, unmapped = remove_post_boundary_lines(
            old_text,
            classify_turns(episode, args.vc),
            boundary_index,
        )
        row = {
            "episode_slug": pitch_path.stem,
            "status": "changed" if removed else "unchanged",
            "boundary_event": boundary,
            "old_sha256": _sha256_text(old_text),
            "new_sha256": _sha256_text(new_text),
            "old_line_count": len(old_text.splitlines()),
            "new_line_count": len(new_text.splitlines()),
            "removed_lines": removed,
            "unmapped_line_count": len(unmapped),
        }
        results.append(row)
        if not args.apply:
            continue

        pitch_path.write_text(new_text, encoding="utf-8")
        audit_path = audits_root / f"{pitch_path.stem}.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["audit_source"] = f"{CUT_PROVENANCE} monotonic deployable audit"
        audit["pitch_sha256"] = _sha256_text(new_text)
        audit["boundary_cleanup"] = {
            "mode": "monotonic-post-boundary-removal",
            "boundary_event": boundary,
            "removed_line_count": len(removed),
            "unmapped_lines_preserved": len(unmapped),
        }
        audit_path.write_text(
            json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        build_episode_manifest(package_root, args.vc, pitch_path.stem)

    report = {
        "schema": "pitch-boundary-refresh-report-v1",
        "cut_provenance": CUT_PROVENANCE,
        "mode": "apply" if args.apply else "dry-run",
        "episode_count": len(results),
        "changed_count": sum(row.get("status") == "changed" for row in results),
        "removed_line_count": sum(
            len(row.get("removed_lines", [])) for row in results
        ),
        "episodes": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
