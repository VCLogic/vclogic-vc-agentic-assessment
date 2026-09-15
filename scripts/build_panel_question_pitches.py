#!/usr/bin/env python3
"""Generate audited neutral panel-question pitch variants."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from vc_clone_graph.evaluation import load_canonical_artifacts, load_registry
from vc_clone_graph.panel_question_pitches import build_panel_question_pitch


VC_INPUT_SLUGS = {
    "charles-hudson": "charles-hudson-precursor-ventures",
    "cyan-banister": "cyan-banister-long-journey-ventures",
    "elizabeth-yin": "elizabeth-yin-hustle-fund",
    "jesse-middleton": "jesse-middleton-flybridge",
    "jillian-manus": "jillian-manus-structure-capital",
    "phil-nadel": "phil-nadel",
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path(
        "evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json"))
    parser.add_argument("--inputs", type=Path, default=Path("inputs"))
    parser.add_argument("--episodes", type=Path, default=Path(
        "../agentic-vc-clone-framework/data/episodes"))
    parser.add_argument("--output", type=Path, default=Path(
        "inputs/data/pitch_variants/panel-question-augmentation-v1"))
    args = parser.parse_args()
    registry = load_registry(args.registry)
    rows = []
    for artifact in load_canonical_artifacts(registry):
        input_slug = VC_INPUT_SLUGS[artifact.vc_slug]
        investor = args.inputs / "data/investors" / input_slug
        destination = args.output / "investors" / input_slug / artifact.episode_slug
        paths = {
            "pitch": investor / "pitches" / f"{artifact.episode_slug}.txt",
            "audit": investor / "audits" / f"{artifact.episode_slug}.json",
            "record": investor / "precedents/records" / f"{artifact.episode_slug}.json",
            "episode": args.episodes / f"{artifact.episode_slug}.json",
        }
        try:
            missing = [name for name, path in paths.items() if not path.is_file()]
            if missing:
                raise ValueError(f"missing sources: {', '.join(missing)}")
            result = build_panel_question_pitch(
                canonical_pitch=paths["pitch"].read_text(encoding="utf-8"),
                source_record=_json(paths["record"]), episode=_json(paths["episode"]),
                audit=_json(paths["audit"]), target_vc_slug=input_slug,
            )
            pitch_hash = sha256(result.pitch_text.encode()).hexdigest()
            canonical_hash = sha256(paths["pitch"].read_bytes()).hexdigest()
            audit_payload = {
                "schema": "panel-question-pitch-audit-v1",
                "status": "audited" if result.insertions else "no_safe_questions",
                "vc_slug": input_slug, "registry_vc_slug": artifact.vc_slug,
                "episode_slug": artifact.episode_slug,
                "actual_decision": artifact.actual_decision,
                "canonical_pitch_path": str(paths["pitch"]),
                "canonical_pitch_sha256": canonical_hash,
                "source_record_path": str(paths["record"]),
                "source_record_sha256": sha256(paths["record"].read_bytes()).hexdigest(),
                "episode_path": str(paths["episode"]),
                "episode_sha256": sha256(paths["episode"].read_bytes()).hexdigest(),
                "pitch_sha256": pitch_hash,
                "boundary_turn_index": result.boundary_turn_index,
                "neutral_speaker_label": "Panel Investor",
                "insertions": [asdict(row) for row in result.insertions],
                "omitted_candidates": [asdict(row) for row in result.omitted],
                "leakage_checklist": {
                    "target_investor_turn_retained": False,
                    "other_investor_identity_model_visible": False,
                    "verdict_or_offer_turn_retained": False,
                    "post_boundary_turn_retained": False,
                    "founder_content_added": False,
                },
            }
            _write(destination / "pitch.txt", result.pitch_text)
            _write(destination / "audit.json", json.dumps(
                audit_payload, indent=2, sort_keys=True) + "\n")
            rows.append({
                "vc_slug": input_slug, "registry_vc_slug": artifact.vc_slug,
                "episode_slug": artifact.episode_slug,
                "actual_decision": artifact.actual_decision,
                "status": audit_payload["status"],
                "insertion_count": len(result.insertions),
                "omitted_count": len(result.omitted),
                "pitch_path": str(destination / "pitch.txt"),
                "audit_path": str(destination / "audit.json"),
                "pitch_sha256": pitch_hash,
            })
        except Exception as exc:
            rows.append({
                "vc_slug": input_slug, "registry_vc_slug": artifact.vc_slug,
                "episode_slug": artifact.episode_slug,
                "actual_decision": artifact.actual_decision,
                "status": "failed", "insertion_count": 0, "omitted_count": 0,
                "error": f"{type(exc).__name__}: {exc}",
            })
    manifest = {
        "schema": "panel-question-pitch-collection-v1",
        "registry": str(args.registry), "case_count": len(rows),
        "audited_count": sum(row["status"] == "audited" for row in rows),
        "no_safe_questions_count": sum(row["status"] == "no_safe_questions" for row in rows),
        "failed_count": sum(row["status"] == "failed" for row in rows),
        "cases": rows,
    }
    _write(args.output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "case_count", "audited_count", "no_safe_questions_count", "failed_count")},
        sort_keys=True))
    return 1 if manifest["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
