#!/usr/bin/env python3
"""Select and stage the 24-case panel-question Phase 1 canary."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil

from vc_clone_graph.evaluation import load_canonical_artifacts, load_registry
from vc_clone_graph.manifest_builder import build_episode_manifest
from vc_clone_graph.panel_question_canary import (
    CanaryCandidate, select_panel_question_canaries,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _copytree(source: Path, target: Path) -> None:
    shutil.copytree(source, target, dirs_exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path(
        "evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json"))
    parser.add_argument("--variants", type=Path, default=Path(
        "inputs/data/pitch_variants/panel-question-augmentation-v1"))
    parser.add_argument("--canonical-inputs", type=Path, default=Path("inputs"))
    parser.add_argument("--experiment-root", type=Path, default=Path(
        "outputs/panel-question-phase1-canary-2026-08-20"))
    args = parser.parse_args()

    manifest = json.loads((args.variants / "manifest.json").read_text())
    registry = load_registry(args.registry)
    artifacts = {
        (row.vc_slug, row.episode_slug): row
        for row in load_canonical_artifacts(registry)
    }
    candidates = []
    for row in manifest["cases"]:
        if row["status"] != "audited":
            continue
        artifact = artifacts[(row["registry_vc_slug"], row["episode_slug"])]
        candidates.append(CanaryCandidate(
            vc_slug=row["vc_slug"], registry_vc_slug=row["registry_vc_slug"],
            episode_slug=row["episode_slug"], actual_decision=row["actual_decision"],
            insertion_count=int(row["insertion_count"]), pitch_path=row["pitch_path"],
            audit_path=row["audit_path"], pitch_sha256=row["pitch_sha256"],
            canonical_artifact_root=str(artifact.artifact_path.parent),
        ))
    selected = select_panel_question_canaries(candidates)
    stage = args.experiment_root / "input-package"
    # Copy only model-visible package resources; labels and reports never enter stage.
    _copytree(args.canonical_inputs / "investors", stage / "investors")
    _copytree(args.canonical_inputs / "taxonomy", stage / "taxonomy")
    _copytree(args.canonical_inputs / "indexes", stage / "indexes")
    for vc in sorted({row.vc_slug for row in selected}):
        _copytree(args.canonical_inputs / "wiki" / vc, stage / "wiki" / vc)
        source = args.canonical_inputs / "data/investors" / vc
        target = stage / "data/investors" / vc
        _copytree(source / "precedents", target / "precedents")
        if (source / "portfolio-memory").is_dir():
            _copytree(source / "portfolio-memory", target / "portfolio-memory")

    config_paths = {}
    for row in selected:
        source_pitch = Path(row.pitch_path)
        treatment_audit = json.loads(Path(row.audit_path).read_text())
        original_audit_path = (
            args.canonical_inputs / "data/investors" / row.vc_slug
            / "audits" / f"{row.episode_slug}.json"
        )
        original_audit = json.loads(original_audit_path.read_text())
        pitch_target = stage / "data/investors" / row.vc_slug / "pitches" / f"{row.episode_slug}.txt"
        pitch_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_pitch, pitch_target)
        runtime_audit = {
            "schema": "pitch-leakage-audit-v1", "status": "audited",
            "vc_slug": row.vc_slug, "episode_slug": row.episode_slug,
            "pitch_path": str(pitch_target.relative_to(stage)),
            "pitch_sha256": sha256(pitch_target.read_bytes()).hexdigest(),
            "target_company_aliases": original_audit["target_company_aliases"],
            "leakage_checklist": {
                "actual_label_in_package": False,
                "complete_transcript_in_package": False,
                "founder_decision_acknowledgement_retained": False,
                "investor_evaluation_retained": False,
                "narrator_evaluation_retained": False,
                "post_decision_material_retained": False,
                "prior_episode_inputs_in_package": False,
                "target_investor_identity_in_pitch": False,
            },
            "panel_question_treatment": treatment_audit,
        }
        audit_target = stage / "data/investors" / row.vc_slug / "audits" / f"{row.episode_slug}.json"
        _write(audit_target, runtime_audit)
        build_episode_manifest(stage, row.vc_slug, row.episode_slug)

        config = json.loads((Path(row.canonical_artifact_root) / "run-config.json").read_text())
        output = args.experiment_root / "runs" / row.vc_slug
        config["run"].update({
            "input_root": str(stage), "output_root": str(output),
            "checkpoint_path": str(args.experiment_root / "checkpoints" / row.vc_slug
                                   / f"{row.episode_slug}.sqlite"),
            "mode": "phase1_only",
        })
        config_path = args.experiment_root / "configs" / row.vc_slug / f"{row.episode_slug}.json"
        _write(config_path, config)
        config_paths[(row.vc_slug, row.episode_slug)] = str(config_path)

    selection = {
        "schema": "panel-question-phase1-canary-selection-v1",
        "selection_policy": "top question count per VC/label; ties by episode slug",
        "per_vc_in": 2, "per_vc_out": 2, "case_count": len(selected),
        "variant_manifest": str(args.variants / "manifest.json"),
        "input_root": str(stage),
        "cases": [
            {**asdict(row), "config_path": config_paths[(row.vc_slug, row.episode_slug)]}
            for row in selected
        ],
    }
    _write(args.experiment_root / "selection.json", selection)
    print(json.dumps({
        "case_count": len(selected), "vc_count": len({r.vc_slug for r in selected}),
        "in_count": sum(r.actual_decision == "In" for r in selected),
        "out_count": sum(r.actual_decision == "Out" for r in selected),
        "input_root": str(stage),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
