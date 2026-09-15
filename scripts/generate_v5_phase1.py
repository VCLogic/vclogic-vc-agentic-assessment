#!/usr/bin/env python3
"""Generate self-contained all-VC investigation-v5 artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil

from vc_clone_graph.phase1_calibration_cases import build_calibration_cases
from vc_clone_graph.phase1_evaluation import load_phase1_cases
from vc_clone_graph.firewall import verify_package
from vc_clone_graph.rationale_completion import build_completion_cases, nested_association_predictions
from vc_clone_graph.v5_enrichment import enrich_investigation_v5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", type=Path, default=Path("evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json"))
    parser.add_argument("--references", type=Path, default=Path("evaluation/phase1_ground_truth_rationales"))
    parser.add_argument("--taxonomy", type=Path, default=Path("../agentic-vc-clone-framework/taxonomy/codebook_v_final.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--registry-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args(argv)

    root = Path.cwd()
    cases, taxonomy = load_phase1_cases(root, args.source_registry, args.references, args.taxonomy)
    calibration = build_calibration_cases(root, cases, taxonomy)
    predictions = nested_association_predictions(build_completion_cases(calibration), max_hypotheses=5)
    prediction_by_key = {(row.vc_slug, row.episode_slug): row for row in predictions}
    counts: Counter[str] = Counter()
    claims = claims_with = statements = statements_with = hypotheses = 0
    records: dict[str, str] = {}
    for case in calibration:
        source_path = Path(case.phase1_case.artifact_path)
        if not source_path.is_absolute():
            source_path = root / source_path
        source_bytes = source_path.read_bytes()
        source = json.loads(source_bytes)
        enriched = enrich_investigation_v5(
            source,
            sha256(source_bytes).hexdigest(),
            prediction_by_key[(case.vc_slug, case.episode_slug)],
            taxonomy_labels=set(taxonomy),
        )
        run_root = args.output_root / "investors" / case.vc_slug / case.episode_slug
        target = run_root / "phase1" / "investigation.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(enriched.model_dump_json(indent=2) + "\n", encoding="utf-8")
        digest = sha256(target.read_bytes()).hexdigest()
        target.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
        source_run_root = source_path.parent.parent
        for name in (
            "precedent-manifest.filtered.json",
            "portfolio-manifest.filtered.json",
            "wiki-sanitization.json",
        ):
            candidate = source_run_root / name
            if candidate.is_file():
                shutil.copyfile(candidate, run_root / name)
        source_config = json.loads(
            (source_run_root / "run-config.json").read_text(encoding="utf-8")
        )
        package = verify_package(
            root / source_config["run"]["input_root"],
            source_config["run"]["vc_slug"],
            case.episode_slug,
            taxonomy_path=source_config["run"]["taxonomy_path"],
        )
        (run_root / "input-provenance.json").write_text(
            json.dumps({
                "pitch_path": str(package.pitch),
                "pitch_sha256": sha256(package.pitch.read_bytes()).hexdigest(),
                "package_manifest_path": str(package.manifest),
                "package_manifest_sha256": sha256(package.manifest.read_bytes()).hexdigest(),
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source_state_path = source_run_root / "state.json"
        if not source_state_path.is_file():
            raise FileNotFoundError(f"canonical source lacks replay state: {source_state_path}")
        source_state = json.loads(source_state_path.read_text(encoding="utf-8"))
        source_state.update({
            "investigation": enriched.model_dump(mode="json"),
            "investigation_sha256": digest,
            "phase2_status": "not_run",
            "decision": {},
            "decision_sha256": "",
            "taxonomy_reflection": None,
        })
        (run_root / "state.json").write_text(
            json.dumps(source_state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run_root / "summary.json").write_text(json.dumps({
            "contract_version": "v5", "episode_slug": case.episode_slug,
            "phase1_status": "enriched_from_canonical", "phase2_status": "not_run",
            "source_investigation_sha256": enriched.source_investigation_sha256,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        records[f"{case.vc_slug}/{case.episode_slug}"] = digest
        counts[case.vc_slug] += 1
        claims += len(enriched.rationales)
        claims_with += sum(bool(row.associated_rationales) for row in enriched.rationales)
        statements += len(enriched.material_statement_coverage)
        statements_with += sum(bool(row.associated_rationales) for row in enriched.material_statement_coverage)
        hypotheses += len(enriched.episode_level_associations)

    source_registry = json.loads(args.source_registry.read_text(encoding="utf-8"))
    registry = {"schema": source_registry["schema"], "investors": {}}
    for vc_slug, spec in source_registry["investors"].items():
        registry["investors"][vc_slug] = {
            **{key: value for key, value in spec.items() if key != "sources"},
            "sources": [{"kind": "summary_glob", "path": f"../{args.output_root.as_posix()}/investors/{vc_slug}/*/summary.json"}],
        }
    args.registry_output.parent.mkdir(parents=True, exist_ok=True)
    args.registry_output.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = {
        "schema": "v5-phase1-generation-v1", "api_cost_usd": 0.0,
        "case_count": len(calibration), "vc_counts": dict(sorted(counts.items())),
        "claim_count": claims, "claims_with_associations": claims_with,
        "claim_association_coverage": claims_with / claims,
        "material_statement_count": statements,
        "material_statements_with_associations": statements_with,
        "material_statement_association_coverage": statements_with / statements if statements else 0.0,
        "episode_hypothesis_count": hypotheses, "records": records,
    }
    args.report_output.mkdir(parents=True, exist_ok=True)
    (args.report_output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.report_output / "evaluation.md").write_text(
        "# V5 Associated-Rationale Phase 1\n\n"
        f"Generated **{len(calibration)}** source-bound v5 investigations across **{len(counts)}** VCs.\n\n"
        f"- Claims with associations: {claims_with}/{claims} ({claims_with/claims:.1%})\n"
        f"- Material statements with associations: {statements_with}/{statements} ({statements_with/statements if statements else 0:.1%})\n"
        f"- Episode-level hypotheses: {hypotheses}\n- API cost: $0.00\n",
        encoding="utf-8",
    )
    print(f"v5 phase1 complete cases={len(calibration)} vcs={len(counts)} cost=$0.00")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
