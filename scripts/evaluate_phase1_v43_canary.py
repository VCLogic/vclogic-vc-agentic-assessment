#!/usr/bin/env python3
"""Evaluate Phase 1 v4.3 against canonical v4/v4.1 on the frozen canary."""

from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import json
from pathlib import Path
from statistics import fmean

from scipy.stats import binomtest

from vc_clone_graph.phase1_evaluation import (
    _prediction_rationales,
    evaluate_cases,
    load_phase1_cases,
    score_set,
    sha256_file,
)


DEFAULT_REGISTRY = Path("evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json")
DEFAULT_REFERENCES = Path("evaluation/phase1_ground_truth_rationales")
DEFAULT_TAXONOMY = Path("../agentic-vc-clone-framework/taxonomy/codebook_v_final.json")
DEFAULT_MANIFEST = Path(
    "reports/evaluation/phase1-v43-diagnostic-2026-08-17/canary-manifest.json"
)
DEFAULT_RUNS = Path("outputs/phase1-v43-canary-2026-08-17/investors")
DEFAULT_STATUS = Path("reports/evaluation/phase1-v43-canary-2026-08-17/status.csv")
DEFAULT_OUTPUT = Path("reports/evaluation/phase1-v43-canary-2026-08-17")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rich_recall(result: dict, subset: str) -> float:
    row = next((row for row in result["rich_subset_rows"] if row["subset"] == subset), None)
    return float(row["recall"]) if row is not None else 0.0


def _vc_f1(result: dict) -> dict[str, float]:
    return {
        row["value"]: float(row["micro_f1"])
        for row in result["scope_rows"]
        if row["dimension"] == "vc"
    }


def _mapping_effect_summary(before: dict, after: dict) -> dict[str, float]:
    """Return deltas attributable to mapping within the same fresh run."""
    keys = (
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "family_micro_f1",
        "average_predicted_size",
    )
    return {key: float(after[key]) - float(before[key]) for key in keys}


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = {
        (row["vc_slug"], row["episode_slug"]): row for row in manifest["cases"]
    }
    all_cases, taxonomy = load_phase1_cases(
        project_root, args.registry, args.references, args.taxonomy
    )
    canonical = [case for case in all_cases if (case.vc_slug, case.episode_slug) in selected]
    if len(canonical) != len(selected):
        raise ValueError("canonical canary coverage is incomplete")

    pre_mapping_cases = []
    v43_cases = []
    missing: list[dict] = []
    mapping_rows: list[dict] = []
    for case in canonical:
        root = args.runs / case.vc_slug / case.episode_slug
        candidate_path = root / "phase1/candidate-investigation.json"
        investigation_path = root / "phase1/investigation.json"
        mapping_path = root / "phase1/rationale-mapping.json"
        if (
            not candidate_path.is_file()
            or not investigation_path.is_file()
            or not mapping_path.is_file()
        ):
            missing.append({"vc_slug": case.vc_slug, "episode_slug": case.episode_slug})
            continue
        candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate_predicted = _prediction_rationales(
            candidate_payload, taxonomy, str(candidate_path)
        )
        pre_mapping_cases.append(
            replace(case, predicted=candidate_predicted, artifact_path=candidate_path)
        )
        payload = json.loads(investigation_path.read_text(encoding="utf-8"))
        predicted = _prediction_rationales(payload, taxonomy, str(investigation_path))
        v43_cases.append(replace(case, predicted=predicted, artifact_path=investigation_path))
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        for row in mapping["dispositions"]:
            mapping_rows.append({
                "vc_slug": case.vc_slug,
                "episode_slug": case.episode_slug,
                "rationale_id": row["rationale_id"],
                "action": row["action"],
                "target_taxonomy_label": row["target_taxonomy_label"],
                "merge_into_rationale_id": row.get("merge_into_rationale_id") or "",
                "mapping_status": payload["mapping_status"],
                "mapping_justification": row["mapping_justification"],
            })
    if not v43_cases:
        raise ValueError("no usable v4.3 canary outputs")
    usable_keys = {(case.vc_slug, case.episode_slug) for case in v43_cases}
    canonical_usable = [
        case for case in canonical if (case.vc_slug, case.episode_slug) in usable_keys
    ]
    canonical_result = evaluate_cases(canonical_usable, taxonomy)
    pre_mapping_result = evaluate_cases(pre_mapping_cases, taxonomy)
    v43_result = evaluate_cases(v43_cases, taxonomy)
    canonical_episode = {
        (row["vc_slug"], row["episode_slug"]): row
        for row in canonical_result["episode_rows"]
    }
    v43_episode = {
        (row["vc_slug"], row["episode_slug"]): row
        for row in v43_result["episode_rows"]
    }
    pre_mapping_episode = {
        (row["vc_slug"], row["episode_slug"]): row
        for row in pre_mapping_result["episode_rows"]
    }
    paired_rows = []
    improved = worsened = 0
    mapping_improved = mapping_worsened = 0
    for old, pre_mapping, new in zip(
        canonical_usable, pre_mapping_cases, v43_cases, strict=True
    ):
        key = old.vc_slug, old.episode_slug
        old_labels = {row.label for row in old.predicted}
        pre_mapping_labels = {row.label for row in pre_mapping.predicted}
        new_labels = {row.label for row in new.predicted}
        references = {row.label for row in old.reference}
        for label in taxonomy:
            before_correct = (label in old_labels) == (label in references)
            after_correct = (label in new_labels) == (label in references)
            improved += int(after_correct and not before_correct)
            worsened += int(before_correct and not after_correct)
            pre_mapping_correct = (label in pre_mapping_labels) == (label in references)
            mapping_improved += int(after_correct and not pre_mapping_correct)
            mapping_worsened += int(pre_mapping_correct and not after_correct)
        old_row = canonical_episode[key]
        pre_mapping_row = pre_mapping_episode[key]
        new_row = v43_episode[key]
        paired_rows.append({
            "vc_slug": old.vc_slug,
            "episode_slug": old.episode_slug,
            "actual_decision": old.actual_decision,
            "role": selected[key]["role"],
            "canonical_precision": old_row["precision"],
            "canonical_recall": old_row["recall"],
            "canonical_f1": old_row["f1"],
            "pre_mapping_precision": pre_mapping_row["precision"],
            "pre_mapping_recall": pre_mapping_row["recall"],
            "pre_mapping_f1": pre_mapping_row["f1"],
            "v43_precision": new_row["precision"],
            "v43_recall": new_row["recall"],
            "v43_f1": new_row["f1"],
            "f1_delta": float(new_row["f1"]) - float(old_row["f1"]),
            "mapping_f1_delta": (
                float(new_row["f1"]) - float(pre_mapping_row["f1"])
            ),
            "canonical_count": old_row["predicted_count"],
            "pre_mapping_count": pre_mapping_row["predicted_count"],
            "v43_count": new_row["predicted_count"],
            "false_labels_removed": len((old_labels - references) - new_labels),
            "correct_labels_lost": len((old_labels & references) - new_labels),
            "correct_labels_added": len((new_labels & references) - old_labels),
        })
    discordant = improved + worsened
    exact_p = (
        float(binomtest(min(improved, worsened), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    old_overall, new_overall = canonical_result["overall"], v43_result["overall"]
    pre_mapping_overall = pre_mapping_result["overall"]
    mapping_effect = _mapping_effect_summary(pre_mapping_overall, new_overall)
    old_vc, new_vc = _vc_f1(canonical_result), _vc_f1(v43_result)
    preserved_vcs = sum(new_vc[vc] >= old_vc[vc] for vc in old_vc)
    old_primary_explicit = _rich_recall(canonical_result, "primary_and_explicit")
    new_primary_explicit = _rich_recall(v43_result, "primary_and_explicit")
    criteria = {
        "all_18_usable": len(v43_cases) == 18,
        "exact_f1_gain_at_least_0_05": (
            float(new_overall["micro_f1"]) - float(old_overall["micro_f1"]) >= 0.05
        ),
        "precision_at_least_0_30": float(new_overall["micro_precision"]) >= 0.30,
        "recall_at_least_0_60": float(new_overall["micro_recall"]) >= 0.60,
        "primary_explicit_recall_loss_at_most_0_03": (
            new_primary_explicit >= old_primary_explicit - 0.03
        ),
        "family_f1_not_reduced": (
            float(new_overall["family_micro_f1"]) >= float(old_overall["family_micro_f1"])
        ),
        "at_least_four_vcs_preserved": preserved_vcs >= 4,
        "rationale_count_not_doubled": (
            float(new_overall["average_predicted_size"])
            <= 2 * float(old_overall["average_predicted_size"])
        ),
    }
    recommendation = (
        "manual_audit_required"
        if missing
        else "promote_to_larger_paired_run"
        if all(criteria.values())
        else "retain_canonical"
    )
    status_rows = []
    total_cost = 0.0
    if args.status.is_file():
        with args.status.open(encoding="utf-8", newline="") as stream:
            status_rows = list(csv.DictReader(stream))
        total_cost = sum(float(row["cost_usd"]) for row in status_rows)
    summary = {
        "schema": "phase1-v43-canary-evaluation-v1",
        "scientific_status": "development_canary_not_untouched_holdout",
        "case_count": len(v43_cases),
        "missing": missing,
        "canonical": old_overall,
        "pre_mapping": pre_mapping_overall,
        "v43": new_overall,
        "deltas": {
            "micro_precision": float(new_overall["micro_precision"]) - float(old_overall["micro_precision"]),
            "micro_recall": float(new_overall["micro_recall"]) - float(old_overall["micro_recall"]),
            "micro_f1": float(new_overall["micro_f1"]) - float(old_overall["micro_f1"]),
            "family_micro_f1": float(new_overall["family_micro_f1"]) - float(old_overall["family_micro_f1"]),
            "average_predicted_size": float(new_overall["average_predicted_size"]) - float(old_overall["average_predicted_size"]),
        },
        "primary_explicit_recall": {
            "canonical": old_primary_explicit,
            "v43": new_primary_explicit,
        },
        "per_vc_f1": {
            vc: {"canonical": old_vc[vc], "v43": new_vc[vc], "delta": new_vc[vc] - old_vc[vc]}
            for vc in old_vc
        },
        "paired_label_correctness": {
            "improved": improved,
            "worsened": worsened,
            "discordant": discordant,
            "exact_two_sided_p": exact_p,
        },
        "mapping_only_label_correctness": {
            "improved": mapping_improved,
            "worsened": mapping_worsened,
            "discordant": mapping_improved + mapping_worsened,
        },
        "mapping_only_deltas": mapping_effect,
        "criteria": criteria,
        "recommendation": recommendation,
        "usage": {"total_cost_usd": total_cost, "status_rows": len(status_rows)},
    }
    _write_json(output / "metrics.json", summary)
    _write_csv(output / "paired-cases.csv", paired_rows)
    _write_csv(output / "mapping-actions.csv", mapping_rows)
    _write_json(output / "canonical-result.json", canonical_result)
    _write_json(output / "pre-mapping-result.json", pre_mapping_result)
    _write_json(output / "v43-result.json", v43_result)
    report = [
        "# Phase 1 v4.3 Paired Canary",
        "",
        "> Development canary using automated transcript-derived rationale references; not an untouched holdout.",
        "",
        f"Usable cases: **{len(v43_cases)}/18**. Provider cost: **${total_cost:.4f}**.",
        "",
        "| Method | Precision | Recall | F1 | Family F1 | Avg rationales |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Canonical v4/v4.1 | {old_overall['micro_precision']:.3f} | {old_overall['micro_recall']:.3f} | {old_overall['micro_f1']:.3f} | {old_overall['family_micro_f1']:.3f} | {old_overall['average_predicted_size']:.2f} |",
        f"| Fresh v4.3 candidates (before mapping) | {pre_mapping_overall['micro_precision']:.3f} | {pre_mapping_overall['micro_recall']:.3f} | {pre_mapping_overall['micro_f1']:.3f} | {pre_mapping_overall['family_micro_f1']:.3f} | {pre_mapping_overall['average_predicted_size']:.2f} |",
        f"| v4.3 mapped | {new_overall['micro_precision']:.3f} | {new_overall['micro_recall']:.3f} | {new_overall['micro_f1']:.3f} | {new_overall['family_micro_f1']:.3f} | {new_overall['average_predicted_size']:.2f} |",
        "",
        f"Paired label correctness improved on **{improved}** discordant labels and worsened on **{worsened}** (exact two-sided p={exact_p:.4f}).",
        "",
        "The canonical-to-v4.3 comparison includes fresh-run variability. The isolated mapping effect compares the fresh candidates immediately before and after mapping.",
        f"Mapping alone changed correctness on **{mapping_improved + mapping_worsened}** label decisions: **{mapping_improved} improved**, **{mapping_worsened} worsened**; exact F1 delta **{mapping_effect['micro_f1']:+.3f}**.",
        "",
        f"Recommendation: **{recommendation}**.",
        "",
        "## Promotion criteria",
        "",
        *[f"- {'PASS' if value else 'FAIL'} — `{name}`" for name, value in criteria.items()],
    ]
    (output / "evaluation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _write_json(output / "manifest.json", {
        "schema": "phase1-v43-canary-evaluation-manifest-v1",
        "inputs": {
            "registry": str(args.registry.resolve()),
            "registry_sha256": sha256_file(args.registry.resolve()),
            "references_manifest_sha256": sha256_file((args.references / "manifest.json").resolve()),
            "taxonomy_sha256": sha256_file(args.taxonomy.resolve()),
            "canary_manifest_sha256": sha256_file(args.manifest.resolve()),
        },
        "canonical_modified": False,
        "recommendation": recommendation,
    })
    print(
        f"v4.3 canary n={len(v43_cases)} P={new_overall['micro_precision']:.3f} "
        f"R={new_overall['micro_recall']:.3f} F1={new_overall['micro_f1']:.3f}; "
        f"recommendation={recommendation} cost=${total_cost:.4f}"
    )


if __name__ == "__main__":
    main()
