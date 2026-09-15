#!/usr/bin/env python3
"""Diagnose canonical Phase 1 rationale errors and select a v4.3 canary."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

from vc_clone_graph.phase1_calibration_cases import build_calibration_cases
from vc_clone_graph.phase1_calibration_models import CalibrationPrediction
from vc_clone_graph.phase1_evaluation import load_phase1_cases
from vc_clone_graph.phase1_v43_diagnostics import (
    ErrorLedgerRow,
    build_error_ledger,
    diagnose_intervention,
    select_canary_cases,
    summarize_diagnostics,
)


DEFAULT_REGISTRY = Path("evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json")
DEFAULT_REFERENCES = Path("evaluation/phase1_ground_truth_rationales")
DEFAULT_TAXONOMY = Path("../agentic-vc-clone-framework/taxonomy/codebook_v_final.json")
DEFAULT_CALIBRATION = Path(
    "reports/evaluation/phase1-calibration-2026-08-17/rationale_predictions.csv"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    result.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    result.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    result.add_argument(
        "--calibration-predictions", type=Path, default=DEFAULT_CALIBRATION
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--neighbor-threshold", type=float, default=0.20)
    result.add_argument("--canary-per-vc", type=int, default=3)
    return result


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean in calibration CSV: {value!r}")


def _load_filtering_predictions(path: Path) -> list[CalibrationPrediction]:
    rows: list[CalibrationPrediction] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["condition"] != "filtering_only":
                continue
            rows.append(
                CalibrationPrediction(
                    vc_slug=raw["vc_slug"],
                    episode_slug=raw["episode_slug"],
                    label=raw["label"],
                    condition=raw["condition"],
                    score=float(raw["score"]),
                    predicted=_boolean(raw["predicted"]),
                    raw_predicted=_boolean(raw["raw_predicted"]),
                    reference=_boolean(raw["reference"]),
                )
            )
    if not rows:
        raise ValueError("calibration CSV contains no filtering_only predictions")
    return rows


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
        temporary.replace(path)
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(field for field in row if field not in fields)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})
    temporary.replace(path)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _severity(row: ErrorLedgerRow) -> float:
    if row.outcome != "false_negative":
        return 0.0
    return (
        1.0
        + 2.0 * (row.reference_salience == "primary")
        + 2.0 * (row.reference_decision_link == "explicit")
        + 1.5 * (row.reference_utterance_type == "decision_reason")
        + 0.5 * (row.reference_activation == "evaluated")
    )


def _failure_rows(ledger: Sequence[ErrorLedgerRow]) -> list[dict[str, object]]:
    grouped: dict[str, list[ErrorLedgerRow]] = defaultdict(list)
    for row in ledger:
        if row.error_type != "none":
            grouped[row.error_type].append(row)
    total_weight = sum(_severity(row) for row in ledger)
    return [
        {
            "error_type": error_type,
            "count": len(rows),
            "weighted_salient_miss_mass": sum(_severity(row) for row in rows),
            "weighted_salient_miss_share": (
                sum(_severity(row) for row in rows) / total_weight
                if total_weight
                else 0.0
            ),
            "vc_count": len({row.vc_slug for row in rows}),
        }
        for error_type, rows in sorted(grouped.items())
    ]


def _render_report(
    ledger: Sequence[ErrorLedgerRow],
    summaries: Mapping[str, Sequence[Mapping[str, object]]],
    canary: Sequence[object],
    diagnosis: Mapping[str, object],
) -> str:
    aggregate = {
        "tp": sum(row.outcome == "true_positive" for row in ledger),
        "fp": sum(row.outcome == "false_positive" for row in ledger),
        "fn": sum(row.outcome == "false_negative" for row in ledger),
    }
    precision = aggregate["tp"] / (aggregate["tp"] + aggregate["fp"])
    recall = aggregate["tp"] / (aggregate["tp"] + aggregate["fn"])
    f1 = 2 * precision * recall / (precision + recall)
    failures = _failure_rows(ledger)
    lines = [
        "# Phase 1 v4.3 Gate A Diagnostic",
        "",
        "> Development analysis using automated transcript-derived candidate references; these are not final human-validated ground truth.",
        "",
        f"Population: **{len({(row.vc_slug, row.episode_slug) for row in ledger})} cases**, "
        f"**{len({row.vc_slug for row in ledger})} VCs**, and **{len(ledger)} case-label rows**.",
        "",
        f"Canonical exact-label performance is precision **{precision:.3f}**, recall "
        f"**{recall:.3f}**, and F1 **{f1:.3f}**.",
        "",
        "This gate made no model or embedding calls and incurred **$0.00 API cost**.",
        "",
        "## Observable error decomposition",
        "",
        "| Error type | Count | Weighted salient-miss share | VCs |",
        "|---|---:|---:|---:|",
    ]
    for row in failures:
        lines.append(
            f"| {row['error_type']} | {row['count']} | "
            f"{float(row['weighted_salient_miss_share']):.3f} | {row['vc_count']} |"
        )
    lines.extend([
        "",
        "## Gate B intervention diagnosis",
        "",
        f"Across weighted salient misses, **{float(diagnosis['family_covered_weighted_share']):.1%}** "
        "already had a canonical rationale in the same broad family, while "
        f"**{float(diagnosis['discovery_weighted_share']):.1%}** lacked family coverage.",
        "",
        f"Selected intervention: **{diagnosis['selected_intervention']}**. "
        f"{diagnosis['reason']}",
    ])
    lines.extend([
        "",
        "## Per-investor diagnosis",
        "",
        "| Investor | Precision | Recall | F1 | Discovery omissions | Family-covered misses | Close-neighbor confusions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summaries["vc_rows"]:
        lines.append(
            f"| {row['vc_name']} | {float(row['precision']):.3f} | "
            f"{float(row['recall']):.3f} | {float(row['f1']):.3f} | "
            f"{row['discovery_omissions']} | {row['family_covered_exact_misses']} | "
            f"{row['taxonomy_neighbor_confusions']} |"
        )
    lines.extend([
        "",
        "## Deterministic canary",
        "",
        "| Investor | Role | Episode | Actual | F1 | Dominant failure |",
        "|---|---|---|---|---:|---|",
    ])
    for selected in canary:
        row = asdict(selected)  # type: ignore[arg-type]
        lines.append(
            f"| {row['vc_name']} | {row['role']} | {row['episode_slug']} | "
            f"{row['actual_decision']} | {float(row['f1']):.3f} | "
            f"{row['dominant_failure_type']} |"
        )
    lines.extend([
        "",
        "Gate A describes observable disagreement with automated references. It does not infer private investor cognition and it does not alter canonical v4/v4.1.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    project_root = Path.cwd().resolve()
    phase1_cases, taxonomy = load_phase1_cases(
        project_root, args.registry, args.references, args.taxonomy
    )
    cases = build_calibration_cases(project_root, phase1_cases, taxonomy)
    filtering = _load_filtering_predictions(args.calibration_predictions)
    ledger = build_error_ledger(
        cases,
        taxonomy,
        filtering,
        semantic_neighbor_threshold=args.neighbor_threshold,
    )
    summaries = summarize_diagnostics(ledger)
    canary = select_canary_cases(
        summaries["case_rows"], ledger, per_vc=args.canary_per_vc
    )
    diagnosis = diagnose_intervention(ledger)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "error_ledger": output / "error-ledger.csv",
        "case_summary": output / "case-summary.csv",
        "vc_summary": output / "vc-summary.csv",
        "label_summary": output / "label-summary.csv",
        "family_summary": output / "family-summary.csv",
        "confusion_pairs": output / "confusion-pairs.csv",
        "missed_salient": output / "missed-salient-rationales.csv",
        "failure_summary": output / "failure-summary.csv",
        "canary_selection": output / "canary-selection.csv",
        "canary_manifest": output / "canary-manifest.json",
        "intervention": output / "intervention-recommendation.json",
        "evaluation": output / "evaluation.md",
        "manifest": output / "manifest.json",
    }
    _write_csv(paths["error_ledger"], [asdict(row) for row in ledger])
    _write_csv(paths["case_summary"], summaries["case_rows"])
    _write_csv(paths["vc_summary"], summaries["vc_rows"])
    _write_csv(paths["label_summary"], summaries["label_rows"])
    _write_csv(paths["family_summary"], summaries["family_rows"])
    _write_csv(paths["confusion_pairs"], summaries["confusion_rows"])
    _write_csv(paths["missed_salient"], summaries["missed_salient_rows"])
    _write_csv(paths["failure_summary"], _failure_rows(ledger))
    _write_csv(paths["canary_selection"], [asdict(row) for row in canary])
    _write_json(paths["canary_manifest"], {
        "schema": "phase1-v43-canary-manifest-v1",
        "scientific_status": "development_canary_not_untouched_holdout",
        "cases": [asdict(row) for row in canary],
        "selected_intervention": diagnosis["selected_intervention"],
    })
    _write_json(paths["intervention"], diagnosis)
    _atomic_text(
        paths["evaluation"], _render_report(ledger, summaries, canary, diagnosis)
    )
    input_paths = {
        "registry": args.registry.resolve(),
        "reference_manifest": (args.references / "manifest.json").resolve(),
        "taxonomy": args.taxonomy.resolve(),
        "calibration_predictions": args.calibration_predictions.resolve(),
    }
    _write_json(paths["manifest"], {
        "schema": "phase1-v43-gate-a-manifest-v1",
        "scientific_status": "development_not_untouched_holdout",
        "reference_status": "automated_candidate_not_human_validated",
        "population": {
            "cases": len(cases),
            "investors": len({case.vc_slug for case in cases}),
            "labels": len(cases[0].labels),
            "case_label_rows": len(ledger),
            "canary_cases": len(canary),
        },
        "settings": {
            "semantic_neighbor_threshold": args.neighbor_threshold,
            "canary_per_vc": args.canary_per_vc,
        },
        "inputs": {
            name: {"path": str(path), "sha256": _hash(path)}
            for name, path in input_paths.items()
        },
        "outputs": sorted(str(path.relative_to(output)) for path in paths.values()),
        "api_cost_usd": 0.0,
        "canonical_phase1_modified": False,
    })
    print(
        f"Phase 1 v4.3 Gate A complete: cases={len(cases)} "
        f"label_rows={len(ledger)} canary={len(canary)} api_cost_usd=0.00 "
        f"report={paths['evaluation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
