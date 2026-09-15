#!/usr/bin/env python3
"""Evaluate the v5 Phase 2 canary panel against labels and canonical decisions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scipy.stats import binomtest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _metrics(rows: list[dict[str, object]], prefix: str) -> dict[str, float | int]:
    targets = [int(row["actual_decision"] == "In") for row in rows]
    decisions = [int(row[f"{prefix}_decision"] == "In") for row in rows]
    scores = [float(row[f"{prefix}_likelihood"]) for row in rows]
    return {
        "correct": sum(target == decision for target, decision in zip(targets, decisions)),
        "balanced_accuracy": balanced_accuracy_score(targets, decisions),
        "in_precision": precision_score(targets, decisions, zero_division=0),
        "in_recall": recall_score(targets, decisions, zero_division=0),
        "in_f1": f1_score(targets, decisions, zero_division=0),
        "average_precision": average_precision_score(targets, scores),
        "roc_auc": roc_auc_score(targets, scores),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, default=Path(
        "outputs/canonical-v4-v41-portfolio-2026-08-15/investors"
    ))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = json.loads((args.run_root / "results.json").read_text(encoding="utf-8"))
    rows = raw["rows"]
    for row in rows:
        canonical = json.loads((
            args.canonical_root / str(row["vc_slug"]) / str(row["episode_slug"])
            / "phase2/decision.json"
        ).read_text(encoding="utf-8"))
        row["v5_decision"] = row.pop("predicted_decision")
        row["v5_likelihood"] = row.pop("investment_likelihood")
        row["canonical_decision"] = canonical["decision"]
        row["canonical_likelihood"] = canonical["investment_likelihood"]
        summary = json.loads((
            args.run_root / "investors" / str(row["vc_slug"])
            / str(row["episode_slug"]) / "summary.json"
        ).read_text(encoding="utf-8"))
        row["phase2_iterations"] = summary["phase2_iterations"]
    metrics = {
        "canonical": _metrics(rows, "canonical"),
        "v5": _metrics(rows, "v5"),
    }
    canonical_only = sum(
        row["canonical_decision"] == row["actual_decision"]
        and row["v5_decision"] != row["actual_decision"] for row in rows
    )
    v5_only = sum(
        row["v5_decision"] == row["actual_decision"]
        and row["canonical_decision"] != row["actual_decision"] for row in rows
    )
    discordant = canonical_only + v5_only
    significance = {
        "canonical_only_correct": canonical_only,
        "v5_only_correct": v5_only,
        "mcnemar_exact_p": (
            binomtest(min(canonical_only, v5_only), discordant, 0.5).pvalue
            if discordant else 1.0
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema": "v5-phase2-canary-evaluation-v1",
        "case_count": len(rows),
        "api_cost_usd": raw["spent_usd"],
        "metrics": metrics,
        "paired_significance": significance,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table = "\n".join(
        f"| {row['vc_slug']} | {row['episode_slug']} | {row['actual_decision']} | "
        f"{row['canonical_decision']} ({float(row['canonical_likelihood']):.2f}) | "
        f"{row['v5_decision']} ({float(row['v5_likelihood']):.2f}) | "
        f"{float(row['review_priority_score']):.2f} | {row['phase2_iterations']} |"
        for row in rows
    )
    (args.output / "evaluation.md").write_text(
        "# V5 Associated-Rationale Phase 2 Canaries\n\n"
        "This fixed panel contains one observed In and one observed Out for each of six VCs. "
        "V5 uses the canonical frozen Phase 1 plus fold-safe associated-rationale hypotheses.\n\n"
        "| Endpoint | Correct | Balanced accuracy | In precision | In recall | In F1 | AP | ROC AUC |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {name} | {value['correct']}/12 | {value['balanced_accuracy']:.3f} | "
            f"{value['in_precision']:.3f} | {value['in_recall']:.3f} | "
            f"{value['in_f1']:.3f} | {value['average_precision']:.3f} | {value['roc_auc']:.3f} |"
            for name, value in metrics.items()
        )
        + "\n\n"
        f"Paired exact McNemar p={significance['mcnemar_exact_p']:.3f}; "
        f"canonical-only correct={canonical_only}, v5-only correct={v5_only}. "
        "This small stochastic canary is diagnostic, not a causal or publication-grade estimate.\n\n"
        "| VC | Episode | Actual | Canonical decision (likelihood) | V5 decision (likelihood) | Review priority | V5 turns |\n"
        "|---|---|---:|---:|---:|---:|---:|\n"
        + table
        + f"\n\nTotal OpenRouter cost: **${raw['spent_usd']:.4f}**.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
