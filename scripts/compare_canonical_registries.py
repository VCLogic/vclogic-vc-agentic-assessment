#!/usr/bin/env python3
"""Compare predictions and metrics from two canonical evaluation registries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from vc_clone_graph.canonical_snapshot import directory_digest
from vc_clone_graph.evaluation import (
    PredictionRow,
    build_evaluation,
    load_canonical_predictions,
    load_registry,
    rank_predictions,
)


TOP_KS = (1, 3, 5, 10, 20)
CHANGED_FIELDS = (
    "vc_slug",
    "vc_name",
    "episode_slug",
    "actual_decision",
    "old_decision",
    "new_decision",
    "old_investment_likelihood",
    "new_investment_likelihood",
    "old_decision_confidence",
    "new_decision_confidence",
    "old_review_priority_score",
    "new_review_priority_score",
    "old_rank",
    "new_rank",
    "old_artifact_path",
    "new_artifact_path",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-registry", type=Path, required=True)
    parser.add_argument("--new-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-changed", type=int)
    return parser


def _key(row: PredictionRow) -> tuple[str, str]:
    return row.vc_slug, row.episode_slug


def _rank_map(rows: list[PredictionRow]) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    for vc_slug in sorted({row.vc_slug for row in rows}):
        vc_rows = [row for row in rows if row.vc_slug == vc_slug]
        result.update({_key(row): rank for rank, row in rank_predictions(vc_rows)})
    return result


def _changed_row(
    old: PredictionRow,
    new: PredictionRow,
    old_rank: int,
    new_rank: int,
) -> dict[str, object]:
    return {
        "vc_slug": old.vc_slug,
        "vc_name": old.vc_name,
        "episode_slug": old.episode_slug,
        "actual_decision": old.actual_decision,
        "old_decision": old.predicted_decision,
        "new_decision": new.predicted_decision,
        "old_investment_likelihood": old.investment_likelihood,
        "new_investment_likelihood": new.investment_likelihood,
        "old_decision_confidence": old.decision_confidence,
        "new_decision_confidence": new.decision_confidence,
        "old_review_priority_score": old.review_priority_score,
        "new_review_priority_score": new.review_priority_score,
        "old_rank": old_rank,
        "new_rank": new_rank,
        "old_artifact_path": str(old.artifact_path.resolve()),
        "new_artifact_path": str(new.artifact_path.resolve()),
    }


def _investors(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["vc_slug"]: item for item in result["investors"]}


def _budget(item: dict[str, Any], requested_k: int) -> dict[str, Any]:
    return next(
        row for row in item["ranking"]["budgets"] if row["requested_k"] == requested_k
    )


def _delta(new: object, old: object) -> str:
    if not isinstance(new, int | float) or not isinstance(old, int | float):
        return ""
    return f"{float(new) - float(old):+.3f}"


def _render_report(
    old_result: dict[str, Any],
    new_result: dict[str, Any],
    *,
    total: int,
    changed: int,
) -> str:
    old_investors = _investors(old_result)
    new_investors = _investors(new_result)
    lines = [
        "# Canonical Registry Comparison",
        "",
        f"Changed artifacts: {changed} / {total}",
        "",
        "Only changed rows are differences; labels and population are identical.",
        "",
        "## Classification and ranking deltas",
        "",
        "| Investor | ΔTP | ΔFP | ΔTN | ΔFN | ΔBalanced accuracy | ΔIn precision | ΔIn recall | ΔIn F1 | ΔAP | ΔROC AUC | ΔRecall@5 | ΔRecall@10 | ΔRecall@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for vc_slug in sorted(old_investors):
        old = old_investors[vc_slug]
        new = new_investors[vc_slug]
        old_c, new_c = old["classification"], new["classification"]
        old_r, new_r = old["ranking"], new["ranking"]
        lines.append(
            f"| {old['vc_name']} | "
            f"{int(new_c['tp']) - int(old_c['tp']):+d} | "
            f"{int(new_c['fp']) - int(old_c['fp']):+d} | "
            f"{int(new_c['tn']) - int(old_c['tn']):+d} | "
            f"{int(new_c['fn']) - int(old_c['fn']):+d} | "
            f"{_delta(new_c['balanced_accuracy'], old_c['balanced_accuracy'])} | "
            f"{_delta(new_c['in_precision'], old_c['in_precision'])} | "
            f"{_delta(new_c['in_recall'], old_c['in_recall'])} | "
            f"{_delta(new_c['in_f1'], old_c['in_f1'])} | "
            f"{_delta(new_r['average_precision'], old_r['average_precision'])} | "
            f"{_delta(new_r['roc_auc'], old_r['roc_auc'])} | "
            f"{_delta(_budget(new, 5)['recall'], _budget(old, 5)['recall'])} | "
            f"{_delta(_budget(new, 10)['recall'], _budget(old, 10)['recall'])} | "
            f"{_delta(_budget(new, 20)['recall'], _budget(old, 20)['recall'])} |"
        )
    lines.extend(
        [
            "",
            "Deltas are new minus old. Ranking is computed independently within each investor.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    old_rows = load_canonical_predictions(load_registry(args.old_registry))
    new_rows = load_canonical_predictions(load_registry(args.new_registry))
    old = {_key(row): row for row in old_rows}
    new = {_key(row): row for row in new_rows}
    if set(old) != set(new):
        raise ValueError("population mismatch between registries")
    for key in sorted(old):
        if old[key].actual_decision != new[key].actual_decision:
            raise ValueError(f"actual-label mismatch: {key}")

    changed_keys = [
        key
        for key in sorted(old)
        if directory_digest(old[key].artifact_path.parent).sha256
        != directory_digest(new[key].artifact_path.parent).sha256
    ]
    if args.expected_changed is not None and len(changed_keys) != args.expected_changed:
        raise ValueError(
            f"expected {args.expected_changed} changed artifacts, "
            f"found {len(changed_keys)}"
        )

    old_ranks = _rank_map(old_rows)
    new_ranks = _rank_map(new_rows)
    changed_rows = [
        _changed_row(old[key], new[key], old_ranks[key], new_ranks[key])
        for key in changed_keys
    ]
    old_result = build_evaluation(old_rows, top_ks=TOP_KS)
    new_result = build_evaluation(new_rows, top_ks=TOP_KS)
    report = _render_report(
        old_result, new_result, total=len(old_rows), changed=len(changed_rows)
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "changed_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CHANGED_FIELDS)
        writer.writeheader()
        writer.writerows(changed_rows)
    (args.output_dir / "comparison.md").write_text(report, encoding="utf-8")
    print(f"Changed artifacts: {len(changed_rows)} / {len(old_rows)}")
    print(f"Outputs: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
