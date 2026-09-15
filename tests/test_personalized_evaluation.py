from __future__ import annotations

import csv
from pathlib import Path

from vc_clone_graph.personalized_evaluation import (
    PredictionRow,
    evaluate_personalized_predictions,
    write_personalized_evaluation,
)


def rows() -> list[PredictionRow]:
    result = []
    for vc in ("alpha", "beta"):
        targets = (1, 0, 1, 0)
        for index, target in enumerate(targets):
            for method, score, predicted in (
                ("candidate", 0.9 if target else 0.1, target),
                ("raw_phase2", 0.1 if target else 0.9, 1 - target),
            ):
                result.append(PredictionRow(
                    method=method,
                    family=method,
                    condition="raw" if method == "raw_phase2" else "all",
                    vc_slug=vc,
                    vc_name=vc.title(),
                    episode_slug=f"{index}-{vc}",
                    target=target,
                    classification_score=score,
                    ranking_score=score,
                    predicted=predicted,
                    status="complete",
                    runtime_seconds=1.0,
                ))
    return result


def test_evaluation_reports_co_primary_metrics_and_paired_comparison() -> None:
    result = evaluate_personalized_predictions(
        rows(), expected_case_count=8, review_budgets=(1, 2, 4),
        bootstrap_samples=100, randomization_samples=100, seed=41,
    )

    candidate = result["classification_macro"]["candidate"]
    ranking = result["ranking_macro"]["candidate"]
    assert candidate["balanced_accuracy"] == 1.0
    assert candidate["in_precision"] == 1.0
    assert candidate["in_recall"] == 1.0
    assert ranking["average_precision"] == 1.0
    assert ranking["precision_at_2"] == 1.0
    assert ranking["recall_at_2"] == 1.0
    assert result["classification_macro"]["raw_phase2"]["balanced_accuracy"] == 0.0
    assert result["coverage"]["candidate"]["complete"] == 8
    assert result["paired_comparisons"]["candidate_vs_raw_phase2"]["classification_delta"] == 1.0


def test_report_writer_emits_csv_json_markdown_and_economics(tmp_path: Path) -> None:
    result = evaluate_personalized_predictions(
        rows(), expected_case_count=8, review_budgets=(1, 2),
        bootstrap_samples=20, randomization_samples=20, seed=43,
    )
    write_personalized_evaluation(
        tmp_path,
        result,
        source_manifest={"registry_sha256": "a" * 64},
        automated_rationale_references=True,
    )

    expected = {
        "population.json", "predictions.csv", "classification_metrics.csv",
        "ranking_metrics.csv", "calibration_metrics.csv", "per_vc_metrics.csv",
        "significance.csv", "transitions.csv", "economics.csv", "report.md",
        "ablations.csv", "error_overlap.csv", "feature_provenance.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    report = (tmp_path / "report.md").read_text()
    assert "automated" in report.lower()
    assert "candidate" in report
    with (tmp_path / "economics.csv").open(newline="") as stream:
        economics = list(csv.DictReader(stream))
    assert all(float(row["api_cost_usd"]) == 0.0 for row in economics)
