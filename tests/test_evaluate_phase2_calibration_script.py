from __future__ import annotations

import json
from pathlib import Path

from vc_clone_graph.phase2_calibration_evaluation import (
    Phase2CalibrationRecord,
    run_phase2_analysis,
    write_phase2_analysis,
)


def _records() -> list[Phase2CalibrationRecord]:
    records: list[Phase2CalibrationRecord] = []
    for episode in range(1, 25):
        for vc_index, vc_slug in enumerate(("alpha", "beta")):
            target = int((episode + vc_index) % 4 == 0)
            likelihood = 0.65 if (episode + vc_index) % 3 == 0 else 0.35
            raw = int(likelihood >= 0.5)
            phase2 = {
                "investment_likelihood": likelihood,
                "decision_in": float(raw),
                "decision_confidence": 0.7,
                "signed_decision_confidence": 0.7 if raw else -0.7,
                "review_priority_score": 0.8 if target else 0.2,
                f"vc__{vc_slug}": 1.0,
            }
            semantic = {
                **phase2,
                "rationale__founder_execution__signed_confidence": (
                    0.9 if target else -0.5
                ),
                f"vc_rationale__{vc_slug}__founder_execution__signed_confidence": (
                    0.9 if target else -0.5
                ),
            }
            records.append(
                Phase2CalibrationRecord(
                    vc_slug=vc_slug,
                    vc_name=vc_slug.title(),
                    episode_slug=f"{episode}-example",
                    group=f"{episode}-example",
                    target=target,
                    raw_decision=raw,
                    raw_likelihood=likelihood,
                    phase2_features=phase2,
                    combined_features={
                        **phase2,
                        "positive_primary_count": float(target),
                        "negative_primary_count": float(1 - target),
                    },
                    artifact_root=f"runs/{vc_slug}/{episode}-example",
                    phase1_sha256=str(episode).zfill(64),
                    phase2_sha256=str(episode + 1).zfill(64),
                    semantic_features=semantic,
                )
            )
    return records


def test_write_phase2_analysis_emits_reproducible_report_and_models(
    tmp_path: Path,
) -> None:
    records = _records()
    result = run_phase2_analysis(
        records,
        outer_repeats=1,
        outer_splits=3,
        inner_splits=2,
        bootstrap_iterations=20,
        permutation_iterations=20,
        seed=71,
        review_budgets=(1, 3, 5),
    )
    output = tmp_path / "report"
    paths = write_phase2_analysis(
        records,
        result,
        output,
        registry_path=tmp_path / "registry.json",
    )

    expected = {
        "evaluation.md",
        "predictions.csv",
        "classification_metrics.csv",
        "ranking_metrics.csv",
        "transitions.csv",
        "uncertainty.csv",
        "significance.csv",
        "fold_provenance.json",
        "manifest.json",
        "models/phase2/model.joblib",
        "models/phase2/manifest.json",
        "models/combined/model.joblib",
        "models/combined/manifest.json",
        "models/semantic/model.joblib",
        "models/semantic/manifest.json",
    }
    assert {str(path.relative_to(output)) for path in paths.values()} == expected
    assert all(path.is_file() for path in paths.values())
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["population"] == {"cases": 48, "investors": 2, "ins": 12, "outs": 36, "episode_groups": 24}
    assert manifest["scientific_status"] == "nested_grouped_development_evaluation"
    report = (output / "evaluation.md").read_text(encoding="utf-8")
    assert "not an untouched holdout" in report
    assert "Full-data fitted models are deployment artifacts, not performance estimates" in report
    assert "## Decision transitions relative to raw Phase 2" in report
    assert "Corrected false Ins" in report
    assert "Introduced false Ins" in report
    assert len(result["methods"]) == 8
    assert len(result["ranking_methods"]) == 10
    assert len(result["significance"]) == 20
    assert "Taxonomy-aware hierarchical classifier" in report
    assert "Within-VC pairwise ranker" in report
    assert "Training-selected top-K reranker" in report
    assert "Fixed raw top-20 / pairwise reranker" in report
    assert "Fixed raw top-20 / rationale reranker" in report
    assert "VCs improved" in report
