from __future__ import annotations

import csv
import json
from pathlib import Path

from vc_clone_graph.phase1_evaluation import (
    Phase1Case,
    PredictedRationale,
    ReferenceRationale,
    TaxonomyLabel,
    evaluate_cases,
    write_evaluation_outputs,
)


def test_write_outputs_creates_complete_machine_readable_bundle(tmp_path: Path):
    taxonomy = {
        "label_a": TaxonomyLabel("label_a", "A", "family"),
        "label_b": TaxonomyLabel("label_b", "B", "family"),
    }
    case = Phase1Case(
        "vc", "VC", "1-one", "In", "newly_extracted", "transcript-observed-rationales-v1",
        (PredictedRationale("label_a", "positive", "primary", 0.8, "Reason"),),
        (ReferenceRationale("label_a", "positive", "primary", 0.9, "evaluated", "assessment", "explicit", ("Evidence",)),),
        Path("outputs/one/phase1/investigation.json"), Path("references/one.json"),
    )
    result = evaluate_cases([case], taxonomy, top_ks=(1, 2))
    result["semantic_rows"] = []
    result["confusion_rows"] = []
    result["semantic_summary"] = {"matched_pairs": 0, "embedding": {"model": "disabled"}}
    paths = write_evaluation_outputs(
        result, [case], taxonomy, tmp_path,
        input_manifest={"registry_sha256": "abc", "reference_manifest_sha256": "def"},
    )
    expected = {
        "evaluation.md", "overall_metrics.csv", "vc_metrics.csv",
        "decision_split_metrics.csv", "source_tier_metrics.csv", "ranking_metrics.csv", "attribute_metrics.csv",
            "rich_subset_metrics.csv", "per_label_metrics.csv", "episode_metrics.csv",
            "family_metrics.csv",
            "salience_strata_metrics.csv", "weighted_recovery_metrics.csv",
            "confidence_threshold_metrics.csv",
            "episode_label_comparisons.csv", "confusion_pairs.csv", "semantic_matches.csv",
        "manifest.json",
    }
    assert {path.name for path in paths.values()} == expected
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["case_count"] == 1
    assert manifest["top_ks"] == [1, 2]
    with (tmp_path / "episode_metrics.csv").open() as handle:
        assert list(csv.DictReader(handle))[0]["episode_slug"] == "1-one"
    assert "automated candidate ground truth" in (tmp_path / "evaluation.md").read_text().lower()
