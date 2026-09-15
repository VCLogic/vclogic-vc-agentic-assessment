from __future__ import annotations

import json

from vc_clone_graph.phase1_evaluation import (
    Phase1Case,
    PredictedRationale,
    ReferenceRationale,
    TaxonomyLabel,
    calibration_metrics,
    evaluate_cases,
    optimal_semantic_matches,
    score_set,
    phase1_artifact_filename,
    weighted_recovery_metrics,
    write_evaluation_outputs,
)


TAXONOMY = {
    "market_size_assessment": TaxonomyLabel(
        "market_size_assessment", "Market size", "market_opportunity"
    ),
    "market_entry_path_assessment": TaxonomyLabel(
        "market_entry_path_assessment", "Entry path", "market_opportunity"
    ),
    "founder_skillset": TaxonomyLabel(
        "founder_skillset", "Founder skill", "founder_team"
    ),
    "competitive_intensity_concern": TaxonomyLabel(
        "competitive_intensity_concern", "Competition", "competition_defensibility"
    ),
}


def predicted(label: str, confidence: float, direction: str = "positive", salience: str = "primary"):
    return PredictedRationale(label, direction, salience, confidence, f"Why {label}")


def reference(
    label: str,
    direction: str = "positive",
    salience: str = "primary",
    confidence: float = 0.95,
    activation: str = "evaluated",
    utterance_type: str = "assessment",
    decision_link: str = "none",
):
    return ReferenceRationale(
        label,
        direction,
        salience,
        confidence,
        activation,
        utterance_type,
        decision_link,
        (f"Evidence for {label}",),
    )


def case(predictions, references, *, vc="vc-a", decision="In", rich=True):
    return Phase1Case(
        vc_slug=vc,
        vc_name=vc,
        episode_slug=f"episode-{vc}-{decision}",
        actual_decision=decision,
        source_tier="newly_extracted" if rich else "legacy_fallback",
        source_format=(
            "transcript-observed-rationales-v1" if rich else "legacy-reference-rationales"
        ),
        predicted=tuple(predictions),
        reference=tuple(references),
        artifact_path=None,
        reference_path=None,
    )


def test_score_set_empty_and_partial_cases():
    assert score_set(set(), set()) == {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "jaccard": 1.0,
    }
    result = score_set({"a", "b"}, {"b", "c"})
    assert result["tp"] == result["fp"] == result["fn"] == 1
    assert result["f1"] == 0.5
    assert result["jaccard"] == 1 / 3


def test_phase1_artifact_filename_selects_final_or_candidate() -> None:
    assert phase1_artifact_filename("final") == "investigation.json"
    assert phase1_artifact_filename("candidate") == "candidate-investigation.json"
    try:
        phase1_artifact_filename("unknown")
    except ValueError as exc:
        assert "artifact" in str(exc)
    else:
        raise AssertionError("unknown artifact selector was accepted")


def test_evaluation_collapses_duplicate_labels_and_scores_attributes():
    item = case(
        [predicted("market_size_assessment", 0.6), predicted("market_size_assessment", 0.9, "negative")],
        [reference("market_size_assessment", "negative"), reference("founder_skillset")],
    )
    result = evaluate_cases([item], TAXONOMY, top_ks=(1, 3))
    overall = result["overall"]
    assert overall["micro_tp"] == 1
    assert overall["micro_fp"] == 0
    assert overall["micro_fn"] == 1
    assert overall["average_predicted_size"] == 1.0
    assert result["attribute_metrics"]["direction"]["micro_tp"] == 1
    assert result["episode_rows"][0]["top_labels"].split(";")[0] == "market_size_assessment"


def test_ranking_uses_full_taxonomy_and_primary_recall():
    item = case(
        [predicted("founder_skillset", 0.7), predicted("market_size_assessment", 0.7)],
        [reference("market_size_assessment")],
    )
    result = evaluate_cases([item], TAXONOMY, top_ks=(1, 2, 4))
    rows = {row["k"]: row for row in result["ranking_rows"] if row["scope"] == "overall"}
    # Alphabetical tie-break puts founder_skillset before market_size_assessment.
    assert rows[1]["recall_at_k"] == 0.0
    assert rows[2]["recall_at_k"] == 1.0
    assert rows[2]["primary_recall_at_k"] == 1.0
    assert result["ranking_summary"]["candidate_pairs"] == 4


def test_calibration_fixed_bins():
    result = calibration_metrics([1, 0], [0.8, 0.2], bins=10)
    assert abs(result["brier"] - 0.04) < 1e-12
    assert abs(result["ece"] - 0.2) < 1e-12


def test_weighted_recovery_uses_minimum_matching_mass():
    item = case(
        [
            predicted("market_size_assessment", 0.8, salience="primary"),
            predicted("founder_skillset", 0.6, salience="secondary"),
        ],
        [
            reference(
                "market_size_assessment", salience="secondary", confidence=0.7
            ),
            reference(
                "competitive_intensity_concern",
                salience="primary",
                confidence=0.9,
            ),
        ],
    )

    salience = weighted_recovery_metrics([item], mode="salience")
    assert salience["matched_mass"] == 1.0
    assert salience["predicted_mass"] == 3.0
    assert salience["reference_mass"] == 3.0
    assert salience["precision"] == salience["recall"] == 1 / 3
    assert salience["f1"] == 1 / 3

    confidence = weighted_recovery_metrics([item], mode="confidence")
    assert abs(confidence["matched_mass"] - 0.7) < 1e-12
    assert abs(confidence["predicted_mass"] - 1.4) < 1e-12
    assert abs(confidence["reference_mass"] - 1.6) < 1e-12
    assert abs(confidence["precision"] - 0.5) < 1e-12
    assert abs(confidence["recall"] - 0.4375) < 1e-12
    assert abs(confidence["f1"] - (7 / 15)) < 1e-12

    combined = weighted_recovery_metrics([item], mode="combined")
    assert abs(combined["matched_mass"] - 0.7) < 1e-12
    assert abs(combined["predicted_mass"] - 2.2) < 1e-12
    assert abs(combined["reference_mass"] - 2.5) < 1e-12


def test_weighted_recovery_has_explicit_empty_mass_rules():
    both_empty = weighted_recovery_metrics([case([], [])], mode="confidence")
    assert both_empty["precision"] == 1.0
    assert both_empty["recall"] == 1.0
    assert both_empty["f1"] == 1.0

    no_predictions = weighted_recovery_metrics(
        [case([], [reference("market_size_assessment")])], mode="confidence"
    )
    assert no_predictions["precision"] == 0.0
    assert no_predictions["recall"] == 0.0

    no_references = weighted_recovery_metrics(
        [case([predicted("market_size_assessment", 0.8)], [])], mode="confidence"
    )
    assert no_references["precision"] == 0.0
    assert no_references["recall"] == 1.0
    assert no_references["f1"] == 0.0


def test_evaluation_stratifies_salience_overall_and_by_vc():
    first = case(
        [
            predicted("market_size_assessment", 0.8, salience="primary"),
            predicted("founder_skillset", 0.6, salience="secondary"),
        ],
        [
            reference("market_size_assessment", salience="primary"),
            reference("competitive_intensity_concern", salience="secondary"),
        ],
        vc="vc-a",
    )
    second = case(
        [predicted("market_size_assessment", 0.5, salience="primary")],
        [reference("market_size_assessment", salience="secondary")],
        vc="vc-b",
    )

    result = evaluate_cases([first, second], TAXONOMY)
    rows = {
        (row["scope"], row["salience"]): row
        for row in result["salience_strata_rows"]
    }
    primary = rows[("overall", "primary")]
    assert primary["reference_count"] == 1
    assert primary["recovered_count"] == 1
    assert primary["recall"] == 1.0
    assert primary["predicted_count"] == 2
    assert primary["correct_count"] == 2
    assert primary["precision"] == 1.0
    assert primary["mean_recovered_predicted_confidence"] == 0.8

    secondary = rows[("overall", "secondary")]
    assert secondary["reference_count"] == 2
    assert secondary["recovered_count"] == 1
    assert secondary["recall"] == 0.5
    assert secondary["predicted_count"] == 1
    assert secondary["correct_count"] == 0
    assert secondary["precision"] == 0.0
    assert {row["scope"] for row in result["salience_strata_rows"]} == {
        "overall",
        "vc-a",
        "vc-b",
    }


def test_confidence_thresholds_are_inclusive_and_monotonic():
    item = case(
        [
            predicted("market_size_assessment", 0.50),
            predicted("founder_skillset", 0.60),
            predicted("competitive_intensity_concern", 0.90),
        ],
        [
            reference("market_size_assessment"),
            reference("competitive_intensity_concern"),
        ],
    )
    result = evaluate_cases([item], TAXONOMY)
    rows = {
        row["threshold"]: row
        for row in result["confidence_threshold_rows"]
        if row["scope"] == "overall"
    }
    assert rows[0.5]["micro_tp"] == 2
    assert rows[0.5]["micro_fp"] == 1
    assert rows[0.5]["micro_recall"] == 1.0
    assert rows[0.5]["average_retained_count"] == 3.0
    assert rows[0.6]["micro_tp"] == 1
    assert rows[0.6]["micro_fp"] == 1
    assert rows[0.6]["micro_recall"] == 0.5
    assert rows[0.9]["micro_tp"] == 1
    assert rows[0.9]["micro_fp"] == 0
    assert rows[0.9]["micro_precision"] == 1.0
    retained = [rows[value]["average_retained_count"] for value in sorted(rows)]
    assert retained == sorted(retained, reverse=True)
    assert {row["scope"] for row in result["confidence_threshold_rows"]} == {
        "overall",
        "vc-a",
    }


def test_evaluation_emits_three_weighted_modes_per_scope():
    items = [
        case(
            [predicted("market_size_assessment", 0.8)],
            [reference("market_size_assessment")],
            vc="vc-a",
        ),
        case([], [reference("founder_skillset")], vc="vc-b"),
    ]
    result = evaluate_cases(items, TAXONOMY)
    assert {
        (row["scope"], row["mode"])
        for row in result["weighted_recovery_rows"]
    } == {
        (scope, mode)
        for scope in ("overall", "vc-a", "vc-b")
        for mode in ("salience", "confidence", "combined")
    }


def test_writer_emits_weighted_csv_markdown_and_manifest_contract(tmp_path):
    item = case(
        [predicted("market_size_assessment", 0.8)],
        [reference("market_size_assessment")],
    )
    result = evaluate_cases([item], TAXONOMY)
    paths = write_evaluation_outputs(
        result,
        [item],
        TAXONOMY,
        tmp_path,
        input_manifest={"registry": "canonical.json"},
    )

    for filename in (
        "salience_strata_metrics.csv",
        "weighted_recovery_metrics.csv",
        "confidence_threshold_metrics.csv",
    ):
        assert filename in paths
        assert paths[filename].read_text(encoding="utf-8").count("\n") > 1
    markdown = paths["evaluation.md"].read_text(encoding="utf-8")
    assert "Salience- and confidence-aware recovery" in markdown
    assert "diagnostic composite" in markdown
    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    importance = manifest["importance_evaluation"]
    assert importance["salience_weights"] == {"primary": 2.0, "secondary": 1.0}
    assert importance["confidence_thresholds"] == [0.5, 0.6, 0.7, 0.8, 0.9]
    assert importance["matching_rule"] == "minimum_mass_on_exact_label_match"
    assert "overall_metrics.csv" in manifest["outputs"]


class FakeEmbedder:
    metadata = {"model": "fake", "revision": "one"}

    def embed_documents(self, texts):
        mapping = {"p1": [1.0, 0.0], "p2": [0.0, 1.0], "r1": [0.0, 1.0], "r2": [1.0, 0.0]}
        return [mapping[text] for text in texts]


def test_semantic_matching_is_optimal_one_to_one():
    matches = optimal_semantic_matches(["p1", "p2"], ["r1", "r2"], FakeEmbedder())
    assert {(row["predicted_index"], row["reference_index"]) for row in matches} == {(0, 1), (1, 0)}
    assert all(row["similarity"] == 1.0 for row in matches)


def test_rich_subset_uses_reference_fields_only():
    rich = case(
        [predicted("market_size_assessment", 0.8)],
        [
            reference("market_size_assessment", activation="queried", utterance_type="question"),
            reference("founder_skillset", decision_link="explicit"),
        ],
    )
    legacy = case([], [reference("market_size_assessment")], vc="vc-b", rich=False)
    result = evaluate_cases([rich, legacy], TAXONOMY)
    rich_rows = {row["subset"]: row for row in result["rich_subset_rows"]}
    assert result["rich_case_count"] == 1
    assert rich_rows["activation:queried"]["recall"] == 1.0
    assert rich_rows["decision_link:explicit"]["recall"] == 0.0
