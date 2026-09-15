from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from vc_clone_graph.per_vc_rationale_models import (
    aggregate_rationale_label_importance,
    decision_feature_view,
    make_per_vc_leave_one_out_folds,
    nested_per_vc_elastic_predictions,
    nested_per_vc_random_forest_predictions,
    nested_per_vc_rulefit_predictions,
    rationale_feature_view,
    extract_tree_rules,
    run_per_vc_analysis,
    write_per_vc_analysis,
)
from vc_clone_graph.phase2_calibration_evaluation import Phase2CalibrationRecord


def model_records() -> list[Phase2CalibrationRecord]:
    records: list[Phase2CalibrationRecord] = []
    for vc_offset, vc_slug in enumerate(("alpha", "beta")):
        for episode in range(1, 13):
            target = int((episode + vc_offset) % 3 == 0)
            raw = int(episode % 4 == 0)
            phase2 = {
                "investment_likelihood": 0.7 if raw else 0.25,
                "review_priority_score": 0.8 if target else 0.2,
                "decision_confidence": 0.75,
                "signed_decision_confidence": 0.75 if raw else -0.75,
                "decision_in": float(raw),
                f"vc__{vc_slug}": 1.0,
            }
            rationale = {
                "rationale__founder_execution__signed_confidence": 0.9 if target else -0.4,
                "rationale__founder_execution__salience_confidence": 0.9 if target else -0.2,
                "rationale__founder_execution__count": 1.0,
                "rationale__founder_execution__pitch_evidence_count": 1.0,
                "constraint__portfolio_conflict__triggered__blocking": float(episode == 5),
                f"vc_rationale__{vc_slug}__founder_execution__signed_confidence": 0.9,
            }
            records.append(
                Phase2CalibrationRecord(
                    vc_slug=vc_slug,
                    vc_name=vc_slug.title(),
                    episode_slug=f"{episode}-example",
                    group=f"{episode}-example",
                    target=target,
                    raw_decision=raw,
                    raw_likelihood=phase2["investment_likelihood"],
                    phase2_features=phase2,
                    combined_features={**phase2, **rationale},
                    semantic_features={**phase2, **rationale},
                    artifact_root=f"runs/{vc_slug}/{episode}-example",
                    phase1_sha256=str(episode).zfill(64),
                    phase2_sha256=str(episode + 1).zfill(64),
                )
            )
    return records


def test_rationale_feature_view_excludes_phase2_and_vc_identity() -> None:
    record = model_records()[0]

    features = rationale_feature_view(record)

    assert features["rationale__founder_execution__signed_confidence"] != 0
    assert "constraint__portfolio_conflict__triggered__blocking" in features
    assert "investment_likelihood" not in features
    assert "decision_in" not in features
    assert not any(name.startswith("vc_") for name in features)
    assert not any("target" in name or "actual" in name for name in features)


def test_decision_feature_view_adds_phase2_without_constant_vc_indicator() -> None:
    record = model_records()[0]

    features = decision_feature_view(record)

    assert features["investment_likelihood"] == pytest.approx(record.raw_likelihood)
    assert "review_priority_score" in features
    assert "rationale__founder_execution__signed_confidence" in features
    assert not any(name.startswith("vc__") for name in features)


def test_per_vc_leave_one_out_folds_hold_exactly_one_pitch() -> None:
    records = model_records()

    folds = make_per_vc_leave_one_out_folds(records)

    assert len(folds) == len(records)
    held = []
    for vc_slug, train_indices, held_index in folds:
        held.append(held_index)
        assert held_index not in train_indices
        assert all(records[index].vc_slug == vc_slug for index in train_indices)
        assert records[held_index].vc_slug == vc_slug
        assert len(train_indices) == 11
    assert sorted(held) == list(range(len(records)))


@pytest.mark.parametrize("family", ["decision", "rationale"])
def test_nested_per_vc_elastic_predictions_are_deterministic_and_isolated(
    family: str,
) -> None:
    records = model_records()

    first = nested_per_vc_elastic_predictions(
        records,
        family=family,
        inner_splits=2,
        seed=101,
    )
    second = nested_per_vc_elastic_predictions(
        records,
        family=family,
        inner_splits=2,
        seed=101,
    )

    assert first == second
    assert len(first) == len(records)
    for record, prediction in zip(records, first, strict=True):
        assert prediction.vc_slug == record.vc_slug
        assert prediction.episode_slug == record.episode_slug
        assert record.episode_slug not in prediction.training_episode_slugs
        assert len(prediction.training_episode_slugs) == 11
        assert prediction.selected_config.startswith("C=")
        assert 0.0 <= prediction.score <= 1.0
        assert prediction.predicted in {0, 1}
        assert prediction.coefficients


def test_tree_rule_extraction_returns_readable_supported_paths() -> None:
    matrix = np.asarray([
        [-1.0, 0.0], [-0.8, 0.0], [-0.4, 1.0], [0.4, 0.0], [0.8, 1.0], [1.0, 1.0]
    ])
    targets = np.asarray([0, 0, 0, 1, 1, 1])
    forest = RandomForestClassifier(
        n_estimators=3,
        max_depth=2,
        min_samples_leaf=1,
        random_state=17,
    ).fit(matrix, targets)

    rules = extract_tree_rules(
        forest,
        feature_names=("founder_execution", "portfolio_conflict"),
        matrix=matrix,
        minimum_support=2,
    )

    assert rules
    assert all(rule.support >= 2 for rule in rules)
    assert all(
        "founder_execution" in rule.description or "portfolio_conflict" in rule.description
        for rule in rules
    )


def test_random_forest_and_rulefit_predictions_are_per_vc_and_out_of_sample() -> None:
    records = model_records()

    forest = nested_per_vc_random_forest_predictions(
        records,
        inner_splits=2,
        n_estimators=12,
        seed=109,
    )
    rulefit = nested_per_vc_rulefit_predictions(
        records,
        inner_splits=2,
        n_estimators=8,
        seed=109,
    )

    assert len(forest) == len(records)
    assert len(rulefit) == len(records)
    for record, prediction in zip(records, forest, strict=True):
        assert prediction.method == "per_vc_rationale_random_forest"
        assert record.episode_slug not in prediction.training_episode_slugs
        assert 0.0 <= prediction.score <= 1.0
    for record, prediction in zip(records, rulefit, strict=True):
        assert prediction.method == "per_vc_rationale_rulefit"
        assert record.episode_slug not in prediction.training_episode_slugs
        assert 0.0 <= prediction.score <= 1.0
    assert any(
        name.startswith("rule__")
        for prediction in rulefit
        for name, _ in prediction.coefficients
    )


def test_per_vc_analysis_reports_metrics_and_rationale_stability() -> None:
    result = run_per_vc_analysis(
        model_records(),
        inner_splits=2,
        forest_trees=8,
        rule_trees=6,
        bootstrap_iterations=20,
        permutation_iterations=20,
        seed=113,
        review_budgets=(1, 3, 5),
    )

    assert set(result["methods"]) == {
        "raw_phase2",
        "per_vc_decision_elastic_net",
        "per_vc_rationale_elastic_net",
        "per_vc_rationale_random_forest",
        "per_vc_rationale_rulefit",
    }
    assert all(len(rows) == 24 for rows in result["methods"].values())
    assert {row["scope"] for row in result["classification"]} == {
        "investor", "macro", "pooled_diagnostic"
    }
    assert any(
        row["method"] == "per_vc_rationale_elastic_net"
        and row["selection_frequency"] > 0
        for row in result["rationale_stability"]
    )
    assert result["api_cost_usd"] == 0.0


def test_per_vc_writer_emits_auditable_artifacts_and_models(tmp_path) -> None:
    records = model_records()
    result = run_per_vc_analysis(
        records,
        inner_splits=2,
        forest_trees=6,
        rule_trees=4,
        bootstrap_iterations=10,
        permutation_iterations=10,
        seed=127,
        review_budgets=(1, 3, 5),
    )
    registry = tmp_path / "registry.json"
    registry.write_text('{"runs": []}\n', encoding="utf-8")

    paths = write_per_vc_analysis(
        records,
        result,
        tmp_path / "report",
        registry_path=registry,
        importance_repeats=2,
    )

    expected = {
        "evaluation", "predictions", "classification", "ranking", "importance",
        "rationale_labels",
        "rules", "uncertainty", "significance", "provenance", "manifest",
    }
    assert expected <= paths.keys()
    assert all(paths[name].is_file() for name in expected)
    model_paths = [path for name, path in paths.items() if name.startswith("model_")]
    assert len(model_paths) == 8
    assert all(path.is_file() for path in model_paths)

    import json

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert manifest["population"] == {
        "cases": 24, "investors": 2, "ins": 8, "outs": 16,
    }
    assert manifest["api_cost_usd"] == 0.0
    assert len(manifest["model_files"]) == 8
    assert len(provenance["folds"]) == 24
    assert all(
        fold["held_episode_slug"] not in fold["training_episode_slugs"]
        for fold in provenance["folds"]
    )
    report = paths["evaluation"].read_text(encoding="utf-8")
    assert "leave-one-pitch-out" in report.lower()
    assert "pooled Phase 2 evaluation" in report


def test_rationale_importance_collapses_encoded_feature_variants() -> None:
    rows = [
        {
            "vc_slug": "alpha", "vc_name": "Alpha",
            "method": "per_vc_rationale_elastic_net",
            "feature": "rationale__market_timing__signed_confidence",
            "selection_frequency": 1.0, "mean_effect": 0.4,
        },
        {
            "vc_slug": "alpha", "vc_name": "Alpha",
            "method": "per_vc_rationale_elastic_net",
            "feature": "rationale__market_timing__count",
            "selection_frequency": 0.5, "mean_effect": -0.2,
        },
    ]

    output = aggregate_rationale_label_importance(rows)

    assert output == [{
        "vc_slug": "alpha",
        "vc_name": "Alpha",
        "method": "per_vc_rationale_elastic_net",
        "rationale_label": "market_timing",
        "association_direction": "positive",
        "importance_score": pytest.approx(0.4),
        "supporting_feature": "rationale__market_timing__signed_confidence",
        "selection_frequency": 1.0,
        "mean_effect": 0.4,
        "encoded_feature_count": 2,
    }]
