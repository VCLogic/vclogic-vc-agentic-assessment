from __future__ import annotations

from vc_clone_graph.actual_rationale_cases import RationaleModelCase
from vc_clone_graph.actual_rationale_evaluation import (
    align_rationale_cases,
    assemble_actual_rationale_analysis,
    evaluate_actual_rationale_results,
    rationale_difference_rows,
    write_actual_rationale_analysis,
)
from vc_clone_graph.actual_rationale_models import RationalePrediction
from vc_clone_graph.phase2_calibration_evaluation import Phase2CalibrationRecord


def fixtures():
    cases = []
    records = []
    for vc_slug in ("alpha", "beta"):
        for episode in range(1, 7):
            target = int(episode % 3 == 0)
            cases.append(RationaleModelCase(
                vc_slug=vc_slug, vc_name=vc_slug.title(), episode_slug=f"{episode}-x",
                group=f"{episode}-x", target=target, source_tier="newly_extracted",
                source_format="transcript-observed-rationales-v1",
                actual_features={"signal": float(target)}, predicted_features={"signal": float(target)},
                actual_items=(("a", "positive", "primary"),),
                predicted_items=(("b", "negative", "secondary"),),
            ))
            records.append(Phase2CalibrationRecord(
                vc_slug=vc_slug, vc_name=vc_slug.title(), episode_slug=f"{episode}-x",
                group=f"{episode}-x", target=target, raw_decision=target,
                raw_likelihood=0.8 if target else 0.2, phase2_features={},
                combined_features={}, semantic_features={}, artifact_root="runs/x",
                phase1_sha256="1" * 64, phase2_sha256="2" * 64,
            ))
    learned = {}
    for family in ("per_vc_logistic", "per_vc_tree", "hierarchical_logistic"):
        for condition in ("actual_to_actual", "actual_to_predicted", "predicted_to_predicted"):
            method = f"{family}__{condition}"
            learned[method] = [RationalePrediction(
                vc_slug=case.vc_slug, vc_name=case.vc_name,
                episode_slug=case.episode_slug, held_group=case.group,
                target=case.target, method=method, model_family=family,
                source_condition=condition,
                training_source="actual" if condition.startswith("actual") else "predicted",
                test_source="actual" if condition.endswith("actual") else "predicted",
                score=0.8 if case.target else 0.2,
                balanced_decision=case.target, precision_decision=case.target,
                balanced_threshold=0.5, precision_threshold=0.5,
                selected_params=(("c", 1.0),), training_episode_slugs=(),
                training_groups=(), training_vcs=(), training_indices=(),
            ) for case in cases]
    return records, cases, learned


def test_endpoint_evaluation_contains_all_models_gaps_and_provenance() -> None:
    records, cases, learned = fixtures()
    result = evaluate_actual_rationale_results(
        records, cases, learned, review_budgets=(5,),
        bootstrap_iterations=20, permutation_iterations=20, seed=3,
    )

    classification_methods = {row["method"] for row in result["classification"]}
    ranking_methods = {row["method"] for row in result["ranking"]}
    assert "raw_phase2" in classification_methods
    assert "per_vc_logistic__actual_to_predicted__balanced" in classification_methods
    assert len(ranking_methods) == 10
    assert result["oracle_gaps"]
    assert result["sensitivity"]
    assert len(result["provenance"]["per_vc_folds"]) == 12 * 3 * 2
    assert len(result["provenance"]["hierarchical_folds"]) == 6 * 3
    assert result["significance"]


def test_rationale_differences_expose_missing_added_and_attribute_conflicts() -> None:
    _, cases, _ = fixtures()
    rows = rationale_difference_rows(cases)

    assert len(rows) == len(cases)
    assert rows[0]["actual_only_labels"] == "a"
    assert rows[0]["predicted_only_labels"] == "b"
    assert rows[0]["shared_label_count"] == 0


def test_assembled_analysis_writes_complete_artifact_set(tmp_path) -> None:
    records, cases, learned = fixtures()
    taxonomy = ("a", *(f"label_{index}" for index in range(43)))
    analysis = assemble_actual_rationale_analysis(
        records, cases, learned, taxonomy, review_budgets=(5,),
        bootstrap_iterations=10, permutation_iterations=10, seed=5,
    )
    input_file = tmp_path / "input.json"
    input_file.write_text("{}", encoding="utf-8")
    paths = write_actual_rationale_analysis(
        analysis, tmp_path / "report", input_paths={"registry": input_file}
    )

    assert paths["evaluation"].is_file()
    assert paths["predictions"].is_file()
    assert paths["provenance"].is_file()
    assert paths["benchmark_policy"].is_file()
    assert paths["manifest"].is_file()
    assert "automated candidate" in paths["evaluation"].read_text(encoding="utf-8")


def test_case_alignment_uses_vc_episode_keys_not_input_order() -> None:
    records, cases, _ = fixtures()
    aligned = align_rationale_cases(records, list(reversed(cases)))

    assert [
        (case.vc_slug, case.episode_slug, case.target) for case in aligned
    ] == [
        (record.vc_slug, record.episode_slug, record.target) for record in records
    ]
