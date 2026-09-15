from __future__ import annotations

from types import SimpleNamespace

import pytest

from vc_clone_graph.rationale_completion import (
    build_completion_cases,
    completion_analysis,
    leave_one_episode_out,
    nested_association_predictions,
    nested_completion_predictions,
    write_completion_analysis,
)


LABELS = ("founder_market_fit", "founder_skillset", "market_size_assessment")


def _source(
    slug: str,
    *,
    predicted: tuple[str, ...] = (),
    actual: tuple[str, ...] = (),
    decision: str = "Out",
):
    signals = {}
    for label in LABELS:
        present = label in predicted
        signals[label] = SimpleNamespace(
            predicted=present,
            confidence=0.8 if present else 0.0,
            direction="positive" if present else "absent",
            salience="primary" if present else "absent",
            signed_salience=2.0 if present else 0.0,
        )
    return SimpleNamespace(
        vc_slug="charles-hudson",
        vc_name="Charles Hudson",
        episode_slug=slug,
        actual_decision=decision,
        labels=LABELS,
        raw_signals=signals,
        reference_targets={label: int(label in actual) for label in LABELS},
    )


def test_completion_case_separates_observed_features_from_reference_targets():
    [case] = build_completion_cases(
        [_source("held", predicted=("founder_market_fit",), actual=("founder_skillset",))]
    )

    assert case.observed_labels == ("founder_market_fit",)
    assert case.reference_targets["founder_skillset"] == 1
    assert case.features["rationale__founder_market_fit__present"] == 1.0
    assert not any("actual" in name or "decision" in name for name in case.features)


def test_completion_features_record_direction_salience_and_confidence():
    [case] = build_completion_cases([_source("held", predicted=("founder_market_fit",))])

    assert case.features["rationale__founder_market_fit__direction__positive"] == 1.0
    assert case.features["rationale__founder_market_fit__salience__primary"] == 1.0
    assert case.features["rationale__founder_market_fit__confidence"] == 0.8
    assert case.features["rationale__founder_market_fit__signed_salience_confidence"] == 1.6
    assert case.features["aggregate__distinct_label_count"] == 1.0


def test_completion_features_count_all_canonical_instances_when_available():
    source = _source("held", predicted=("founder_market_fit",))
    source.phase1_case = SimpleNamespace(
        predicted=(
            SimpleNamespace(label="founder_market_fit", confidence=0.8, direction="positive", salience="primary"),
            SimpleNamespace(label="founder_market_fit", confidence=0.6, direction="neutral", salience="secondary"),
        )
    )
    [case] = build_completion_cases([source])

    assert case.features["rationale__founder_market_fit__count"] == 2.0
    assert case.features["rationale__founder_market_fit__direction__positive"] == 1.0
    assert case.features["rationale__founder_market_fit__direction__neutral"] == 1.0
    assert case.features["aggregate__instance_count"] == 2.0


def test_outer_fold_never_trains_on_held_episode():
    cases = build_completion_cases([_source("a"), _source("b"), _source("c")])
    folds = leave_one_episode_out(cases)

    assert len(folds) == 3
    assert all(fold.held_slug not in fold.training_slugs for fold in folds)
    assert {fold.held_slug for fold in folds} == {"a", "b", "c"}


def test_completion_cases_reject_duplicate_episode_for_same_vc():
    with pytest.raises(ValueError, match="duplicate completion case"):
        build_completion_cases([_source("a"), _source("a")])


def _model_cases():
    sources = [
        _source("a", predicted=("founder_market_fit",), actual=("founder_skillset",)),
        _source("b", predicted=("founder_market_fit",), actual=("founder_skillset",)),
        _source("c", predicted=("founder_market_fit",), actual=()),
        _source("d", predicted=("founder_market_fit",), actual=()),
        _source("e", predicted=("market_size_assessment",), actual=()),
        _source("f", predicted=("market_size_assessment",), actual=()),
        _source("g", predicted=(), actual=()),
        _source("held", predicted=("founder_market_fit",), actual=("founder_skillset",)),
    ]
    return build_completion_cases(sources)


def test_association_rule_uses_only_outer_training_and_minimum_support():
    held = next(
        row for row in nested_completion_predictions(_model_cases())
        if row.episode_slug == "held"
    )
    hypothesis = next(row for row in held.association if row.label == "founder_skillset")

    assert hypothesis.association_antecedent == ("founder_market_fit",)
    assert hypothesis.association_support == 4
    assert "held" not in hypothesis.training_slugs


def test_association_only_predictions_match_full_association_rules_without_logistic():
    full = {
        row.episode_slug: row for row in nested_completion_predictions(_model_cases())
    }
    association_only = {
        row.episode_slug: row for row in nested_association_predictions(_model_cases())
    }

    assert association_only.keys() == full.keys()
    for slug, prediction in association_only.items():
        assert prediction.training_slugs == full[slug].training_slugs
        assert prediction.observed_labels == full[slug].observed_labels
        assert [
            (row.label, row.association_rules) for row in prediction.candidates
        ] == [
            (row.label, row.association_rules) for row in full[slug].candidates
        ]
        assert prediction.logistic == ()
        assert prediction.ensemble == ()


def test_rare_logistic_label_uses_beta_smoothed_training_prevalence():
    held = next(
        row for row in nested_completion_predictions(_model_cases())
        if row.episode_slug == "held"
    )
    hypothesis = next(row for row in held.logistic if row.label == "founder_skillset")

    # Two positives among seven outer-training rows with Beta(1, 1) smoothing.
    assert hypothesis.logistic_probability == pytest.approx(3 / 9)
    assert hypothesis.logistic_fallback is True


def test_ensemble_uses_prespecified_weighting():
    held = next(
        row for row in nested_completion_predictions(_model_cases())
        if row.episode_slug == "held"
    )
    hypothesis = next(row for row in held.ensemble if row.label == "founder_skillset")

    assert hypothesis.completion_probability == pytest.approx(
        0.60 * hypothesis.logistic_probability
        + 0.40 * hypothesis.association_probability
    )


def test_generators_exclude_observed_labels_and_select_at_most_three():
    held = next(
        row for row in nested_completion_predictions(_model_cases(), max_hypotheses=3)
        if row.episode_slug == "held"
    )

    assert all(len(rows) <= 3 for rows in (held.association, held.logistic, held.ensemble))
    assert all(
        "founder_market_fit" not in {item.label for item in rows}
        for rows in (held.association, held.logistic, held.ensemble)
    )
    assert tuple(item.rank for item in held.ensemble) == tuple(
        range(1, len(held.ensemble) + 1)
    )


def test_completion_analysis_reports_top_k_recovery_and_zero_api_cost(tmp_path):
    predictions = nested_completion_predictions(_model_cases())
    analysis = completion_analysis(predictions)
    paths = write_completion_analysis(
        predictions,
        analysis,
        tmp_path,
        input_paths={"registry": tmp_path / "registry.json"},
    )

    assert set(analysis["methods"]) == {"association", "logistic", "ensemble"}
    assert analysis["api_cost_usd"] == 0.0
    assert analysis["methods"]["ensemble"]["hypothesis_precision_at_3"] >= 0.0
    assert paths["manifest"].is_file()
    assert paths["predictions"].is_file()
    assert paths["association_rules"].is_file()
    assert paths["logistic_coefficients"].is_file()
    assert "automated candidate" in paths["evaluation"].read_text(encoding="utf-8")
