from __future__ import annotations

from dataclasses import replace

from vc_clone_graph.personalized_cases import PersonalizedCase
from vc_clone_graph.personalized_folds import make_episode_folds
from vc_clone_graph.personalized_rationale_teacher import (
    cross_fit_rationale_teacher,
    teacher_comparison_features,
)


def cases() -> list[PersonalizedCase]:
    rows = []
    for index in range(8):
        positive = index % 2
        rows.append(PersonalizedCase(
            vc_slug="alpha", vc_input_slug="alpha-fund", vc_name="Alpha",
            episode_slug=f"{index}-episode", group=f"{index}-episode",
            target=positive, pitch_text="pitch", phase1_text="p1", phase2_text="p2",
            phase1_features={
                "label__market__signed_confidence": 0.8 if positive else 0.0,
                "label_direction__market__positive__primary__count": float(positive),
            },
            phase2_features={}, wiki_text="wiki",
            actual_rationale_targets={"market": positive, "rare": 0},
            actual_rationale_available=True,
            actual_rationale_source_tier="test",
            actual_rationale_source_format="test",
            source_hashes={"pitch": f"{index:064x}"},
        ))
    return rows


def test_teacher_cross_fits_training_rows_and_excludes_held_episode() -> None:
    population = cases()
    fold = make_episode_folds(population, inner_splits=3, seed=9)[0]
    deployable = [
        {"signal": float(row.target), "vc_identity__alpha": 1.0}
        for row in population
    ]

    train, held = cross_fit_rationale_teacher(
        population, fold, deployable, ("market", "rare"), c_value=1.0, seed=9
    )

    assert len(train) == len(fold.train_indices)
    assert len(held) == len(fold.test_indices)
    assert all(fold.held_episode not in output.training_episodes for output in train + held)
    assert all(0.0 <= output.probabilities["market"] <= 1.0 for output in train + held)
    assert held[0].status_by_label["market"] == "trained"
    assert held[0].status_by_label["rare"] == "untrainable_prevalence"


def test_held_actual_rationales_cannot_change_teacher_prediction() -> None:
    original = cases()
    fold = make_episode_folds(original, inner_splits=3, seed=11)[0]
    deployable = [{"signal": float(row.target)} for row in original]
    first = cross_fit_rationale_teacher(
        original, fold, deployable, ("market",), c_value=1.0, seed=11
    )[1]
    held_index = fold.test_indices[0]
    changed = list(original)
    changed[held_index] = replace(
        changed[held_index],
        target=1 - changed[held_index].target,
        actual_rationale_targets={"market": 1 - changed[held_index].actual_rationale_targets["market"]},
    )
    second = cross_fit_rationale_teacher(
        changed, fold, deployable, ("market",), c_value=1.0, seed=11
    )[1]

    assert first == second


def test_teacher_comparison_marks_missing_and_spurious_hypotheses() -> None:
    population = cases()
    fold = make_episode_folds(population, inner_splits=3, seed=13)[0]
    output = cross_fit_rationale_teacher(
        population,
        fold,
        [{"signal": float(row.target)} for row in population],
        ("market", "rare"),
        c_value=1.0,
        seed=13,
    )[1][0]

    features = teacher_comparison_features(
        output,
        {"label__market__signed_confidence": 0.0},
        ("market", "rare"),
    )

    assert features["teacher__market__missing_rationale"] == output.probabilities["market"]
    assert features["teacher__market__spurious_rationale"] == 0.0
    assert not any("actual" in key or "target" in key for key in features)
