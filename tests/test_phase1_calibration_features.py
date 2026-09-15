from dataclasses import replace
from pathlib import Path

import numpy as np

from vc_clone_graph.phase1_calibration_cases import (
    CalibrationCase,
    RawLabelSignal,
)
from vc_clone_graph.phase1_calibration_features import build_fold_features


LABELS = ("a", "b")
FAMILIES = {"a": "one", "b": "one"}


def _case(vc: str, episode: str, target: tuple[int, int], predicted: str) -> CalibrationCase:
    signals = {
        label: RawLabelSignal(
            predicted=label == predicted,
            confidence=0.8 if label == predicted else 0.0,
            direction="positive" if label == predicted else "absent",
            salience="primary" if label == predicted else "absent",
            signed_salience=2.0 if label == predicted else 0.0,
        )
        for label in LABELS
    }
    return CalibrationCase(
        vc_slug=vc,
        vc_name=vc,
        episode_slug=episode,
        actual_decision="In",
        source_tier="newly_extracted",
        source_format="transcript-observed-rationales-v1",
        pitch_text=episode,
        pitch_path=Path(f"{episode}.txt"),
        pitch_sha256=episode,
        labels=LABELS,
        families=FAMILIES,
        raw_signals=signals,
        reference_targets=dict(zip(LABELS, target, strict=True)),
        reference_attributes={},
        phase1_case=None,  # type: ignore[arg-type]
    )


def test_fold_features_mask_phase1_signals_in_semantic_control() -> None:
    cases = [
        _case("vc1", "e1", (1, 0), "a"),
        _case("vc1", "e2", (0, 1), "b"),
        _case("vc2", "e3", (1, 0), "b"),
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])

    with_phase1 = build_fold_features(
        cases, embeddings, [0, 1], [2], include_phase1=True, pca_components=2
    )
    semantic_only = build_fold_features(
        cases, embeddings, [0, 1], [2], include_phase1=False, pca_components=2
    )

    assert any(name.startswith("phase1__") for name in with_phase1.feature_names)
    assert not any(name.startswith("phase1__") for name in semantic_only.feature_names)
    assert with_phase1.train.shape[0] == 2
    assert with_phase1.evaluate.shape[0] == 1
    assert with_phase1.targets.shape == (2, 2)


def test_held_reference_does_not_change_held_features() -> None:
    cases = [
        _case("vc1", "e1", (1, 0), "a"),
        _case("vc1", "e2", (0, 1), "b"),
        _case("vc2", "e3", (1, 0), "b"),
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    original = build_fold_features(cases, embeddings, [0, 1], [2], pca_components=2)
    changed_cases = list(cases)
    changed_cases[2] = replace(
        changed_cases[2], reference_targets={"a": 0, "b": 1}
    )
    changed = build_fold_features(
        changed_cases, embeddings, [0, 1], [2], pca_components=2
    )

    np.testing.assert_allclose(original.evaluate, changed.evaluate)


def test_training_prototypes_exclude_same_episode_group() -> None:
    cases = [
        _case("vc1", "shared", (1, 0), "a"),
        _case("vc2", "shared", (1, 0), "a"),
        _case("vc1", "other", (0, 1), "b"),
        _case("vc1", "held", (0, 0), "a"),
    ]
    embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

    features = build_fold_features(cases, embeddings, [0, 1, 2], [3], pca_components=2)
    column = features.feature_names.index("prototype__a__similarity")

    assert features.train[0, column] == 0.0
    assert features.train[1, column] == 0.0
    assert features.evaluate[0, column] > 0.0
