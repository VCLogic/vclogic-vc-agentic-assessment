from dataclasses import replace
from pathlib import Path

import numpy as np

from vc_clone_graph.phase1_calibration_cases import CalibrationCase, RawLabelSignal
from vc_clone_graph.phase1_calibration_models import (
    apply_condition,
    run_nested_calibration,
)


LABELS = ("a", "b")
FAMILIES = {"a": "one", "b": "one"}


def _case(index: int) -> CalibrationCase:
    label = LABELS[index % 2]
    signals = {
        item: RawLabelSignal(
            predicted=item == label,
            confidence=0.8 if item == label else 0.0,
            direction="positive" if item == label else "absent",
            salience="primary" if item == label else "absent",
            signed_salience=2.0 if item == label else 0.0,
        )
        for item in LABELS
    }
    return CalibrationCase(
        vc_slug=f"vc{index % 2}",
        vc_name=f"VC {index % 2}",
        episode_slug=f"episode-{index}",
        actual_decision="In" if index % 2 == 0 else "Out",
        source_tier="newly_extracted",
        source_format="transcript-observed-rationales-v1",
        pitch_text=label,
        pitch_path=Path(f"{index}.txt"),
        pitch_sha256=str(index),
        labels=LABELS,
        families=FAMILIES,
        raw_signals=signals,
        reference_targets={item: int(item == label) for item in LABELS},
        reference_attributes={},
        phase1_case=None,  # type: ignore[arg-type]
    )


def test_apply_condition_masks_filter_but_allows_expansion() -> None:
    scores = np.asarray([[0.8, 0.9]])
    raw = np.asarray([[1, 0]], dtype=bool)

    filtered = apply_condition(scores, raw, threshold=0.5, condition="filtering_only")
    expanded = apply_condition(scores, raw, threshold=0.5, condition="selector_expansion")

    assert filtered.tolist() == [[True, False]]
    assert expanded.tolist() == [[True, True]]


def test_nested_calibration_holds_episode_out_and_is_deterministic() -> None:
    cases = [_case(index) for index in range(8)]
    embeddings = np.asarray(
        [[1.0, 0.0] if index % 2 == 0 else [0.0, 1.0] for index in range(8)]
    )
    kwargs = dict(
        jobs=1,
        c_grid=(0.1,),
        thresholds=(0.4, 0.5),
        inner_splits=2,
        pca_components=2,
    )

    first = run_nested_calibration(cases, embeddings, **kwargs)
    second = run_nested_calibration(cases, embeddings, **kwargs)

    assert first.predictions == second.predictions
    assert len(first.predictions) == len(cases) * len(LABELS) * 3
    assert {row.condition for row in first.predictions} == {
        "filtering_only",
        "selector_expansion",
        "semantic_control",
    }
    assert all(
        fold.held_episode_slug not in fold.training_episode_slugs
        for fold in first.provenance
    )
    assert len(first.provenance) == len(cases) * 3


def test_semantic_control_does_not_change_when_raw_signal_changes() -> None:
    cases = [_case(index) for index in range(8)]
    changed = list(cases)
    changed[0] = replace(
        changed[0],
        raw_signals={
            "a": RawLabelSignal(False, 0.0, "absent", "absent", 0.0),
            "b": RawLabelSignal(True, 1.0, "negative", "secondary", -1.0),
        },
    )
    embeddings = np.asarray(
        [[1.0, 0.0] if index % 2 == 0 else [0.0, 1.0] for index in range(8)]
    )
    kwargs = dict(
        jobs=1,
        c_grid=(0.1,),
        thresholds=(0.5,),
        inner_splits=2,
        pca_components=2,
    )

    original = run_nested_calibration(cases, embeddings, **kwargs)
    revised = run_nested_calibration(changed, embeddings, **kwargs)
    original_rows = [row for row in original.predictions if row.condition == "semantic_control"]
    revised_rows = [row for row in revised.predictions if row.condition == "semantic_control"]

    assert [(row.score, row.predicted) for row in original_rows] == [
        (row.score, row.predicted) for row in revised_rows
    ]
