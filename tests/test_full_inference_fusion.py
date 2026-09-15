from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import vc_clone_graph.full_inference_fusion as fusion
from vc_clone_graph.full_inference_fusion import (
    EARLY_FUSION_METHODS,
    FIXED_FUSION_METHODS,
    FullInferenceRecord,
    FullFusionPrediction,
    build_full_inference_population,
    fixed_score_fusions,
    native_phase2_score_fusions,
    nested_error_correction_predictions,
    nested_full_fusion_predictions,
    write_full_fusion_analysis,
)
from vc_clone_graph.posthoc_semantic_models import SemanticCalibrationRecord
from vc_clone_graph.standalone_pitch_models import StandalonePitchRecord


def _semantic(slug: str = "1-example") -> SemanticCalibrationRecord:
    return SemanticCalibrationRecord(
        episode_slug=slug,
        target=1,
        target_decision="In",
        original_decision="In",
        original_likelihood=0.61,
        review_priority_score=0.73,
        phase1_text="phase one rationale",
        full_text="phase one and phase two reasoning",
        structured_features={"decision_path__standard": 1.0},
        artifact_root=f"runs/{slug}",
        phase1_sha256="a" * 64,
        phase2_sha256="b" * 64,
    )


def _pitch(slug: str = "1-example") -> StandalonePitchRecord:
    return StandalonePitchRecord(
        episode_slug=slug,
        target=1,
        target_decision="In",
        original_decision="In",
        original_likelihood=0.61,
        pitch_text="Founder describes the leakage-free pitch.",
        rationale_features={"positive_primary_confidence_sum": 0.8},
        pitch_sha256="c" * 64,
        phase1_sha256="a" * 64,
    )


def test_builds_aligned_full_inference_population() -> None:
    semantic = _semantic()
    pitch = _pitch()

    rows = build_full_inference_population([semantic], [pitch])

    assert len(rows) == 1
    assert rows[0].episode_slug == semantic.episode_slug
    assert rows[0].pitch_text == pitch.pitch_text
    assert rows[0].phase1_text == semantic.phase1_text
    assert rows[0].full_reasoning == semantic.full_text
    assert rows[0].structured_features == semantic.structured_features
    assert rows[0].original_decision == "In"
    assert rows[0].original_likelihood == pytest.approx(0.61)
    assert rows[0].review_priority_score == pytest.approx(0.73)
    assert rows[0].pitch_sha256 == pitch.pitch_sha256
    assert rows[0].phase1_sha256 == semantic.phase1_sha256
    assert rows[0].phase2_sha256 == semantic.phase2_sha256


@pytest.mark.parametrize(
    "semantic,pitch,match",
    [
        ([_semantic()], [_pitch("2-other")], "population"),
        ([_semantic()], [replace(_pitch(), target=0)], "target"),
        ([_semantic()], [replace(_pitch(), phase1_sha256="d" * 64)], "Phase 1"),
        ([_semantic()], [replace(_pitch(), original_decision="Out")], "decision"),
        ([_semantic()], [replace(_pitch(), original_likelihood=0.4)], "likelihood"),
    ],
)
def test_full_inference_population_fails_closed(
    semantic: list[SemanticCalibrationRecord],
    pitch: list[StandalonePitchRecord],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_full_inference_population(semantic, pitch)


def _full_records(*, flip_first: bool = False) -> list[FullInferenceRecord]:
    rows = []
    for index in range(18):
        latent_positive = index % 3 == 0
        target = int(latent_positive)
        if flip_first and index == 0:
            target = 1 - target
        signal = 1.0 if latent_positive else -1.0
        rows.append(
            FullInferenceRecord(
                episode_slug=f"{index + 1}-example",
                target=target,
                target_decision="In" if target else "Out",
                original_decision="In" if index % 4 == 0 else "Out",
                original_likelihood=0.5 + 0.1 * signal,
                review_priority_score=0.6 + 0.05 * signal,
                pitch_text=(
                    "strong founder traction and market "
                    if latent_positive
                    else "weak economics and fatal market constraint "
                )
                + f"case {index % 2}",
                phase1_text=(
                    "positive primary founder execution "
                    if latent_positive
                    else "negative primary market assessment "
                ),
                full_reasoning=(
                    "agent sees founder exception and compelling upside "
                    if latent_positive
                    else "agent sees controlling downside and no exception "
                )
                + f"case {index % 2}",
                structured_features={
                    "positive_primary_confidence_sum": max(signal, 0.0),
                    "negative_primary_confidence_sum": max(-signal, 0.0),
                    "label__founder_execution__signed_confidence": signal,
                    "decision_path__standard": float(index % 2 == 0),
                },
                pitch_sha256=str(index).zfill(64),
                phase1_sha256=str(index + 1).zfill(64),
                phase2_sha256=str(index + 2).zfill(64),
                artifact_root=f"runs/{index + 1}-example",
            )
        )
    return rows


def _full_embeddings(records: list[FullInferenceRecord]) -> dict[str, np.ndarray]:
    pitch = np.asarray(
        [
            [1.0, 0.0, 0.1] if "strong founder" in row.pitch_text else [0.0, 1.0, 0.1]
            for row in records
        ],
        dtype=float,
    )
    reasoning = np.asarray(
        [
            [0.9, 0.1, 0.2]
            if "compelling upside" in row.full_reasoning
            else [0.1, 0.9, 0.2]
            for row in records
        ],
        dtype=float,
    )
    return {"pitch": pitch, "full_reasoning": reasoning}


def test_nested_full_fusion_excludes_held_episode_and_label() -> None:
    records = _full_records()
    embeddings = _full_embeddings(records)

    original = nested_full_fusion_predictions(records, embeddings)
    flipped = nested_full_fusion_predictions(
        _full_records(flip_first=True), embeddings, methods=EARLY_FUSION_METHODS
    )

    assert set(original) == set(EARLY_FUSION_METHODS)
    for method, rows in original.items():
        assert len(rows) == len(records), method
        assert all(
            row.episode_slug not in row.training_slugs
            and len(row.training_slugs) == len(records) - 1
            for row in rows
        )
        assert flipped[method][0].score == pytest.approx(rows[0].score)
        assert flipped[method][0].threshold == pytest.approx(rows[0].threshold)
        assert flipped[method][0].selected_config == rows[0].selected_config


def test_high_dimensional_nonlinear_grid_avoids_histogram_binning() -> None:
    configs = fusion._nonlinear_configurations(feature_count=1600)

    assert configs
    assert {config["kind"] for config in configs} == {"extra_trees"}


def _component_predictions(
    records: list[FullInferenceRecord], *, flip_targets: bool = False
) -> dict[str, list[FullFusionPrediction]]:
    offsets = {
        "full_embedding": 0.07,
        "structured": 0.04,
        "full_tfidf": -0.02,
        "pitch_embedding": 0.05,
        "pitch_tfidf": -0.01,
    }
    result = {}
    for name, offset in offsets.items():
        rows = []
        for index, record in enumerate(records):
            target = 1 - record.target if flip_targets else record.target
            score = record.original_likelihood + offset
            threshold = 0.5 + (0.01 if index % 2 else 0.0)
            rows.append(
                FullFusionPrediction(
                    method=name,
                    episode_slug=record.episode_slug,
                    target=target,
                    score=score,
                    predicted=int(score >= threshold),
                    threshold=threshold,
                    selected_config={"kind": "synthetic"},
                    training_slugs=tuple(
                        other.episode_slug
                        for other in records
                        if other.episode_slug != record.episode_slug
                    ),
                )
            )
        result[name] = rows
    return result


def test_fixed_score_fusions_are_label_blind() -> None:
    records = _full_records()
    components = _component_predictions(records)

    original = fixed_score_fusions(records, components)
    flipped_records = [
        replace(row, target=1 - row.target, target_decision="Out" if row.target else "In")
        for row in records
    ]
    flipped = fixed_score_fusions(
        flipped_records, _component_predictions(records, flip_targets=True)
    )

    assert set(original) == set(FIXED_FUSION_METHODS)
    for method, rows in original.items():
        assert len(rows) == len(records), method
        assert [row.score for row in flipped[method]] == pytest.approx(
            [row.score for row in rows]
        )
        assert [row.predicted for row in flipped[method]] == [
            row.predicted for row in rows
        ]


def test_native_phase2_score_fusions_are_label_blind() -> None:
    records = _full_records()
    original = native_phase2_score_fusions(records)
    flipped = native_phase2_score_fusions(
        [replace(row, target=1 - row.target) for row in records]
    )

    assert set(original) == {
        "phase2_likelihood_priority_70_30",
        "phase2_likelihood_priority_mean",
        "phase2_likelihood_priority_min",
    }
    for method, rows in original.items():
        assert [row.score for row in flipped[method]] == pytest.approx(
            [row.score for row in rows]
        )
        assert [row.predicted for row in flipped[method]] == [
            row.predicted for row in rows
        ]


def test_fixed_score_fusion_rejects_misaligned_components() -> None:
    records = _full_records()
    components = _component_predictions(records)
    components["full_embedding"][0] = replace(
        components["full_embedding"][0], episode_slug="999-wrong"
    )

    with pytest.raises(ValueError, match="component population"):
        fixed_score_fusions(records, components)


def test_nested_error_correction_excludes_held_episode_and_label() -> None:
    records = _full_records()
    embeddings = _full_embeddings(records)

    original = nested_error_correction_predictions(records, embeddings)
    flipped = nested_error_correction_predictions(
        _full_records(flip_first=True), embeddings
    )

    assert len(original) == len(records)
    assert all(
        row.episode_slug not in row.training_slugs
        and len(row.training_slugs) == len(records) - 1
        for row in original
    )
    assert flipped[0].score == pytest.approx(original[0].score)
    assert flipped[0].threshold == pytest.approx(original[0].threshold)
    assert flipped[0].selected_config == original[0].selected_config


def test_error_correction_preserves_raw_when_heads_cannot_train() -> None:
    records = [
        replace(row, original_decision="Out", original_likelihood=0.35)
        for row in _full_records()
    ]

    predictions = nested_error_correction_predictions(records, _full_embeddings(records))

    assert [row.predicted for row in predictions] == [0] * len(records)
    assert all(row.selected_config["fallback"] == "raw_decision" for row in predictions)


def test_write_full_fusion_analysis_records_provenance(tmp_path: Path) -> None:
    records = _full_records()
    predictions = fixed_score_fusions(records, _component_predictions(records))

    result = write_full_fusion_analysis(
        records,
        predictions,
        tmp_path,
        embedding_metadata={"model": "local-test-model", "revision": "pinned"},
        source_manifest={"labels_sha256": "f" * 64},
    )

    assert result["population"] == {"episodes": 18, "ins": 6, "outs": 12}
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["api_cost_usd"] == 0.0
    assert manifest["prediction_reruns"] == 0
    assert manifest["pipeline_modified"] is False
    assert manifest["pitch_sha256"]["1-example"] == records[0].pitch_sha256
    assert manifest["phase2_sha256"]["1-example"] == records[0].phase2_sha256
    assert json.loads((tmp_path / "embedding-metadata.json").read_text())["revision"] == "pinned"
    assert (tmp_path / "predictions.csv").is_file()
    assert (tmp_path / "rankings.csv").is_file()
    assert (tmp_path / "report.md").is_file()


def test_full_fusion_cli_exposes_required_offline_inputs() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/analyze_full_inference_fusion.py", "--help"],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    for option in (
        "--input-root",
        "--status",
        "--labels",
        "--semantic-predictions",
        "--pitch-predictions",
        "--embedding-model",
        "--embedding-revision",
        "--resume",
    ):
        assert option in completed.stdout
