import json
from pathlib import Path

import pytest

from vc_clone_graph.phase2_calibration_evaluation import Phase2CalibrationRecord
from vc_clone_graph.rehearsal_classifier import ClassifierRegistry
from vc_clone_graph.rehearsal_classifier_training import (
    _rehearsal_estimator,
    build_classifier_artifacts,
    train_classifier_artifact,
    write_classifier_registry,
)


VC = "charles-hudson-precursor-ventures"


def test_rehearsal_estimator_has_convergence_budget() -> None:
    estimator = _rehearsal_estimator(1.0, 0.5, 42)

    assert estimator.named_steps["logisticregression"].max_iter == 20_000


def _records() -> list[Phase2CalibrationRecord]:
    records: list[Phase2CalibrationRecord] = []
    for index in range(8):
        target = int(index % 2 == 0)
        common = {
            "rationale__founder_execution__signed_confidence": (
                0.8 if target else -0.6
            ),
            f"held_only_{index}": 1.0,
        }
        records.append(
            Phase2CalibrationRecord(
                vc_slug=VC,
                vc_name="Charles Hudson",
                episode_slug=f"{index + 1}-episode-{index}",
                group=f"{index + 1}-episode-{index}",
                target=target,
                raw_decision=target,
                raw_likelihood=0.8 if target else 0.2,
                phase2_features={
                    "investment_likelihood": 0.8 if target else 0.2,
                    f"held_only_{index}": 1.0,
                },
                combined_features=common,
                artifact_root=f"/tmp/{index}",
                phase1_sha256="a" * 64,
                phase2_sha256="b" * 64,
                semantic_features=common,
            )
        )
    return records


def test_trained_json_artifact_reproduces_sklearn_probability() -> None:
    records = _records()
    trained = train_classifier_artifact(
        records,
        vc_slug=VC,
        model_version="test-v1",
        source_registry="canonical.json",
        label_version="audited-v1",
        seed=42,
    )
    features = {
        "rationale__founder_execution__signed_confidence": 0.3,
        "investment_likelihood": 0.65,
    }

    sklearn_probability = float(trained.estimator.predict_proba([features])[0, 1])
    runtime_probability = trained.artifact.score(features).probability_in

    assert runtime_probability == pytest.approx(sklearn_probability, abs=1e-10)
    assert trained.artifact.source_registry == "canonical.json"
    assert trained.artifact.label_version == "audited-v1"
    assert trained.artifact.feature_schema == "per-vc-decision-v1"
    assert trained.artifact.seed == 42
    assert trained.artifact.class_counts == {"Out": 4, "In": 4}


def test_builds_production_and_every_leave_one_episode_out_artifact() -> None:
    records = _records()

    trained = build_classifier_artifacts(
        records,
        vc_slug=VC,
        model_version="test-v1",
        source_registry="canonical.json",
        label_version="audited-v1",
        seed=42,
    )

    assert len(trained) == 9
    production = next(
        row.artifact for row in trained if row.artifact.training_context == "production"
    )
    assert set(production.training_episode_slugs) == {
        row.episode_slug for row in records
    }
    for index, record in enumerate(records):
        held = next(
            row.artifact
            for row in trained
            if row.artifact.excluded_episode_slug == record.episode_slug
        )
        assert record.episode_slug not in held.training_episode_slugs
        assert f"held_only_{index}" not in held.feature_names


def test_fold_vectorizer_does_not_see_held_only_feature() -> None:
    records = _records()
    held_slug = records[0].episode_slug

    trained = train_classifier_artifact(
        records,
        vc_slug=VC,
        model_version="test-v1",
        source_registry="canonical.json",
        label_version="audited-v1",
        seed=42,
        excluded_episode_slug=held_slug,
    )

    assert "held_only_0" not in trained.artifact.feature_names


def test_written_registry_is_hash_verified_and_resolvable(tmp_path: Path) -> None:
    trained = build_classifier_artifacts(
        _records(),
        vc_slug=VC,
        model_version="test-v1",
        source_registry="canonical.json",
        label_version="audited-v1",
        seed=42,
    )

    registry_path = write_classifier_registry(tmp_path, trained)
    registry = ClassifierRegistry.load(registry_path)

    assert registry.resolve(VC).training_context == "production"
    assert registry.resolve(
        VC, excluded_episode_slug="1-episode-0"
    ).excluded_episode_slug == "1-episode-0"
    manifest = json.loads(registry_path.read_text(encoding="utf-8"))
    assert len(manifest["artifacts"]) == 9


def test_training_rejects_fold_without_both_classes() -> None:
    records = _records()
    single_class = [row for row in records if row.target == 1]

    with pytest.raises(ValueError, match="both classes"):
        train_classifier_artifact(
            single_class,
            vc_slug=VC,
            model_version="test-v1",
            source_registry="canonical.json",
            label_version="audited-v1",
            seed=42,
        )
