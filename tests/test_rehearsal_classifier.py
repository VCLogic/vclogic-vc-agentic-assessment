import hashlib
import json
import math
from pathlib import Path

import pytest

from vc_clone_graph.rehearsal_classifier import (
    ClassifierRegistry,
    LinearClassifierArtifact,
)


VC = "charles-hudson-precursor-ventures"


def _artifact(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "rehearsal-linear-classifier-v1",
        "artifact_id": "charles-production-v1",
        "model_version": "v1",
        "vc_slug": VC,
        "training_context": "production",
        "excluded_episode_slug": None,
        "feature_names": ["rationale__founder_execution"],
        "scales": [2.0],
        "coefficients": [1.5],
        "intercept": -0.4,
        "decision_threshold": 0.5,
        "training_episode_slugs": ["18-rowvigor", "20-harper-wilde"],
    }
    payload.update(overrides)
    return payload


def _write_registry(tmp_path: Path, artifacts: list[dict[str, object]]) -> Path:
    entries: list[dict[str, object]] = []
    for index, payload in enumerate(artifacts):
        path = tmp_path / f"artifact-{index}.json"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        entries.append(
            {
                "artifact_id": payload["artifact_id"],
                "vc_slug": payload["vc_slug"],
                "training_context": payload["training_context"],
                "excluded_episode_slug": payload["excluded_episode_slug"],
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {"schema": "rehearsal-classifier-registry-v1", "artifacts": entries}
        ),
        encoding="utf-8",
    )
    return registry


def test_linear_artifact_scores_sparse_features_like_exported_pipeline() -> None:
    artifact = LinearClassifierArtifact.model_validate(_artifact())

    result = artifact.score(
        {
            "rationale__founder_execution": 4.0,
            "unrecognized_runtime_feature": 100.0,
        }
    )

    expected = 1 / (1 + math.exp(-(-0.4 + (4.0 / 2.0) * 1.5)))
    assert result.probability_in == pytest.approx(expected)
    assert result.predicted_decision == "In"
    assert result.artifact_id == "charles-production-v1"


def test_absent_features_are_scored_as_zero() -> None:
    artifact = LinearClassifierArtifact.model_validate(_artifact())

    result = artifact.score({})

    assert result.probability_in == pytest.approx(1 / (1 + math.exp(0.4)))
    assert result.predicted_decision == "Out"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"schema": "unknown"}, "schema"),
        ({"feature_names": ["x", "x"], "scales": [1, 1], "coefficients": [1, 2]}, "feature_names"),
        ({"scales": [0]}, "scales"),
        ({"coefficients": [1, 2]}, "same length"),
        ({"decision_threshold": 1.2}, "less than or equal to 1"),
    ],
)
def test_artifact_rejects_malformed_contract(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LinearClassifierArtifact.model_validate(_artifact(**overrides))


def test_registry_resolves_production_for_live_pitch(tmp_path: Path) -> None:
    registry = ClassifierRegistry.load(
        _write_registry(tmp_path, [_artifact()])
    )

    resolved = registry.resolve(VC)

    assert resolved.training_context == "production"


def test_registry_resolves_only_exact_held_episode_for_replay(tmp_path: Path) -> None:
    registry = ClassifierRegistry.load(
        _write_registry(
            tmp_path,
            [
                _artifact(),
                _artifact(
                    artifact_id="charles-held-20-v1",
                    training_context="leave_one_episode_out",
                    excluded_episode_slug="20-harper-wilde",
                    training_episode_slugs=["18-rowvigor"],
                ),
            ],
        )
    )

    held = registry.resolve(VC, excluded_episode_slug="20-harper-wilde")

    assert held.artifact_id == "charles-held-20-v1"
    assert "20-harper-wilde" not in held.training_episode_slugs
    with pytest.raises(LookupError, match="held-episode"):
        registry.resolve(VC, excluded_episode_slug="39-this-pitch-is-damn-near-perfect")


def test_registry_rejects_wrong_vc_inside_artifact(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_artifact()])
    artifact_path = tmp_path / "artifact-0.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["vc_slug"] = "elizabeth-yin-hustle-fund"
    artifact_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    registry_payload = json.loads(path.read_text(encoding="utf-8"))
    registry_payload["artifacts"][0]["sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(registry_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="VC identity"):
        ClassifierRegistry.load(path)


def test_registry_detects_artifact_hash_mismatch(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_artifact()])
    (tmp_path / "artifact-0.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        ClassifierRegistry.load(path)


def test_registry_rejects_path_escape(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_artifact()])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifacts"][0]["path"] = "../artifact-0.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="safe relative"):
        ClassifierRegistry.load(path)


def test_registry_can_return_structured_fallback(tmp_path: Path) -> None:
    registry = ClassifierRegistry.load(_write_registry(tmp_path, [_artifact()]))

    resolution = registry.resolve_or_fallback(
        VC, excluded_episode_slug="999-missing"
    )

    assert resolution.status == "fallback"
    assert resolution.artifact is None
    assert "held-episode" in (resolution.reason or "")
