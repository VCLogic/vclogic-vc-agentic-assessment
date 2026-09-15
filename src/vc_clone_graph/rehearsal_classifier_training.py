"""Train and export trusted per-VC rehearsal classifier artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .per_vc_rationale_models import (
    decision_feature_view,
)
from .phase2_calibration_evaluation import Phase2CalibrationRecord
from .rehearsal_classifier import LinearClassifierArtifact


@dataclass(frozen=True)
class TrainedClassifierArtifact:
    artifact: LinearClassifierArtifact
    estimator: Any


_ELASTIC_CONFIGS = ((0.1, 0.5), (1.0, 0.5), (10.0, 0.5))


def _rehearsal_estimator(c_value: float, l1_ratio: float, seed: int):
    """Use the established elastic model with a reliable convergence budget."""
    return make_pipeline(
        DictVectorizer(sparse=True),
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=c_value,
            solver="saga",
            l1_ratio=l1_ratio,
            class_weight="balanced",
            max_iter=20_000,
            tol=1e-3,
            random_state=seed,
        ),
    )


def _best_threshold(targets: np.ndarray, scores: np.ndarray) -> float:
    candidates = sorted({0.0, 0.5, 1.0, *(float(value) for value in scores)})
    return max(
        candidates,
        key=lambda threshold: (
            balanced_accuracy_score(targets, scores >= threshold),
            threshold,
        ),
    )


def _select_elastic_config(
    records: Sequence[Phase2CalibrationRecord], *, seed: int
) -> tuple[float, float, float]:
    targets = np.asarray([record.target for record in records], dtype=int)
    split_count = min(3, int(targets.sum()), int(len(targets) - targets.sum()))
    if split_count < 2:
        raise ValueError("inner elastic selection needs at least two cases per class")
    maps = [decision_feature_view(record) for record in records]
    folds = tuple(
        StratifiedKFold(
            n_splits=split_count, shuffle=True, random_state=seed
        ).split(np.zeros(len(records)), targets)
    )
    candidates: list[tuple[float, float, float, np.ndarray]] = []
    for c_value, l1_ratio in _ELASTIC_CONFIGS:
        scores = np.zeros(len(records), dtype=float)
        for train_indices, held_indices in folds:
            estimator = _rehearsal_estimator(c_value, l1_ratio, seed)
            estimator.fit(
                [maps[int(index)] for index in train_indices],
                targets[train_indices],
            )
            scores[held_indices] = estimator.predict_proba(
                [maps[int(index)] for index in held_indices]
            )[:, 1]
        candidates.append(
            (average_precision_score(targets, scores), -c_value, c_value, scores)
        )
    _, _, selected_c, selected_scores = max(
        candidates, key=lambda row: (row[0], row[1])
    )
    return selected_c, 0.5, _best_threshold(targets, selected_scores)


def _artifact_id(
    vc_slug: str, model_version: str, excluded_episode_slug: str | None
) -> str:
    context = (
        f"held-{excluded_episode_slug}"
        if excluded_episode_slug is not None
        else "production"
    )
    return f"{vc_slug}--{context}--{model_version}"


def train_classifier_artifact(
    records: Sequence[Phase2CalibrationRecord],
    *,
    vc_slug: str,
    model_version: str,
    source_registry: str,
    label_version: str,
    seed: int,
    excluded_episode_slug: str | None = None,
) -> TrainedClassifierArtifact:
    """Fit one production or held-episode model using training rows only."""
    vc_records = [record for record in records if record.vc_slug == vc_slug]
    if not vc_records:
        raise ValueError(f"no records for VC: {vc_slug}")
    if excluded_episode_slug is not None:
        if not any(
            record.episode_slug == excluded_episode_slug for record in vc_records
        ):
            raise ValueError(f"excluded episode is not present: {excluded_episode_slug}")
        training = [
            record
            for record in vc_records
            if record.episode_slug != excluded_episode_slug
        ]
    else:
        training = list(vc_records)
    targets = np.asarray([record.target for record in training], dtype=int)
    if set(int(value) for value in targets) != {0, 1}:
        raise ValueError("classifier training requires both classes")
    class_counts = {
        "Out": int((targets == 0).sum()),
        "In": int((targets == 1).sum()),
    }
    if min(class_counts.values()) < 2:
        raise ValueError("classifier training requires at least two cases per class")
    c_value, l1_ratio, threshold = _select_elastic_config(training, seed=seed)
    estimator = _rehearsal_estimator(c_value, l1_ratio, seed)
    feature_maps = [decision_feature_view(record) for record in training]
    estimator.fit(feature_maps, targets)
    vectorizer = estimator.named_steps["dictvectorizer"]
    scaler = estimator.named_steps["standardscaler"]
    classifier = estimator.named_steps["logisticregression"]
    feature_names = tuple(str(value) for value in vectorizer.get_feature_names_out())
    scales = tuple(float(value) for value in scaler.scale_)
    coefficients = tuple(float(value) for value in classifier.coef_[0])
    artifact = LinearClassifierArtifact(
        schema="rehearsal-linear-classifier-v1",
        artifact_id=_artifact_id(vc_slug, model_version, excluded_episode_slug),
        model_version=model_version,
        vc_slug=vc_slug,
        training_context=(
            "leave_one_episode_out"
            if excluded_episode_slug is not None
            else "production"
        ),
        excluded_episode_slug=excluded_episode_slug,
        feature_names=feature_names,
        scales=scales,
        coefficients=coefficients,
        intercept=float(classifier.intercept_[0]),
        decision_threshold=float(threshold),
        training_episode_slugs=tuple(record.episode_slug for record in training),
        source_registry=source_registry,
        label_version=label_version,
        feature_schema="per-vc-decision-v1",
        seed=seed,
        class_counts=class_counts,
        selected_config=f"C={c_value:g};l1_ratio={l1_ratio:g}",
    )
    return TrainedClassifierArtifact(artifact=artifact, estimator=estimator)


def build_classifier_artifacts(
    records: Sequence[Phase2CalibrationRecord],
    *,
    vc_slug: str,
    model_version: str,
    source_registry: str,
    label_version: str,
    seed: int,
) -> tuple[TrainedClassifierArtifact, ...]:
    """Build a live model plus one leakage-safe model for every VC episode."""
    vc_records = sorted(
        (record for record in records if record.vc_slug == vc_slug),
        key=lambda record: record.episode_slug,
    )
    result = [
        train_classifier_artifact(
            vc_records,
            vc_slug=vc_slug,
            model_version=model_version,
            source_registry=source_registry,
            label_version=label_version,
            seed=seed,
        )
    ]
    for index, record in enumerate(vc_records, start=1):
        result.append(
            train_classifier_artifact(
                vc_records,
                vc_slug=vc_slug,
                model_version=model_version,
                source_registry=source_registry,
                label_version=label_version,
                seed=seed + index,
                excluded_episode_slug=record.episode_slug,
            )
        )
    return tuple(result)


def write_classifier_registry(
    output_root: Path,
    trained: Sequence[TrainedClassifierArtifact],
) -> Path:
    """Write canonical JSON artifacts and a digest-verified registry."""
    root = Path(output_root).resolve()
    artifacts_root = root / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for row in trained:
        artifact = row.artifact
        relative = Path("artifacts") / f"{artifact.artifact_id}.json"
        path = root / relative
        payload = artifact.model_dump(mode="json", by_alias=True)
        content = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        path.write_bytes(content)
        entries.append(
            {
                "artifact_id": artifact.artifact_id,
                "vc_slug": artifact.vc_slug,
                "training_context": artifact.training_context,
                "excluded_episode_slug": artifact.excluded_episode_slug,
                "path": relative.as_posix(),
                "sha256": sha256(content).hexdigest(),
            }
        )
    registry = root / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "rehearsal-classifier-registry-v1",
                "artifacts": sorted(
                    entries,
                    key=lambda row: (
                        str(row["vc_slug"]),
                        str(row["training_context"]),
                        str(row["excluded_episode_slug"] or ""),
                    ),
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return registry
