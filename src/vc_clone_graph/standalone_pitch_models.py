"""Standalone classifiers over frozen Phase 1 rationales and audited pitches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .artifacts import read_verified_frozen, write_json
from .firewall import sha256_file, verify_package
from .posthoc_semantic_models import (
    _C_VALUES,
    _estimator_score,
    _linear_estimator,
    _metrics,
    _raw_metrics,
    _select_candidate,
    _tfidf,
    _write_csv,
    load_semantic_population,
)
from .schemas_v4 import InvestigationV4
from .two_head_calibration import ThresholdSelection


_SEED = 20260809
_METHODS = (
    "rationale_only",
    "pitch_tfidf_only",
    "rationale_plus_pitch_tfidf",
    "pitch_embedding_only",
    "rationale_plus_pitch_embedding",
)


@dataclass(frozen=True)
class StandalonePitchRecord:
    episode_slug: str
    target: int
    target_decision: str
    original_decision: str
    original_likelihood: float
    pitch_text: str
    rationale_features: Mapping[str, float]
    pitch_sha256: str
    phase1_sha256: str


def extract_phase1_rationale_features(
    investigation: Mapping[str, Any],
) -> dict[str, float]:
    """Extract structured features exclusively from a frozen investigation."""
    features: dict[str, float] = {
        "rationale_count": float(len(investigation.get("rationales", []))),
        "question_count": float(len(investigation.get("questions", []))),
        "conflict_count": float(len(investigation.get("conflicts", []))),
        "unmapped_observation_count": float(
            len(investigation.get("unmapped_observations", []))
        ),
        "information_sufficient": float(bool(investigation.get("information_sufficient"))),
        "searchable_question_count": float(
            len(investigation.get("searchable_questions", []))
        ),
        "diligence_question_count": float(
            len(investigation.get("diligence_questions", []))
        ),
        "next_search_objective_count": float(
            len(investigation.get("next_search_objectives", []))
        ),
    }
    direction_weight = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    for rationale in investigation.get("rationales", []):
        label = str(rationale.get("taxonomy_label", "unknown"))
        direction = str(rationale.get("direction", "neutral"))
        salience = str(rationale.get("salience", "secondary"))
        confidence = float(rationale.get("confidence", 0.0))
        features[f"direction__{direction}__count"] = (
            features.get(f"direction__{direction}__count", 0.0) + 1.0
        )
        features[f"salience__{salience}__count"] = (
            features.get(f"salience__{salience}__count", 0.0) + 1.0
        )
        strength_key = f"{direction}_{salience}_confidence_sum"
        features[strength_key] = features.get(strength_key, 0.0) + confidence
        signed_key = f"label__{label}__signed_confidence"
        features[signed_key] = (
            features.get(signed_key, 0.0)
            + direction_weight.get(direction, 0.0) * confidence
        )
        confidence_key = f"label__{label}__confidence_sum"
        features[confidence_key] = features.get(confidence_key, 0.0) + confidence
        activation_key = f"label_direction__{label}__{direction}__{salience}__count"
        features[activation_key] = features.get(activation_key, 0.0) + 1.0
        features["pitch_evidence_count"] = features.get("pitch_evidence_count", 0.0) + len(
            rationale.get("pitch_evidence_ids", [])
        )
        features["wiki_evidence_count"] = features.get("wiki_evidence_count", 0.0) + len(
            rationale.get("wiki_evidence_ids", [])
        )
        features["historical_evidence_count"] = features.get(
            "historical_evidence_count", 0.0
        ) + len(rationale.get("historical_evidence_ids", []))
        for source in ("pitch", "wiki", "historical", "portfolio"):
            source_key = f"{source}_evidence_ids"
            label_key = f"label__{label}__{source}_evidence_count"
            features[label_key] = features.get(label_key, 0.0) + len(
                rationale.get(source_key, [])
            )
    for question in investigation.get("questions", []):
        status = str(question.get("status", "unanswered"))
        key = f"question_status__{status}__count"
        features[key] = features.get(key, 0.0) + 1.0
    return {name: float(value) for name, value in features.items()}


def load_verified_pitch_population(
    input_root: Path,
    status_path: Path,
    labels_path: Path,
    vc_slug: str,
) -> list[StandalonePitchRecord]:
    """Join frozen Phase 1 artifacts to firewall-verified pitch-only text."""
    records = load_semantic_population(status_path, labels_path)
    result: list[StandalonePitchRecord] = []
    for record in records:
        package = verify_package(input_root, vc_slug, record.episode_slug)
        raw, digest = read_verified_frozen(
            Path(record.artifact_root) / "phase1/investigation.json",
            Path(record.artifact_root) / "phase1/investigation.sha256",
        )
        if digest != record.phase1_sha256:
            raise ValueError(f"Phase 1 digest mismatch: {record.episode_slug}")
        investigation = InvestigationV4.model_validate_json(raw)
        if investigation.episode_slug != record.episode_slug:
            raise ValueError(f"Phase 1 episode mismatch: {record.episode_slug}")
        pitch_text = package.pitch.read_text(encoding="utf-8")
        if not pitch_text.strip():
            raise ValueError(f"empty verified pitch: {record.episode_slug}")
        result.append(
            StandalonePitchRecord(
                episode_slug=record.episode_slug,
                target=record.target,
                target_decision=record.target_decision,
                original_decision=record.original_decision,
                original_likelihood=record.original_likelihood,
                pitch_text=pitch_text,
                rationale_features=extract_phase1_rationale_features(
                    investigation.model_dump(mode="json")
                ),
                pitch_sha256=sha256_file(package.pitch),
                phase1_sha256=digest,
            )
        )
    return result


@dataclass(frozen=True)
class StandalonePrediction:
    method: str
    episode_slug: str
    target: int
    score: float
    predicted: int
    threshold: float
    selected_config: Mapping[str, object]
    training_slugs: tuple[str, ...]


def _inner_split(targets: np.ndarray) -> StratifiedKFold:
    count = min(3, int(targets.sum()), int(len(targets) - targets.sum()))
    if count < 2:
        raise ValueError("nested standalone evaluation requires two cases per class")
    return StratifiedKFold(n_splits=count, shuffle=True, random_state=_SEED)


def _rationale_names(records: Sequence[StandalonePitchRecord]) -> tuple[str, ...]:
    return tuple(sorted({name for record in records for name in record.rationale_features}))


def _rationale_matrix(
    records: Sequence[StandalonePitchRecord], names: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [
            [float(record.rationale_features.get(name, 0.0)) for name in names]
            for record in records
        ],
        dtype=float,
    )


def _choose_linear_candidate(
    train_targets: np.ndarray,
    fold_matrices: Sequence[tuple[np.ndarray, np.ndarray, Any, Any]],
    cap: float,
) -> tuple[Mapping[str, object], ThresholdSelection]:
    candidates = []
    for kind in ("logistic", "linear_svc"):
        for c_value in _C_VALUES:
            oof = np.zeros(len(train_targets), dtype=float)
            for fit, validate, x_fit, x_validate in fold_matrices:
                estimator = _linear_estimator(kind, c_value)
                estimator.fit(x_fit, train_targets[fit])
                oof[validate] = _estimator_score(estimator, x_validate)
            candidates.append(({"kind": kind, "c": c_value}, oof))
    config, selection, _ = _select_candidate(train_targets, candidates, cap)
    return config, selection


def _dense_outer(
    train_records: Sequence[StandalonePitchRecord],
    train_targets: np.ndarray,
    held_record: StandalonePitchRecord,
    train_embeddings: np.ndarray,
    held_embedding: np.ndarray,
    cap: float,
    *,
    include_rationales: bool,
    include_embeddings: bool,
) -> tuple[float, ThresholdSelection, Mapping[str, object]]:
    folds = []
    for fit, validate in _inner_split(train_targets).split(
        np.zeros(len(train_targets)), train_targets
    ):
        fit_records = [train_records[index] for index in fit]
        validate_records = [train_records[index] for index in validate]
        parts_fit = []
        parts_validate = []
        if include_rationales:
            names = _rationale_names(fit_records)
            parts_fit.append(_rationale_matrix(fit_records, names))
            parts_validate.append(_rationale_matrix(validate_records, names))
        if include_embeddings:
            parts_fit.append(train_embeddings[fit])
            parts_validate.append(train_embeddings[validate])
        x_fit = np.concatenate(parts_fit, axis=1)
        x_validate = np.concatenate(parts_validate, axis=1)
        scaler = StandardScaler()
        folds.append(
            (fit, validate, scaler.fit_transform(x_fit), scaler.transform(x_validate))
        )
    config, selection = _choose_linear_candidate(train_targets, folds, cap)

    train_parts = []
    held_parts = []
    if include_rationales:
        names = _rationale_names(train_records)
        train_parts.append(_rationale_matrix(train_records, names))
        held_parts.append(_rationale_matrix([held_record], names))
    if include_embeddings:
        train_parts.append(train_embeddings)
        held_parts.append(held_embedding.reshape(1, -1))
    x_train = np.concatenate(train_parts, axis=1)
    x_held = np.concatenate(held_parts, axis=1)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_held = scaler.transform(x_held)
    estimator = _linear_estimator(str(config["kind"]), float(config["c"]))
    estimator.fit(x_train, train_targets)
    return float(_estimator_score(estimator, x_held)[0]), selection, config


def _tfidf_outer(
    train_records: Sequence[StandalonePitchRecord],
    train_targets: np.ndarray,
    held_record: StandalonePitchRecord,
    cap: float,
    *,
    include_rationales: bool,
) -> tuple[float, ThresholdSelection, Mapping[str, object]]:
    folds = []
    for fit, validate in _inner_split(train_targets).split(
        np.zeros(len(train_targets)), train_targets
    ):
        fit_records = [train_records[index] for index in fit]
        validate_records = [train_records[index] for index in validate]
        vectorizer = _tfidf()
        x_fit = vectorizer.fit_transform([record.pitch_text for record in fit_records])
        x_validate = vectorizer.transform(
            [record.pitch_text for record in validate_records]
        )
        if include_rationales:
            names = _rationale_names(fit_records)
            scaler = StandardScaler()
            rationale_fit = scaler.fit_transform(_rationale_matrix(fit_records, names))
            rationale_validate = scaler.transform(
                _rationale_matrix(validate_records, names)
            )
            x_fit = hstack((x_fit, csr_matrix(rationale_fit)), format="csr")
            x_validate = hstack(
                (x_validate, csr_matrix(rationale_validate)), format="csr"
            )
        folds.append((fit, validate, x_fit, x_validate))
    config, selection = _choose_linear_candidate(train_targets, folds, cap)

    vectorizer = _tfidf()
    x_train = vectorizer.fit_transform([record.pitch_text for record in train_records])
    x_held = vectorizer.transform([held_record.pitch_text])
    if include_rationales:
        names = _rationale_names(train_records)
        scaler = StandardScaler()
        rationale_train = scaler.fit_transform(_rationale_matrix(train_records, names))
        rationale_held = scaler.transform(_rationale_matrix([held_record], names))
        x_train = hstack((x_train, csr_matrix(rationale_train)), format="csr")
        x_held = hstack((x_held, csr_matrix(rationale_held)), format="csr")
    estimator = _linear_estimator(str(config["kind"]), float(config["c"]))
    estimator.fit(x_train, train_targets)
    return float(_estimator_score(estimator, x_held)[0]), selection, config


def nested_standalone_predictions(
    records: Sequence[StandalonePitchRecord],
    pitch_embeddings: np.ndarray,
    *,
    methods: Sequence[str] = _METHODS,
    false_positive_rate_cap: float = 0.20,
) -> dict[str, list[StandalonePrediction]]:
    """Generate leakage-controlled standalone predictions for every held episode."""
    if len(records) < 6 or len({record.episode_slug for record in records}) != len(records):
        raise ValueError("standalone population must contain six unique episodes")
    unknown = set(methods) - set(_METHODS)
    if unknown:
        raise ValueError(f"unknown standalone methods: {sorted(unknown)}")
    embeddings = np.asarray(pitch_embeddings, dtype=float)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise ValueError("pitch embedding row mismatch")

    result: dict[str, list[StandalonePrediction]] = {method: [] for method in methods}
    for method in methods:
        for held_index, held in enumerate(records):
            training_indices = [index for index in range(len(records)) if index != held_index]
            train_records = [records[index] for index in training_indices]
            train_targets = np.asarray([record.target for record in train_records], dtype=int)
            if method in {"pitch_tfidf_only", "rationale_plus_pitch_tfidf"}:
                score, selection, config = _tfidf_outer(
                    train_records,
                    train_targets,
                    held,
                    false_positive_rate_cap,
                    include_rationales=method == "rationale_plus_pitch_tfidf",
                )
            else:
                score, selection, config = _dense_outer(
                    train_records,
                    train_targets,
                    held,
                    embeddings[training_indices],
                    embeddings[held_index],
                    false_positive_rate_cap,
                    include_rationales=method
                    in {"rationale_only", "rationale_plus_pitch_embedding"},
                    include_embeddings=method
                    in {"pitch_embedding_only", "rationale_plus_pitch_embedding"},
                )
            result[method].append(
                StandalonePrediction(
                    method=method,
                    episode_slug=held.episode_slug,
                    target=held.target,
                    score=score,
                    predicted=int(score >= selection.threshold),
                    threshold=selection.threshold,
                    selected_config=dict(config),
                    training_slugs=tuple(
                        records[index].episode_slug for index in training_indices
                    ),
                )
            )
    return result


def write_standalone_analysis(
    records: Sequence[StandalonePitchRecord],
    predictions: Mapping[str, Sequence[StandalonePrediction]],
    output_root: Path,
    *,
    embedding_metadata: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Write metrics and provenance for the standalone comparison."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    targets = np.asarray([record.target for record in records], dtype=int)
    methods: dict[str, dict[str, Any]] = {"raw_v4": _raw_metrics(records)}
    for method, rows in predictions.items():
        if len(rows) != len(records):
            raise ValueError(f"prediction count mismatch: {method}")
        methods[method] = _metrics(
            targets,
            np.asarray([row.predicted for row in rows], dtype=int),
            np.asarray([row.score for row in rows], dtype=float),
        )
    population = {
        "episodes": len(records),
        "ins": int(targets.sum()),
        "outs": int(len(records) - targets.sum()),
    }
    result = {
        "schema": "rationale-pitch-standalone-v1",
        "evaluation": "nested_leave_one_out_development_not_untouched_holdout",
        "population": population,
        "success_criterion": {"minimum_tp": 7, "maximum_fp": 12},
        "methods": methods,
    }
    manifest = {
        "schema": result["schema"],
        "api_cost_usd": 0.0,
        "prediction_reruns": 0,
        "pipeline_modified": False,
        "random_seed": _SEED,
        "false_positive_rate_cap": 0.20,
        "population": population,
        "methods": list(methods),
        "sources": dict(source_manifest),
        "pitch_sha256": {
            record.episode_slug: record.pitch_sha256 for record in records
        },
        "phase1_sha256": {
            record.episode_slug: record.phase1_sha256 for record in records
        },
    }
    prediction_rows = [
        {
            "method": method,
            "episode_slug": row.episode_slug,
            "target": row.target,
            "score": row.score,
            "predicted": row.predicted,
            "threshold": row.threshold,
            "selected_config": row.selected_config,
            "training_slugs": row.training_slugs,
        }
        for method, rows in predictions.items()
        for row in rows
    ]
    ranking_rows = [
        {"method": method, "budget": budget, **values}
        for method, metrics in methods.items()
        for budget, values in metrics["ranking"].items()
    ]
    lines = [
        "# Rationale and pitch standalone classifier comparison",
        "",
        f"Population: {population['episodes']} episodes "
        f"({population['ins']} Ins, {population['outs']} Outs).",
        "",
        "Success requires TP >= 7 and FP <= 12.",
        "",
        "| Method | TP | FP | TN | FN | Balanced accuracy | Precision | Recall | F1 | AP | AUC | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for method, metrics in methods.items():
        matrix = metrics["confusion_matrix"]
        passed = matrix["tp"] >= 7 and matrix["fp"] <= 12
        lines.append(
            f"| {method} | {matrix['tp']} | {matrix['fp']} | {matrix['tn']} | "
            f"{matrix['fn']} | {metrics['balanced_accuracy']:.3f} | "
            f"{metrics['in_precision']:.3f} | {metrics['in_recall']:.3f} | "
            f"{metrics['in_f1']:.3f} | {metrics['average_precision']:.3f} | "
            f"{metrics['roc_auc']:.3f} | {'YES' if passed else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "Pitch-only methods are controls for the incremental value of Phase 1 rationales.",
            "These are nested development results, not an untouched holdout estimate.",
            "No inference-pipeline changes, LLM calls, or prediction reruns were performed.",
            "",
        ]
    )
    write_json(output / "manifest.json", manifest)
    write_json(output / "embedding-metadata.json", dict(embedding_metadata))
    write_json(output / "metrics.json", result)
    _write_csv(output / "predictions.csv", prediction_rows)
    _write_csv(output / "rankings.csv", ranking_rows)
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result
