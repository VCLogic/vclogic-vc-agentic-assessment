"""Leakage-controlled fusion of frozen Phase 1, Phase 2, and pitch signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .posthoc_semantic_models import (
    _C_VALUES,
    _estimator_score,
    _linear_estimator,
    _metrics,
    _raw_metrics,
    _select_candidate,
    _tfidf,
    _write_csv,
    SemanticCalibrationRecord,
)
from .artifacts import write_json
from .standalone_pitch_models import StandalonePitchRecord
from .two_head_calibration import ThresholdSelection


_SEED = 20260810
EARLY_FUSION_METHODS = (
    "phase2_structured_linear",
    "phase2_structured_nonlinear",
    "dual_embedding_linear",
    "all_dense_linear",
    "all_dense_nonlinear",
    "full_structured_pitch_tfidf",
    "multiview_tfidf",
)
FIXED_FUSION_METHODS = (
    "phase2_semantic_mean",
    "all_view_mean",
    "raw_heavy_mean",
    "all_view_median",
    "supportive_max",
    "conservative_min",
    "majority_vote",
    "raw_with_consensus_override",
)
_COMPONENT_NAMES = (
    "full_embedding",
    "structured",
    "full_tfidf",
    "pitch_embedding",
    "pitch_tfidf",
)


@dataclass(frozen=True)
class FullInferenceRecord:
    episode_slug: str
    target: int
    target_decision: str
    original_decision: str
    original_likelihood: float
    review_priority_score: float
    pitch_text: str
    phase1_text: str
    full_reasoning: str
    structured_features: Mapping[str, float]
    pitch_sha256: str
    phase1_sha256: str
    phase2_sha256: str
    artifact_root: str


def build_full_inference_population(
    semantic_records: Sequence[SemanticCalibrationRecord],
    pitch_records: Sequence[StandalonePitchRecord],
) -> list[FullInferenceRecord]:
    """Strictly align already-verified inference and pitch populations."""
    semantic_slugs = [record.episode_slug for record in semantic_records]
    pitch_by_slug = {record.episode_slug: record for record in pitch_records}
    if len(pitch_by_slug) != len(pitch_records) or set(semantic_slugs) != set(pitch_by_slug):
        raise ValueError("semantic and pitch population mismatch")

    rows: list[FullInferenceRecord] = []
    for semantic in semantic_records:
        pitch = pitch_by_slug[semantic.episode_slug]
        if semantic.target != pitch.target or semantic.target_decision != pitch.target_decision:
            raise ValueError(f"target mismatch: {semantic.episode_slug}")
        if semantic.phase1_sha256 != pitch.phase1_sha256:
            raise ValueError(f"Phase 1 hash mismatch: {semantic.episode_slug}")
        if semantic.original_decision != pitch.original_decision:
            raise ValueError(f"original decision mismatch: {semantic.episode_slug}")
        if abs(semantic.original_likelihood - pitch.original_likelihood) > 1e-12:
            raise ValueError(f"original likelihood mismatch: {semantic.episode_slug}")
        rows.append(
            FullInferenceRecord(
                episode_slug=semantic.episode_slug,
                target=semantic.target,
                target_decision=semantic.target_decision,
                original_decision=semantic.original_decision,
                original_likelihood=semantic.original_likelihood,
                review_priority_score=semantic.review_priority_score,
                pitch_text=pitch.pitch_text,
                phase1_text=semantic.phase1_text,
                full_reasoning=semantic.full_text,
                structured_features=dict(semantic.structured_features),
                pitch_sha256=pitch.pitch_sha256,
                phase1_sha256=semantic.phase1_sha256,
                phase2_sha256=semantic.phase2_sha256,
                artifact_root=semantic.artifact_root,
            )
        )
    return rows


@dataclass(frozen=True)
class FullFusionPrediction:
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
        raise ValueError("nested full-fusion evaluation requires two cases per class")
    return StratifiedKFold(n_splits=count, shuffle=True, random_state=_SEED)


def _feature_names(records: Sequence[FullInferenceRecord]) -> tuple[str, ...]:
    return tuple(sorted({name for row in records for name in row.structured_features}))


def _phase2_matrix(
    records: Sequence[FullInferenceRecord], names: Sequence[str]
) -> np.ndarray:
    structured = [
        [float(row.structured_features.get(name, 0.0)) for name in names]
        for row in records
    ]
    for values, row in zip(structured, records, strict=True):
        values.extend(
            (
                row.original_likelihood,
                float(row.original_decision == "In"),
                row.review_priority_score,
            )
        )
    return np.asarray(structured, dtype=float)


def _linear_candidates(
    targets: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray, Any, Any]],
    cap: float,
) -> tuple[Mapping[str, object], ThresholdSelection]:
    candidates = []
    for kind in ("logistic", "linear_svc"):
        for c_value in _C_VALUES:
            oof = np.zeros(len(targets), dtype=float)
            for fit, validate, x_fit, x_validate in folds:
                estimator = _linear_estimator(kind, c_value)
                estimator.fit(x_fit, targets[fit])
                oof[validate] = _estimator_score(estimator, x_validate)
            candidates.append(({"kind": kind, "c": c_value}, oof))
    config, selection, _ = _select_candidate(targets, candidates, cap)
    return config, selection


def _nonlinear_estimator(config: Mapping[str, object]):
    if config["kind"] == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=200,
            max_depth=int(config["max_depth"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            class_weight="balanced",
            random_state=_SEED,
            n_jobs=1,
        )
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        early_stopping=False,
        learning_rate=0.05,
        max_depth=int(config["max_depth"]),
        max_iter=100,
        min_samples_leaf=int(config["min_samples_leaf"]),
        random_state=_SEED,
    )


def _nonlinear_configurations(feature_count: int) -> list[dict[str, object]]:
    kinds = ("extra_trees",) if feature_count > 256 else (
        "extra_trees",
        "hist_gradient_boosting",
    )
    return [
        {"kind": kind, "max_depth": depth, "min_samples_leaf": leaf}
        for kind in kinds
        for depth in (1, 2)
        for leaf in (3, 5)
    ]


def _nonlinear_candidates(
    targets: np.ndarray,
    matrices: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    cap: float,
) -> tuple[Mapping[str, object], ThresholdSelection]:
    configs = _nonlinear_configurations(int(matrices[0][2].shape[1]))
    candidates = []
    for config in configs:
        oof = np.zeros(len(targets), dtype=float)
        for fit, validate, x_fit, x_validate in matrices:
            estimator = _nonlinear_estimator(config)
            estimator.fit(x_fit, targets[fit])
            oof[validate] = _estimator_score(estimator, x_validate)
        candidates.append((config, oof))
    config, selection, _ = _select_candidate(targets, candidates, cap)
    return config, selection


def _dense_outer(
    train_records: Sequence[FullInferenceRecord],
    train_targets: np.ndarray,
    held_record: FullInferenceRecord,
    train_pitch_embeddings: np.ndarray,
    held_pitch_embedding: np.ndarray,
    train_reasoning_embeddings: np.ndarray,
    held_reasoning_embedding: np.ndarray,
    cap: float,
    *,
    include_structured: bool,
    include_embeddings: bool,
    nonlinear: bool,
) -> tuple[float, ThresholdSelection, Mapping[str, object]]:
    fold_matrices = []
    for fit, validate in _inner_split(train_targets).split(
        np.zeros(len(train_targets)), train_targets
    ):
        fit_records = [train_records[index] for index in fit]
        validate_records = [train_records[index] for index in validate]
        fit_parts: list[np.ndarray] = []
        validate_parts: list[np.ndarray] = []
        if include_structured:
            names = _feature_names(fit_records)
            fit_parts.append(_phase2_matrix(fit_records, names))
            validate_parts.append(_phase2_matrix(validate_records, names))
        if include_embeddings:
            fit_parts.extend((train_pitch_embeddings[fit], train_reasoning_embeddings[fit]))
            validate_parts.extend(
                (train_pitch_embeddings[validate], train_reasoning_embeddings[validate])
            )
        x_fit = np.concatenate(fit_parts, axis=1)
        x_validate = np.concatenate(validate_parts, axis=1)
        scaler = StandardScaler()
        fold_matrices.append(
            (fit, validate, scaler.fit_transform(x_fit), scaler.transform(x_validate))
        )
    if nonlinear:
        config, selection = _nonlinear_candidates(train_targets, fold_matrices, cap)
    else:
        config, selection = _linear_candidates(train_targets, fold_matrices, cap)

    train_parts = []
    held_parts = []
    if include_structured:
        names = _feature_names(train_records)
        train_parts.append(_phase2_matrix(train_records, names))
        held_parts.append(_phase2_matrix([held_record], names))
    if include_embeddings:
        train_parts.extend((train_pitch_embeddings, train_reasoning_embeddings))
        held_parts.extend(
            (held_pitch_embedding.reshape(1, -1), held_reasoning_embedding.reshape(1, -1))
        )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(np.concatenate(train_parts, axis=1))
    x_held = scaler.transform(np.concatenate(held_parts, axis=1))
    estimator = (
        _nonlinear_estimator(config)
        if nonlinear
        else _linear_estimator(str(config["kind"]), float(config["c"]))
    )
    estimator.fit(x_train, train_targets)
    return float(_estimator_score(estimator, x_held)[0]), selection, config


def _sparse_outer(
    train_records: Sequence[FullInferenceRecord],
    train_targets: np.ndarray,
    held_record: FullInferenceRecord,
    cap: float,
    *,
    multiview: bool,
) -> tuple[float, ThresholdSelection, Mapping[str, object]]:
    folds = []
    for fit, validate in _inner_split(train_targets).split(
        np.zeros(len(train_targets)), train_targets
    ):
        fit_records = [train_records[index] for index in fit]
        validate_records = [train_records[index] for index in validate]
        pitch_vectorizer = _tfidf()
        x_fit = pitch_vectorizer.fit_transform([row.pitch_text for row in fit_records])
        x_validate = pitch_vectorizer.transform([row.pitch_text for row in validate_records])
        if multiview:
            reasoning_vectorizer = _tfidf()
            x_fit = hstack(
                (
                    x_fit,
                    reasoning_vectorizer.fit_transform(
                        [row.full_reasoning for row in fit_records]
                    ),
                ),
                format="csr",
            )
            x_validate = hstack(
                (
                    x_validate,
                    reasoning_vectorizer.transform(
                        [row.full_reasoning for row in validate_records]
                    ),
                ),
                format="csr",
            )
            names: tuple[str, ...] = ()
        else:
            names = _feature_names(fit_records)
        scaler = StandardScaler()
        dense_fit = scaler.fit_transform(_phase2_matrix(fit_records, names))
        dense_validate = scaler.transform(_phase2_matrix(validate_records, names))
        folds.append(
            (
                fit,
                validate,
                hstack((x_fit, csr_matrix(dense_fit)), format="csr"),
                hstack((x_validate, csr_matrix(dense_validate)), format="csr"),
            )
        )
    config, selection = _linear_candidates(train_targets, folds, cap)

    pitch_vectorizer = _tfidf()
    x_train = pitch_vectorizer.fit_transform([row.pitch_text for row in train_records])
    x_held = pitch_vectorizer.transform([held_record.pitch_text])
    if multiview:
        reasoning_vectorizer = _tfidf()
        x_train = hstack(
            (
                x_train,
                reasoning_vectorizer.fit_transform(
                    [row.full_reasoning for row in train_records]
                ),
            ),
            format="csr",
        )
        x_held = hstack(
            (x_held, reasoning_vectorizer.transform([held_record.full_reasoning])),
            format="csr",
        )
        names = ()
    else:
        names = _feature_names(train_records)
    scaler = StandardScaler()
    dense_train = scaler.fit_transform(_phase2_matrix(train_records, names))
    dense_held = scaler.transform(_phase2_matrix([held_record], names))
    x_train = hstack((x_train, csr_matrix(dense_train)), format="csr")
    x_held = hstack((x_held, csr_matrix(dense_held)), format="csr")
    estimator = _linear_estimator(str(config["kind"]), float(config["c"]))
    estimator.fit(x_train, train_targets)
    return float(_estimator_score(estimator, x_held)[0]), selection, config


def nested_full_fusion_predictions(
    records: Sequence[FullInferenceRecord],
    embeddings: Mapping[str, np.ndarray],
    *,
    methods: Sequence[str] = EARLY_FUSION_METHODS,
    false_positive_rate_cap: float = 0.20,
) -> dict[str, list[FullFusionPrediction]]:
    """Evaluate full inference-time feature views with nested LOEO."""
    if len(records) < 6 or len({row.episode_slug for row in records}) != len(records):
        raise ValueError("full-fusion population must contain six unique episodes")
    unknown = set(methods) - set(EARLY_FUSION_METHODS)
    if unknown:
        raise ValueError(f"unknown full-fusion methods: {sorted(unknown)}")
    vectors = {name: np.asarray(value, dtype=float) for name, value in embeddings.items()}
    for name in ("pitch", "full_reasoning"):
        if vectors.get(name, np.empty((0,))).ndim != 2 or vectors[name].shape[0] != len(records):
            raise ValueError(f"embedding row mismatch: {name}")

    result: dict[str, list[FullFusionPrediction]] = {method: [] for method in methods}
    for method in methods:
        for held_index, held in enumerate(records):
            training_indices = [index for index in range(len(records)) if index != held_index]
            train_records = [records[index] for index in training_indices]
            train_targets = np.asarray([row.target for row in train_records], dtype=int)
            if method in {"full_structured_pitch_tfidf", "multiview_tfidf"}:
                score, selection, config = _sparse_outer(
                    train_records,
                    train_targets,
                    held,
                    false_positive_rate_cap,
                    multiview=method == "multiview_tfidf",
                )
            else:
                score, selection, config = _dense_outer(
                    train_records,
                    train_targets,
                    held,
                    vectors["pitch"][training_indices],
                    vectors["pitch"][held_index],
                    vectors["full_reasoning"][training_indices],
                    vectors["full_reasoning"][held_index],
                    false_positive_rate_cap,
                    include_structured=method != "dual_embedding_linear",
                    include_embeddings=method
                    in {"dual_embedding_linear", "all_dense_linear", "all_dense_nonlinear"},
                    nonlinear=method
                    in {"phase2_structured_nonlinear", "all_dense_nonlinear"},
                )
            result[method].append(
                FullFusionPrediction(
                    method=method,
                    episode_slug=held.episode_slug,
                    target=held.target,
                    score=score,
                    predicted=int(score >= selection.threshold),
                    threshold=selection.threshold,
                    selected_config=dict(config),
                    training_slugs=tuple(records[index].episode_slug for index in training_indices),
                )
            )
    return result


def fixed_score_fusions(
    records: Sequence[FullInferenceRecord],
    components: Mapping[str, Sequence[FullFusionPrediction]],
) -> dict[str, list[FullFusionPrediction]]:
    """Combine already-held-out component margins using label-blind rules."""
    missing = set(_COMPONENT_NAMES) - set(components)
    if missing:
        raise ValueError(f"missing fusion components: {sorted(missing)}")
    expected_slugs = [row.episode_slug for row in records]
    expected_training = {
        row.episode_slug: frozenset(expected_slugs) - {row.episode_slug} for row in records
    }
    aligned: dict[str, dict[str, FullFusionPrediction]] = {}
    for name in _COMPONENT_NAMES:
        rows = components[name]
        by_slug = {row.episode_slug: row for row in rows}
        if len(by_slug) != len(rows) or set(by_slug) != set(expected_slugs):
            raise ValueError(f"component population mismatch: {name}")
        aligned[name] = by_slug
    result: dict[str, list[FullFusionPrediction]] = {
        method: [] for method in FIXED_FUSION_METHODS
    }
    for record in records:
        rows = {name: aligned[name][record.episode_slug] for name in _COMPONENT_NAMES}
        for name, row in rows.items():
            if row.target != record.target:
                raise ValueError(f"component target mismatch: {name}/{record.episode_slug}")
            if frozenset(row.training_slugs) != expected_training[record.episode_slug]:
                raise ValueError(
                    f"component training provenance mismatch: {name}/{record.episode_slug}"
                )
        margins = {name: row.score - row.threshold for name, row in rows.items()}
        raw_margin = record.original_likelihood - 0.5
        all_margins = np.asarray([raw_margin, *margins.values()], dtype=float)
        learned = np.asarray(list(margins.values()), dtype=float)
        phase2 = np.asarray(
            [raw_margin, margins["full_embedding"], margins["structured"]], dtype=float
        )
        raw_vote = 1 if record.original_decision == "In" else -1
        learned_votes = np.where(learned > 0.0, 1, -1)
        consensus_override = raw_vote
        if int(np.sum(learned_votes == -raw_vote)) >= 3:
            consensus_override = -raw_vote
        formulas = {
            "phase2_semantic_mean": float(np.mean(phase2)),
            "all_view_mean": float(np.mean(all_margins)),
            "raw_heavy_mean": float(0.5 * raw_margin + 0.1 * np.sum(learned)),
            "all_view_median": float(np.median(all_margins)),
            "supportive_max": float(np.max(all_margins)),
            "conservative_min": float(np.min(all_margins)),
            "majority_vote": float(np.mean(np.where(all_margins > 0.0, 1, -1))),
            "raw_with_consensus_override": float(consensus_override),
        }
        for method, score in formulas.items():
            result[method].append(
                FullFusionPrediction(
                    method=method,
                    episode_slug=record.episode_slug,
                    target=record.target,
                    score=score,
                    predicted=int(score > 0.0),
                    threshold=0.0,
                    selected_config={"kind": "fixed_label_blind_score_fusion"},
                    training_slugs=tuple(sorted(expected_training[record.episode_slug])),
                )
            )
    return result


def native_phase2_score_fusions(
    records: Sequence[FullInferenceRecord],
) -> dict[str, list[FullFusionPrediction]]:
    """Apply transparent fixed rules to Phase 2's two native continuous scores."""
    methods = {
        "phase2_likelihood_priority_70_30": lambda row: (
            0.7 * row.original_likelihood + 0.3 * row.review_priority_score
        ),
        "phase2_likelihood_priority_mean": lambda row: (
            row.original_likelihood + row.review_priority_score
        )
        / 2.0,
        "phase2_likelihood_priority_min": lambda row: min(
            row.original_likelihood, row.review_priority_score
        ),
    }
    slugs = {row.episode_slug for row in records}
    result = {method: [] for method in methods}
    for method, formula in methods.items():
        for row in records:
            score = float(formula(row))
            result[method].append(
                FullFusionPrediction(
                    method=method,
                    episode_slug=row.episode_slug,
                    target=row.target,
                    score=score,
                    predicted=int(score >= 0.5),
                    threshold=0.5,
                    selected_config={
                        "kind": "fixed_native_phase2_score_fusion",
                        "exploratory_posthoc": True,
                    },
                    training_slugs=tuple(sorted(slugs - {row.episode_slug})),
                )
            )
    return result


def _correction_matrix(
    records: Sequence[FullInferenceRecord],
    names: Sequence[str],
    pitch_embeddings: np.ndarray,
    reasoning_embeddings: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        (
            _phase2_matrix(records, names),
            np.asarray(pitch_embeddings, dtype=float),
            np.asarray(reasoning_embeddings, dtype=float),
        ),
        axis=1,
    )


def _head_trainable(labels: np.ndarray) -> bool:
    return len(labels) >= 4 and int(labels.sum()) >= 2 and int(len(labels) - labels.sum()) >= 2


def _head_oof(
    matrix: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, Mapping[str, object]]:
    splitter = _inner_split(labels)
    candidates = []
    for c_value in _C_VALUES:
        oof = np.zeros(len(labels), dtype=float)
        for fit, validate in splitter.split(matrix, labels):
            estimator = make_pipeline(
                StandardScaler(), _linear_estimator("logistic", c_value)
            )
            estimator.fit(matrix[fit], labels[fit])
            oof[validate] = _estimator_score(estimator, matrix[validate])
        candidates.append((float(average_precision_score(labels, oof)), c_value, oof))
    _, c_value, scores = max(candidates, key=lambda item: (item[0], -item[1]))
    return scores, {"kind": "logistic", "c": c_value}


def _select_correction_thresholds(
    records: Sequence[FullInferenceRecord],
    suppress_indices: np.ndarray,
    suppress_scores: np.ndarray,
    rescue_indices: np.ndarray,
    rescue_scores: np.ndarray,
    cap: float,
) -> tuple[float, float]:
    targets = np.asarray([row.target for row in records], dtype=int)
    raw = np.asarray([row.original_decision == "In" for row in records], dtype=bool)
    candidates = (0.35, 0.5, 0.65)
    best = None
    for suppress_threshold in candidates:
        for rescue_threshold in candidates:
            predicted = raw.copy()
            predicted[suppress_indices] = suppress_scores < suppress_threshold
            predicted[rescue_indices] = rescue_scores >= rescue_threshold
            tp = int(np.sum(predicted & (targets == 1)))
            fp = int(np.sum(predicted & (targets == 0)))
            tn = int(np.sum(~predicted & (targets == 0)))
            fn = int(np.sum(~predicted & (targets == 1)))
            fpr = fp / (fp + tn) if fp + tn else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            precision = tp / (tp + fp) if tp + fp else 0.0
            specificity = tn / (tn + fp) if tn + fp else 0.0
            within_cap = fpr <= cap + 1e-12
            key = (
                int(within_cap),
                recall if within_cap else -fpr,
                precision,
                (recall + specificity) / 2.0,
                suppress_threshold,
                rescue_threshold,
            )
            if best is None or key > best[0]:
                best = (key, suppress_threshold, rescue_threshold)
    assert best is not None
    return best[1], best[2]


def nested_error_correction_predictions(
    records: Sequence[FullInferenceRecord],
    embeddings: Mapping[str, np.ndarray],
    *,
    false_positive_rate_cap: float = 0.20,
) -> list[FullFusionPrediction]:
    """Train separate false-In suppression and missed-In rescue heads."""
    pitch_vectors = np.asarray(embeddings["pitch"], dtype=float)
    reasoning_vectors = np.asarray(embeddings["full_reasoning"], dtype=float)
    if (
        pitch_vectors.ndim != 2
        or reasoning_vectors.ndim != 2
        or pitch_vectors.shape[0] != len(records)
        or reasoning_vectors.shape[0] != len(records)
    ):
        raise ValueError("error-correction embedding row mismatch")
    predictions = []
    for held_index, held in enumerate(records):
        training_indices = np.asarray(
            [index for index in range(len(records)) if index != held_index], dtype=int
        )
        train_records = [records[index] for index in training_indices]
        names = _feature_names(train_records)
        x_train = _correction_matrix(
            train_records,
            names,
            pitch_vectors[training_indices],
            reasoning_vectors[training_indices],
        )
        x_held = _correction_matrix(
            [held],
            names,
            pitch_vectors[held_index].reshape(1, -1),
            reasoning_vectors[held_index].reshape(1, -1),
        )
        targets = np.asarray([row.target for row in train_records], dtype=int)
        raw_in = np.asarray(
            [row.original_decision == "In" for row in train_records], dtype=bool
        )
        suppress_indices = np.flatnonzero(raw_in)
        rescue_indices = np.flatnonzero(~raw_in)
        suppress_labels = 1 - targets[suppress_indices]
        rescue_labels = targets[rescue_indices]
        training_slugs = tuple(row.episode_slug for row in train_records)
        if not (_head_trainable(suppress_labels) and _head_trainable(rescue_labels)):
            predictions.append(
                FullFusionPrediction(
                    method="rich_error_correction",
                    episode_slug=held.episode_slug,
                    target=held.target,
                    score=held.original_likelihood,
                    predicted=int(held.original_decision == "In"),
                    threshold=0.5,
                    selected_config={"fallback": "raw_decision"},
                    training_slugs=training_slugs,
                )
            )
            continue

        suppress_oof, suppress_config = _head_oof(
            x_train[suppress_indices], suppress_labels
        )
        rescue_oof, rescue_config = _head_oof(x_train[rescue_indices], rescue_labels)
        suppress_threshold, rescue_threshold = _select_correction_thresholds(
            train_records,
            suppress_indices,
            suppress_oof,
            rescue_indices,
            rescue_oof,
            false_positive_rate_cap,
        )
        if held.original_decision == "In":
            head_indices = suppress_indices
            head_labels = suppress_labels
            config = suppress_config
        else:
            head_indices = rescue_indices
            head_labels = rescue_labels
            config = rescue_config
        estimator = make_pipeline(
            StandardScaler(), _linear_estimator("logistic", float(config["c"]))
        )
        estimator.fit(x_train[head_indices], head_labels)
        head_score = float(_estimator_score(estimator, x_held)[0])
        if held.original_decision == "In":
            score = 0.5 + (suppress_threshold - head_score) / 2.0
        else:
            score = 0.5 + (head_score - rescue_threshold) / 2.0
        predictions.append(
            FullFusionPrediction(
                method="rich_error_correction",
                episode_slug=held.episode_slug,
                target=held.target,
                score=float(np.clip(score, 0.0, 1.0)),
                predicted=int(score >= 0.5),
                threshold=0.5,
                selected_config={
                    "fallback": None,
                    "suppressor": dict(suppress_config),
                    "rescuer": dict(rescue_config),
                    "suppress_threshold": suppress_threshold,
                    "rescue_threshold": rescue_threshold,
                },
                training_slugs=training_slugs,
            )
        )
    return predictions


def write_full_fusion_analysis(
    records: Sequence[FullInferenceRecord],
    predictions: Mapping[str, Sequence[FullFusionPrediction]],
    output_root: Any,
    *,
    embedding_metadata: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Write metrics, rankings, predictions, and immutable provenance."""
    from pathlib import Path

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    targets = np.asarray([row.target for row in records], dtype=int)
    methods: dict[str, dict[str, Any]] = {"raw_v4": _raw_metrics(records)}
    for method, rows in predictions.items():
        if len(rows) != len(records):
            raise ValueError(f"prediction count mismatch: {method}")
        if [row.episode_slug for row in rows] != [row.episode_slug for row in records]:
            raise ValueError(f"prediction order mismatch: {method}")
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
        "schema": "full-inference-fusion-v1",
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
        "pitch_sha256": {row.episode_slug: row.pitch_sha256 for row in records},
        "phase1_sha256": {row.episode_slug: row.phase1_sha256 for row in records},
        "phase2_sha256": {row.episode_slug: row.phase2_sha256 for row in records},
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
        "# Full inference-time fusion comparison",
        "",
        f"Population: {population['episodes']} episodes "
        f"({population['ins']} Ins, {population['outs']} Outs).",
        "",
        "Success requires TP >= 7 and FP <= 12.",
        "",
        "| Method | TP | FP | TN | FN | BA | Precision | Recall | F1 | AP | AUC | Pass |",
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
        (
            "",
            "All learned methods use nested leave-one-episode-out development evaluation.",
            "Fixed fusion rules combine already-held-out component margins without labels.",
            "No LLM calls, inference reruns, or pipeline changes were performed.",
            "",
        )
    )
    write_json(output / "manifest.json", manifest)
    write_json(output / "embedding-metadata.json", dict(embedding_metadata))
    write_json(output / "metrics.json", result)
    _write_csv(output / "predictions.csv", prediction_rows)
    _write_csv(output / "rankings.csv", ranking_rows)
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result
