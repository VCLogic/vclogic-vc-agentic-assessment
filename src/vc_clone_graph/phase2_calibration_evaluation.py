"""All-VC evaluation of raw and calibrated Phase 2 decisions."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import binomtest, rankdata

from .artifacts import read_verified_frozen
from .evaluation import load_canonical_artifacts, load_registry
from .schemas_v4 import DecisionV4, DecisionV41, InvestigationV4, InvestigationV41
from .two_head_calibration import extract_compact_v4_features


FeatureMap = dict[str, float]
Fold = tuple[tuple[int, ...], tuple[int, ...]]
FoldMaps = tuple[tuple[Fold, ...], ...]
_C_VALUES = (0.01, 0.1, 1.0, 10.0)


def _token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "unknown"


def _increment(features: FeatureMap, key: str, value: float = 1.0) -> None:
    features[key] = features.get(key, 0.0) + float(value)


def extract_phase2_features(
    decision: Mapping[str, Any], summary: Mapping[str, Any]
) -> FeatureMap:
    """Extract structured Phase 2 fields without prose or target labels."""
    decision_in = float(decision.get("decision") == "In")
    features: FeatureMap = {
        "decision_in": decision_in,
        "investment_likelihood": float(decision.get("investment_likelihood", 0.0)),
        "decision_confidence": float(decision.get("decision_confidence", 0.0)),
        "review_priority_score": float(decision.get("review_priority_score", 0.0)),
        "signed_decision_confidence": float(
            decision.get("decision_confidence", 0.0)
        ) * (1.0 if decision_in else -1.0),
        "controlling_rationale_count": float(
            len(decision.get("controlling_rationale_ids", []))
        ),
        "searchable_question_count": float(
            len(decision.get("searchable_questions", []))
        ),
        "diligence_question_count": float(
            len(decision.get("diligence_questions", []))
        ),
        "reversal_condition_count": float(
            len(decision.get("reversal_conditions", []))
        ),
        "next_search_count": float(
            len(decision.get("next_search_objectives", []))
        ),
        "missing_consideration_count": float(
            len(decision.get("missing_considerations", []))
        ),
        "information_sufficient": float(bool(decision.get("information_sufficient"))),
        "phase1_reopen_recommended": float(
            bool(decision.get("phase1_reopen_recommended"))
        ),
        "founder_exception_considered": float(
            bool(decision.get("founder_exception_considered"))
        ),
        "founder_exception_rationale_count": float(
            len(decision.get("founder_exception_rationale_ids", []))
        ),
        "founder_exception_precedent_count": float(
            len(decision.get("founder_exception_precedent_ids", []))
        ),
        "phase1_provisional": float(summary.get("phase1_status") == "provisional"),
        "phase2_provisional": float(summary.get("phase2_status") == "provisional"),
        "phase1_finding_count": float(len(summary.get("phase1_findings", []))),
        "phase2_finding_count": float(len(summary.get("phase2_findings", []))),
    }
    features[f"decision_path__{_token(decision.get('decision_path'))}"] = 1.0
    features[f"check_tier__{_token(decision.get('recommended_check_tier'))}"] = 1.0
    for evidence in decision.get("evidence_basis", []):
        source = _token(evidence.get("source_type"))
        effect = _token(evidence.get("effect_on_decision"))
        _increment(features, f"evidence_source__{source}__count")
        _increment(features, f"evidence_effect__{effect}__count")
        _increment(features, f"evidence_source_effect__{source}__{effect}__count")
        _increment(features, "decision_evidence_id_count", len(evidence.get("evidence_ids", [])))
    return {name: float(value) for name, value in features.items()}


def extract_taxonomy_features(
    investigation: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    vc_slug: str,
) -> FeatureMap:
    """Encode rationale identity and VC-specific rationale effects without labels."""
    features: FeatureMap = {}
    direction_weights = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    salience_weights = {"primary": 1.0, "secondary": 0.5}
    controlling = {str(value) for value in decision.get("controlling_rationale_ids", [])}
    for rationale in investigation.get("rationales", []):
        label = _token(rationale.get("taxonomy_label"))
        rationale_id = str(rationale.get("rationale_id", ""))
        direction = str(rationale.get("direction", "neutral"))
        salience = str(rationale.get("salience", "secondary"))
        confidence = float(rationale.get("confidence", 0.0))
        signed = direction_weights.get(direction, 0.0) * confidence
        salient = signed * salience_weights.get(salience, 0.5)
        prefix = f"rationale__{label}"
        _increment(features, f"{prefix}__count")
        _increment(features, f"{prefix}__signed_confidence", signed)
        _increment(features, f"{prefix}__salience_confidence", salient)
        _increment(features, f"{prefix}__direction__{_token(direction)}")
        _increment(features, f"{prefix}__salience__{_token(salience)}")
        _increment(
            features,
            f"{prefix}__pitch_evidence_count",
            len(rationale.get("pitch_evidence_ids", [])),
        )
        _increment(
            features,
            f"{prefix}__wiki_evidence_count",
            len(rationale.get("wiki_evidence_ids", [])),
        )
        _increment(
            features,
            f"{prefix}__historical_evidence_count",
            len(rationale.get("historical_evidence_ids", [])),
        )
        if rationale_id in controlling:
            _increment(features, f"{prefix}__controlling")
        vc_prefix = f"vc_rationale__{_token(vc_slug)}__{label}"
        _increment(features, f"{vc_prefix}__signed_confidence", signed)
        _increment(features, f"{vc_prefix}__salience_confidence", salient)
    for constraint in investigation.get("constraint_assessments", []):
        kind = _token(constraint.get("constraint_kind"))
        status = _token(constraint.get("status"))
        severity = _token(constraint.get("severity"))
        _increment(features, f"constraint__{kind}__{status}__{severity}")
        _increment(
            features,
            f"vc_constraint__{_token(vc_slug)}__{kind}__{status}__{severity}",
        )
    return {name: float(value) for name, value in features.items()}


@dataclass(frozen=True)
class Phase2CalibrationRecord:
    """One canonical investor-pitch case with label-blind feature families."""

    vc_slug: str
    vc_name: str
    episode_slug: str
    group: str
    target: int
    raw_decision: int
    raw_likelihood: float
    phase2_features: Mapping[str, float]
    combined_features: Mapping[str, float]
    artifact_root: str
    phase1_sha256: str
    phase2_sha256: str
    semantic_features: Mapping[str, float] = field(default_factory=dict)


def validate_phase2_payloads(
    investigation_raw: str, decision_raw: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a matching v4 or v4.1 frozen artifact pair."""
    investigation_payload = json.loads(investigation_raw)
    decision_payload = json.loads(decision_raw)
    if not isinstance(investigation_payload, dict) or not isinstance(decision_payload, dict):
        raise ValueError("Phase 2 calibration artifacts must be JSON objects")
    schema_pair = (
        investigation_payload.get("schema_version"),
        decision_payload.get("schema_version"),
    )
    models = {
        ("investigation-v4", "decision-v4"): (InvestigationV4, DecisionV4),
        ("investigation-v4.1", "decision-v4.1"): (
            InvestigationV41,
            DecisionV41,
        ),
    }.get(schema_pair)
    if models is None:
        raise ValueError(f"unsupported or mismatched Phase 2 schema pair: {schema_pair}")
    investigation_model, decision_model = models
    investigation = investigation_model.model_validate(investigation_payload)
    decision = decision_model.model_validate(decision_payload)
    return (
        investigation.model_dump(mode="json"),
        decision.model_dump(mode="json"),
    )


def load_phase2_records(registry_path: Path) -> list[Phase2CalibrationRecord]:
    """Load and hash-verify the canonical v4/v4.1 Phase 1 and Phase 2 artifacts."""
    registry = load_registry(Path(registry_path))
    artifacts = load_canonical_artifacts(registry)
    records: list[Phase2CalibrationRecord] = []
    seen: set[tuple[str, str]] = set()
    for artifact in artifacts:
        key = (artifact.vc_slug, artifact.episode_slug)
        if key in seen:
            raise ValueError(f"duplicate canonical investor-pitch case: {key}")
        seen.add(key)
        root = artifact.artifact_path.parent
        summary = json.loads(artifact.artifact_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError(f"summary must be an object: {artifact.artifact_path}")
        investigation_raw, investigation_digest = read_verified_frozen(
            root / "phase1/investigation.json",
            root / "phase1/investigation.sha256",
        )
        decision_raw, decision_digest = read_verified_frozen(
            root / "phase2/decision.json",
            root / "phase2/decision.sha256",
        )
        investigation_dict, decision_dict = validate_phase2_payloads(
            investigation_raw, decision_raw
        )
        if (
            investigation_dict["episode_slug"] != artifact.episode_slug
            or decision_dict["episode_slug"] != artifact.episode_slug
        ):
            raise ValueError(f"artifact episode mismatch: {artifact.episode_slug}")
        if decision_dict["investigation_sha256"] != investigation_digest:
            raise ValueError(
                f"decision Phase 1 digest mismatch: {artifact.episode_slug}"
            )
        vc_feature = {f"vc__{artifact.vc_slug}": 1.0}
        phase2 = {**extract_phase2_features(decision_dict, summary), **vc_feature}
        combined = {
            **extract_compact_v4_features(
                investigation_dict,
                decision_dict,
                summary,
            ),
            **vc_feature,
        }
        semantic = {
            **combined,
            **extract_taxonomy_features(
                investigation_dict,
                decision_dict,
                vc_slug=artifact.vc_slug,
            ),
        }
        records.append(
            Phase2CalibrationRecord(
                vc_slug=artifact.vc_slug,
                vc_name=artifact.vc_name,
                episode_slug=artifact.episode_slug,
                group=artifact.episode_slug,
                target=int(artifact.actual_decision == "In"),
                raw_decision=int(decision_dict["decision"] == "In"),
                raw_likelihood=float(decision_dict["investment_likelihood"]),
                phase2_features=phase2,
                combined_features=combined,
                artifact_root=str(root),
                phase1_sha256=investigation_digest,
                phase2_sha256=decision_digest,
                semantic_features=semantic,
            )
        )
    return sorted(records, key=lambda row: (row.vc_slug, row.episode_slug))


def make_repeated_grouped_folds(
    records: Sequence[Phase2CalibrationRecord],
    *,
    repeats: int = 5,
    splits: int = 5,
    seed: int = 20260816,
) -> FoldMaps:
    """Create deterministic stratified folds that never split an episode group."""
    if not records or repeats < 1 or splits < 2:
        raise ValueError("grouped folds need records, positive repeats, and >=2 splits")
    targets = np.asarray([record.target for record in records], dtype=int)
    groups = np.asarray([record.group for record in records], dtype=object)
    if len(set(targets)) != 2:
        raise ValueError("grouped folds require both target classes")
    actual_splits = min(
        splits,
        len(set(groups)),
        int(targets.sum()),
        int(len(targets) - targets.sum()),
    )
    if actual_splits < 2:
        raise ValueError("not enough groups or classes for grouped folds")
    dummy = np.zeros((len(records), 1), dtype=float)
    repetitions: list[tuple[Fold, ...]] = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=actual_splits,
            shuffle=True,
            random_state=seed + repeat,
        )
        repetition: list[Fold] = []
        for train, held in splitter.split(dummy, targets, groups):
            train_groups = set(groups[train])
            held_groups = set(groups[held])
            if train_groups & held_groups:
                raise RuntimeError("episode group appears in both train and held folds")
            repetition.append(
                (
                    tuple(int(index) for index in train),
                    tuple(int(index) for index in held),
                )
            )
        repetitions.append(tuple(repetition))
    return tuple(repetitions)


@dataclass(frozen=True)
class CalibratedPrediction:
    """Repeated out-of-fold prediction and its leakage-audit provenance."""

    vc_slug: str
    episode_slug: str
    group: str
    target: int
    method: str
    score: float
    predicted: int
    outer_repeats: int
    selected_cs: tuple[float, ...]
    thresholds: tuple[float, ...]
    outer_training_groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class MethodPrediction:
    """One aligned raw or learned decision used by metric functions."""

    vc_slug: str
    vc_name: str
    episode_slug: str
    group: str
    target: int
    method: str
    score: float
    predicted: int


def _feature_maps(
    records: Sequence[Phase2CalibrationRecord], family: str
) -> list[dict[str, float]]:
    if family == "phase2":
        return [dict(record.phase2_features) for record in records]
    if family == "combined":
        return [dict(record.combined_features) for record in records]
    if family == "semantic":
        return [dict(record.semantic_features) for record in records]
    raise ValueError(f"unknown Phase 2 calibration feature family: {family}")


def _estimator(c_value: float, seed: int):
    return make_pipeline(
        DictVectorizer(sparse=True),
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=20_000,
            random_state=seed,
        ),
    )


def _best_threshold(targets: np.ndarray, scores: np.ndarray) -> float:
    candidates = sorted({0.0, 0.5, 1.0, *(float(score) for score in scores)})
    best: tuple[float, float] | None = None
    for threshold in candidates:
        value = float(balanced_accuracy_score(targets, scores >= threshold))
        key = (value, threshold)
        if best is None or key > best:
            best = key
    assert best is not None
    return best[1]


def _inner_selection(
    feature_maps: Sequence[dict[str, float]],
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    inner_splits: int,
    seed: int,
) -> tuple[float, LogisticRegression, float]:
    class_counts = (int(targets.sum()), int(len(targets) - targets.sum()))
    split_count = min(inner_splits, len(set(groups)), *class_counts)
    if split_count < 2:
        raise ValueError("outer training fold lacks classes for nested calibration")
    splitter = StratifiedGroupKFold(
        n_splits=split_count,
        shuffle=True,
        random_state=seed,
    )
    dummy = np.zeros((len(targets), 1), dtype=float)
    inner_folds = tuple(splitter.split(dummy, targets, groups))
    best_c = _C_VALUES[0]
    best_ap = float("-inf")
    best_scores = np.zeros(len(targets), dtype=float)
    for c_value in _C_VALUES:
        scores = np.zeros(len(targets), dtype=float)
        for fit_indices, validation_indices in inner_folds:
            estimator = _estimator(c_value, seed)
            estimator.fit(
                [feature_maps[int(index)] for index in fit_indices],
                targets[fit_indices],
            )
            scores[validation_indices] = estimator.predict_proba(
                [feature_maps[int(index)] for index in validation_indices]
            )[:, 1]
        average_precision = float(average_precision_score(targets, scores))
        if average_precision > best_ap:
            best_ap = average_precision
            best_c = c_value
            best_scores = scores
    calibrator = LogisticRegression(
        C=1.0,
        max_iter=20_000,
        random_state=seed,
    )
    calibrator.fit(best_scores.reshape(-1, 1), targets)
    calibrated = calibrator.predict_proba(best_scores.reshape(-1, 1))[:, 1]
    return best_c, calibrator, _best_threshold(targets, calibrated)


def nested_grouped_predictions(
    records: Sequence[Phase2CalibrationRecord],
    family: str,
    *,
    folds: FoldMaps | None = None,
    outer_repeats: int = 5,
    outer_splits: int = 5,
    inner_splits: int = 3,
    seed: int = 20260816,
) -> list[CalibratedPrediction]:
    """Generate repeated nested out-of-fold probabilities for every record."""
    if not records:
        raise ValueError("calibration population cannot be empty")
    fold_maps = folds or make_repeated_grouped_folds(
        records,
        repeats=outer_repeats,
        splits=outer_splits,
        seed=seed,
    )
    maps = _feature_maps(records, family)
    targets = np.asarray([record.target for record in records], dtype=int)
    groups = np.asarray([record.group for record in records], dtype=object)
    score_values: list[list[float]] = [[] for _ in records]
    decision_values: list[list[int]] = [[] for _ in records]
    c_values: list[list[float]] = [[] for _ in records]
    thresholds: list[list[float]] = [[] for _ in records]
    training_groups: list[list[tuple[str, ...]]] = [[] for _ in records]
    for repeat_index, repetition in enumerate(fold_maps):
        covered: set[int] = set()
        for train_tuple, held_tuple in repetition:
            train_indices = np.asarray(train_tuple, dtype=int)
            held_indices = np.asarray(held_tuple, dtype=int)
            if covered & set(held_tuple):
                raise ValueError("an outer repetition holds a case more than once")
            covered.update(held_tuple)
            train_group_values = tuple(sorted(set(str(group) for group in groups[train_indices])))
            held_group_values = set(str(group) for group in groups[held_indices])
            if set(train_group_values) & held_group_values:
                raise ValueError("outer fold leaks a held episode group")
            train_maps = [maps[int(index)] for index in train_indices]
            train_targets = targets[train_indices]
            train_groups_array = groups[train_indices]
            chosen_c, calibrator, threshold = _inner_selection(
                train_maps,
                train_targets,
                train_groups_array,
                inner_splits=inner_splits,
                seed=seed + repeat_index,
            )
            estimator = _estimator(chosen_c, seed + repeat_index)
            estimator.fit(train_maps, train_targets)
            raw_scores = estimator.predict_proba(
                [maps[int(index)] for index in held_indices]
            )[:, 1]
            held_scores = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
            for offset, held_index in enumerate(held_indices):
                index = int(held_index)
                score = float(held_scores[offset])
                score_values[index].append(score)
                decision_values[index].append(int(score >= threshold))
                c_values[index].append(float(chosen_c))
                thresholds[index].append(float(threshold))
                training_groups[index].append(train_group_values)
        if covered != set(range(len(records))):
            raise ValueError("outer repetition does not cover every case")
    predictions: list[CalibratedPrediction] = []
    repeat_count = len(fold_maps)
    for index, record in enumerate(records):
        if len(score_values[index]) != repeat_count:
            raise ValueError(f"case lacks repeated predictions: {record.vc_slug}/{record.episode_slug}")
        predictions.append(
            CalibratedPrediction(
                vc_slug=record.vc_slug,
                episode_slug=record.episode_slug,
                group=record.group,
                target=record.target,
                method=family,
                score=float(np.mean(score_values[index])),
                predicted=int(float(np.mean(decision_values[index])) >= 0.5),
                outer_repeats=repeat_count,
                selected_cs=tuple(c_values[index]),
                thresholds=tuple(thresholds[index]),
                outer_training_groups=tuple(training_groups[index]),
            )
        )
    return predictions


def raw_method_predictions(
    records: Sequence[Phase2CalibrationRecord], method: str
) -> list[MethodPrediction]:
    """Create aligned raw endpoints without fitting a model."""
    if method not in {"raw_phase2", "raw_likelihood_050"}:
        raise ValueError(f"unknown raw Phase 2 method: {method}")
    return [
        MethodPrediction(
            vc_slug=record.vc_slug,
            vc_name=record.vc_name,
            episode_slug=record.episode_slug,
            group=record.group,
            target=record.target,
            method=method,
            score=record.raw_likelihood,
            predicted=(
                record.raw_decision
                if method == "raw_phase2"
                else int(record.raw_likelihood >= 0.5)
            ),
        )
        for record in records
    ]


def learned_method_predictions(
    records: Sequence[Phase2CalibrationRecord],
    predictions: Sequence[CalibratedPrediction],
    method: str,
) -> list[MethodPrediction]:
    """Align calibrated predictions with display metadata from canonical records."""
    if len(records) != len(predictions):
        raise ValueError("records and calibrated predictions must align")
    rows: list[MethodPrediction] = []
    for record, prediction in zip(records, predictions, strict=True):
        if (
            record.vc_slug != prediction.vc_slug
            or record.episode_slug != prediction.episode_slug
            or record.target != prediction.target
        ):
            raise ValueError("calibrated prediction is not aligned to its record")
        rows.append(
            MethodPrediction(
                vc_slug=record.vc_slug,
                vc_name=record.vc_name,
                episode_slug=record.episode_slug,
                group=record.group,
                target=record.target,
                method=method,
                score=prediction.score,
                predicted=prediction.predicted,
            )
        )
    return rows


def _ece(targets: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(targets)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (scores >= lower) & (
            scores <= upper if index == bins - 1 else scores < upper
        )
        count = int(selected.sum())
        if count:
            error += count / total * abs(
                float(targets[selected].mean()) - float(scores[selected].mean())
            )
    return float(error)


def _classification_metrics(predictions: Sequence[MethodPrediction]) -> dict[str, Any]:
    targets = np.asarray([row.target for row in predictions], dtype=int)
    decided = np.asarray([row.predicted for row in predictions], dtype=int)
    scores = np.asarray([row.score for row in predictions], dtype=float)
    tn, fp, fn, tp = confusion_matrix(targets, decided, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    return {
        "n": len(predictions),
        "in_count": int(targets.sum()),
        "out_count": int(len(targets) - targets.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": float(accuracy_score(targets, decided)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, decided)),
        "in_precision": float(precision_score(targets, decided, zero_division=0)),
        "in_recall": float(recall_score(targets, decided, zero_division=0)),
        "in_f1": float(f1_score(targets, decided, zero_division=0)),
        "specificity": specificity,
        "matthews_correlation": float(matthews_corrcoef(targets, decided)),
        "brier_score": float(np.mean((scores - targets) ** 2)),
        "ece": _ece(targets, scores),
        "review_count": int(decided.sum()),
    }


def classification_metric_rows(
    predictions: Sequence[MethodPrediction],
) -> list[dict[str, Any]]:
    """Return per-investor, macro, and pooled classification diagnostics."""
    if not predictions:
        raise ValueError("classification predictions cannot be empty")
    method = predictions[0].method
    if any(row.method != method for row in predictions):
        raise ValueError("classification rows must contain one method")
    investors = sorted({row.vc_slug for row in predictions})
    rows: list[dict[str, Any]] = []
    for vc_slug in investors:
        subset = [row for row in predictions if row.vc_slug == vc_slug]
        rows.append(
            {
                "scope": "investor",
                "vc_slug": vc_slug,
                "vc_name": subset[0].vc_name,
                "method": method,
                **_classification_metrics(subset),
            }
        )
    averaged = (
        "accuracy",
        "balanced_accuracy",
        "in_precision",
        "in_recall",
        "in_f1",
        "specificity",
        "matthews_correlation",
        "brier_score",
        "ece",
    )
    macro: dict[str, Any] = {
        "scope": "macro",
        "vc_slug": "",
        "vc_name": "Macro average",
        "method": method,
        "n": "",
        "in_count": "",
        "out_count": "",
        "tp": "",
        "fp": "",
        "tn": "",
        "fn": "",
        "review_count": "",
    }
    for name in averaged:
        macro[name] = float(np.mean([float(row[name]) for row in rows]))
    rows.append(macro)
    rows.append(
        {
            "scope": "pooled_diagnostic",
            "vc_slug": "",
            "vc_name": "Pooled diagnostic",
            "method": method,
            **_classification_metrics(predictions),
        }
    )
    return rows


def _ranking_summary(predictions: Sequence[MethodPrediction]) -> dict[str, float]:
    targets = np.asarray([row.target for row in predictions], dtype=int)
    scores = np.asarray([row.score for row in predictions], dtype=float)
    return {
        "average_precision": float(average_precision_score(targets, scores)),
        "roc_auc": float(roc_auc_score(targets, scores)),
    }


def ranking_metric_rows(
    predictions: Sequence[MethodPrediction],
    *,
    review_budgets: Sequence[int] = (1, 3, 5, 10, 20),
) -> list[dict[str, Any]]:
    """Rank independently within each investor and return macro summaries."""
    if not predictions or not review_budgets:
        raise ValueError("ranking needs predictions and review budgets")
    method = predictions[0].method
    investors = sorted({row.vc_slug for row in predictions})
    rows: list[dict[str, Any]] = []
    for vc_slug in investors:
        subset = [row for row in predictions if row.vc_slug == vc_slug]
        ranked = sorted(subset, key=lambda row: (-row.score, row.episode_slug))
        positives = sum(row.target for row in subset)
        summary = _ranking_summary(subset)
        for requested in review_budgets:
            reviewed = min(int(requested), len(ranked))
            hits = sum(row.target for row in ranked[:reviewed])
            rows.append(
                {
                    "scope": "investor",
                    "vc_slug": vc_slug,
                    "vc_name": subset[0].vc_name,
                    "method": method,
                    "n": len(subset),
                    "in_count": positives,
                    "review_budget": int(requested),
                    "reviewed": reviewed,
                    "hits": int(hits),
                    "precision_at_k": hits / reviewed if reviewed else 0.0,
                    "recall_at_k": hits / positives if positives else 0.0,
                    **summary,
                }
            )
    for requested in review_budgets:
        budget_rows = [
            row
            for row in rows
            if row["scope"] == "investor" and row["review_budget"] == requested
        ]
        rows.append(
            {
                "scope": "macro",
                "vc_slug": "",
                "vc_name": "Macro average",
                "method": method,
                "n": "",
                "in_count": "",
                "review_budget": int(requested),
                "reviewed": "",
                "hits": "",
                "precision_at_k": float(
                    np.mean([row["precision_at_k"] for row in budget_rows])
                ),
                "recall_at_k": float(
                    np.mean([row["recall_at_k"] for row in budget_rows])
                ),
                "average_precision": float(
                    np.mean([row["average_precision"] for row in budget_rows])
                ),
                "roc_auc": float(np.mean([row["roc_auc"] for row in budget_rows])),
            }
        )
    return rows


_TRANSITIONS = (
    "preserved_correct_in",
    "lost_correct_in",
    "corrected_false_in",
    "remaining_false_in",
    "rescued_in",
    "remaining_missed_in",
    "preserved_correct_out",
    "introduced_false_in",
)


def _transition_name(raw: int, target: int, learned: int) -> str:
    return {
        (1, 1, 1): "preserved_correct_in",
        (1, 1, 0): "lost_correct_in",
        (1, 0, 0): "corrected_false_in",
        (1, 0, 1): "remaining_false_in",
        (0, 1, 1): "rescued_in",
        (0, 1, 0): "remaining_missed_in",
        (0, 0, 0): "preserved_correct_out",
        (0, 0, 1): "introduced_false_in",
    }[(raw, target, learned)]


def transition_metric_rows(
    records: Sequence[Phase2CalibrationRecord],
    learned: Sequence[MethodPrediction],
) -> list[dict[str, Any]]:
    """Account for every calibrated decision relative to raw Phase 2."""
    if len(records) != len(learned) or not records:
        raise ValueError("transition records and predictions must align")
    aligned: list[tuple[Phase2CalibrationRecord, MethodPrediction]] = []
    for record, prediction in zip(records, learned, strict=True):
        if (record.vc_slug, record.episode_slug, record.target) != (
            prediction.vc_slug,
            prediction.episode_slug,
            prediction.target,
        ):
            raise ValueError("transition prediction is not aligned")
        aligned.append((record, prediction))
    rows: list[dict[str, Any]] = []
    scopes = [("overall", "", aligned)] + [
        (
            "investor",
            vc_slug,
            [pair for pair in aligned if pair[0].vc_slug == vc_slug],
        )
        for vc_slug in sorted({record.vc_slug for record in records})
    ]
    for scope, vc_slug, pairs in scopes:
        counts = {name: 0 for name in _TRANSITIONS}
        for record, prediction in pairs:
            counts[_transition_name(record.raw_decision, record.target, prediction.predicted)] += 1
        rows.append(
            {
                "scope": scope,
                "vc_slug": vc_slug,
                "method": learned[0].method,
                "n": len(pairs),
                **counts,
            }
        )
    return rows


def _aligned_methods(
    first: Sequence[MethodPrediction], second: Sequence[MethodPrediction]
) -> None:
    if len(first) != len(second) or not first:
        raise ValueError("paired methods must be nonempty and equally sized")
    for left, right in zip(first, second, strict=True):
        if (left.vc_slug, left.episode_slug, left.target, left.group) != (
            right.vc_slug,
            right.episode_slug,
            right.target,
            right.group,
        ):
            raise ValueError("paired methods are not case-aligned")


def _macro_metric_arrays(
    targets: np.ndarray,
    decided: np.ndarray,
    scores: np.ndarray,
    vc_slugs: np.ndarray,
    metric: str,
) -> float:
    values: list[float] = []
    for vc_slug in np.unique(vc_slugs):
        selected = vc_slugs == vc_slug
        y = targets[selected]
        prediction = decided[selected]
        probability = scores[selected]
        positives = int(y.sum())
        negatives = int(len(y) - positives)
        if not positives or not negatives:
            raise ValueError("macro resample lacks both classes for an investor")
        if metric == "balanced_accuracy":
            true_positive = int(((y == 1) & (prediction == 1)).sum())
            true_negative = int(((y == 0) & (prediction == 0)).sum())
            value = (true_positive / positives + true_negative / negatives) / 2
        elif metric == "in_f1":
            true_positive = int(((y == 1) & (prediction == 1)).sum())
            false_positive = int(((y == 0) & (prediction == 1)).sum())
            false_negative = int(((y == 1) & (prediction == 0)).sum())
            denominator = 2 * true_positive + false_positive + false_negative
            value = 2 * true_positive / denominator if denominator else 0.0
        elif metric == "average_precision":
            order = np.argsort(-probability, kind="stable")
            ordered = y[order]
            ordered_scores = probability[order]
            cumulative_positives = np.cumsum(ordered)
            group_ends = np.r_[
                np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]),
                len(ordered_scores) - 1,
            ]
            true_positives = cumulative_positives[group_ends]
            precision = true_positives / (group_ends + 1)
            recall = true_positives / positives
            value = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
        elif metric == "roc_auc":
            ranks = rankdata(probability, method="average")
            positive_rank_sum = float(ranks[y == 1].sum())
            value = (
                positive_rank_sum - positives * (positives + 1) / 2
            ) / (positives * negatives)
        else:
            raise ValueError(f"unsupported macro metric: {metric}")
        values.append(float(value))
    return float(np.mean(values))


def _prediction_arrays(
    predictions: Sequence[MethodPrediction],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([row.target for row in predictions], dtype=int),
        np.asarray([row.predicted for row in predictions], dtype=int),
        np.asarray([row.score for row in predictions], dtype=float),
        np.asarray([row.vc_slug for row in predictions], dtype=object),
    )


def _macro_metric(predictions: Sequence[MethodPrediction], metric: str) -> float:
    return _macro_metric_arrays(*_prediction_arrays(predictions), metric)


def cluster_resample_indices(
    predictions: Sequence[MethodPrediction], rng: np.random.Generator
) -> list[int]:
    """Sample episode groups with replacement and retain every row in each group."""
    if not predictions:
        raise ValueError("cluster resampling needs predictions")
    by_group: dict[str, list[int]] = {}
    for index, row in enumerate(predictions):
        by_group.setdefault(row.group, []).append(index)
    groups = sorted(by_group)
    sampled = rng.choice(groups, size=len(groups), replace=True)
    return [index for group in sampled for index in by_group[str(group)]]


def cluster_bootstrap_intervals(
    methods: Mapping[str, Sequence[MethodPrediction]],
    *,
    raw_method: str,
    metrics: Sequence[str] = (
        "balanced_accuracy",
        "in_f1",
        "average_precision",
        "roc_auc",
    ),
    iterations: int = 5_000,
    seed: int = 20260816,
) -> list[dict[str, Any]]:
    """Return cluster-bootstrap method intervals and paired learned-minus-raw deltas."""
    if raw_method not in methods or iterations < 1:
        raise ValueError("bootstrap needs a raw method and positive iterations")
    names = list(methods)
    raw = list(methods[raw_method])
    for name in names:
        _aligned_methods(raw, methods[name])
    observed = {
        (name, metric): _macro_metric(methods[name], metric)
        for name in names
        for metric in metrics
    }
    samples: dict[tuple[str, str], list[float]] = {
        (name, metric): [] for name in names for metric in metrics
    }
    deltas: dict[tuple[str, str], list[float]] = {
        (name, metric): []
        for name in names
        if name != raw_method
        for metric in metrics
    }
    rng = np.random.default_rng(seed)
    arrays = {name: _prediction_arrays(methods[name]) for name in names}
    for _ in range(iterations):
        indices = cluster_resample_indices(raw, rng)
        try:
            values = {
                (name, metric): _macro_metric_arrays(
                    arrays[name][0][indices],
                    arrays[name][1][indices],
                    arrays[name][2][indices],
                    arrays[name][3][indices],
                    metric,
                )
                for name in names
                for metric in metrics
            }
        except ValueError:
            continue
        for key, value in values.items():
            samples[key].append(value)
        for name in names:
            if name == raw_method:
                continue
            for metric in metrics:
                deltas[(name, metric)].append(
                    values[(name, metric)] - values[(raw_method, metric)]
                )
    rows: list[dict[str, Any]] = []
    for name in names:
        for metric in metrics:
            values = np.asarray(samples[(name, metric)], dtype=float)
            if not len(values):
                raise ValueError("bootstrap produced no valid resamples")
            rows.append(
                {
                    "kind": "method_interval",
                    "method": name,
                    "reference_method": "",
                    "metric": metric,
                    "estimate": observed[(name, metric)],
                    "lower_95": float(np.quantile(values, 0.025)),
                    "upper_95": float(np.quantile(values, 0.975)),
                    "requested_iterations": iterations,
                    "valid_iterations": len(values),
                }
            )
    for name in names:
        if name == raw_method:
            continue
        for metric in metrics:
            values = np.asarray(deltas[(name, metric)], dtype=float)
            rows.append(
                {
                    "kind": "paired_delta",
                    "method": name,
                    "reference_method": raw_method,
                    "metric": metric,
                    "estimate": observed[(name, metric)]
                    - observed[(raw_method, metric)],
                    "lower_95": float(np.quantile(values, 0.025)),
                    "upper_95": float(np.quantile(values, 0.975)),
                    "requested_iterations": iterations,
                    "valid_iterations": len(values),
                }
            )
    return rows


def cluster_permutation_test(
    raw: Sequence[MethodPrediction],
    learned: Sequence[MethodPrediction],
    *,
    metric: str,
    iterations: int = 10_000,
    seed: int = 20260816,
) -> dict[str, Any]:
    """Paired randomization test that swaps complete episode clusters."""
    _aligned_methods(raw, learned)
    if iterations < 1:
        raise ValueError("permutation iterations must be positive")
    observed = _macro_metric(learned, metric) - _macro_metric(raw, metric)
    groups = sorted({row.group for row in raw})
    group_index = {group: index for index, group in enumerate(groups)}
    row_groups = np.asarray([group_index[row.group] for row in raw], dtype=int)
    targets, raw_decided, raw_scores, vc_slugs = _prediction_arrays(raw)
    _, learned_decided, learned_scores, _ = _prediction_arrays(learned)
    rng = np.random.default_rng(seed)
    extreme = 0
    valid = 0
    for _ in range(iterations):
        swap_rows = rng.integers(0, 2, size=len(groups), dtype=np.int8)[row_groups].astype(bool)
        permuted_raw_decided = np.where(swap_rows, learned_decided, raw_decided)
        permuted_learned_decided = np.where(swap_rows, raw_decided, learned_decided)
        permuted_raw_scores = np.where(swap_rows, learned_scores, raw_scores)
        permuted_learned_scores = np.where(swap_rows, raw_scores, learned_scores)
        try:
            delta = _macro_metric_arrays(
                targets,
                permuted_learned_decided,
                permuted_learned_scores,
                vc_slugs,
                metric,
            ) - _macro_metric_arrays(
                targets,
                permuted_raw_decided,
                permuted_raw_scores,
                vc_slugs,
                metric,
            )
        except ValueError:
            continue
        valid += 1
        extreme += int(abs(delta) >= abs(observed) - 1e-12)
    if not valid:
        raise ValueError("permutation test produced no valid iterations")
    return {
        "metric": metric,
        "raw_method": raw[0].method,
        "learned_method": learned[0].method,
        "observed_delta": observed,
        "p_value": (extreme + 1) / (valid + 1),
        "requested_iterations": iterations,
        "valid_iterations": valid,
        "cluster_count": len(groups),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Apply Holm family-wise correction while preserving input order."""
    if not p_values or any(not 0.0 <= float(value) <= 1.0 for value in p_values):
        raise ValueError("Holm adjustment needs p-values between zero and one")
    ordered = sorted(enumerate(float(value) for value in p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(ordered)
    running = 0.0
    total = len(ordered)
    for rank, (original_index, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[original_index] = running
    return adjusted


def mcnemar_exact(
    raw: Sequence[MethodPrediction], learned: Sequence[MethodPrediction]
) -> dict[str, Any]:
    """Secondary pooled exact McNemar diagnostic for aligned classifications."""
    _aligned_methods(raw, learned)
    raw_correct_learned_wrong = 0
    raw_wrong_learned_correct = 0
    for left, right in zip(raw, learned, strict=True):
        left_correct = left.predicted == left.target
        right_correct = right.predicted == right.target
        raw_correct_learned_wrong += int(left_correct and not right_correct)
        raw_wrong_learned_correct += int(not left_correct and right_correct)
    discordant = raw_correct_learned_wrong + raw_wrong_learned_correct
    p_value = (
        float(
            binomtest(
                min(raw_correct_learned_wrong, raw_wrong_learned_correct),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "raw_method": raw[0].method,
        "learned_method": learned[0].method,
        "raw_correct_learned_wrong": raw_correct_learned_wrong,
        "raw_wrong_learned_correct": raw_wrong_learned_correct,
        "discordant": discordant,
        "p_value": p_value,
    }


def run_phase2_analysis(
    records: Sequence[Phase2CalibrationRecord],
    *,
    outer_repeats: int = 5,
    outer_splits: int = 5,
    inner_splits: int = 3,
    bootstrap_iterations: int = 5_000,
    permutation_iterations: int = 10_000,
    seed: int = 20260816,
    review_budgets: Sequence[int] = (1, 3, 5, 10, 20),
) -> dict[str, Any]:
    """Run raw and nested calibrated endpoints over one canonical population."""
    from .phase2_advanced_models import (
        constrained_fusion_predictions,
        pairwise_grouped_predictions,
        top_k_rerank_scores,
        top_k_grouped_predictions,
    )

    if not records:
        raise ValueError("Phase 2 analysis population cannot be empty")
    folds = make_repeated_grouped_folds(
        records,
        repeats=outer_repeats,
        splits=outer_splits,
        seed=seed,
    )
    phase2_calibrated = nested_grouped_predictions(
        records,
        "phase2",
        folds=folds,
        inner_splits=inner_splits,
        seed=seed,
    )
    combined_calibrated = nested_grouped_predictions(
        records,
        "combined",
        folds=folds,
        inner_splits=inner_splits,
        seed=seed,
    )
    semantic_calibrated = nested_grouped_predictions(
        records,
        "semantic",
        folds=folds,
        inner_splits=inner_splits,
        seed=seed,
    )
    constrained_fusion = constrained_fusion_predictions(
        records,
        folds=folds,
        seed=seed,
    )
    pairwise_ranked = pairwise_grouped_predictions(
        records,
        folds=folds,
        inner_splits=inner_splits,
        seed=seed,
    )
    top_k_reranked = top_k_grouped_predictions(
        records,
        folds=folds,
        inner_splits=inner_splits,
        seed=seed,
    )
    methods: dict[str, list[MethodPrediction]] = {
        "raw_phase2": raw_method_predictions(records, "raw_phase2"),
        "raw_likelihood_050": raw_method_predictions(records, "raw_likelihood_050"),
        "phase2_calibrated": learned_method_predictions(
            records, phase2_calibrated, "phase2_calibrated"
        ),
        "rationale_plus_phase2_calibrated": learned_method_predictions(
            records,
            combined_calibrated,
            "rationale_plus_phase2_calibrated",
        ),
        "taxonomy_hierarchical_calibrated": learned_method_predictions(
            records,
            semantic_calibrated,
            "taxonomy_hierarchical_calibrated",
        ),
        "constrained_phase2_fusion": learned_method_predictions(
            records,
            constrained_fusion,
            "constrained_phase2_fusion",
        ),
        "within_vc_pairwise_ranker": learned_method_predictions(
            records,
            pairwise_ranked,
            "within_vc_pairwise_ranker",
        ),
        "training_selected_top_k_reranker": learned_method_predictions(
            records,
            top_k_reranked,
            "training_selected_top_k_reranker",
        ),
    }
    ranking_only: dict[str, list[MethodPrediction]] = {}
    for method_name, learned_rows in (
        (
            "fixed_raw_top20_pairwise_reranker",
            methods["within_vc_pairwise_ranker"],
        ),
        (
            "fixed_raw_top20_rationale_reranker",
            methods["rationale_plus_phase2_calibrated"],
        ),
    ):
        reranked_scores = top_k_rerank_scores(
            records,
            np.asarray([row.score for row in learned_rows], dtype=float),
            candidate_depth=20,
        )
        ranking_only[method_name] = [
            MethodPrediction(
                vc_slug=record.vc_slug,
                vc_name=record.vc_name,
                episode_slug=record.episode_slug,
                group=record.group,
                target=record.target,
                method=method_name,
                score=float(reranked_scores[index]),
                predicted=record.raw_decision,
            )
            for index, record in enumerate(records)
        ]
    ranking_methods = {**methods, **ranking_only}
    classification = [
        row
        for predictions in methods.values()
        for row in classification_metric_rows(predictions)
    ]
    ranking = [
        row
        for predictions in ranking_methods.values()
        for row in ranking_metric_rows(
            predictions, review_budgets=review_budgets
        )
    ]
    learned_names = (
        "phase2_calibrated",
        "rationale_plus_phase2_calibrated",
        "taxonomy_hierarchical_calibrated",
        "constrained_phase2_fusion",
        "within_vc_pairwise_ranker",
        "training_selected_top_k_reranker",
    )
    transitions = [
        row
        for name in learned_names
        for row in transition_metric_rows(records, methods[name])
    ]
    uncertainty = cluster_bootstrap_intervals(
        {
            "raw_phase2": methods["raw_phase2"],
            **{name: methods[name] for name in learned_names},
        },
        raw_method="raw_phase2",
        iterations=bootstrap_iterations,
        seed=seed,
    )
    uncertainty.extend(
        cluster_bootstrap_intervals(
            {
                "raw_phase2": methods["raw_phase2"],
                **ranking_only,
            },
            raw_method="raw_phase2",
            metrics=("average_precision",),
            iterations=bootstrap_iterations,
            seed=seed + 101,
        )
    )
    permutation_rows: list[dict[str, Any]] = []
    for offset, name in enumerate(learned_names):
        for metric_offset, metric in enumerate(
            ("balanced_accuracy", "average_precision")
        ):
            result = cluster_permutation_test(
                methods["raw_phase2"],
                methods[name],
                metric=metric,
                iterations=permutation_iterations,
                seed=seed + offset * 10 + metric_offset,
            )
            permutation_rows.append(
                {
                    "test": "episode_cluster_randomization",
                    **result,
                    "adjusted_p_value": None,
                    "significant_0_05": None,
                    "note": "primary paired test",
                }
            )
    for offset, name in enumerate(ranking_only, start=len(learned_names)):
        result = cluster_permutation_test(
            methods["raw_phase2"],
            ranking_only[name],
            metric="average_precision",
            iterations=permutation_iterations,
            seed=seed + offset * 10 + 1,
        )
        permutation_rows.append(
            {
                "test": "episode_cluster_randomization",
                **result,
                "adjusted_p_value": None,
                "significant_0_05": None,
                "note": "exploratory fixed top-20 ranking-only comparison",
            }
        )
    adjusted = holm_adjust([float(row["p_value"]) for row in permutation_rows])
    for row, adjusted_p in zip(permutation_rows, adjusted, strict=True):
        row["adjusted_p_value"] = adjusted_p
        row["significant_0_05"] = adjusted_p < 0.05
    mcnemar_rows = []
    for name in learned_names:
        result = mcnemar_exact(methods["raw_phase2"], methods[name])
        mcnemar_rows.append(
            {
                "test": "pooled_exact_mcnemar",
                "metric": "classification_accuracy",
                **result,
                "observed_delta": None,
                "adjusted_p_value": None,
                "significant_0_05": None,
                "requested_iterations": None,
                "valid_iterations": None,
                "cluster_count": len({record.group for record in records}),
                "note": "secondary diagnostic; rows sharing a pitch are dependent",
            }
        )
    fold_provenance = {
        "schema": "phase2-grouped-fold-provenance-v1",
        "group_key": "episode_slug",
        "repetitions": [
            [
                {
                    "train_groups": sorted({records[index].group for index in train}),
                    "held_groups": sorted({records[index].group for index in held}),
                    "train_cases": len(train),
                    "held_cases": len(held),
                }
                for train, held in repetition
            ]
            for repetition in folds
        ],
    }
    return {
        "schema": "phase2-all-vc-calibration-evaluation-v1",
        "scientific_status": "nested_grouped_development_evaluation",
        "parameters": {
            "outer_repeats": outer_repeats,
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
            "bootstrap_iterations": bootstrap_iterations,
            "permutation_iterations": permutation_iterations,
            "seed": seed,
            "regularization_grid": list(_C_VALUES),
            "review_budgets": list(review_budgets),
        },
        "methods": methods,
        "ranking_methods": ranking_methods,
        "calibrated_predictions": {
            "phase2": phase2_calibrated,
            "combined": combined_calibrated,
            "semantic": semantic_calibrated,
            "fusion": constrained_fusion,
            "pairwise": pairwise_ranked,
            "top_k": top_k_reranked,
        },
        "classification": classification,
        "ranking": ranking,
        "transitions": transitions,
        "uncertainty": uncertainty,
        "significance": [*permutation_rows, *mcnemar_rows],
        "fold_provenance": fold_provenance,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name, "")) for name in fieldnames})


def _fit_full_model(
    records: Sequence[Phase2CalibrationRecord],
    family: str,
    *,
    inner_splits: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    maps = _feature_maps(records, family)
    targets = np.asarray([record.target for record in records], dtype=int)
    groups = np.asarray([record.group for record in records], dtype=object)
    chosen_c, calibrator, threshold = _inner_selection(
        maps,
        targets,
        groups,
        inner_splits=inner_splits,
        seed=seed,
    )
    estimator = _estimator(chosen_c, seed)
    estimator.fit(maps, targets)
    vectorizer = estimator.named_steps["dictvectorizer"]
    bundle = {
        "schema": "phase2-full-data-deployment-model-v1",
        "family": family,
        "estimator": estimator,
        "probability_calibrator": calibrator,
        "threshold": threshold,
    }
    manifest = {
        "schema": bundle["schema"],
        "family": family,
        "training_cases": len(records),
        "training_episode_groups": len(set(groups)),
        "chosen_c": chosen_c,
        "threshold": threshold,
        "feature_count": len(vectorizer.get_feature_names_out()),
        "feature_names": list(vectorizer.get_feature_names_out()),
        "performance_estimate": False,
        "note": "Full-data fitted model for deployment; evaluation uses nested out-of-fold predictions.",
    }
    return bundle, manifest


def _render_phase2_report(
    records: Sequence[Phase2CalibrationRecord], result: Mapping[str, Any]
) -> str:
    methods: Mapping[str, Sequence[MethodPrediction]] = result["methods"]
    ranking_methods: Mapping[str, Sequence[MethodPrediction]] = result["ranking_methods"]
    classification = result["classification"]
    ranking = result["ranking"]
    uncertainty = result["uncertainty"]
    significance = result["significance"]
    population_ins = sum(record.target for record in records)
    names = {
        "raw_phase2": "Raw Phase 2 decision",
        "raw_likelihood_050": "Raw likelihood >= 0.50",
        "phase2_calibrated": "Phase 2 calibrated",
        "rationale_plus_phase2_calibrated": "Rationale + Phase 2 calibrated",
        "taxonomy_hierarchical_calibrated": "Taxonomy-aware hierarchical classifier",
        "constrained_phase2_fusion": "Constrained Phase 2 fusion",
        "within_vc_pairwise_ranker": "Within-VC pairwise ranker",
        "training_selected_top_k_reranker": "Training-selected top-K reranker",
        "fixed_raw_top20_pairwise_reranker": "Fixed raw top-20 / pairwise reranker",
        "fixed_raw_top20_rationale_reranker": "Fixed raw top-20 / rationale reranker",
    }
    lines = [
        "# All-VC Phase 2 Raw-vs-Calibrated Evaluation",
        "",
        "> **Scientific status:** repeated nested episode-grouped development evaluation; this is not an untouched holdout estimate.",
        "",
        "## Population and protocol",
        "",
        f"The analysis covers **{len(records)} investor-pitch cases**, **{population_ins} Ins**, "
        f"**{len(records) - population_ins} Outs**, **{len(set(record.vc_slug for record in records))} investors**, "
        f"and **{len(set(record.group for record in records))} distinct pitch episodes**.",
        "",
        "Learned predictions use repeated nested cross-validation. Every occurrence of the same pitch is held out across all investors together. Regularization, probability calibration, and classification thresholds are selected using outer-training data only.",
        "",
        "Full-data fitted models are deployment artifacts, not performance estimates.",
        "",
        "## Headline classification",
        "",
        "| Method | Balanced accuracy | In precision | In recall | In F1 | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    macro_classification: dict[str, Mapping[str, Any]] = {}
    for method in methods:
        row = next(
            item
            for item in classification
            if item["method"] == method and item["scope"] == "macro"
        )
        macro_classification[method] = row
        lines.append(
            f"| {names[method]} | {row['balanced_accuracy']:.3f} | "
            f"{row['in_precision']:.3f} | {row['in_recall']:.3f} | "
            f"{row['in_f1']:.3f} | {row['brier_score']:.3f} | {row['ece']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Classification headline values are unweighted macro averages across the six investors.",
            "",
            "### Per-investor balanced accuracy and In F1",
            "",
            "| Investor | Method | Balanced accuracy | In F1 | TP | FP | TN | FN |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in classification:
        if row["scope"] != "investor":
            continue
        lines.append(
            f"| {row['vc_name']} | {names[row['method']]} | "
            f"{row['balanced_accuracy']:.3f} | {row['in_f1']:.3f} | "
            f"{row['tp']} | {row['fp']} | {row['tn']} | {row['fn']} |"
        )
    lines.extend(
        [
            "",
            "## Headline ranking",
            "",
            "Ranking is computed independently within each investor; pooled ranking is not used as a headline.",
            "",
            "| Method | Macro AP | Macro ROC AUC | Top-5 precision | Top-5 recall | Top-10 precision | Top-10 recall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    macro_ranking: dict[str, Mapping[str, Any]] = {}
    for method in ranking_methods:
        method_rows = [
            item
            for item in ranking
            if item["method"] == method and item["scope"] == "macro"
        ]
        row5 = next((item for item in method_rows if item["review_budget"] == 5), method_rows[0])
        row10 = next((item for item in method_rows if item["review_budget"] == 10), method_rows[-1])
        macro_ranking[method] = row5
        lines.append(
            f"| {names[method]} | {row5['average_precision']:.3f} | "
            f"{row5['roc_auc']:.3f} | {row5['precision_at_k']:.3f} | "
            f"{row5['recall_at_k']:.3f} | {row10['precision_at_k']:.3f} | "
            f"{row10['recall_at_k']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### VCs improved relative to raw Phase 2",
            "",
            "| Method | Balanced accuracy | Average precision | Precision@5 | Recall@5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    investor_slugs = sorted({record.vc_slug for record in records})
    raw_class_by_vc = {
        row["vc_slug"]: row
        for row in classification
        if row["scope"] == "investor" and row["method"] == "raw_phase2"
    }
    raw_rank_by_vc = {
        row["vc_slug"]: row
        for row in ranking
        if row["scope"] == "investor"
        and row["method"] == "raw_phase2"
        and row["review_budget"] == 5
    }
    for method in ranking_methods:
        if method in {"raw_phase2", "raw_likelihood_050"}:
            continue
        class_by_vc = {
            row["vc_slug"]: row
            for row in classification
            if row["scope"] == "investor" and row["method"] == method
        }
        rank_by_vc = {
            row["vc_slug"]: row
            for row in ranking
            if row["scope"] == "investor"
            and row["method"] == method
            and row["review_budget"] == 5
        }
        ba_count = (
            sum(
                class_by_vc[vc]["balanced_accuracy"]
                > raw_class_by_vc[vc]["balanced_accuracy"]
                for vc in investor_slugs
            )
            if class_by_vc
            else None
        )
        counts = {
            "ap": sum(
                rank_by_vc[vc]["average_precision"]
                > raw_rank_by_vc[vc]["average_precision"]
                for vc in investor_slugs
            ),
            "p5": sum(
                rank_by_vc[vc]["precision_at_k"]
                > raw_rank_by_vc[vc]["precision_at_k"]
                for vc in investor_slugs
            ),
            "r5": sum(
                rank_by_vc[vc]["recall_at_k"]
                > raw_rank_by_vc[vc]["recall_at_k"]
                for vc in investor_slugs
            ),
        }
        lines.append(
            f"| {names[method]} | "
            f"{f'{ba_count}/{len(investor_slugs)}' if ba_count is not None else 'n/a'} | "
            f"{counts['ap']}/{len(investor_slugs)} | {counts['p5']}/{len(investor_slugs)} | "
            f"{counts['r5']}/{len(investor_slugs)} |"
        )
    lines.extend(
        [
            "",
            "## Decision transitions relative to raw Phase 2",
            "",
            "| Learned method | Preserved correct Ins | Lost correct Ins | Corrected false Ins | Remaining false Ins | Rescued Ins | Remaining missed Ins | Preserved correct Outs | Introduced false Ins |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["transitions"]:
        if row["scope"] != "overall":
            continue
        lines.append(
            f"| {names[row['method']]} | {row['preserved_correct_in']} | "
            f"{row['lost_correct_in']} | {row['corrected_false_in']} | "
            f"{row['remaining_false_in']} | {row['rescued_in']} | "
            f"{row['remaining_missed_in']} | {row['preserved_correct_out']} | "
            f"{row['introduced_false_in']} |"
        )
    lines.extend(
        [
            "",
            "## Paired uncertainty and significance",
            "",
            "Primary p-values use paired episode-cluster randomization and are Holm-adjusted across all learned methods and both primary endpoints.",
            "",
            "| Learned method | Metric | Delta vs raw | Bootstrap 95% CI | Raw p | Holm p | Significant |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    delta_lookup = {
        (row["method"], row["metric"]): row
        for row in uncertainty
        if row["kind"] == "paired_delta"
    }
    primary_tests = [
        row for row in significance if row["test"] == "episode_cluster_randomization"
    ]
    for row in primary_tests:
        delta = delta_lookup[(row["learned_method"], row["metric"])]
        lines.append(
            f"| {names[row['learned_method']]} | {row['metric']} | "
            f"{row['observed_delta']:+.3f} | [{delta['lower_95']:+.3f}, {delta['upper_95']:+.3f}] | "
            f"{row['p_value']:.4f} | {row['adjusted_p_value']:.4f} | "
            f"{'yes' if row['significant_0_05'] else 'no'} |"
        )
    raw_ba = float(macro_classification["raw_phase2"]["balanced_accuracy"])
    raw_ap = float(macro_ranking["raw_phase2"]["average_precision"])
    best_ba_name = max(methods, key=lambda name: float(macro_classification[name]["balanced_accuracy"]))
    best_ap_name = max(ranking_methods, key=lambda name: float(macro_ranking[name]["average_precision"]))
    significant_rows = [row for row in primary_tests if row["significant_0_05"]]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The raw agent's macro balanced accuracy is **{raw_ba:.3f}** and macro average precision is **{raw_ap:.3f}**. "
            f"The highest classification point estimate is **{names[best_ba_name]}** at "
            f"**{float(macro_classification[best_ba_name]['balanced_accuracy']):.3f}**; "
            f"the highest ranking point estimate is **{names[best_ap_name]}** at "
            f"**{float(macro_ranking[best_ap_name]['average_precision']):.3f}**.",
            "",
            (
                f"{len(significant_rows)} of the {len(primary_tests)} prespecified paired comparisons remain significant after Holm correction."
                if significant_rows
                else f"None of the {len(primary_tests)} prespecified learned-versus-raw comparisons is statistically significant after Holm correction."
            ),
            "",
            "A calibrated point estimate above raw Phase 2 indicates that the structured agent output contains information not well expressed by its direct threshold. It does not establish future-pitch generalization because the models and comparisons were developed on this canonical cohort.",
            "",
            "The fixed top-20 methods are ranking-only exploratory analyses using an operational review budget that was already present in the evaluation. They preserve the raw classification decision and must be locked before any future holdout evaluation.",
            "",
            "Exact pooled McNemar tests are exported only as secondary diagnostics because investor-cases sharing a pitch are dependent.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase2_analysis(
    records: Sequence[Phase2CalibrationRecord],
    result: Mapping[str, Any],
    output_root: Path,
    *,
    registry_path: Path,
) -> dict[str, Path]:
    """Write report tables, provenance, and full-data deployment model bundles."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    methods: Mapping[str, Sequence[MethodPrediction]] = result["methods"]
    ranking_methods: Mapping[str, Sequence[MethodPrediction]] = result["ranking_methods"]
    calibrated: Mapping[str, Sequence[CalibratedPrediction]] = result[
        "calibrated_predictions"
    ]
    learned_lookup = {
        "phase2_calibrated": calibrated["phase2"],
        "rationale_plus_phase2_calibrated": calibrated["combined"],
        "taxonomy_hierarchical_calibrated": calibrated["semantic"],
        "constrained_phase2_fusion": calibrated["fusion"],
        "within_vc_pairwise_ranker": calibrated["pairwise"],
        "training_selected_top_k_reranker": calibrated["top_k"],
        "fixed_raw_top20_pairwise_reranker": calibrated["pairwise"],
        "fixed_raw_top20_rationale_reranker": calibrated["combined"],
    }
    prediction_rows: list[dict[str, Any]] = []
    for method, predictions in ranking_methods.items():
        provenance = learned_lookup.get(method)
        for index, row in enumerate(predictions):
            learned = provenance[index] if provenance is not None else None
            prediction_rows.append(
                {
                    "method": method,
                    "vc_slug": row.vc_slug,
                    "vc_name": row.vc_name,
                    "episode_slug": row.episode_slug,
                    "group": row.group,
                    "actual_decision": "In" if row.target else "Out",
                    "target": row.target,
                    "predicted_decision": "In" if row.predicted else "Out",
                    "predicted": row.predicted,
                    "score": row.score,
                    "selected_cs": getattr(learned, "selected_cs", ()) if learned else (),
                    "selected_weights": getattr(learned, "selected_weights", ()) if learned else (),
                    "selected_parameters": getattr(learned, "selected_parameters", ()) if learned else (),
                    "thresholds": learned.thresholds if learned else (),
                    "outer_repeats": learned.outer_repeats if learned else 0,
                }
            )
    paths: dict[str, Path] = {
        "evaluation": output / "evaluation.md",
        "predictions": output / "predictions.csv",
        "classification": output / "classification_metrics.csv",
        "ranking": output / "ranking_metrics.csv",
        "transitions": output / "transitions.csv",
        "uncertainty": output / "uncertainty.csv",
        "significance": output / "significance.csv",
        "fold_provenance": output / "fold_provenance.json",
        "manifest": output / "manifest.json",
    }
    _write_csv(paths["predictions"], prediction_rows)
    _write_csv(paths["classification"], result["classification"])
    _write_csv(paths["ranking"], result["ranking"])
    _write_csv(paths["transitions"], result["transitions"])
    _write_csv(paths["uncertainty"], result["uncertainty"])
    _write_csv(paths["significance"], result["significance"])
    _write_json(paths["fold_provenance"], result["fold_provenance"])
    paths["evaluation"].write_text(
        _render_phase2_report(records, result), encoding="utf-8"
    )
    for family in ("phase2", "combined", "semantic"):
        bundle, model_manifest = _fit_full_model(
            records,
            family,
            inner_splits=int(result["parameters"]["inner_splits"]),
            seed=int(result["parameters"]["seed"]),
        )
        model_dir = output / "models" / family
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.joblib"
        model_manifest_path = model_dir / "manifest.json"
        joblib.dump(bundle, model_path)
        model_manifest["model_sha256"] = sha256(model_path.read_bytes()).hexdigest()
        _write_json(model_manifest_path, model_manifest)
        paths[f"model_{family}"] = model_path
        paths[f"model_manifest_{family}"] = model_manifest_path
    registry = Path(registry_path)
    population = {
        "cases": len(records),
        "investors": len(set(record.vc_slug for record in records)),
        "ins": sum(record.target for record in records),
        "outs": len(records) - sum(record.target for record in records),
        "episode_groups": len(set(record.group for record in records)),
    }
    manifest = {
        "schema": result["schema"],
        "scientific_status": result["scientific_status"],
        "population": population,
        "parameters": result["parameters"],
        "registry_path": str(registry.resolve()),
        "registry_sha256": sha256(registry.read_bytes()).hexdigest()
        if registry.is_file()
        else None,
        "api_cost_usd": 0.0,
        "artifact_hashes": {
            f"{record.vc_slug}/{record.episode_slug}": {
                "phase1": record.phase1_sha256,
                "phase2": record.phase2_sha256,
            }
            for record in records
        },
        "outputs": sorted(str(path.relative_to(output)) for path in paths.values()),
        "notes": [
            "All evaluation predictions are nested out of fold.",
            "All occurrences of an episode are held out together across investors.",
            "Full-data models are deployment artifacts and are not used as performance estimates.",
            "No LLM or network calls were made.",
        ],
    }
    _write_json(paths["manifest"], manifest)
    return paths
