"""Leakage-controlled calibration over frozen rationale and decision artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import read_verified_frozen, write_json
from .schemas import DecisionV2, DecisionV3, InvestigationV2, InvestigationV3


FeatureMap = dict[str, float]

_DIRECTION_WEIGHT = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
_DIRECTIONS = tuple(_DIRECTION_WEIGHT)
_SALIENCES = ("primary", "secondary")
FEATURE_FAMILIES = (
    "legacy_phase1",
    "semantic_phase1",
    "phase2",
    "legacy_plus_phase2",
    "semantic_plus_phase2",
)


def _token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "unknown"


def _increment(features: FeatureMap, key: str, value: float = 1.0) -> None:
    features[key] = features.get(key, 0.0) + float(value)


def extract_legacy_phase1(investigation: Mapping[str, Any]) -> FeatureMap:
    """Map the v2 investigation onto the original rationale-core aggregates."""
    features: FeatureMap = {
        "rationale_count": 0.0,
        "unanswered_question_count": float(
            len(investigation.get("unanswered_questions", []))
        ),
        "conflict_count": float(len(investigation.get("conflicts", []))),
    }
    for direction in _DIRECTIONS:
        for salience in _SALIENCES:
            features[f"{direction}_{salience}_count"] = 0.0
            features[f"{direction}_{salience}_confidence_sum"] = 0.0

    for rationale in investigation.get("rationales", []):
        direction = str(rationale.get("direction"))
        salience = str(rationale.get("salience"))
        if direction not in _DIRECTION_WEIGHT:
            raise ValueError(f"invalid rationale direction: {direction!r}")
        if salience not in _SALIENCES:
            raise ValueError(f"invalid rationale salience: {salience!r}")
        confidence = float(rationale.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"invalid rationale confidence: {confidence!r}")
        features["rationale_count"] += 1.0
        features[f"{direction}_{salience}_count"] += 1.0
        features[f"{direction}_{salience}_confidence_sum"] += confidence

    for salience in _SALIENCES:
        features[f"{salience}_signed_confidence_sum"] = sum(
            _DIRECTION_WEIGHT[direction]
            * features[f"{direction}_{salience}_confidence_sum"]
            for direction in _DIRECTIONS
        )
    return features


def extract_semantic_phase1(investigation: Mapping[str, Any]) -> FeatureMap:
    """Extend the legacy core with v2 taxonomy and evidence semantics."""
    features = extract_legacy_phase1(investigation)
    for rationale in investigation.get("rationales", []):
        direction = str(rationale["direction"])
        salience = str(rationale["salience"])
        confidence = float(rationale["confidence"])
        label = _token(rationale["label"])
        evidence_status = _token(rationale["evidence_status"])
        constraint_kind = _token(rationale["constraint_kind"])
        severity = _token(rationale["constraint_severity"])

        _increment(
            features,
            f"label__{label}__signed_confidence",
            _DIRECTION_WEIGHT[direction] * confidence,
        )
        _increment(features, f"label_direction__{label}__{direction}__count")
        _increment(features, f"evidence__{evidence_status}__count")
        _increment(
            features,
            f"evidence__{evidence_status}__confidence_sum",
            confidence,
        )
        _increment(features, f"constraint__{constraint_kind}__count")
        _increment(
            features,
            f"constraint__{constraint_kind}__confidence_sum",
            confidence,
        )
        _increment(features, f"severity__{severity}__count")
        _increment(features, f"severity__{severity}__confidence_sum", confidence)
        _increment(
            features,
            f"severity_salience__{severity}__{salience}__count",
        )

    deal_context = investigation.get("deal_context", {})
    for fact_name, fact in deal_context.items():
        status = _token(fact.get("status", "unknown"))
        features[f"deal__{_token(fact_name)}__{status}"] = 1.0
    return features


def _finding_category(finding: object) -> str:
    text = str(finding)
    if text.startswith("NORMALIZED_UNTYPED_CONSTRAINT:"):
        return "normalized_untyped_constraint"
    return _token(text)


def extract_phase2(
    decision: Mapping[str, Any],
    *,
    phase1_status: str,
    phase2_status: str,
    phase1_findings: Sequence[str],
    phase2_findings: Sequence[str],
) -> FeatureMap:
    """Extract compact numeric Phase 2 measurements without using prose."""
    any_check = decision["any_check"]
    standard = decision["standard_check"]
    any_in = float(any_check["decision"] == "In")
    standard_in = float(standard["decision"] == "In")
    any_confidence = float(any_check["confidence"])
    standard_confidence = float(standard["confidence"])
    any_likelihood = float(any_check["likelihood"])
    standard_likelihood = float(standard["likelihood"])
    features: FeatureMap = {
        "any_check_in": any_in,
        "any_check_likelihood": any_likelihood,
        "any_check_confidence": any_confidence,
        "any_check_signed_confidence": any_confidence * (1.0 if any_in else -1.0),
        "standard_check_in": standard_in,
        "standard_check_likelihood": standard_likelihood,
        "standard_check_confidence": standard_confidence,
        "standard_check_signed_confidence": standard_confidence
        * (1.0 if standard_in else -1.0),
        "likelihood_gap": any_likelihood - standard_likelihood,
        "ranking_score": float(decision["ranking_score"]),
        "top_level_decision_in": float(decision.get("decision") == "In"),
        "top_level_confidence": float(decision.get("decision_confidence", 0.0)),
        "supporting_any_count": float(
            len(any_check.get("supporting_rationale_ids", []))
        ),
        "opposing_any_count": float(len(any_check.get("opposing_rationale_ids", []))),
        "supporting_standard_count": float(
            len(standard.get("supporting_rationale_ids", []))
        ),
        "opposing_standard_count": float(
            len(standard.get("opposing_rationale_ids", []))
        ),
        "failure_standard_count": float(
            len(standard.get("failure_rationale_ids", []))
        ),
        "controlling_count": float(
            len(decision.get("controlling_rationale_ids", []))
        ),
        "reversal_condition_count": float(
            len(decision.get("reversal_conditions", []))
        ),
        "phase1_provisional": float(phase1_status == "provisional"),
        "phase2_provisional": float(phase2_status == "provisional"),
        "phase1_finding_count": float(len(phase1_findings)),
        "phase2_finding_count": float(len(phase2_findings)),
    }
    tier = decision.get("recommended_check_tier", decision.get("check_tier", "unknown"))
    features[f"check_tier__{_token(tier)}"] = 1.0
    market_gate = standard.get("market_gate", {})
    features[f"market_gate__{_token(market_gate.get('status', 'unknown'))}"] = 1.0

    for entry in decision.get("risk_ledger", []):
        risk_type = _token(entry.get("risk_type", "unknown"))
        controlling = bool(entry.get("controlling_for_any_check", False))
        _increment(features, f"risk__{risk_type}__count")
        if controlling:
            _increment(features, f"risk__{risk_type}__controlling_count")
            _increment(features, "risk__all__controlling_count")

    for assessment in decision.get("rationale_assessments", []):
        direction = _token(assessment.get("effective_direction", "unknown"))
        weight = _token(assessment.get("decision_weight", "unknown"))
        _increment(features, f"assessment__{direction}__count")
        _increment(features, f"assessment_weight__{weight}__count")
        _increment(features, f"assessment__{direction}__{weight}_count")

    for phase, findings in (("phase1", phase1_findings), ("phase2", phase2_findings)):
        for finding in findings:
            _increment(
                features,
                f"quality__{phase}__{_finding_category(finding)}__count",
            )
    return features


@dataclass(frozen=True)
class CalibrationRecord:
    """One verified episode and its label-blind feature families."""

    episode_slug: str
    target: int
    target_decision: str
    artifact_root: str
    phase1_sha256: str
    phase2_sha256: str
    features: Mapping[str, FeatureMap]


def _combined_features(phase1: FeatureMap, phase2: FeatureMap) -> FeatureMap:
    return {
        **{f"p1__{name}": value for name, value in phase1.items()},
        **{f"p2__{name}": value for name, value in phase2.items()},
    }


def _slug_sort_key(slug: str) -> tuple[bool, int, str]:
    prefix = slug.split("-", 1)[0]
    return (not prefix.isdigit(), int(prefix) if prefix.isdigit() else 0, slug)


def load_frozen_population(
    status_paths: Sequence[Path],
    labels_path: Path,
    *,
    label_overrides: Mapping[str, str] | None = None,
) -> list[CalibrationRecord]:
    """Join completed batch records to audited labels and verify both phases."""
    labels_raw = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    if type(labels_raw) is not list:
        raise ValueError("label file must contain a list")
    labels: dict[str, str] = {}
    for row in labels_raw:
        if row.get("evaluation_eligible") is not True:
            continue
        slug = str(row.get("episode_slug", ""))
        target = row.get("pitch_window_decision")
        if not slug or target not in {"In", "Out"}:
            raise ValueError("eligible label row is invalid")
        if slug in labels:
            raise ValueError(f"duplicate eligible label: {slug}")
        labels[slug] = str(target)

    overrides = dict(label_overrides or {})
    if set(overrides) - set(labels):
        raise ValueError("label override is outside the eligible population")
    if any(value not in {"In", "Out"} for value in overrides.values()):
        raise ValueError("label override is not binary")

    records: list[CalibrationRecord] = []
    seen: set[str] = set()
    for status_path in status_paths:
        status = json.loads(Path(status_path).read_text(encoding="utf-8"))
        progress = status.get("progress", {})
        if progress.get("processed") != progress.get("total"):
            raise ValueError(f"batch is incomplete: {status_path}")
        batch_records = status.get("completed_records")
        if type(batch_records) is not list:
            raise ValueError(f"batch has no completed records: {status_path}")
        for record in batch_records:
            slug = str(record.get("episode_slug", ""))
            if not slug or slug in seen:
                raise ValueError(f"duplicate or missing episode slug: {slug!r}")
            if record.get("status") != "completed":
                raise ValueError(f"batch record is not completed: {slug}")
            if slug not in labels:
                raise ValueError(f"completed episode has no eligible label: {slug}")
            seen.add(slug)

            artifact_root = Path(record["artifact_root"])
            investigation_raw, investigation_digest = read_verified_frozen(
                artifact_root / "phase1/investigation.json",
                artifact_root / "phase1/investigation.sha256",
            )
            decision_raw, decision_digest = read_verified_frozen(
                artifact_root / "phase2/decision.json",
                artifact_root / "phase2/decision.sha256",
            )
            investigation_payload = json.loads(investigation_raw)
            investigation_schema = investigation_payload.get("schema_version")
            decision_schema = json.loads(decision_raw).get("schema_version")
            investigation_model = {
                "investigation-v2": InvestigationV2,
                "investigation-v3": InvestigationV3,
            }.get(investigation_schema)
            decision_model = {
                "decision-v2": DecisionV2,
                "decision-v3": DecisionV3,
            }.get(decision_schema)
            if investigation_model is None or decision_model is None:
                raise ValueError(f"unsupported frozen artifact schema: {slug}")
            if (
                investigation_schema == "investigation-v3"
                and "taxonomy_dispositions" not in investigation_payload
            ):
                investigation_payload["taxonomy_dispositions"] = [
                    {
                        "label": label,
                        "disposition": disposition,
                        "basis": (
                            "Reconstructed from the frozen legacy v3 "
                            f"{disposition}_candidates list."
                        ),
                    }
                    for disposition, field_name in (
                        ("activated", "activated_candidates"),
                        ("rejected", "rejected_candidates"),
                    )
                    for label in investigation_payload.get(field_name, [])
                ]
            investigation = investigation_model.model_validate(investigation_payload)
            decision = decision_model.model_validate_json(decision_raw)
            if investigation.episode_slug != slug or decision.episode_slug != slug:
                raise ValueError(f"artifact episode mismatch: {slug}")
            if decision.investigation_sha256 != investigation_digest:
                raise ValueError(f"decision Phase 1 digest mismatch: {slug}")

            investigation_dict = investigation.model_dump(mode="json")
            decision_dict = decision.model_dump(mode="json")
            legacy = extract_legacy_phase1(investigation_dict)
            semantic = extract_semantic_phase1(investigation_dict)
            phase2 = extract_phase2(
                decision_dict,
                phase1_status=str(record.get("phase1_status", "unknown")),
                phase2_status=str(record.get("phase2_status", "unknown")),
                phase1_findings=record.get("phase1_findings", []),
                phase2_findings=record.get("phase2_findings", []),
            )
            target_decision = overrides.get(slug, labels[slug])
            records.append(
                CalibrationRecord(
                    episode_slug=slug,
                    target=int(target_decision == "In"),
                    target_decision=target_decision,
                    artifact_root=str(artifact_root),
                    phase1_sha256=investigation_digest,
                    phase2_sha256=decision_digest,
                    features={
                        "legacy_phase1": legacy,
                        "semantic_phase1": semantic,
                        "phase2": phase2,
                        "legacy_plus_phase2": _combined_features(legacy, phase2),
                        "semantic_plus_phase2": _combined_features(semantic, phase2),
                    },
                )
            )

    missing = set(labels) - seen
    if missing:
        raise ValueError("eligible episodes missing from batches: " + ", ".join(sorted(missing)))
    return sorted(records, key=lambda record: _slug_sort_key(record.episode_slug))


@dataclass(frozen=True)
class CalibrationPrediction:
    """One outer-fold probability and its fully training-derived decision rule."""

    episode_slug: str
    target: int
    score: float
    predicted: int
    threshold: float
    chosen_c: float
    feature_family: str
    training_slugs: tuple[str, ...]


def feature_names(
    records: Sequence[CalibrationRecord], feature_family: str
) -> tuple[str, ...]:
    if not records or any(feature_family not in record.features for record in records):
        raise ValueError(f"unknown or empty feature family: {feature_family}")
    return tuple(
        sorted(
            {
                name
                for record in records
                for name in record.features[feature_family]
            }
        )
    )


def _matrix(
    records: Sequence[CalibrationRecord],
    feature_family: str,
    names: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [
            [float(record.features[feature_family].get(name, 0.0)) for name in names]
            for record in records
        ],
        dtype=float,
    )


def _model(c_value: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=10_000,
            random_state=0,
        ),
    )


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidates = sorted({0.0, 0.5, 1.0, *(float(value) for value in scores)})
    best_score = -1.0
    best_threshold = 0.5
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        score = float(balanced_accuracy_score(y_true, predicted))
        if score > best_score or (
            np.isclose(score, best_score) and threshold > best_threshold
        ):
            best_score = score
            best_threshold = threshold
    return best_threshold


def _tune_training_fold(
    x_train: np.ndarray, y_train: np.ndarray
) -> tuple[float, float]:
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    if not positive_count or not negative_count:
        raise ValueError("each training fold must contain both classes")
    split_count = min(3, positive_count, negative_count)
    if split_count < 2:
        return 1.0, 0.5

    splitter = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=0)
    best_c = 0.01
    best_ap = -1.0
    oof_by_c: dict[float, np.ndarray] = {}
    for c_value in (0.01, 0.1, 1.0, 10.0):
        oof = np.zeros(len(y_train), dtype=float)
        for inner_train, inner_validation in splitter.split(x_train, y_train):
            model = _model(c_value)
            model.fit(x_train[inner_train], y_train[inner_train])
            oof[inner_validation] = model.predict_proba(
                x_train[inner_validation]
            )[:, 1]
        oof_by_c[c_value] = oof
        average_precision = float(average_precision_score(y_train, oof))
        if average_precision > best_ap:
            best_ap = average_precision
            best_c = c_value
    return best_c, _best_threshold(y_train, oof_by_c[best_c])


def nested_leave_one_out(
    records: Sequence[CalibrationRecord], feature_family: str
) -> list[CalibrationPrediction]:
    """Generate probabilities with model and threshold selection inside each fold."""
    if len(records) < 3:
        raise ValueError("leave-one-out evaluation needs at least three records")
    names = feature_names(records, feature_family)
    x_all = _matrix(records, feature_family, names)
    y_all = np.asarray([record.target for record in records], dtype=int)
    predictions: list[CalibrationPrediction] = []
    for held_out_index, held_out in enumerate(records):
        training_indices = np.asarray(
            [index for index in range(len(records)) if index != held_out_index],
            dtype=int,
        )
        x_train = x_all[training_indices]
        y_train = y_all[training_indices]
        chosen_c, threshold = _tune_training_fold(x_train, y_train)
        model = _model(chosen_c)
        model.fit(x_train, y_train)
        score = float(model.predict_proba(x_all[[held_out_index]])[0, 1])
        predictions.append(
            CalibrationPrediction(
                episode_slug=held_out.episode_slug,
                target=int(held_out.target),
                score=score,
                predicted=int(score >= threshold),
                threshold=threshold,
                chosen_c=chosen_c,
                feature_family=feature_family,
                training_slugs=tuple(
                    records[index].episode_slug for index in training_indices
                ),
            )
        )
    return predictions


def evaluate_predictions(
    predictions: Sequence[CalibrationPrediction],
    *,
    review_budgets: Sequence[int] = (5, 10, 20, 30, 40),
) -> dict[str, Any]:
    """Compute imbalanced classification and diamond-search metrics."""
    if not predictions:
        raise ValueError("predictions cannot be empty")
    true = np.asarray([prediction.target for prediction in predictions], dtype=int)
    predicted = np.asarray(
        [prediction.predicted for prediction in predictions], dtype=int
    )
    scores = np.asarray([prediction.score for prediction in predictions], dtype=float)
    tn, fp, fn, tp = confusion_matrix(true, predicted, labels=[0, 1]).ravel()
    ranked = sorted(predictions, key=lambda item: (-item.score, item.episode_slug))
    positives = int(true.sum())
    precision_at_k: dict[str, float] = {}
    recall_at_k: dict[str, float] = {}
    hits_at_k: dict[str, int] = {}
    for budget in review_budgets:
        actual_budget = min(int(budget), len(ranked))
        hits = sum(item.target for item in ranked[:actual_budget])
        key = str(budget)
        hits_at_k[key] = int(hits)
        precision_at_k[key] = hits / actual_budget if actual_budget else 0.0
        recall_at_k[key] = hits / positives if positives else 0.0
    return {
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "in_precision": float(precision_score(true, predicted, zero_division=0)),
        "in_recall": float(recall_score(true, predicted, zero_division=0)),
        "in_f1": float(f1_score(true, predicted, zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(true, predicted)),
        "average_precision": float(average_precision_score(true, scores)),
        "roc_auc": float(roc_auc_score(true, scores)),
        "ranking": {
            "hits_at_k": hits_at_k,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "rows": [
                {
                    "rank": rank,
                    "episode_slug": item.episode_slug,
                    "target": item.target,
                    "score": item.score,
                }
                for rank, item in enumerate(ranked, start=1)
            ],
        },
    }


def _reference_predictions(
    records: Sequence[CalibrationRecord], name: str
) -> list[CalibrationPrediction]:
    predictions: list[CalibrationPrediction] = []
    for record in records:
        phase2 = record.features["phase2"]
        if name == "majority_out":
            score, predicted, threshold = 0.0, 0, 1.0
        elif name == "raw_any_check":
            score = float(phase2["any_check_likelihood"])
            predicted = int(phase2["any_check_in"])
            threshold = 0.5
        elif name == "raw_standard_check":
            score = float(phase2["standard_check_likelihood"])
            predicted = int(phase2["standard_check_in"])
            threshold = 0.5
        elif name == "raw_ranking_score":
            score = float(phase2["ranking_score"])
            predicted = int(score >= 0.5)
            threshold = 0.5
        else:
            raise ValueError(f"unknown reference method: {name}")
        predictions.append(
            CalibrationPrediction(
                episode_slug=record.episode_slug,
                target=record.target,
                score=score,
                predicted=predicted,
                threshold=threshold,
                chosen_c=0.0,
                feature_family=name,
                training_slugs=(),
            )
        )
    return predictions


def run_analysis(records: Sequence[CalibrationRecord]) -> dict[str, Any]:
    """Evaluate five learned families and four fixed reference methods."""
    if not records:
        raise ValueError("calibration population cannot be empty")
    methods: dict[str, list[CalibrationPrediction]] = {
        family: nested_leave_one_out(records, family) for family in FEATURE_FAMILIES
    }
    for reference in (
        "majority_out",
        "raw_any_check",
        "raw_standard_check",
        "raw_ranking_score",
    ):
        methods[reference] = _reference_predictions(records, reference)

    metrics = {
        name: evaluate_predictions(predictions)
        for name, predictions in methods.items()
    }
    prediction_rows = []
    for name, predictions in methods.items():
        for prediction in predictions:
            prediction_rows.append(
                {
                    "method": name,
                    "episode_slug": prediction.episode_slug,
                    "target": prediction.target,
                    "score": prediction.score,
                    "predicted": prediction.predicted,
                    "threshold": prediction.threshold,
                    "chosen_c": prediction.chosen_c,
                    "training_slugs": list(prediction.training_slugs),
                }
            )
    return {
        "dataset": {
            "episodes": len(records),
            "ins": sum(record.target for record in records),
            "outs": len(records) - sum(record.target for record in records),
        },
        "evaluation": "nested_leave_one_out_development_not_untouched_holdout",
        "methods": metrics,
        "feature_manifest": {
            family: list(feature_names(records, family)) for family in FEATURE_FAMILIES
        },
        "prediction_rows": prediction_rows,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {
                key: json.dumps(value, separators=(",", ":"))
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
            writer.writerow(normalized)


def _report_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Semantic calibration comparison",
        "",
        f"Population: {result['dataset']['episodes']} episodes, "
        f"{result['dataset']['ins']} Ins, {result['dataset']['outs']} Outs.",
        "",
        "| Method | Balanced accuracy | In precision | In recall | In F1 | AP | ROC AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in result["methods"].items():
        lines.append(
            f"| {name} | {metrics['balanced_accuracy']:.3f} | "
            f"{metrics['in_precision']:.3f} | {metrics['in_recall']:.3f} | "
            f"{metrics['in_f1']:.3f} | {metrics['average_precision']:.3f} | "
            f"{metrics['roc_auc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "These are nested leave-one-out development results, not an untouched holdout estimate.",
            "",
        ]
    )
    return "\n".join(lines)


def write_analysis(
    records: Sequence[CalibrationRecord], output_root: Path
) -> dict[str, Any]:
    """Run comparison and write its reproducibility artifacts."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    result = run_analysis(records)
    write_json(
        output / "metrics.json",
        {key: value for key, value in result.items() if key != "prediction_rows"},
    )
    write_json(output / "feature-manifest.json", result["feature_manifest"])
    _write_csv(output / "predictions.csv", result["prediction_rows"])
    ranking_rows = [
        {"method": method, **row}
        for method, metrics in result["methods"].items()
        for row in metrics["ranking"]["rows"]
    ]
    _write_csv(output / "rankings.csv", ranking_rows)
    (output / "report.md").write_text(
        _report_markdown(result), encoding="utf-8"
    )
    return result
