"""Fold-safe probabilistic completion of canonical Phase 1 rationales."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import csv
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from statistics import fmean

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CompletionCase:
    vc_slug: str
    vc_name: str
    episode_slug: str
    actual_decision: str
    labels: tuple[str, ...]
    observed_labels: tuple[str, ...]
    features: Mapping[str, float]
    reference_targets: Mapping[str, int]


@dataclass(frozen=True)
class CompletionFold:
    held_index: int
    held_slug: str
    training_indices: tuple[int, ...]
    training_slugs: tuple[str, ...]


@dataclass(frozen=True)
class AssociationRuleEstimate:
    antecedent: tuple[str, ...]
    support: int
    posterior_probability: float
    training_prevalence: float
    lift: float


@dataclass(frozen=True)
class CompletionHypothesis:
    label: str
    rank: int
    logistic_probability: float
    logistic_fallback: bool
    association_probability: float
    association_support: int
    association_antecedent: tuple[str, ...]
    association_lift: float
    completion_probability: float
    training_slugs: tuple[str, ...]
    logistic_coefficients: tuple[tuple[str, float], ...] = ()
    association_rules: tuple[AssociationRuleEstimate, ...] = ()


@dataclass(frozen=True)
class CompletionPrediction:
    vc_slug: str
    episode_slug: str
    actual_decision: str
    observed_labels: tuple[str, ...]
    reference_labels: tuple[str, ...]
    training_slugs: tuple[str, ...]
    candidates: tuple[CompletionHypothesis, ...]
    association: tuple[CompletionHypothesis, ...]
    logistic: tuple[CompletionHypothesis, ...]
    ensemble: tuple[CompletionHypothesis, ...]


def _features(
    source: object, labels: Sequence[str], signals: Mapping[str, object]
) -> dict[str, float]:
    result: dict[str, float] = {}
    observed = 0
    instances_by_label: dict[str, list[object]] = {}
    phase1_case = getattr(source, "phase1_case", None)
    for row in getattr(phase1_case, "predicted", ()):
        instances_by_label.setdefault(str(getattr(row, "label")), []).append(row)
    total_instances = 0
    for label in labels:
        signal = signals[label]
        present = bool(getattr(signal, "predicted"))
        observed += int(present)
        prefix = f"rationale__{label}"
        instances = instances_by_label.get(label, [])
        if not instances and present:
            instances = [signal]
        total_instances += len(instances)
        result[f"{prefix}__present"] = float(present)
        result[f"{prefix}__count"] = float(len(instances))
        result[f"{prefix}__confidence"] = float(getattr(signal, "confidence"))
        result[f"{prefix}__signed_salience"] = float(
            getattr(signal, "signed_salience")
        )
        signed_confidence = 0.0
        for row in instances:
            sign = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}.get(
                str(getattr(row, "direction")), 0.0
            )
            weight = {"primary": 2.0, "secondary": 1.0}.get(
                str(getattr(row, "salience")), 0.0
            )
            signed_confidence += sign * weight * float(getattr(row, "confidence"))
        result[f"{prefix}__signed_salience_confidence"] = signed_confidence
        for value in ("positive", "negative", "neutral"):
            result[f"{prefix}__direction__{value}"] = float(
                sum(str(getattr(row, "direction")) == value for row in instances)
            )
        for value in ("primary", "secondary"):
            result[f"{prefix}__salience__{value}"] = float(
                sum(str(getattr(row, "salience")) == value for row in instances)
            )
    result["aggregate__distinct_label_count"] = float(observed)
    result["aggregate__instance_count"] = float(total_instances)
    return result


def build_completion_cases(cases: Sequence[object]) -> list[CompletionCase]:
    result: list[CompletionCase] = []
    seen: set[tuple[str, str]] = set()
    for source in cases:
        key = (str(getattr(source, "vc_slug")), str(getattr(source, "episode_slug")))
        if key in seen:
            raise ValueError(f"duplicate completion case: {key}")
        seen.add(key)
        labels = tuple(str(label) for label in getattr(source, "labels"))
        signals = getattr(source, "raw_signals")
        observed = tuple(label for label in labels if bool(signals[label].predicted))
        result.append(
            CompletionCase(
                vc_slug=key[0],
                vc_name=str(getattr(source, "vc_name")),
                episode_slug=key[1],
                actual_decision=str(getattr(source, "actual_decision")),
                labels=labels,
                observed_labels=observed,
                features=_features(source, labels, signals),
                reference_targets={
                    label: int(getattr(source, "reference_targets")[label])
                    for label in labels
                },
            )
        )
    return result


def leave_one_episode_out(cases: Sequence[CompletionCase]) -> tuple[CompletionFold, ...]:
    folds: list[CompletionFold] = []
    for held_index, held in enumerate(cases):
        training = tuple(
            index
            for index, case in enumerate(cases)
            if case.vc_slug == held.vc_slug and case.episode_slug != held.episode_slug
        )
        folds.append(
            CompletionFold(
                held_index=held_index,
                held_slug=held.episode_slug,
                training_indices=training,
                training_slugs=tuple(cases[index].episode_slug for index in training),
            )
        )
    return tuple(folds)


def _smoothed_prevalence(values: Sequence[int]) -> float:
    return (sum(values) + 1.0) / (len(values) + 2.0)


def _association_probability(
    training: Sequence[CompletionCase], held: CompletionCase, label: str
) -> tuple[float, int, tuple[str, ...], float, tuple[AssociationRuleEstimate, ...]]:
    targets = [int(case.reference_targets[label]) for case in training]
    prevalence = _smoothed_prevalence(targets)
    antecedents = [
        antecedent
        for size in (1, 2)
        for antecedent in combinations(held.observed_labels, size)
    ]
    candidates: list[AssociationRuleEstimate] = []
    for antecedent in antecedents:
        matched = [
            case
            for case in training
            if set(antecedent).issubset(case.observed_labels)
        ]
        if len(matched) < 3:
            continue
        probability = _smoothed_prevalence(
            [int(case.reference_targets[label]) for case in matched]
        )
        candidates.append(
            AssociationRuleEstimate(
                antecedent=antecedent,
                support=len(matched),
                posterior_probability=probability,
                training_prevalence=prevalence,
                lift=probability / prevalence,
            )
        )
    if not candidates:
        return prevalence, 0, (), 1.0, ()
    ordered = tuple(sorted(
        candidates,
        key=lambda row: (-row.posterior_probability, -row.support, row.antecedent),
    ))
    selected = ordered[0]
    return (
        selected.posterior_probability,
        selected.support,
        selected.antecedent,
        selected.lift,
        ordered,
    )


def _logistic_probability(
    training: Sequence[CompletionCase], held: CompletionCase, label: str, seed: int
) -> tuple[float, bool, tuple[tuple[str, float], ...]]:
    targets = np.asarray(
        [int(case.reference_targets[label]) for case in training], dtype=int
    )
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if positives < 3 or negatives < 3:
        return _smoothed_prevalence(targets.tolist()), True, ()
    split_count = min(3, positives, negatives)
    folds = tuple(
        StratifiedKFold(
            n_splits=split_count, shuffle=True, random_state=seed
        ).split(np.zeros(len(targets)), targets)
    )
    candidate_rows: list[tuple[float, float]] = []
    for c_value in (0.1, 1.0, 10.0):
        scores = np.zeros(len(targets), dtype=float)
        for train_indices, validation_indices in folds:
            estimator = _logistic_estimator(c_value, seed)
            estimator.fit(
                [training[int(index)].features for index in train_indices],
                targets[train_indices],
            )
            scores[validation_indices] = estimator.predict_proba(
                [training[int(index)].features for index in validation_indices]
            )[:, 1]
        candidate_rows.append((float(average_precision_score(targets, scores)), c_value))
    _, selected_c = max(candidate_rows, key=lambda row: (row[0], -row[1]))
    estimator = _logistic_estimator(selected_c, seed)
    estimator.fit([case.features for case in training], targets)
    vectorizer = estimator.named_steps["dictvectorizer"]
    model = estimator.named_steps["logisticregression"]
    coefficients = tuple(
        sorted(
            zip(vectorizer.get_feature_names_out(), model.coef_[0], strict=True),
            key=lambda row: row[0],
        )
    )
    return (
        float(estimator.predict_proba([held.features])[0, 1]),
        False,
        tuple((str(name), float(value)) for name, value in coefficients),
    )


def _logistic_estimator(c_value: float, seed: int):
    return make_pipeline(
        DictVectorizer(sparse=True),
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=c_value,
            solver="liblinear",
            max_iter=2_000,
            random_state=seed,
        ),
    )


def _ranked(
    raw: Sequence[CompletionHypothesis], *, method: str, limit: int
) -> tuple[CompletionHypothesis, ...]:
    def score(row: CompletionHypothesis) -> float:
        if method == "association":
            return row.association_probability
        if method == "logistic":
            return row.logistic_probability
        return row.completion_probability

    ordered = sorted(
        raw,
        key=lambda row: (-score(row), -row.association_support, row.label),
    )[:limit]
    return tuple(
        CompletionHypothesis(
            **{
                **row.__dict__,
                "rank": rank,
                "completion_probability": score(row),
            }
        )
        for rank, row in enumerate(ordered, start=1)
    )


def nested_completion_predictions(
    cases: Sequence[CompletionCase], *, max_hypotheses: int = 3, seed: int = 20260819
) -> list[CompletionPrediction]:
    if max_hypotheses < 1:
        raise ValueError("max_hypotheses must be positive")
    result: list[CompletionPrediction] = []
    for fold_index, fold in enumerate(leave_one_episode_out(cases)):
        held = cases[fold.held_index]
        training = [cases[index] for index in fold.training_indices]
        if not training:
            raise ValueError(f"completion fold has no training cases: {held.episode_slug}")
        raw: list[CompletionHypothesis] = []
        for label in held.labels:
            if label in held.observed_labels:
                continue
            association, support, antecedent, lift, rules = _association_probability(
                training, held, label
            )
            logistic, fallback, coefficients = _logistic_probability(
                training, held, label, seed + fold_index
            )
            raw.append(
                CompletionHypothesis(
                    label=label,
                    rank=0,
                    logistic_probability=logistic,
                    logistic_fallback=fallback,
                    association_probability=association,
                    association_support=support,
                    association_antecedent=antecedent,
                    association_lift=lift,
                    completion_probability=0.60 * logistic + 0.40 * association,
                    training_slugs=fold.training_slugs,
                    logistic_coefficients=coefficients,
                    association_rules=rules,
                )
            )
        result.append(
            CompletionPrediction(
                vc_slug=held.vc_slug,
                episode_slug=held.episode_slug,
                actual_decision=held.actual_decision,
                observed_labels=held.observed_labels,
                reference_labels=tuple(
                    label for label in held.labels if held.reference_targets[label]
                ),
                training_slugs=fold.training_slugs,
                candidates=tuple(raw),
                association=_ranked(raw, method="association", limit=max_hypotheses),
                logistic=_ranked(raw, method="logistic", limit=max_hypotheses),
                ensemble=_ranked(raw, method="ensemble", limit=max_hypotheses),
            )
        )
    return result


def nested_association_predictions(
    cases: Sequence[CompletionCase], *, max_hypotheses: int = 5
) -> list[CompletionPrediction]:
    """Generate the same fold-safe association rules without fitting logistic models."""
    if max_hypotheses < 1:
        raise ValueError("max_hypotheses must be positive")
    result: list[CompletionPrediction] = []
    for fold in leave_one_episode_out(cases):
        held = cases[fold.held_index]
        training = [cases[index] for index in fold.training_indices]
        if not training:
            raise ValueError(f"completion fold has no training cases: {held.episode_slug}")
        raw: list[CompletionHypothesis] = []
        for label in held.labels:
            if label in held.observed_labels:
                continue
            association, support, antecedent, lift, rules = _association_probability(
                training, held, label
            )
            raw.append(
                CompletionHypothesis(
                    label=label,
                    rank=0,
                    logistic_probability=0.0,
                    logistic_fallback=True,
                    association_probability=association,
                    association_support=support,
                    association_antecedent=antecedent,
                    association_lift=lift,
                    completion_probability=association,
                    training_slugs=fold.training_slugs,
                    association_rules=rules,
                )
            )
        ranked = _ranked(raw, method="association", limit=max_hypotheses)
        result.append(
            CompletionPrediction(
                vc_slug=held.vc_slug,
                episode_slug=held.episode_slug,
                actual_decision=held.actual_decision,
                observed_labels=held.observed_labels,
                reference_labels=tuple(
                    label for label in held.labels if held.reference_targets[label]
                ),
                training_slugs=fold.training_slugs,
                candidates=tuple(raw),
                association=ranked,
                logistic=(),
                ensemble=(),
            )
        )
    return result


def _method_score(row: CompletionHypothesis, method: str) -> float:
    if method == "association":
        return row.association_probability
    if method == "logistic":
        return row.logistic_probability
    if method == "ensemble":
        return row.completion_probability
    raise ValueError(f"unknown completion method: {method}")


def _method_metrics(
    predictions: Sequence[CompletionPrediction], method: str
) -> dict[str, float | int]:
    totals = {1: [0, 0], 3: [0, 0]}
    missing_total = 0
    recoverable = 0
    missing_cases = 0
    aps: list[float] = []
    for prediction in predictions:
        missing = set(prediction.reference_labels) - set(prediction.observed_labels)
        missing_total += len(missing)
        ranked = sorted(
            prediction.candidates,
            key=lambda row: (
                -_method_score(row, method),
                -row.association_support,
                row.label,
            ),
        )
        for budget in (1, 3):
            selected = {row.label for row in ranked[:budget]}
            totals[budget][0] += len(selected & missing)
            totals[budget][1] += len(selected)
        if missing:
            missing_cases += 1
            recoverable += int(bool({row.label for row in ranked[:3]} & missing))
            targets = np.asarray([int(row.label in missing) for row in ranked], dtype=int)
            scores = np.asarray([_method_score(row, method) for row in ranked], dtype=float)
            aps.append(float(average_precision_score(targets, scores)))
    result: dict[str, float | int] = {
        "case_count": len(predictions),
        "missing_reference_count": missing_total,
        "missing_reference_case_count": missing_cases,
        "recoverable_case_rate_at_3": recoverable / missing_cases if missing_cases else 1.0,
        "mean_average_precision_missing": fmean(aps) if aps else 1.0,
    }
    for budget, (hits, selected) in totals.items():
        result[f"missing_recall_at_{budget}"] = hits / missing_total if missing_total else 1.0
        result[f"hypothesis_precision_at_{budget}"] = hits / selected if selected else 1.0
    return result


def completion_analysis(
    predictions: Sequence[CompletionPrediction],
) -> dict[str, object]:
    if not predictions:
        raise ValueError("completion analysis requires predictions")
    keys = [(row.vc_slug, row.episode_slug) for row in predictions]
    if len(keys) != len(set(keys)):
        raise ValueError("completion analysis contains duplicate cases")
    return {
        "schema": "rationale-completion-analysis-v1",
        "scientific_status": "nested_development_automated_candidate_references",
        "api_cost_usd": 0.0,
        "case_count": len(predictions),
        "methods": {
            method: _method_metrics(predictions, method)
            for method in ("association", "logistic", "ensemble")
        },
        "decision_splits": {
            decision: {
                method: _method_metrics(
                    [row for row in predictions if row.actual_decision == decision],
                    method,
                )
                for method in ("association", "logistic", "ensemble")
            }
            for decision in ("In", "Out")
            if any(row.actual_decision == decision for row in predictions)
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_completion_analysis(
    predictions: Sequence[CompletionPrediction],
    analysis: Mapping[str, object],
    output: Path,
    *,
    input_paths: Mapping[str, Path],
) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    prediction_rows = [
        {
            "vc_slug": row.vc_slug,
            "episode_slug": row.episode_slug,
            "actual_decision": row.actual_decision,
            "observed_labels": ";".join(row.observed_labels),
            "reference_labels": ";".join(row.reference_labels),
            "training_count": len(row.training_slugs),
        }
        for row in predictions
    ]
    hypothesis_rows: list[dict[str, object]] = []
    association_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for row in predictions:
        for method in ("association", "logistic", "ensemble"):
            ordered = sorted(
                row.candidates,
                key=lambda item: (
                    -_method_score(item, method),
                    -item.association_support,
                    item.label,
                ),
            )
            for rank, item in enumerate(ordered, start=1):
                hypothesis_rows.append(
                    {
                        "vc_slug": row.vc_slug,
                        "episode_slug": row.episode_slug,
                        "actual_decision": row.actual_decision,
                        "method": method,
                        "rank": rank,
                        "label": item.label,
                        "score": _method_score(item, method),
                        "logistic_probability": item.logistic_probability,
                        "association_probability": item.association_probability,
                        "association_support": item.association_support,
                        "association_antecedent": ";".join(item.association_antecedent),
                        "reference_missing_hit": int(
                            item.label in row.reference_labels
                            and item.label not in row.observed_labels
                        ),
                    }
                )
        for item in row.candidates:
            for rule_rank, rule in enumerate(item.association_rules, start=1):
                association_rows.append({
                    "vc_slug": row.vc_slug,
                    "episode_slug": row.episode_slug,
                    "target_label": item.label,
                    "rule_rank": rule_rank,
                    "selected": int(rule_rank == 1),
                    "antecedent": ";".join(rule.antecedent),
                    "support": rule.support,
                    "posterior_probability": rule.posterior_probability,
                    "training_prevalence": rule.training_prevalence,
                    "lift": rule.lift,
                    "training_count": len(item.training_slugs),
                })
            for feature, coefficient in item.logistic_coefficients:
                coefficient_rows.append(
                    {
                        "vc_slug": row.vc_slug,
                        "episode_slug": row.episode_slug,
                        "target_label": item.label,
                        "feature": feature,
                        "coefficient": coefficient,
                        "fallback": int(item.logistic_fallback),
                    }
                )
    metric_rows = [
        {"population": "all", "method": method, **values}
        for method, values in dict(analysis["methods"]).items()
    ]
    for decision, methods in dict(analysis.get("decision_splits", {})).items():
        metric_rows.extend(
            {"population": f"actual_{decision}", "method": method, **values}
            for method, values in dict(methods).items()
        )
    per_label_rows: list[dict[str, object]] = []
    labels = sorted({item.label for row in predictions for item in row.candidates})
    for label in labels:
        proposed = [
            item
            for row in predictions
            for item in row.ensemble
            if item.label == label
        ]
        hits = sum(
            item.label in row.reference_labels and item.label not in row.observed_labels
            for row in predictions
            for item in row.ensemble
            if item.label == label
        )
        per_label_rows.append(
            {
                "label": label,
                "reference_support": sum(label in row.reference_labels for row in predictions),
                "missing_reference_support": sum(
                    label in row.reference_labels and label not in row.observed_labels
                    for row in predictions
                ),
                "ensemble_top3_proposals": len(proposed),
                "ensemble_top3_hits": hits,
                "ensemble_top3_precision": hits / len(proposed) if proposed else 0.0,
                "ensemble_top3_recall": hits
                / sum(
                    label in row.reference_labels and label not in row.observed_labels
                    for row in predictions
                )
                if any(
                    label in row.reference_labels and label not in row.observed_labels
                    for row in predictions
                )
                else 0.0,
            }
        )
    paths = {
        "predictions": output / "predictions.csv",
        "hypotheses": output / "hypotheses.csv",
        "association_rules": output / "association-rules.csv",
        "logistic_coefficients": output / "logistic-coefficients.csv",
        "metrics": output / "metrics.csv",
        "per_label": output / "per-label.csv",
        "fold_provenance": output / "fold-provenance.json",
        "manifest": output / "manifest.json",
        "evaluation": output / "evaluation.md",
    }
    _write_csv(paths["predictions"], prediction_rows)
    _write_csv(paths["hypotheses"], hypothesis_rows)
    _write_csv(paths["association_rules"], association_rows)
    _write_csv(
        paths["logistic_coefficients"],
        coefficient_rows
        or [{
            "vc_slug": "",
            "episode_slug": "",
            "target_label": "",
            "feature": "",
            "coefficient": "",
            "fallback": 1,
        }],
    )
    _write_csv(paths["metrics"], metric_rows)
    _write_csv(paths["per_label"], per_label_rows)
    paths["fold_provenance"].write_text(
        json.dumps(
            [
                {
                    "vc_slug": row.vc_slug,
                    "held_episode_slug": row.episode_slug,
                    "training_episode_slugs": list(row.training_slugs),
                }
                for row in predictions
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        **dict(analysis),
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            }
            for name, path in input_paths.items()
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ensemble = dict(analysis["methods"])["ensemble"]
    paths["evaluation"].write_text(
        "# Probabilistic Rationale Completion\n\n"
        "This is a nested development evaluation using automated candidate "
        "transcript-observed rationale references, not human-validated ground truth.\n\n"
        "## Ensemble\n\n"
        f"- Missing-rationale recall@3: {float(ensemble['missing_recall_at_3']):.3f}\n"
        f"- Hypothesis precision@3: {float(ensemble['hypothesis_precision_at_3']):.3f}\n"
        f"- Missing-label MAP: {float(ensemble['mean_average_precision_missing']):.3f}\n",
        encoding="utf-8",
    )
    return paths
