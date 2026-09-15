"""Investor-specific leave-one-pitch-out decision and rationale models."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .phase2_calibration_evaluation import (
    MethodPrediction,
    Phase2CalibrationRecord,
    classification_metric_rows,
    cluster_bootstrap_intervals,
    cluster_permutation_test,
    holm_adjust,
    ranking_metric_rows,
    raw_method_predictions,
)


FeatureMap = dict[str, float]
PerVCFold = tuple[str, tuple[int, ...], int]


@dataclass(frozen=True)
class PerVCPrediction:
    vc_slug: str
    vc_name: str
    episode_slug: str
    target: int
    method: str
    score: float
    predicted: int
    threshold: float
    selected_config: str
    training_episode_slugs: tuple[str, ...]
    coefficients: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class TreeRule:
    description: str
    conditions: tuple[tuple[int, str, float], ...]
    support: int

    def activation(self, matrix: np.ndarray) -> np.ndarray:
        selected = np.ones(matrix.shape[0], dtype=bool)
        for feature_index, operator, threshold in self.conditions:
            if operator == "<=":
                selected &= matrix[:, feature_index] <= threshold
            else:
                selected &= matrix[:, feature_index] > threshold
        return selected.astype(float)


def rationale_feature_view(record: Phase2CalibrationRecord) -> FeatureMap:
    """Return Phase 1 rationale and constraint semantics only."""
    allowed = ("rationale__", "constraint__")
    return {
        name: float(value)
        for name, value in record.semantic_features.items()
        if name.startswith(allowed)
        and "target" not in name
        and "actual" not in name
    }


def decision_feature_view(record: Phase2CalibrationRecord) -> FeatureMap:
    """Combine rationale semantics with non-identity structured Phase 2 fields."""
    features = rationale_feature_view(record)
    features.update(
        {
            name: float(value)
            for name, value in record.phase2_features.items()
            if not name.startswith("vc__")
            and "target" not in name
            and "actual" not in name
        }
    )
    return features


def make_per_vc_leave_one_out_folds(
    records: Sequence[Phase2CalibrationRecord],
) -> tuple[PerVCFold, ...]:
    """Hold each investor-pitch case out once, using only that VC for training."""
    folds: list[PerVCFold] = []
    for vc_slug in sorted({record.vc_slug for record in records}):
        indices = [index for index, record in enumerate(records) if record.vc_slug == vc_slug]
        if len(indices) < 3:
            raise ValueError(f"per-VC leave-one-out needs at least three cases: {vc_slug}")
        targets = {records[index].target for index in indices}
        if targets != {0, 1}:
            raise ValueError(f"per-VC leave-one-out needs both classes: {vc_slug}")
        for held_index in indices:
            train = tuple(index for index in indices if index != held_index)
            folds.append((vc_slug, train, held_index))
    return tuple(folds)


_ELASTIC_CONFIGS = ((0.1, 0.5), (1.0, 0.5), (10.0, 0.5))


def _elastic_estimator(c_value: float, l1_ratio: float, seed: int):
    return make_pipeline(
        DictVectorizer(sparse=True),
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=c_value,
            solver="saga",
            l1_ratio=l1_ratio,
            class_weight="balanced",
            max_iter=1_000,
            tol=1e-3,
            random_state=seed,
        ),
    )


def _feature_view(record: Phase2CalibrationRecord, family: str) -> FeatureMap:
    if family == "decision":
        return decision_feature_view(record)
    if family == "rationale":
        return rationale_feature_view(record)
    raise ValueError(f"unknown per-VC feature family: {family}")


def _best_threshold(targets: np.ndarray, scores: np.ndarray) -> float:
    candidates = sorted({0.0, 0.5, 1.0, *(float(value) for value in scores)})
    return max(
        candidates,
        key=lambda threshold: (
            balanced_accuracy_score(targets, scores >= threshold),
            threshold,
        ),
    )


def _inner_elastic_selection(
    records: Sequence[Phase2CalibrationRecord],
    *,
    family: str,
    inner_splits: int,
    seed: int,
) -> tuple[float, float, float]:
    targets = np.asarray([record.target for record in records], dtype=int)
    split_count = min(inner_splits, int(targets.sum()), int(len(targets) - targets.sum()))
    if split_count < 2:
        raise ValueError("inner elastic selection needs at least two cases per class")
    splitter = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=seed)
    maps = [_feature_view(record, family) for record in records]
    folds = tuple(splitter.split(np.zeros(len(records)), targets))
    candidates: list[tuple[float, float, float, np.ndarray]] = []
    for c_value, l1_ratio in _ELASTIC_CONFIGS:
        scores = np.zeros(len(records), dtype=float)
        for train_indices, held_indices in folds:
            estimator = _elastic_estimator(c_value, l1_ratio, seed)
            estimator.fit([maps[int(index)] for index in train_indices], targets[train_indices])
            scores[held_indices] = estimator.predict_proba(
                [maps[int(index)] for index in held_indices]
            )[:, 1]
        candidates.append(
            (average_precision_score(targets, scores), -c_value, c_value, scores)
        )
    _, _, selected_c, selected_scores = max(candidates, key=lambda row: (row[0], row[1]))
    return selected_c, 0.5, _best_threshold(targets, selected_scores)


def _coefficient_rows(estimator) -> tuple[tuple[str, float], ...]:
    vectorizer = estimator.named_steps["dictvectorizer"]
    classifier = estimator.named_steps["logisticregression"]
    names = vectorizer.get_feature_names_out()
    values = classifier.coef_[0]
    return tuple(
        (str(name), float(value))
        for name, value in zip(names, values, strict=True)
        if abs(float(value)) > 1e-10
    )


def nested_per_vc_elastic_predictions(
    records: Sequence[Phase2CalibrationRecord],
    *,
    family: str,
    inner_splits: int = 3,
    seed: int = 20260816,
) -> list[PerVCPrediction]:
    """Predict every pitch from an elastic-net model trained on that VC's others."""
    predictions: list[PerVCPrediction | None] = [None] * len(records)
    for fold_index, (vc_slug, train_indices, held_index) in enumerate(
        make_per_vc_leave_one_out_folds(records)
    ):
        train_records = [records[index] for index in train_indices]
        held_record = records[held_index]
        c_value, l1_ratio, threshold = _inner_elastic_selection(
            train_records,
            family=family,
            inner_splits=inner_splits,
            seed=seed + fold_index,
        )
        estimator = _elastic_estimator(c_value, l1_ratio, seed + fold_index)
        estimator.fit(
            [_feature_view(record, family) for record in train_records],
            np.asarray([record.target for record in train_records], dtype=int),
        )
        score = float(estimator.predict_proba([_feature_view(held_record, family)])[0, 1])
        predictions[held_index] = PerVCPrediction(
            vc_slug=vc_slug,
            vc_name=held_record.vc_name,
            episode_slug=held_record.episode_slug,
            target=held_record.target,
            method=f"per_vc_{family}_elastic_net",
            score=score,
            predicted=int(score >= threshold),
            threshold=float(threshold),
            selected_config=f"C={c_value:g};l1_ratio={l1_ratio:g}",
            training_episode_slugs=tuple(record.episode_slug for record in train_records),
            coefficients=_coefficient_rows(estimator),
        )
    if any(prediction is None for prediction in predictions):
        raise RuntimeError("per-VC leave-one-out did not predict every case")
    return [prediction for prediction in predictions if prediction is not None]


def _forest_estimator(
    training_count: int,
    *,
    n_estimators: int,
    seed: int,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=2,
        min_samples_leaf=max(2, ceil(training_count * 0.10)),
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        # These forests are tiny and invoked thousands of times during nested
        # evaluation. Parallel prediction overhead is much larger than the work.
        n_jobs=1,
    )


def extract_tree_rules(
    forest: RandomForestClassifier,
    *,
    feature_names: Sequence[str],
    matrix: np.ndarray,
    minimum_support: int,
) -> tuple[TreeRule, ...]:
    """Extract unique supported root-to-leaf conjunctions from shallow trees."""
    rules: dict[str, TreeRule] = {}
    for estimator in forest.estimators_:
        tree = estimator.tree_

        def visit(node: int, conditions: tuple[tuple[int, str, float], ...]) -> None:
            feature = int(tree.feature[node])
            if feature < 0:
                if not conditions:
                    return
                parts = [
                    f"{feature_names[index]} {operator} {threshold:.6g}"
                    for index, operator, threshold in conditions
                ]
                description = " AND ".join(parts)
                candidate = TreeRule(description, conditions, 0)
                support = int(candidate.activation(matrix).sum())
                if minimum_support <= support < len(matrix):
                    rules.setdefault(
                        description,
                        TreeRule(description, conditions, support),
                    )
                return
            threshold = float(tree.threshold[node])
            visit(
                int(tree.children_left[node]),
                (*conditions, (feature, "<=", threshold)),
            )
            visit(
                int(tree.children_right[node]),
                (*conditions, (feature, ">", threshold)),
            )

        visit(0, ())
    return tuple(rules[name] for name in sorted(rules))


def _inner_indices(targets: np.ndarray, inner_splits: int, seed: int):
    split_count = min(inner_splits, int(targets.sum()), int(len(targets) - targets.sum()))
    if split_count < 2:
        raise ValueError("inner model selection needs at least two cases per class")
    splitter = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=seed)
    return tuple(splitter.split(np.zeros(len(targets)), targets))


def nested_per_vc_random_forest_predictions(
    records: Sequence[Phase2CalibrationRecord],
    *,
    inner_splits: int = 3,
    n_estimators: int = 100,
    seed: int = 20260816,
) -> list[PerVCPrediction]:
    """Evaluate a constrained rationale-only forest for each VC."""
    predictions: list[PerVCPrediction | None] = [None] * len(records)
    for fold_index, (vc_slug, train_indices, held_index) in enumerate(
        make_per_vc_leave_one_out_folds(records)
    ):
        train_records = [records[index] for index in train_indices]
        held_record = records[held_index]
        maps = [rationale_feature_view(record) for record in train_records]
        targets = np.asarray([record.target for record in train_records], dtype=int)
        inner_scores = np.zeros(len(train_records), dtype=float)
        for inner_index, (fit_indices, validation_indices) in enumerate(
            _inner_indices(targets, inner_splits, seed + fold_index)
        ):
            vectorizer = DictVectorizer(sparse=False)
            fit_matrix = vectorizer.fit_transform([maps[int(index)] for index in fit_indices])
            validation_matrix = vectorizer.transform(
                [maps[int(index)] for index in validation_indices]
            )
            forest = _forest_estimator(
                len(fit_indices),
                n_estimators=n_estimators,
                seed=seed + fold_index * 10 + inner_index,
            )
            forest.fit(fit_matrix, targets[fit_indices])
            inner_scores[validation_indices] = forest.predict_proba(validation_matrix)[:, 1]
        threshold = _best_threshold(targets, inner_scores)
        vectorizer = DictVectorizer(sparse=False)
        matrix = vectorizer.fit_transform(maps)
        forest = _forest_estimator(
            len(train_records),
            n_estimators=n_estimators,
            seed=seed + fold_index,
        )
        forest.fit(matrix, targets)
        score = float(
            forest.predict_proba(vectorizer.transform([rationale_feature_view(held_record)]))[0, 1]
        )
        importances = tuple(
            (str(name), float(value))
            for name, value in zip(
                vectorizer.get_feature_names_out(),
                forest.feature_importances_,
                strict=True,
            )
            if float(value) > 0.0
        )
        predictions[held_index] = PerVCPrediction(
            vc_slug=vc_slug,
            vc_name=held_record.vc_name,
            episode_slug=held_record.episode_slug,
            target=held_record.target,
            method="per_vc_rationale_random_forest",
            score=score,
            predicted=int(score >= threshold),
            threshold=float(threshold),
            selected_config=f"trees={n_estimators};depth=2",
            training_episode_slugs=tuple(record.episode_slug for record in train_records),
            coefficients=importances,
        )
    return [prediction for prediction in predictions if prediction is not None]


@dataclass
class _RuleFitBundle:
    vectorizer: DictVectorizer
    rules: tuple[TreeRule, ...]
    scaler: StandardScaler
    classifier: LogisticRegression

    def transform(self, maps: Sequence[FeatureMap]) -> np.ndarray:
        base = self.vectorizer.transform(maps)
        rule_matrix = np.column_stack(
            [rule.activation(base) for rule in self.rules]
        ) if self.rules else np.empty((len(maps), 0), dtype=float)
        return self.scaler.transform(np.column_stack((base, rule_matrix)))

    def predict_proba(self, maps: Sequence[FeatureMap]) -> np.ndarray:
        return self.classifier.predict_proba(self.transform(maps))[:, 1]

    def coefficient_rows(self) -> tuple[tuple[str, float], ...]:
        names = [
            *(f"feature__{name}" for name in self.vectorizer.get_feature_names_out()),
            *(f"rule__{rule.description}" for rule in self.rules),
        ]
        return tuple(
            (str(name), float(value))
            for name, value in zip(names, self.classifier.coef_[0], strict=True)
            if abs(float(value)) > 1e-10
        )


def _fit_rulefit(
    maps: Sequence[FeatureMap],
    targets: np.ndarray,
    *,
    c_value: float,
    n_estimators: int,
    seed: int,
) -> _RuleFitBundle:
    vectorizer = DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform(maps)
    forest = _forest_estimator(
        len(maps),
        n_estimators=n_estimators,
        seed=seed,
    )
    forest.fit(matrix, targets)
    rules = extract_tree_rules(
        forest,
        feature_names=tuple(str(name) for name in vectorizer.get_feature_names_out()),
        matrix=matrix,
        minimum_support=max(2, ceil(len(maps) * 0.10)),
    )
    rule_matrix = np.column_stack(
        [rule.activation(matrix) for rule in rules]
    ) if rules else np.empty((len(maps), 0), dtype=float)
    combined = np.column_stack((matrix, rule_matrix))
    scaler = StandardScaler().fit(combined)
    classifier = LogisticRegression(
        C=c_value,
        solver="liblinear",
        l1_ratio=1.0,
        class_weight="balanced",
        max_iter=10_000,
        random_state=seed,
    ).fit(scaler.transform(combined), targets)
    return _RuleFitBundle(vectorizer, rules, scaler, classifier)


def nested_per_vc_rulefit_predictions(
    records: Sequence[Phase2CalibrationRecord],
    *,
    inner_splits: int = 3,
    n_estimators: int = 40,
    seed: int = 20260816,
) -> list[PerVCPrediction]:
    """Evaluate shallow rationale rules selected by sparse logistic regression."""
    c_values = (0.1, 1.0, 10.0)
    predictions: list[PerVCPrediction | None] = [None] * len(records)
    for fold_index, (vc_slug, train_indices, held_index) in enumerate(
        make_per_vc_leave_one_out_folds(records)
    ):
        train_records = [records[index] for index in train_indices]
        held_record = records[held_index]
        maps = [rationale_feature_view(record) for record in train_records]
        targets = np.asarray([record.target for record in train_records], dtype=int)
        folds = _inner_indices(targets, inner_splits, seed + fold_index)
        candidates: list[tuple[float, float, float, np.ndarray]] = []
        for c_value in c_values:
            scores = np.zeros(len(train_records), dtype=float)
            for inner_index, (fit_indices, validation_indices) in enumerate(folds):
                bundle = _fit_rulefit(
                    [maps[int(index)] for index in fit_indices],
                    targets[fit_indices],
                    c_value=c_value,
                    n_estimators=n_estimators,
                    seed=seed + fold_index * 10 + inner_index,
                )
                scores[validation_indices] = bundle.predict_proba(
                    [maps[int(index)] for index in validation_indices]
                )
            candidates.append(
                (average_precision_score(targets, scores), -c_value, c_value, scores)
            )
        _, _, selected_c, inner_scores = max(candidates, key=lambda row: (row[0], row[1]))
        threshold = _best_threshold(targets, inner_scores)
        bundle = _fit_rulefit(
            maps,
            targets,
            c_value=selected_c,
            n_estimators=n_estimators,
            seed=seed + fold_index,
        )
        score = float(bundle.predict_proba([rationale_feature_view(held_record)])[0])
        predictions[held_index] = PerVCPrediction(
            vc_slug=vc_slug,
            vc_name=held_record.vc_name,
            episode_slug=held_record.episode_slug,
            target=held_record.target,
            method="per_vc_rationale_rulefit",
            score=score,
            predicted=int(score >= threshold),
            threshold=float(threshold),
            selected_config=f"C={selected_c:g};trees={n_estimators};depth=2",
            training_episode_slugs=tuple(record.episode_slug for record in train_records),
            coefficients=bundle.coefficient_rows(),
        )
    return [prediction for prediction in predictions if prediction is not None]


def _method_rows(
    records: Sequence[Phase2CalibrationRecord],
    predictions: Sequence[PerVCPrediction],
    method: str,
) -> list[MethodPrediction]:
    if len(records) != len(predictions):
        raise ValueError("per-VC predictions must align with canonical records")
    rows = []
    for record, prediction in zip(records, predictions, strict=True):
        if (
            record.vc_slug != prediction.vc_slug
            or record.episode_slug != prediction.episode_slug
            or record.target != prediction.target
        ):
            raise ValueError("per-VC prediction alignment failure")
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


def _coefficient_stability(
    predictions: Sequence[PerVCPrediction],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = sorted({prediction.method for prediction in predictions})
    for method in methods:
        method_rows = [prediction for prediction in predictions if prediction.method == method]
        for vc_slug in sorted({prediction.vc_slug for prediction in method_rows}):
            subset = [prediction for prediction in method_rows if prediction.vc_slug == vc_slug]
            names = sorted({name for prediction in subset for name, _ in prediction.coefficients})
            for name in names:
                values = np.asarray(
                    [dict(prediction.coefficients).get(name, 0.0) for prediction in subset],
                    dtype=float,
                )
                selected = np.abs(values) > 1e-10
                selected_values = values[selected]
                rows.append(
                    {
                        "vc_slug": vc_slug,
                        "vc_name": subset[0].vc_name,
                        "method": method,
                        "feature": name,
                        "feature_type": (
                            "rule" if name.startswith("rule__") else
                            "constraint" if "constraint__" in name else
                            "rationale"
                        ),
                        "fold_count": len(subset),
                        "selected_count": int(selected.sum()),
                        "selection_frequency": float(selected.mean()),
                        "mean_effect": float(values.mean()),
                        "median_effect": float(np.median(values)),
                        "lower_fold_95": float(np.quantile(values, 0.025)),
                        "upper_fold_95": float(np.quantile(values, 0.975)),
                        "positive_when_selected": (
                            float((selected_values > 0).mean()) if len(selected_values) else 0.0
                        ),
                        "odds_ratio": (
                            float(np.exp(np.clip(values.mean(), -20.0, 20.0)))
                            if "elastic_net" in method or "rulefit" in method
                            else None
                        ),
                    }
                )
    return rows


def run_per_vc_analysis(
    records: Sequence[Phase2CalibrationRecord],
    *,
    inner_splits: int = 3,
    forest_trees: int = 100,
    rule_trees: int = 40,
    bootstrap_iterations: int = 2_000,
    permutation_iterations: int = 5_000,
    seed: int = 20260816,
    review_budgets: Sequence[int] = (1, 3, 5, 10, 20),
) -> dict[str, object]:
    """Run independent per-investor leave-one-pitch-out model evaluation."""
    decision = nested_per_vc_elastic_predictions(
        records, family="decision", inner_splits=inner_splits, seed=seed
    )
    rationale = nested_per_vc_elastic_predictions(
        records, family="rationale", inner_splits=inner_splits, seed=seed + 1
    )
    forest = nested_per_vc_random_forest_predictions(
        records,
        inner_splits=inner_splits,
        n_estimators=forest_trees,
        seed=seed + 2,
    )
    rulefit = nested_per_vc_rulefit_predictions(
        records,
        inner_splits=inner_splits,
        n_estimators=rule_trees,
        seed=seed + 3,
    )
    detailed = {
        "per_vc_decision_elastic_net": decision,
        "per_vc_rationale_elastic_net": rationale,
        "per_vc_rationale_random_forest": forest,
        "per_vc_rationale_rulefit": rulefit,
    }
    methods: dict[str, list[MethodPrediction]] = {
        "raw_phase2": raw_method_predictions(records, "raw_phase2"),
        **{
            name: _method_rows(records, predictions, name)
            for name, predictions in detailed.items()
        },
    }
    classification = [
        row
        for predictions in methods.values()
        for row in classification_metric_rows(predictions)
    ]
    ranking = [
        row
        for predictions in methods.values()
        for row in ranking_metric_rows(predictions, review_budgets=review_budgets)
    ]
    uncertainty = cluster_bootstrap_intervals(
        methods,
        raw_method="raw_phase2",
        metrics=("balanced_accuracy", "average_precision"),
        iterations=bootstrap_iterations,
        seed=seed,
    )
    significance = []
    for method_index, method in enumerate(name for name in methods if name != "raw_phase2"):
        for metric_index, metric in enumerate(("balanced_accuracy", "average_precision")):
            significance.append(
                {
                    "test": "episode_cluster_randomization",
                    **cluster_permutation_test(
                        methods["raw_phase2"],
                        methods[method],
                        metric=metric,
                        iterations=permutation_iterations,
                        seed=seed + method_index * 10 + metric_index,
                    ),
                    "adjusted_p_value": None,
                    "significant_0_05": None,
                }
            )
    adjusted = holm_adjust([float(row["p_value"]) for row in significance])
    for row, adjusted_p in zip(significance, adjusted, strict=True):
        row["adjusted_p_value"] = adjusted_p
        row["significant_0_05"] = adjusted_p < 0.05
    return {
        "schema": "per-vc-rationale-model-evaluation-v1",
        "scientific_status": "nested_per_vc_leave_one_pitch_out_development_evaluation",
        "methods": methods,
        "detailed_predictions": detailed,
        "classification": classification,
        "ranking": ranking,
        "rationale_stability": _coefficient_stability(
            [*rationale, *forest, *rulefit]
        ),
        "uncertainty": uncertainty,
        "significance": significance,
        "parameters": {
            "inner_splits": inner_splits,
            "forest_trees": forest_trees,
            "rule_trees": rule_trees,
            "bootstrap_iterations": bootstrap_iterations,
            "permutation_iterations": permutation_iterations,
            "seed": seed,
            "review_budgets": list(review_budgets),
        },
        "api_cost_usd": 0.0,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, tuple, dict))
                    else value
                    for name, value in row.items()
                }
            )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _fit_full_forest(
    records: Sequence[Phase2CalibrationRecord],
    *,
    inner_splits: int,
    n_estimators: int,
    seed: int,
):
    maps = [rationale_feature_view(record) for record in records]
    targets = np.asarray([record.target for record in records], dtype=int)
    inner_scores = np.zeros(len(records), dtype=float)
    for fold_index, (fit_indices, validation_indices) in enumerate(
        _inner_indices(targets, inner_splits, seed)
    ):
        vectorizer = DictVectorizer(sparse=False)
        fit_matrix = vectorizer.fit_transform([maps[int(index)] for index in fit_indices])
        validation_matrix = vectorizer.transform([maps[int(index)] for index in validation_indices])
        forest = _forest_estimator(
            len(fit_indices), n_estimators=n_estimators, seed=seed + fold_index
        )
        forest.fit(fit_matrix, targets[fit_indices])
        inner_scores[validation_indices] = forest.predict_proba(validation_matrix)[:, 1]
    vectorizer = DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform(maps)
    forest = _forest_estimator(len(records), n_estimators=n_estimators, seed=seed)
    forest.fit(matrix, targets)
    return {
        "schema": "per-vc-rationale-random-forest-v1",
        "vectorizer": vectorizer,
        "estimator": forest,
        "threshold": _best_threshold(targets, inner_scores),
    }


def _fit_full_rulefit(
    records: Sequence[Phase2CalibrationRecord],
    *,
    inner_splits: int,
    n_estimators: int,
    seed: int,
):
    maps = [rationale_feature_view(record) for record in records]
    targets = np.asarray([record.target for record in records], dtype=int)
    folds = _inner_indices(targets, inner_splits, seed)
    candidates = []
    for c_value in (0.1, 1.0, 10.0):
        scores = np.zeros(len(records), dtype=float)
        for fold_index, (fit_indices, validation_indices) in enumerate(folds):
            bundle = _fit_rulefit(
                [maps[int(index)] for index in fit_indices],
                targets[fit_indices],
                c_value=c_value,
                n_estimators=n_estimators,
                seed=seed + fold_index,
            )
            scores[validation_indices] = bundle.predict_proba(
                [maps[int(index)] for index in validation_indices]
            )
        candidates.append((average_precision_score(targets, scores), -c_value, c_value, scores))
    _, _, selected_c, inner_scores = max(candidates, key=lambda row: (row[0], row[1]))
    return {
        "schema": "per-vc-rationale-rulefit-v1",
        "bundle": _fit_rulefit(
            maps,
            targets,
            c_value=selected_c,
            n_estimators=n_estimators,
            seed=seed,
        ),
        "threshold": _best_threshold(targets, inner_scores),
        "selected_c": selected_c,
    }


def _cross_validated_forest_importance(
    records: Sequence[Phase2CalibrationRecord],
    *,
    inner_splits: int,
    n_estimators: int,
    repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    maps = [rationale_feature_view(record) for record in records]
    targets = np.asarray([record.target for record in records], dtype=int)
    vectorizer = DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform(maps)
    names = [str(name) for name in vectorizer.get_feature_names_out()]
    drops: list[list[float]] = [[] for _ in names]
    rng = np.random.default_rng(seed)
    for fold_index, (fit_indices, validation_indices) in enumerate(
        _inner_indices(targets, inner_splits, seed)
    ):
        forest = _forest_estimator(
            len(fit_indices), n_estimators=n_estimators, seed=seed + fold_index
        )
        forest.fit(matrix[fit_indices], targets[fit_indices])
        baseline_scores = forest.predict_proba(matrix[validation_indices])[:, 1]
        baseline = average_precision_score(targets[validation_indices], baseline_scores)
        for feature_index in range(matrix.shape[1]):
            for _ in range(repeats):
                permuted = matrix[validation_indices].copy()
                permuted[:, feature_index] = rng.permutation(permuted[:, feature_index])
                value = average_precision_score(
                    targets[validation_indices], forest.predict_proba(permuted)[:, 1]
                )
                drops[feature_index].append(float(baseline - value))
    return [
        {
            "feature": name,
            "permutation_importance_mean": float(np.mean(values)),
            "permutation_importance_std": float(np.std(values)),
        }
        for name, values in zip(names, drops, strict=True)
    ]


def _rationale_label(feature: str) -> str | None:
    if feature.startswith("rationale__"):
        parts = feature.split("__")
        return parts[1] if len(parts) > 2 else None
    if feature.startswith("constraint__"):
        parts = feature.split("__")
        return f"constraint:{parts[1]}" if len(parts) > 2 else None
    return None


def aggregate_rationale_label_importance(
    importance_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Collapse encoded feature variants into investor/rationale associations."""
    candidates: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    supported = {
        "per_vc_rationale_elastic_net",
        "per_vc_rationale_random_forest_cv_permutation",
    }
    for row in importance_rows:
        method = str(row.get("method", ""))
        feature = str(row.get("feature", ""))
        label = _rationale_label(feature)
        if method not in supported or label is None:
            continue
        key = (str(row["vc_slug"]), method, label)
        candidates.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for (vc_slug, method, label), rows in sorted(candidates.items()):
        if method == "per_vc_rationale_elastic_net":
            representative = max(
                rows,
                key=lambda row: abs(float(row.get("mean_effect", 0.0)))
                * float(row.get("selection_frequency", 0.0)),
            )
            effect = float(representative.get("mean_effect", 0.0))
            frequency = float(representative.get("selection_frequency", 0.0))
            score = abs(effect) * frequency
            direction = "positive" if effect > 0 else "negative" if effect < 0 else "neutral"
        else:
            representative = max(
                rows,
                key=lambda row: max(
                    0.0, float(row.get("permutation_importance_mean", 0.0))
                ),
            )
            effect = None
            frequency = None
            score = max(
                0.0, float(representative.get("permutation_importance_mean", 0.0))
            )
            direction = "non_directional"
        output.append(
            {
                "vc_slug": vc_slug,
                "vc_name": representative["vc_name"],
                "method": method,
                "rationale_label": label,
                "association_direction": direction,
                "importance_score": score,
                "supporting_feature": representative["feature"],
                "selection_frequency": frequency,
                "mean_effect": effect,
                "encoded_feature_count": len(rows),
            }
        )
    return output


def _render_report(
    records: Sequence[Phase2CalibrationRecord],
    result: dict[str, object],
    rationale_labels: Sequence[dict[str, object]],
) -> str:
    names = {
        "raw_phase2": "Raw Phase 2",
        "per_vc_decision_elastic_net": "Per-VC decision elastic net",
        "per_vc_rationale_elastic_net": "Per-VC rationale elastic net",
        "per_vc_rationale_random_forest": "Per-VC rationale random forest",
        "per_vc_rationale_rulefit": "Per-VC rationale RuleFit",
    }
    classification = result["classification"]
    ranking = result["ranking"]
    lines = [
        "# Per-VC Leave-One-Pitch-Out Rationale Models",
        "",
        "> **Scientific status:** nested per-investor leave-one-pitch-out development evaluation; full-data explanatory models are not performance estimates.",
        "",
        f"Population: **{len(records)} investor-pitch cases**, **{sum(row.target for row in records)} Ins**, "
        f"**{len(records)-sum(row.target for row in records)} Outs**, and **{len(set(row.vc_slug for row in records))} VCs**.",
        "",
        "Every held pitch is predicted using only the remaining pitches belonging to the same VC. Rationale-only models cannot access Phase 2 fields.",
        "",
        "## Macro classification",
        "",
        "| Method | Balanced accuracy | In precision | In recall | In F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    macro_class = {
        row["method"]: row for row in classification if row["scope"] == "macro"
    }
    for method in result["methods"]:
        row = macro_class[method]
        lines.append(
            f"| {names[method]} | {row['balanced_accuracy']:.3f} | "
            f"{row['in_precision']:.3f} | {row['in_recall']:.3f} | {row['in_f1']:.3f} |"
        )
    lines.extend([
        "",
        "## Macro ranking",
        "",
        "| Method | Average precision | ROC AUC | Precision@5 | Recall@5 |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in result["methods"]:
        rows = [
            row for row in ranking
            if row["scope"] == "macro" and row["method"] == method
        ]
        row = next((item for item in rows if item["review_budget"] == 5), rows[0])
        lines.append(
            f"| {names[method]} | {row['average_precision']:.3f} | {row['roc_auc']:.3f} | "
            f"{row['precision_at_k']:.3f} | {row['recall_at_k']:.3f} |"
        )
    lines.extend([
        "",
        "## Strongest rationale-level associations",
        "",
        "These are the strongest stable elastic-net associations within each VC. Positive means the encoded rationale signal is associated with an In; negative means it is associated with an Out. Scores combine coefficient magnitude and leave-one-out selection frequency.",
        "",
        "| VC | Rationale | Direction | Association score | Selection frequency |",
        "|---|---|---|---:|---:|",
    ])
    for vc_slug in sorted({row["vc_slug"] for row in rationale_labels}):
        rows = [
            row for row in rationale_labels
            if row["vc_slug"] == vc_slug
            and row["method"] == "per_vc_rationale_elastic_net"
        ]
        rows.sort(key=lambda row: float(row["importance_score"]), reverse=True)
        for row in rows[:5]:
            lines.append(
                f"| {row['vc_name']} | `{row['rationale_label']}` | "
                f"{row['association_direction']} | {float(row['importance_score']):.3f} | "
                f"{float(row['selection_frequency']):.3f} |"
            )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "Rationale coefficients, forest permutation importance, and selected rules are predictive associations with observed decisions. They are not causal estimates of why an investor invested.",
        "",
        "The pooled Phase 2 evaluation remains available separately and has not been replaced or modified by this experiment.",
        "",
    ])
    return "\n".join(lines)


def write_per_vc_analysis(
    records: Sequence[Phase2CalibrationRecord],
    result: dict[str, object],
    output_root: Path,
    *,
    registry_path: Path,
    importance_repeats: int = 10,
) -> dict[str, Path]:
    """Write per-VC metrics, explanations, provenance, and full-data models."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    detailed = result["detailed_predictions"]
    for method, predictions in result["methods"].items():
        detail = detailed.get(method)
        for index, prediction in enumerate(predictions):
            audit = detail[index] if detail is not None else None
            prediction_rows.append(
                {
                    "method": method,
                    "vc_slug": prediction.vc_slug,
                    "vc_name": prediction.vc_name,
                    "episode_slug": prediction.episode_slug,
                    "actual_decision": "In" if prediction.target else "Out",
                    "predicted_decision": "In" if prediction.predicted else "Out",
                    "score": prediction.score,
                    "threshold": audit.threshold if audit else 0.5,
                    "selected_config": audit.selected_config if audit else "raw",
                    "training_episode_slugs": audit.training_episode_slugs if audit else (),
                }
            )
    importance_rows = [dict(row) for row in result["rationale_stability"]]
    rule_rows: list[dict[str, object]] = []
    model_paths: list[Path] = []
    parameters = result["parameters"]
    for vc_index, vc_slug in enumerate(sorted({record.vc_slug for record in records})):
        vc_records = [record for record in records if record.vc_slug == vc_slug]
        targets = np.asarray([record.target for record in vc_records], dtype=int)
        models_dir = output / "models" / vc_slug
        models_dir.mkdir(parents=True, exist_ok=True)
        for family in ("decision", "rationale"):
            c_value, l1_ratio, threshold = _inner_elastic_selection(
                vc_records,
                family=family,
                inner_splits=int(parameters["inner_splits"]),
                seed=int(parameters["seed"]) + vc_index,
            )
            estimator = _elastic_estimator(c_value, l1_ratio, int(parameters["seed"]) + vc_index)
            estimator.fit(
                [_feature_view(record, family) for record in vc_records], targets
            )
            bundle = {
                "schema": f"per-vc-{family}-elastic-net-v1",
                "vc_slug": vc_slug,
                "estimator": estimator,
                "threshold": threshold,
                "selected_c": c_value,
                "l1_ratio": l1_ratio,
            }
            path = models_dir / f"{family}_elastic_net.joblib"
            joblib.dump(bundle, path)
            model_paths.append(path)
        forest_bundle = _fit_full_forest(
            vc_records,
            inner_splits=int(parameters["inner_splits"]),
            n_estimators=int(parameters["forest_trees"]),
            seed=int(parameters["seed"]) + 100 + vc_index,
        )
        forest_path = models_dir / "rationale_random_forest.joblib"
        joblib.dump(forest_bundle, forest_path)
        model_paths.append(forest_path)
        for row in _cross_validated_forest_importance(
            vc_records,
            inner_splits=int(parameters["inner_splits"]),
            n_estimators=int(parameters["forest_trees"]),
            repeats=importance_repeats,
            seed=int(parameters["seed"]) + 200 + vc_index,
        ):
            importance_rows.append(
                {
                    "vc_slug": vc_slug,
                    "vc_name": vc_records[0].vc_name,
                    "method": "per_vc_rationale_random_forest_cv_permutation",
                    **row,
                    "feature_type": "constraint" if "constraint__" in row["feature"] else "rationale",
                }
            )
        rulefit_bundle = _fit_full_rulefit(
            vc_records,
            inner_splits=int(parameters["inner_splits"]),
            n_estimators=int(parameters["rule_trees"]),
            seed=int(parameters["seed"]) + 300 + vc_index,
        )
        rulefit_path = models_dir / "rationale_rulefit.joblib"
        joblib.dump(rulefit_bundle, rulefit_path)
        model_paths.append(rulefit_path)
        bundle = rulefit_bundle["bundle"]
        base_matrix = bundle.vectorizer.transform(
            [rationale_feature_view(record) for record in vc_records]
        )
        coefficient_lookup = dict(bundle.coefficient_rows())
        base_rate = float(targets.mean())
        for rule in bundle.rules:
            coefficient = coefficient_lookup.get(f"rule__{rule.description}", 0.0)
            if abs(coefficient) <= 1e-10:
                continue
            activated = rule.activation(base_matrix).astype(bool)
            in_rate = float(targets[activated].mean()) if activated.any() else 0.0
            rule_rows.append(
                {
                    "vc_slug": vc_slug,
                    "vc_name": vc_records[0].vc_name,
                    "rule": rule.description,
                    "coefficient": coefficient,
                    "odds_ratio": float(np.exp(np.clip(coefficient, -20.0, 20.0))),
                    "support": int(activated.sum()),
                    "support_fraction": float(activated.mean()),
                    "in_rate": in_rate,
                    "base_in_rate": base_rate,
                    "lift": in_rate / base_rate if base_rate else 0.0,
                }
            )
    if not rule_rows:
        rule_rows.append({
            "vc_slug": "", "vc_name": "", "rule": "No non-zero rules selected",
            "coefficient": 0.0, "odds_ratio": 1.0, "support": 0,
            "support_fraction": 0.0, "in_rate": 0.0, "base_in_rate": 0.0, "lift": 0.0,
        })
    rationale_label_rows = aggregate_rationale_label_importance(importance_rows)
    paths = {
        "evaluation": output / "evaluation.md",
        "predictions": output / "predictions.csv",
        "classification": output / "classification_metrics.csv",
        "ranking": output / "ranking_metrics.csv",
        "importance": output / "rationale_importance.csv",
        "rationale_labels": output / "rationale_label_importance.csv",
        "rules": output / "rules.csv",
        "uncertainty": output / "uncertainty.csv",
        "significance": output / "significance.csv",
        "provenance": output / "fold_provenance.json",
        "manifest": output / "manifest.json",
    }
    _write_csv(paths["predictions"], prediction_rows)
    _write_csv(paths["classification"], result["classification"])
    _write_csv(paths["ranking"], result["ranking"])
    _write_csv(paths["importance"], importance_rows)
    _write_csv(paths["rationale_labels"], rationale_label_rows)
    _write_csv(paths["rules"], rule_rows)
    _write_csv(paths["uncertainty"], result["uncertainty"])
    _write_csv(paths["significance"], result["significance"])
    _write_json(
        paths["provenance"],
        {
            "schema": "per-vc-leave-one-pitch-out-provenance-v1",
            "folds": [
                {
                    "vc_slug": row["vc_slug"],
                    "held_episode_slug": row["episode_slug"],
                    "training_episode_slugs": row["training_episode_slugs"],
                }
                for row in prediction_rows
                if row["method"] == "per_vc_decision_elastic_net"
            ],
        },
    )
    paths["evaluation"].write_text(
        _render_report(records, result, rationale_label_rows), encoding="utf-8"
    )
    registry = Path(registry_path)
    manifest = {
        "schema": result["schema"],
        "scientific_status": result["scientific_status"],
        "population": {
            "cases": len(records),
            "investors": len(set(record.vc_slug for record in records)),
            "ins": sum(record.target for record in records),
            "outs": len(records) - sum(record.target for record in records),
        },
        "parameters": parameters,
        "registry_path": str(registry.resolve()),
        "registry_sha256": sha256(registry.read_bytes()).hexdigest() if registry.is_file() else None,
        "api_cost_usd": 0.0,
        "outputs": sorted(str(path.relative_to(output)) for path in paths.values()),
        "model_files": sorted(str(path.relative_to(output)) for path in model_paths),
        "notes": [
            "Each evaluation prediction is leave-one-pitch-out within one VC.",
            "Full-data models are explanation/deployment artifacts, not performance estimates.",
            "Rationale importance is predictive association, not causal effect.",
            "The pooled Phase 2 evaluator and reports are preserved separately.",
            "No LLM, network, or API calls were made.",
        ],
    }
    _write_json(paths["manifest"], manifest)
    return {**paths, **{f"model_{index}": path for index, path in enumerate(model_paths)}}
