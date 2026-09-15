"""Post-hoc semantic models over frozen v4 reasoning artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .artifacts import read_verified_frozen, write_json
from .schemas_v4 import DecisionV4, InvestigationV4
from .two_head_calibration import (
    ThresholdSelection,
    extract_compact_v4_features,
    load_v4_calibration_population,
)


_SEED = 20260809
_C_VALUES = (0.01, 0.1, 1.0)
_METHODS = (
    "tfidf_phase1",
    "tfidf_full",
    "embedding_prototype",
    "embedding_logistic",
    "structured_elastic",
    "structured_boost",
    "late_fusion",
)


def _line(name: str, value: object) -> str:
    return f"{name}: {str(value).strip()}"


def extract_reasoning_views(
    investigation: Mapping[str, Any], decision: Mapping[str, Any]
) -> dict[str, str]:
    """Render deterministic label-blind text views from frozen reasoning."""
    phase1: list[str] = []
    for rationale in investigation.get("rationales", []):
        phase1.append(
            "rationale "
            f"{rationale.get('rationale_id')}: {rationale.get('taxonomy_label')} "
            f"{rationale.get('direction')} {rationale.get('salience')} "
            f"confidence={float(rationale.get('confidence', 0.0)):.3f}; "
            f"justification={rationale.get('justification', '')}"
        )
    for question in investigation.get("questions", []):
        phase1.append(
            "question: "
            f"{question.get('question', '')}; status={question.get('status', '')}; "
            f"answer={question.get('answer', '')}"
        )
    for conflict in investigation.get("conflicts", []):
        phase1.append(_line("conflict", conflict))
    for observation in investigation.get("unmapped_observations", []):
        phase1.append(
            _line("unmapped observation", observation.get("description", ""))
        )
    phase1.append(_line("information sufficient", investigation.get("information_sufficient")))
    phase1.append(_line("investigation summary", investigation.get("summary", "")))
    phase1_text = "\n".join(phase1)

    full = [phase1_text]
    full.append(_line("agent decision", decision.get("decision", "")))
    full.append(_line("decision path", decision.get("decision_path", "")))
    full.append(_line("investment likelihood", decision.get("investment_likelihood", "")))
    full.append(_line("decision confidence", decision.get("decision_confidence", "")))
    full.append(_line("decision justification", decision.get("decision_justification", "")))
    for evidence in decision.get("evidence_basis", []):
        full.append(
            "evidence basis: "
            f"source={evidence.get('source_type', '')}; "
            f"effect={evidence.get('effect_on_decision', '')}; "
            f"interpretation={evidence.get('interpretation', '')}"
        )
    opposing = decision.get("strongest_opposing_case", {})
    full.append(_line("strongest opposing case", opposing.get("argument", "")))
    full.append(_line("response to strongest opposing case", opposing.get("response", "")))
    full.append(
        _line("founder exception assessment", decision.get("founder_exception_assessment", ""))
    )
    full.append(_line("sufficiency assessment", decision.get("sufficiency_assessment", "")))
    for condition in decision.get("reversal_conditions", []):
        full.append(_line("reversal condition", condition))
    return {
        "phase1_rationales": phase1_text,
        "full_reasoning": "\n".join(full),
    }


def extract_phase2_reasoning_view(decision: Mapping[str, Any]) -> str:
    """Render the decision synthesis without repeating the Phase 1 investigation."""
    lines = [
        _line("agent decision", decision.get("decision", "")),
        _line("decision path", decision.get("decision_path", "")),
        _line("investment likelihood", decision.get("investment_likelihood", "")),
        _line("decision confidence", decision.get("decision_confidence", "")),
        _line("decision justification", decision.get("decision_justification", "")),
    ]
    for evidence in decision.get("evidence_basis", []):
        lines.append(
            "evidence basis: "
            f"source={evidence.get('source_type', '')}; "
            f"effect={evidence.get('effect_on_decision', '')}; "
            f"interpretation={evidence.get('interpretation', '')}"
        )
    opposing = decision.get("strongest_opposing_case", {})
    lines.extend((
        _line("strongest opposing case", opposing.get("argument", "")),
        _line("response to strongest opposing case", opposing.get("response", "")),
        _line(
            "founder exception assessment",
            decision.get("founder_exception_assessment", ""),
        ),
        _line("sufficiency assessment", decision.get("sufficiency_assessment", "")),
    ))
    for condition in decision.get("reversal_conditions", []):
        lines.append(_line("reversal condition", condition))
    return "\n".join(lines)


def extract_structured_semantic_features(
    investigation: Mapping[str, Any],
    decision: Mapping[str, Any],
    batch_record: Mapping[str, Any],
) -> dict[str, float]:
    """Combine compact v4 measurements with explicit rationale semantics."""
    features = extract_compact_v4_features(investigation, decision, batch_record)
    weights = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    for rationale in investigation.get("rationales", []):
        label = str(rationale.get("taxonomy_label", "unknown"))
        direction = str(rationale.get("direction", "neutral"))
        salience = str(rationale.get("salience", "secondary"))
        confidence = float(rationale.get("confidence", 0.0))
        key = f"label__{label}__signed_confidence"
        features[key] = features.get(key, 0.0) + weights.get(direction, 0.0) * confidence
        count_key = f"label_direction__{label}__{direction}__{salience}__count"
        features[count_key] = features.get(count_key, 0.0) + 1.0
    return {name: float(value) for name, value in features.items()}


@dataclass(frozen=True)
class SemanticCalibrationRecord:
    episode_slug: str
    target: int
    target_decision: str
    original_decision: str
    original_likelihood: float
    review_priority_score: float
    phase1_text: str
    full_text: str
    structured_features: Mapping[str, float]
    artifact_root: str
    phase1_sha256: str
    phase2_sha256: str


def load_semantic_population(
    status_path: Path, labels_path: Path
) -> list[SemanticCalibrationRecord]:
    """Join verified v4 records to deterministic reasoning views."""
    base_records = load_v4_calibration_population(status_path, labels_path)
    status = json.loads(Path(status_path).read_text(encoding="utf-8"))
    batch_by_slug = {
        str(row["episode_slug"]): row for row in status.get("completed_records", [])
    }
    records: list[SemanticCalibrationRecord] = []
    for base in base_records:
        batch_record = batch_by_slug[base.episode_slug]
        root = Path(base.artifact_root)
        investigation_raw, investigation_digest = read_verified_frozen(
            root / "phase1/investigation.json", root / "phase1/investigation.sha256"
        )
        decision_raw, decision_digest = read_verified_frozen(
            root / "phase2/decision.json", root / "phase2/decision.sha256"
        )
        investigation = InvestigationV4.model_validate_json(investigation_raw)
        decision = DecisionV4.model_validate_json(decision_raw)
        if investigation_digest != base.phase1_sha256 or decision_digest != base.phase2_sha256:
            raise ValueError(f"semantic population digest mismatch: {base.episode_slug}")
        investigation_dict = investigation.model_dump(mode="json")
        decision_dict = decision.model_dump(mode="json")
        views = extract_reasoning_views(investigation_dict, decision_dict)
        records.append(
            SemanticCalibrationRecord(
                episode_slug=base.episode_slug,
                target=base.target,
                target_decision=base.target_decision,
                original_decision=base.original_decision,
                original_likelihood=base.original_likelihood,
                review_priority_score=base.review_priority_score,
                phase1_text=views["phase1_rationales"],
                full_text=views["full_reasoning"],
                structured_features=extract_structured_semantic_features(
                    investigation_dict, decision_dict, batch_record
                ),
                artifact_root=base.artifact_root,
                phase1_sha256=base.phase1_sha256,
                phase2_sha256=base.phase2_sha256,
            )
        )
    return records


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def prototype_probability(
    train_vectors: np.ndarray, targets: np.ndarray, held_vector: np.ndarray
) -> float:
    """Return a bounded cosine-margin score from training-only class centroids."""
    positives = train_vectors[targets == 1]
    negatives = train_vectors[targets == 0]
    if not len(positives) or not len(negatives):
        raise ValueError("prototype scoring requires both classes")
    positive_centroid = _normalize(np.mean(positives, axis=0))
    negative_centroid = _normalize(np.mean(negatives, axis=0))
    candidate = _normalize(np.asarray(held_vector, dtype=float))
    margin = float(candidate @ positive_centroid - candidate @ negative_centroid)
    return 1.0 / (1.0 + math.exp(-5.0 * margin))


def _fast_threshold(
    targets: np.ndarray, scores: np.ndarray, cap: float
) -> ThresholdSelection:
    best: tuple[tuple[float, ...], ThresholdSelection] | None = None
    upper = float(np.max(scores)) + max(1e-9, abs(float(np.max(scores))) * 1e-9)
    for threshold in sorted({0.0, upper, *(float(score) for score in scores)}):
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (targets == 1)))
        fp = int(np.sum(predicted & (targets == 0)))
        tn = int(np.sum(~predicted & (targets == 0)))
        fn = int(np.sum(~predicted & (targets == 1)))
        fpr = fp / (fp + tn) if fp + tn else 0.0
        if fpr > cap + 1e-12:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        selection = ThresholdSelection(
            threshold=threshold,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            balanced_accuracy=(recall + specificity) / 2.0,
            false_positive_rate=fpr,
        )
        key = (recall, precision, selection.balanced_accuracy, threshold)
        if best is None or key > best[0]:
            best = (key, selection)
    if best is None:
        raise RuntimeError("no capped threshold found")
    return best[1]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _tfidf() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=1.0,
                    max_features=3000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=1,
                    max_features=3000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def _linear_estimator(kind: str, c_value: float):
    if kind == "logistic":
        return LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=20_000,
            random_state=_SEED,
        )
    if kind == "linear_svc":
        return LinearSVC(
            C=c_value,
            class_weight="balanced",
            dual="auto",
            max_iter=20_000,
            random_state=_SEED,
        )
    raise ValueError(f"unknown linear estimator: {kind}")


def _estimator_score(estimator: object, values: Any) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(values)[:, 1], dtype=float)
    return _sigmoid(np.asarray(estimator.decision_function(values), dtype=float))


def _inner_split(targets: np.ndarray) -> StratifiedKFold:
    count = min(3, int(targets.sum()), int(len(targets) - targets.sum()))
    if count < 2:
        raise ValueError("nested semantic evaluation requires two cases per class")
    return StratifiedKFold(n_splits=count, shuffle=True, random_state=_SEED)


def _select_candidate(
    targets: np.ndarray,
    scored_candidates: Sequence[tuple[Mapping[str, object], np.ndarray]],
    cap: float,
) -> tuple[Mapping[str, object], ThresholdSelection, np.ndarray]:
    best: tuple[tuple[float, ...], Mapping[str, object], ThresholdSelection, np.ndarray] | None = None
    for config, scores in scored_candidates:
        selection = _fast_threshold(targets, scores, cap)
        ap = float(average_precision_score(targets, scores))
        key = (
            selection.recall,
            selection.precision,
            selection.balanced_accuracy,
            ap,
            selection.threshold,
        )
        if best is None or key > best[0]:
            best = (key, config, selection, scores)
    assert best is not None
    return best[1], best[2], best[3]


def _tfidf_outer(
    train_texts: Sequence[str],
    train_targets: np.ndarray,
    held_text: str,
    cap: float,
) -> tuple[float, ThresholdSelection, Mapping[str, object]]:
    splitter = _inner_split(train_targets)
    fold_matrices = []
    for fit, validate in splitter.split(np.zeros(len(train_targets)), train_targets):
        vectorizer = _tfidf()
        x_fit = vectorizer.fit_transform([train_texts[index] for index in fit])
        x_validate = vectorizer.transform([train_texts[index] for index in validate])
        fold_matrices.append((fit, validate, x_fit, x_validate))

    candidates: list[tuple[Mapping[str, object], np.ndarray]] = []
    for kind in ("logistic", "linear_svc"):
        for c_value in _C_VALUES:
            oof = np.zeros(len(train_targets), dtype=float)
            for fit, validate, x_fit, x_validate in fold_matrices:
                estimator = _linear_estimator(kind, c_value)
                estimator.fit(x_fit, train_targets[fit])
                oof[validate] = _estimator_score(estimator, x_validate)
            candidates.append(({"kind": kind, "c": c_value}, oof))
    config, selection, _ = _select_candidate(train_targets, candidates, cap)
    vectorizer = _tfidf()
    x_train = vectorizer.fit_transform(train_texts)
    estimator = _linear_estimator(str(config["kind"]), float(config["c"]))
    estimator.fit(x_train, train_targets)
    held_score = float(_estimator_score(estimator, vectorizer.transform([held_text]))[0])
    return held_score, selection, config


def _prototype_oof(vectors: np.ndarray, targets: np.ndarray) -> np.ndarray:
    splitter = _inner_split(targets)
    oof = np.zeros(len(targets), dtype=float)
    for fit, validate in splitter.split(vectors, targets):
        for index in validate:
            oof[index] = prototype_probability(vectors[fit], targets[fit], vectors[index])
    return oof


def _prototype_outer(
    train_vectors: np.ndarray,
    train_targets: np.ndarray,
    held_vector: np.ndarray,
    cap: float,
) -> tuple[float, ThresholdSelection, Mapping[str, object], np.ndarray]:
    oof = _prototype_oof(train_vectors, train_targets)
    selection = _fast_threshold(train_targets, oof, cap)
    held_score = prototype_probability(train_vectors, train_targets, held_vector)
    return held_score, selection, {"kind": "cosine_centroid", "temperature": 5.0}, oof


def _dense_outer(
    train_vectors: np.ndarray,
    train_targets: np.ndarray,
    held_vector: np.ndarray,
    cap: float,
) -> tuple[float, ThresholdSelection, Mapping[str, object]]:
    splitter = _inner_split(train_targets)
    candidates: list[tuple[Mapping[str, object], np.ndarray]] = []
    for kind in ("logistic", "linear_svc"):
        for c_value in _C_VALUES:
            oof = np.zeros(len(train_targets), dtype=float)
            for fit, validate in splitter.split(train_vectors, train_targets):
                estimator = make_pipeline(
                    StandardScaler(), _linear_estimator(kind, c_value)
                )
                estimator.fit(train_vectors[fit], train_targets[fit])
                oof[validate] = _estimator_score(estimator, train_vectors[validate])
            candidates.append(({"kind": kind, "c": c_value}, oof))
    config, selection, _ = _select_candidate(train_targets, candidates, cap)
    estimator = make_pipeline(
        StandardScaler(), _linear_estimator(str(config["kind"]), float(config["c"]))
    )
    estimator.fit(train_vectors, train_targets)
    held_score = float(_estimator_score(estimator, held_vector.reshape(1, -1))[0])
    return held_score, selection, config


def _structured_matrix(
    records: Sequence[SemanticCalibrationRecord], names: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [
            [float(record.structured_features.get(name, 0.0)) for name in names]
            for record in records
        ],
        dtype=float,
    )


def _structured_estimator(config: Mapping[str, object]):
    if config["kind"] == "elastic_net":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(config["c"]),
                class_weight="balanced",
                l1_ratio=float(config["l1_ratio"]),
                max_iter=20_000,
                random_state=_SEED,
                solver="saga",
            ),
        )
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        early_stopping=False,
        l2_regularization=1.0,
        learning_rate=0.05,
        max_depth=int(config["max_depth"]),
        max_iter=100,
        max_leaf_nodes=5,
        min_samples_leaf=5,
        random_state=_SEED,
    )


def _structured_outer(
    train_records: Sequence[SemanticCalibrationRecord],
    train_targets: np.ndarray,
    held_record: SemanticCalibrationRecord,
    cap: float,
    *,
    family: str,
) -> tuple[float, ThresholdSelection, Mapping[str, object]]:
    names = tuple(
        sorted({name for record in train_records for name in record.structured_features})
    )
    x_train = _structured_matrix(train_records, names)
    x_held = _structured_matrix([held_record], names)
    if family == "structured_elastic":
        configs = [
            {"kind": "elastic_net", "c": c_value, "l1_ratio": ratio}
            for c_value in _C_VALUES
            for ratio in (0.25, 0.5, 0.75)
        ]
    else:
        configs = [
            {"kind": "gradient_boosting", "max_depth": depth} for depth in (1, 2)
        ]
    splitter = _inner_split(train_targets)
    candidates: list[tuple[Mapping[str, object], np.ndarray]] = []
    for config in configs:
        oof = np.zeros(len(train_targets), dtype=float)
        for fit, validate in splitter.split(x_train, train_targets):
            estimator = _structured_estimator(config)
            estimator.fit(x_train[fit], train_targets[fit])
            oof[validate] = _estimator_score(estimator, x_train[validate])
        candidates.append((config, oof))
    config, selection, _ = _select_candidate(train_targets, candidates, cap)
    estimator = _structured_estimator(config)
    estimator.fit(x_train, train_targets)
    held_score = float(_estimator_score(estimator, x_held)[0])
    return held_score, selection, config


def _fusion_outer(
    train_records: Sequence[SemanticCalibrationRecord],
    train_targets: np.ndarray,
    train_vectors: np.ndarray,
    held_record: SemanticCalibrationRecord,
    held_vector: np.ndarray,
    cap: float,
) -> tuple[float, ThresholdSelection, Mapping[str, object]]:
    _, _, _, prototype_oof = _prototype_outer(
        train_vectors, train_targets, held_vector, cap
    )
    raw = np.asarray([record.original_likelihood for record in train_records], dtype=float)
    held_prototype = prototype_probability(train_vectors, train_targets, held_vector)
    candidates = []
    for alpha in np.linspace(0.0, 1.0, 11):
        scores = alpha * raw + (1.0 - alpha) * prototype_oof
        candidates.append(({"raw_weight": float(alpha)}, scores))
    config, selection, _ = _select_candidate(train_targets, candidates, cap)
    alpha = float(config["raw_weight"])
    held_score = alpha * held_record.original_likelihood + (1.0 - alpha) * held_prototype
    return held_score, selection, config


@dataclass(frozen=True)
class PosthocPrediction:
    method: str
    episode_slug: str
    target: int
    score: float
    predicted: int
    threshold: float
    selected_config: Mapping[str, object]
    training_slugs: tuple[str, ...]


def nested_posthoc_predictions(
    records: Sequence[SemanticCalibrationRecord],
    embeddings: Mapping[str, np.ndarray],
    *,
    methods: Sequence[str] = _METHODS,
    false_positive_rate_cap: float = 0.20,
) -> dict[str, list[PosthocPrediction]]:
    """Generate outer-held-out predictions for each post-hoc method."""
    if len(records) < 6 or len({record.episode_slug for record in records}) != len(records):
        raise ValueError("semantic population must contain six unique episodes")
    unknown = set(methods) - set(_METHODS)
    if unknown:
        raise ValueError(f"unknown semantic methods: {sorted(unknown)}")
    for view in ("phase1_rationales", "full_reasoning"):
        if embeddings[view].shape[0] != len(records):
            raise ValueError(f"embedding row mismatch: {view}")
    result: dict[str, list[PosthocPrediction]] = {method: [] for method in methods}
    for method in methods:
        for held_index, held in enumerate(records):
            training_indices = [index for index in range(len(records)) if index != held_index]
            train_records = [records[index] for index in training_indices]
            train_targets = np.asarray([record.target for record in train_records], dtype=int)
            if method.startswith("tfidf"):
                view = "phase1_rationales" if method == "tfidf_phase1" else "full_reasoning"
                train_texts = [
                    record.phase1_text if view == "phase1_rationales" else record.full_text
                    for record in train_records
                ]
                held_text = held.phase1_text if view == "phase1_rationales" else held.full_text
                score, selection, config = _tfidf_outer(
                    train_texts, train_targets, held_text, false_positive_rate_cap
                )
            elif method == "embedding_prototype":
                vectors = embeddings["full_reasoning"]
                score, selection, config, _ = _prototype_outer(
                    vectors[training_indices],
                    train_targets,
                    vectors[held_index],
                    false_positive_rate_cap,
                )
            elif method == "embedding_logistic":
                vectors = embeddings["full_reasoning"]
                score, selection, config = _dense_outer(
                    vectors[training_indices],
                    train_targets,
                    vectors[held_index],
                    false_positive_rate_cap,
                )
            elif method in {"structured_elastic", "structured_boost"}:
                score, selection, config = _structured_outer(
                    train_records,
                    train_targets,
                    held,
                    false_positive_rate_cap,
                    family=method,
                )
            else:
                vectors = embeddings["full_reasoning"]
                score, selection, config = _fusion_outer(
                    train_records,
                    train_targets,
                    vectors[training_indices],
                    held,
                    vectors[held_index],
                    false_positive_rate_cap,
                )
            result[method].append(
                PosthocPrediction(
                    method=method,
                    episode_slug=held.episode_slug,
                    target=held.target,
                    score=float(score),
                    predicted=int(score >= selection.threshold),
                    threshold=selection.threshold,
                    selected_config=dict(config),
                    training_slugs=tuple(records[index].episode_slug for index in training_indices),
                )
            )
    return result


def _metrics(
    targets: np.ndarray, predicted: np.ndarray, scores: np.ndarray
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(targets, predicted, labels=[0, 1]).ravel()
    ranked = np.argsort(-scores, kind="stable")
    positives = int(targets.sum())
    ranking: dict[str, Any] = {}
    for budget in (1, 3, 5, 10, 20):
        actual = min(budget, len(targets))
        hits = int(targets[ranked[:actual]].sum())
        ranking[str(budget)] = {
            "hits": hits,
            "precision": hits / actual,
            "recall": hits / positives if positives else 0.0,
        }
    return {
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": float(accuracy_score(targets, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "in_precision": float(precision_score(targets, predicted, zero_division=0)),
        "in_recall": float(recall_score(targets, predicted, zero_division=0)),
        "in_f1": float(f1_score(targets, predicted, zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(targets, predicted)),
        "average_precision": float(average_precision_score(targets, scores)),
        "roc_auc": float(roc_auc_score(targets, scores)),
        "ranking": ranking,
    }


def evaluate_posthoc_predictions(
    records: Sequence[SemanticCalibrationRecord],
    predictions: Mapping[str, Sequence[PosthocPrediction]],
) -> dict[str, dict[str, Any]]:
    targets = np.asarray([record.target for record in records], dtype=int)
    metrics: dict[str, dict[str, Any]] = {}
    for method, rows in predictions.items():
        if len(rows) != len(records):
            raise ValueError(f"prediction count mismatch: {method}")
        metrics[method] = _metrics(
            targets,
            np.asarray([row.predicted for row in rows], dtype=int),
            np.asarray([row.score for row in rows], dtype=float),
        )
    return metrics


def _descending_rank_score(values: np.ndarray) -> np.ndarray:
    """Return deterministic 1..0 rank scores without consulting labels."""
    order = np.argsort(-np.asarray(values, dtype=float), kind="stable")
    result = np.empty(len(order), dtype=float)
    if len(order) == 1:
        result[order] = 1.0
    else:
        result[order] = 1.0 - (np.arange(len(order), dtype=float) / (len(order) - 1))
    return result


def evaluate_fixed_budget_shortlists(
    records: Sequence[SemanticCalibrationRecord],
    predictions: Mapping[str, Sequence[PosthocPrediction]],
    *,
    review_budget: int = 20,
) -> dict[str, dict[str, Any]]:
    """Evaluate label-blind batch shortlists over already held-out model scores.

    These policies are transductive deal-flow ranking rules, not pointwise
    classifiers.  Targets are consulted only after every shortlist is fixed.
    """
    if review_budget <= 0:
        raise ValueError("review_budget must be positive")
    required = {"tfidf_full", "embedding_logistic", "structured_boost"}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"missing shortlist inputs: {sorted(missing)}")

    slugs = [record.episode_slug for record in records]
    if len(slugs) != len(set(slugs)):
        raise ValueError("shortlist population must contain unique episodes")

    aligned: dict[str, list[PosthocPrediction]] = {}
    for method in required:
        by_slug = {row.episode_slug: row for row in predictions[method]}
        if set(by_slug) != set(slugs):
            raise ValueError(f"shortlist episode mismatch: {method}")
        aligned[method] = [by_slug[slug] for slug in slugs]

    embedding_scores = np.asarray(
        [row.score for row in aligned["embedding_logistic"]], dtype=float
    )
    structured_scores = np.asarray(
        [row.score for row in aligned["structured_boost"]], dtype=float
    )
    raw_scores = np.asarray([record.original_likelihood for record in records], dtype=float)
    embedding_ranks = _descending_rank_score(embedding_scores)
    policies: dict[str, tuple[np.ndarray, int, str]] = {
        "semantic_structured_rank_fusion": (
            (embedding_ranks + _descending_rank_score(structured_scores)) / 2.0,
            min(review_budget, len(records)),
            "equal-weight semantic and structured rank fusion",
        ),
        "raw_semantic_rank_fusion": (
            (_descending_rank_score(raw_scores) + embedding_ranks) / 2.0,
            min(review_budget, len(records)),
            "equal-weight raw-v4 likelihood and semantic rank fusion",
        ),
    }

    raw_selected = np.asarray(
        [record.original_decision == "In" for record in records], dtype=bool
    )
    tfidf_selected = np.asarray(
        [row.predicted == 1 for row in aligned["tfidf_full"]], dtype=bool
    )
    candidate = raw_selected | tfidf_selected
    policies["budget_matched_semantic_rescue"] = (
        embedding_ranks + (2.0 * candidate.astype(float)),
        int(raw_selected.sum()),
        "raw-v4-or-tfidf candidates reranked semantically at the raw-v4 review budget",
    )

    targets = np.asarray([record.target for record in records], dtype=int)
    result: dict[str, dict[str, Any]] = {}
    for method, (scores, budget, rule) in policies.items():
        order = np.argsort(-scores, kind="stable")
        selected = np.zeros(len(records), dtype=int)
        selected[order[:budget]] = 1
        rows = [
            {
                "method": method,
                "episode_slug": record.episode_slug,
                "target": record.target,
                "score": float(scores[index]),
                "rank": int(np.flatnonzero(order == index)[0]) + 1,
                "selected": int(selected[index]),
            }
            for index, record in enumerate(records)
        ]
        result[method] = {
            "review_budget": budget,
            "selection_rule": rule,
            "metrics": _metrics(targets, selected, scores),
            "rows": rows,
        }
    return result


def _raw_metrics(records: Sequence[SemanticCalibrationRecord]) -> dict[str, Any]:
    targets = np.asarray([record.target for record in records], dtype=int)
    return _metrics(
        targets,
        np.asarray([record.original_decision == "In" for record in records], dtype=int),
        np.asarray([record.original_likelihood for record in records], dtype=float),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def write_fixed_budget_shortlist_analysis(
    records: Sequence[SemanticCalibrationRecord],
    predictions: Mapping[str, Sequence[PosthocPrediction]],
    output_root: Path,
    *,
    review_budget: int = 20,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Write batch-prioritization results without retraining or LLM calls."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    evaluated = evaluate_fixed_budget_shortlists(
        records, predictions, review_budget=review_budget
    )
    methods = {
        method: {
            "review_budget": values["review_budget"],
            "selection_rule": values["selection_rule"],
            **values["metrics"],
        }
        for method, values in evaluated.items()
    }
    result = {
        "schema": "v4-fixed-budget-shortlists-v1",
        "evaluation": "label_blind_fixed_budget_batch_prioritization",
        "api_cost_usd": 0.0,
        "prediction_reruns": 0,
        "population": {
            "episodes": len(records),
            "ins": sum(record.target for record in records),
            "outs": len(records) - sum(record.target for record in records),
        },
        "methods": methods,
    }
    rows = [row for values in evaluated.values() for row in values["rows"]]
    manifest = {
        "schema": result["schema"],
        "evaluation": result["evaluation"],
        "api_cost_usd": 0.0,
        "prediction_reruns": 0,
        "pipeline_modified": False,
        "review_budget": review_budget,
        "sources": dict(source_manifest),
    }
    lines = [
        "# V4 fixed-budget batch prioritization",
        "",
        f"Population: {result['population']['episodes']} episodes "
        f"({result['population']['ins']} Ins, {result['population']['outs']} Outs).",
        "",
        "| Method | Budget | TP | FP | TN | FN | Balanced accuracy | Precision | Recall | F1 | AP | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in methods.items():
        matrix = values["confusion_matrix"]
        lines.append(
            f"| {method} | {values['review_budget']} | {matrix['tp']} | {matrix['fp']} | "
            f"{matrix['tn']} | {matrix['fn']} | {values['balanced_accuracy']:.3f} | "
            f"{values['in_precision']:.3f} | {values['in_recall']:.3f} | "
            f"{values['in_f1']:.3f} | {values['average_precision']:.3f} | "
            f"{values['roc_auc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "These are deterministic, label-blind shortlist rules applied to outer-held-out model scores.",
            "They evaluate batch prioritization, not a pointwise classifier or untouched holdout.",
            "No LLM calls or prediction reruns were performed.",
            "",
        ]
    )
    write_json(output / "manifest.json", manifest)
    write_json(output / "metrics.json", result)
    _write_csv(output / "shortlists.csv", rows)
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def _report(result: Mapping[str, Any]) -> str:
    lines = [
        "# V4 post-hoc semantic model comparison",
        "",
        f"Population: {result['population']['episodes']} episodes "
        f"({result['population']['ins']} Ins, {result['population']['outs']} Outs).",
        "",
        "Success requires TP > 6 and FP <= 12.",
        "",
        "| Method | TP | FP | TN | FN | Balanced accuracy | Precision | Recall | F1 | AP | AUC | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for method, metrics in result["methods"].items():
        matrix = metrics["confusion_matrix"]
        passed = matrix["tp"] >= result["success_criterion"]["minimum_tp"] and matrix["fp"] <= result["success_criterion"]["maximum_fp"]
        lines.append(
            f"| {method} | {matrix['tp']} | {matrix['fp']} | {matrix['tn']} | {matrix['fn']} | "
            f"{metrics['balanced_accuracy']:.3f} | {metrics['in_precision']:.3f} | "
            f"{metrics['in_recall']:.3f} | {metrics['in_f1']:.3f} | "
            f"{metrics['average_precision']:.3f} | {metrics['roc_auc']:.3f} | "
            f"{'YES' if passed else 'NO'} |"
        )
    winners = [
        name
        for name, metrics in result["methods"].items()
        if metrics["confusion_matrix"]["tp"] >= result["success_criterion"]["minimum_tp"]
        and metrics["confusion_matrix"]["fp"] <= result["success_criterion"]["maximum_fp"]
    ]
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                "Methods meeting the locked development criterion: " + ", ".join(winners) + "."
                if winners
                else "No method meets the locked development criterion."
            ),
            "",
            "These are nested leave-one-episode-out development results, not an untouched holdout estimate.",
            "No LLM calls or prediction reruns were performed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_posthoc_analysis(
    records: Sequence[SemanticCalibrationRecord],
    predictions: Mapping[str, Sequence[PosthocPrediction]],
    output_root: Path,
    *,
    embedding_metadata: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Write reproducibility artifacts for the frozen post-hoc benchmark."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    learned = evaluate_posthoc_predictions(records, predictions)
    methods = {"raw_v4": _raw_metrics(records), **learned}
    population = {
        "episodes": len(records),
        "ins": sum(record.target for record in records),
        "outs": len(records) - sum(record.target for record in records),
    }
    result = {
        "schema": "v4-posthoc-semantic-models-v1",
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
    }
    prediction_rows: list[dict[str, Any]] = []
    for method, rows in predictions.items():
        for row in rows:
            prediction_rows.append(
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
            )
    ranking_rows = [
        {"method": method, "budget": budget, **values}
        for method, metrics in methods.items()
        for budget, values in metrics["ranking"].items()
    ]
    write_json(output / "manifest.json", manifest)
    write_json(output / "embedding-metadata.json", dict(embedding_metadata))
    write_json(output / "metrics.json", result)
    _write_csv(output / "predictions.csv", prediction_rows)
    _write_csv(output / "rankings.csv", ranking_rows)
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
