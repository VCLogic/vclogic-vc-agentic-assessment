"""Leakage-controlled confirmation/rescue calibration for v4 decisions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

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
from .schemas_v4 import DecisionV4, InvestigationV4


FeatureMap = dict[str, float]
_C_VALUES = (0.01, 0.1, 1.0, 10.0)
_RANDOM_SEED = 20260808


def _token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "unknown"


def _increment(features: FeatureMap, key: str, value: float = 1.0) -> None:
    features[key] = features.get(key, 0.0) + float(value)


def extract_compact_v4_features(
    investigation: Mapping[str, Any],
    decision: Mapping[str, Any],
    batch_record: Mapping[str, Any],
) -> FeatureMap:
    """Extract compact aggregates without target labels, prose, or taxonomy labels."""
    features: FeatureMap = {
        "rationale_count": float(len(investigation.get("rationales", []))),
        "question_count": float(len(investigation.get("questions", []))),
        "conflict_count": float(len(investigation.get("conflicts", []))),
        "unmapped_count": float(len(investigation.get("unmapped_observations", []))),
        "phase1_searchable_count": float(
            len(investigation.get("searchable_questions", []))
        ),
        "phase1_diligence_count": float(
            len(investigation.get("diligence_questions", []))
        ),
        "phase1_next_search_count": float(
            len(investigation.get("next_search_objectives", []))
        ),
        "phase1_information_sufficient": float(
            bool(investigation.get("information_sufficient"))
        ),
        "decision_in": float(decision.get("decision") == "In"),
        "investment_likelihood": float(decision.get("investment_likelihood", 0.0)),
        "decision_confidence": float(decision.get("decision_confidence", 0.0)),
        "review_priority_score": float(decision.get("review_priority_score", 0.0)),
        "controlling_rationale_count": float(
            len(decision.get("controlling_rationale_ids", []))
        ),
        "phase2_searchable_count": float(len(decision.get("searchable_questions", []))),
        "phase2_diligence_count": float(len(decision.get("diligence_questions", []))),
        "reversal_condition_count": float(len(decision.get("reversal_conditions", []))),
        "phase2_next_search_count": float(
            len(decision.get("next_search_objectives", []))
        ),
        "missing_consideration_count": float(
            len(decision.get("missing_considerations", []))
        ),
        "phase2_information_sufficient": float(
            bool(decision.get("information_sufficient"))
        ),
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
        "phase1_provisional": float(batch_record.get("phase1_status") == "provisional"),
        "phase2_provisional": float(batch_record.get("phase2_status") == "provisional"),
        "phase1_finding_count": float(len(batch_record.get("phase1_findings", []))),
        "phase2_finding_count": float(len(batch_record.get("phase2_findings", []))),
    }
    decision_in = features["decision_in"]
    features["signed_decision_confidence"] = features["decision_confidence"] * (
        1.0 if decision_in else -1.0
    )

    for direction in ("positive", "neutral", "negative"):
        for salience in ("primary", "secondary"):
            features[f"{direction}_{salience}_count"] = 0.0
            features[f"{direction}_{salience}_confidence_sum"] = 0.0
    for question in investigation.get("questions", []):
        _increment(features, f"question_status__{_token(question.get('status'))}__count")
    direction_weights = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    for rationale in investigation.get("rationales", []):
        direction = str(rationale.get("direction"))
        salience = str(rationale.get("salience"))
        confidence = float(rationale.get("confidence", 0.0))
        if direction not in direction_weights or salience not in {"primary", "secondary"}:
            raise ValueError("invalid v4 rationale direction or salience")
        _increment(features, f"{direction}_{salience}_count")
        _increment(features, f"{direction}_{salience}_confidence_sum", confidence)
        _increment(
            features,
            f"{salience}_signed_confidence_sum",
            direction_weights[direction] * confidence,
        )
        _increment(features, "pitch_evidence_count", len(rationale.get("pitch_evidence_ids", [])))
        _increment(features, "wiki_evidence_count", len(rationale.get("wiki_evidence_ids", [])))
        _increment(
            features,
            "historical_evidence_count",
            len(rationale.get("historical_evidence_ids", [])),
        )

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


@dataclass(frozen=True)
class V4CalibrationRecord:
    """One verified v4 episode with label-blind compact features."""

    episode_slug: str
    target: int
    target_decision: str
    original_decision: str
    original_likelihood: float
    review_priority_score: float
    features: Mapping[str, float]
    artifact_root: str
    phase1_sha256: str
    phase2_sha256: str


def conditional_head_targets(
    records: Sequence[V4CalibrationRecord], head: str
) -> tuple[int, ...]:
    """Return the error event each conditional head is responsible for detecting."""
    if head == "confirmation":
        return tuple(1 - record.target for record in records)
    if head == "rescue":
        return tuple(record.target for record in records)
    raise ValueError(f"unknown two-head calibration branch: {head}")


def _load_labels(labels_path: Path) -> dict[str, str]:
    payload = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    if type(payload) is not list:
        raise ValueError("label file must contain a list")
    labels: dict[str, str] = {}
    for row in payload:
        if row.get("evaluation_eligible") is not True:
            continue
        slug = str(row.get("episode_slug", ""))
        decision = row.get("pitch_window_decision")
        if not slug or decision not in {"In", "Out"} or slug in labels:
            raise ValueError("eligible label row is invalid or duplicated")
        labels[slug] = str(decision)
    return labels


def load_v4_calibration_population(
    status_path: Path,
    labels_path: Path,
) -> list[V4CalibrationRecord]:
    """Load strictly completed v4 records and verify their frozen artifacts."""
    labels = _load_labels(labels_path)
    status = json.loads(Path(status_path).read_text(encoding="utf-8"))
    completed = status.get("completed_records")
    if type(completed) is not list:
        raise ValueError("batch status has no completed_records list")
    records: list[V4CalibrationRecord] = []
    seen: set[str] = set()
    for batch_record in completed:
        slug = str(batch_record.get("episode_slug", ""))
        if not slug or slug in seen:
            raise ValueError(f"duplicate or missing episode slug: {slug!r}")
        if batch_record.get("status") != "completed":
            raise ValueError(f"completed record has non-completed status: {slug}")
        if slug not in labels:
            raise ValueError(f"completed episode lacks an eligible label: {slug}")
        seen.add(slug)
        artifact_root = Path(str(batch_record["artifact_root"]))
        investigation_raw, investigation_digest = read_verified_frozen(
            artifact_root / "phase1/investigation.json",
            artifact_root / "phase1/investigation.sha256",
        )
        decision_raw, decision_digest = read_verified_frozen(
            artifact_root / "phase2/decision.json",
            artifact_root / "phase2/decision.sha256",
        )
        investigation = InvestigationV4.model_validate_json(investigation_raw)
        decision = DecisionV4.model_validate_json(decision_raw)
        if investigation.episode_slug != slug or decision.episode_slug != slug:
            raise ValueError(f"artifact episode mismatch: {slug}")
        if decision.investigation_sha256 != investigation_digest:
            raise ValueError(f"decision Phase 1 digest mismatch: {slug}")
        investigation_dict = investigation.model_dump(mode="json")
        decision_dict = decision.model_dump(mode="json")
        target_decision = labels[slug]
        records.append(
            V4CalibrationRecord(
                episode_slug=slug,
                target=int(target_decision == "In"),
                target_decision=target_decision,
                original_decision=decision.decision,
                original_likelihood=decision.investment_likelihood,
                review_priority_score=decision.review_priority_score,
                features=extract_compact_v4_features(
                    investigation_dict, decision_dict, batch_record
                ),
                artifact_root=str(artifact_root),
                phase1_sha256=investigation_digest,
                phase2_sha256=decision_digest,
            )
        )
    return sorted(records, key=lambda record: record.episode_slug)


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    balanced_accuracy: float
    false_positive_rate: float


def _counts(targets: np.ndarray, predicted: np.ndarray) -> tuple[int, int, int, int]:
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        targets, predicted, labels=[0, 1]
    ).ravel()
    return (
        int(true_positives),
        int(false_positives),
        int(true_negatives),
        int(false_negatives),
    )


def select_capped_threshold(
    targets: Sequence[int],
    scores: Sequence[float],
    *,
    false_positive_rate_cap: float,
) -> ThresholdSelection:
    """Maximize recall subject to an explicit training false-positive-rate cap."""
    if not 0.0 <= false_positive_rate_cap <= 1.0:
        raise ValueError("false_positive_rate_cap must be between zero and one")
    y_true = np.asarray(targets, dtype=int)
    values = np.asarray(scores, dtype=float)
    if len(y_true) != len(values) or not len(y_true):
        raise ValueError("targets and scores must be nonempty and equally sized")
    if set(y_true) - {0, 1}:
        raise ValueError("targets must be binary")
    candidates = sorted({0.0, 1.0000001, *(float(value) for value in values)})
    best: tuple[tuple[float, ...], ThresholdSelection] | None = None
    for threshold in candidates:
        predicted = (values >= threshold).astype(int)
        tp, fp, tn, fn = _counts(y_true, predicted)
        false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
        if false_positive_rate > false_positive_rate_cap + 1e-12:
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
            balanced_accuracy=(recall + specificity) / 2,
            false_positive_rate=false_positive_rate,
        )
        key = (recall, precision, selection.balanced_accuracy, threshold)
        if best is None or key > best[0]:
            best = (key, selection)
    if best is None:
        raise RuntimeError("no threshold satisfies the false-positive-rate cap")
    return best[1]


@dataclass(frozen=True)
class TwoHeadPrediction:
    episode_slug: str
    target: int
    original_decision: str
    original_likelihood: float
    review_priority_score: float
    head: str
    score: float
    threshold: float
    predicted: int
    chosen_c: float
    false_positive_rate_cap: float
    fallback_reason: str | None
    training_slugs: tuple[str, ...]
    head_training_slugs: tuple[str, ...]


def _feature_names(records: Sequence[V4CalibrationRecord]) -> tuple[str, ...]:
    return tuple(sorted({name for record in records for name in record.features}))


def _matrix(
    records: Sequence[V4CalibrationRecord], names: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [[float(record.features.get(name, 0.0)) for name in names] for record in records],
        dtype=float,
    )


def _model(c_value: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=20_000,
            random_state=_RANDOM_SEED,
        ),
    )


def _fit_conditional_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_held: np.ndarray,
    *,
    false_positive_rate_cap: float,
) -> tuple[float, ThresholdSelection, float] | None:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    split_count = min(3, positives, negatives)
    if split_count < 2:
        return None
    splitter = StratifiedKFold(
        n_splits=split_count, shuffle=True, random_state=_RANDOM_SEED
    )
    best: tuple[tuple[float, ...], float, ThresholdSelection] | None = None
    for c_value in _C_VALUES:
        out_of_fold = np.zeros(len(y_train), dtype=float)
        for fit_indices, validation_indices in splitter.split(x_train, y_train):
            estimator = _model(c_value)
            estimator.fit(x_train[fit_indices], y_train[fit_indices])
            out_of_fold[validation_indices] = estimator.predict_proba(
                x_train[validation_indices]
            )[:, 1]
        selection = select_capped_threshold(
            y_train,
            out_of_fold,
            false_positive_rate_cap=false_positive_rate_cap,
        )
        key = (
            selection.recall,
            selection.precision,
            selection.balanced_accuracy,
            selection.threshold,
            -c_value,
        )
        if best is None or key > best[0]:
            best = (key, c_value, selection)
    assert best is not None
    _, chosen_c, selection = best
    estimator = _model(chosen_c)
    estimator.fit(x_train, y_train)
    held_score = float(estimator.predict_proba(x_held)[0, 1])
    return held_score, selection, chosen_c


def nested_two_head_leave_one_out(
    records: Sequence[V4CalibrationRecord],
    *,
    false_positive_rate_cap: float,
) -> list[TwoHeadPrediction]:
    """Evaluate the conditional cascade with all selection inside each outer fold."""
    if len(records) < 5:
        raise ValueError("two-head leave-one-out evaluation needs at least five records")
    names = _feature_names(records)
    x_all = _matrix(records, names)
    predictions: list[TwoHeadPrediction] = []
    for held_index, held in enumerate(records):
        training_indices = [index for index in range(len(records)) if index != held_index]
        head = "confirmation" if held.original_decision == "In" else "rescue"
        head_decision = held.original_decision
        head_indices = [
            index
            for index in training_indices
            if records[index].original_decision == head_decision
        ]
        head_records = [records[index] for index in head_indices]
        y_train = np.asarray(conditional_head_targets(head_records, head), dtype=int)
        fitted = _fit_conditional_head(
            x_all[head_indices],
            y_train,
            x_all[[held_index]],
            false_positive_rate_cap=false_positive_rate_cap,
        )
        fallback_reason: str | None = None
        if fitted is None:
            fallback_reason = "insufficient_conditional_classes"
            score = held.original_likelihood
            threshold = 0.5
            chosen_c = 0.0
            predicted = int(held.original_decision == "In")
        else:
            score, selection, chosen_c = fitted
            threshold = selection.threshold
            event_detected = score >= threshold
            predicted = int(not event_detected) if head == "confirmation" else int(event_detected)
        predictions.append(
            TwoHeadPrediction(
                episode_slug=held.episode_slug,
                target=held.target,
                original_decision=held.original_decision,
                original_likelihood=held.original_likelihood,
                review_priority_score=held.review_priority_score,
                head=head,
                score=score,
                threshold=threshold,
                predicted=predicted,
                chosen_c=chosen_c,
                false_positive_rate_cap=false_positive_rate_cap,
                fallback_reason=fallback_reason,
                training_slugs=tuple(records[index].episode_slug for index in training_indices),
                head_training_slugs=tuple(records[index].episode_slug for index in head_indices),
            )
        )
    return predictions


def _transition(record: V4CalibrationRecord, predicted: int) -> str:
    original = int(record.original_decision == "In")
    target = record.target
    table = {
        (1, 1, 1): "preserved_correct_in",
        (1, 1, 0): "lost_correct_in",
        (1, 0, 0): "corrected_false_in",
        (1, 0, 1): "remaining_false_in",
        (0, 1, 1): "rescued_in",
        (0, 1, 0): "remaining_missed_in",
        (0, 0, 0): "preserved_correct_out",
        (0, 0, 1): "introduced_false_in",
    }
    return table[(original, target, predicted)]


def _classification_metrics(
    targets: np.ndarray, predicted: np.ndarray, ranking_scores: np.ndarray
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(targets, predicted, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": float(accuracy_score(targets, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "in_precision": float(precision_score(targets, predicted, zero_division=0)),
        "in_recall": float(recall_score(targets, predicted, zero_division=0)),
        "in_f1": float(f1_score(targets, predicted, zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(targets, predicted)),
        "average_precision": float(average_precision_score(targets, ranking_scores)),
        "roc_auc": float(roc_auc_score(targets, ranking_scores)),
    }
    ranked_indices = np.argsort(-ranking_scores, kind="stable")
    positives = int(targets.sum())
    metrics["ranking"] = {}
    for budget in (1, 3, 5, 10, 20):
        actual_budget = min(budget, len(targets))
        hits = int(targets[ranked_indices[:actual_budget]].sum())
        metrics["ranking"][str(budget)] = {
            "hits": hits,
            "precision": hits / actual_budget,
            "recall": hits / positives if positives else 0.0,
        }
    return metrics


def evaluate_two_head_predictions(
    records: Sequence[V4CalibrationRecord],
    predictions: Sequence[TwoHeadPrediction],
) -> dict[str, Any]:
    """Measure final decisions and explicitly account for every transition."""
    if len(records) != len(predictions) or not records:
        raise ValueError("records and predictions must be nonempty and aligned")
    by_slug = {record.episode_slug: record for record in records}
    targets = np.asarray([prediction.target for prediction in predictions], dtype=int)
    predicted = np.asarray([prediction.predicted for prediction in predictions], dtype=int)
    ranking_scores = np.asarray(
        [prediction.original_likelihood for prediction in predictions], dtype=float
    )
    result = _classification_metrics(targets, predicted, ranking_scores)
    transitions = {
        name: 0
        for name in (
            "preserved_correct_in",
            "lost_correct_in",
            "corrected_false_in",
            "remaining_false_in",
            "rescued_in",
            "remaining_missed_in",
            "preserved_correct_out",
            "introduced_false_in",
        )
    }
    episodes = {name: [] for name in transitions}
    for prediction in predictions:
        record = by_slug[prediction.episode_slug]
        name = _transition(record, prediction.predicted)
        transitions[name] += 1
        episodes[name].append(prediction.episode_slug)
    result["transitions"] = transitions
    result["transition_episodes"] = episodes
    result["ranking_score"] = "raw_v4_investment_likelihood"
    head_diagnostics: dict[str, Any] = {}
    for head, positive_event in (
        ("confirmation", "false_in"),
        ("rescue", "missed_in"),
    ):
        head_rows = [prediction for prediction in predictions if prediction.head == head]
        event_targets = np.asarray(
            [
                1 - prediction.target if head == "confirmation" else prediction.target
                for prediction in head_rows
            ],
            dtype=int,
        )
        event_scores = np.asarray([prediction.score for prediction in head_rows], dtype=float)
        diagnostic: dict[str, Any] = {
            "positive_event": positive_event,
            "cases": len(head_rows),
            "positives": int(event_targets.sum()),
        }
        if len(set(event_targets)) == 2:
            diagnostic["average_precision"] = float(
                average_precision_score(event_targets, event_scores)
            )
            diagnostic["roc_auc"] = float(roc_auc_score(event_targets, event_scores))
        else:
            diagnostic["average_precision"] = None
            diagnostic["roc_auc"] = None
        head_diagnostics[head] = diagnostic
    result["head_diagnostics"] = head_diagnostics
    return result


def _raw_metrics(records: Sequence[V4CalibrationRecord]) -> dict[str, Any]:
    targets = np.asarray([record.target for record in records], dtype=int)
    predicted = np.asarray(
        [int(record.original_decision == "In") for record in records], dtype=int
    )
    scores = np.asarray([record.original_likelihood for record in records], dtype=float)
    return _classification_metrics(targets, predicted, scores)


def _cap_name(cap: float) -> str:
    return f"two_head_fpr_{cap:.2f}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty predictions CSV")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, tuple, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _report(result: Mapping[str, Any]) -> str:
    population = result["population"]
    lines = [
        "# V4 two-head calibration",
        "",
        f"Population: {population['episodes']} strict episodes "
        f"({population['ins']} Ins, {population['outs']} Outs).",
        "",
        "| Method | TP | FP | TN | FN | Balanced accuracy | In precision | In recall | In F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in result["methods"].items():
        matrix = metrics["confusion_matrix"]
        lines.append(
            f"| {name} | {matrix['tp']} | {matrix['fp']} | {matrix['tn']} | "
            f"{matrix['fn']} | {metrics['balanced_accuracy']:.3f} | "
            f"{metrics['in_precision']:.3f} | {metrics['in_recall']:.3f} | "
            f"{metrics['in_f1']:.3f} |"
        )
    learned_methods = [
        (name, metrics)
        for name, metrics in result["methods"].items()
        if name != "raw_v4"
    ]
    lines.extend(["", "## Decision transitions", ""])
    for name, metrics in learned_methods:
        transitions = metrics["transitions"]
        confirmation = metrics["head_diagnostics"]["confirmation"]
        rescue = metrics["head_diagnostics"]["rescue"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Preserved correct Ins: {transitions['preserved_correct_in']}",
                f"- Lost correct Ins: {transitions['lost_correct_in']}",
                f"- Corrected false Ins: {transitions['corrected_false_in']}",
                f"- Remaining false Ins: {transitions['remaining_false_in']}",
                f"- Rescued Ins: {transitions['rescued_in']}",
                f"- Introduced false Ins: {transitions['introduced_false_in']}",
                f"- Remaining missed Ins: {transitions['remaining_missed_in']}",
                f"- Confirmation error-detection AUC: {confirmation['roc_auc']:.3f}",
                f"- Rescue AUC: {rescue['roc_auc']:.3f}",
                "",
            ]
        )
    corrected = max(
        metrics["transitions"]["corrected_false_in"]
        for _, metrics in learned_methods
    )
    introduced = min(
        metrics["transitions"]["introduced_false_in"]
        for _, metrics in learned_methods
    )
    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "No tested policy corrected an original false In. "
                if corrected == 0
                else f"The strongest policy corrected {corrected} original false Ins. "
            )
            + f"The least permissive cascade still introduced {introduced} new false Ins through rescue. "
            + "The compact frozen features therefore do not support the intended false-In suppression objective on this development population.",
            "",
            "Higher recall in the cascade must not be described as an overall improvement without reporting the additional false Ins.",
            "",
            "",
            "Every learned result is outer leave-one-episode-out development evaluation.",
            "Thresholds and regularization are selected only from each outer training fold.",
            "Ranking metrics retain raw v4 investment likelihood because conditional head scores are not cross-head comparable.",
            "No LLM or network call is used.",
            "",
        ]
    )
    return "\n".join(lines)


def write_two_head_analysis(
    records: Sequence[V4CalibrationRecord],
    output_root: Path,
    *,
    false_positive_rate_caps: Sequence[float],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Run all capped policies and write reproducibility artifacts."""
    if not records or not false_positive_rate_caps:
        raise ValueError("records and false-positive-rate caps cannot be empty")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    methods: dict[str, Any] = {"raw_v4": _raw_metrics(records)}
    prediction_rows: list[dict[str, Any]] = []
    for cap in false_positive_rate_caps:
        predictions = nested_two_head_leave_one_out(
            records, false_positive_rate_cap=float(cap)
        )
        name = _cap_name(float(cap))
        methods[name] = evaluate_two_head_predictions(records, predictions)
        for prediction in predictions:
            prediction_rows.append(
                {
                    "method": name,
                    "episode_slug": prediction.episode_slug,
                    "target": prediction.target,
                    "original_decision": prediction.original_decision,
                    "original_likelihood": prediction.original_likelihood,
                    "review_priority_score": prediction.review_priority_score,
                    "head": prediction.head,
                    "score": prediction.score,
                    "threshold": prediction.threshold,
                    "predicted": prediction.predicted,
                    "chosen_c": prediction.chosen_c,
                    "false_positive_rate_cap": prediction.false_positive_rate_cap,
                    "fallback_reason": prediction.fallback_reason or "",
                    "training_slugs": prediction.training_slugs,
                    "head_training_slugs": prediction.head_training_slugs,
                }
            )
    population = {
        "episodes": len(records),
        "ins": sum(record.target for record in records),
        "outs": len(records) - sum(record.target for record in records),
    }
    result = {
        "schema": "v4-two-head-calibration-v1",
        "evaluation": "nested_leave_one_out_development_not_untouched_holdout",
        "population": population,
        "methods": methods,
    }
    manifest = {
        "schema": result["schema"],
        "api_cost_usd": 0.0,
        "random_seed": _RANDOM_SEED,
        "regularization_grid": list(_C_VALUES),
        "false_positive_rate_caps": [float(cap) for cap in false_positive_rate_caps],
        "population": population,
        "sources": dict(source_manifest),
        "feature_names": list(_feature_names(records)),
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "metrics.json", result)
    _write_csv(output / "predictions.csv", prediction_rows)
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
