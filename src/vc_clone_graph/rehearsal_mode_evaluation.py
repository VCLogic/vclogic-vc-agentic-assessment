"""Classification, ranking, usage, and audit evaluation for rehearsal modes."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class RehearsalPrediction:
    mode: str
    method: str
    episode_slug: str
    target: int
    score: float
    predicted: int
    artifact_id: str | None


@dataclass(frozen=True)
class RehearsalMetric:
    method: str
    cases: int
    ins: int
    balanced_accuracy: float
    in_precision: float
    in_recall: float
    in_f1: float
    average_precision: float
    roc_auc: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float


@dataclass(frozen=True)
class RehearsalAudit:
    replay_count: int
    fallback_count: int
    total_questions: int
    accepted_answers: int
    total_cost_usd: float


@dataclass(frozen=True)
class RehearsalEvaluation:
    predictions: tuple[RehearsalPrediction, ...]
    metrics: tuple[RehearsalMetric, ...]
    audit: RehearsalAudit


def _snapshot(
    classification: Mapping[str, object], stage: str
) -> Mapping[str, object] | None:
    rows = classification.get("snapshots", [])
    if not isinstance(rows, list):
        return None
    return next(
        (
            row
            for row in rows
            if isinstance(row, dict) and row.get("stage") == stage
        ),
        None,
    )


def _recall_at(targets: np.ndarray, scores: np.ndarray, k: int) -> float:
    positives = int(targets.sum())
    if positives == 0:
        return 0.0
    ordered = np.argsort(-scores, kind="stable")[: min(k, len(scores))]
    return float(targets[ordered].sum() / positives)


def _metric(method: str, rows: Sequence[RehearsalPrediction]) -> RehearsalMetric:
    targets = np.asarray([row.target for row in rows], dtype=int)
    predicted = np.asarray([row.predicted for row in rows], dtype=int)
    scores = np.asarray([row.score for row in rows], dtype=float)
    both_classes = len(set(int(value) for value in targets)) == 2
    return RehearsalMetric(
        method=method,
        cases=len(rows),
        ins=int(targets.sum()),
        balanced_accuracy=(
            float(balanced_accuracy_score(targets, predicted))
            if both_classes
            else float("nan")
        ),
        in_precision=float(precision_score(targets, predicted, zero_division=0)),
        in_recall=float(recall_score(targets, predicted, zero_division=0)),
        in_f1=float(f1_score(targets, predicted, zero_division=0)),
        average_precision=(
            float(average_precision_score(targets, scores))
            if int(targets.sum())
            else float("nan")
        ),
        roc_auc=(
            float(roc_auc_score(targets, scores))
            if both_classes
            else float("nan")
        ),
        recall_at_5=_recall_at(targets, scores, 5),
        recall_at_10=_recall_at(targets, scores, 10),
        recall_at_20=_recall_at(targets, scores, 20),
    )


def evaluate_replay_files(
    paths_by_mode: Mapping[str, Sequence[Path]],
) -> RehearsalEvaluation:
    predictions: list[RehearsalPrediction] = []
    fallback_count = 0
    total_questions = 0
    accepted_answers = 0
    total_cost = 0.0
    replay_count = 0
    seen: set[tuple[str, str]] = set()
    for mode, paths in sorted(paths_by_mode.items()):
        for path in paths:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if payload.get("status") != "complete":
                continue
            episode = str(payload["episode_slug"])
            identity = (mode, episode)
            if identity in seen:
                raise ValueError(f"duplicate replay for mode and episode: {identity}")
            seen.add(identity)
            replay_count += 1
            target = int(payload.get("actual_decision") == "In")
            direct_score = float(payload.get("investment_likelihood", 0.0))
            direct_predicted = int(payload.get("predicted_decision") == "In")
            founder_report_path = Path(path).parent / "founder-report.json"
            founder_report = (
                json.loads(founder_report_path.read_text(encoding="utf-8"))
                if founder_report_path.is_file()
                else {}
            )
            grounded = founder_report.get("grounded")
            if isinstance(grounded, dict) and isinstance(
                grounded.get("baseline"), dict
            ):
                baseline = grounded["baseline"]
                predictions.append(
                    RehearsalPrediction(
                        mode=mode,
                        method="canonical_v41_baseline",
                        episode_slug=episode,
                        target=target,
                        score=float(baseline["investment_likelihood"]),
                        predicted=int(baseline["decision"] == "In"),
                        artifact_id=str(baseline["phase2"]["sha256"]),
                    )
                )
            predictions.append(
                RehearsalPrediction(
                    mode=mode,
                    method=(
                        "v41_grounded_final"
                        if isinstance(grounded, dict)
                        else f"{mode}_direct"
                    ),
                    episode_slug=episode,
                    target=target,
                    score=direct_score,
                    predicted=direct_predicted,
                    artifact_id=None,
                )
            )
            turns = payload.get("replay_turns", [])
            if isinstance(turns, list):
                total_questions += len(turns)
                accepted_answers += sum(
                    bool(row.get("accepted")) for row in turns if isinstance(row, dict)
                )
            usage = payload.get("total_usage", {})
            if isinstance(usage, dict):
                total_cost += float(usage.get("cost_usd", 0.0))
            classification = payload.get("classification")
            if not isinstance(classification, dict):
                continue
            if classification.get("status") == "fallback":
                fallback_count += 1
                continue
            artifact_id = (
                str(classification["artifact_id"])
                if classification.get("artifact_id")
                else None
            )
            for stage, method in (
                ("initial", "static_classifier"),
                ("final", f"{mode}_classifier"),
            ):
                row = _snapshot(classification, stage)
                if row is None:
                    continue
                predictions.append(
                    RehearsalPrediction(
                        mode=mode,
                        method=method,
                        episode_slug=episode,
                        target=target,
                        score=float(row["probability_in"]),
                        predicted=int(row["predicted_decision"] == "In"),
                        artifact_id=artifact_id,
                    )
                )
    metrics = tuple(
        _metric(method, [row for row in predictions if row.method == method])
        for method in sorted({row.method for row in predictions})
    )
    return RehearsalEvaluation(
        predictions=tuple(predictions),
        metrics=metrics,
        audit=RehearsalAudit(
            replay_count=replay_count,
            fallback_count=fallback_count,
            total_questions=total_questions,
            accepted_answers=accepted_answers,
            total_cost_usd=total_cost,
        ),
    )


def write_evaluation_outputs(
    output_root: Path, result: RehearsalEvaluation
) -> dict[str, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(asdict(result.metrics[0])),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in result.metrics)
    predictions_path = root / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(asdict(result.predictions[0])),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in result.predictions)
    report_path = root / "report.md"
    lines = [
        "# Rehearsal mode evaluation",
        "",
        f"Replays: {result.audit.replay_count}; fallbacks: {result.audit.fallback_count}; "
        f"questions: {result.audit.total_questions}; accepted answers: "
        f"{result.audit.accepted_answers}; cost: ${result.audit.total_cost_usd:.4f}.",
        "",
        "**Engineering-canary warning:** this sample is too small for significance claims or stable predictive-performance estimates.",
        "",
        "| Method | Cases | Ins | Balanced accuracy | In precision | In recall | AP | ROC AUC | R@5 | R@10 | R@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.metrics:
        lines.append(
            f"| {row.method} | {row.cases} | {row.ins} | {row.balanced_accuracy:.3f} | "
            f"{row.in_precision:.3f} | {row.in_recall:.3f} | "
            f"{row.average_precision:.3f} | {row.roc_auc:.3f} | "
            f"{row.recall_at_5:.3f} | {row.recall_at_10:.3f} | {row.recall_at_20:.3f} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "metrics_csv": metrics_path,
        "predictions_csv": predictions_path,
        "report_md": report_path,
    }
