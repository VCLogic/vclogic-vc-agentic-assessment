"""Co-primary evaluation and paired uncertainty for personalized models."""

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
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .phase2_calibration_evaluation import holm_adjust


LOCKED_BENCHMARKS = {
    "raw_phase2_macro_balanced_accuracy": 0.545,
    "raw_phase2_in_precision": 0.371,
    "raw_phase2_in_recall": 0.496,
    "raw_phase2_in_f1": 0.417,
    "nominal_rationale_ranking_ap": 0.461,
    "rationale_reranker_precision_at_5": 0.500,
    "rationale_reranker_recall_at_5": 0.195,
}


@dataclass(frozen=True)
class PredictionRow:
    method: str
    family: str
    condition: str
    vc_slug: str
    vc_name: str
    episode_slug: str
    target: int
    classification_score: float
    ranking_score: float
    predicted: int
    status: str
    runtime_seconds: float = 0.0


def _classification(rows: Sequence[PredictionRow]) -> dict[str, float | int]:
    y = np.asarray([row.target for row in rows], dtype=int)
    predicted = np.asarray([row.predicted for row in rows], dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "n": len(rows), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "in_precision": float(precision_score(y, predicted, zero_division=0)),
        "in_recall": float(recall_score(y, predicted, zero_division=0)),
        "in_f1": float(f1_score(y, predicted, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "accuracy": float(np.mean(y == predicted)),
    }


def _ece(targets: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    total = len(targets)
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (scores >= edges[index]) & (
            scores <= edges[index + 1] if index == bins - 1 else scores < edges[index + 1]
        )
        if selected.any():
            value += float(selected.sum() / total) * abs(
                float(scores[selected].mean()) - float(targets[selected].mean())
            )
    return value


def _ranking(
    rows: Sequence[PredictionRow], budgets: Sequence[int]
) -> dict[str, float | int]:
    y = np.asarray([row.target for row in rows], dtype=int)
    scores = np.asarray([row.ranking_score for row in rows], dtype=float)
    result: dict[str, float | int] = {
        "n": len(rows),
        "in_count": int(y.sum()),
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(set(y.tolist())) == 2 else 0.5,
    }
    order = np.argsort(-scores, kind="stable")
    for budget in budgets:
        selected = order[:min(budget, len(order))]
        found = int(y[selected].sum())
        result[f"ins_at_{budget}"] = found
        result[f"precision_at_{budget}"] = float(found / max(1, len(selected)))
        result[f"recall_at_{budget}"] = float(found / max(1, int(y.sum())))
    return result


def _macro(
    per_vc: Mapping[str, Mapping[str, float | int]], keys: Sequence[str]
) -> dict[str, float]:
    return {
        key: float(np.mean([float(row[key]) for row in per_vc.values()]))
        for key in keys
    }


def _method_metrics(
    rows: Sequence[PredictionRow], budgets: Sequence[int]
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    by_vc = {
        vc: [row for row in rows if row.vc_slug == vc]
        for vc in sorted({row.vc_slug for row in rows})
    }
    classification = {vc: _classification(selected) for vc, selected in by_vc.items()}
    ranking = {vc: _ranking(selected, budgets) for vc, selected in by_vc.items()}
    classification_keys = (
        "balanced_accuracy", "in_precision", "in_recall", "in_f1",
        "specificity", "accuracy",
    )
    ranking_keys = (
        "average_precision", "roc_auc",
        *(name for budget in budgets for name in (
            f"precision_at_{budget}", f"recall_at_{budget}", f"ins_at_{budget}"
        )),
    )
    macro_classification = _macro(classification, classification_keys)
    macro_ranking = _macro(ranking, ranking_keys)
    y = np.asarray([row.target for row in rows], dtype=int)
    score = np.asarray([row.classification_score for row in rows], dtype=float)
    calibration = {
        "brier_score": float(brier_score_loss(y, score)),
        "expected_calibration_error": _ece(y, score),
    }
    return (
        {"macro": macro_classification, "per_vc": classification, "pooled": _classification(rows)},
        {"macro": macro_ranking, "per_vc": ranking, "pooled": _ranking(rows, budgets)},
        calibration,
    )


def _metric(rows: Sequence[PredictionRow], kind: str) -> float:
    by_vc = {
        vc: [row for row in rows if row.vc_slug == vc]
        for vc in sorted({row.vc_slug for row in rows})
    }
    if kind == "classification":
        values = []
        for selected in by_vc.values():
            targets = np.asarray([row.target for row in selected], dtype=int)
            if len(set(targets.tolist())) != 2:
                continue
            predicted = np.asarray([row.predicted for row in selected], dtype=int)
            values.append(float(balanced_accuracy_score(targets, predicted)))
        return float(np.mean(values)) if values else 0.5
    values = []
    for selected in by_vc.values():
        targets = np.asarray([row.target for row in selected], dtype=int)
        if len(set(targets.tolist())) != 2:
            continue
        scores = np.asarray([row.ranking_score for row in selected], dtype=float)
        values.append(float(average_precision_score(targets, scores)))
    return float(np.mean(values)) if values else 0.0


def _aligned(
    candidate: Sequence[PredictionRow], baseline: Sequence[PredictionRow]
) -> tuple[list[PredictionRow], list[PredictionRow]]:
    left = {(row.vc_slug, row.episode_slug): row for row in candidate}
    right = {(row.vc_slug, row.episode_slug): row for row in baseline}
    if set(left) != set(right):
        raise ValueError("paired methods do not cover identical investor episodes")
    keys = sorted(left)
    return [left[key] for key in keys], [right[key] for key in keys]


def _paired_uncertainty(
    candidate: Sequence[PredictionRow],
    baseline: Sequence[PredictionRow],
    kind: str,
    *,
    bootstrap_samples: int,
    randomization_samples: int,
    seed: int,
) -> dict[str, float]:
    candidate, baseline = _aligned(candidate, baseline)
    observed = _metric(candidate, kind) - _metric(baseline, kind)
    episodes = sorted({row.episode_slug for row in candidate})
    candidate_by_episode = {
        episode: [row for row in candidate if row.episode_slug == episode]
        for episode in episodes
    }
    baseline_by_episode = {
        episode: [row for row in baseline if row.episode_slug == episode]
        for episode in episodes
    }
    random = np.random.default_rng(seed)
    bootstrapped = []
    for _ in range(bootstrap_samples):
        selected = random.choice(episodes, size=len(episodes), replace=True)
        left = [row for episode in selected for row in candidate_by_episode[str(episode)]]
        right = [row for episode in selected for row in baseline_by_episode[str(episode)]]
        bootstrapped.append(_metric(left, kind) - _metric(right, kind))
    permuted = []
    for _ in range(randomization_samples):
        left, right = [], []
        for episode in episodes:
            if random.random() < 0.5:
                left.extend(candidate_by_episode[episode]); right.extend(baseline_by_episode[episode])
            else:
                left.extend(baseline_by_episode[episode]); right.extend(candidate_by_episode[episode])
        permuted.append(_metric(left, kind) - _metric(right, kind))
    return {
        "delta": observed,
        "ci_low": float(np.quantile(bootstrapped, 0.025)),
        "ci_high": float(np.quantile(bootstrapped, 0.975)),
        "raw_p": float((1 + sum(abs(value) >= abs(observed) for value in permuted)) / (1 + len(permuted))),
    }


def evaluate_personalized_predictions(
    predictions: Sequence[PredictionRow],
    *,
    expected_case_count: int,
    review_budgets: Sequence[int],
    bootstrap_samples: int = 1000,
    randomization_samples: int = 2000,
    seed: int = 20260821,
) -> dict[str, object]:
    complete_by_method: dict[str, list[PredictionRow]] = {}
    for row in predictions:
        if row.status == "complete":
            complete_by_method.setdefault(row.method, []).append(row)
    if "raw_phase2" not in complete_by_method:
        raise ValueError("personalized evaluation requires raw_phase2")
    classification, ranking, calibration, per_vc, coverage = {}, {}, {}, {}, {}
    for method, rows in sorted(complete_by_method.items()):
        keys = [(row.vc_slug, row.episode_slug) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate predictions for method {method}")
        class_result, rank_result, calibration_result = _method_metrics(rows, review_budgets)
        classification[method] = class_result["macro"]
        ranking[method] = rank_result["macro"]
        calibration[method] = calibration_result
        per_vc[method] = {
            vc: {
                "classification": class_result["per_vc"][vc],
                "ranking": rank_result["per_vc"][vc],
            }
            for vc in class_result["per_vc"]
        }
        coverage[method] = {
            "complete": len(rows),
            "missing_or_failed": max(0, expected_case_count - len(rows)),
            "coverage": len(rows) / expected_case_count,
        }

    baseline = complete_by_method["raw_phase2"]
    comparisons = {}
    p_values, p_keys = [], []
    for method, candidate in sorted(complete_by_method.items()):
        if method == "raw_phase2" or len(candidate) != len(baseline):
            continue
        class_test = _paired_uncertainty(
            candidate, baseline, "classification",
            bootstrap_samples=bootstrap_samples,
            randomization_samples=randomization_samples, seed=seed,
        )
        rank_test = _paired_uncertainty(
            candidate, baseline, "ranking",
            bootstrap_samples=bootstrap_samples,
            randomization_samples=randomization_samples, seed=seed + 1,
        )
        key = f"{method}_vs_raw_phase2"
        comparisons[key] = {
            "classification_delta": class_test["delta"],
            "classification_ci_low": class_test["ci_low"],
            "classification_ci_high": class_test["ci_high"],
            "classification_raw_p": class_test["raw_p"],
            "ranking_delta": rank_test["delta"],
            "ranking_ci_low": rank_test["ci_low"],
            "ranking_ci_high": rank_test["ci_high"],
            "ranking_raw_p": rank_test["raw_p"],
        }
        for kind, test in (("classification", class_test), ("ranking", rank_test)):
            p_keys.append((key, kind)); p_values.append(test["raw_p"])
    for (key, kind), adjusted in zip(p_keys, holm_adjust(p_values), strict=True):
        comparisons[key][f"{kind}_holm_p"] = adjusted

    transitions = {}
    baseline_map = {(row.vc_slug, row.episode_slug): row for row in baseline}
    for method, rows in complete_by_method.items():
        if method == "raw_phase2" or len(rows) != len(baseline):
            continue
        counts = {"candidate_only_correct": 0, "baseline_only_correct": 0, "both_correct": 0, "both_wrong": 0}
        for row in rows:
            base = baseline_map[(row.vc_slug, row.episode_slug)]
            candidate_ok, baseline_ok = row.predicted == row.target, base.predicted == base.target
            name = "both_correct" if candidate_ok and baseline_ok else "both_wrong" if not candidate_ok and not baseline_ok else "candidate_only_correct" if candidate_ok else "baseline_only_correct"
            counts[name] += 1
        transitions[method] = counts

    candidates = [method for method in classification if method != "raw_phase2"]
    best = max(candidates, key=lambda method: (
        classification[method]["balanced_accuracy"], ranking[method]["average_precision"]
    )) if candidates else None
    promotion = "retain_current"
    if best is not None:
        comparison = comparisons.get(f"{best}_vs_raw_phase2", {})
        point_improves = (
            classification[best]["balanced_accuracy"] > LOCKED_BENCHMARKS["raw_phase2_macro_balanced_accuracy"]
            and ranking[best]["average_precision"] > LOCKED_BENCHMARKS["nominal_rationale_ranking_ap"]
        )
        supported = (
            comparison.get("classification_holm_p", 1.0) < 0.05
            and comparison.get("ranking_holm_p", 1.0) < 0.05
            and coverage[best]["complete"] == expected_case_count
        )
        promotion = "promote" if point_improves and supported else "exploratory_only" if point_improves else "retain_current"
    return {
        "schema": "personalized-model-evaluation-v1",
        "expected_case_count": expected_case_count,
        "locked_benchmarks": LOCKED_BENCHMARKS,
        "classification_macro": classification,
        "ranking_macro": ranking,
        "calibration": calibration,
        "per_vc": per_vc,
        "coverage": coverage,
        "paired_comparisons": comparisons,
        "transitions": transitions,
        "promotion_status": promotion,
        "best_candidate": best,
        "predictions": [asdict(row) for row in predictions],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_personalized_evaluation(
    output: Path,
    result: Mapping[str, object],
    *,
    source_manifest: Mapping[str, object],
    automated_rationale_references: bool,
) -> None:
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    population = {
        "schema": "personalized-evaluation-population-v1",
        "expected_case_count": result["expected_case_count"],
        "source_manifest": dict(source_manifest),
        "automated_rationale_references": automated_rationale_references,
    }
    (root / "population.json").write_text(json.dumps(population, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(root / "predictions.csv", result["predictions"])
    class_rows = [{"method": method, **values} for method, values in result["classification_macro"].items()]
    rank_rows = [{"method": method, **values} for method, values in result["ranking_macro"].items()]
    calibration_rows = [{"method": method, **values} for method, values in result["calibration"].items()]
    per_vc_rows = []
    for method, vcs in result["per_vc"].items():
        for vc, values in vcs.items():
            per_vc_rows.append({
                "method": method, "vc_slug": vc,
                **{f"classification__{key}": value for key, value in values["classification"].items()},
                **{f"ranking__{key}": value for key, value in values["ranking"].items()},
            })
    significance = [{"comparison": key, **values} for key, values in result["paired_comparisons"].items()]
    transitions = [{"method": key, **values} for key, values in result["transitions"].items()]
    ablations = []
    for method in sorted(result["classification_macro"]):
        family, _, condition = method.partition("__")
        ablations.append({
            "method": method,
            "family": family,
            "condition": condition or "raw",
            **{f"classification__{key}": value for key, value in result["classification_macro"][method].items()},
            **{f"ranking__{key}": value for key, value in result["ranking_macro"][method].items()},
        })
    complete_maps = {}
    for row in result["predictions"]:
        if row["status"] == "complete":
            complete_maps.setdefault(row["method"], {})[
                (row["vc_slug"], row["episode_slug"])
            ] = row
    error_overlap = []
    methods = sorted(complete_maps)
    for left_index, left_method in enumerate(methods):
        for right_method in methods[left_index + 1:]:
            shared = sorted(set(complete_maps[left_method]) & set(complete_maps[right_method]))
            both_wrong = left_only_wrong = right_only_wrong = 0
            for key in shared:
                left = complete_maps[left_method][key]
                right = complete_maps[right_method][key]
                left_wrong = int(left["predicted"]) != int(left["target"])
                right_wrong = int(right["predicted"]) != int(right["target"])
                both_wrong += int(left_wrong and right_wrong)
                left_only_wrong += int(left_wrong and not right_wrong)
                right_only_wrong += int(right_wrong and not left_wrong)
            error_overlap.append({
                "left_method": left_method, "right_method": right_method,
                "shared_cases": len(shared), "both_wrong": both_wrong,
                "left_only_wrong": left_only_wrong,
                "right_only_wrong": right_only_wrong,
            })
    economics = []
    for method in sorted({row["method"] for row in result["predictions"]}):
        selected = [row for row in result["predictions"] if row["method"] == method]
        economics.append({
            "method": method,
            "runtime_seconds": sum(float(row.get("runtime_seconds", 0.0)) for row in selected),
            "api_cost_usd": 0.0,
        })
    for name, rows in (
        ("classification_metrics.csv", class_rows), ("ranking_metrics.csv", rank_rows),
        ("calibration_metrics.csv", calibration_rows), ("per_vc_metrics.csv", per_vc_rows),
        ("significance.csv", significance), ("transitions.csv", transitions),
        ("economics.csv", economics), ("ablations.csv", ablations),
        ("error_overlap.csv", error_overlap),
    ):
        _write_csv(root / name, rows)
    (root / "feature_provenance.json").write_text(json.dumps({
        "schema": "personalized-feature-provenance-v1",
        "sources": dict(source_manifest),
        "automated_rationale_references": automated_rationale_references,
        "locked_benchmarks": result["locked_benchmarks"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Personalized Multimodal Model Evaluation", "",
        f"Promotion status: **{result['promotion_status']}**.", "",
        f"Best candidate: `{result['best_candidate']}`.", "",
        "Automated transcript-observed rationale references are auxiliary candidate ground truth and remain pending human validation." if automated_rationale_references else "Rationale references were human validated.",
        "", "## Classification", "",
    ]
    for row in class_rows:
        report.append(f"- {row['method']}: macro balanced accuracy {row['balanced_accuracy']:.3f}")
    report.extend(("", "## Ranking", ""))
    for row in rank_rows:
        report.append(f"- {row['method']}: macro within-VC AP {row['average_precision']:.3f}")
    (root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
