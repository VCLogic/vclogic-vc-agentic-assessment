"""Canonical classification and ranking evaluation for VC inference artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from sklearn.metrics import average_precision_score, roc_auc_score


REGISTRY_SCHEMA = "canonical-vc-evaluation-registry-v1"


@dataclass(frozen=True)
class ArtifactSource:
    kind: str
    paths: tuple[str, ...] = ()
    path: str | None = None
    vc_name: str | None = None


@dataclass(frozen=True)
class InvestorSpec:
    slug: str
    display_name: str
    label_file: Path
    eligible_count: int
    sources: tuple[ArtifactSource, ...]


@dataclass(frozen=True)
class EvaluationRegistry:
    path: Path
    investors: dict[str, InvestorSpec]


@dataclass(frozen=True)
class PredictionRow:
    vc_slug: str
    vc_name: str
    episode_slug: str
    actual_decision: str
    predicted_decision: str
    investment_likelihood: float
    decision_confidence: float | None
    review_priority_score: float | None
    phase1_status: str | None
    phase1_iterations: int | None
    phase2_status: str | None
    phase2_iterations: int | None
    cost_usd: float
    artifact_path: Path


@dataclass(frozen=True)
class CanonicalArtifact:
    vc_slug: str
    vc_name: str
    episode_slug: str
    actual_decision: str
    artifact_path: Path


def _require_dict(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def load_registry(path: Path) -> EvaluationRegistry:
    path = path.resolve()
    payload = _require_dict(json.loads(path.read_text(encoding="utf-8")), "registry")
    if payload.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(f"unsupported evaluation registry schema: {payload.get('schema')!r}")
    raw_investors = _require_dict(payload.get("investors"), "registry investors")
    investors: dict[str, InvestorSpec] = {}
    for slug, raw_value in raw_investors.items():
        raw = _require_dict(raw_value, f"investor {slug}")
        sources: list[ArtifactSource] = []
        raw_sources = raw.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError(f"investor {slug} must declare at least one source")
        for index, raw_source_value in enumerate(raw_sources):
            source = _require_dict(raw_source_value, f"investor {slug} source {index}")
            paths = source.get("paths", [])
            if not isinstance(paths, list) or not all(isinstance(item, str) for item in paths):
                raise ValueError(f"investor {slug} source {index} paths must be strings")
            sources.append(
                ArtifactSource(
                    kind=str(source.get("kind", "")),
                    paths=tuple(paths),
                    path=source.get("path"),
                    vc_name=source.get("vc_name"),
                )
            )
        investors[slug] = InvestorSpec(
            slug=slug,
            display_name=str(raw["display_name"]),
            label_file=(path.parent / str(raw["label_file"])).resolve(),
            eligible_count=int(raw["eligible_count"]),
            sources=tuple(sources),
        )
    return EvaluationRegistry(path=path, investors=investors)


def _eligible_labels(spec: InvestorSpec) -> dict[str, str]:
    payload = json.loads(spec.label_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"label file must contain a list: {spec.label_file}")
    labels: dict[str, str] = {}
    for raw in payload:
        if not isinstance(raw, dict) or raw.get("evaluation_eligible") is not True:
            continue
        slug = raw.get("episode_slug")
        decision = raw.get("pitch_window_decision")
        if not isinstance(slug, str) or decision not in {"In", "Out"}:
            raise ValueError(f"invalid eligible label in {spec.label_file}")
        if slug in labels:
            raise ValueError(f"duplicate eligible label for {slug}")
        labels[slug] = decision
    if len(labels) != spec.eligible_count:
        raise ValueError(
            f"eligible label count for {spec.slug} is {len(labels)}, expected {spec.eligible_count}"
        )
    return labels


def _source_summary_paths(registry: EvaluationRegistry, source: ArtifactSource) -> tuple[Path, ...]:
    base = registry.path.parent
    if source.kind == "summary_files":
        return tuple((base / item).resolve() for item in source.paths)
    if source.kind == "summary_glob":
        if not source.path:
            raise ValueError("summary_glob source requires path")
        return tuple(sorted(base.glob(source.path)))
    if source.kind in {"batch_status", "recovery_status"}:
        if not source.path:
            raise ValueError(f"{source.kind} source requires path")
        status_path = (base / source.path).resolve()
        status = _require_dict(
            json.loads(status_path.read_text(encoding="utf-8")),
            f"status {status_path}",
        )
        records = status.get("completed_records")
        if not isinstance(records, list):
            raise ValueError(f"status has no completed_records list: {status_path}")
        paths: list[Path] = []
        for record_value in records:
            record = _require_dict(record_value, f"completed record in {status_path}")
            if source.kind == "recovery_status" and record.get("vc") != source.vc_name:
                continue
            artifact_root = record.get("artifact_root")
            if not isinstance(artifact_root, str):
                raise ValueError(f"completed record has no artifact_root: {status_path}")
            root = Path(artifact_root)
            if not root.is_absolute():
                root = (base.parent / root).resolve()
            paths.append(root / "summary.json")
        return tuple(paths)
    raise ValueError(f"unsupported artifact source kind: {source.kind!r}")


def _optional_number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def load_canonical_artifacts(
    registry: EvaluationRegistry,
    selected_vcs: Iterable[str] | None = None,
) -> list[CanonicalArtifact]:
    """Resolve eligible canonical summaries without requiring a Phase 2 decision."""
    selected = list(selected_vcs) if selected_vcs is not None else list(registry.investors)
    unknown = sorted(set(selected) - set(registry.investors))
    if unknown:
        raise ValueError(f"unknown VC slug(s): {', '.join(unknown)}")
    rows: list[CanonicalArtifact] = []
    for vc_slug in selected:
        spec = registry.investors[vc_slug]
        labels = _eligible_labels(spec)
        artifacts: dict[str, Path] = {}
        for source in spec.sources:
            source_rows: dict[str, Path] = {}
            for path in _source_summary_paths(registry, source):
                if not path.is_file():
                    raise ValueError(f"missing summary artifact: {path}")
                payload = _require_dict(
                    json.loads(path.read_text(encoding="utf-8")), f"summary {path}"
                )
                decision = (
                    payload.get("decision")
                    if isinstance(payload.get("decision"), dict)
                    else {}
                )
                slug = payload.get("episode_slug") or decision.get("episode_slug") or path.parent.name
                if slug in source_rows:
                    raise ValueError(f"duplicate artifact at equal precedence for {slug}")
                source_rows[str(slug)] = path
            artifacts.update(source_rows)
        missing = sorted(set(labels) - set(artifacts))
        unexpected = sorted(set(artifacts) - set(labels))
        if missing:
            raise ValueError(f"missing canonical artifacts for {vc_slug}: {', '.join(missing)}")
        if unexpected:
            raise ValueError(f"unexpected canonical artifacts for {vc_slug}: {', '.join(unexpected)}")
        rows.extend(
            CanonicalArtifact(
                vc_slug=spec.slug,
                vc_name=spec.display_name,
                episode_slug=slug,
                actual_decision=labels[slug],
                artifact_path=artifacts[slug],
            )
            for slug in sorted(labels)
        )
    return rows


def _prediction_from_summary(spec: InvestorSpec, actual: str, path: Path) -> PredictionRow:
    summary = _require_dict(json.loads(path.read_text(encoding="utf-8")), f"summary {path}")
    decision = _require_dict(summary.get("decision"), f"summary decision {path}")
    predicted = decision.get("decision")
    likelihood = decision.get("investment_likelihood")
    if predicted not in {"In", "Out"}:
        raise ValueError(f"summary has no usable decision: {path}")
    if not isinstance(likelihood, int | float) or not 0 <= float(likelihood) <= 1:
        raise ValueError(f"summary has invalid investment likelihood: {path}")
    episode_slug = summary.get("episode_slug") or decision.get("episode_slug") or path.parent.name
    if not isinstance(episode_slug, str):
        raise ValueError(f"summary has no episode slug: {path}")
    usage = summary.get("usage") if isinstance(summary.get("usage"), dict) else {}
    return PredictionRow(
        vc_slug=spec.slug,
        vc_name=spec.display_name,
        episode_slug=episode_slug,
        actual_decision=actual,
        predicted_decision=predicted,
        investment_likelihood=float(likelihood),
        decision_confidence=_optional_number(decision.get("decision_confidence")),
        review_priority_score=_optional_number(decision.get("review_priority_score")),
        phase1_status=summary.get("phase1_status"),
        phase1_iterations=summary.get("phase1_iterations"),
        phase2_status=summary.get("phase2_status"),
        phase2_iterations=summary.get("phase2_iterations"),
        cost_usd=float(usage.get("cost_usd", 0.0) or 0.0),
        artifact_path=path,
    )


def prediction_from_summary(spec: InvestorSpec, actual: str, path: Path) -> PredictionRow:
    """Load one validated prediction from a summary artifact."""
    return _prediction_from_summary(spec, actual, path)


def load_canonical_predictions(
    registry: EvaluationRegistry,
    selected_vcs: Iterable[str] | None = None,
) -> list[PredictionRow]:
    selected = list(selected_vcs) if selected_vcs is not None else list(registry.investors)
    unknown = sorted(set(selected) - set(registry.investors))
    if unknown:
        raise ValueError(f"unknown VC slug(s): {', '.join(unknown)}")
    rows: list[PredictionRow] = []
    for vc_slug in selected:
        spec = registry.investors[vc_slug]
        labels = _eligible_labels(spec)
        artifacts: dict[str, Path] = {}
        for source in spec.sources:
            source_rows: dict[str, Path] = {}
            for path in _source_summary_paths(registry, source):
                if not path.is_file():
                    raise ValueError(f"missing summary artifact: {path}")
                payload = _require_dict(json.loads(path.read_text(encoding="utf-8")), f"summary {path}")
                decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
                slug = payload.get("episode_slug") or decision.get("episode_slug") or path.parent.name
                if slug in source_rows:
                    raise ValueError(f"duplicate artifact at equal precedence for {slug}")
                source_rows[str(slug)] = path
            artifacts.update(source_rows)
        missing = sorted(set(labels) - set(artifacts))
        unexpected = sorted(set(artifacts) - set(labels))
        if missing:
            raise ValueError(f"missing canonical artifacts for {vc_slug}: {', '.join(missing)}")
        if unexpected:
            raise ValueError(f"unexpected canonical artifacts for {vc_slug}: {', '.join(unexpected)}")
        rows.extend(
            _prediction_from_summary(spec, labels[slug], artifacts[slug])
            for slug in sorted(labels)
        )
    return rows


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(rows: Iterable[PredictionRow]) -> dict[str, int | float]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("classification metrics require at least one prediction")
    tp = sum(row.actual_decision == "In" and row.predicted_decision == "In" for row in materialized)
    fp = sum(row.actual_decision == "Out" and row.predicted_decision == "In" for row in materialized)
    tn = sum(row.actual_decision == "Out" and row.predicted_decision == "Out" for row in materialized)
    fn = sum(row.actual_decision == "In" and row.predicted_decision == "Out" for row in materialized)
    n = len(materialized)
    in_count = tp + fn
    out_count = tn + fp
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, in_count)
    specificity = _safe_ratio(tn, out_count)
    return {
        "n": n,
        "in_count": in_count,
        "out_count": out_count,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": _safe_ratio(tp + tn, n),
        "balanced_accuracy": (recall + specificity) / 2,
        "in_precision": precision,
        "in_recall": recall,
        "in_f1": _safe_ratio(2 * precision * recall, precision + recall),
        "specificity": specificity,
        "prevalence": _safe_ratio(in_count, n),
    }


def rank_predictions(rows: Iterable[PredictionRow]) -> list[tuple[int, PredictionRow]]:
    ranked = sorted(
        rows,
        key=lambda row: (-row.investment_likelihood, row.episode_slug),
    )
    return [(index, row) for index, row in enumerate(ranked, start=1)]


def ranking_metrics(
    rows: Iterable[PredictionRow],
    *,
    top_ks: Iterable[int] = (1, 3, 5, 10, 20),
) -> dict[str, object]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("ranking metrics require at least one prediction")
    budgets = tuple(top_ks)
    if not budgets or any(not isinstance(k, int) or k <= 0 for k in budgets):
        raise ValueError("top-k review budgets must be positive integers")
    true = [1 if row.actual_decision == "In" else 0 for row in materialized]
    scores = [row.investment_likelihood for row in materialized]
    positive_count = sum(true)
    negative_count = len(true) - positive_count
    average_precision = (
        float(average_precision_score(true, scores)) if positive_count else None
    )
    roc_auc = (
        float(roc_auc_score(true, scores))
        if positive_count and negative_count
        else None
    )
    ranked = rank_predictions(materialized)
    budget_rows: list[dict[str, int | float]] = []
    for requested_k in budgets:
        reviewed = min(requested_k, len(ranked))
        hits = sum(row.actual_decision == "In" for _, row in ranked[:reviewed])
        budget_rows.append(
            {
                "requested_k": requested_k,
                "reviewed": reviewed,
                "hits": hits,
                "precision": _safe_ratio(hits, reviewed),
                "recall": _safe_ratio(hits, positive_count),
            }
        )
    return {
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "budgets": budget_rows,
    }


def _phase_counts(rows: list[PredictionRow]) -> dict[str, int | float]:
    return {
        "phase1_accepted": sum(row.phase1_status == "accepted" for row in rows),
        "phase1_provisional": sum(row.phase1_status == "provisional" for row in rows),
        "phase2_accepted": sum(row.phase2_status == "accepted" for row in rows),
        "phase2_provisional": sum(row.phase2_status == "provisional" for row in rows),
        "total_cost_usd": sum(row.cost_usd for row in rows),
    }


def _mean_optional(values: Iterable[float | None]) -> float | None:
    materialized = [value for value in values if value is not None]
    return fmean(materialized) if materialized else None


def build_evaluation(
    rows: Iterable[PredictionRow],
    *,
    top_ks: Iterable[int] = (1, 3, 5, 10, 20),
) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("evaluation requires at least one prediction")
    budgets = tuple(top_ks)
    grouped: dict[str, list[PredictionRow]] = {}
    for row in materialized:
        grouped.setdefault(row.vc_slug, []).append(row)
    investors: list[dict[str, Any]] = []
    for vc_slug in sorted(grouped):
        vc_rows = grouped[vc_slug]
        investors.append(
            {
                "scope": "investor",
                "vc_slug": vc_slug,
                "vc_name": vc_rows[0].vc_name,
                "classification": classification_metrics(vc_rows),
                "ranking": ranking_metrics(vc_rows, top_ks=budgets),
                "runtime": _phase_counts(vc_rows),
            }
        )
    classification_rate_keys = (
        "accuracy",
        "balanced_accuracy",
        "in_precision",
        "in_recall",
        "in_f1",
        "specificity",
        "prevalence",
    )
    macro_classification = {
        key: fmean(float(item["classification"][key]) for item in investors)
        for key in classification_rate_keys
    }
    macro_ranking = {
        "average_precision": _mean_optional(
            item["ranking"]["average_precision"] for item in investors
        ),
        "roc_auc": _mean_optional(item["ranking"]["roc_auc"] for item in investors),
        "budgets": [],
    }
    for index, requested_k in enumerate(budgets):
        budget_values = [item["ranking"]["budgets"][index] for item in investors]
        macro_ranking["budgets"].append(
            {
                "requested_k": requested_k,
                "reviewed": None,
                "hits": None,
                "precision": fmean(float(item["precision"]) for item in budget_values),
                "recall": fmean(float(item["recall"]) for item in budget_values),
            }
        )
    return {
        "investors": investors,
        "macro": {
            "scope": "macro",
            "vc_slug": "",
            "vc_name": "Macro average",
            "classification": macro_classification,
            "ranking": macro_ranking,
            "runtime": {},
        },
        "pooled": {
            "scope": "pooled_diagnostic",
            "vc_slug": "",
            "vc_name": "Pooled diagnostic",
            "classification": classification_metrics(materialized),
            "ranking": ranking_metrics(materialized, top_ks=budgets),
            "runtime": _phase_counts(materialized),
        },
        "top_ks": list(budgets),
    }


def _format_metric(value: object, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Canonical VC Classification and Ranking Evaluation",
        "",
        "Classification uses the Phase 2 `decision`. Ranking uses the Phase 2 "
        "`investment_likelihood` independently within each investor.",
        "",
        "## Classification",
        "",
        "| Investor | N | In / Out | TP | FP | TN | FN | Accuracy | Balanced accuracy | In precision | In recall | In F1 | AP | ROC AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["investors"]:
        c = item["classification"]
        r = item["ranking"]
        lines.append(
            f"| {item['vc_name']} | {c['n']} | {c['in_count']} / {c['out_count']} | "
            f"{c['tp']} | {c['fp']} | {c['tn']} | {c['fn']} | "
            f"{_format_metric(c['accuracy'])} | {_format_metric(c['balanced_accuracy'])} | "
            f"{_format_metric(c['in_precision'])} | {_format_metric(c['in_recall'])} | "
            f"{_format_metric(c['in_f1'])} | {_format_metric(r['average_precision'])} | "
            f"{_format_metric(r['roc_auc'])} |"
        )
    macro = result["macro"]
    lines.extend(
        [
            f"| **Macro average** |  |  |  |  |  |  | "
            f"{_format_metric(macro['classification']['accuracy'])} | "
            f"{_format_metric(macro['classification']['balanced_accuracy'])} | "
            f"{_format_metric(macro['classification']['in_precision'])} | "
            f"{_format_metric(macro['classification']['in_recall'])} | "
            f"{_format_metric(macro['classification']['in_f1'])} | "
            f"{_format_metric(macro['ranking']['average_precision'])} | "
            f"{_format_metric(macro['ranking']['roc_auc'])} |",
            "",
            "The pooled diagnostic is exported for completeness, but pooled ranking is not "
            "substantively comparable because likelihood scales are investor-specific.",
            "",
            "## Ranking at review budgets",
            "",
            "| Investor | Review budget | Reviewed | Hits | Precision | Recall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in result["investors"]:
        for budget in item["ranking"]["budgets"]:
            lines.append(
                f"| {item['vc_name']} | {budget['requested_k']} | {budget['reviewed']} | "
                f"{budget['hits']} | {_format_metric(budget['precision'])} | "
                f"{_format_metric(budget['recall'])} |"
            )
    lines.append("")
    return "\n".join(lines)


CLASSIFICATION_FIELDS = (
    "scope",
    "vc_slug",
    "vc_name",
    "n",
    "in_count",
    "out_count",
    "tp",
    "fp",
    "tn",
    "fn",
    "accuracy",
    "balanced_accuracy",
    "in_precision",
    "in_recall",
    "in_f1",
    "specificity",
    "prevalence",
    "phase1_accepted",
    "phase1_provisional",
    "phase2_accepted",
    "phase2_provisional",
    "total_cost_usd",
)

RANKING_FIELDS = (
    "scope",
    "vc_slug",
    "vc_name",
    "average_precision",
    "roc_auc",
    "requested_k",
    "reviewed",
    "hits",
    "precision_at_k",
    "recall_at_k",
)

PREDICTION_FIELDS = (
    "vc_slug",
    "vc_name",
    "rank",
    "episode_slug",
    "actual_decision",
    "predicted_decision",
    "correct",
    "investment_likelihood",
    "decision_confidence",
    "review_priority_score",
    "phase1_status",
    "phase1_iterations",
    "phase2_status",
    "phase2_iterations",
    "cost_usd",
    "artifact_path",
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_outputs(
    result: dict[str, Any],
    rows: Iterable[PredictionRow],
    output_dir: Path,
    *,
    include_csv: bool = True,
    include_markdown: bool = True,
) -> dict[str, Path]:
    if not include_csv and not include_markdown:
        raise ValueError("at least one output format must be enabled")
    output_dir.mkdir(parents=True, exist_ok=True)
    classification_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    for item in [*result["investors"], result["macro"], result["pooled"]]:
        classification_rows.append(
            {
                "scope": item["scope"],
                "vc_slug": item["vc_slug"],
                "vc_name": item["vc_name"],
                **item["classification"],
                **item["runtime"],
            }
        )
        for budget in item["ranking"]["budgets"]:
            ranking_rows.append(
                {
                    "scope": item["scope"],
                    "vc_slug": item["vc_slug"],
                    "vc_name": item["vc_name"],
                    "average_precision": item["ranking"]["average_precision"],
                    "roc_auc": item["ranking"]["roc_auc"],
                    "requested_k": budget["requested_k"],
                    "reviewed": budget["reviewed"],
                    "hits": budget["hits"],
                    "precision_at_k": budget["precision"],
                    "recall_at_k": budget["recall"],
                }
            )
    materialized = list(rows)
    prediction_rows: list[dict[str, object]] = []
    for vc_slug in sorted({row.vc_slug for row in materialized}):
        for rank, row in rank_predictions(item for item in materialized if item.vc_slug == vc_slug):
            prediction_rows.append(
                {
                    "vc_slug": row.vc_slug,
                    "vc_name": row.vc_name,
                    "rank": rank,
                    "episode_slug": row.episode_slug,
                    "actual_decision": row.actual_decision,
                    "predicted_decision": row.predicted_decision,
                    "correct": row.actual_decision == row.predicted_decision,
                    "investment_likelihood": row.investment_likelihood,
                    "decision_confidence": row.decision_confidence,
                    "review_priority_score": row.review_priority_score,
                    "phase1_status": row.phase1_status,
                    "phase1_iterations": row.phase1_iterations,
                    "phase2_status": row.phase2_status,
                    "phase2_iterations": row.phase2_iterations,
                    "cost_usd": row.cost_usd,
                    "artifact_path": str(row.artifact_path),
                }
            )
    paths: dict[str, Path] = {}
    if include_csv:
        paths.update(
            {
                "classification_metrics": output_dir / "classification_metrics.csv",
                "ranking_metrics": output_dir / "ranking_metrics.csv",
                "predictions": output_dir / "predictions.csv",
            }
        )
        _write_csv(paths["classification_metrics"], CLASSIFICATION_FIELDS, classification_rows)
        _write_csv(paths["ranking_metrics"], RANKING_FIELDS, ranking_rows)
        _write_csv(paths["predictions"], PREDICTION_FIELDS, prediction_rows)
    if include_markdown:
        paths["markdown"] = output_dir / "evaluation.md"
        paths["markdown"].write_text(render_markdown(result), encoding="utf-8")
    return paths
