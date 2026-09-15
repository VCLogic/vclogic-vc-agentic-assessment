"""Compact, interpretable Phase 1 rationale feature construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .phase2_calibration_evaluation import Phase2CalibrationRecord


AGGREGATE_FEATURES = (
    "positive_count",
    "negative_count",
    "neutral_count",
    "primary_count",
    "secondary_count",
    "controlling_count",
    "pitch_evidence_total",
    "external_evidence_total",
)
CONSTRAINT_FEATURES = (
    "triggered_hard",
    "triggered_material",
    "possible_hard",
    "possible_material",
    "portfolio_conflict_triggered",
    "portfolio_conflict_possible",
)
PHASE2_FIELDS = (
    "investment_likelihood",
    "review_priority_score",
    "decision_confidence",
    "signed_decision_confidence",
)


def _labels(taxonomy: Mapping[str, object] | Sequence[str]) -> tuple[str, ...]:
    values = tuple(taxonomy) if isinstance(taxonomy, Mapping) else tuple(taxonomy)
    if len(values) != 44 or len(set(values)) != 44:
        raise ValueError(f"compact rationale features require 44 unique labels, found {len(values)}")
    return values


def _rationale_value(features: Mapping[str, float], label: str) -> float:
    prefix = f"rationale__{label}__"
    signed = float(features.get(prefix + "signed_confidence", 0.0))
    if float(features.get(prefix + "salience__primary", 0.0)) > 0:
        weight = 1.0
    elif float(features.get(prefix + "salience__secondary", 0.0)) > 0:
        weight = 0.5
    else:
        weight = 1.0
    return signed * weight


def _constraint_totals(features: Mapping[str, float]) -> dict[str, float]:
    totals = {name: 0.0 for name in CONSTRAINT_FEATURES}
    for feature, raw_value in features.items():
        if not feature.startswith("constraint__"):
            continue
        parts = feature.split("__")
        if len(parts) != 4:
            continue
        _, kind, status, severity = parts
        value = float(raw_value)
        compact_severity = "hard" if severity in {"blocking", "hard"} else severity
        key = f"{status}_{compact_severity}"
        if key in totals:
            totals[key] += value
        if kind == "portfolio_conflict" and status in {"triggered", "possible"}:
            totals[f"portfolio_conflict_{status}"] += value
    return totals


def compact_rationale_view(
    record: Phase2CalibrationRecord,
    taxonomy: Mapping[str, object] | Sequence[str],
) -> dict[str, float]:
    """Return 44 signed rationales plus fourteen transparent aggregate fields."""
    labels = _labels(taxonomy)
    source = record.semantic_features
    result = {
        f"rationale_core__{label}": _rationale_value(source, label)
        for label in labels
    }
    aggregates = {name: 0.0 for name in AGGREGATE_FEATURES}
    for label in labels:
        prefix = f"rationale__{label}__"
        aggregates["positive_count"] += float(source.get(prefix + "direction__positive", 0.0))
        aggregates["negative_count"] += float(source.get(prefix + "direction__negative", 0.0))
        aggregates["neutral_count"] += float(source.get(prefix + "direction__neutral", 0.0))
        aggregates["primary_count"] += float(source.get(prefix + "salience__primary", 0.0))
        aggregates["secondary_count"] += float(source.get(prefix + "salience__secondary", 0.0))
        aggregates["controlling_count"] += float(source.get(prefix + "controlling", 0.0))
        aggregates["pitch_evidence_total"] += float(
            source.get(prefix + "pitch_evidence_count", 0.0)
        )
        aggregates["external_evidence_total"] += float(
            source.get(prefix + "wiki_evidence_count", 0.0)
        ) + float(source.get(prefix + "historical_evidence_count", 0.0))
    result.update({f"aggregate__{name}": value for name, value in aggregates.items()})
    result.update({
        f"constraint__{name}": value
        for name, value in _constraint_totals(source).items()
    })
    return result


def compact_decision_view(
    record: Phase2CalibrationRecord,
    taxonomy: Mapping[str, object] | Sequence[str],
) -> dict[str, float]:
    """Add four structured Phase 2 fields to the rationale core."""
    result = compact_rationale_view(record, taxonomy)
    result.update({
        f"phase2__{field}": float(record.phase2_features.get(field, 0.0))
        for field in PHASE2_FIELDS
    })
    return result


def compact_feature_dictionary(
    taxonomy: Mapping[str, object] | Sequence[str],
) -> list[dict[str, str]]:
    """Describe every rationale-only and Phase 2 extension feature."""
    labels = _labels(taxonomy)
    rows: list[dict[str, str]] = []
    for label in labels:
        item = taxonomy.get(label) if isinstance(taxonomy, Mapping) else None
        definition = str(getattr(item, "definition", ""))
        rows.append({
            "feature": f"rationale_core__{label}",
            "family": "rationale",
            "definition": definition or f"Signed, salience-weighted activation of {label}.",
            "source": "phase1",
            "directionality": "signed",
        })
    rows.extend({
        "feature": f"aggregate__{name}",
        "family": "aggregate",
        "definition": f"Phase 1 aggregate: {name.replace('_', ' ')}.",
        "source": "phase1",
        "directionality": "count",
    } for name in AGGREGATE_FEATURES)
    rows.extend({
        "feature": f"constraint__{name}",
        "family": "constraint",
        "definition": f"Aggregated constraint signal: {name.replace('_', ' ')}.",
        "source": "phase1",
        "directionality": "count",
    } for name in CONSTRAINT_FEATURES)
    rows.extend({
        "feature": f"phase2__{name}",
        "family": "phase2",
        "definition": f"Structured Phase 2 field: {name.replace('_', ' ')}.",
        "source": "phase2",
        "directionality": "signed" if name == "signed_decision_confidence" else "nonnegative",
    } for name in PHASE2_FIELDS)
    return rows
