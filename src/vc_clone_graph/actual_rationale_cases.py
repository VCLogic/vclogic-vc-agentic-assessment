"""Aligned observed and predicted rationale cases for decision modeling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from .actual_rationale_features import structured_rationale_features
from .phase1_evaluation import Phase1Case


RationaleItem = tuple[str, str, str]


@dataclass(frozen=True)
class RationaleModelCase:
    vc_slug: str
    vc_name: str
    episode_slug: str
    group: str
    target: int
    source_tier: str
    source_format: str
    actual_features: dict[str, float]
    predicted_features: dict[str, float]
    actual_items: tuple[RationaleItem, ...]
    predicted_items: tuple[RationaleItem, ...]


def _actual_rows(case: Phase1Case) -> list[dict[str, object]]:
    return [
        {
            "rationale_label": item.label,
            "direction": item.direction,
            "salience": item.salience,
        }
        for item in case.reference
    ]


def _predicted_rows(case: Phase1Case) -> list[dict[str, object]]:
    return [
        {
            "taxonomy_label": item.label,
            "direction": item.direction,
            "salience": item.salience,
        }
        for item in case.predicted
    ]


def _items(rows: Sequence[Mapping[str, object]], label_field: str) -> tuple[RationaleItem, ...]:
    return tuple(sorted(
        (str(row[label_field]), str(row["direction"]), str(row["salience"]))
        for row in rows
    ))


def build_rationale_model_cases(
    cases: Sequence[Phase1Case],
    taxonomy: Mapping[str, object] | Sequence[str],
) -> list[RationaleModelCase]:
    """Create aligned feature pairs while keeping the audited target separate."""
    result: list[RationaleModelCase] = []
    seen: set[tuple[str, str]] = set()
    for case in cases:
        key = (case.vc_slug, case.episode_slug)
        if key in seen:
            raise ValueError(f"duplicate rationale model case: {key}")
        seen.add(key)
        if case.actual_decision not in {"In", "Out"}:
            raise ValueError(f"invalid actual decision for {key}: {case.actual_decision!r}")
        actual_rows = _actual_rows(case)
        predicted_rows = _predicted_rows(case)
        actual_features = structured_rationale_features(
            actual_rows, taxonomy, label_field="rationale_label"
        )
        predicted_features = structured_rationale_features(
            predicted_rows, taxonomy, label_field="taxonomy_label"
        )
        if set(actual_features) != set(predicted_features):
            raise ValueError(f"feature contract mismatch for {key}")
        if not all(
            math.isfinite(value)
            for features in (actual_features, predicted_features)
            for value in features.values()
        ):
            raise ValueError(f"non-finite rationale features for {key}")
        result.append(RationaleModelCase(
            vc_slug=case.vc_slug,
            vc_name=case.vc_name,
            episode_slug=case.episode_slug,
            group=case.episode_slug,
            target=int(case.actual_decision == "In"),
            source_tier=case.source_tier,
            source_format=case.source_format,
            actual_features=actual_features,
            predicted_features=predicted_features,
            actual_items=_items(actual_rows, "rationale_label"),
            predicted_items=_items(predicted_rows, "taxonomy_label"),
        ))
    return result
