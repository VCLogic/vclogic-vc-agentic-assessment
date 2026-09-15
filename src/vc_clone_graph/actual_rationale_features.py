"""Source-neutral structured features for observed and predicted rationales."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


DIRECTIONS = ("positive", "negative", "neutral")
SALIENCES = ("primary", "secondary")
AGGREGATES = (
    "positive_count",
    "negative_count",
    "neutral_count",
    "primary_count",
    "secondary_count",
    "distinct_label_count",
    "instance_count",
)


def _labels(taxonomy: Mapping[str, object] | Sequence[str]) -> tuple[str, ...]:
    labels = tuple(taxonomy) if isinstance(taxonomy, Mapping) else tuple(taxonomy)
    if len(labels) != 44 or len(set(labels)) != 44:
        raise ValueError(f"rationale features require 44 unique labels, found {len(labels)}")
    return labels


def _increment(features: dict[str, float], name: str, value: float = 1.0) -> None:
    features[name] = features.get(name, 0.0) + float(value)


def structured_rationale_features(
    rationales: Sequence[Mapping[str, object]],
    taxonomy: Mapping[str, object] | Sequence[str],
    *,
    label_field: str,
) -> dict[str, float]:
    """Encode only taxonomy label, direction, and salience in a shared space."""
    labels = _labels(taxonomy)
    valid_labels = set(labels)
    features: dict[str, float] = {}
    for label in labels:
        prefix = f"rationale__{label}"
        features[f"{prefix}__present"] = 0.0
        features[f"{prefix}__count"] = 0.0
        features[f"{prefix}__signed_salience"] = 0.0
        for direction in DIRECTIONS:
            features[f"{prefix}__direction__{direction}"] = 0.0
        for salience in SALIENCES:
            features[f"{prefix}__salience__{salience}"] = 0.0
    for aggregate in AGGREGATES:
        features[f"aggregate__{aggregate}"] = 0.0

    seen: set[str] = set()
    direction_weight = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    salience_weight = {"primary": 1.0, "secondary": 0.5}
    for index, rationale in enumerate(rationales):
        label = rationale.get(label_field)
        direction = rationale.get("direction")
        salience = rationale.get("salience")
        if label not in valid_labels:
            raise ValueError(f"invalid rationale label at index {index}: {label!r}")
        if direction not in DIRECTIONS:
            raise ValueError(f"invalid rationale direction at index {index}: {direction!r}")
        if salience not in SALIENCES:
            raise ValueError(f"invalid rationale salience at index {index}: {salience!r}")
        label = str(label)
        direction = str(direction)
        salience = str(salience)
        prefix = f"rationale__{label}"
        features[f"{prefix}__present"] = 1.0
        _increment(features, f"{prefix}__count")
        _increment(features, f"{prefix}__direction__{direction}")
        _increment(features, f"{prefix}__salience__{salience}")
        _increment(
            features,
            f"{prefix}__signed_salience",
            direction_weight[direction] * salience_weight[salience],
        )
        _increment(features, f"aggregate__{direction}_count")
        _increment(features, f"aggregate__{salience}_count")
        _increment(features, "aggregate__instance_count")
        seen.add(label)
    features["aggregate__distinct_label_count"] = float(len(seen))
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("non-finite rationale feature")
    return features


def actual_rationale_feature_dictionary(
    taxonomy: Mapping[str, object] | Sequence[str],
) -> list[dict[str, str]]:
    """Describe every field in the shared actual/predicted feature contract."""
    labels = _labels(taxonomy)
    rows: list[dict[str, str]] = []
    for label in labels:
        prefix = f"rationale__{label}"
        rows.extend([
            {"feature": f"{prefix}__present", "family": "label", "definition": f"Presence of {label}."},
            {"feature": f"{prefix}__count", "family": "label", "definition": f"Number of {label} annotations."},
            {"feature": f"{prefix}__signed_salience", "family": "interaction", "definition": f"Signed salience-weighted {label} activation."},
        ])
        rows.extend(
            {"feature": f"{prefix}__direction__{value}", "family": "direction", "definition": f"Count of {value} {label} annotations."}
            for value in DIRECTIONS
        )
        rows.extend(
            {"feature": f"{prefix}__salience__{value}", "family": "salience", "definition": f"Count of {value} {label} annotations."}
            for value in SALIENCES
        )
    rows.extend(
        {"feature": f"aggregate__{name}", "family": "aggregate", "definition": f"Rationale aggregate: {name.replace('_', ' ')}."}
        for name in AGGREGATES
    )
    return rows
