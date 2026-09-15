"""Label-blind features for learning investor-specific decision overrides."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .calibration import CalibrationRecord, FeatureMap


def _token(value: object) -> str:
    return str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")


def _increment(features: FeatureMap, name: str, amount: float = 1.0) -> None:
    features[name] = features.get(name, 0.0) + float(amount)


def exclude_changed_pitch_records(
    status: Mapping[str, Any], changed_slugs: set[str]
) -> dict[str, Any]:
    """Return a batch view containing no record whose pitch changed in cleanup."""
    filtered = deepcopy(dict(status))
    records = [
        deepcopy(row)
        for row in status.get("completed_records", [])
        if row.get("episode_slug") not in changed_slugs
    ]
    present = {str(row.get("episode_slug")) for row in status.get("completed_records", [])}
    excluded = sorted(present & changed_slugs)
    filtered["completed_records"] = records
    filtered["failed_records"] = []
    filtered["episode_count"] = len(records)
    filtered["in_count"] = sum(row.get("actual_decision") == "In" for row in records)
    filtered["out_count"] = sum(row.get("actual_decision") == "Out" for row in records)
    filtered["leakage_exclusions"] = excluded
    return filtered


def extract_override_features(
    investigation: Mapping[str, Any], decision: Mapping[str, Any]
) -> FeatureMap:
    """Extract numeric exception-versus-blocker structure without observed labels."""
    any_check = decision.get("any_check", {})
    standard_check = decision.get("standard_check", {})
    any_likelihood = float(any_check.get("likelihood", 0.0))
    standard_likelihood = float(standard_check.get("likelihood", 0.0))
    fatal = bool(any_check.get("fatal_constraint_present", False))
    any_out = any_check.get("decision") == "Out"
    supporting = len(any_check.get("supporting_rationale_ids", []))
    opposing = len(any_check.get("opposing_rationale_ids", []))
    features: FeatureMap = {
        "override__any_likelihood": any_likelihood,
        "override__standard_likelihood": standard_likelihood,
        "override__likelihood_gap": any_likelihood - standard_likelihood,
        "override__fatal_constraint_present": float(fatal),
        "override__out_without_fatal": float(any_out and not fatal),
        "override__support_minus_opposition": float(supporting - opposing),
        "override__exception_analogy_count": float(
            len(decision.get("exception_analogies", []))
        ),
        "override__any_counterargument_rationale_count": float(
            len(any_check.get("strongest_counterargument", {}).get("rationale_ids", []))
        ),
        "override__standard_counterargument_rationale_count": float(
            len(
                standard_check.get("strongest_counterargument", {}).get(
                    "rationale_ids", []
                )
            )
        ),
        "override__founder_positive_primary_count": 0.0,
        "override__founder_positive_confidence_sum": 0.0,
        "override__founder_negative_count": 0.0,
    }
    tier = _token(
        decision.get("recommended_check_tier", decision.get("check_tier", "unknown"))
    )
    features[f"override__check_tier__{tier}"] = 1.0

    for risk in decision.get("risk_ledger", []):
        risk_type = _token(risk.get("risk_type", "unknown"))
        _increment(features, f"override__risk__{risk_type}__count")
        if risk.get("controlling_for_any_check") is True:
            _increment(
                features,
                f"override__risk__{risk_type}__controlling_count",
            )
            _increment(features, "override__risk__all__controlling_count")

    for rationale in investigation.get("rationales", []):
        label = _token(rationale.get("label", ""))
        if not (label.startswith("founder_") or label.startswith("founding_team")):
            continue
        direction = _token(rationale.get("direction", "neutral"))
        confidence = float(rationale.get("confidence", 0.0))
        salience = _token(rationale.get("salience", "secondary"))
        if direction == "positive":
            _increment(
                features,
                "override__founder_positive_confidence_sum",
                confidence,
            )
            if salience == "primary":
                _increment(features, "override__founder_positive_primary_count")
        elif direction == "negative":
            _increment(features, "override__founder_negative_count")
    return features


def augment_override_records(
    records: Sequence[CalibrationRecord],
) -> list[CalibrationRecord]:
    """Add a semantic+risk+override family to verified calibration records."""
    augmented: list[CalibrationRecord] = []
    for record in records:
        root = Path(record.artifact_root)
        investigation = json.loads(
            (root / "phase1/investigation.json").read_text(encoding="utf-8")
        )
        decision = json.loads(
            (root / "phase2/decision.json").read_text(encoding="utf-8")
        )
        combined = {
            **{
                f"p1__{name}": value
                for name, value in record.features["semantic_phase1"].items()
            },
            **{
                f"p2__{name}": value
                for name, value in record.features["phase2"].items()
            },
            **extract_override_features(investigation, decision),
        }
        families = dict(record.features)
        families["semantic_plus_phase2_plus_override"] = combined
        augmented.append(replace(record, features=families))
    return augmented
