"""Observable error diagnostics for the Phase 1 v4.3 development gate."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import fmean, median
from typing import Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .phase1_calibration_cases import CalibrationCase
from .phase1_calibration_models import CalibrationPrediction
from .phase1_evaluation import ReferenceRationale, TaxonomyLabel


@dataclass(frozen=True)
class ErrorLedgerRow:
    vc_slug: str
    vc_name: str
    episode_slug: str
    actual_decision: str
    source_tier: str
    source_format: str
    label: str
    family: str
    outcome: str
    error_type: str
    raw_predicted: bool
    raw_confidence: float
    raw_direction: str
    raw_salience: str
    reference: bool
    reference_confidence: float | None
    reference_direction: str | None
    reference_salience: str | None
    reference_activation: str | None
    reference_utterance_type: str | None
    reference_decision_link: str | None
    direction_disagreement: bool
    salience_disagreement: bool
    neighbor_label: str | None
    neighbor_similarity: float | None
    filtering_selected: bool
    filtering_score: float
    filtering_effect: str


@dataclass(frozen=True)
class CanarySelection:
    vc_slug: str
    vc_name: str
    episode_slug: str
    role: str
    actual_decision: str
    source_tier: str
    source_format: str
    precision: float
    recall: float
    f1: float
    false_negatives: int
    false_positives: int
    weighted_omission_severity: float
    dominant_failure_type: str
    selection_reason: str
    fallback_used: bool


def _reference_rows(case: CalibrationCase) -> dict[str, ReferenceRationale]:
    grouped: dict[str, list[ReferenceRationale]] = {}
    for row in case.phase1_case.reference:
        grouped.setdefault(row.label, []).append(row)
    return {
        label: max(
            rows,
            key=lambda row: (
                row.decision_link == "explicit",
                row.utterance_type == "decision_reason",
                row.activation == "evaluated",
                row.salience == "primary",
                row.confidence,
            ),
        )
        for label, rows in grouped.items()
    }


def _taxonomy_similarities(
    labels: Sequence[str], taxonomy: Mapping[str, TaxonomyLabel]
) -> np.ndarray:
    documents = [
        f"{label.replace('_', ' ')} {taxonomy[label].definition} "
        f"{taxonomy[label].coarse_parent.replace('_', ' ')}"
        for label in labels
    ]
    matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(documents)
    return np.asarray(cosine_similarity(matrix), dtype=float)


def _filtering_map(
    cases: Sequence[CalibrationCase], predictions: Sequence[CalibrationPrediction]
) -> dict[tuple[str, str, str], CalibrationPrediction]:
    expected = {
        (case.vc_slug, case.episode_slug, label)
        for case in cases
        for label in case.labels
    }
    result: dict[tuple[str, str, str], CalibrationPrediction] = {}
    for row in predictions:
        if row.condition != "filtering_only":
            raise ValueError("Gate A requires filtering_only calibration predictions")
        key = row.vc_slug, row.episode_slug, row.label
        if key in result:
            raise ValueError(f"duplicate filtering prediction: {key}")
        result[key] = row
    if set(result) != expected:
        raise ValueError(
            f"filtering prediction coverage mismatch; "
            f"missing={len(expected-set(result))} unexpected={len(set(result)-expected)}"
        )
    return result


def _outcome(reference: bool, predicted: bool) -> str:
    if reference and predicted:
        return "true_positive"
    if reference:
        return "false_negative"
    if predicted:
        return "false_positive"
    return "true_negative"


def _filtering_effect(reference: bool, raw: bool, selected: bool) -> str:
    if raw and reference:
        return "retained_true" if selected else "harmful_removal"
    if raw:
        return "retained_false" if selected else "correct_removal"
    if reference:
        return "unchanged_miss"
    return "unchanged"


def build_error_ledger(
    cases: Sequence[CalibrationCase],
    taxonomy: Mapping[str, TaxonomyLabel],
    filtering_predictions: Sequence[CalibrationPrediction],
    *,
    semantic_neighbor_threshold: float = 0.55,
) -> list[ErrorLedgerRow]:
    """Classify observable canonical errors without using Phase 2 or new calls."""
    if not cases:
        raise ValueError("Gate A diagnostics require cases")
    if not 0 <= semantic_neighbor_threshold <= 1:
        raise ValueError("semantic neighbor threshold must be between zero and one")
    labels = tuple(cases[0].labels)
    if set(labels) != set(taxonomy):
        raise ValueError("taxonomy and calibration label coverage differ")
    if any(tuple(case.labels) != labels for case in cases):
        raise ValueError("calibration cases do not share one ordered label set")
    similarities = _taxonomy_similarities(labels, taxonomy)
    label_index = {label: index for index, label in enumerate(labels)}
    filtering = _filtering_map(cases, filtering_predictions)
    rows: list[ErrorLedgerRow] = []
    for case in cases:
        references = _reference_rows(case)
        activated = [label for label in labels if case.raw_signals[label].predicted]
        for label in labels:
            signal = case.raw_signals[label]
            is_reference = bool(case.reference_targets[label])
            reference = references.get(label)
            attribute = case.reference_attributes.get(label)
            outcome = _outcome(is_reference, signal.predicted)
            neighbor_label: str | None = None
            neighbor_similarity: float | None = None
            error_type = "none"
            if outcome == "false_negative":
                same_family = [
                    candidate
                    for candidate in activated
                    if case.families[candidate] == case.families[label]
                ]
                candidates = same_family or activated
                if candidates:
                    neighbor_label = max(
                        candidates,
                        key=lambda candidate: (
                            similarities[label_index[label], label_index[candidate]],
                            case.raw_signals[candidate].confidence,
                            candidate,
                        ),
                    )
                    neighbor_similarity = float(
                        similarities[label_index[label], label_index[neighbor_label]]
                    )
                if (
                    neighbor_similarity is not None
                    and neighbor_similarity >= semantic_neighbor_threshold
                ):
                    error_type = "taxonomy_neighbor_confusion"
                elif same_family:
                    error_type = "family_covered_exact_miss"
                else:
                    error_type = "discovery_omission"
            elif outcome == "false_positive":
                error_type = "unsupported_prediction"
            direction_disagreement = bool(
                outcome == "true_positive"
                and reference is not None
                and signal.direction != reference.direction
            )
            salience_disagreement = bool(
                outcome == "true_positive"
                and reference is not None
                and signal.salience != reference.salience
            )
            if outcome == "true_positive" and (
                direction_disagreement or salience_disagreement
            ):
                error_type = "attribute_disagreement"
            filtered = filtering[(case.vc_slug, case.episode_slug, label)]
            rows.append(
                ErrorLedgerRow(
                    vc_slug=case.vc_slug,
                    vc_name=case.vc_name,
                    episode_slug=case.episode_slug,
                    actual_decision=case.actual_decision,
                    source_tier=case.source_tier,
                    source_format=case.source_format,
                    label=label,
                    family=case.families[label],
                    outcome=outcome,
                    error_type=error_type,
                    raw_predicted=signal.predicted,
                    raw_confidence=signal.confidence,
                    raw_direction=signal.direction,
                    raw_salience=signal.salience,
                    reference=is_reference,
                    reference_confidence=reference.confidence if reference else None,
                    reference_direction=reference.direction if reference else None,
                    reference_salience=reference.salience if reference else None,
                    reference_activation=attribute.activation if attribute else None,
                    reference_utterance_type=(attribute.utterance_type if attribute else None),
                    reference_decision_link=(attribute.decision_link if attribute else None),
                    direction_disagreement=direction_disagreement,
                    salience_disagreement=salience_disagreement,
                    neighbor_label=neighbor_label,
                    neighbor_similarity=neighbor_similarity,
                    filtering_selected=filtered.predicted,
                    filtering_score=filtered.score,
                    filtering_effect=_filtering_effect(
                        is_reference, signal.predicted, filtered.predicted
                    ),
                )
            )
    return rows


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _omission_severity(row: ErrorLedgerRow) -> float:
    if row.outcome != "false_negative":
        return 0.0
    return (
        1.0
        + 2.0 * (row.reference_salience == "primary")
        + 2.0 * (row.reference_decision_link == "explicit")
        + 1.5 * (row.reference_utterance_type == "decision_reason")
        + 0.5 * (row.reference_activation == "evaluated")
    )


def _aggregate(rows: Sequence[ErrorLedgerRow]) -> dict[str, int | float]:
    tp = sum(row.outcome == "true_positive" for row in rows)
    fp = sum(row.outcome == "false_positive" for row in rows)
    fn = sum(row.outcome == "false_negative" for row in rows)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    predicted = [row for row in rows if row.raw_predicted]
    references = [row for row in rows if row.reference]
    return {
        "label_rows": len(rows),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "raw_rationale_count": len(predicted),
        "reference_rationale_count": len(references),
        "average_raw_confidence": fmean(row.raw_confidence for row in predicted)
        if predicted
        else 0.0,
        "weighted_omission_severity": sum(_omission_severity(row) for row in rows),
        "discovery_omissions": sum(row.error_type == "discovery_omission" for row in rows),
        "taxonomy_neighbor_confusions": sum(
            row.error_type == "taxonomy_neighbor_confusion" for row in rows
        ),
        "family_covered_exact_misses": sum(
            row.error_type == "family_covered_exact_miss" for row in rows
        ),
        "attribute_disagreements": sum(
            row.error_type == "attribute_disagreement" for row in rows
        ),
        "correct_filtering_removals": sum(
            row.filtering_effect == "correct_removal" for row in rows
        ),
        "harmful_filtering_removals": sum(
            row.filtering_effect == "harmful_removal" for row in rows
        ),
    }


def _dominant_failure(rows: Sequence[ErrorLedgerRow]) -> str:
    weighted = Counter()
    for row in rows:
        if row.error_type in {
            "discovery_omission",
            "taxonomy_neighbor_confusion",
            "family_covered_exact_miss",
        }:
            weighted[row.error_type] += _omission_severity(row)
        elif row.error_type == "attribute_disagreement":
            weighted[row.error_type] += 1.0
    return max(weighted, key=lambda value: (weighted[value], value)) if weighted else "none"


def _group_rows(
    rows: Sequence[ErrorLedgerRow], field: str, output_field: str
) -> list[dict[str, object]]:
    grouped: dict[str, list[ErrorLedgerRow]] = defaultdict(list)
    for row in rows:
        grouped[str(getattr(row, field))].append(row)
    result = []
    for value in sorted(grouped):
        scoped = grouped[value]
        result.append(
            {
                output_field: value,
                **(
                    {"vc_name": scoped[0].vc_name}
                    if output_field == "vc_slug"
                    else {}
                ),
                **(
                    {"family": scoped[0].family}
                    if output_field == "label"
                    else {}
                ),
                **_aggregate(scoped),
                "dominant_failure_type": _dominant_failure(scoped),
            }
        )
    return result


def summarize_diagnostics(rows: Sequence[ErrorLedgerRow]) -> dict[str, list[dict[str, object]]]:
    """Create case and population summaries from a complete error ledger."""
    if not rows:
        raise ValueError("diagnostic summary requires error rows")
    grouped_cases: dict[tuple[str, str], list[ErrorLedgerRow]] = defaultdict(list)
    for row in rows:
        grouped_cases[(row.vc_slug, row.episode_slug)].append(row)
    case_rows: list[dict[str, object]] = []
    for key in sorted(grouped_cases):
        scoped = grouped_cases[key]
        first = scoped[0]
        case_rows.append(
            {
                "vc_slug": first.vc_slug,
                "vc_name": first.vc_name,
                "episode_slug": first.episode_slug,
                "actual_decision": first.actual_decision,
                "source_tier": first.source_tier,
                "source_format": first.source_format,
                **_aggregate(scoped),
                "dominant_failure_type": _dominant_failure(scoped),
            }
        )
    confusion_counts = Counter(
        (row.label, row.neighbor_label, row.family)
        for row in rows
        if row.error_type == "taxonomy_neighbor_confusion" and row.neighbor_label
    )
    confusion_rows = [
        {
            "reference_label": reference,
            "predicted_neighbor_label": neighbor,
            "family": family,
            "count": count,
        }
        for (reference, neighbor, family), count in sorted(
            confusion_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    missed_salient = [
        asdict(row)
        for row in rows
        if row.outcome == "false_negative"
        and (
            row.reference_salience == "primary"
            or row.reference_decision_link == "explicit"
            or row.reference_utterance_type == "decision_reason"
        )
    ]
    missed_salient.sort(
        key=lambda row: (
            -_omission_severity(ErrorLedgerRow(**row)),
            row["vc_slug"],
            row["episode_slug"],
            row["label"],
        )
    )
    return {
        "case_rows": case_rows,
        "vc_rows": _group_rows(rows, "vc_slug", "vc_slug"),
        "label_rows": _group_rows(rows, "label", "label"),
        "family_rows": _group_rows(rows, "family", "family"),
        "confusion_rows": confusion_rows,
        "missed_salient_rows": missed_salient,
    }


def _selection(
    row: Mapping[str, object], role: str, reason: str, fallback: bool
) -> CanarySelection:
    return CanarySelection(
        vc_slug=str(row["vc_slug"]),
        vc_name=str(row["vc_name"]),
        episode_slug=str(row["episode_slug"]),
        role=role,
        actual_decision=str(row["actual_decision"]),
        source_tier=str(row["source_tier"]),
        source_format=str(row["source_format"]),
        precision=float(row["precision"]),
        recall=float(row["recall"]),
        f1=float(row["f1"]),
        false_negatives=int(row["false_negatives"]),
        false_positives=int(row["false_positives"]),
        weighted_omission_severity=float(row["weighted_omission_severity"]),
        dominant_failure_type=str(row["dominant_failure_type"]),
        selection_reason=reason,
        fallback_used=fallback,
    )


def select_canary_cases(
    case_rows: Sequence[Mapping[str, object]],
    ledger_rows: Sequence[ErrorLedgerRow],
    *,
    per_vc: int = 3,
) -> list[CanarySelection]:
    """Select coverage, precision, and control cases deterministically per VC."""
    if per_vc != 3:
        raise ValueError("v4.3 canary design requires exactly three cases per VC")
    if not case_rows or not ledger_rows:
        raise ValueError("canary selection requires case and ledger rows")
    expected_keys = {(row.vc_slug, row.episode_slug) for row in ledger_rows}
    observed_keys = {
        (str(row["vc_slug"]), str(row["episode_slug"])) for row in case_rows
    }
    if observed_keys != expected_keys:
        raise ValueError("case summaries and ledger rows do not align")
    by_vc: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in case_rows:
        by_vc[str(row["vc_slug"])].append(row)
    result: list[CanarySelection] = []
    for vc_slug in sorted(by_vc):
        available = list(by_vc[vc_slug])
        if len(available) < 3:
            raise ValueError(f"VC has fewer than three canary candidates: {vc_slug}")
        coverage = min(
            available,
            key=lambda row: (
                -float(row["weighted_omission_severity"]),
                -int(row["false_negatives"]),
                row["source_tier"] == "legacy_fallback",
                str(row["episode_slug"]),
            ),
        )
        result.append(
            _selection(
                coverage,
                "coverage",
                "highest weighted missed-rationale severity",
                float(coverage["weighted_omission_severity"]) == 0.0,
            )
        )
        available.remove(coverage)
        precision = min(
            available,
            key=lambda row: (
                -int(row["false_positives"]),
                -int(row["raw_rationale_count"]),
                str(row["episode_slug"]),
            ),
        )
        result.append(
            _selection(
                precision,
                "precision",
                "highest unsupported-rationale count",
                int(precision["false_positives"]) == 0,
            )
        )
        available.remove(precision)
        vc_reference_median = median(
            float(row["reference_rationale_count"]) for row in by_vc[vc_slug]
        )
        control = min(
            available,
            key=lambda row: (
                -float(row["f1"]),
                abs(float(row["reference_rationale_count"]) - vc_reference_median),
                str(row["episode_slug"]),
            ),
        )
        result.append(
            _selection(
                control,
                "control",
                "highest remaining exact-label F1 near the VC reference-count median",
                float(control["f1"]) == 0.0,
            )
        )
    role_order = {"coverage": 0, "precision": 1, "control": 2}
    return sorted(result, key=lambda row: (row.vc_slug, role_order[row.role]))


def diagnose_intervention(rows: Sequence[ErrorLedgerRow]) -> dict[str, object]:
    """Select one Gate B intervention only when the observable pattern is broad."""
    if not rows:
        raise ValueError("intervention diagnosis requires error rows")
    by_vc: dict[str, list[ErrorLedgerRow]] = defaultdict(list)
    for row in rows:
        by_vc[row.vc_slug].append(row)

    def masses(scoped: Sequence[ErrorLedgerRow]) -> tuple[float, float, float]:
        family = sum(
            _omission_severity(row)
            for row in scoped
            if row.error_type
            in {"taxonomy_neighbor_confusion", "family_covered_exact_miss"}
        )
        discovery = sum(
            _omission_severity(row)
            for row in scoped
            if row.error_type == "discovery_omission"
        )
        return family, discovery, family + discovery

    family, discovery, total = masses(rows)
    per_vc = []
    for vc_slug in sorted(by_vc):
        vc_family, vc_discovery, vc_total = masses(by_vc[vc_slug])
        per_vc.append(
            {
                "vc_slug": vc_slug,
                "vc_name": by_vc[vc_slug][0].vc_name,
                "family_covered_weighted_share": _ratio(vc_family, vc_total),
                "discovery_weighted_share": _ratio(vc_discovery, vc_total),
                "dominant_pattern": (
                    "family_covered" if vc_family > vc_discovery else "discovery"
                ),
            }
        )
    family_support = sum(row["dominant_pattern"] == "family_covered" for row in per_vc)
    discovery_support = sum(row["dominant_pattern"] == "discovery" for row in per_vc)
    family_share, discovery_share = _ratio(family, total), _ratio(discovery, total)
    if family_share >= 0.60 and family_support >= 4:
        selected = "contrastive_within_family_mapping"
        support = family_support
        reason = (
            "At least 60% of weighted salient misses had an activated rationale in "
            "the same family, and that pattern dominated in at least four VCs."
        )
    elif discovery_share >= 0.50 and discovery_support >= 4:
        selected = "recall_oriented_hypothesis_pass"
        support = discovery_support
        reason = (
            "At least 50% of weighted salient misses lacked family coverage, and "
            "that pattern dominated in at least four VCs."
        )
    else:
        selected = "inconclusive_no_api_run"
        support = max(family_support, discovery_support)
        reason = "No single observable failure class met the prespecified breadth gate."
    return {
        "schema": "phase1-v43-intervention-diagnosis-v1",
        "selected_intervention": selected,
        "reason": reason,
        "family_covered_weighted_share": family_share,
        "discovery_weighted_share": discovery_share,
        "supporting_vc_count": support,
        "vc_count": len(per_vc),
        "per_vc": per_vc,
        "api_cost_usd": 0.0,
    }
