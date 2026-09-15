"""Deterministic evaluation of canonical Phase 1 rationale investigations."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score

from vc_clone_graph.evaluation import load_canonical_artifacts, load_registry


RICH_SOURCE_FORMAT = "transcript-observed-rationales-v1"
REFERENCE_SCHEMA = "phase1-rationale-reference-v1"
SALIENCE_WEIGHTS = {"primary": 2.0, "secondary": 1.0}
CONFIDENCE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


def phase1_artifact_filename(artifact: str) -> str:
    if artifact == "final":
        return "investigation.json"
    if artifact == "candidate":
        return "candidate-investigation.json"
    raise ValueError(f"unsupported Phase 1 artifact selector: {artifact}")


@dataclass(frozen=True)
class TaxonomyLabel:
    label: str
    definition: str
    coarse_parent: str


@dataclass(frozen=True)
class PredictedRationale:
    label: str
    direction: str
    salience: str
    confidence: float
    justification: str


@dataclass(frozen=True)
class ReferenceRationale:
    label: str
    direction: str
    salience: str
    confidence: float
    activation: str
    utterance_type: str
    decision_link: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class Phase1Case:
    vc_slug: str
    vc_name: str
    episode_slug: str
    actual_decision: str
    source_tier: str
    source_format: str
    predicted: tuple[PredictedRationale, ...]
    reference: tuple[ReferenceRationale, ...]
    artifact_path: Path | None
    reference_path: Path | None


def _object(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _sequence(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def _confidence(value: object, context: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{context} confidence must be numeric")
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError(f"{context} confidence must be between zero and one")
    return result


def load_taxonomy(path: Path) -> dict[str, TaxonomyLabel]:
    payload = _sequence(json.loads(path.read_text(encoding="utf-8")), f"taxonomy {path}")
    result: dict[str, TaxonomyLabel] = {}
    for index, raw_value in enumerate(payload):
        raw = _object(raw_value, f"taxonomy row {index}")
        label = raw.get("label")
        definition = raw.get("definition")
        parent = raw.get("coarse_parent")
        if not all(isinstance(item, str) and item for item in (label, definition, parent)):
            raise ValueError(f"malformed taxonomy row {index}")
        if label in result:
            raise ValueError(f"duplicate taxonomy label: {label}")
        result[label] = TaxonomyLabel(label, definition, parent)
    if len(result) != 44:
        raise ValueError(f"expected 44 taxonomy labels, found {len(result)}")
    return result


def _verify_sha256(path: Path) -> None:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing Phase 1 hash sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != observed:
        raise ValueError(f"Phase 1 hash mismatch: {path}")


def _prediction_rationales(
    payload: Mapping[str, Any], taxonomy: Mapping[str, TaxonomyLabel], context: str
) -> tuple[PredictedRationale, ...]:
    version = payload.get("schema_version")
    if not isinstance(version, str) or not version.startswith("investigation-v4"):
        raise ValueError(f"unsupported Phase 1 schema {version!r}: {context}")
    result: list[PredictedRationale] = []
    for index, raw_value in enumerate(_sequence(payload.get("rationales"), f"{context} rationales")):
        raw = _object(raw_value, f"{context} rationale {index}")
        label = raw.get("taxonomy_label")
        if label not in taxonomy:
            raise ValueError(f"invalid predicted taxonomy label {label!r}: {context}")
        direction, salience = raw.get("direction"), raw.get("salience")
        if direction not in {"positive", "negative", "neutral"}:
            raise ValueError(f"invalid predicted direction {direction!r}: {context}")
        if salience not in {"primary", "secondary"}:
            raise ValueError(f"invalid predicted salience {salience!r}: {context}")
        justification = raw.get("justification")
        if not isinstance(justification, str) or not justification.strip():
            raise ValueError(f"missing predicted justification: {context}")
        result.append(
            PredictedRationale(
                str(label), str(direction), str(salience),
                _confidence(raw.get("confidence"), context), justification.strip()
            )
        )
    return tuple(result)


def _reference_rationales(
    payload: Mapping[str, Any], taxonomy: Mapping[str, TaxonomyLabel], context: str
) -> tuple[ReferenceRationale, ...]:
    result: list[ReferenceRationale] = []
    for index, raw_value in enumerate(_sequence(payload.get("rationales"), f"{context} rationales")):
        raw = _object(raw_value, f"{context} rationale {index}")
        label = raw.get("rationale_label")
        if label not in taxonomy:
            raise ValueError(f"invalid reference taxonomy label {label!r}: {context}")
        evidence = _sequence(raw.get("evidence"), f"{context} evidence {index}")
        if not all(isinstance(item, str) and item.strip() for item in evidence):
            raise ValueError(f"invalid reference evidence: {context}")
        result.append(
            ReferenceRationale(
                label=str(label),
                direction=str(raw.get("direction")),
                salience=str(raw.get("salience")),
                confidence=_confidence(raw.get("confidence"), context),
                activation=str(raw.get("activation")),
                utterance_type=str(raw.get("utterance_type")),
                decision_link=str(raw.get("decision_link")),
                evidence=tuple(item.strip() for item in evidence),
            )
        )
    return tuple(result)


def load_phase1_cases(
    project_root: Path,
    registry_path: Path,
    reference_root: Path,
    taxonomy_path: Path,
    selected_vcs: Sequence[str] | None = None,
    artifact: str = "final",
) -> tuple[list[Phase1Case], dict[str, TaxonomyLabel]]:
    project_root = project_root.resolve()
    registry = load_registry(registry_path)
    taxonomy = load_taxonomy(taxonomy_path.resolve())
    predictions = load_canonical_artifacts(registry, selected_vcs)
    reference_root = reference_root.resolve()
    manifest = _object(
        json.loads((reference_root / "manifest.json").read_text(encoding="utf-8")),
        "reference manifest",
    )
    if manifest.get("missing_count") != 0:
        raise ValueError("reference manifest reports missing cases")
    reference_files: dict[tuple[str, str], Path] = {}
    for path in (reference_root / "records").rglob("*.json"):
        payload = _object(json.loads(path.read_text(encoding="utf-8")), f"reference {path}")
        key = (str(payload.get("vc_slug")), str(payload.get("episode_slug")))
        if key in reference_files:
            raise ValueError(f"duplicate reference record: {key}")
        reference_files[key] = path
    expected = {(row.vc_slug, row.episode_slug) for row in predictions}
    selected_reference_files = {key: path for key, path in reference_files.items() if key[0] in {row.vc_slug for row in predictions}}
    if set(selected_reference_files) != expected:
        missing = sorted(expected - set(selected_reference_files))
        unexpected = sorted(set(selected_reference_files) - expected)
        raise ValueError(f"reference coverage mismatch; missing={missing}, unexpected={unexpected}")
    cases: list[Phase1Case] = []
    artifact_filename = phase1_artifact_filename(artifact)
    for row in predictions:
        artifact_path = row.artifact_path.parent / "phase1" / artifact_filename
        if not artifact_path.is_file():
            raise ValueError(f"missing Phase 1 investigation: {artifact_path}")
        _verify_sha256(artifact_path)
        predicted_payload = _object(json.loads(artifact_path.read_text(encoding="utf-8")), f"Phase 1 {artifact_path}")
        if predicted_payload.get("episode_slug") != row.episode_slug:
            raise ValueError(f"Phase 1 episode mismatch: {artifact_path}")
        reference_path = selected_reference_files[(row.vc_slug, row.episode_slug)]
        reference_payload = _object(json.loads(reference_path.read_text(encoding="utf-8")), f"reference {reference_path}")
        if reference_payload.get("schema") != REFERENCE_SCHEMA:
            raise ValueError(f"unsupported reference schema: {reference_path}")
        if reference_payload.get("actual_decision") != row.actual_decision:
            raise ValueError(f"reference decision mismatch: {reference_path}")
        cases.append(
            Phase1Case(
                vc_slug=row.vc_slug,
                vc_name=row.vc_name,
                episode_slug=row.episode_slug,
                actual_decision=row.actual_decision,
                source_tier=str(reference_payload.get("source_tier")),
                source_format=str(reference_payload.get("source_format")),
                predicted=_prediction_rationales(predicted_payload, taxonomy, str(artifact_path)),
                reference=_reference_rationales(reference_payload, taxonomy, str(reference_path)),
                artifact_path=artifact_path.relative_to(project_root) if artifact_path.is_relative_to(project_root) else artifact_path,
                reference_path=reference_path.relative_to(project_root) if reference_path.is_relative_to(project_root) else reference_path,
            )
        )
    if selected_vcs is None and len(cases) != int(manifest.get("record_count", -1)):
        raise ValueError(f"canonical/reference count mismatch: {len(cases)}")
    return cases, taxonomy


def _ratio(numerator: int | float, denominator: int | float, *, empty: float = 0.0) -> float:
    return float(numerator) / float(denominator) if denominator else empty


def score_set(predicted: set[Any], reference: set[Any]) -> dict[str, int | float]:
    tp = len(predicted & reference)
    fp = len(predicted - reference)
    fn = len(reference - predicted)
    both_empty = not predicted and not reference
    precision = _ratio(tp, tp + fp, empty=1.0 if both_empty else 0.0)
    recall = _ratio(tp, tp + fn, empty=1.0)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall, empty=0.0),
        "jaccard": _ratio(tp, tp + fp + fn, empty=1.0),
    }


def _label_sets(case: Phase1Case) -> tuple[set[str], set[str]]:
    return ({item.label for item in case.predicted}, {item.label for item in case.reference})


def _aggregate_set_metrics(pairs: Iterable[tuple[set[Any], set[Any]]]) -> dict[str, int | float]:
    scored = [score_set(predicted, reference) for predicted, reference in pairs]
    if not scored:
        raise ValueError("metric aggregation requires at least one case")
    tp, fp, fn = (sum(int(row[key]) for row in scored) for key in ("tp", "fp", "fn"))
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return {
        "n": len(scored),
        "micro_tp": tp,
        "micro_fp": fp,
        "micro_fn": fn,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": _ratio(2 * precision * recall, precision + recall),
        "macro_precision": fmean(float(row["precision"]) for row in scored),
        "macro_recall": fmean(float(row["recall"]) for row in scored),
        "macro_f1": fmean(float(row["f1"]) for row in scored),
        "macro_jaccard": fmean(float(row["jaccard"]) for row in scored),
    }


def _scope_metrics(cases: Sequence[Phase1Case], taxonomy: Mapping[str, TaxonomyLabel]) -> dict[str, int | float]:
    pairs = [_label_sets(case) for case in cases]
    result = _aggregate_set_metrics(pairs)
    result.update(
        {
            "exact_set_match_rate": fmean(pred == ref for pred, ref in pairs),
            "at_least_one_correct_rate": fmean(bool(pred & ref) for pred, ref in pairs),
            "zero_correct_rate": fmean(not bool(pred & ref) for pred, ref in pairs),
            "average_predicted_size": fmean(len(pred) for pred, _ in pairs),
            "average_reference_size": fmean(len(ref) for _, ref in pairs),
        }
    )
    family_pairs = [
        (
            {taxonomy[label].coarse_parent for label in predicted},
            {taxonomy[label].coarse_parent for label in reference},
        )
        for predicted, reference in pairs
    ]
    family = _aggregate_set_metrics(family_pairs)
    for key, value in family.items():
        if key != "n":
            result[f"family_{key}"] = value
    return result


def _weighted_precision(matched: float, predicted_mass: float, reference_mass: float) -> float:
    if predicted_mass:
        return matched / predicted_mass
    return 1.0 if not reference_mass else 0.0


def _weighted_recall(matched: float, reference_mass: float) -> float:
    return matched / reference_mass if reference_mass else 1.0


def _rationale_masses(
    rationales: Sequence[PredictedRationale | ReferenceRationale],
    mode: str,
    salience_weights: Mapping[str, float],
) -> dict[str, float]:
    if mode not in {"salience", "confidence", "combined"}:
        raise ValueError(f"unsupported weighted recovery mode: {mode}")
    masses: dict[str, float] = {}
    for rationale in rationales:
        salience = float(salience_weights[rationale.salience])
        mass = (
            salience
            if mode == "salience"
            else rationale.confidence
            if mode == "confidence"
            else rationale.confidence * salience
        )
        masses[rationale.label] = max(masses.get(rationale.label, 0.0), mass)
    return masses


def weighted_recovery_metrics(
    cases: Sequence[Phase1Case],
    *,
    mode: str,
    salience_weights: Mapping[str, float] = SALIENCE_WEIGHTS,
) -> dict[str, int | float]:
    if not cases:
        raise ValueError("weighted recovery requires at least one case")
    if set(salience_weights) != {"primary", "secondary"} or any(
        not isinstance(value, int | float) or value <= 0
        for value in salience_weights.values()
    ):
        raise ValueError("salience weights must be positive primary and secondary values")
    episode_rows: list[tuple[float, float, float, float, float, float]] = []
    for case in cases:
        predicted = _rationale_masses(case.predicted, mode, salience_weights)
        reference = _rationale_masses(case.reference, mode, salience_weights)
        predicted_mass = sum(predicted.values())
        reference_mass = sum(reference.values())
        matched_mass = sum(
            min(predicted[label], reference[label])
            for label in predicted.keys() & reference.keys()
        )
        precision = _weighted_precision(matched_mass, predicted_mass, reference_mass)
        recall = _weighted_recall(matched_mass, reference_mass)
        f1 = _ratio(2 * precision * recall, precision + recall)
        episode_rows.append(
            (matched_mass, predicted_mass, reference_mass, precision, recall, f1)
        )
    matched_mass = sum(row[0] for row in episode_rows)
    predicted_mass = sum(row[1] for row in episode_rows)
    reference_mass = sum(row[2] for row in episode_rows)
    precision = _weighted_precision(matched_mass, predicted_mass, reference_mass)
    recall = _weighted_recall(matched_mass, reference_mass)
    return {
        "n": len(cases),
        "matched_mass": matched_mass,
        "predicted_mass": predicted_mass,
        "reference_mass": reference_mass,
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "macro_precision": fmean(row[3] for row in episode_rows),
        "macro_recall": fmean(row[4] for row in episode_rows),
        "macro_f1": fmean(row[5] for row in episode_rows),
    }


def calibration_metrics(true: Sequence[int], scores: Sequence[float], *, bins: int = 10) -> dict[str, float]:
    if len(true) != len(scores) or not true:
        raise ValueError("calibration arrays must be non-empty and equally sized")
    brier = fmean((float(score) - int(label)) ** 2 for label, score in zip(true, scores, strict=True))
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, score in enumerate(scores) if low <= score < high or (index == bins - 1 and score == 1.0)]
        if members:
            accuracy = fmean(true[i] for i in members)
            confidence = fmean(scores[i] for i in members)
            ece += len(members) / len(true) * abs(accuracy - confidence)
    return {"brier": brier, "ece": ece}


def _ranked_labels(case: Phase1Case, taxonomy: Mapping[str, TaxonomyLabel]) -> list[tuple[str, float]]:
    scores: dict[str, float] = {label: 0.0 for label in taxonomy}
    for item in case.predicted:
        scores[item.label] = max(scores[item.label], item.confidence)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _ranking(
    cases: Sequence[Phase1Case], taxonomy: Mapping[str, TaxonomyLabel], top_ks: Sequence[int]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    true: list[int] = []
    scores: list[float] = []
    ranks: dict[str, list[tuple[str, float]]] = {}
    for case in cases:
        reference = {item.label for item in case.reference}
        ranked = _ranked_labels(case, taxonomy)
        ranks[case.episode_slug + "\0" + case.vc_slug] = ranked
        for label, score in ranked:
            true.append(int(label in reference))
            scores.append(score)
    positives = sum(true)
    negatives = len(true) - positives
    summary: dict[str, Any] = {
        "candidate_pairs": len(true),
        "positive_pairs": positives,
        "prevalence": _ratio(positives, len(true)),
        "average_precision": float(average_precision_score(true, scores)) if positives else None,
        "roc_auc": float(roc_auc_score(true, scores)) if positives and negatives else None,
        **calibration_metrics(true, scores),
    }
    rows: list[dict[str, Any]] = []
    for k in top_ks:
        recalls, precisions, primary_recalls = [], [], []
        hits_total = reference_total = primary_total = 0
        for case in cases:
            ranked = ranks[case.episode_slug + "\0" + case.vc_slug]
            selected = {label for label, _ in ranked[: min(k, len(ranked))]}
            reference = {item.label for item in case.reference}
            primary = {item.label for item in case.reference if item.salience == "primary"}
            hits = len(selected & reference)
            hits_total += hits
            reference_total += len(reference)
            primary_total += len(primary)
            recalls.append(_ratio(hits, len(reference), empty=1.0))
            precisions.append(_ratio(hits, min(k, len(ranked))))
            primary_recalls.append(_ratio(len(selected & primary), len(primary), empty=1.0))
        rows.append(
            {
                "scope": "overall",
                "k": k,
                "macro_precision_at_k": fmean(precisions),
                "macro_recall_at_k": fmean(recalls),
                "precision_at_k": _ratio(hits_total, len(cases) * min(k, len(taxonomy))),
                "recall_at_k": _ratio(hits_total, reference_total),
                "primary_recall_at_k": _ratio(
                    sum(
                        len({label for label, _ in ranks[c.episode_slug + "\0" + c.vc_slug][:k]} & {r.label for r in c.reference if r.salience == "primary"})
                        for c in cases
                    ),
                    primary_total,
                ),
                "macro_primary_recall_at_k": fmean(primary_recalls),
            }
        )
    return summary, rows


def _attribute_pairs(case: Phase1Case, kind: str) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
    def encode(item: PredictedRationale | ReferenceRationale) -> tuple[str, ...]:
        if kind == "direction":
            return item.label, item.direction
        if kind == "salience":
            return item.label, item.salience
        return item.label, item.direction, item.salience
    return ({encode(item) for item in case.predicted}, {encode(item) for item in case.reference})


def _conditional_agreement(cases: Sequence[Phase1Case], attribute: str) -> float | None:
    agreements: list[bool] = []
    for case in cases:
        pred_by_label: dict[str, set[str]] = defaultdict(set)
        ref_by_label: dict[str, set[str]] = defaultdict(set)
        for item in case.predicted:
            pred_by_label[item.label].add(str(getattr(item, attribute)))
        for item in case.reference:
            ref_by_label[item.label].add(str(getattr(item, attribute)))
        for label in pred_by_label.keys() & ref_by_label.keys():
            agreements.append(bool(pred_by_label[label] & ref_by_label[label]))
    return fmean(agreements) if agreements else None


def _rich_rows(cases: Sequence[Phase1Case]) -> list[dict[str, Any]]:
    rich = [case for case in cases if case.source_format == RICH_SOURCE_FORMAT]
    definitions = [
        ("activation:queried", lambda r: r.activation == "queried"),
        ("activation:evaluated", lambda r: r.activation == "evaluated"),
        ("utterance_type:question", lambda r: r.utterance_type == "question"),
        ("utterance_type:assessment", lambda r: r.utterance_type == "assessment"),
        ("utterance_type:decision_reason", lambda r: r.utterance_type == "decision_reason"),
        ("decision_link:explicit", lambda r: r.decision_link == "explicit"),
        ("primary_and_explicit", lambda r: r.salience == "primary" and r.decision_link == "explicit"),
    ]
    rows: list[dict[str, Any]] = []
    for name, predicate in definitions:
        reference_count = recovered = cases_with_reference = 0
        for case in rich:
            targets = [item for item in case.reference if predicate(item)]
            if targets:
                cases_with_reference += 1
            predicted = {item.label for item in case.predicted}
            reference_count += len(targets)
            recovered += sum(item.label in predicted for item in targets)
        rows.append(
            {
                "subset": name,
                "case_count": len(rich),
                "cases_with_reference": cases_with_reference,
                "reference_rationales": reference_count,
                "recovered_rationales": recovered,
                "recall": _ratio(recovered, reference_count),
            }
        )
    return rows


def _group_rows(cases: Sequence[Phase1Case], taxonomy: Mapping[str, TaxonomyLabel]) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[Phase1Case]]] = [("overall", "all", list(cases))]
    for dimension, getter in (
        ("vc", lambda c: c.vc_slug),
        ("decision", lambda c: c.actual_decision),
        ("source_tier", lambda c: c.source_tier),
    ):
        values: dict[str, list[Phase1Case]] = defaultdict(list)
        for case in cases:
            values[getter(case)].append(case)
        groups.extend((dimension, value, grouped) for value, grouped in sorted(values.items()))
    rows = []
    for dimension, value, grouped in groups:
        rows.append({"dimension": dimension, "value": value, **_scope_metrics(grouped, taxonomy)})
    return rows


def _evaluation_scopes(
    cases: Sequence[Phase1Case],
) -> list[tuple[str, list[Phase1Case]]]:
    scopes: list[tuple[str, list[Phase1Case]]] = [("overall", list(cases))]
    for vc_slug in sorted({case.vc_slug for case in cases}):
        scopes.append((vc_slug, [case for case in cases if case.vc_slug == vc_slug]))
    return scopes


def _salience_strata(cases: Sequence[Phase1Case]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, scoped_cases in _evaluation_scopes(cases):
        for salience in ("primary", "secondary"):
            cases_with_reference = 0
            reference_count = recovered_count = predicted_count = correct_count = 0
            recovered_confidences: list[float] = []
            for case in scoped_cases:
                predicted_labels = {item.label for item in case.predicted}
                reference_labels = {item.label for item in case.reference}
                predicted_by_label: dict[str, float] = defaultdict(float)
                for item in case.predicted:
                    predicted_by_label[item.label] = max(
                        predicted_by_label[item.label], item.confidence
                    )
                reference_stratum = {
                    item.label for item in case.reference if item.salience == salience
                }
                predicted_stratum = {
                    item.label for item in case.predicted if item.salience == salience
                }
                cases_with_reference += bool(reference_stratum)
                reference_count += len(reference_stratum)
                recovered = reference_stratum & predicted_labels
                recovered_count += len(recovered)
                recovered_confidences.extend(
                    predicted_by_label[label] for label in recovered
                )
                predicted_count += len(predicted_stratum)
                correct_count += len(predicted_stratum & reference_labels)
            rows.append(
                {
                    "scope": scope,
                    "salience": salience,
                    "n": len(scoped_cases),
                    "cases_with_reference": cases_with_reference,
                    "reference_count": reference_count,
                    "recovered_count": recovered_count,
                    "recall": _ratio(recovered_count, reference_count, empty=1.0),
                    "mean_recovered_predicted_confidence": (
                        fmean(recovered_confidences) if recovered_confidences else None
                    ),
                    "predicted_count": predicted_count,
                    "correct_count": correct_count,
                    "precision": _ratio(
                        correct_count,
                        predicted_count,
                        empty=1.0 if not reference_count else 0.0,
                    ),
                }
            )
    return rows


def _weighted_recovery_rows(cases: Sequence[Phase1Case]) -> list[dict[str, Any]]:
    return [
        {
            "scope": scope,
            "mode": mode,
            **weighted_recovery_metrics(scoped_cases, mode=mode),
        }
        for scope, scoped_cases in _evaluation_scopes(cases)
        for mode in ("salience", "confidence", "combined")
    ]


def _confidence_threshold_rows(
    cases: Sequence[Phase1Case],
    thresholds: Sequence[float] = CONFIDENCE_THRESHOLDS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, scoped_cases in _evaluation_scopes(cases):
        for threshold in thresholds:
            pairs: list[tuple[set[str], set[str]]] = []
            retained_counts: list[int] = []
            has_correct: list[bool] = []
            for case in scoped_cases:
                confidence_by_label: dict[str, float] = defaultdict(float)
                for item in case.predicted:
                    confidence_by_label[item.label] = max(
                        confidence_by_label[item.label], item.confidence
                    )
                predicted = {
                    label
                    for label, confidence in confidence_by_label.items()
                    if confidence >= threshold
                }
                reference = {item.label for item in case.reference}
                pairs.append((predicted, reference))
                retained_counts.append(len(predicted))
                has_correct.append(bool(predicted & reference))
            metrics = _aggregate_set_metrics(pairs)
            rows.append(
                {
                    "scope": scope,
                    "threshold": float(threshold),
                    **metrics,
                    "average_retained_count": fmean(retained_counts),
                    "at_least_one_correct_rate": fmean(has_correct),
                }
            )
    return rows


def evaluate_cases(
    cases: Sequence[Phase1Case],
    taxonomy: Mapping[str, TaxonomyLabel],
    *,
    top_ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    if not cases:
        raise ValueError("evaluation requires at least one case")
    if not top_ks or any(not isinstance(k, int) or k <= 0 for k in top_ks):
        raise ValueError("top-k values must be positive integers")
    scope_rows = _group_rows(cases, taxonomy)
    overall = next(row for row in scope_rows if row["dimension"] == "overall")
    ranking_summary, ranking_rows = _ranking(cases, taxonomy, tuple(top_ks))
    ranking_summaries = [{"scope": "overall", **ranking_summary}]
    for vc in sorted({case.vc_slug for case in cases}):
        summary, rows = _ranking([case for case in cases if case.vc_slug == vc], taxonomy, tuple(top_ks))
        for row in rows:
            row["scope"] = vc
        ranking_rows.extend(rows)
        ranking_summaries.append({"scope": vc, **summary})
    attribute_metrics: dict[str, dict[str, Any]] = {}
    for kind in ("direction", "salience", "triple"):
        metrics = _aggregate_set_metrics([_attribute_pairs(case, kind) for case in cases])
        if kind in {"direction", "salience"}:
            metrics["conditional_agreement"] = _conditional_agreement(cases, kind)
        attribute_metrics[kind] = metrics
    episode_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    label_counts: dict[str, Counter[str]] = {label: Counter() for label in taxonomy}
    for case in cases:
        predicted, reference = _label_sets(case)
        metric = score_set(predicted, reference)
        ranks = _ranked_labels(case, taxonomy)
        episode_rows.append(
            {
                "vc_slug": case.vc_slug,
                "episode_slug": case.episode_slug,
                "actual_decision": case.actual_decision,
                "source_tier": case.source_tier,
                "source_format": case.source_format,
                "predicted_count": len(predicted),
                "reference_count": len(reference),
                **metric,
                "exact_set_match": predicted == reference,
                "top_labels": ";".join(label for label, _ in ranks if label in predicted),
                "artifact_path": str(case.artifact_path or ""),
                "reference_path": str(case.reference_path or ""),
            }
        )
        confidence_by_label: dict[str, float] = defaultdict(float)
        for item in case.predicted:
            confidence_by_label[item.label] = max(confidence_by_label[item.label], item.confidence)
        for label in taxonomy:
            pred, ref = label in predicted, label in reference
            status = "tp" if pred and ref else "fp" if pred else "fn" if ref else "tn"
            label_counts[label][status] += 1
            if pred or ref:
                comparison_rows.append(
                    {
                        "vc_slug": case.vc_slug,
                        "episode_slug": case.episode_slug,
                        "actual_decision": case.actual_decision,
                        "label": label,
                        "coarse_parent": taxonomy[label].coarse_parent,
                        "predicted": pred,
                        "reference": ref,
                        "status": status,
                        "confidence": confidence_by_label[label] if pred else 0.0,
                    }
                )
    per_label_rows = []
    for label in taxonomy:
        counts = label_counts[label]
        precision, recall = _ratio(counts["tp"], counts["tp"] + counts["fp"]), _ratio(counts["tp"], counts["tp"] + counts["fn"])
        per_label_rows.append(
            {
                "label": label,
                "coarse_parent": taxonomy[label].coarse_parent,
                "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"], "tn": counts["tn"],
                "precision": precision, "recall": recall,
                "f1": _ratio(2 * precision * recall, precision + recall),
                "support": counts["tp"] + counts["fn"],
            }
        )
    family_rows = []
    for parent in sorted({item.coarse_parent for item in taxonomy.values()}):
        labels = {label for label, item in taxonomy.items() if item.coarse_parent == parent}
        pairs = []
        for case in cases:
            predicted, reference = _label_sets(case)
            pairs.append((predicted & labels, reference & labels))
        family_rows.append({"coarse_parent": parent, **_aggregate_set_metrics(pairs)})
    vc_metrics = [row for row in scope_rows if row["dimension"] == "vc"]
    rate_keys = ["micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1", "macro_jaccard"]
    macro_vc = {key: fmean(float(row[key]) for row in vc_metrics) for key in rate_keys}
    return {
        "case_count": len(cases),
        "rich_case_count": sum(case.source_format == RICH_SOURCE_FORMAT for case in cases),
        "legacy_case_count": sum(case.source_format != RICH_SOURCE_FORMAT for case in cases),
        "vc_count": len({case.vc_slug for case in cases}),
        "overall": overall,
        "macro_vc": macro_vc,
        "scope_rows": scope_rows,
        "ranking_summary": ranking_summary,
        "ranking_summaries": ranking_summaries,
        "ranking_rows": ranking_rows,
        "attribute_metrics": attribute_metrics,
        "rich_subset_rows": _rich_rows(cases),
        "salience_strata_rows": _salience_strata(cases),
        "weighted_recovery_rows": _weighted_recovery_rows(cases),
        "confidence_threshold_rows": _confidence_threshold_rows(cases),
        "salience_weights": dict(SALIENCE_WEIGHTS),
        "confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
        "per_label_rows": per_label_rows,
        "family_rows": family_rows,
        "episode_rows": episode_rows,
        "comparison_rows": comparison_rows,
        "top_ks": list(top_ks),
    }


def optimal_semantic_matches(
    predicted_texts: Sequence[str], reference_texts: Sequence[str], embedder: Any
) -> list[dict[str, int | float]]:
    if not predicted_texts or not reference_texts:
        return []
    predicted = np.asarray(embedder.embed_documents(predicted_texts), dtype=float)
    reference = np.asarray(embedder.embed_documents(reference_texts), dtype=float)
    predicted /= np.maximum(np.linalg.norm(predicted, axis=1, keepdims=True), 1e-12)
    reference /= np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    similarities = predicted @ reference.T
    predicted_indices, reference_indices = linear_sum_assignment(-similarities)
    return [
        {
            "predicted_index": int(p_index),
            "reference_index": int(r_index),
            "similarity": float(similarities[p_index, r_index]),
        }
        for p_index, r_index in zip(predicted_indices, reference_indices, strict=True)
    ]


def add_semantic_diagnostics(
    result: dict[str, Any],
    cases: Sequence[Phase1Case],
    taxonomy: Mapping[str, TaxonomyLabel],
    embedder: Any,
) -> None:
    semantic_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for case in cases:
        predicted_texts = [item.justification for item in case.predicted]
        reference_texts = [" ".join(item.evidence) for item in case.reference]
        for match in optimal_semantic_matches(predicted_texts, reference_texts, embedder):
            predicted = case.predicted[int(match["predicted_index"])]
            reference = case.reference[int(match["reference_index"])]
            semantic_rows.append(
                {
                    "vc_slug": case.vc_slug,
                    "episode_slug": case.episode_slug,
                    "predicted_label": predicted.label,
                    "reference_label": reference.label,
                    "same_label": predicted.label == reference.label,
                    "same_family": taxonomy[predicted.label].coarse_parent == taxonomy[reference.label].coarse_parent,
                    "similarity": match["similarity"],
                }
            )
        predicted_labels, reference_labels = _label_sets(case)
        false_items = [item for item in case.predicted if item.label in predicted_labels - reference_labels]
        missed_items = [item for item in case.reference if item.label in reference_labels - predicted_labels]
        for parent in sorted({taxonomy[item.label].coarse_parent for item in [*false_items, *missed_items]}):
            parent_pred = [item for item in false_items if taxonomy[item.label].coarse_parent == parent]
            parent_ref = [item for item in missed_items if taxonomy[item.label].coarse_parent == parent]
            matches = optimal_semantic_matches(
                [item.justification for item in parent_pred],
                [" ".join(item.evidence) for item in parent_ref],
                embedder,
            )
            for match in matches:
                pred = parent_pred[int(match["predicted_index"])]
                ref = parent_ref[int(match["reference_index"])]
                confusion_rows.append(
                    {
                        "vc_slug": case.vc_slug,
                        "episode_slug": case.episode_slug,
                        "coarse_parent": parent,
                        "predicted_label": pred.label,
                        "missed_label": ref.label,
                        "similarity": match["similarity"],
                    }
                )
    result["semantic_rows"] = semantic_rows
    result["confusion_rows"] = confusion_rows
    result["semantic_summary"] = {
        "matched_pairs": len(semantic_rows),
        "mean_similarity": fmean(float(row["similarity"]) for row in semantic_rows) if semantic_rows else None,
        "same_label_mean_similarity": fmean(
            float(row["similarity"]) for row in semantic_rows if row["same_label"]
        ) if any(row["same_label"] for row in semantic_rows) else None,
        "cross_label_mean_similarity": fmean(
            float(row["similarity"]) for row in semantic_rows if not row["same_label"]
        ) if any(not row["same_label"] for row in semantic_rows) else None,
        "embedding": dict(embedder.metadata),
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_evaluation_markdown(result: Mapping[str, Any]) -> str:
    overall = result["overall"]
    ranking = result["ranking_summary"]
    lines = [
        "# Phase 1 Rationale Evaluation",
        "",
        "> **Scientific status:** the transcript-derived references are automated candidate ground truth, not final human-validated annotations. The results below are scaled automated-evaluation results and must be interpreted accordingly.",
        "",
        "## Population",
        "",
        f"The evaluation covers **{result['case_count']} investor–pitch cases** across **{result['vc_count']} VCs**: **{result['rich_case_count']} rich transcript-observed references** and **{result['legacy_case_count']} legacy references**.",
        "",
        "## Headline exact-label recovery",
        "",
        f"Across all cases, Phase 1 recovered **{overall['micro_tp']}** reference rationale labels, emitted **{overall['micro_fp']}** false labels, and missed **{overall['micro_fn']}** labels. Micro precision was **{_fmt(overall['micro_precision'])}**, micro recall **{_fmt(overall['micro_recall'])}**, and micro F1 **{_fmt(overall['micro_f1'])}**. Episode-macro Jaccard was **{_fmt(overall['macro_jaccard'])}**.",
        "",
        f"The average Phase 1 output contained **{_fmt(overall['average_predicted_size'], 2)}** unique labels, versus **{_fmt(overall['average_reference_size'], 2)}** in the reference. At least one reference rationale was recovered in **{_fmt(overall['at_least_one_correct_rate'])}** of cases; exact set match occurred in **{_fmt(overall['exact_set_match_rate'])}**.",
        "",
        "| Scope | N | Micro P | Micro R | Micro F1 | Macro F1 | Jaccard | Family F1 | Avg predicted | Avg reference |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["scope_rows"]:
        if row["dimension"] not in {"overall", "vc"}:
            continue
        lines.append(
            f"| {row['value']} | {row['n']} | {_fmt(row['micro_precision'])} | {_fmt(row['micro_recall'])} | {_fmt(row['micro_f1'])} | {_fmt(row['macro_f1'])} | {_fmt(row['macro_jaccard'])} | {_fmt(row['family_micro_f1'])} | {_fmt(row['average_predicted_size'], 2)} | {_fmt(row['average_reference_size'], 2)} |"
        )
    lines.extend(
        [
            "",
            "### Reference-source sensitivity",
            "",
            "| Source tier | N | Micro P | Micro R | Micro F1 | Avg predicted | Avg reference |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["scope_rows"]:
        if row["dimension"] == "source_tier":
            lines.append(
                f"| {row['value']} | {row['n']} | {_fmt(row['micro_precision'])} | {_fmt(row['micro_recall'])} | {_fmt(row['micro_f1'])} | {_fmt(row['average_predicted_size'], 2)} | {_fmt(row['average_reference_size'], 2)} |"
            )
    lines.extend(
        [
            "",
            "## Salience- and confidence-aware recovery",
            "",
            "Exact taxonomy-label recovery remains the primary evaluation. The weighted measures below are companion diagnostics: primary rationales use a fixed weight of 2 and secondary rationales a weight of 1; matched mass is capped by the smaller predicted/reference mass. Prediction and reference confidence may not be calibrated on the same scale. The combined salience-confidence measure is a diagnostic composite, not a replacement headline score.",
            "",
            "### Salience strata",
            "",
            "| Scope | Salience | Reference | Recovered | Recall | Predicted | Correct | Precision | Mean confidence when recovered |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["salience_strata_rows"]:
        lines.append(
            f"| {row['scope']} | {row['salience']} | {row['reference_count']} | {row['recovered_count']} | {_fmt(row['recall'])} | {row['predicted_count']} | {row['correct_count']} | {_fmt(row['precision'])} | {_fmt(row['mean_recovered_predicted_confidence'])} |"
        )
    lines.extend(
        [
            "",
            "### Weighted exact-label recovery",
            "",
            "| Scope | Weighting | Precision | Recall | F1 | Matched mass | Predicted mass | Reference mass |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["weighted_recovery_rows"]:
        lines.append(
            f"| {row['scope']} | {row['mode']} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['f1'])} | {_fmt(row['matched_mass'], 2)} | {_fmt(row['predicted_mass'], 2)} | {_fmt(row['reference_mass'], 2)} |"
        )
    lines.extend(
        [
            "",
            "### Confidence-threshold sensitivity",
            "",
            "No threshold is selected as best without a separate calibration or held-out procedure.",
            "",
            "| Threshold | Precision | Recall | F1 | Avg retained | Cases with a correct retained label |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["confidence_threshold_rows"]:
        if row["scope"] != "overall":
            continue
        lines.append(
            f"| {_fmt(row['threshold'], 2)} | {_fmt(row['micro_precision'])} | {_fmt(row['micro_recall'])} | {_fmt(row['micro_f1'])} | {_fmt(row['average_retained_count'], 2)} | {_fmt(row['at_least_one_correct_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Ranking and confidence",
            "",
            f"Across the 44-label candidate universe, label-ranking average precision was **{_fmt(ranking['average_precision'])}** against prevalence **{_fmt(ranking['prevalence'])}**; ROC AUC was **{_fmt(ranking['roc_auc'])}**. Confidence Brier score was **{_fmt(ranking['brier'])}** and fixed-bin ECE **{_fmt(ranking['ece'])}**.",
            "",
            "| Scope | k | Precision@k | Recall@k | Primary recall@k |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["ranking_rows"]:
        lines.append(
            f"| {row['scope']} | {row['k']} | {_fmt(row['precision_at_k'])} | {_fmt(row['recall_at_k'])} | {_fmt(row['primary_recall_at_k'])} |"
        )
    lines.extend(
        [
            "",
            "## Direction, salience, and hierarchy",
            "",
            "| Representation | Micro P | Micro R | Micro F1 | Conditional agreement |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in result["attribute_metrics"].items():
        lines.append(
            f"| {name} | {_fmt(row['micro_precision'])} | {_fmt(row['micro_recall'])} | {_fmt(row['micro_f1'])} | {_fmt(row.get('conditional_agreement'))} |"
        )
    lines.extend(
        [
            "",
            "## Rich-reference diagnostics",
            "",
            "These are reference-stratified recovery rates. Phase 1 does not predict activation, utterance type, or decision linkage as attributes.",
            "",
            "| Reference subset | Cases containing subset | Reference rationales | Recovered | Recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["rich_subset_rows"]:
        lines.append(
            f"| {row['subset']} | {row['cases_with_reference']} | {row['reference_rationales']} | {row['recovered_rationales']} | {_fmt(row['recall'])} |"
        )
    missed = sorted(result["per_label_rows"], key=lambda row: (-row["fn"], row["label"]))[:10]
    false = sorted(result["per_label_rows"], key=lambda row: (-row["fp"], row["label"]))[:10]
    lines.extend(
        [
            "",
            "## Most frequent errors",
            "",
            "| Most missed label | Missed | Support | Recall | Most false label | False | Precision |",
            "|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for left, right in zip(missed, false, strict=True):
        lines.append(
            f"| {left['label']} | {left['fn']} | {left['support']} | {_fmt(left['recall'])} | {right['label']} | {right['fp']} | {_fmt(right['precision'])} |"
        )
    semantic = result.get("semantic_summary")
    if semantic:
        lines.extend(
            [
                "",
                "## Semantic diagnostic",
                "",
                f"Optimal one-to-one matching produced **{semantic['matched_pairs']}** pairs with mean cosine similarity **{_fmt(semantic.get('mean_similarity'))}**. Same-label mean similarity was **{_fmt(semantic.get('same_label_mean_similarity'))}** and cross-label mean similarity **{_fmt(semantic.get('cross_label_mean_similarity'))}**. These values are diagnostic and do not alter exact-label scores.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This evaluation measures whether Phase 1 recovers taxonomy-coded considerations expressed by the investor in the full episode transcript. It does not evaluate Phase 2 decisions, and it does not imply that the automated references are error-free. Human validation remains the final annotation step.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_outputs(
    result: dict[str, Any],
    cases: Sequence[Phase1Case],
    taxonomy: Mapping[str, TaxonomyLabel],
    output_dir: Path,
    *,
    input_manifest: Mapping[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_rows = [
        {"scope": "pooled", **result["overall"]},
        {"scope": "macro_vc", **result["macro_vc"]},
    ]
    vc_rows = [row for row in result["scope_rows"] if row["dimension"] == "vc"]
    decision_rows = [row for row in result["scope_rows"] if row["dimension"] == "decision"]
    source_tier_rows = [row for row in result["scope_rows"] if row["dimension"] == "source_tier"]
    ranking_by_scope = {row["scope"]: row for row in result["ranking_summaries"]}
    ranking_rows = [
        {**ranking_by_scope[row["scope"]], **row} for row in result["ranking_rows"]
    ]
    attribute_rows = [{"representation": name, **row} for name, row in result["attribute_metrics"].items()]
    files_and_rows: dict[str, Sequence[Mapping[str, Any]]] = {
        "overall_metrics.csv": overall_rows,
        "vc_metrics.csv": vc_rows,
        "decision_split_metrics.csv": decision_rows,
        "source_tier_metrics.csv": source_tier_rows,
        "ranking_metrics.csv": ranking_rows,
        "attribute_metrics.csv": attribute_rows,
        "rich_subset_metrics.csv": result["rich_subset_rows"],
        "salience_strata_metrics.csv": result["salience_strata_rows"],
        "weighted_recovery_metrics.csv": result["weighted_recovery_rows"],
        "confidence_threshold_metrics.csv": result["confidence_threshold_rows"],
        "per_label_metrics.csv": result["per_label_rows"],
        "family_metrics.csv": result["family_rows"],
        "episode_metrics.csv": result["episode_rows"],
        "episode_label_comparisons.csv": result["comparison_rows"],
        "confusion_pairs.csv": result.get("confusion_rows", []),
        "semantic_matches.csv": result.get("semantic_rows", []),
    }
    paths: dict[str, Path] = {}
    for filename, rows in files_and_rows.items():
        path = output_dir / filename
        _write_csv(path, rows)
        paths[filename] = path
    markdown_path = output_dir / "evaluation.md"
    markdown_path.write_text(render_evaluation_markdown(result), encoding="utf-8")
    paths[markdown_path.name] = markdown_path
    manifest = {
        "schema": "phase1-rationale-evaluation-v1",
        "scientific_status": "automated_candidate_ground_truth_not_human_validated",
        "case_count": result["case_count"],
        "rich_case_count": result["rich_case_count"],
        "legacy_case_count": result["legacy_case_count"],
        "vc_count": result["vc_count"],
        "taxonomy_label_count": len(taxonomy),
        "top_ks": result["top_ks"],
        "semantic": result.get("semantic_summary"),
        "importance_evaluation": {
            "salience_weights": result["salience_weights"],
            "confidence_thresholds": result["confidence_thresholds"],
            "matching_rule": "minimum_mass_on_exact_label_match",
            "modes": {
                "salience": "fixed_salience_weight",
                "confidence": "rationale_confidence",
                "combined": "rationale_confidence_times_salience_weight",
            },
            "empty_mass_rules": {
                "precision": "one_only_when_both_predicted_and_reference_mass_are_zero",
                "recall": "one_when_reference_mass_is_zero",
            },
            "headline_status": "diagnostic_companion_exact_labels_remain_primary",
            "outputs": [
                "salience_strata_metrics.csv",
                "weighted_recovery_metrics.csv",
                "confidence_threshold_metrics.csv",
            ],
        },
        "inputs": dict(input_manifest),
        "outputs": {name: sha256_file(path) for name, path in paths.items()},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths[manifest_path.name] = manifest_path
    return paths
