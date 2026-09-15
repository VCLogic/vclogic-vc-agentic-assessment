"""Aligned, immutable inputs for offline Phase 1 rationale calibration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from .phase1_evaluation import Phase1Case, ReferenceRationale, TaxonomyLabel


_DIRECTION_SIGN = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
_SALIENCE_WEIGHT = {"secondary": 1.0, "primary": 2.0}


@dataclass(frozen=True)
class RawLabelSignal:
    predicted: bool
    confidence: float
    direction: str
    salience: str
    signed_salience: float


@dataclass(frozen=True)
class ReferenceAttribute:
    activation: str
    utterance_type: str
    decision_link: str
    salience: str


@dataclass(frozen=True)
class CalibrationCase:
    vc_slug: str
    vc_name: str
    episode_slug: str
    actual_decision: str
    source_tier: str
    source_format: str
    pitch_text: str
    pitch_path: Path
    pitch_sha256: str
    labels: tuple[str, ...]
    families: Mapping[str, str]
    raw_signals: Mapping[str, RawLabelSignal]
    reference_targets: Mapping[str, int]
    reference_attributes: Mapping[str, ReferenceAttribute]
    phase1_case: Phase1Case

    @property
    def key(self) -> tuple[str, str]:
        return self.vc_slug, self.episode_slug


def resolve_pitch_path(project_root: Path, vc_slug: str, episode_slug: str) -> Path:
    investor_root = project_root / "inputs" / "data" / "investors"
    matches = sorted(investor_root.glob(f"{vc_slug}*/pitches/{episode_slug}.txt"))
    if not matches:
        raise ValueError(f"missing audited pitch for {vc_slug}/{episode_slug}")
    if len(matches) != 1:
        raise ValueError(f"ambiguous audited pitch for {vc_slug}/{episode_slug}: {matches}")
    return matches[0]


def _reference_attribute(rows: Sequence[ReferenceRationale]) -> ReferenceAttribute:
    ordered = sorted(
        rows,
        key=lambda row: (
            row.decision_link == "explicit",
            row.utterance_type == "decision_reason",
            row.activation == "evaluated",
            row.salience == "primary",
            row.confidence,
        ),
        reverse=True,
    )
    selected = ordered[0]
    return ReferenceAttribute(
        activation=selected.activation,
        utterance_type=selected.utterance_type,
        decision_link=selected.decision_link,
        salience=selected.salience,
    )


def build_calibration_cases(
    project_root: Path,
    cases: Sequence[Phase1Case],
    taxonomy: Mapping[str, TaxonomyLabel],
) -> list[CalibrationCase]:
    labels = tuple(sorted(taxonomy))
    families = {label: taxonomy[label].coarse_parent for label in labels}
    result: list[CalibrationCase] = []
    for case in cases:
        pitch_path = resolve_pitch_path(project_root, case.vc_slug, case.episode_slug)
        pitch_text = pitch_path.read_text(encoding="utf-8").strip()
        if not pitch_text:
            raise ValueError(f"empty audited pitch: {pitch_path}")
        predicted_by_label = {}
        for row in case.predicted:
            if row.label not in predicted_by_label or row.confidence > predicted_by_label[row.label].confidence:
                predicted_by_label[row.label] = row
        raw_signals: dict[str, RawLabelSignal] = {}
        for label in labels:
            row = predicted_by_label.get(label)
            if row is None:
                raw_signals[label] = RawLabelSignal(False, 0.0, "absent", "absent", 0.0)
            else:
                raw_signals[label] = RawLabelSignal(
                    True,
                    row.confidence,
                    row.direction,
                    row.salience,
                    _DIRECTION_SIGN[row.direction] * _SALIENCE_WEIGHT[row.salience],
                )
        reference_by_label: dict[str, list[ReferenceRationale]] = {}
        for row in case.reference:
            reference_by_label.setdefault(row.label, []).append(row)
        result.append(
            CalibrationCase(
                vc_slug=case.vc_slug,
                vc_name=case.vc_name,
                episode_slug=case.episode_slug,
                actual_decision=case.actual_decision,
                source_tier=case.source_tier,
                source_format=case.source_format,
                pitch_text=pitch_text,
                pitch_path=pitch_path,
                pitch_sha256=hashlib.sha256(pitch_text.encode("utf-8")).hexdigest(),
                labels=labels,
                families=families,
                raw_signals=raw_signals,
                reference_targets={label: int(label in reference_by_label) for label in labels},
                reference_attributes={
                    label: _reference_attribute(rows)
                    for label, rows in reference_by_label.items()
                },
                phase1_case=case,
            )
        )
    keys = [row.key for row in result]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate calibration case key")
    return result
