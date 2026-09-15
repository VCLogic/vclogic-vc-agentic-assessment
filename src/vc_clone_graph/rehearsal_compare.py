"""Deterministic comparison of independent completed rehearsal sessions."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .rehearsal_artifacts import RehearsalArtifactStore
from .rehearsal_schemas import FounderReport


class ComparisonRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    vc_slug: str
    session_id: str
    questions: tuple[str, ...]
    rationale_labels: tuple[str, ...]
    decision: Literal["In", "Out"]
    investment_likelihood: float = Field(ge=0, le=1)
    decision_confidence: float = Field(ge=0, le=1)
    unresolved_uncertainties: tuple[str, ...]
    pitch_improvement_suggestions: tuple[str, ...]


class CrossVCComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pitch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shared_state_used: bool = False
    rows: tuple[ComparisonRow, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def require_independent_distinct_vcs(self) -> "CrossVCComparison":
        if self.shared_state_used:
            raise ValueError("cross-VC comparison must not use shared evolving state")
        if len({row.vc_slug for row in self.rows}) != len(self.rows):
            raise ValueError("comparison requires distinct VC sessions")
        return self


def compare_sessions(paths: Sequence[Path]) -> CrossVCComparison:
    if len(paths) < 2:
        raise ValueError("comparison requires at least two completed sessions")
    rows: list[ComparisonRow] = []
    pitch_digests: set[str] = set()
    for path in paths:
        store = RehearsalArtifactStore.open(path)
        verified = store.verify()
        pitch_digests.add(str(verified["pitch_sha256"]))
        state = store.read_json("state.json")
        if state.get("status") != "complete":
            raise ValueError(f"rehearsal session is not complete: {path}")
        report = FounderReport.model_validate(store.read_json("founder-report.json"))
        final = report.final_assessment
        rows.append(
            ComparisonRow(
                vc_slug=report.vc_slug,
                session_id=report.session_id,
                questions=tuple(
                    str(answer["question"])
                    for answer in state.get("answers", [])
                    if isinstance(answer, dict) and answer.get("question")
                ),
                rationale_labels=tuple(
                    dict.fromkeys(
                        rationale.taxonomy_label for rationale in final.rationale_state
                    )
                ),
                decision=final.decision,
                investment_likelihood=final.investment_likelihood,
                decision_confidence=final.decision_confidence,
                unresolved_uncertainties=final.unresolved_uncertainties,
                pitch_improvement_suggestions=report.pitch_improvement_suggestions,
            )
        )
    if len(pitch_digests) != 1:
        raise ValueError("cross-VC comparison requires the same pitch")
    ordered = tuple(sorted(rows, key=lambda row: row.vc_slug))
    return CrossVCComparison(
        pitch_sha256=next(iter(pitch_digests)),
        shared_state_used=False,
        rows=ordered,
    )


def comparison_csv(comparison: CrossVCComparison) -> str:
    stream = StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "vc_slug",
            "decision",
            "investment_likelihood",
            "decision_confidence",
            "questions",
            "rationale_labels",
            "unresolved_uncertainties",
            "pitch_improvement_suggestions",
        )
    )
    for row in comparison.rows:
        writer.writerow(
            (
                row.vc_slug,
                row.decision,
                row.investment_likelihood,
                row.decision_confidence,
                " | ".join(row.questions),
                " | ".join(row.rationale_labels),
                " | ".join(row.unresolved_uncertainties),
                " | ".join(row.pitch_improvement_suggestions),
            )
        )
    return stream.getvalue()
