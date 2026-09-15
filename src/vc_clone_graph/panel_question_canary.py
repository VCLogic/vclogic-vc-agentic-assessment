"""Deterministic selection for panel-question Phase 1 canaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CanaryCandidate:
    vc_slug: str
    registry_vc_slug: str
    episode_slug: str
    actual_decision: str
    insertion_count: int
    pitch_path: str
    audit_path: str
    pitch_sha256: str
    canonical_artifact_root: str


def select_panel_question_canaries(
    rows: Sequence[CanaryCandidate], *, per_label: int = 2
) -> tuple[CanaryCandidate, ...]:
    if per_label < 1:
        raise ValueError("per_label must be positive")
    result = []
    vcs = sorted({row.vc_slug for row in rows})
    for vc in vcs:
        for label in ("In", "Out"):
            eligible = sorted(
                (row for row in rows if row.vc_slug == vc
                 and row.actual_decision == label and row.insertion_count > 0),
                key=lambda row: (-row.insertion_count, row.episode_slug),
            )
            if len(eligible) < per_label:
                raise ValueError(
                    f"insufficient eligible panel-question cases for {vc}/{label}: "
                    f"{len(eligible)} < {per_label}"
                )
            result.extend(eligible[:per_label])
    return tuple(result)
