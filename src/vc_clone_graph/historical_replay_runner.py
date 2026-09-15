"""Load leakage-safe historical episodes for founder-rehearsal replay."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .precedents import PrecedentEpisode
from .rehearsal_canary import HistoricalRehearsalCanary, build_canary
from .rehearsal_replay import PanelFounderStatement


@dataclass(frozen=True)
class HistoricalReplayCase:
    episode: PrecedentEpisode
    canary: HistoricalRehearsalCanary
    target_company_aliases: tuple[str, ...]
    observed_rationales: tuple[dict[str, Any], ...]
    panel_founder_statements: tuple[PanelFounderStatement, ...]


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _speaker_aliases(rows: list[dict[str, Any]]) -> set[str]:
    aliases: set[str] = set()
    for row in rows:
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        aliases.add(name)
        aliases.add(name.split()[0])
    return aliases


def _ground_truth_path(
    evaluation_root: Path, vc_slug: str, vc_name: str, episode_slug: str
) -> Path:
    candidates = sorted(
        (evaluation_root / "phase1_ground_truth_rationales" / "records").glob(
            f"*/{episode_slug}.json"
        )
    )
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("vc_slug") == vc_slug or payload.get("vc_name") == vc_name:
            return path
    raise ValueError(f"ground-truth rationale record is missing: {vc_slug}/{episode_slug}")


def _question_label_map(
    episode: PrecedentEpisode, rationales: list[dict[str, Any]]
) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    question_rows = [
        row for row in rationales if row.get("utterance_type") == "question"
    ]
    for turn in episode.turns:
        if "?" not in turn.text:
            continue
        normalized_turn = _normalized(turn.text)
        for row in question_rows:
            label = row.get("rationale_label")
            if not isinstance(label, str) or not label:
                continue
            for evidence in row.get("evidence", []):
                normalized_evidence = _normalized(str(evidence))
                if normalized_evidence and (
                    normalized_evidence in normalized_turn
                    or normalized_turn in normalized_evidence
                ):
                    result.setdefault(turn.turn_index, set()).add(label)
                    break
    return result


def load_historical_case(
    *,
    input_root: Path,
    evaluation_root: Path,
    vc_slug: str,
    vc_name: str,
    episode_slug: str,
) -> HistoricalReplayCase:
    """Load one audited pitch, observed Q&A, and decision-hidden rationale reference."""
    base = Path(input_root) / "data" / "investors" / vc_slug
    episode = PrecedentEpisode.model_validate_json(
        (base / "precedents" / "records" / f"{episode_slug}.json").read_text(
            encoding="utf-8"
        )
    )
    pitch = (base / "pitches" / f"{episode_slug}.txt").read_text(encoding="utf-8")
    source = json.loads(
        (base / "precedents" / "sources" / f"{episode_slug}.json").read_text(
            encoding="utf-8"
        )
    )
    audit = json.loads(
        (base / "audits" / f"{episode_slug}.json").read_text(encoding="utf-8")
    )
    if audit.get("status") != "audited":
        raise ValueError(f"pitch is not audited: {episode_slug}")
    gt = json.loads(
        _ground_truth_path(Path(evaluation_root), vc_slug, vc_name, episode_slug).read_text(
            encoding="utf-8"
        )
    )
    rationales = list(gt.get("rationales", []))
    founder_speakers = _speaker_aliases(list(source.get("founders", [])))
    canary = build_canary(
        episode,
        target_vc=vc_name,
        pitch_text=pitch,
        founder_speakers=founder_speakers,
        investor_speakers=_speaker_aliases(list(source.get("panel", []))),
        observed_question_labels=_question_label_map(episode, rationales),
    )
    decision_boundary = min(
        evidence.turn_start for evidence in episode.decision.evidence
    )
    normalized_founders = {_normalized(value) for value in founder_speakers}
    panel_founder_statements = tuple(
        PanelFounderStatement(
            turn_index=turn.turn_index,
            speaker=turn.speaker,
            text=turn.text,
        )
        for turn in episode.turns
        if turn.turn_index < decision_boundary
        and _normalized(turn.speaker) in normalized_founders
        and turn.text.strip()
    )
    return HistoricalReplayCase(
        episode=episode,
        canary=canary,
        target_company_aliases=tuple(
            str(value) for value in audit.get("target_company_aliases", [])
        ),
        observed_rationales=tuple(rationales),
        panel_founder_statements=panel_founder_statements,
    )
