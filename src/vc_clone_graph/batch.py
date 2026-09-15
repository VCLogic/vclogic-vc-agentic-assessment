"""Small, deterministic helpers for sequential episode batches."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import re
from typing import Any

from .config import RunConfig


_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _episode_sort_key(slug: str) -> tuple[int, int | str, str]:
    prefix = slug.split("-", 1)[0]
    if prefix.isdigit():
        return (0, int(prefix), slug)
    return (1, prefix, slug)


def round_robin_order(rows: tuple[tuple[str, str], ...] | list[tuple[str, str]]) -> tuple[str, ...]:
    """Alternate observed Ins and Outs, then append the remaining class."""
    seen: set[str] = set()
    by_decision: dict[str, list[str]] = {"In": [], "Out": []}
    for slug, decision in rows:
        if slug in seen:
            raise ValueError(f"duplicate episode slug: {slug}")
        seen.add(slug)
        if decision not in by_decision:
            raise ValueError(f"episode {slug} is not an observed In or Out")
        by_decision[decision].append(slug)

    ins = sorted(by_decision["In"], key=_episode_sort_key)
    outs = sorted(by_decision["Out"], key=_episode_sort_key)
    ordered: list[str] = []
    for position in range(max(len(ins), len(outs))):
        if position < len(ins):
            ordered.append(ins[position])
        if position < len(outs):
            ordered.append(outs[position])
    return tuple(ordered)


def discover_observed_pitch_rows(
    pitch_dir: Path,
    record_dir: Path,
) -> tuple[tuple[str, str], ...]:
    """Join runnable pitch files to their target-excluded audited decisions."""
    rows: list[tuple[str, str]] = []
    for pitch_path in sorted(Path(pitch_dir).glob("*.txt")):
        slug = pitch_path.stem
        record_path = Path(record_dir) / f"{slug}.json"
        if not record_path.is_file():
            raise ValueError(f"missing audited precedent record for {slug}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("episode_slug") != slug:
            raise ValueError(f"audited precedent record slug mismatch for {slug}")
        decision = record.get("decision")
        status = decision.get("status") if type(decision) is dict else None
        if status not in {"In", "Out"}:
            raise ValueError(f"pitch {slug} is not an observed In or Out")
        rows.append((slug, status))
    if not rows:
        raise ValueError("no observed pitch files found")
    return tuple(sorted(rows, key=lambda row: _episode_sort_key(row[0])))


def load_cohort(path: Path) -> dict[str, Any]:
    """Load a label-free execution cohort and its documented exclusions."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != "vc-clone-batch-v1":
        raise ValueError("unsupported batch cohort schema")
    episodes = raw.get("episodes")
    excluded = raw.get("excluded", [])
    if (
        type(episodes) is not list
        or not episodes
        or any(type(slug) is not str or not _SLUG.fullmatch(slug) for slug in episodes)
        or len(set(episodes)) != len(episodes)
    ):
        raise ValueError("batch episodes are invalid")
    if type(excluded) is not list or any(type(row) is not dict for row in excluded):
        raise ValueError("batch exclusions are invalid")
    return {"episodes": tuple(episodes), "excluded": tuple(excluded)}


def scoped_config(
    base: RunConfig,
    episode_slug: str,
    output_root: str,
    *,
    attempt: int,
) -> RunConfig:
    """Return an immutable config scoped to one fresh episode attempt."""
    if attempt < 1:
        raise ValueError("attempt must be positive")
    attempt_root = f"{output_root}/attempt-{attempt}"
    run = base.run.model_copy(
        update={
            "episode_slug": episode_slug,
            "output_root": attempt_root,
            "checkpoint_path": (
                f"{attempt_root}/checkpoints/{episode_slug}.sqlite"
            ),
        }
    )
    return base.model_copy(update={"run": run})


def _has_decision(summary: dict[str, Any]) -> bool:
    decision = summary.get("decision")
    if type(decision) is dict and decision.get("decision") in {"In", "Out"}:
        return True
    return (
        summary.get("phase1_status") in {"accepted", "provisional"}
        and summary.get("phase2_status") == "not_run"
    )


def run_with_retries(
    attempt: Callable[[int], dict[str, Any]],
    *,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Run until a frozen full decision or Phase-1-only artifact exists."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    summaries: list[dict[str, Any]] = []
    for number in range(1, max_attempts + 1):
        summary = attempt(number)
        summaries.append(summary)
        if _has_decision(summary):
            return {
                "status": "completed",
                "selected_attempt": number,
                "summary": summary,
                "attempt_summaries": summaries,
            }
    return {
        "status": "failed",
        "selected_attempt": None,
        "summary": summaries[-1],
        "attempt_summaries": summaries,
    }
