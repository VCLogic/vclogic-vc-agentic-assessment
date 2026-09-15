"""Monotonic cleanup helpers for already-packaged pitch-only transcripts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").split())


def remove_post_boundary_lines(
    pitch_text: str,
    turns: Sequence[Mapping[str, Any]],
    boundary_index: int | None,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Remove only existing lines proven to occur after a decision boundary.

    The operation is monotonic: it never introduces source text and preserves
    lines that cannot be mapped safely or that also occur before the boundary.
    """
    lookup: dict[str, list[int]] = {}
    for turn in turns:
        text = _normalized(turn.get("text"))
        if not text:
            continue
        speaker = _normalized(turn.get("speaker"))
        keys = {text}
        if speaker:
            keys.add(f"{speaker}: {text}")
        for key in keys:
            lookup.setdefault(key, []).append(int(turn["idx"]))

    kept: list[str] = []
    removed: list[dict[str, Any]] = []
    unmapped: list[str] = []
    for line in pitch_text.splitlines():
        normalized_line = _normalized(line)
        if not normalized_line:
            continue
        indices = lookup.get(normalized_line, [])
        if not indices:
            kept.append(line)
            unmapped.append(line)
            continue
        if boundary_index is not None and min(indices) >= boundary_index:
            removed.append({"turn_index": min(indices), "line": line})
            continue
        kept.append(line)

    return ("\n".join(kept) + ("\n" if kept else ""), removed, unmapped)
