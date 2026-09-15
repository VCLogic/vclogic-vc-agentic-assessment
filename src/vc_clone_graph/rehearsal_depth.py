"""Stable founder-selectable rehearsal depth limits."""

from typing import Literal


RehearsalDepth = Literal["quick", "standard", "deep"]

DEPTH_LIMITS: dict[RehearsalDepth, int] = {
    "quick": 3,
    "standard": 5,
    "deep": 8,
}


def effective_question_limit(
    depth: RehearsalDepth, server_ceiling: int
) -> int:
    """Resolve a founder choice without exceeding the configured safety ceiling."""
    if not 1 <= server_ceiling <= 8:
        raise ValueError("server_ceiling must be between 1 and 8")
    return min(DEPTH_LIMITS[depth], server_ceiling)
