"""An investor's distinctive rationales, from annotations of their other episodes.

For each rationale the investor raised in at least ``min_episodes`` other
episodes, compare the share of their episodes in which they raised it with the
share for all investors (the current episode is excluded from both). The
rationales with the highest ratio are what sets this investor apart in
practice, which the Investment Memory does not capture.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

SCOPES = frozenset({"original_pitch", "post_founder_debrief"})


def _mentions(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))["rationale_mentions"]
    return [m for m in rows if m["scope"] in SCOPES and m["direction"] != "unclear"]


def build_profile(
    annotation_root: Path,
    vc_slug: str,
    exclude_episode: str,
    *,
    min_episodes: int = 2,
    top: int = 10,
    quotes: int = 2,
) -> list[dict[str, Any]]:
    root = Path(annotation_root)
    own: Counter[str] = Counter()
    everyone: Counter[str] = Counter()
    directions: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)
    own_episodes = all_episodes = 0
    for path in sorted(root.glob("*/*.json")):
        episode = path.stem
        if episode == exclude_episode:
            continue
        mentions = _mentions(path)
        labels = {m["rationale_label"] for m in mentions}
        everyone.update(labels)
        all_episodes += 1
        if path.parent.name != vc_slug:
            continue
        own.update(labels)
        own_episodes += 1
        for m in mentions:
            directions[m["rationale_label"]][m["direction"]] += 1
            if len(examples[m["rationale_label"]]) < quotes:
                examples[m["rationale_label"]].append(f"{episode}: \"{m['quote'][:200]}\"")
    if not own_episodes:
        return []
    rows = []
    for label, count in own.items():
        if count < min_episodes:
            continue
        rate = count / own_episodes
        base = everyone[label] / all_episodes
        rows.append({
            "taxonomy_label": label,
            "investor_rate": round(rate, 2),
            "all_investors_rate": round(base, 2),
            "ratio": round(rate / base, 2) if base else None,
            "usual_direction": directions[label].most_common(1)[0][0],
            "examples": examples[label],
        })
    rows.sort(key=lambda r: -(r["ratio"] or 0))
    for number, row in enumerate(rows[:top], start=1):
        row["profile_id"] = f"V{number}"
    return [{**row, "episodes": own_episodes} for row in rows[:top]]
