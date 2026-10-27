"""Uniqueness page: what sets one investor apart from the other investors.

The Investment Memory records what an investor values, much of which every
investor says. The page is written once per investor from their memory and
from statistics over all investors' memories, so that it states only the
differences: what they stress more or less than others, where they are harsher
or more lenient, their hard limits, and what they rarely weigh.

For evaluation the page is written per pitch, from the memory filtered for that
pitch exactly as the pipeline filters it (sections mentioning the pitched
company are excluded), and the page is checked for the company's names.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Sequence

from ..providers.base import GenerationProvider, GenerationRequest
from .config import LensSettings
from .lens import MEMORY_SOURCES, _EXCLUDED_TEXT, _memory_index, _run_package
from .run_inputs import RunArtifacts

SCHEMA_VERSION = "uniqueness-page-v1"
KINDS = ("raises_more", "raises_less", "harsher", "more_lenient", "hard_limit", "rarely_weighs")
_LABEL = re.compile(r"label=(\S+) direction=(\S+)")


def wiki_statistics(wiki_root: Path) -> dict[str, dict[str, Any]]:
    """Per investor: share of evidence items per label and share positive per label."""
    stats = {}
    for evidence in sorted(Path(wiki_root).glob("*/evidence")):
        labels: Counter[str] = Counter()
        positive: Counter[str] = Counter()
        for path in evidence.glob("*.md"):
            for label, direction in _LABEL.findall(path.read_text(encoding="utf-8")):
                labels[label] += 1
                positive[label] += direction == "positive"
        stats[evidence.parent.name] = {"labels": labels, "positive": positive, "total": sum(labels.values())}
    return stats


def contrast_table(stats: dict[str, dict[str, Any]], vc: str, top: int = 12) -> list[dict[str, Any]]:
    """Labels where this investor's evidence share or positivity differs most from all others."""
    others = [s for name, s in stats.items() if name != vc]
    own = stats[vc]
    rows = []
    for label in set(own["labels"]) | {l for s in others for l in s["labels"]}:
        share = own["labels"][label] / max(1, own["total"])
        base = sum(s["labels"][label] / max(1, s["total"]) for s in others) / len(others)
        pos = own["positive"][label] / own["labels"][label] if own["labels"][label] else None
        n_other = sum(s["labels"][label] for s in others)
        base_pos = sum(s["positive"][label] for s in others) / n_other if n_other else None
        rows.append({"taxonomy_label": label, "investor_share": round(share, 3), "others_share": round(base, 3),
                     "ratio": round((share + 0.002) / (base + 0.002), 2),
                     "investor_positive": None if pos is None else round(pos, 2),
                     "others_positive": None if base_pos is None else round(base_pos, 2)})
    rows.sort(key=lambda r: -r["ratio"])
    low = [r for r in rows if r["others_share"] >= 0.01][-top // 2:]
    return rows[:top] + [r for r in low if r not in rows[:top]]


def pitched_aliases(aliases_path: Path, episodes: Sequence[str]) -> list[str]:
    data = json.loads(Path(aliases_path).read_text(encoding="utf-8"))["episodes"]
    return sorted({a for e in episodes for a in data.get(e, {}).get("aliases", []) if a.strip()})


def filtered_memory(run: RunArtifacts, aliases: Sequence[str], max_chars: int, workspace: Path = Path(".")) -> list[dict[str, Any]]:
    """The investor's memory with every section mentioning a pitched company excluded."""
    root, package = _run_package(run, workspace)
    canonical, _ = _memory_index(run, root, package)
    index, _ = canonical.for_pitch(list(aliases))
    chosen, used = [], 0
    for chunk in sorted(index.chunks, key=lambda c: (MEMORY_SOURCES.index(c.source_path)
                                                     if c.source_path in MEMORY_SOURCES else 9, c.source_path)):
        if not (chunk.source_path in MEMORY_SOURCES or chunk.source_path.startswith("evidence/")):
            continue
        if chunk.text.strip() == _EXCLUDED_TEXT or used + len(chunk.text) > max_chars:
            continue
        used += len(chunk.text)
        chosen.append({"memory_id": chunk.chunk_id, "source": chunk.source_path, "heading": chunk.heading,
                       "text": chunk.text})
    return chosen


def build_prompt(investor: str, table: Sequence[dict[str, Any]], memory: Sequence[dict[str, Any]], n_investors: int) -> str:
    rows = "\n".join(
        f"- {r['taxonomy_label']}: {r['investor_share']:.1%} of this investor's evidence vs {r['others_share']:.1%} "
        f"for the other investors (x{r['ratio']}); positive {r['investor_positive']} vs {r['others_positive']}"
        for r in table)
    mem = "\n\n".join(f"[{m['memory_id']}] {m['heading']} ({m['source']})\n{m['text']}" for m in memory)
    return f"""You are writing a short page on what sets one venture investor apart from {n_investors - 1} other investors.

Investor: {investor}

Do NOT describe what most investors value (strong founders, large markets, traction) unless this investor weighs it clearly more or less than the others. The page is used to tell this investor apart from the others on the same pitch.

How this investor's public statements differ from the other investors' (share of evidence on each rationale, and share positive):
{rows}

Write 6 to 12 items. Each item:
- kind: raises_more (raises it more than others), raises_less, harsher (more often negative or demanding on it), more_lenient, hard_limit (stage, check size, sector, geography, structure or conflict they will not do), rarely_weighs;
- taxonomy_labels: the related rationale labels from the table or memory;
- statement: one sentence on how this investor differs, specific enough to change an assessment;
- memory_ids: the memory excerpts that support it.
Do not name any company, fund portfolio company or person other than the investor.

Investment Memory excerpts:
{mem}"""


SCHEMA = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": list(KINDS)},
            "taxonomy_labels": {"type": "array", "items": {"type": "string"}},
            "statement": {"type": "string"},
            "memory_ids": {"type": "array", "items": {"type": "string"}},
        }}},
    },
}


def build_page(
    run: RunArtifacts,
    settings: LensSettings,
    provider: GenerationProvider,
    *,
    stats: dict[str, dict[str, Any]],
    aliases: Sequence[str],
    workspace: Path = Path("."),
) -> dict[str, Any]:
    table = contrast_table(stats, run.vc_slug)
    memory = filtered_memory(run, aliases, settings.max_memory_chars, workspace)
    prompt = build_prompt(run.investor_name, table, memory, len(stats))
    result = provider.generate(GenerationRequest(
        phase="uniqueness_page", prompt=prompt, schema=SCHEMA,
        max_output_tokens=settings.max_output_tokens, reasoning_effort=settings.reasoning_effort))
    labels = {r["label"] for r in run.taxonomy}
    ids = {m["memory_id"] for m in memory}
    items = []
    for item in (result.parsed or {}).get("items") or []:
        if item.get("kind") not in KINDS or not str(item.get("statement", "")).strip():
            continue
        items.append({
            "page_id": f"U{len(items) + 1}",
            "kind": item["kind"],
            "taxonomy_labels": [l for l in item.get("taxonomy_labels") or [] if l in labels],
            "statement": item["statement"].strip(),
            "memory_ids": [i for i in item.get("memory_ids") or [] if i in ids],
        })
    text = json.dumps(items).casefold()
    leaked = [a for a in aliases if len(a) >= 3 and re.search(rf"\b{re.escape(a.casefold())}\b", text)]
    return {
        "schema": SCHEMA_VERSION,
        "vc_slug": run.vc_slug,
        "investor_name": run.investor_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.model,
        "excluded_company_aliases": len(aliases),
        "memory_ids": sorted(ids),
        "contrast_table": table,
        "items": items,
        "leaked_aliases": leaked,
        "usage": result.usage.model_dump(),
    }


def render(page: dict[str, Any]) -> str:
    groups: dict[str, list[str]] = defaultdict(list)
    for item in page["items"]:
        groups[item["kind"]].append(f"- [{item['page_id']}] {item['statement']} ({', '.join(item['taxonomy_labels'])})")
    return f"# What sets {page['investor_name']} apart\n\n" + "\n\n".join(
        f"## {kind.replace('_', ' ')}\n" + "\n".join(lines) for kind, lines in groups.items())


def page_for_pitch(page: dict[str, Any], aliases: Sequence[str]) -> list[dict[str, Any]]:
    """Page items with any item naming this pitch's company removed."""
    names = [a.casefold() for a in aliases if len(a) >= 3]
    return [i for i in page["items"] if not any(re.search(rf"\b{re.escape(n)}\b", i["statement"].casefold()) for n in names)]
