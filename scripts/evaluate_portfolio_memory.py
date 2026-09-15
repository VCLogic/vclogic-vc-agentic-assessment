#!/usr/bin/env python3
"""Evaluate temporal availability and retrieval of episode-derived portfolio memory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vc_clone_graph.portfolio_memory import PortfolioMemoryCorpus, PortfolioMemoryIndex
from vc_clone_graph.providers.sentence_transformers import SentenceTransformerEmbeddingProvider


DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
REFERENCE_SLUGS = {
    "charles-hudson-precursor-ventures": "charles-hudson",
    "cyan-banister-long-journey-ventures": "cyan-banister",
    "elizabeth-yin-hustle-fund": "elizabeth-yin",
    "jesse-middleton-flybridge": "jesse-middleton",
    "jillian-manus-structure-capital": "jillian-manus",
    "phil-nadel": "phil-nadel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("inputs"))
    parser.add_argument(
        "--references", type=Path,
        default=Path("evaluation/phase1_ground_truth_rationales/records"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, action="append", dest="top_ks")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-revision", default=DEFAULT_REVISION)
    parser.add_argument("--embedding-device", default="auto")
    return parser.parse_args()


def _normal(value: str) -> str:
    return " ".join(value.casefold().split())


def _conflict_reference_slugs(root: Path, short_slug: str) -> list[str]:
    result: list[str] = []
    for path in sorted((root / short_slug).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if any(
            row.get("rationale_label") == "portfolio_conflict_constraint"
            for row in payload.get("rationales", [])
        ):
            result.append(payload["episode_slug"])
    return result


def _availability(corpus: PortfolioMemoryCorpus, target_slug: str) -> tuple[str, set[str]]:
    current = [
        row for row in corpus.disclosures if row.source_episode_slug == target_slug
    ]
    if not current:
        return "unavailable", set()
    current_names = {
        _normal(row.company_name) for row in current if row.company_name is not None
    }
    prefix = target_slug.split("-", 1)[0]
    if not prefix.isdecimal():
        return "unavailable", current_names
    target_number = int(prefix)
    prior_names = {
        _normal(row.company_name)
        for row in corpus.disclosures
        if row.company_name is not None
        and row.source_episode_number is not None
        and row.source_episode_number < target_number
    }
    relevant = current_names & prior_names
    if relevant:
        return "prior_observable", relevant
    return "first_disclosed_current", current_names


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else ["vc_slug", "episode_slug"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    top_ks = sorted(set(args.top_ks or [1, 3, 5]))
    embedder = SentenceTransformerEmbeddingProvider(
        args.embedding_model,
        revision=args.embedding_revision,
        device=args.embedding_device,
        batch_size=32,
    )
    rows: list[dict[str, object]] = []
    leakage_violations = 0
    inventory: list[dict[str, object]] = []
    for vc_slug, short_slug in REFERENCE_SLUGS.items():
        root = args.inputs / "data/investors" / vc_slug / "portfolio-memory"
        events = root / "disclosure-events.jsonl"
        corpus = PortfolioMemoryCorpus.load_events(vc_slug, events)
        index = PortfolioMemoryIndex.load(root / "embedding-index.json", corpus, embedder, events)
        inventory.append({
            "vc_slug": vc_slug,
            "disclosure_count": len(corpus.disclosures),
            "named_count": sum(row.company_name is not None for row in corpus.disclosures),
            "anonymous_count": sum(row.company_name is None for row in corpus.disclosures),
        })
        for target_slug in _conflict_reference_slugs(args.references, short_slug):
            availability, relevant_names = _availability(corpus, target_slug)
            if not target_slug.split("-", 1)[0].isdecimal():
                rows.append({
                    "vc_slug": vc_slug,
                    "episode_slug": target_slug,
                    "availability": "unavailable",
                    "relevant_prior_companies": "|".join(sorted(relevant_names)),
                    "eligible_disclosure_count": 0,
                    "pitch_available": False,
                    "retrieved_entities": "",
                    **{f"recall_at_{k}": "" for k in top_ks},
                })
                continue
            filtered = index.for_target(target_slug)
            leakage_violations += sum(
                row.source_episode_number is None
                or row.source_episode_number >= filtered.filtered.target_episode_number
                for row in filtered.filtered.disclosures
            )
            pitch_path = args.inputs / "data/investors" / vc_slug / "pitches" / f"{target_slug}.txt"
            retrieved_names: list[str] = []
            if pitch_path.is_file() and filtered.entities:
                hits = filtered.search(
                    pitch_path.read_text(encoding="utf-8"),
                    limit=max(top_ks),
                    candidate_pool_k=max(15, max(top_ks)),
                )
                retrieved_names = [
                    _normal(hit.company_name) if hit.company_name else hit.entity_id
                    for hit in hits
                ]
            row: dict[str, object] = {
                "vc_slug": vc_slug,
                "episode_slug": target_slug,
                "availability": availability,
                "relevant_prior_companies": "|".join(sorted(relevant_names)),
                "eligible_disclosure_count": len(filtered.filtered.disclosures),
                "pitch_available": pitch_path.is_file(),
                "retrieved_entities": "|".join(retrieved_names),
            }
            for k in top_ks:
                row[f"recall_at_{k}"] = (
                    int(bool(relevant_names & set(retrieved_names[:k])))
                    if availability == "prior_observable"
                    else ""
                )
            rows.append(row)
    counts = {
        label: sum(row["availability"] == label for row in rows)
        for label in ("prior_observable", "first_disclosed_current", "unavailable")
    }
    observable = [row for row in rows if row["availability"] == "prior_observable"]
    recall = {
        f"recall_at_{k}": (
            sum(int(row[f"recall_at_{k}"]) for row in observable) / len(observable)
            if observable else None
        )
        for k in top_ks
    }
    summary = {
        "schema": "portfolio-memory-evaluation-v1",
        "vc_count": len(REFERENCE_SLUGS),
        "conflict_reference_case_count": len(rows),
        "availability_counts": counts,
        "observable_case_count": len(observable),
        "retrieval": recall,
        "temporal_leakage_violations": leakage_violations,
        "phase1_overlap_assessment_runs": 0,
        "inventory": inventory,
        "embedding": {
            "model": args.embedding_model,
            "revision": args.embedding_revision,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "cases.csv", rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    recall_text = ", ".join(
        f"R@{k}={recall[f'recall_at_{k}']:.3f}"
        for k in top_ks if recall[f"recall_at_{k}"] is not None
    ) or "not estimable (no prior-observable reference cases)"
    report = f"""# Temporal Portfolio-Memory Evaluation

The six investor ledgers contain **{sum(int(row['disclosure_count']) for row in inventory)}** source-bound disclosures. Evaluation found **{len(rows)}** transcript-reference cases in which portfolio conflict was material.

## Availability

- Prior observable before the target pitch: **{counts['prior_observable']}**
- First disclosed in the current episode: **{counts['first_disclosed_current']}**
- Unavailable from the accepted ledger: **{counts['unavailable']}**

Only prior-observable cases are valid retrieval tests. Current-episode first disclosures are not counted as retrieval failures because exposing them would leak the target transcript.

## Retrieval

On the **{len(observable)}** prior-observable cases: {recall_text}.

## Leakage and model assessment

Temporal-filter violations: **{leakage_violations}**. No Phase 1 v4.1 runs have yet been regenerated with the new `portfolio_overlap_assessments` field, so model overlap accuracy is explicitly reported as not yet available rather than inferred from older outputs.
"""
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
