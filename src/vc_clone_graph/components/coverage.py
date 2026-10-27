"""LLM coverage expansion for rationales Phase 1 did not address.

Input is deliberately small: the pitch, the run's already-selected investor
policy evidence, a label/direction/salience list of Phase 1 rationales and the
definitions of uncovered taxonomy labels only. No retrieval, precedents or
Phase 1 justifications are sent. Additions are kept separate from Phase 1.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from typing import Any

from ..providers.base import GenerationProvider, GenerationRequest
from .config import CoverageSettings
from .run_inputs import RunArtifacts

SCHEMA_VERSION = "coverage-expansion-v1"
DIRECTIONS = ("positive", "negative", "neutral")
SALIENCES = ("primary", "secondary")


def uncovered_labels(run: RunArtifacts) -> list[dict[str, Any]]:
    covered = {r["taxonomy_label"] for r in run.phase1.get("rationales", [])}
    return [r for r in run.taxonomy if r["label"] not in covered]


def build_prompt(run: RunArtifacts, settings: CoverageSettings) -> tuple[str, list[str]]:
    candidates = uncovered_labels(run)
    labels = [r["label"] for r in candidates]
    phase1_lines = "\n".join(
        f"- {r['taxonomy_label']} ({r['direction']}, {r['salience']})"
        for r in run.phase1.get("rationales", [])
    ) or "- none"
    label_lines = "\n".join(
        f"- {r['label']} [{r.get('coarse_parent', '')}]: {r.get('definition', '')}"
        for r in candidates
    )
    limit = settings.max_policy_chars_per_excerpt
    policy_lines = "\n\n".join(
        f"[{e.get('evidence_id')}] {e.get('heading', '')}\n{str(e.get('text', ''))[:limit]}"
        for e in run.policy_evidence
    ) or "No investor policy evidence was saved for this run."
    prompt = f"""You are auditing the coverage of an investment-rationale analysis.

Investor: {run.investor_name}

An earlier analysis of this pitch already produced these rationales (taxonomy label, direction, salience):
{phase1_lines}

Task: decide whether any of the UNCOVERED taxonomy labels below would plausibly matter to this investor's In/Out decision on this pitch. Consider only the uncovered labels. Do not repeat or revise the covered ones.

Rules:
- Add a label only when the pitch contains specific evidence bearing on it. Quote that evidence verbatim in pitch_quote (one contiguous excerpt, at most 40 words, copied exactly from the pitch).
- Direction is from this investor's perspective: positive supports investing, negative weighs against, neutral means a material open question the investor would need answered.
- Do not default to positive. Investor-fit constraints (stage, check size, sector, geography, mandate), timing and deal terms (valuation, round size, structure) are frequently missed; check them against the investor policy evidence.
- Cite investor policy evidence IDs in policy_evidence_ids when they make the label material; use an empty list otherwise.
- Salience primary means it would likely be among the investor's deciding reasons; otherwise secondary.
- Return at most {settings.max_additions} additions. An empty list is the correct answer when nothing qualifies.

Uncovered taxonomy labels:
{label_lines}

Investor policy evidence (earlier excerpts about this investor; the pitch episode itself is excluded):
{policy_lines}

Pitch (founder statements only):
<pitch>
{run.pitch_text}
</pitch>"""
    return prompt, labels


def output_schema(labels: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "additions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "taxonomy_label": {"type": "string", "enum": labels},
                        "direction": {"type": "string", "enum": list(DIRECTIONS)},
                        "salience": {"type": "string", "enum": list(SALIENCES)},
                        "confidence": {"type": "number"},
                        "pitch_quote": {"type": "string"},
                        "policy_evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "justification": {"type": "string"},
                    },
                },
            },
            "review_note": {"type": "string"},
        },
    }


def _normalize(text: str) -> str:
    text = text.casefold().replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"').replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", text).strip()


def validate_additions(
    raw: dict[str, Any] | None, run: RunArtifacts, labels: list[str], settings: CoverageSettings
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep additions whose label is uncovered and whose quote occurs in the pitch."""
    if not isinstance(raw, dict) or not isinstance(raw.get("additions"), list):
        raise ValueError("coverage output is not a valid structured object")
    pitch = _normalize(run.pitch_text)
    policy_ids = {e.get("evidence_id") for e in run.policy_evidence}
    allowed = set(labels)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw["additions"]:
        reason = None
        label = item.get("taxonomy_label")
        quote = _normalize(str(item.get("pitch_quote", "")))
        if label not in allowed:
            reason = "label_already_covered_or_unknown"
        elif label in seen:
            reason = "duplicate_label"
        elif item.get("direction") not in DIRECTIONS or item.get("salience") not in SALIENCES:
            reason = "invalid_direction_or_salience"
        elif len(quote) < 8 or quote not in pitch:
            reason = "quote_not_found_in_pitch"
        elif len(kept) >= settings.max_additions:
            reason = "over_max_additions"
        if reason:
            rejected.append({**item, "rejection_reason": reason})
            continue
        seen.add(label)
        unknown_ids = [i for i in item.get("policy_evidence_ids", []) if i not in policy_ids]
        kept.append({
            "addition_id": f"C{len(kept) + 1}",
            "taxonomy_label": label,
            "direction": item["direction"],
            "salience": item["salience"],
            "confidence": min(max(float(item.get("confidence", 0.5)), 0.0), 1.0),
            "pitch_quote": item["pitch_quote"],
            "policy_evidence_ids": [i for i in item.get("policy_evidence_ids", []) if i in policy_ids],
            "unknown_policy_evidence_ids": unknown_ids,
            "justification": item.get("justification", ""),
        })
    return kept, rejected


def run_coverage(
    run: RunArtifacts, settings: CoverageSettings, provider: GenerationProvider
) -> dict[str, Any]:
    prompt, labels = build_prompt(run, settings)
    base = {
        "schema": SCHEMA_VERSION,
        "vc_slug": run.vc_slug,
        "episode_slug": run.episode_slug,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.model,
        "input_hashes": run.hashes,
        "prompt_chars": len(prompt),
        "uncovered_label_count": len(labels),
    }
    if not labels:
        return {**base, "status": "skipped_no_uncovered_labels", "additions": [], "rejected": []}
    result = provider.generate(GenerationRequest(
        phase="coverage_expansion",
        prompt=prompt,
        schema=output_schema(labels),
        max_output_tokens=settings.max_output_tokens,
        reasoning_effort=settings.reasoning_effort,
    ))
    additions, rejected = validate_additions(result.parsed, run, labels, settings)
    return {
        **base,
        "status": "completed",
        "additions": additions,
        "rejected": rejected,
        "review_note": (result.parsed or {}).get("review_note", ""),
        "usage": result.usage.model_dump(),
        "provider_metadata": result.raw_metadata,
    }


def openrouter_provider(settings: CoverageSettings) -> GenerationProvider:
    from ..providers.openrouter import OpenRouterProvider

    return OpenRouterProvider(
        settings.model,
        base_url=settings.base_url,
        api_key_env=settings.api_key_env,
        max_output_tokens=settings.max_output_tokens,
        data_collection=settings.data_collection,
    )


def dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
