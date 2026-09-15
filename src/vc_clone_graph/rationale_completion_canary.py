"""Auditable, cost-gated verification for rationale-completion canaries."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import json
from pathlib import Path
from typing import Mapping, Sequence

from .providers.base import GenerationProvider, GenerationRequest
from .rationale_completion_verification import (
    CompletionVerification,
    build_completion_verification_prompt,
    completion_verification_schema,
    validate_completion_verification,
)


@dataclass(frozen=True)
class VerificationCallResult:
    verification: CompletionVerification
    input_tokens: int
    output_tokens: int
    cost_usd: float
    elapsed_seconds: float


def conservative_pricing_preflight(
    *,
    prompt: str,
    schema: Mapping[str, object],
    max_output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
    spent_cost_usd: float,
    max_cost_usd: float,
) -> dict[str, float | int]:
    if input_price_per_million <= 0 or output_price_per_million <= 0:
        raise ValueError("asserted provider prices must be positive")
    schema_bytes = len(
        json.dumps(schema, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    # Each UTF-8 byte is a conservative upper bound on BPE pieces. Add explicit
    # schema bytes plus a generous fixed allowance for message/provider framing.
    input_token_upper_bound = len(prompt.encode("utf-8")) + schema_bytes + 4_096
    conservative_call_cost_usd = (
        input_token_upper_bound * input_price_per_million
        + max_output_tokens * output_price_per_million
    ) / 1_000_000
    if spent_cost_usd + conservative_call_cost_usd > max_cost_usd:
        raise ValueError("conservative request estimate exceeds cost cap")
    return {
        "input_price_per_million_usd": input_price_per_million,
        "output_price_per_million_usd": output_price_per_million,
        "input_token_upper_bound": input_token_upper_bound,
        "schema_byte_count": schema_bytes,
        "framing_token_allowance": 4_096,
        "max_output_tokens": max_output_tokens,
        "conservative_call_cost_usd": conservative_call_cost_usd,
        "spent_before_call_usd": spent_cost_usd,
        "experiment_cost_cap_usd": max_cost_usd,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_minor_contract_fields(
    payload: Mapping[str, object],
    *,
    pitch_evidence_ids: set[str] | None = None,
    investor_evidence_ids: set[str] | None = None,
) -> dict[str, object]:
    """Clear semantically inapplicable fields without changing model dispositions."""
    normalized = deepcopy(dict(payload))
    findings = list(normalized.get("validator_findings") or [])
    dispositions = normalized.get("dispositions")
    if not isinstance(dispositions, list):
        return normalized
    for row in dispositions:
        if not isinstance(row, dict):
            continue
        label = row.get("taxonomy_label", "unknown")
        for field, allowed, kind in (
            ("pitch_evidence_ids", pitch_evidence_ids, "pitch"),
            ("investor_evidence_ids", investor_evidence_ids, "investor"),
        ):
            if allowed is None or not isinstance(row.get(field), list):
                continue
            unknown = [value for value in row[field] if value not in allowed]
            if unknown:
                row[field] = [value for value in row[field] if value in allowed]
                findings.append(
                    f"dropped unknown {kind} evidence IDs for {label}: "
                    + ", ".join(unknown)
                )
        if row.get("disposition") in {
            "activated_core",
            "activated_candidate",
        }:
            continue
        fields = ("direction", "salience", "confidence")
        if any(row.get(field) is not None for field in fields):
            for field in fields:
                row[field] = None
            findings.append(
                "normalized inapplicable activation attributes for "
                f"{label}"
            )
    normalized["validator_findings"] = findings
    return normalized


def run_verification_call(
    *,
    provider: GenerationProvider,
    investor_name: str,
    episode_slug: str,
    pitch_evidence: Sequence[Mapping[str, object]],
    canonical_rationales: Sequence[Mapping[str, object]],
    candidate_definitions: Mapping[str, str],
    investor_evidence: Sequence[Mapping[str, object]],
    output_dir: Path,
    max_cost_usd: float,
    spent_cost_usd: float,
    input_price_per_million: float,
    output_price_per_million: float,
    max_output_tokens: int = 8192,
) -> VerificationCallResult:
    """Make one decision-blind verification call and preserve its audit artifacts."""
    if max_cost_usd < 0 or spent_cost_usd < 0:
        raise ValueError("cost values must be non-negative")
    if not candidate_definitions:
        raise ValueError("at least one rationale hypothesis is required")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_completion_verification_prompt(
        investor_name=investor_name,
        episode_slug=episode_slug,
        pitch_evidence=pitch_evidence,
        canonical_rationales=canonical_rationales,
        candidate_definitions=candidate_definitions,
        investor_evidence=investor_evidence,
    )
    # A byte is a conservative upper bound on tokenizer pieces for provider BPEs;
    # reserve an additional 1,024 tokens for message/schema framing.
    schema = completion_verification_schema()
    pricing = conservative_pricing_preflight(
        prompt=prompt,
        schema=schema,
        max_output_tokens=max_output_tokens,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        spent_cost_usd=spent_cost_usd,
        max_cost_usd=max_cost_usd,
    )
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    _write_json(
        output_dir / "pricing-preflight.json",
        pricing,
    )
    _write_json(
        output_dir / "input-packet.json",
        {
            "episode_slug": episode_slug,
            "pitch_evidence": list(pitch_evidence),
            "canonical_rationales": list(canonical_rationales),
            "candidate_definitions": dict(candidate_definitions),
            "investor_evidence": list(investor_evidence),
        },
    )

    response = provider.generate(
        GenerationRequest(
            phase="rationale_completion_verification",
            prompt=prompt,
            schema=schema,
            max_output_tokens=max_output_tokens,
            reasoning_effort="high",
        )
    )
    _write_json(
        output_dir / "raw-response.json",
        {
            "content": response.content,
            "parsed": response.parsed,
            "raw_metadata": response.raw_metadata,
        },
    )
    usage = {
        "input_tokens": response.usage.input_tokens,
        "cached_input_tokens": response.usage.cached_input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": response.usage.cost_usd,
        "elapsed_seconds": response.elapsed_seconds,
    }
    _write_json(output_dir / "usage.json", usage)
    if response.parsed is None:
        raise ValueError("verification provider returned no parsed object")
    pitch_ids = {str(row["evidence_id"]) for row in pitch_evidence}
    investor_ids = {str(row["evidence_id"]) for row in investor_evidence}
    normalized = normalize_minor_contract_fields(
        response.parsed,
        pitch_evidence_ids=pitch_ids,
        investor_evidence_ids=investor_ids,
    )
    verification = validate_completion_verification(
        normalized,
        episode_slug=episode_slug,
        selected_labels=tuple(candidate_definitions),
        pitch_evidence_ids=pitch_ids,
        investor_evidence_ids=investor_ids,
    )
    if spent_cost_usd + response.usage.cost_usd > max_cost_usd:
        raise ValueError("provider-reported usage exceeds cost cap")
    _write_json(output_dir / "verification.json", verification.model_dump(mode="json"))
    return VerificationCallResult(
        verification=verification,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=response.usage.cost_usd,
        elapsed_seconds=response.elapsed_seconds,
    )
