"""One-logical-call execution and resumable baseline batching."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from config import BaselineConfig
from input import assemble_context
from prompt import BaselineContext, build_prompt
from schema import OneShotBaselineResponse, baseline_json_schema
from vc_clone_graph.batch import round_robin_order
from vc_clone_graph.providers.base import GenerationProvider, GenerationRequest


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def pilot_slugs(rows):
    labels = dict(rows)
    order = round_robin_order(rows)
    first_in = next(slug for slug in order if labels[slug] == "In")
    first_out = next(slug for slug in order if labels[slug] == "Out")
    return first_in, first_out


def _cited_precedent_slugs(citations: list[str]) -> set[str]:
    """Accept a bare slug or the model's auditable `slug — explanation` form."""
    return {
        citation.strip().split(maxsplit=1)[0].rstrip(":,;.!?")
        for citation in citations
    }


def execute_context(
    context: BaselineContext,
    context_manifest: dict,
    provider: GenerationProvider,
    output_dir: Path,
    actual_label: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(context)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    manifest = dict(context_manifest)
    manifest["prompt_sha256"] = sha256(prompt.encode()).hexdigest()
    manifest["prompt_characters"] = len(prompt)
    write_json_atomic(output_dir / "context-manifest.json", manifest)
    write_json_atomic(output_dir / "selected-precedents.json", [
        {"episode_slug": row.episode_slug, "similarity": row.similarity,
         "observed_decision": row.observed_decision}
        for row in context.precedents
    ])
    labels = {row["label"] for row in context.taxonomy}
    request = GenerationRequest(
        phase="one_shot_baseline", prompt=prompt,
        schema=baseline_json_schema(context.episode_slug, labels),
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    try:
        response = provider.generate(request)
    except Exception as exc:
        summary = {
            "schema": "one-shot-baseline-summary-v1", "episode_slug": context.episode_slug,
            "actual_label": actual_label, "status": "failed",
            "error": f"{type(exc).__name__}: {exc}", "usage": {},
        }
        write_json_atomic(output_dir / "validation-findings.json", {"errors": [summary["error"]]})
        write_json_atomic(output_dir / "summary.json", summary)
        return summary

    raw = response.model_dump(mode="json")
    write_json_atomic(output_dir / "provider-response.json", raw)
    findings = []
    value = None
    try:
        if response.parsed is None:
            raise ValueError("provider response is not JSON")
        value = OneShotBaselineResponse.model_validate(response.parsed)
        value.validate_taxonomy(labels)
        selected = {row.episode_slug for row in context.precedents}
        unknown_precedents = _cited_precedent_slugs(value.decision.decisive_precedents) - selected
        if unknown_precedents:
            raise ValueError(f"decision cites unselected precedent: {sorted(unknown_precedents)[0]}")
    except (ValidationError, ValueError) as exc:
        findings.append(f"{type(exc).__name__}: {exc}")
        value = None

    if value is not None:
        result = value.model_dump(mode="json")
        write_json_atomic(output_dir / "result.json", result)
        result_sha = sha256((json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()).hexdigest()
        status = "completed"
    else:
        result_sha = None
        status = "failed"
    write_json_atomic(output_dir / "validation-findings.json", {"errors": findings})
    summary = {
        "schema": "one-shot-baseline-summary-v1", "episode_slug": context.episode_slug,
        "actual_label": actual_label, "status": status,
        "prompt_sha256": manifest["prompt_sha256"], "result_sha256": result_sha,
        "usage": response.usage.model_dump(mode="json"),
        "elapsed_seconds": response.elapsed_seconds,
        "provider_metadata": response.raw_metadata, "validation_errors": findings,
    }
    if value is not None:
        summary["prediction"] = {
            "decision": value.decision.decision,
            "investment_likelihood": value.decision.investment_likelihood,
            "decision_confidence": value.decision.decision_confidence,
            "ranking_score": value.decision.ranking_score,
            "any_check": value.decision.any_check.model_dump(mode="json"),
            "standard_check": value.decision.standard_check.model_dump(mode="json"),
        }
        summary["rationale_count"] = len(value.rationales)
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def run_episode(config: BaselineConfig, slug: str, actual_label: str, provider):
    context, manifest = assemble_context(config, slug)
    return execute_context(
        context, manifest, provider, Path(config.output_root) / slug,
        actual_label, config.provider.reasoning_effort,
        config.provider.max_output_tokens,
    )
