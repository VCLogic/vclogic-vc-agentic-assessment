#!/usr/bin/env python3
"""Retry only the v4.3 mapping node from a frozen candidate investigation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vc_clone_graph.artifacts import (
    freeze_model,
    read_verified_frozen,
    verify_phase1_artifacts,
    write_json,
)
from vc_clone_graph.cli import _generation_provider
from vc_clone_graph.config import load_config
from vc_clone_graph.graph import _usage, _usage_by_phase
from vc_clone_graph.prompts_v4 import phase1_rationale_mapping_v43_prompt
from vc_clone_graph.providers.base import GenerationRequest
from vc_clone_graph.schemas_v4 import (
    InvestigationV41,
    RationaleMappingV43,
    model_facing_taxonomy,
    normalize_rationale_mapping_v43_payload,
    rationale_mapping_v43_json_schema,
)
from vc_clone_graph.workflow_v4 import (
    apply_rationale_mapping_v43,
    constrain_rationale_mapping_v43,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, choices=(1, 2), default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.run.contract_version != "v4.3" or config.run.mode != "phase1_only":
        raise ValueError("mapping retry requires a v4.3 phase1-only config")
    root = args.run_root.resolve()
    raw_candidate, candidate_digest = read_verified_frozen(
        root / "phase1/candidate-investigation.json",
        root / "phase1/candidate-investigation.sha256",
    )
    candidate = InvestigationV41.model_validate_json(raw_candidate)
    if candidate.episode_slug != config.run.episode_slug:
        raise ValueError("candidate episode does not match config")
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    registry = state.get("evidence_registry", {})
    cited_ids = {
        evidence_id
        for rationale in candidate.rationales
        for evidence_id in [
            *rationale.wiki_evidence_ids,
            *rationale.historical_evidence_ids,
        ]
    }
    cited_evidence = {
        evidence_id: registry[evidence_id]
        for evidence_id in sorted(cited_ids)
        if evidence_id in registry
    }
    taxonomy_path = Path(config.run.input_root) / config.run.taxonomy_path
    taxonomy = model_facing_taxonomy(
        json.loads(taxonomy_path.read_text(encoding="utf-8"))
    )
    families = {row["label"]: row["coarse_parent"] for row in taxonomy}
    definitions = {row["label"]: row["definition"] for row in taxonomy}
    activated_families = {
        families[row.taxonomy_label] for row in candidate.rationales
    }
    allowed_labels = sorted(
        label for label, family in families.items() if family in activated_families
    )
    prompt = phase1_rationale_mapping_v43_prompt(
        state.get("investor_name", config.run.vc_slug),
        candidate.episode_slug,
        taxonomy,
        candidate.model_dump(mode="json"),
        candidate_digest,
        cited_evidence,
    )
    schema = rationale_mapping_v43_json_schema(
        episode_slug=candidate.episode_slug,
        candidate_investigation_sha256=candidate_digest,
        candidate_rationale_ids=[row.rationale_id for row in candidate.rationales],
        taxonomy_labels=allowed_labels,
    )
    provider = _generation_provider(config, "phase1")
    findings: list[str] = []
    accepted = None
    for attempt in range(1, args.max_attempts + 1):
        request_prompt = prompt
        if attempt > 1:
            request_prompt += (
                "\nREPAIR INSTRUCTION: Return one compact complete JSON object. "
                "For non-merge actions use the literal string none as the merge target."
            )
        request = GenerationRequest(
            "phase1_mapping",
            request_prompt,
            schema,
            max_output_tokens=config.phase1.max_output_tokens or config.provider.max_output_tokens,
            reasoning_effort=config.phase1.reasoning_effort,
        )
        result = provider.generate(request)
        write_json(
            root / "phase1/mapping-repair" / f"attempt-{attempt:02d}-model-response.json",
            {
                "prompt": request.prompt,
                "schema": request.schema,
                "parsed": result.parsed,
                "content": result.content,
                "usage": result.usage.model_dump(mode="json"),
                "elapsed_seconds": result.elapsed_seconds,
                "raw_metadata": result.raw_metadata,
            },
        )
        state["usage"] = _usage(state, result)
        state["usage_by_phase"] = _usage_by_phase(state, result, "phase1")
        try:
            normalized, normalization_findings = normalize_rationale_mapping_v43_payload(
                result.parsed
            )
            findings.extend(normalization_findings)
            parsed = RationaleMappingV43.model_validate(normalized)
            if parsed.episode_slug != candidate.episode_slug:
                raise ValueError("episode slug mismatch")
            if parsed.candidate_investigation_sha256 != candidate_digest:
                raise ValueError("candidate investigation hash mismatch")
            parsed, constraint_findings = constrain_rationale_mapping_v43(
                candidate,
                parsed,
                taxonomy=families,
                definitions=definitions,
            )
            findings.extend(constraint_findings)
            apply_rationale_mapping_v43(
                candidate, parsed, taxonomy=families, mapping_sha256="0" * 64
            )
            accepted = parsed
            break
        except Exception as exc:
            findings.append(f"PHASE1_RATIONALE_MAPPING_REPAIR_INVALID_{attempt}: {exc}")
    if accepted is None:
        write_json(state_path, state)
        raise ValueError("mapping repair exhausted without a usable response")
    _, mapping_digest = freeze_model(root / "phase1", "rationale-mapping", accepted)
    final = apply_rationale_mapping_v43(
        candidate, accepted, taxonomy=families, mapping_sha256=mapping_digest
    )
    _, investigation_digest = freeze_model(root / "phase1", "investigation", final)
    state["investigation"] = final.model_dump(mode="json")
    state["investigation_sha256"] = investigation_digest
    state["rationale_mapping"] = accepted.model_dump(mode="json")
    state["rationale_mapping_sha256"] = mapping_digest
    state["phase1_findings"] = [
        item
        for item in state.get("phase1_findings", [])
        if not item.startswith("PHASE1_RATIONALE_MAPPING_INVALID_ATTEMPT_")
        and item != "PHASE1_RATIONALE_MAPPING_FALLBACK_ORIGINAL"
    ]
    state["phase1_findings"].extend(
        item for item in findings if item not in state["phase1_findings"]
    )
    state.setdefault("events", []).append(
        {
            "name": "phase1_rationale_mapping_repaired",
            "valid": True,
            "relabeled": final.relabeled_candidate_count,
            "merged": final.merged_candidate_count,
        }
    )
    write_json(state_path, state)
    verify_phase1_artifacts(root)
    print(
        f"repaired {candidate.episode_slug}: relabeled={final.relabeled_candidate_count} "
        f"merged={final.merged_candidate_count} total_cost=${state['usage']['cost_usd']:.6f}"
    )


if __name__ == "__main__":
    main()
