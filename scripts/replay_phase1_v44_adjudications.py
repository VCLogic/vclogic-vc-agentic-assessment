#!/usr/bin/env python3
"""Replay recorded Phase 1 v4.4 adjudications without provider/API calls."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from pydantic import BaseModel

from vc_clone_graph.adjudication_postprocess_v44 import (
    finalize_adjudication_v44,
    normalize_adjudication_payload_v44,
)
from vc_clone_graph.artifacts import verify_phase1_artifacts
from vc_clone_graph.config import load_config
from vc_clone_graph.firewall import verify_package
from vc_clone_graph.phase1_v44 import (
    ClaimMapV44,
    ClaimRetrievalManifestV44,
    RationaleAdjudicationV44,
    TaxonomyNeighborhoodManifestV44,
    adjudication_findings_v44,
    build_investigation_v44,
)
from vc_clone_graph.providers.base import (
    GenerationRequest,
    GenerationResult,
    Usage,
)


class RecordedReplayProvider:
    """Return recorded structured outputs without making provider/API calls."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        model: str,
        terminal_adjudication_repeats: int = 0,
    ) -> None:
        if terminal_adjudication_repeats < 0:
            raise ValueError("terminal adjudication repeats cannot be negative")
        self._records = list(records)
        terminal = self._records[-1] if self._records else None
        self._terminal_adjudication = (
            json.loads(json.dumps(terminal))
            if terminal is not None
            and terminal.get("phase") == "phase1_adjudication"
            else None
        )
        self._terminal_adjudication_repeats = terminal_adjudication_repeats
        self.model = model
        self.local_replay_calls = 0
        self.api_calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        started = perf_counter()
        if self._records:
            record = self._records.pop(0)
        elif (
            self._terminal_adjudication_repeats > 0
            and self._terminal_adjudication is not None
            and request.phase == "phase1_adjudication"
        ):
            record = self._terminal_adjudication
            self._terminal_adjudication_repeats -= 1
        else:
            raise RuntimeError("recorded replay provider output script exhausted")
        if record.get("phase") != request.phase:
            raise RuntimeError(
                "recorded replay phase mismatch: "
                f"expected {request.phase}, got {record.get('phase')}"
            )
        parsed = record.get("parsed")
        if type(parsed) is not dict:
            raise RuntimeError("recorded replay output is not a JSON object")
        self.local_replay_calls += 1
        content = json.dumps(parsed, ensure_ascii=True, sort_keys=True)
        return GenerationResult(
            parsed=parsed,
            content=content,
            usage=Usage(),
            elapsed_seconds=perf_counter() - started,
            raw_metadata={
                "provider": "openrouter",
                "requested_model": self.model,
                "returned_model": self.model,
                "model": self.model,
                "replay": "recorded-local",
            },
        )


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"expected JSON object: {path}")
    return value


def _model_sha256(value: BaseModel) -> str:
    raw = json.dumps(
        value.model_dump(mode="json", warnings=False),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def source_hash_inventory(root: Path) -> dict[str, str]:
    """Hash a source tree while rejecting symlink-based redirection."""
    candidate = Path(root)
    if candidate.is_symlink():
        raise ValueError(f"source root is a symlink: {candidate}")
    if not candidate.is_dir():
        raise ValueError(f"attempt root is not a directory: {candidate}")
    inventory: dict[str, str] = {}
    for path in sorted(candidate.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"source tree contains symlink: {path}")
        if path.is_file():
            inventory[path.relative_to(candidate).as_posix()] = sha256(
                path.read_bytes()
            ).hexdigest()
    return inventory


def _terminal_adjudication_call(root: Path) -> tuple[Path, dict[str, Any]]:
    calls = sorted((root / "phase1/adjudication").glob("turn-*/call-*.json"))
    if not calls:
        raise ValueError(f"attempt lacks an adjudication call: {root}")
    path = calls[-1]
    payload = _load_object(path)
    parsed = payload.get("parsed")
    if type(parsed) is not dict:
        raise ValueError(f"terminal adjudication call lacks parsed JSON: {path}")
    return path, payload


def _eligible_ids(
    retrieval: ClaimRetrievalManifestV44,
) -> tuple[list[str], list[str]]:
    investor: set[str] = set()
    portfolio: set[str] = set()
    for bundle in retrieval.claim_bundles:
        investor.update(
            row.evidence_id
            for row in (*bundle.wiki_evidence, *bundle.historical_evidence)
            if row.eligible
        )
        portfolio.update(
            row.evidence_id for row in bundle.portfolio_disclosures if row.eligible
        )
    return sorted(investor), sorted(portfolio)


def replay_attempt(root: Path) -> dict[str, Any]:
    """Semantically replay one recorded attempt without mutating its files."""
    candidate = Path(root)
    before = source_hash_inventory(candidate)
    claim_map = ClaimMapV44.model_validate_json(
        (candidate / "phase1/claim-map.json").read_bytes()
    )
    retrieval = ClaimRetrievalManifestV44.model_validate_json(
        (candidate / "phase1/claim-retrieval.json").read_bytes()
    )
    neighborhood = TaxonomyNeighborhoodManifestV44.model_validate_json(
        (candidate / "phase1/taxonomy-neighborhood.json").read_bytes()
    )
    call_path, call = _terminal_adjudication_call(candidate)
    parsed = call["parsed"]
    raw_dispositions = parsed.get("dispositions")
    if type(raw_dispositions) is not list:
        raise ValueError("terminal adjudication lacks dispositions")
    normalized, normalization_audit = normalize_adjudication_payload_v44(parsed)
    adjudication = RationaleAdjudicationV44.model_validate_json(
        json.dumps(
            normalized,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    finalized, reuse_audit, mapping_audit = finalize_adjudication_v44(
        adjudication,
        neighborhood,
        retrieval,
    )
    investor_ids, portfolio_ids = _eligible_ids(retrieval)
    pitch_ids = sorted(
        {row.pitch_evidence_id for row in claim_map.claim_coverage},
        key=lambda value: int(value.split("-", 1)[1]),
    )
    findings = sorted(
        {
            *finalized.validator_findings,
            *(
                f"UNRESOLVED_CONSTRAINT_MAPPING:{constraint_id}"
                for constraint_id in mapping_audit.unresolved_constraint_ids
            ),
            *adjudication_findings_v44(
                claim_map,
                finalized,
                neighborhood,
                pitch_evidence_ids=pitch_ids,
                investor_evidence_ids=investor_ids,
                portfolio_disclosure_ids=portfolio_ids,
                retrieval_manifest=retrieval,
            ),
        }
    )
    investigation = build_investigation_v44(
        episode_slug=claim_map.episode_slug,
        claim_map=claim_map,
        adjudication=finalized,
        retrieval_manifest=retrieval,
        neighborhood=neighborhood,
        claim_map_sha256=_model_sha256(claim_map),
        adjudication_sha256=_model_sha256(finalized),
        pitch_evidence_ids=pitch_ids,
    )
    after = source_hash_inventory(candidate)
    state = _load_object(candidate / "state.json")
    usage = state.get("usage", {})
    recorded_cost = (
        float(usage.get("cost_usd", 0.0)) if type(usage) is dict else 0.0
    )
    return {
        "attempt_root": str(candidate),
        "episode_slug": claim_map.episode_slug,
        "terminal_call": call_path.relative_to(candidate).as_posix(),
        "terminal_call_sha256": before[call_path.relative_to(candidate).as_posix()],
        "raw_disposition_count": len(raw_dispositions),
        "normalized_disposition_count": len(finalized.dispositions),
        "activated_instance_count": len(investigation.candidate_rationales),
        "unique_activated_label_count": len(
            {row.taxonomy_label for row in investigation.candidate_rationales}
        ),
        "exact_duplicate_dispositions_removed": (
            normalization_audit.exact_duplicate_dispositions_removed
        ),
        "cleared_nonactivating_dispositions": (
            normalization_audit.cleared_nonactivating_dispositions
        ),
        "removed_disposition_positions": list(
            normalization_audit.removed_disposition_positions
        ),
        "cross_target_evidence_reuse": [
            {
                **asdict(row),
                "target_ids": list(row.target_ids),
            }
            for row in reuse_audit
        ],
        "constraint_mapping_revisions": [
            {
                "constraint_id": row.constraint_id,
                "pitch_evidence_ids": list(
                    next(
                        constraint.pitch_evidence_ids
                        for constraint in finalized.constraint_assessments
                        if constraint.constraint_id == row.constraint_id
                    )
                ),
                "original_mapped_ids": list(row.original_mapped_ids),
                "recomputed_mapped_ids": list(row.recomputed_mapped_ids),
            }
            for row in mapping_audit.constraint_mappings
        ],
        "unresolved_constraint_ids": list(
            mapping_audit.unresolved_constraint_ids
        ),
        "findings": findings,
        "investigation_status": investigation.investigation_status,
        "source_files_unchanged": before == after,
        "recorded_source_cost_usd": recorded_cost,
        "api_calls": 0,
        "api_cost_usd": 0.0,
    }


def _recorded_call_sequence(root: Path) -> list[dict[str, Any]]:
    state = _load_object(root / "state.json")
    records = state.get("call_records")
    if type(records) is not list or not records:
        raise ValueError("full replay source lacks recorded call sequence")
    sequence: list[dict[str, Any]] = []
    for record in records:
        if type(record) is not dict or type(record.get("relative_path")) is not str:
            raise ValueError("full replay source call ledger is malformed")
        call_path = root / record["relative_path"]
        raw = call_path.read_bytes()
        if record.get("sha256") != sha256(raw).hexdigest():
            raise ValueError("full replay source call hash mismatch")
        payload = json.loads(raw)
        if type(payload) is not dict:
            raise ValueError("full replay source call is not a JSON object")
        sequence.append(payload)
    return sequence


def full_replay_latest(
    source_root: Path,
    *,
    config_path: Path,
    replay_run_root: Path,
) -> dict[str, Any]:
    """Run the normal CLI-backed v4.4 workflow with recorded local outputs."""
    from vc_clone_graph import cli as cli_module

    project_root = Path(__file__).resolve().parents[1]
    source = Path(source_root)
    before = source_hash_inventory(source)
    config = load_config(Path(config_path))
    target = config.run.episode_slug
    requested_root = Path(replay_run_root).resolve()
    if requested_root.name != target:
        raise ValueError("full replay root must end with the target episode slug")
    try:
        relative_root = requested_root.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("full replay root must be inside the project") from exc
    if requested_root.exists() and any(requested_root.iterdir()):
        raise ValueError(f"full replay root is not empty: {requested_root}")
    checkpoint = relative_root.parent / "checkpoints" / f"{target}.sqlite"
    checkpoint_absolute = project_root / checkpoint
    if checkpoint_absolute.exists():
        raise ValueError(f"full replay checkpoint already exists: {checkpoint}")
    replay_config = config.model_copy(
        update={
            "run": config.run.model_copy(
                update={
                    "output_root": relative_root.parent.as_posix(),
                    "checkpoint_path": checkpoint.as_posix(),
                }
            )
        }
    )
    calls = _recorded_call_sequence(source)
    model = replay_config.phase1.model or replay_config.provider.model
    delegate = RecordedReplayProvider(
        calls,
        model=model,
        terminal_adjudication_repeats=replay_config.phase1_v44.max_revisits,
    )
    configured_provider = cli_module._ConfiguredPhase1ProviderV44(
        delegate, replay_config
    )
    original_factory = cli_module._generation_provider

    def local_provider_factory(candidate_config, phase="phase1"):
        if phase != "phase1" or candidate_config != replay_config:
            raise RuntimeError("full replay attempted an unexpected provider path")
        return configured_provider

    cli_module._generation_provider = local_provider_factory
    try:
        cli_module.command_run(replay_config)
    finally:
        cli_module._generation_provider = original_factory

    package = verify_package(
        Path(replay_config.run.input_root),
        replay_config.run.vc_slug,
        replay_config.run.episode_slug,
        taxonomy_path=replay_config.run.taxonomy_path,
    )
    verify_phase1_artifacts(
        requested_root,
        provenance_mode="cli",
        expected_config=replay_config,
        expected_package=package,
    )
    after = source_hash_inventory(source)
    if after != before:
        raise ValueError("full replay mutated its recorded source attempt")
    state = _load_object(requested_root / "state.json")
    usage = state.get("usage")
    if type(usage) is not dict or float(usage.get("cost_usd", -1)) != 0.0:
        raise ValueError("full local replay recorded non-zero provider cost")
    return {
        "run_root": relative_root.as_posix(),
        "full_artifact_verification": "passed",
        "source_files_unchanged": True,
        "local_replay_calls": delegate.local_replay_calls,
        "api_calls": delegate.api_calls,
        "api_cost_usd": 0.0,
        "usage": usage,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_cases_csv(path: Path, cases: Sequence[dict[str, Any]]) -> None:
    fields = (
        "attempt_root",
        "episode_slug",
        "terminal_call",
        "raw_disposition_count",
        "normalized_disposition_count",
        "activated_instance_count",
        "unique_activated_label_count",
        "exact_duplicate_dispositions_removed",
        "cleared_nonactivating_dispositions",
        "investigation_status",
        "source_files_unchanged",
        "recorded_source_cost_usd",
        "api_calls",
        "api_cost_usd",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for case in cases:
            writer.writerow({field: case[field] for field in fields})


def run_replays(
    attempt_roots: Sequence[Path],
    output: Path,
) -> dict[str, Any]:
    """Replay exactly three immutable attempts and publish one local report."""
    roots = [Path(root) for root in attempt_roots]
    resolved = [root.resolve() for root in roots]
    if len(roots) != 3 or len(set(resolved)) != 3:
        raise ValueError("replay requires exactly three distinct attempt roots")
    destination = Path(output)
    if any(destination.resolve().is_relative_to(root) for root in resolved):
        raise ValueError("replay output cannot be inside an attempt root")
    before = {str(root): source_hash_inventory(root) for root in roots}
    cases = [replay_attempt(root) for root in roots]
    after = {str(root): source_hash_inventory(root) for root in roots}
    unchanged = before == after and all(
        case["source_files_unchanged"] for case in cases
    )
    report = {
        "schema": "phase1-v44-adjudication-replay-v1",
        "attempt_count": len(cases),
        "all_source_files_unchanged": unchanged,
        "new_api_calls": 0,
        "new_api_cost_usd": 0.0,
        "recorded_source_cost_usd": sum(
            float(case["recorded_source_cost_usd"]) for case in cases
        ),
        "cases": cases,
        "latest_attempt": cases[-1],
    }
    if not unchanged:
        raise ValueError("source attempt files changed during replay")
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "source-hashes-before.json", before)
    _write_json(destination / "source-hashes-after.json", after)
    _write_json(destination / "replay-report.json", report)
    _write_cases_csv(destination / "replay-cases.csv", cases)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attempt-root", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-replay-root", type=Path)
    parser.add_argument("--full-replay-config", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_replays(args.attempt_root, args.output)
    if (args.full_replay_root is None) != (args.full_replay_config is None):
        raise ValueError(
            "full replay requires both --full-replay-root and --full-replay-config"
        )
    if args.full_replay_root is not None:
        full = full_replay_latest(
            args.attempt_root[-1],
            config_path=args.full_replay_config,
            replay_run_root=args.full_replay_root,
        )
        report["full_replay"] = full
        report["latest_attempt"]["full_artifact_verification"] = full[
            "full_artifact_verification"
        ]
        report["new_api_calls"] = full["api_calls"]
        report["new_api_cost_usd"] = full["api_cost_usd"]
        _write_json(args.output / "replay-report.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
