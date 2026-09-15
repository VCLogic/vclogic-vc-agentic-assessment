"""Trusted adapters from canonical v4/v4.1 runs to founder rehearsal."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
from pathlib import Path
from typing import Any, Literal

from .artifacts import read_verified_frozen
from .prompts_v4 import pitch_evidence_index
from .rehearsal_schemas import (
    EvidenceRef,
    GroundedArtifactRef,
    GroundedBaselineSummary,
    InitialAssessment,
    RationaleState,
)
from .schemas_v4 import DecisionV4, DecisionV41, InvestigationV4, InvestigationV41


CanonicalStatus = Literal["accepted", "provisional"]
CanonicalContract = Literal["v4", "v4.1"]


@dataclass(frozen=True)
class CanonicalRehearsalBaseline:
    canonical_vc_slug: str
    episode_slug: str
    contract_version: CanonicalContract
    phase1_status: CanonicalStatus
    phase2_status: CanonicalStatus
    run_root: Path
    investigation: InvestigationV4 | InvestigationV41
    investigation_sha256: str
    decision: DecisionV4 | DecisionV41
    decision_sha256: str
    phase1_state: dict[str, Any]
    pitch_evidence: dict[str, str]


def _inside(workspace: Path, candidate: Path, *, label: str) -> Path:
    resolved = candidate.resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"{label} escapes the workspace")
    return resolved


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _status(value: object, *, phase: str) -> CanonicalStatus:
    if value not in {"accepted", "provisional"}:
        raise ValueError(f"canonical {phase} status is not usable: {value}")
    return value


def _pitch_map(workspace: Path, run_root: Path) -> dict[str, str]:
    provenance = _json(run_root / "input-provenance.json")
    raw_path = provenance.get("pitch_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("canonical input provenance lacks pitch_path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = workspace / path
    path = _inside(workspace, path, label="canonical pitch path")
    expected = provenance.get("pitch_sha256")
    from hashlib import sha256

    actual = sha256(path.read_bytes()).hexdigest()
    if expected != actual:
        # A canonical v4 artifact embeds the exact cited pitch excerpts. Do not
        # reinterpret a later, changed source file as the original P-ID index.
        return {}
    return {
        row["evidence_id"]: row["text"]
        for row in pitch_evidence_index(path.read_text(encoding="utf-8"))
    }


def load_canonical_baseline_from_run(
    *,
    workspace: Path,
    run_root: Path,
    canonical_vc_slug: str,
    expected_episode_slug: str,
    expected_pitch_sha256: str | None = None,
    expected_run_vc_slug: str | None = None,
) -> CanonicalRehearsalBaseline:
    """Load one complete canonical run without consulting an evaluation registry."""
    root = Path(workspace).resolve()
    canonical_run = _inside(root, Path(run_root), label="canonical run")
    summary = _json(canonical_run / "summary.json")
    state = _json(canonical_run / "state.json")
    run_config = _json(canonical_run / "run-config.json")
    provenance = _json(canonical_run / "input-provenance.json")
    configured_run = run_config.get("run")
    if not isinstance(configured_run, dict):
        raise ValueError("canonical run configuration lacks run identity")
    if summary.get("episode_slug") != expected_episode_slug:
        raise ValueError("canonical summary episode mismatch")
    if (
        expected_pitch_sha256 is not None
        and provenance.get("pitch_sha256") != expected_pitch_sha256
    ):
        raise ValueError("canonical baseline pitch digest mismatch")

    contract = summary.get("contract_version")
    if contract not in {"v4", "v4.1"}:
        raise ValueError(f"unsupported canonical contract version: {contract}")
    if configured_run.get("contract_version") != contract:
        raise ValueError("canonical run contract differs from summary")
    phase1_status = _status(summary.get("phase1_status"), phase="Phase 1")
    phase2_status = _status(summary.get("phase2_status"), phase="Phase 2")

    investigation_raw, investigation_digest = read_verified_frozen(
        canonical_run / "phase1/investigation.json",
        canonical_run / "phase1/investigation.sha256",
    )
    decision_raw, decision_digest = read_verified_frozen(
        canonical_run / "phase2/decision.json",
        canonical_run / "phase2/decision.sha256",
    )
    if (
        expected_run_vc_slug is not None
        and configured_run.get("vc_slug") != expected_run_vc_slug
    ):
        raise ValueError("canonical baseline investor mismatch")
    if configured_run.get("episode_slug") != expected_episode_slug:
        raise ValueError("canonical run configuration episode mismatch")
    if state.get("investigation_sha256") != investigation_digest:
        raise ValueError("canonical investigation state digest mismatch")
    if state.get("decision_sha256") != decision_digest:
        raise ValueError("canonical decision state digest mismatch")

    if contract == "v4.1":
        investigation = InvestigationV41.model_validate_json(investigation_raw)
        decision = DecisionV41.model_validate_json(decision_raw)
    else:
        investigation = InvestigationV4.model_validate_json(investigation_raw)
        decision = DecisionV4.model_validate_json(decision_raw)
    if (
        investigation.episode_slug != expected_episode_slug
        or decision.episode_slug != expected_episode_slug
    ):
        raise ValueError("canonical baseline episode mismatch")
    if decision.investigation_sha256 != investigation_digest:
        raise ValueError("canonical decision is not bound to the investigation")
    if state.get("investigation") != investigation.model_dump(mode="json"):
        raise ValueError("canonical state investigation differs from frozen artifact")
    if state.get("decision") != decision.model_dump(mode="json"):
        raise ValueError("canonical state decision differs from frozen artifact")

    return CanonicalRehearsalBaseline(
        canonical_vc_slug=canonical_vc_slug,
        episode_slug=expected_episode_slug,
        contract_version=contract,
        phase1_status=phase1_status,
        phase2_status=phase2_status,
        run_root=canonical_run,
        investigation=investigation,
        investigation_sha256=investigation_digest,
        decision=decision,
        decision_sha256=decision_digest,
        phase1_state=state,
        pitch_evidence=_pitch_map(root, canonical_run),
    )


def load_canonical_baseline(
    *,
    workspace: Path,
    registry_path: Path,
    canonical_vc_slug: str,
    episode_slug: str,
) -> CanonicalRehearsalBaseline:
    """Resolve one exact canonical run and verify every trusted boundary."""
    root = Path(workspace).resolve()
    registry_file = _inside(root, Path(registry_path), label="canonical registry")
    registry = _json(registry_file)
    if registry.get("schema") != "canonical-vc-evaluation-registry-v1":
        raise ValueError("unsupported canonical registry schema")
    investors = registry.get("investors")
    if not isinstance(investors, dict) or canonical_vc_slug not in investors:
        raise ValueError(f"canonical investor is unavailable: {canonical_vc_slug}")
    investor = investors[canonical_vc_slug]
    if not isinstance(investor, dict):
        raise ValueError("canonical investor registry entry must be an object")
    sources = investor.get("sources")
    if not isinstance(sources, list):
        raise ValueError("canonical investor registry lacks sources")
    matches: list[Path] = []
    for source in sources:
        if not isinstance(source, dict) or source.get("kind") != "summary_glob":
            continue
        pattern = source.get("path")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("canonical summary source lacks path")
        absolute_pattern = registry_file.parent / pattern
        for candidate_text in glob.glob(str(absolute_pattern)):
            candidate = _inside(root, Path(candidate_text), label="canonical summary")
            summary = _json(candidate)
            if summary.get("episode_slug") == episode_slug:
                matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one canonical baseline for "
            f"{canonical_vc_slug}/{episode_slug}; found {len(matches)}"
        )

    summary_path = matches[0]
    run_config = _json(summary_path.parent / "run-config.json")
    configured_run = run_config.get("run")
    expected_run_vc_slug = (
        str(configured_run.get("vc_slug"))
        if isinstance(configured_run, dict) and configured_run.get("vc_slug")
        else None
    )
    return load_canonical_baseline_from_run(
        workspace=root,
        run_root=summary_path.parent,
        canonical_vc_slug=canonical_vc_slug,
        expected_run_vc_slug=expected_run_vc_slug,
        expected_episode_slug=episode_slug,
    )


def _evidence_refs(
    baseline: CanonicalRehearsalBaseline, rationale: Any
) -> tuple[EvidenceRef, ...]:
    rows: list[EvidenceRef] = []
    explicit_pitch = list(getattr(rationale, "pitch_evidence", []))
    for position, evidence_id in enumerate(rationale.pitch_evidence_ids):
        excerpt = baseline.pitch_evidence.get(evidence_id)
        if excerpt is None and position < len(explicit_pitch):
            excerpt = explicit_pitch[position]
        if not excerpt:
            raise ValueError(f"canonical pitch evidence is unavailable: {evidence_id}")
        rows.append(
            EvidenceRef(
                evidence_id=evidence_id,
                source_kind="pitch",
                source_path=str(baseline.run_root / "input-provenance.json"),
                excerpt=excerpt,
            )
        )
    registry = baseline.phase1_state.get("evidence_registry", {})
    if not isinstance(registry, dict):
        raise ValueError("canonical evidence registry is invalid")
    for source_kind, identifiers in (
        ("wiki", rationale.wiki_evidence_ids),
        ("precedent", rationale.historical_evidence_ids),
    ):
        for evidence_id in identifiers:
            source = registry.get(evidence_id)
            if not isinstance(source, dict) or not source.get("text"):
                raise ValueError(f"canonical investor evidence is unavailable: {evidence_id}")
            rows.append(
                EvidenceRef(
                    evidence_id=evidence_id,
                    source_kind=source_kind,
                    source_path=str(
                        source.get("source_path")
                        or source.get("episode_slug")
                        or "canonical-evidence-registry"
                    ),
                    excerpt=str(source["text"]),
                )
            )
    return tuple(rows)


def baseline_initial_assessment(
    baseline: CanonicalRehearsalBaseline,
) -> InitialAssessment:
    """Adapt, but never reinterpret, the canonical baseline."""
    rationales = tuple(
        RationaleState(
            rationale_id=row.rationale_id,
            taxonomy_label=row.taxonomy_label,
            direction=row.direction,
            salience=row.salience,
            confidence=row.confidence,
            assessment=row.justification,
            evidence_refs=_evidence_refs(baseline, row),
        )
        for row in baseline.investigation.rationales
    )
    unresolved = tuple(
        dict.fromkeys(
            [
                *baseline.investigation.searchable_questions,
                *baseline.investigation.diligence_questions,
                *baseline.decision.searchable_questions,
                *baseline.decision.diligence_questions,
                *baseline.decision.missing_considerations,
            ]
        )
    )
    return InitialAssessment(
        decision=baseline.decision.decision,
        investment_likelihood=baseline.decision.investment_likelihood,
        decision_confidence=baseline.decision.decision_confidence,
        rationales=rationales,
        unresolved_questions=unresolved,
        candidate_questions=(),
    )


def baseline_summary(
    baseline: CanonicalRehearsalBaseline,
    *,
    workspace: Path,
) -> GroundedBaselineSummary:
    """Produce the auditable public reference stored with the session."""
    root = Path(workspace).resolve()
    return GroundedBaselineSummary(
        contract_version=baseline.contract_version,
        episode_slug=baseline.episode_slug,
        phase1=GroundedArtifactRef(
            path=str(
                (baseline.run_root / "phase1/investigation.json").relative_to(root)
            ),
            sha256=baseline.investigation_sha256,
            status=baseline.phase1_status,
        ),
        phase2=GroundedArtifactRef(
            path=str((baseline.run_root / "phase2/decision.json").relative_to(root)),
            sha256=baseline.decision_sha256,
            status=baseline.phase2_status,
        ),
        decision=baseline.decision.decision,
        investment_likelihood=baseline.decision.investment_likelihood,
        decision_confidence=baseline.decision.decision_confidence,
        controlling_rationale_ids=tuple(baseline.decision.controlling_rationale_ids),
    )
