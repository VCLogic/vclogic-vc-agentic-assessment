"""Build comparable Phase 1 rationale-reference records from prior annotations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any

from .evaluation import EvaluationRegistry, InvestorSpec


REFERENCE_SCHEMA = "phase1-rationale-reference-v1"
SOURCE_TIERS = {
    3: "newly_extracted",
    2: "previously_validated",
    1: "legacy_fallback",
}


@dataclass(frozen=True)
class ReferenceIdentity:
    vc_slug: str
    vc_name: str
    rich_speaker: str
    legacy_suffix: str


@dataclass(frozen=True)
class _Candidate:
    path: Path
    payload: dict[str, Any]
    source_format: str
    precedence: int


def _probability(value: object, context: str) -> float:
    if not isinstance(value, int | float) or not 0 <= float(value) <= 1:
        raise ValueError(f"{context} must be between zero and one")
    return float(value)


def _evidence_texts(value: object, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} requires evidence")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{context} evidence must contain nonempty strings")
    return list(value)


def normalize_reference(
    payload: dict[str, Any],
    *,
    source_path: Path,
    source_format: str,
    vc_slug: str,
    vc_name: str,
    actual_decision: str,
    taxonomy_labels: set[str],
) -> dict[str, Any]:
    """Normalize a rich or legacy transcript annotation to a shared label-level schema."""
    if actual_decision not in {"In", "Out"}:
        raise ValueError("actual decision must be In or Out")
    episode_slug = payload.get("episode_slug")
    if not isinstance(episode_slug, str) or not episode_slug:
        raise ValueError("reference episode slug is missing")
    raw_rationales = payload.get("rationales")
    if not isinstance(raw_rationales, list) or not raw_rationales:
        raise ValueError(f"reference has no rationales: {source_path}")
    evidence_by_id: dict[str, str] = {}
    if source_format == "transcript-observed-rationales-v1":
        raw_evidence = payload.get("evidence_registry")
        if not isinstance(raw_evidence, list):
            raise ValueError("observed reference has no evidence registry")
        evidence_by_id = {
            str(item["evidence_id"]): str(item["text"])
            for item in raw_evidence
            if isinstance(item, dict) and "evidence_id" in item and "text" in item
        }
    normalized_rationales: list[dict[str, Any]] = []
    for index, raw_value in enumerate(raw_rationales, start=1):
        if not isinstance(raw_value, dict):
            raise ValueError(f"rationale {index} is not an object")
        label = raw_value.get("rationale_label", raw_value.get("label"))
        if label not in taxonomy_labels:
            raise ValueError(f"rationale {index} has invalid taxonomy label: {label!r}")
        if source_format == "transcript-observed-rationales-v1":
            evidence_ids = raw_value.get("evidence_ids")
            if not isinstance(evidence_ids, list) or not evidence_ids:
                raise ValueError(f"rationale {index} has no evidence IDs")
            try:
                evidence = [evidence_by_id[str(item)] for item in evidence_ids]
            except KeyError as exc:
                raise ValueError(f"rationale {index} cites unknown evidence") from exc
            activation = raw_value.get("activation")
            utterance_type = raw_value.get("utterance_type")
            decision_link = raw_value.get("decision_link")
            source_rationale_id = raw_value.get("rationale_id")
        else:
            evidence = _evidence_texts(raw_value.get("evidence_span"), f"rationale {index}")
            activation = "evaluated"
            utterance_type = "assessment"
            decision_link = "unspecified"
            source_rationale_id = None
        normalized_rationales.append(
            {
                "rationale_label": label,
                "direction": raw_value.get("direction"),
                "salience": raw_value.get("salience"),
                "confidence": _probability(raw_value.get("confidence"), f"rationale {index} confidence"),
                "activation": activation,
                "utterance_type": utterance_type,
                "decision_link": decision_link,
                "evidence": evidence,
                "source_rationale_id": source_rationale_id,
            }
        )
    source_sha256 = sha256(source_path.read_bytes()).hexdigest() if source_path.is_file() else None
    return {
        "schema": REFERENCE_SCHEMA,
        "episode_slug": episode_slug,
        "vc_slug": vc_slug,
        "vc_name": vc_name,
        "actual_decision": actual_decision,
        "reference_kind": "automated_transcript_observed_candidate",
        "human_validated": False,
        "source_format": source_format,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "rationales": normalized_rationales,
    }


def _labels(spec: InvestorSpec) -> dict[str, str]:
    payload = json.loads(spec.label_file.read_text(encoding="utf-8"))
    labels = {
        str(item["episode_slug"]): str(item["pitch_window_decision"])
        for item in payload
        if isinstance(item, dict) and item.get("evaluation_eligible") is True
    }
    if len(labels) != spec.eligible_count or any(value not in {"In", "Out"} for value in labels.values()):
        raise ValueError(f"audited labels are inconsistent for {spec.slug}")
    return labels


def _observed_candidate(path: Path, identity: ReferenceIdentity) -> _Candidate | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != "transcript-observed-rationales-v1":
        return None
    evidence = payload.get("evidence_registry")
    if not isinstance(evidence, list) or not evidence:
        return None
    speakers = {item.get("speaker") for item in evidence if isinstance(item, dict)}
    if speakers != {identity.rich_speaker}:
        return None
    return _Candidate(
        path=path,
        payload=payload,
        source_format="transcript-observed-rationales-v1",
        precedence=2,
    )


def _legacy_candidate(path: Path, identity: ReferenceIdentity) -> _Candidate | None:
    suffix = f"__{identity.legacy_suffix}.json"
    if not path.name.endswith(suffix):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("vc_slug") != identity.legacy_suffix:
        return None
    return _Candidate(
        path=path,
        payload=payload,
        source_format="legacy-reference-rationales",
        precedence=1,
    )


def _decision(candidate: _Candidate) -> object:
    if candidate.source_format == "transcript-observed-rationales-v1":
        return candidate.payload.get("observed_decision")
    return candidate.payload.get("decision")


def summarize_extraction_usage(extraction_root: Path) -> dict[str, int]:
    """Sum token usage across every fresh extraction attempt, including retries."""
    totals = {
        "model_call_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    if not extraction_root.is_dir():
        return totals
    for path in sorted(extraction_root.rglob("usage.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        totals["model_call_count"] += 1
        for field in ("input_tokens", "output_tokens", "reasoning_output_tokens"):
            amount = value.get(field, 0)
            if isinstance(amount, int) and amount >= 0:
                totals[field] += amount
    return totals


def build_reference_corpus(
    *,
    registry: EvaluationRegistry,
    identities: dict[str, ReferenceIdentity],
    agentic_root: Path,
    output_root: Path,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Copy and normalize the best existing reference for every eligible episode."""
    unknown = set(registry.investors) - set(identities)
    if unknown:
        raise ValueError(f"missing reference identities: {', '.join(sorted(unknown))}")
    taxonomy_payload = json.loads(
        (agentic_root / "taxonomy" / "codebook_v_final.json").read_text(encoding="utf-8")
    )
    taxonomy_labels = {str(item["label"]) for item in taxonomy_payload}
    candidates: dict[tuple[str, str], list[_Candidate]] = {}

    def consider(vc_slug: str, candidate: _Candidate, *, fresh: bool = False) -> None:
        slug = candidate.payload.get("episode_slug")
        if not isinstance(slug, str):
            return
        adjusted = (
            _Candidate(candidate.path, candidate.payload, candidate.source_format, 3)
            if fresh
            else candidate
        )
        key = (vc_slug, slug)
        candidates.setdefault(key, []).append(adjusted)

    fresh_root = output_root / "extraction_runs"
    if fresh_root.is_dir():
        for path in sorted(fresh_root.rglob("reference.json")):
            for vc_slug, identity in identities.items():
                candidate = _observed_candidate(path, identity)
                if candidate is not None:
                    consider(vc_slug, candidate, fresh=True)
                    break
    prior_outputs = agentic_root / "outputs"
    if prior_outputs.is_dir():
        for path in sorted(prior_outputs.rglob("reference.json")):
            for vc_slug, identity in identities.items():
                candidate = _observed_candidate(path, identity)
                if candidate is not None:
                    consider(vc_slug, candidate)
                    break
    for relative_root in ("data/reference_rationales", "data/reference_rationales_test"):
        legacy_root = agentic_root / relative_root
        if not legacy_root.is_dir():
            continue
        for vc_slug, identity in identities.items():
            for path in sorted(legacy_root.glob(f"*__{identity.legacy_suffix}.json")):
                candidate = _legacy_candidate(path, identity)
                if candidate is not None:
                    consider(vc_slug, candidate)

    records_root = output_root / "records"
    sources_root = output_root / "source_artifacts"
    missing: dict[str, list[str]] = {}
    source_counts: dict[str, int] = {}
    source_tier_counts: dict[str, int] = {}
    record_count = 0
    by_vc: dict[str, dict[str, int]] = {}
    for vc_slug, spec in registry.investors.items():
        identity = identities[vc_slug]
        labels = _labels(spec)
        vc_missing: list[str] = []
        vc_written = 0
        vc_source_tier_counts: dict[str, int] = {}
        for episode_slug, actual in sorted(labels.items()):
            compatible = [
                candidate
                for candidate in candidates.get((vc_slug, episode_slug), [])
                if _decision(candidate) == actual
            ]
            if not compatible:
                vc_missing.append(episode_slug)
                continue
            compatible.sort(key=lambda candidate: candidate.precedence, reverse=True)
            candidate = compatible[0]
            equally_preferred = [
                item
                for item in compatible
                if item.precedence == candidate.precedence and item.path != candidate.path
            ]
            if equally_preferred:
                raise ValueError(
                    f"duplicate references at equal precedence for {vc_slug}/{episode_slug}"
                )
            source_destination = sources_root / vc_slug / f"{episode_slug}.json"
            source_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate.path, source_destination)
            normalized = normalize_reference(
                candidate.payload,
                source_path=source_destination,
                source_format=candidate.source_format,
                vc_slug=vc_slug,
                vc_name=identity.vc_name,
                actual_decision=actual,
                taxonomy_labels=taxonomy_labels,
            )
            source_tier = SOURCE_TIERS[candidate.precedence]
            normalized["source_tier"] = source_tier
            normalized["source_origin_path"] = str(candidate.path)
            record_destination = records_root / vc_slug / f"{episode_slug}.json"
            record_destination.parent.mkdir(parents=True, exist_ok=True)
            record_destination.write_text(
                json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            source_counts[candidate.source_format] = source_counts.get(candidate.source_format, 0) + 1
            source_tier_counts[source_tier] = source_tier_counts.get(source_tier, 0) + 1
            vc_source_tier_counts[source_tier] = vc_source_tier_counts.get(source_tier, 0) + 1
            record_count += 1
            vc_written += 1
        if vc_missing:
            missing[vc_slug] = vc_missing
        by_vc[vc_slug] = {
            "eligible_count": len(labels),
            "record_count": vc_written,
            "missing_count": len(vc_missing),
            "source_tier_counts": dict(sorted(vc_source_tier_counts.items())),
        }
    manifest = {
        "schema": "phase1-rationale-reference-corpus-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_kind": "automated_transcript_observed_candidate",
        "human_validated": False,
        "eligible_count": sum(spec.eligible_count for spec in registry.investors.values()),
        "record_count": record_count,
        "missing_count": sum(len(items) for items in missing.values()),
        "source_format_counts": source_counts,
        "source_tier_counts": dict(sorted(source_tier_counts.items())),
        "extraction_usage": summarize_extraction_usage(fresh_root),
        "by_vc": by_vc,
        "missing": missing,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if missing and not allow_missing:
        raise ValueError(f"reference corpus is incomplete: {manifest['missing_count']} missing")
    return manifest
