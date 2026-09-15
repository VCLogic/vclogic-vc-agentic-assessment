"""Canonical artifact writing and frozen-output verification."""

from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

from pydantic import BaseModel, ValidationError


_V44_CLI_PROVIDER_CLASS = "vc_clone_graph.cli._ConfiguredPhase1ProviderV44"
_V44_CLI_PROVIDER_CLASS_ALIASES = {
    _V44_CLI_PROVIDER_CLASS,
    "__main__._ConfiguredPhase1ProviderV44",
}


def _canonical_v44_cli_provider_class(value: str) -> str:
    """Normalize only the exact class alias produced by ``python -m``."""
    if value in _V44_CLI_PROVIDER_CLASS_ALIASES:
        return _V44_CLI_PROVIDER_CLASS
    return value


def canonical_bytes(value: BaseModel | dict[str, Any]) -> bytes:
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return (json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode()


def v44_execution_settings_sha256(config: Any) -> str:
    """Bind complete provider/embedding settings without persisting secret-like fields."""
    embedding = config.embedding
    payload = {
        "provider": config.provider.model_dump(mode="json"),
        "embedding": (
            None if embedding is None else embedding.model_dump(mode="json")
        ),
    }
    return sha256(canonical_bytes(payload)).hexdigest()


def freeze_model(root: Path, name: str, value: BaseModel) -> tuple[Path, str]:
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(value)
    digest = sha256(raw).hexdigest()
    json_path = directory / f"{name}.json"
    sha_path = directory / f"{name}.sha256"
    json_path.write_bytes(raw)
    sha_path.write_text(digest + "\n", encoding="ascii")
    return json_path, digest


def read_verified_frozen(json_path: Path, sha_path: Path) -> tuple[bytes, str]:
    raw = Path(json_path).read_bytes()
    expected = Path(sha_path).read_text(encoding="ascii").strip()
    actual = sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(f"frozen artifact hash mismatch: {json_path}")
    return raw, actual


def verify_frozen(json_path: Path, sha_path: Path) -> str:
    _, digest = read_verified_frozen(json_path, sha_path)
    return digest


def write_json(path: Path, value: Any) -> None:
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_phase1_artifacts(
    run_root: Path,
    *,
    provenance_mode: str = "direct",
    expected_config: Any | None = None,
    expected_package: Any | None = None,
) -> None:
    """Verify a frozen Phase-1-only run and its review binding."""
    from .schemas_v4 import InvestigationV41, InvestigationV42, RationaleLockV42

    if provenance_mode not in {"direct", "cli"}:
        raise ValueError("Phase 1 provenance mode is invalid")
    root = Path(run_root)
    final_payload = _load_object(root / "phase1/investigation.json")
    if final_payload.get("schema_version") == "investigation-v4.4":
        _verify_phase1_v44_artifacts(
            root,
            provenance_mode=provenance_mode,
            expected_config=expected_config,
            expected_package=expected_package,
        )
        return
    if final_payload.get("schema_version") == "investigation-v4.3":
        _verify_phase1_v43_artifacts(root)
        return
    candidate_digest = verify_frozen(
        root / "phase1/candidate-investigation.json",
        root / "phase1/candidate-investigation.sha256",
    )
    lock_digest = verify_frozen(
        root / "phase1/rationale-lock.json",
        root / "phase1/rationale-lock.sha256",
    )
    investigation_digest = verify_frozen(
        root / "phase1/investigation.json",
        root / "phase1/investigation.sha256",
    )
    candidate = InvestigationV41.model_validate(
        _load_object(root / "phase1/candidate-investigation.json")
    )
    lock = RationaleLockV42.model_validate(
        _load_object(root / "phase1/rationale-lock.json")
    )
    final = InvestigationV42.model_validate(
        _load_object(root / "phase1/investigation.json")
    )
    if lock.candidate_investigation_sha256 != candidate_digest:
        raise ValueError("rationale lock does not bind candidate investigation")
    if final.candidate_investigation_sha256 != candidate_digest:
        raise ValueError("final investigation does not bind candidate investigation")
    if final.rationale_lock_sha256 != lock_digest:
        raise ValueError("final investigation does not bind rationale lock")
    candidate_by_id = {row.rationale_id: row for row in candidate.rationales}
    disposition_by_id = {row.rationale_id: row for row in lock.dispositions}
    if set(candidate_by_id) != set(disposition_by_id):
        raise ValueError("rationale lock candidate coverage mismatch")
    expected_locked = {
        identifier
        for identifier, row in disposition_by_id.items()
        if row.decision == "locked"
    }
    final_by_id = {row.rationale_id: row for row in final.rationales}
    if set(final_by_id) != expected_locked:
        raise ValueError("final rationale set does not match rationale lock")
    for identifier, row in final_by_id.items():
        candidate_row = candidate_by_id[identifier]
        for field in (
            "taxonomy_label",
            "pitch_evidence_ids",
            "wiki_evidence_ids",
            "historical_evidence_ids",
            "justification",
        ):
            if getattr(row, field) != getattr(candidate_row, field):
                raise ValueError("locked rationale changed immutable candidate evidence")
    state = _load_object(root / "state.json")
    if state.get("phase1_status") not in {"accepted", "provisional"}:
        raise ValueError("Phase 1 is not frozen")
    if state.get("phase2_status") != "not_run":
        raise ValueError("Phase-1-only run has an invalid Phase 2 status")
    if state.get("investigation_sha256") != investigation_digest:
        raise ValueError("state investigation hash does not match frozen Phase 1")
    if state.get("investigation") != final.model_dump(mode="json"):
        raise ValueError("frozen investigation does not match state")
    if any((root / "phase2").glob("**/*-model-response.json")):
        raise ValueError("Phase-1-only run contains a Phase 2 model call")
    target = final.episode_slug
    registry = state.get("evidence_registry")
    if not isinstance(registry, dict):
        raise ValueError("v4.2 evidence registry is invalid")
    if any(
        isinstance(row, dict) and row.get("episode_slug") == target
        for row in registry.values()
    ):
        raise ValueError("target appears in v4.2 evidence registry")
    for investigation, label in ((candidate, "candidate"), (final, "final")):
        cited = {
            evidence_id
            for rationale in investigation.rationales
            for evidence_id in [
                *rationale.wiki_evidence_ids,
                *rationale.historical_evidence_ids,
            ]
        }
        cited.update(
            evidence_id
            for question in investigation.questions
            for evidence_id in question.evidence_ids
        )
        cited.update(
            evidence_id
            for observation in investigation.unmapped_observations
            for evidence_id in observation.evidence_ids
        )
        cited.update(
            evidence_id
            for constraint in investigation.constraint_assessments
            for evidence_id in [
                *constraint.wiki_evidence_ids,
                *constraint.historical_evidence_ids,
            ]
        )
        if not cited <= set(registry):
            raise ValueError(f"v4.2 {label} investigation cites inaccessible evidence")
    for response_path in root.glob("**/*model-response*.json"):
        prompt = _load_object(response_path).get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("v4.2 response artifact lacks exact prompt")
        if "accessible_episode_inventory" in prompt:
            raise ValueError("v4.2 prompt contains registry inventory noise")


def _verify_phase1_v43_artifacts(root: Path) -> None:
    from .schemas_v4 import InvestigationV41, InvestigationV43, RationaleMappingV43

    candidate_digest = verify_frozen(
        root / "phase1/candidate-investigation.json",
        root / "phase1/candidate-investigation.sha256",
    )
    mapping_digest = verify_frozen(
        root / "phase1/rationale-mapping.json",
        root / "phase1/rationale-mapping.sha256",
    )
    investigation_digest = verify_frozen(
        root / "phase1/investigation.json",
        root / "phase1/investigation.sha256",
    )
    candidate = InvestigationV41.model_validate(
        _load_object(root / "phase1/candidate-investigation.json")
    )
    mapping = RationaleMappingV43.model_validate(
        _load_object(root / "phase1/rationale-mapping.json")
    )
    final = InvestigationV43.model_validate(
        _load_object(root / "phase1/investigation.json")
    )
    if mapping.candidate_investigation_sha256 != candidate_digest:
        raise ValueError("rationale mapping does not bind candidate investigation")
    if final.candidate_investigation_sha256 != candidate_digest:
        raise ValueError("final investigation does not bind candidate investigation")
    if final.rationale_mapping_sha256 != mapping_digest:
        raise ValueError("final investigation does not bind rationale mapping")
    candidate_by_id = {row.rationale_id: row for row in candidate.rationales}
    disposition_by_id = {row.rationale_id: row for row in mapping.dispositions}
    if set(candidate_by_id) != set(disposition_by_id):
        raise ValueError("rationale mapping candidate coverage mismatch")
    final_by_id = {row.rationale_id: row for row in final.rationales}
    expected_final = {
        identifier
        for identifier, row in disposition_by_id.items()
        if row.action != "merge"
    }
    if set(final_by_id) != expected_final:
        raise ValueError("final rationale set does not match rationale mapping")
    for identifier, row in final_by_id.items():
        candidate_row = candidate_by_id[identifier]
        disposition = disposition_by_id[identifier]
        if row.taxonomy_label != disposition.target_taxonomy_label:
            raise ValueError("final rationale label does not match mapping")
        merged_sources = [
            candidate_by_id[source_id]
            for source_id, source_mapping in disposition_by_id.items()
            if source_mapping.action == "merge"
            and source_mapping.merge_into_rationale_id == identifier
        ]
        sources = [candidate_row, *merged_sources]
        for field in (
            "pitch_evidence_ids",
            "wiki_evidence_ids",
            "historical_evidence_ids",
        ):
            expected = {
                value for source in sources for value in getattr(source, field)
            }
            if set(getattr(row, field)) != expected:
                raise ValueError("mapped rationale changed immutable candidate evidence")
    state = _load_object(root / "state.json")
    if state.get("phase1_status") not in {"accepted", "provisional"}:
        raise ValueError("Phase 1 is not frozen")
    if state.get("phase2_status") != "not_run":
        raise ValueError("Phase-1-only run has an invalid Phase 2 status")
    if state.get("investigation_sha256") != investigation_digest:
        raise ValueError("state investigation hash does not match frozen Phase 1")
    if state.get("investigation") != final.model_dump(mode="json"):
        raise ValueError("frozen investigation does not match state")
    if any((root / "phase2").glob("**/*-model-response.json")):
        raise ValueError("Phase-1-only run contains a Phase 2 model call")
    registry = state.get("evidence_registry")
    if not isinstance(registry, dict):
        raise ValueError("v4.3 evidence registry is invalid")
    target = final.episode_slug
    if any(
        isinstance(row, dict) and row.get("episode_slug") == target
        for row in registry.values()
    ):
        raise ValueError("target appears in v4.3 evidence registry")
    cited = {
        evidence_id
        for investigation in (candidate, final)
        for rationale in investigation.rationales
        for evidence_id in [
            *rationale.wiki_evidence_ids,
            *rationale.historical_evidence_ids,
        ]
    }
    if not cited <= set(registry):
        raise ValueError("v4.3 investigation cites inaccessible evidence")
    for response_path in root.glob("**/*model-response*.json"):
        prompt = _load_object(response_path).get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("v4.3 response artifact lacks exact prompt")
        if "accessible_episode_inventory" in prompt:
            raise ValueError("v4.3 prompt contains registry inventory noise")


_V44_CALL_PATH = re.compile(
    r"phase1/(?:claim-extraction|adjudication/turn-[0-9]{2})/call-[0-9]{2}\.json"
)
_V44_SHA256 = re.compile(r"[0-9a-f]{64}")
_V44_FORBIDDEN_KEYS = {
    "actual_decision",
    "current_decision",
    "decision",
    "outcome",
    "target_outcome",
    "target_decision",
    "reference_rationale",
    "reference_rationales",
    "previous_prediction",
    "evaluation_report",
    "evaluation_payload",
    "model_memory",
}
_V44_CALL_PAYLOAD_KEYS = {
    "phase",
    "prompt",
    "schema",
    "raw",
    "parsed",
    "provider_metadata",
    "usage",
    "cost_usd",
    "elapsed_seconds",
    "max_output_tokens",
    "reasoning_effort",
}
_V44_ZERO_USAGE = {
    "input_tokens": 0,
    "cached_input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0.0,
}
_V44_STATE_KEYS = {
    "contract_version",
    "episode_slug",
    "run_fingerprint",
    "run_thread_id",
    "call_records",
    "pending_call",
    "pending_response",
    "repair_context",
    "phase1_iteration",
    "phase1_status",
    "phase2_status",
    "phase1_findings",
    "targeted_retrieval_ids",
    "evidence_registry",
    "events",
    "usage",
    "usage_by_phase",
    "phase1_action",
    "claim_map",
    "claim_map_sha256",
    "claim_retrieval",
    "claim_retrieval_manifest",
    "claim_retrieval_sha256",
    "taxonomy_neighborhood",
    "taxonomy_neighborhood_manifest",
    "taxonomy_neighborhood_sha256",
    "adjudication",
    "adjudication_sha256",
    "last_valid_adjudication",
    "investigation",
    "investigation_sha256",
}
_V44_OWNER_KEYS = {
    "schema_version",
    "thread_id",
    "run_fingerprint",
    "fingerprint_payload",
}
_V44_FINGERPRINT_KEYS = {
    "schema_version",
    "workflow_settings",
    "pitch",
    "pitch_evidence",
    "phase1_v44_settings",
    "wiki_index",
    "precedent",
    "portfolio",
    "provider",
}
_V44_WORKFLOW_KEYS = {
    "contract_version",
    "execution_mode",
    "episode_slug",
    "investor_name",
    "taxonomy_records",
    "taxonomy_labels",
    "check_tiers",
    "phase1_min_iterations",
    "phase1_max_iterations",
    "retrieval_top_k",
    "max_exact_reads",
    "phase1_max_output_tokens",
    "phase1_reasoning_effort",
    "phase1_planning_max_output_tokens",
    "phase1_planning_reasoning_effort",
    "precedent_manifest_sha256",
    "accessible_precedent_count",
    "phase1_max_precedent_searches",
    "phase1_max_precedent_reads",
    "allow_full_transcript",
    "precedent_selection_policy",
    "precedent_candidate_pool_k",
    "precedent_in_slots",
    "precedent_out_slots",
    "portfolio_memory_enabled",
    "portfolio_retrieval_top_k",
    "portfolio_candidate_pool_k",
}
_V44_SETTINGS_KEYS = {
    "claim_retrieval_top_k",
    "max_wiki_reads_per_claim",
    "max_precedent_reads_per_claim",
    "taxonomy_top_k",
    "max_revisits",
}
_V44_WIKI_IDENTITY_KEYS = {
    "root",
    "embedding",
    "embedding_index",
    "require_complete_embeddings",
    "chunks",
}


def _verify_v44_safe_tree(root: Path) -> None:
    try:
        root_mode = root.lstat().st_mode
    except OSError as exc:
        raise ValueError("v4.4 run root is inaccessible") from exc
    if stat.S_ISLNK(root_mode):
        raise ValueError("v4.4 verification rejects a symlink run root")
    if not stat.S_ISDIR(root_mode):
        raise ValueError("v4.4 run root is not a directory")
    for directory, names, files in os.walk(root, followlinks=False):
        for name in [*names, *files]:
            path = Path(directory) / name
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                raise ValueError("v4.4 artifact tree is inaccessible") from exc
            if stat.S_ISLNK(mode):
                raise ValueError(f"v4.4 verification rejects symlink artifact: {path}")
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError(f"v4.4 verification rejects special artifact: {path}")


def _verify_v44_inventory(root: Path, call_paths: set[str]) -> None:
    allowed = {
        "run-owner-v44.json",
        "state.json",
        "phase1/claim-map.json",
        "phase1/claim-map.sha256",
        "phase1/claim-retrieval.json",
        "phase1/claim-retrieval.sha256",
        "phase1/evidence-registry.json",
        "phase1/taxonomy-neighborhood.json",
        "phase1/taxonomy-neighborhood.sha256",
        "phase1/adjudication.json",
        "phase1/adjudication.sha256",
        "phase1/investigation.json",
        "phase1/investigation.sha256",
        *call_paths,
    }
    optional = {
        "run-config.json",
        "input-provenance.json",
        "wiki-sanitization.json",
        "precedent-manifest.filtered.json",
        "portfolio-manifest.filtered.json",
    }
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    unexpected = actual - allowed - optional
    if unexpected:
        if any("phase2" in {part.casefold() for part in Path(item).parts} for item in unexpected):
            raise ValueError("v4.4 Phase 2 artifact is forbidden")
        raise ValueError(
            "v4.4 artifact inventory contains unexpected files: "
            + ", ".join(sorted(unexpected))
        )

    allowed_directories = {".phase1-v44-wal", "phase1"}
    for relative in call_paths:
        parent = Path(relative).parent
        while parent != Path("."):
            allowed_directories.add(parent.as_posix())
            parent = parent.parent
    actual_directories = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir()
    }
    unexpected_directories = actual_directories - allowed_directories
    if unexpected_directories:
        raise ValueError(
            "v4.4 artifact inventory contains unexpected directories: "
            + ", ".join(sorted(unexpected_directories))
        )

    for relative in actual:
        if not relative.endswith(".json"):
            continue
        try:
            value = json.loads((root / relative).read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON in v4.4 artifact inventory: {relative}") from exc
        if _v44_forbidden_payload_key(value):
            raise ValueError(
                "v4.4 artifact inventory contains forbidden target/evaluation fields"
            )
        if _v44_semantic_leakage(value, artifact=relative):
            raise ValueError("v4.4 artifact inventory contains semantic leakage")
        if not (
            _v44_secret_values_safe(value)
            if relative in optional
            else _v44_secret_safe(value)
        ):
            raise ValueError("v4.4 artifact inventory contains secret credentials")


def _load_v44_object(path: Path, *, canonical: bool = False) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid v4.4 artifact: {path}") from exc
    if type(value) is not dict:
        raise ValueError(f"v4.4 artifact must be a JSON object: {path}")
    if canonical and raw != canonical_bytes(value):
        raise ValueError(f"v4.4 artifact bytes are not canonical: {path}")
    return value


def _load_v44_array(path: Path, *, canonical: bool = False) -> list[Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid v4.4 artifact: {path}") from exc
    if type(value) is not list:
        raise ValueError(f"v4.4 artifact must be a JSON array: {path}")
    if canonical and raw != canonical_bytes(value):
        raise ValueError(f"v4.4 artifact bytes are not canonical: {path}")
    return value


def _v44_digest(value: BaseModel | dict[str, Any]) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def _v44_model_bytes(value: BaseModel) -> bytes:
    return json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _v44_openai_secret(value: str) -> bool:
    if re.search(r"(?i)\bsk-(?:live|proj)-[a-z0-9_-]{8,}\b", value) or re.search(
        r"(?i)\bsk-or-v1-[a-z0-9_-]{8,}\b", value
    ):
        return True
    return re.search(r"\bsk-[A-Za-z0-9]{20,}\b", value) is not None


def _v44_secret_safe(value: object) -> bool:
    if type(value) is dict:
        for key, nested in value.items():
            lowered = str(key).casefold().replace("-", "_")
            if any(
                marker in lowered
                for marker in (
                    "secret",
                    "password",
                    "authorization",
                    "credential",
                    "bearer",
                    "api_key",
                    "apikey",
                )
            ):
                return False
            if not _v44_secret_safe(nested):
                return False
        return True
    if type(value) is list:
        return all(_v44_secret_safe(item) for item in value)
    if type(value) is str:
        lowered = value.casefold()
        return not (
            re.search(r"://[^/?#\s]+:[^@/?#\s]+@", value)
            or re.search(r"[?&](?:api[_-]?key|token|secret)=", lowered)
            or re.search(r"(?i)\b(?:bearer|basic)\s+[a-z0-9._~+/=-]{8,}", value)
            or _v44_openai_secret(value)
            or re.search(r"\bAKIA[A-Z0-9]{16}\b", value)
            or re.search(r"(?i)\bgh[pousr]_[a-z0-9]{20,}\b", value)
            or re.search(r"(?i)\bxox[baprs]-[a-z0-9-]{20,}\b", value)
            or re.search(
                r"\beyJ[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\."
                r"[a-zA-Z0-9_-]{8,}\b",
                value,
            )
            or re.search(r"(?i)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", value)
            or re.search(
                r"(?i)(?<![a-z0-9_-])(?:authorization|api[_-]?key|token|secret)"
                r"\s*[:=]\s*"
                r"[^\s,;]{8,}",
                value,
            )
        )
    return True


def _v44_secret_values_safe(value: object) -> bool:
    if type(value) is dict:
        return all(_v44_secret_values_safe(nested) for nested in value.values())
    if type(value) is list:
        return all(_v44_secret_values_safe(nested) for nested in value)
    if type(value) is str:
        return _v44_secret_safe(value)
    return True


def _v44_pitch_prose_context(artifact: str, path: tuple[object, ...]) -> bool:
    if artifact == "run-owner-v44.json":
        return path == ("fingerprint_payload", "pitch") or (
            len(path) == 4
            and path[:2] == ("fingerprint_payload", "pitch_evidence")
            and type(path[2]) is int
            and path[3] == "text"
        )
    if artifact.startswith("phase1/") and artifact.endswith(".json"):
        return path == ("prompt",)
    return artifact == "state.json" and path[-1:] == ("prompt",) and (
        "call_records" in path
    )


def _v44_semantic_leakage(
    value: object,
    *,
    artifact: str,
    path: tuple[object, ...] = (),
) -> bool:
    if type(value) is dict:
        return any(
            _v44_semantic_leakage(
                nested,
                artifact=artifact,
                path=(*path, key),
            )
            for key, nested in value.items()
        )
    if type(value) is list:
        return any(
            _v44_semantic_leakage(
                nested,
                artifact=artifact,
                path=(*path, index),
            )
            for index, nested in enumerate(value)
        )
    if type(value) is not str:
        return False
    normalized = " ".join(re.sub(r"[_-]+", " ", value.casefold()).split())
    if any(
        re.search(pattern, normalized)
        for pattern in (
            r"\bactual\s+(?:current\s+)?target\s+decision\b",
            r"\bcurrent\s+target\s+decision\b",
            r"\btarget\s+(?:decision|outcome)\b",
            r"\breference\s+rationales?\b",
            r"\bevaluation\s+(?:report|payload|result|score)\b",
            r"\bprevious\s+prediction\b",
        )
    ):
        return True
    return not _v44_pitch_prose_context(artifact, path) and any(
        re.search(pattern, normalized)
        for pattern in (
            r"\bground\s+truth\s+(?:decision|outcome|label)\b",
            r"\bgold\s+(?:decision|outcome|label)\b",
        )
    )


def _v44_forbidden_payload_key(value: object) -> str | None:
    if type(value) is dict:
        for key, nested in value.items():
            if key in _V44_FORBIDDEN_KEYS:
                return key
            found = _v44_forbidden_payload_key(nested)
            if found:
                return found
    elif type(value) is list:
        for nested in value:
            found = _v44_forbidden_payload_key(nested)
            if found:
                return found
    return None


def _v44_usage(value: object, *, context: str) -> dict[str, int | float]:
    if type(value) is not dict or set(value) != set(_USAGE_KEYS):
        raise ValueError(f"v4.4 {context} usage is invalid")
    result: dict[str, int | float] = {}
    for key in _USAGE_KEYS[:-1]:
        item = value.get(key)
        if type(item) is not int or item < 0:
            raise ValueError(f"v4.4 {context} usage is invalid")
        result[key] = item
    cost = value.get("cost_usd")
    if type(cost) not in {int, float} or not math.isfinite(cost) or cost < 0:
        raise ValueError(f"v4.4 {context} usage is invalid")
    result["cost_usd"] = float(cost)
    return result


def _verify_v44_filtered_inputs(
    root: Path, owner_payload: dict[str, Any], target: str
) -> None:
    precedent = owner_payload.get("precedent")
    if precedent is not None:
        if type(precedent) is not dict:
            raise ValueError("v4.4 precedent fingerprint is invalid")
        manifest = _load_v44_object(root / "precedent-manifest.filtered.json")
        if manifest != precedent.get("filtered_manifest"):
            raise ValueError("v4.4 precedent manifest does not match fingerprint")
        rows = manifest.get("accessible_episodes")
        if (
            manifest.get("target_episode_slug") != target
            or type(rows) is not list
            or any(type(row) is dict and row.get("episode_slug") == target for row in rows)
        ):
            raise ValueError("target appears in v4.4 precedent corpus")
    portfolio = owner_payload.get("portfolio")
    if portfolio is not None:
        if type(portfolio) is not dict:
            raise ValueError("v4.4 portfolio fingerprint is invalid")
        manifest = _load_v44_object(root / "portfolio-manifest.filtered.json")
        if manifest != portfolio.get("filtered_manifest"):
            raise ValueError("v4.4 portfolio manifest does not match fingerprint")
        target_number_text = target.split("-", 1)[0]
        target_number = int(target_number_text) if target_number_text.isdecimal() else None
        for row in portfolio.get("disclosures", []):
            if (
                type(row) is not dict
                or row.get("source_episode_slug") == target
                or target_number is None
                or type(row.get("source_episode_number")) is not int
                or row["source_episode_number"] >= target_number
            ):
                raise ValueError("v4.4 portfolio temporal eligibility mismatch")


def _verify_v44_upstream_evidence(
    owner_payload: dict[str, Any],
    retrieval: BaseModel,
    registry_payload: list[dict[str, Any]],
    target: str,
) -> None:
    from .portfolio_memory import (
        PortfolioDisclosure,
        PortfolioEntity,
        make_disclosure_id,
    )
    from .precedents import FilteredPrecedentManifest

    wiki = owner_payload["wiki_index"]
    chunks = wiki.get("chunks")
    embedding = wiki.get("embedding")
    embedding_index = wiki.get("embedding_index")
    if (
        type(chunks) is not list
        or not chunks
        or type(embedding) is not dict
        or set(embedding)
        != {
            "backend",
            "model",
            "revision",
            "normalize",
            "document_prefix",
            "query_prefix",
        }
        or type(embedding_index) is not dict
        or set(embedding_index)
        != {
            *embedding,
            "embedded_count",
            "total_count",
            "coverage",
            "dimension",
        }
    ):
        raise ValueError("v4.4 wiki source/index fingerprint is incomplete")
    if any(embedding_index[key] != value for key, value in embedding.items()):
        raise ValueError("v4.4 wiki embedding index identity is contradictory")
    chunk_by_id: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    dimension = embedding_index.get("dimension")
    for chunk in chunks:
        if (
            type(chunk) is not dict
            or set(chunk)
            != {
                "chunk_id",
                "source_path",
                "source_sha256",
                "heading",
                "text",
                "embedding",
            }
            or not all(
                type(chunk.get(key)) is str and chunk[key]
                for key in ("chunk_id", "source_path", "source_sha256", "heading", "text")
            )
            or _V44_SHA256.fullmatch(chunk["source_sha256"]) is None
            or chunk["chunk_id"] in chunk_by_id
        ):
            raise ValueError("v4.4 wiki chunk fingerprint is invalid or duplicated")
        vector = chunk.get("embedding")
        if (
            type(vector) is not list
            or type(dimension) is not int
            or dimension < 1
            or len(vector) != dimension
            or any(
                type(value) not in {int, float} or not math.isfinite(value)
                for value in vector
            )
        ):
            raise ValueError("v4.4 wiki chunk embedding coverage is invalid")
        prior_hash = source_hashes.setdefault(chunk["source_path"], chunk["source_sha256"])
        if prior_hash != chunk["source_sha256"]:
            raise ValueError("v4.4 wiki source path has contradictory hashes")
        expected_chunk_id = "W-" + sha256(
            f"{chunk['source_path']}\0{chunk['text']}".encode("utf-8")
        ).hexdigest()[:20]
        if chunk["chunk_id"] != expected_chunk_id:
            raise ValueError("v4.4 wiki chunk identity does not match source text")
        chunk_by_id[chunk["chunk_id"]] = chunk
    if (
        embedding_index.get("total_count") != len(chunks)
        or embedding_index.get("embedded_count") != len(chunks)
        or embedding_index.get("coverage") != 1
    ):
        raise ValueError("v4.4 wiki embedding index coverage is contradictory")

    registry_by_kind: dict[str, list[dict[str, Any]]] = {
        "wiki": [],
        "historical": [],
        "portfolio": [],
    }
    for row in registry_payload:
        registry_by_kind[row["source_kind"]].append(row)
    for row in registry_by_kind["wiki"]:
        chunk = chunk_by_id.get(row["evidence_id"])
        if chunk is None or any(
            row[field] != expected
            for field, expected in (
                ("source_locator", chunk["source_path"]),
                ("source_sha256", chunk["source_sha256"]),
                ("text", chunk["text"]),
            )
        ):
            raise ValueError("v4.4 wiki evidence does not match owner chunk fingerprint")

    precedent = owner_payload.get("precedent")
    historical = registry_by_kind["historical"]
    if historical and type(precedent) is not dict:
        raise ValueError("v4.4 historical evidence lacks owner precedent provenance")
    if type(precedent) is dict:
        if set(precedent) != {
            "target_episode_slug",
            "filtered_manifest",
            "embedding",
            "embedding_index",
        } or precedent.get("target_episode_slug") != target:
            raise ValueError("v4.4 precedent fingerprint has an invalid shape")
        precedent_embedding = precedent.get("embedding")
        precedent_index = precedent.get("embedding_index")
        if (
            type(precedent_embedding) is not dict
            or type(precedent_index) is not dict
            or any(
                precedent_index.get(key) != value
                for key, value in precedent_embedding.items()
            )
            or precedent_index.get("coverage") != 1
            or precedent_index.get("embedded_count")
            != precedent_index.get("total_count")
        ):
            raise ValueError("v4.4 precedent embedding fingerprint is contradictory")
        try:
            manifest = FilteredPrecedentManifest.model_validate_json(
                json.dumps(precedent.get("filtered_manifest")), strict=True
            )
        except ValidationError as exc:
            raise ValueError("v4.4 precedent fingerprint manifest is invalid") from exc
        manifest_payload = manifest.model_dump(mode="json")
        digest_payload = {key: value for key, value in manifest_payload.items() if key != "sha256"}
        if manifest.sha256 != sha256(
            json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest():
            raise ValueError("v4.4 precedent fingerprint manifest hash is invalid")
        episodes = {row.episode_slug: row for row in manifest.accessible_episodes}
        if len(episodes) != len(manifest.accessible_episodes):
            raise ValueError("v4.4 precedent fingerprint contains duplicate episodes")
        for row in historical:
            episode = episodes.get(row["episode_slug"])
            if (
                episode is None
                or row["source_sha256"] != episode.source_sha256
                or row["decision_status"] != episode.decision_status
                or row["episode_slug"] == target
            ):
                raise ValueError("v4.4 historical evidence contradicts precedent fingerprint")

    portfolio = owner_payload.get("portfolio")
    portfolio_rows = registry_by_kind["portfolio"]
    if portfolio_rows and type(portfolio) is not dict:
        raise ValueError("v4.4 portfolio evidence lacks owner disclosure provenance")
    if type(portfolio) is dict:
        if set(portfolio) != {
            "filtered_manifest",
            "disclosures",
            "entities",
            "embedding",
        }:
            raise ValueError("v4.4 portfolio fingerprint has an invalid shape")
        raw_disclosures = portfolio.get("disclosures")
        raw_entities = portfolio.get("entities")
        if type(raw_disclosures) is not list or type(raw_entities) is not list:
            raise ValueError("v4.4 portfolio fingerprint provenance is incomplete")
        try:
            disclosures = [
                PortfolioDisclosure.model_validate_json(json.dumps(row), strict=True)
                for row in raw_disclosures
            ]
            entities = [
                PortfolioEntity.model_validate_json(json.dumps(row), strict=True)
                for row in raw_entities
            ]
        except ValidationError as exc:
            raise ValueError("v4.4 portfolio fingerprint provenance is invalid") from exc
        disclosure_by_id = {row.disclosure_id: row for row in disclosures}
        if len(disclosure_by_id) != len(disclosures) or any(
            row.disclosure_id
            != make_disclosure_id(
                row.vc_slug,
                row.source_episode_slug,
                min(item.turn_index for item in row.evidence),
                row.company_name or row.descriptor,
            )
            for row in disclosures
        ):
            raise ValueError("v4.4 portfolio fingerprint has invalid disclosure IDs")
        known_ids = set(disclosure_by_id)
        if any(not set(entity.disclosure_ids) <= known_ids for entity in entities):
            raise ValueError("v4.4 portfolio entity references an unknown disclosure")
        portfolio_embedding = portfolio.get("embedding")
        if (
            type(portfolio_embedding) is not dict
            or portfolio_embedding.get("model") != embedding.get("model")
            or portfolio_embedding.get("revision") != embedding.get("revision")
            or any(
                entity.embedding is None
                or not entity.embedding
                or any(not math.isfinite(value) for value in entity.embedding)
                for entity in entities
            )
        ):
            raise ValueError("v4.4 portfolio embedding fingerprint is contradictory")
        manifest = portfolio.get("filtered_manifest")
        if (
            type(manifest) is not dict
            or manifest.get("target_episode_slug") != target
            or manifest.get("eligible_disclosure_ids")
            != [row.disclosure_id for row in disclosures]
        ):
            raise ValueError("v4.4 portfolio filtered manifest contradicts disclosures")
        manifest_payload = {
            key: value
            for key, value in manifest.items()
            if key not in {"schema_version", "sha256"}
        }
        expected_manifest_hash = sha256(
            (
                json.dumps(
                    manifest_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        if manifest.get("sha256") != expected_manifest_hash:
            raise ValueError("v4.4 portfolio filtered manifest hash is invalid")
        for row in portfolio_rows:
            disclosure = disclosure_by_id.get(row["evidence_id"])
            if disclosure is None:
                raise ValueError("v4.4 portfolio evidence is absent from owner disclosures")
            evidence = tuple(disclosure.evidence)
            hashes = {item.source_sha256 for item in evidence}
            indexes = sorted({item.turn_index for item in evidence})
            contiguous = indexes == list(range(indexes[0], indexes[-1] + 1))
            locator = (
                f"{disclosure.source_episode_slug}:turns-"
                + (
                    f"{indexes[0]}-{indexes[-1]}"
                    if contiguous and len(indexes) > 1
                    else ",".join(str(value) for value in indexes)
                )
            )
            if len(hashes) != 1 or any(
                row[field] != expected
                for field, expected in (
                    ("source_locator", locator),
                    ("source_sha256", next(iter(hashes))),
                    ("text", "\n".join(item.text for item in evidence)),
                )
            ):
                raise ValueError("v4.4 portfolio evidence contradicts owner disclosure")


def _verify_v44_optional_artifacts(
    root: Path,
    owner_payload: dict[str, Any],
    state: dict[str, Any],
    *,
    provenance_mode: str,
    expected_config: Any | None,
    expected_package: Any | None,
) -> None:
    from .config import RunConfig

    if provenance_mode == "cli":
        if expected_config is None or expected_package is None:
            raise ValueError("v4.4 CLI verification requires config and package identity")
        required = {
            "run-config.json",
            "input-provenance.json",
            "wiki-sanitization.json",
        }
        if owner_payload.get("precedent") is not None:
            required.add("precedent-manifest.filtered.json")
        if owner_payload.get("portfolio") is not None:
            required.add("portfolio-manifest.filtered.json")
        missing = sorted(relative for relative in required if not (root / relative).is_file())
        if missing:
            raise ValueError(
                "v4.4 CLI audit artifact is missing: " + ", ".join(missing)
            )
    config: RunConfig | None = None
    run_config_path = root / "run-config.json"
    if run_config_path.is_file():
        raw_config = _load_v44_object(run_config_path)
        if (
            provenance_mode == "cli"
            and raw_config != expected_config.model_dump(mode="json")
        ):
            raise ValueError("v4.4 run-config does not match supplied CLI config")
        try:
            config = RunConfig.model_validate(raw_config, strict=True)
        except ValidationError as exc:
            raise ValueError("v4.4 run-config artifact is invalid") from exc
        provider = dict(owner_payload["provider"])
        provider["class"] = _canonical_v44_cli_provider_class(
            str(provider.get("class", ""))
        )
        workflow = owner_payload["workflow_settings"]
        embedding = owner_payload["wiki_index"]["embedding"]
        effective_model = config.phase1.model or config.provider.model
        expected_workflow = {
            "contract_version": config.run.contract_version,
            "execution_mode": config.run.mode,
            "episode_slug": config.run.episode_slug,
            "phase1_min_iterations": config.phase1.min_iterations,
            "phase1_max_iterations": config.phase1.max_iterations,
            "retrieval_top_k": config.retrieval.top_k,
            "max_exact_reads": config.retrieval.max_exact_reads,
            "phase1_max_output_tokens": (
                config.phase1.max_output_tokens or config.provider.max_output_tokens
            ),
            "phase1_reasoning_effort": config.phase1.reasoning_effort,
            "phase1_planning_max_output_tokens": (
                config.phase1.planning_max_output_tokens
            ),
            "phase1_planning_reasoning_effort": (
                config.phase1.planning_reasoning_effort
            ),
            "phase1_max_precedent_searches": config.phase1.max_precedent_searches,
            "phase1_max_precedent_reads": config.phase1.max_precedent_reads,
            "allow_full_transcript": config.precedents.allow_full_transcript,
            "precedent_selection_policy": config.precedents.selection_policy,
            "precedent_candidate_pool_k": config.precedents.candidate_pool_k,
            "precedent_in_slots": config.precedents.in_slots,
            "precedent_out_slots": config.precedents.out_slots,
            "portfolio_memory_enabled": config.portfolio_memory.enabled,
            "portfolio_retrieval_top_k": config.portfolio_memory.retrieval_top_k,
            "portfolio_candidate_pool_k": config.portfolio_memory.candidate_pool_k,
        }
        expected_provider = {
            "schema_version": "phase1-provider-fingerprint-v1",
            "provider": config.provider.kind,
            "class": _V44_CLI_PROVIDER_CLASS,
            "model": effective_model,
            "execution_settings_sha256": v44_execution_settings_sha256(config),
        }
        if config.embedding is None:
            raise ValueError("v4.4 run-config does not match owner fingerprint")
        expected_embedding = {
            "backend": config.embedding.kind,
            "model": config.embedding.model,
            "revision": config.embedding.revision,
            "normalize": config.embedding.normalize,
            "document_prefix": config.embedding.document_prefix,
            "query_prefix": config.embedding.query_prefix,
        }
        if (
            config.run.contract_version != "v4.4"
            or config.run.mode != "phase1_only"
            or config.run.episode_slug != state["episode_slug"]
            or state.get("run_thread_id")
            != f"{config.run.vc_slug}:{config.run.episode_slug}"
            or config.phase1_v44 is None
            or config.phase1_v44.model_dump(mode="json")
            != owner_payload["phase1_v44_settings"]
            or any(workflow.get(key) != value for key, value in expected_workflow.items())
            or provider != expected_provider
            or embedding != expected_embedding
            or bool(owner_payload["precedent"] is not None)
            != config.precedents.enabled
            or bool(owner_payload["portfolio"] is not None)
            != config.portfolio_memory.enabled
        ):
            raise ValueError("v4.4 run-config does not match owner fingerprint")

    provenance_path = root / "input-provenance.json"
    if provenance_path.is_file():
        provenance = _load_v44_object(provenance_path)
        if config is None:
            raise ValueError("v4.4 input provenance requires its run-config")
        try:
            if provenance_mode == "cli":
                package = expected_package
            else:
                from .firewall import verify_package

                package = verify_package(
                    config.resolve_path(config.run.input_root),
                    config.run.vc_slug,
                    config.run.episode_slug,
                    taxonomy_path=config.run.taxonomy_path,
                )
            pitch_bytes = package.pitch.read_bytes()
            manifest_bytes = package.manifest.read_bytes()
            normalized_pitch = package.pitch.read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            raise ValueError("v4.4 input package provenance cannot be verified") from exc
        if (
            set(provenance)
            != {
                "pitch_path",
                "pitch_sha256",
                "package_manifest_path",
                "package_manifest_sha256",
            }
            or type(provenance.get("pitch_path")) is not str
            or Path(provenance["pitch_path"]) != package.pitch
            or type(provenance.get("package_manifest_path")) is not str
            or Path(provenance["package_manifest_path"]) != package.manifest
            or provenance.get("pitch_sha256")
            != sha256(pitch_bytes).hexdigest()
            or provenance.get("package_manifest_sha256")
            != sha256(manifest_bytes).hexdigest()
            or normalized_pitch != owner_payload["pitch"]
        ):
            raise ValueError("v4.4 input-provenance artifact is invalid or stale")

    sanitization_path = root / "wiki-sanitization.json"
    if sanitization_path.is_file():
        sanitization = _load_v44_object(sanitization_path)
        if set(sanitization) != {
            "alias_hashes",
            "changed_chunk_ids",
            "changed_chunk_count",
            "reused_embedding_count",
            "recomputed_embedding_count",
            "lexical_only_chunk_ids",
            "quality_findings",
        }:
            raise ValueError("v4.4 wiki-sanitization artifact has an invalid shape")
        lists = (
            "alias_hashes",
            "changed_chunk_ids",
            "lexical_only_chunk_ids",
            "quality_findings",
        )
        counts = (
            "changed_chunk_count",
            "reused_embedding_count",
            "recomputed_embedding_count",
        )
        known_chunks = {
            row["chunk_id"] for row in owner_payload["wiki_index"]["chunks"]
        }
        if (
            any(type(sanitization.get(key)) is not list for key in lists)
            or any(
                type(sanitization.get(key)) is not int or sanitization[key] < 0
                for key in counts
            )
            or sanitization["changed_chunk_count"]
            != len(sanitization["changed_chunk_ids"])
            or not set(sanitization["changed_chunk_ids"]) <= known_chunks
            or not set(sanitization["lexical_only_chunk_ids"])
            <= set(sanitization["changed_chunk_ids"])
        ):
            raise ValueError("v4.4 wiki-sanitization artifact is invalid or stale")

    for relative, key in (
        ("precedent-manifest.filtered.json", "precedent"),
        ("portfolio-manifest.filtered.json", "portfolio"),
    ):
        if (root / relative).is_file() and owner_payload.get(key) is None:
            raise ValueError(f"v4.4 unexpected {key} filtered manifest artifact")


def _verify_phase1_v44_artifacts(
    root: Path,
    *,
    provenance_mode: str,
    expected_config: Any | None,
    expected_package: Any | None,
) -> None:
    from .adjudication_postprocess_v44 import (
        finalize_adjudication_v44,
        normalize_adjudication_payload_v44,
    )
    from .phase1_v44 import (
        ClaimMapV44,
        ClaimRetrievalManifestV44,
        InvestigationV44,
        RationaleAdjudicationV44,
        TaxonomyNeighborhoodManifestV44,
        adjudication_findings_v44,
        adjudication_v44_json_schema,
        build_investigation_v44,
        claim_map_v44_json_schema,
    )
    from .prompts_v44 import (
        claim_extraction_repair_v44_prompt,
        claim_extraction_v44_prompt,
        rationale_adjudication_repair_v44_prompt,
        rationale_adjudication_v44_prompt,
    )
    from .providers.base import strict_provider_schema
    from .workflow_v44 import (
        BoundSchemaValidationError,
        _SequenceSet,
        _eligible_ids,
        _targeted_ids_from_findings,
        _validate_claim_map_runtime,
        _validate_json_model,
        _validation_errors,
    )

    _verify_v44_safe_tree(root)
    phase1 = root / "phase1"
    frozen_specs = {
        "claim_map": ("claim-map", ClaimMapV44),
        "claim_retrieval": ("claim-retrieval", ClaimRetrievalManifestV44),
        "taxonomy_neighborhood": (
            "taxonomy-neighborhood",
            TaxonomyNeighborhoodManifestV44,
        ),
        "adjudication": ("adjudication", RationaleAdjudicationV44),
        "investigation": ("investigation", InvestigationV44),
    }
    models: dict[str, BaseModel] = {}
    digests: dict[str, str] = {}
    for state_key, (name, model_type) in frozen_specs.items():
        json_path = phase1 / f"{name}.json"
        sha_path = phase1 / f"{name}.sha256"
        try:
            raw, digest = read_verified_frozen(json_path, sha_path)
            if state_key == "investigation":
                unchecked = json.loads(raw)
                if (
                    type(unchecked) is dict
                    and unchecked.get("rationales")
                    != unchecked.get("core_rationales")
                ):
                    raise ValueError(
                        "v4.4 rationales must be the exact core compatibility view"
                    )
            model = model_type.model_validate_json(raw, strict=True)
        except Exception as exc:
            if "core compatibility view" in str(exc):
                raise
            if isinstance(exc, ValueError) and "hash mismatch" in str(exc):
                raise
            raise ValueError(f"invalid frozen v4.4 {name} artifact") from exc
        if raw != _v44_model_bytes(model):
            raise ValueError(f"frozen v4.4 {name} bytes are not canonical")
        if sha_path.read_bytes() != (digest + "\n").encode("ascii"):
            raise ValueError(f"frozen v4.4 {name} hash bytes are not canonical")
        models[state_key] = model
        digests[state_key] = digest

    claim_map = models["claim_map"]
    retrieval = models["claim_retrieval"]
    neighborhood = models["taxonomy_neighborhood"]
    adjudication = models["adjudication"]
    investigation = models["investigation"]
    assert isinstance(claim_map, ClaimMapV44)
    assert isinstance(retrieval, ClaimRetrievalManifestV44)
    assert isinstance(neighborhood, TaxonomyNeighborhoodManifestV44)
    assert isinstance(adjudication, RationaleAdjudicationV44)
    assert isinstance(investigation, InvestigationV44)

    state = _load_v44_object(root / "state.json", canonical=True)
    owner = _load_v44_object(root / "run-owner-v44.json", canonical=True)
    if set(state) != _V44_STATE_KEYS:
        raise ValueError("v4.4 state.json does not have the exact terminal shape")
    if state.get("contract_version") != "v4.4":
        raise ValueError("v4.4 state contract binding is invalid")
    if state.get("phase1_status") not in {"accepted", "provisional"}:
        raise ValueError("v4.4 Phase 1 is not frozen")
    if state.get("phase2_status") != "not_run":
        raise ValueError("v4.4 Phase 2 status must be not_run")
    if state.get("pending_call") is not None or state.get("pending_response") is not None:
        raise ValueError("v4.4 terminal state retains a pending call/response")
    wal = root / ".phase1-v44-wal"
    if not wal.is_dir() or any(wal.iterdir()):
        raise ValueError("v4.4 provider-call WAL is missing or unsettled")

    if not _v44_secret_safe(owner):
        raise ValueError("v4.4 run owner is not secret-safe")
    if set(owner) != _V44_OWNER_KEYS or (
        owner.get("schema_version") != "phase1-v4.4-run-owner-v1"
        or owner.get("thread_id") != state.get("run_thread_id")
        or owner.get("run_fingerprint") != state.get("run_fingerprint")
        or type(owner.get("fingerprint_payload")) is not dict
    ):
        raise ValueError("v4.4 run owner binding mismatch")
    fingerprint_payload = owner["fingerprint_payload"]
    if set(fingerprint_payload) != _V44_FINGERPRINT_KEYS:
        raise ValueError("v4.4 run fingerprint payload does not have the exact shape")
    expected_fingerprint = sha256(canonical_bytes(fingerprint_payload)).hexdigest()
    if expected_fingerprint != state.get("run_fingerprint"):
        raise ValueError("v4.4 run fingerprint mismatch")
    if (
        fingerprint_payload.get("schema_version")
        != "phase1-v4.4-run-fingerprint-v1"
    ):
        raise ValueError("v4.4 run fingerprint payload is invalid")
    workflow_identity = fingerprint_payload.get("workflow_settings")
    if (
        type(workflow_identity) is not dict
        or set(workflow_identity) != _V44_WORKFLOW_KEYS
        or workflow_identity.get("contract_version") != "v4.4"
        or workflow_identity.get("execution_mode") != "phase1_only"
        or workflow_identity.get("episode_slug") != state.get("episode_slug")
    ):
        raise ValueError("v4.4 run fingerprint settings are stale")
    provider_identity = fingerprint_payload.get("provider")
    if (
        type(provider_identity) is not dict
        or provider_identity.get("schema_version")
        != "phase1-provider-fingerprint-v1"
        or not provider_identity.get("provider")
        or not provider_identity.get("model")
    ):
        raise ValueError("v4.4 provider/model fingerprint metadata is missing")
    v44_settings = fingerprint_payload.get("phase1_v44_settings")
    wiki_identity = fingerprint_payload.get("wiki_index")
    if type(v44_settings) is not dict or set(v44_settings) != _V44_SETTINGS_KEYS:
        raise ValueError("v4.4 settings fingerprint does not have the exact shape")
    if type(wiki_identity) is not dict or set(wiki_identity) != _V44_WIKI_IDENTITY_KEYS:
        raise ValueError("v4.4 wiki fingerprint does not have the exact shape")
    taxonomy_records = workflow_identity.get("taxonomy_records")
    if type(taxonomy_records) is not list or not taxonomy_records:
        raise ValueError("v4.4 taxonomy fingerprint records are missing")
    expected_taxonomy_hash = sha256(
        json.dumps(
            taxonomy_records,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    embedding_identity = wiki_identity.get("embedding")
    embedding_index = wiki_identity.get("embedding_index")
    if (
        type(embedding_identity) is not dict
        or type(embedding_index) is not dict
        or embedding_identity.get("model") != embedding_index.get("model")
        or embedding_identity.get("revision") != embedding_index.get("revision")
        or wiki_identity.get("require_complete_embeddings") is not True
    ):
        raise ValueError("v4.4 embedding fingerprint identity is invalid")
    forbidden = _v44_forbidden_payload_key(
        {"state": state, "fingerprint_payload": fingerprint_payload}
    )
    if forbidden:
        raise ValueError("v4.4 artifacts contain forbidden target/evaluation payload fields")

    target = investigation.episode_slug
    if any(
        value != target
        for value in (
            state.get("episode_slug"),
            claim_map.episode_slug,
            retrieval.episode_slug,
            adjudication.episode_slug,
        )
    ):
        raise ValueError("v4.4 episode bindings are inconsistent")
    if retrieval.claim_map_sha256 != digests["claim_map"]:
        raise ValueError("v4.4 claim retrieval does not bind claim-map hash")
    if neighborhood.taxonomy_sha256 != expected_taxonomy_hash:
        raise ValueError("v4.4 taxonomy hash does not match owner fingerprint")
    if (
        neighborhood.embedding_model != embedding_identity.get("model")
        or neighborhood.embedding_revision != embedding_identity.get("revision")
    ):
        raise ValueError("v4.4 taxonomy embedding identity does not match fingerprint")
    embedded_hashes = {
        "claim_map_sha256": digests["claim_map"],
        "claim_retrieval_sha256": digests["claim_retrieval"],
        "taxonomy_neighborhood_sha256": digests["taxonomy_neighborhood"],
        "adjudication_sha256": digests["adjudication"],
    }
    if any(
        getattr(investigation, key) != digest
        for key, digest in embedded_hashes.items()
    ):
        raise ValueError("v4.4 investigation contains a stale artifact hash")
    if (
        investigation.claim_retrieval_manifest.model_dump(mode="json")
        != retrieval.model_dump(mode="json")
        or investigation.taxonomy_neighborhood_manifest.model_dump(mode="json")
        != neighborhood.model_dump(mode="json")
    ):
        raise ValueError("v4.4 investigation embeds stale retrieval artifacts")
    for key, model in models.items():
        if state.get(key) != model.model_dump(mode="json"):
            raise ValueError(f"v4.4 frozen {key} does not exactly match state")
        if state.get(f"{key}_sha256") != digests[key]:
            raise ValueError(f"v4.4 state {key} hash mismatch")
    if state.get("claim_retrieval_manifest") != state.get("claim_retrieval"):
        raise ValueError("v4.4 claim retrieval state aliases disagree")
    if state.get("taxonomy_neighborhood_manifest") != state.get(
        "taxonomy_neighborhood"
    ):
        raise ValueError("v4.4 taxonomy state aliases disagree")
    if state.get("last_valid_adjudication") != state.get("adjudication"):
        raise ValueError("v4.4 last-valid adjudication state is stale")
    if state.get("repair_context") is not None:
        raise ValueError("v4.4 terminal state retains repair context")
    if state["phase1_status"] == "accepted" and state.get("phase1_findings") != []:
        raise ValueError("v4.4 accepted state must not retain validator findings")

    candidate = [row.model_dump(mode="json") for row in investigation.candidate_rationales]
    core = [row.model_dump(mode="json") for row in investigation.core_rationales]
    compatibility = [row.model_dump(mode="json") for row in investigation.rationales]
    candidate_by_id = {row["rationale_id"]: row for row in candidate}
    core_by_id = {row["rationale_id"]: row for row in core}
    if (
        len(candidate_by_id) != len(candidate)
        or len(core_by_id) != len(core)
        or not set(core_by_id) <= set(candidate_by_id)
        or any(candidate_by_id[key] != row for key, row in core_by_id.items())
        or compatibility != core
    ):
        raise ValueError("v4.4 rationales must be the exact core compatibility view")

    pitch_rows = fingerprint_payload.get("pitch_evidence")
    if type(pitch_rows) is not list or not pitch_rows:
        raise ValueError("v4.4 pitch evidence fingerprint is missing")
    pitch_ids = [row.get("evidence_id") for row in pitch_rows if type(row) is dict]
    if (
        len(pitch_ids) != len(pitch_rows)
        or len(set(pitch_ids)) != len(pitch_ids)
        or pitch_ids != [f"P-{index:03d}" for index in range(1, len(pitch_ids) + 1)]
        or [str(row.get("text")) for row in pitch_rows]
        != [
            line.strip()
            for line in str(fingerprint_payload.get("pitch", "")).splitlines()
            if line.strip()
        ]
    ):
        raise ValueError("v4.4 pitch evidence fingerprint is invalid")
    expected = build_investigation_v44(
        episode_slug=target,
        claim_map=claim_map,
        adjudication=adjudication,
        retrieval_manifest=retrieval,
        neighborhood=neighborhood,
        claim_map_sha256=digests["claim_map"],
        adjudication_sha256=digests["adjudication"],
        pitch_evidence_ids=pitch_ids,
    )
    if state["phase1_status"] == "provisional":
        expected = expected.model_copy(
            update={
                "investigation_status": "provisional",
                "validator_findings": tuple(
                    sorted(
                        {
                            *expected.validator_findings,
                            *state.get("phase1_findings", []),
                        }
                    )
                ),
            }
        )
    if expected.model_dump(mode="json") != investigation.model_dump(mode="json"):
        raise ValueError("v4.4 dual-view or disposition separation is inconsistent")
    if investigation.investigation_status != state["phase1_status"]:
        raise ValueError("v4.4 investigation status does not match state")

    registry_payload = [row.model_dump(mode="json") for row in retrieval.evidence_registry]
    if _load_v44_array(phase1 / "evidence-registry.json", canonical=True) != registry_payload:
        raise ValueError("v4.4 evidence registry artifact is stale")
    registry = {row["evidence_id"]: row for row in registry_payload}
    if state.get("evidence_registry") != registry:
        raise ValueError("v4.4 evidence registry state is stale")
    _verify_v44_upstream_evidence(
        fingerprint_payload,
        retrieval,
        registry_payload,
        target,
    )
    if any(row.get("episode_slug") == target for row in registry_payload):
        raise ValueError("target appears in v4.4 evidence registry")
    for row in registry_payload:
        evidence_id = row["evidence_id"]
        if row["source_kind"] == "wiki":
            expected_id = "W-" + sha256(
                f"{row['source_locator']}\0{row['text']}".encode("utf-8")
            ).hexdigest()[:20]
        elif row["source_kind"] == "historical":
            expected_id = _historical_identity(
                row["source_sha256"], row["turn_start"], row["turn_end"], row["text"]
            )
        else:
            continue
        if evidence_id != expected_id:
            raise ValueError("v4.4 evidence registry identity binding is invalid")
    accessible_portfolio = {
        row.evidence_id
        for bundle in retrieval.claim_bundles
        for row in bundle.portfolio_disclosures
    }
    cited_investor: set[str] = set()
    cited_portfolio: set[str] = set()
    cited_pitch: set[str] = set()
    for row in [*investigation.candidate_rationales, *investigation.unmapped_observations]:
        cited_pitch.update(row.pitch_evidence_ids)
        cited_investor.update(row.wiki_evidence_ids)
        cited_investor.update(row.historical_evidence_ids)
        cited_portfolio.update(row.portfolio_disclosure_ids)
    if not cited_pitch <= set(pitch_ids):
        raise ValueError("v4.4 investigation cites inaccessible pitch evidence")
    if not cited_investor <= set(registry):
        raise ValueError("v4.4 investigation cites inaccessible investor evidence")
    if not cited_portfolio <= accessible_portfolio:
        raise ValueError("v4.4 investigation cites inaccessible portfolio evidence")
    for bundle in retrieval.claim_bundles:
        if any(row.episode_slug == target for row in bundle.historical_evidence):
            raise ValueError("target appears in v4.4 precedent retrieval")
        if any(
            row.source_locator.startswith(f"{target}:")
            for row in bundle.portfolio_disclosures
        ):
            raise ValueError("target appears in v4.4 portfolio retrieval")
    if any(
        target in action.opened_episode_slugs for action in retrieval.retrieval_actions
    ):
        raise ValueError("target appears in v4.4 retrieval actions")
    if not retrieval.target_episode_excluded:
        raise ValueError("v4.4 retrieval does not attest target exclusion")
    allowed_labels = set(neighborhood.ordered_labels)
    for row in adjudication.dispositions:
        if row.taxonomy_label not in allowed_labels:
            raise ValueError("v4.4 disposition is outside taxonomy neighborhood")
    _verify_v44_filtered_inputs(root, fingerprint_payload, target)

    records = state.get("call_records")
    if type(records) is not list or not records:
        raise ValueError("v4.4 provider-call record set is missing")
    expected_paths: set[str] = set()
    total_usage: dict[str, int | float] = dict(_V44_ZERO_USAGE)
    claim_calls: list[tuple[str, dict[str, Any]]] = []
    adjudication_calls: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    events = state.get("events")
    if type(events) is not list or [row.get("sequence") for row in events if type(row) is dict] != list(
        range(1, len(events) + 1)
    ):
        raise ValueError("v4.4 event sequence is invalid")
    for record in records:
        if type(record) is not dict or set(record) != {
            "relative_path",
            "sha256",
            "payload",
            "wal_id",
            "request_sha256",
        }:
            raise ValueError("v4.4 provider-call record binding is invalid")
        relative = record["relative_path"]
        payload = record["payload"]
        if (
            type(relative) is not str
            or _V44_CALL_PATH.fullmatch(relative) is None
            or relative in expected_paths
            or type(payload) is not dict
            or set(payload) != _V44_CALL_PAYLOAD_KEYS
        ):
            raise ValueError("v4.4 provider-call record/path is invalid")
        expected_paths.add(relative)
        raw = canonical_bytes(payload)
        path = root / relative
        try:
            artifact_raw = path.read_bytes()
        except OSError as exc:
            raise ValueError("v4.4 call artifact set is missing a record") from exc
        if artifact_raw != raw or record.get("sha256") != sha256(raw).hexdigest():
            raise ValueError("v4.4 call record hash/bytes mismatch")
        request_payload = {
            key: payload[key]
            for key in (
                "phase",
                "prompt",
                "schema",
                "max_output_tokens",
                "reasoning_effort",
            )
        }
        request_digest = sha256(canonical_bytes(request_payload)).hexdigest()
        if record.get("request_sha256") != request_digest:
            raise ValueError("v4.4 call record request hash mismatch")
        wal_identity = {
            "run_fingerprint": state["run_fingerprint"],
            "thread_id": state["run_thread_id"],
            "relative_call_path": relative,
            "request_sha256": request_digest,
        }
        if record.get("wal_id") != sha256(canonical_bytes(wal_identity)).hexdigest():
            raise ValueError("v4.4 call record WAL binding mismatch")
        if (
            type(payload["prompt"]) is not str
            or not payload["prompt"]
            or type(payload["schema"]) is not dict
            or type(payload["raw"]) is not str
            or type(payload["parsed"]) not in {dict, list}
            or type(payload["max_output_tokens"]) is not int
            or payload["max_output_tokens"] < 1
            or payload["reasoning_effort"] not in {None, "low", "medium", "high"}
            or type(payload["elapsed_seconds"]) not in {int, float}
            or not math.isfinite(payload["elapsed_seconds"])
            or payload["elapsed_seconds"] < 0
            or type(payload["cost_usd"]) not in {int, float}
            or isinstance(payload["cost_usd"], bool)
            or not math.isfinite(payload["cost_usd"])
            or payload["cost_usd"] < 0
        ):
            raise ValueError("v4.4 call payload metadata/timing is invalid")
        metadata = payload["provider_metadata"]
        if (
            type(metadata) is not dict
            or not metadata.get("provider")
            or not (
                metadata.get("model")
                or metadata.get("returned_model")
                or metadata.get("requested_model")
            )
        ):
            raise ValueError("v4.4 call lacks provider/model metadata")
        if not _v44_secret_safe(metadata):
            raise ValueError("v4.4 call provider metadata is not secret-safe")
        fingerprint_model = provider_identity.get("model")
        if metadata.get("provider") != provider_identity.get("provider"):
            raise ValueError("v4.4 call provider identity does not match fingerprint")
        requested_model = metadata.get("requested_model")
        returned_model = metadata.get("returned_model")
        reported_model = metadata.get("model")
        if (
            provider_identity.get("provider") == "openrouter"
            and requested_model is None
        ):
            raise ValueError("v4.4 OpenRouter call lacks requested model binding")
        if requested_model is not None:
            if (
                type(requested_model) is not str
                or not requested_model
                or requested_model != fingerprint_model
                or type(returned_model or reported_model) is not str
                or not (returned_model or reported_model)
                or (
                    returned_model is not None
                    and reported_model is not None
                    and returned_model != reported_model
                )
            ):
                raise ValueError(
                    "v4.4 call model metadata does not match provider fingerprint"
                )
        else:
            metadata_models = [
                value
                for value in (reported_model, returned_model)
                if value is not None
            ]
            if not metadata_models or any(
                type(value) is not str
                or not value
                or value != fingerprint_model
                for value in metadata_models
            ):
                raise ValueError(
                    "v4.4 call model metadata does not match provider fingerprint"
                )
        try:
            if json.loads(payload["raw"]) != payload["parsed"]:
                raise ValueError("v4.4 call raw/parsed payload mismatch")
        except json.JSONDecodeError as exc:
            raise ValueError("v4.4 call raw payload is not exact JSON") from exc
        usage = _v44_usage(payload["usage"], context="provider-call")
        if float(payload["cost_usd"]) != float(usage["cost_usd"]):
            raise ValueError("v4.4 provider-call cost/usage mismatch")
        for key in _USAGE_KEYS[:-1]:
            total_usage[key] = int(total_usage[key]) + int(usage[key])
        total_usage["cost_usd"] = float(total_usage["cost_usd"]) + float(
            usage["cost_usd"]
        )
        forbidden = _v44_forbidden_payload_key(
            {"schema": payload["schema"], "parsed": payload["parsed"]}
        )
        prompt_forbidden = next(
            (
                key
                for key in _V44_FORBIDDEN_KEYS
                if re.search(rf'"{re.escape(key)}"\s*:', payload["prompt"])
            ),
            None,
        )
        if forbidden or prompt_forbidden:
            raise ValueError("v4.4 call contains forbidden target/evaluation payload fields")
        match = re.fullmatch(r"phase1/claim-extraction/call-([0-9]{2})\.json", relative)
        if match:
            claim_calls.append((match.group(1), payload))
        else:
            match = re.fullmatch(
                r"phase1/adjudication/turn-([0-9]{2})/call-([0-9]{2})\.json",
                relative,
            )
            assert match is not None
            adjudication_calls.setdefault(int(match.group(1)), []).append(
                (match.group(2), payload)
            )

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in phase1.rglob("call-*.json")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("v4.4 call artifact set mismatch")
    _verify_v44_inventory(root, expected_paths)
    _verify_v44_optional_artifacts(
        root,
        fingerprint_payload,
        state,
        provenance_mode=provenance_mode,
        expected_config=expected_config,
        expected_package=expected_package,
    )
    if [number for number, _ in claim_calls] not in [["01"], ["01", "02"]]:
        raise ValueError("v4.4 claim call topology is invalid")
    if claim_calls[0][1]["phase"] != "phase1_claim_extraction" or (
        len(claim_calls) == 2
        and claim_calls[1][1]["phase"] != "phase1_claim_extraction_repair"
    ):
        raise ValueError("v4.4 claim call topology is invalid")
    iterations = state.get("phase1_iteration")
    v44_settings = fingerprint_payload.get("phase1_v44_settings")
    max_revisits = (
        v44_settings.get("max_revisits") if type(v44_settings) is dict else None
    )
    if (
        type(max_revisits) is not int
        or max_revisits not in {0, 1}
        or type(iterations) is not int
        or iterations < 1
        or iterations > 1 + max_revisits
    ):
        raise ValueError("v4.4 targeted revisit count is invalid")
    if set(adjudication_calls) != set(range(1, iterations + 1)):
        raise ValueError("v4.4 adjudication call topology is invalid")
    for calls in adjudication_calls.values():
        calls.sort()
        if [number for number, _ in calls] not in [["01"], ["01", "02"]]:
            raise ValueError("v4.4 adjudication call topology is invalid")
        if calls[0][1]["phase"] != "phase1_adjudication" or (
            len(calls) == 2
            and calls[1][1]["phase"] != "phase1_adjudication_repair"
        ):
            raise ValueError("v4.4 adjudication call topology is invalid")
    chronological_paths = [
        f"phase1/claim-extraction/call-{number}.json"
        for number, _payload in sorted(claim_calls)
    ]
    for iteration in range(1, iterations + 1):
        chronological_paths.extend(
            f"phase1/adjudication/turn-{iteration:02d}/call-{number}.json"
            for number, _payload in adjudication_calls[iteration]
        )
    if [record["relative_path"] for record in records] != chronological_paths:
        raise ValueError("v4.4 provider calls are not in exact chronological order")
    investor_name = workflow_identity["investor_name"]
    expected_max_tokens = workflow_identity["phase1_max_output_tokens"]
    expected_reasoning = workflow_identity["phase1_reasoning_effort"]

    def verify_request(
        payload: dict[str, Any],
        *,
        schema: dict[str, Any],
        prompt: str,
        context: str,
    ) -> None:
        if payload["schema"] != schema:
            raise ValueError(f"v4.4 {context} request schema does not match contract")
        if payload["prompt"] != prompt:
            raise ValueError(f"v4.4 {context} prompt does not match exact reconstruction")
        if (
            payload["max_output_tokens"] != expected_max_tokens
            or payload["reasoning_effort"] != expected_reasoning
        ):
            raise ValueError(f"v4.4 {context} request settings do not match fingerprint")

    claim_schema = strict_provider_schema(
        claim_map_v44_json_schema(
            episode_slug=target,
            pitch_evidence_ids=pitch_ids,
        )
    )
    claim_primary = claim_calls[0][1]
    verify_request(
        claim_primary,
        schema=claim_schema,
        prompt=claim_extraction_v44_prompt(investor_name, target, pitch_rows),
        context="claim extraction",
    )
    try:
        selected_claim = _validate_json_model(
            ClaimMapV44, claim_primary["parsed"], claim_schema
        )
        assert isinstance(selected_claim, ClaimMapV44)
        _validate_claim_map_runtime(selected_claim, pitch_ids)
        if len(claim_calls) != 1:
            raise ValueError("v4.4 claim repair exists after a valid primary response")
    except (ValidationError, BoundSchemaValidationError) as primary_error:
        if len(claim_calls) != 2:
            raise ValueError("v4.4 invalid claim primary lacks its exact repair") from primary_error
        repair = claim_calls[1][1]
        verify_request(
            repair,
            schema=claim_schema,
            prompt=claim_extraction_repair_v44_prompt(
                investor_name,
                target,
                pitch_rows,
                claim_primary["raw"],
                _validation_errors(primary_error),
            ),
            context="claim repair",
        )
        try:
            selected_claim = _validate_json_model(
                ClaimMapV44, repair["parsed"], claim_schema
            )
            assert isinstance(selected_claim, ClaimMapV44)
            _validate_claim_map_runtime(selected_claim, pitch_ids)
        except (ValidationError, BoundSchemaValidationError) as exc:
            raise ValueError("v4.4 terminal claim repair is invalid") from exc
    if selected_claim != claim_map:
        raise ValueError("v4.4 claim calls do not match frozen claim map")

    investor_ids, portfolio_ids = _eligible_ids(retrieval)
    claim_ids = [
        row.claim_id for row in (*claim_map.material_claims, *claim_map.adverse_claims)
    ]
    question_ids = [row.question_id for row in claim_map.unanswered_questions]
    adjudication_schema = strict_provider_schema(
        adjudication_v44_json_schema(
            episode_slug=target,
            claim_ids=claim_ids,
            question_ids=question_ids,
            pitch_evidence_ids=pitch_ids,
            investor_evidence_ids=investor_ids,
            portfolio_disclosure_ids=portfolio_ids,
            taxonomy_labels=neighborhood.ordered_labels,
        )
    )
    prior: RationaleAdjudicationV44 | None = None
    prior_findings: list[str] = []
    terminal: RationaleAdjudicationV44 | None = None
    terminal_findings: list[str] = []
    expected_targeted_ids: list[str] = []
    expected_events: list[dict[str, Any]] = [
        {"sequence": 1, "kind": "phase1_claim_extraction", "status": "valid"}
    ]
    terminal_action: str | None = None

    def append_event(kind: str, **payload: Any) -> None:
        expected_events.append(
            {"sequence": len(expected_events) + 1, "kind": kind, **payload}
        )

    for iteration in range(1, iterations + 1):
        append_event(
            "phase1_claim_retrieval",
            targeted_ids=list(_SequenceSet(expected_targeted_ids)),
        )
        append_event("phase1_taxonomy_retrieval")
        calls = adjudication_calls[iteration]
        primary = calls[0][1]
        verify_request(
            primary,
            schema=adjudication_schema,
            prompt=rationale_adjudication_v44_prompt(
                investor_name,
                target,
                claim_map,
                retrieval,
                neighborhood,
                prior,
                prior_findings,
            ),
            context=f"adjudication turn {iteration}",
        )
        normalized_primary, normalization_audit = (
            normalize_adjudication_payload_v44(primary["parsed"])
        )
        try:
            selected = _validate_json_model(
                RationaleAdjudicationV44,
                normalized_primary,
                adjudication_schema,
            )
            assert isinstance(selected, RationaleAdjudicationV44)
            if len(calls) != 1:
                raise ValueError(
                    "v4.4 adjudication repair exists after a valid primary response"
                )
        except (ValidationError, BoundSchemaValidationError) as primary_error:
            if len(calls) != 2:
                raise ValueError(
                    "v4.4 invalid adjudication primary lacks its exact repair"
                ) from primary_error
            repair = calls[1][1]
            verify_request(
                repair,
                schema=adjudication_schema,
                prompt=rationale_adjudication_repair_v44_prompt(
                    investor_name,
                    target,
                    claim_map,
                    retrieval,
                    neighborhood,
                    primary["raw"],
                    _validation_errors(primary_error),
                    prior,
                    prior_findings,
                ),
                context=f"adjudication repair turn {iteration}",
            )
            normalized_repair, normalization_audit = (
                normalize_adjudication_payload_v44(repair["parsed"])
            )
            try:
                selected = _validate_json_model(
                    RationaleAdjudicationV44,
                    normalized_repair,
                    adjudication_schema,
                )
                assert isinstance(selected, RationaleAdjudicationV44)
            except (ValidationError, BoundSchemaValidationError) as repair_error:
                if prior is None or prior.adjudication_status != "provisional":
                    raise ValueError(
                        "v4.4 terminal adjudication repair is invalid"
                    ) from repair_error
                selected = prior
                terminal_findings = sorted(
                    {
                        *prior.validator_findings,
                        *prior_findings,
                        "ADJUDICATION_REPAIR_FAILED_USING_PRIOR",
                    }
                )
                terminal = selected
                terminal_action = "freeze_provisional"
                append_event("phase1_adjudication", status="prior_preserved")
                continue

        selected, reuse_audit, mapping_audit = finalize_adjudication_v44(
            selected,
            neighborhood,
            retrieval,
        )
        unresolved_findings = {
            f"UNRESOLVED_CONSTRAINT_MAPPING:{constraint_id}"
            for constraint_id in mapping_audit.unresolved_constraint_ids
        }
        findings = sorted(
            {
                *selected.validator_findings,
                *unresolved_findings,
                *adjudication_findings_v44(
                    claim_map,
                    selected,
                    neighborhood,
                    pitch_evidence_ids=pitch_ids,
                    investor_evidence_ids=investor_ids,
                    portfolio_disclosure_ids=portfolio_ids,
                    retrieval_manifest=retrieval,
                ),
            }
        )
        structural = [
            value
            for value in findings
            if value.startswith(
                (
                    "DUPLICATE_",
                    "INACCESSIBLE_",
                    "INVESTOR_EVIDENCE_",
                    "LABEL_OUTSIDE_",
                    "PITCH_EVIDENCE_",
                    "PORTFOLIO_EVIDENCE_",
                    "UNRESOLVED_CONSTRAINT_MAPPING:",
                    "UNKNOWN_",
                )
            )
        ]
        if structural:
            if prior is None:
                raise ValueError("v4.4 first adjudication has structural blockers")
            selected = prior
            findings = sorted(
                {
                    *prior_findings,
                    *structural,
                    "STRUCTURAL_ADJUDICATION_REJECTED_USING_PRIOR",
                }
            )
            requested: list[str] = []
            terminal_action = "freeze_provisional"
            event_status = "prior_preserved"
        else:
            requested = _targeted_ids_from_findings(claim_map, selected, findings)
            if requested and iteration <= max_revisits:
                terminal_action = "targeted_retrieval"
                event_status = "provisional"
            elif requested or selected.adjudication_status == "provisional" or findings:
                terminal_action = "freeze_provisional"
                event_status = "provisional"
            else:
                terminal_action = "freeze_accepted"
                event_status = "valid"
        if requested:
            expected_targeted_ids = list(requested)
        append_event(
            "phase1_adjudication",
            status=event_status,
            normalized_nonactivating_dispositions=(
                normalization_audit.cleared_nonactivating_dispositions
            ),
            exact_duplicate_dispositions_removed=(
                normalization_audit.exact_duplicate_dispositions_removed
            ),
            removed_disposition_positions=list(
                normalization_audit.removed_disposition_positions
            ),
            cross_target_evidence_reuse=[
                {
                    "disposition_position": row.disposition_position,
                    "taxonomy_label": row.taxonomy_label,
                    "target_ids": list(row.target_ids),
                    "evidence_kind": row.evidence_kind,
                    "evidence_id": row.evidence_id,
                }
                for row in reuse_audit
            ],
            constraint_mapping_revisions=[
                {
                    "constraint_id": row.constraint_id,
                    "original_mapped_ids": list(row.original_mapped_ids),
                    "recomputed_mapped_ids": list(row.recomputed_mapped_ids),
                }
                for row in mapping_audit.constraint_mappings
            ],
        )
        terminal = selected
        terminal_findings = findings
        prior = selected
        prior_findings = findings

        if iteration < iterations and terminal_action != "targeted_retrieval":
            raise ValueError("v4.4 adjudication topology continued without a revisit")
        if iteration == iterations and terminal_action == "targeted_retrieval":
            raise ValueError("v4.4 adjudication topology froze before its targeted revisit")

    if terminal is None or adjudication != terminal:
        raise ValueError("v4.4 frozen adjudication is not the terminal valid turn")
    if state.get("phase1_findings") != terminal_findings:
        raise ValueError("v4.4 terminal validator findings do not match adjudication")
    assert terminal_action is not None
    expected_phase1_status = (
        "accepted" if terminal_action == "freeze_accepted" else "provisional"
    )
    append_event("phase1_freeze", status=expected_phase1_status)
    if events != expected_events:
        raise ValueError("v4.4 calls do not match exact event/state topology")
    if (
        state.get("phase1_iteration") != iterations
        or state.get("phase1_action") != terminal_action
        or state.get("targeted_retrieval_ids") != expected_targeted_ids
        or state.get("phase1_status") != expected_phase1_status
    ):
        raise ValueError("v4.4 terminal action/targeted state does not match replay")
    if state.get("usage") != total_usage:
        raise ValueError("v4.4 provider-call usage does not match state")
    by_phase = state.get("usage_by_phase")
    if (
        type(by_phase) is not dict
        or by_phase.get("phase1") != total_usage
        or by_phase.get("phase2") != _V44_ZERO_USAGE
    ):
        raise ValueError("v4.4 usage mismatch or Phase 2 usage is nonzero")

    phase2_paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            "phase2" in {part.casefold() for part in path.relative_to(root).parts}
            or re.search(r"phase[-_]?2.*(?:call|response)", path.name, re.I)
        )
    ]
    if phase2_paths or any(
        type(record.get("payload")) is dict
        and str(record["payload"].get("phase", "")).startswith("phase2")
        for record in records
    ):
        raise ValueError("v4.4 Phase 2 response/call artifact is forbidden")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return value


def _historical_identity(source_sha256: str, start: int, end: int, text: str) -> str:
    material = f"{source_sha256}\0{start}\0{end}\0{text}".encode("utf-8")
    return "H-" + sha256(material).hexdigest()[:20]


_AUDIT_ARRAY_NAMES = (
    "wiki-searches.json",
    "wiki-reads.json",
    "precedent-searches.json",
    "precedent-reads.json",
    "historical-evidence.json",
)
_USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "cost_usd")


def _load_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"audit artifact must be a JSON object array: {path}")
    return value


def _same_usage(actual: dict[str, int | float], expected: object) -> bool:
    if not isinstance(expected, dict) or set(expected) != set(_USAGE_KEYS):
        return False
    for key in _USAGE_KEYS[:-1]:
        if type(expected.get(key)) is not int or expected[key] != actual[key]:
            return False
    cost = expected.get("cost_usd")
    return (
        isinstance(cost, (int, float))
        and not isinstance(cost, bool)
        and abs(float(cost) - float(actual["cost_usd"])) <= 1e-9
    )


def _verify_strict_v3_audit(
    root: Path,
    state: dict[str, Any],
    config: dict[str, Any] | None,
) -> None:
    required_roots = {
        "run-config.json": "run config",
        "input-provenance.json": "package provenance",
        "precedent-manifest.filtered.json": "filtered precedent manifest",
    }
    for name, label in required_roots.items():
        if not (root / name).is_file():
            raise ValueError(f"v3 precedent run lacks required {label}")
    if config is None:
        raise ValueError("v3 precedent run lacks required run config")
    provenance = _load_object(root / "input-provenance.json")
    provenance_keys = {
        "pitch_path",
        "pitch_sha256",
        "package_manifest_path",
        "package_manifest_sha256",
    }
    if not provenance_keys.issubset(provenance):
        raise ValueError("v3 precedent run package provenance is incomplete")

    expected_responses: set[Path] = set()
    phase_artifacts: dict[str, list[dict[str, list[dict[str, Any]]]]] = {}
    for phase, terminal_step in (
        ("phase1", "investigation"),
        ("phase2", "decision"),
    ):
        iterations = state.get(f"{phase}_iteration")
        if type(iterations) is not int or iterations < 1:
            raise ValueError(f"v3 precedent run has invalid {phase} iteration count")
        phase_artifacts[phase] = []
        for turn in range(1, iterations + 1):
            turn_root = root / phase / f"turn-{turn:02d}"
            loaded: dict[str, list[dict[str, Any]]] = {}
            for name in _AUDIT_ARRAY_NAMES:
                path = turn_root / name
                if not path.is_file():
                    raise ValueError(f"v3 precedent run lacks required audit artifact: {path}")
                loaded[name] = _load_array(path)
            phase_artifacts[phase].append(loaded)
            expected_responses.add(turn_root / "plan-model-response.json")
            expected_responses.add(turn_root / f"{terminal_step}-model-response.json")

    actual_responses = set(root.glob("phase*/turn-*/*-model-response.json"))
    if actual_responses != expected_responses:
        raise ValueError("v3 precedent run model-response artifact set is not exact")

    phase_usage: dict[str, dict[str, int | float]] = {}
    provider_kind = config.get("provider", {}).get("kind")
    for phase in ("phase1", "phase2"):
        summed: dict[str, int | float] = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }
        for path in sorted(
            path for path in expected_responses if path.parts[-3] == phase
        ):
            response = _load_object(path)
            metadata = response.get("raw_metadata")
            if (
                not isinstance(metadata, dict)
                or not metadata.get("provider")
                or not metadata.get("model")
            ):
                raise ValueError("generation artifact lacks provider/model metadata")
            if provider_kind == "openrouter" and metadata.get("provider") != "openrouter":
                raise ValueError("OpenRouter generation artifact lacks provider/model metadata")
            usage = response.get("usage")
            if not isinstance(usage, dict) or set(usage) != set(_USAGE_KEYS):
                raise ValueError("generation artifact usage is invalid")
            for key in _USAGE_KEYS[:-1]:
                value = usage.get(key)
                if type(value) is not int or value < 0:
                    raise ValueError("generation artifact usage is invalid")
                summed[key] = int(summed[key]) + value
            cost = usage.get("cost_usd")
            if (
                not isinstance(cost, (int, float))
                or isinstance(cost, bool)
                or cost < 0
            ):
                raise ValueError("generation artifact usage is invalid")
            summed["cost_usd"] = float(summed["cost_usd"]) + float(cost)
        phase_usage[phase] = summed

    state_by_phase = state.get("usage_by_phase")
    if not isinstance(state_by_phase, dict) or set(state_by_phase) != {
        "phase1",
        "phase2",
    }:
        raise ValueError("artifact usage does not match state usage_by_phase")
    for phase in ("phase1", "phase2"):
        if not _same_usage(phase_usage[phase], state_by_phase.get(phase)):
            raise ValueError("artifact usage does not match state usage_by_phase")
    total: dict[str, int | float] = {
        key: sum(int(phase_usage[phase][key]) for phase in ("phase1", "phase2"))
        for key in _USAGE_KEYS[:-1]
    }
    total["cost_usd"] = sum(
        float(phase_usage[phase]["cost_usd"]) for phase in ("phase1", "phase2")
    )
    if not _same_usage(total, state.get("usage")):
        raise ValueError("artifact usage does not match state total usage")

    for phase in ("phase1", "phase2"):
        turns = phase_artifacts[phase]
        prefix = "" if phase == "phase1" else "phase2_"
        queries = list(
            dict.fromkeys(
                row["query"]
                for turn in turns
                for row in turn["wiki-searches.json"]
                if isinstance(row.get("query"), str)
            )
        )
        if queries != state.get(f"{prefix}query_history", []):
            raise ValueError("wiki search artifacts do not match recorded state")
        if turns[-1]["wiki-reads.json"] != state.get(
            "exact_reads" if phase == "phase1" else "phase2_exact_reads", []
        ):
            raise ValueError("wiki read artifacts do not match recorded state")
        for artifact_name, state_suffix in (
            ("precedent-searches.json", "precedent_searches"),
            ("precedent-reads.json", "precedent_reads"),
        ):
            rows = [row for turn in turns for row in turn[artifact_name]]
            if rows != state.get(f"{phase}_{state_suffix}", []):
                label = state_suffix.replace("_", " ")
                raise ValueError(f"{label} artifacts do not match recorded state")
        if turns[-1]["historical-evidence.json"] != state.get(
            f"{phase}_historical_evidence", []
        ):
            raise ValueError("historical evidence artifacts do not match recorded state")


def verify_v3_phase1_replay_source(
    run_root: Path, state: dict[str, Any]
) -> None:
    """Verify the complete frozen Phase 1 audit before starting a v3 replay."""
    root = Path(run_root)
    iterations = state.get("phase1_iteration")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("v3 replay source has invalid Phase 1 iteration count")

    turns: list[dict[str, list[dict[str, Any]]]] = []
    expected_responses: set[Path] = set()
    for turn in range(1, iterations + 1):
        turn_root = root / "phase1" / f"turn-{turn:02d}"
        loaded: dict[str, list[dict[str, Any]]] = {}
        for name in _AUDIT_ARRAY_NAMES:
            path = turn_root / name
            if not path.is_file():
                raise ValueError(f"v3 replay source lacks Phase 1 audit artifact: {path}")
            loaded[name] = _load_array(path)
        turns.append(loaded)
        expected_responses.update(
            {
                turn_root / "plan-model-response.json",
                turn_root / "investigation-model-response.json",
            }
        )
    if set((root / "phase1").glob("turn-*/*-model-response.json")) != expected_responses:
        raise ValueError("v3 replay source Phase 1 model-response artifact set is not exact")

    usage: dict[str, int | float] = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    for path in sorted(expected_responses):
        response = _load_object(path)
        metadata = response.get("raw_metadata")
        if (
            not isinstance(metadata, dict)
            or not metadata.get("provider")
            or not metadata.get("model")
        ):
            raise ValueError("v3 replay source response lacks provider/model metadata")
        response_usage = response.get("usage")
        if not isinstance(response_usage, dict) or set(response_usage) != set(_USAGE_KEYS):
            raise ValueError("v3 replay source response usage is invalid")
        for key in _USAGE_KEYS[:-1]:
            value = response_usage.get(key)
            if type(value) is not int or value < 0:
                raise ValueError("v3 replay source response usage is invalid")
            usage[key] = int(usage[key]) + value
        cost = response_usage.get("cost_usd")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0:
            raise ValueError("v3 replay source response usage is invalid")
        usage["cost_usd"] = float(usage["cost_usd"]) + float(cost)
    if not _same_usage(usage, state.get("usage_by_phase", {}).get("phase1")):
        raise ValueError("v3 replay source Phase 1 usage does not match artifacts")

    queries = list(
        dict.fromkeys(
            row["query"]
            for turn in turns
            for row in turn["wiki-searches.json"]
            if isinstance(row.get("query"), str)
        )
    )
    if queries != state.get("query_history"):
        raise ValueError("v3 replay source wiki search history does not match artifacts")
    if turns[-1]["wiki-reads.json"] != state.get("exact_reads"):
        raise ValueError("v3 replay source wiki read history does not match artifacts")
    for artifact_name, state_key in (
        ("precedent-searches.json", "phase1_precedent_searches"),
        ("precedent-reads.json", "phase1_precedent_reads"),
    ):
        rows = [row for turn in turns for row in turn[artifact_name]]
        if rows != state.get(state_key):
            raise ValueError(f"v3 replay source {state_key} does not match artifacts")
    if turns[-1]["historical-evidence.json"] != state.get(
        "phase1_historical_evidence"
    ):
        raise ValueError("v3 replay source historical registry does not match artifacts")


def verify_run_artifacts(run_root: Path) -> None:
    """Fail closed over frozen outputs, provenance, usage, and exact precedent reads."""
    root = Path(run_root)
    investigation_digest = verify_frozen(
        root / "phase1/investigation.json", root / "phase1/investigation.sha256"
    )
    decision_digest = verify_frozen(
        root / "phase2/decision.json", root / "phase2/decision.sha256"
    )
    state = _load_object(root / "state.json")
    if state.get("phase1_status") not in {"accepted", "provisional"}:
        raise ValueError("Phase 1 is not frozen")
    if state.get("phase2_status") not in {
        "accepted",
        "accepted_with_warnings",
        "provisional",
    }:
        raise ValueError("Phase 2 is not frozen")
    if state.get("investigation_sha256") != investigation_digest:
        raise ValueError("state investigation hash does not match frozen Phase 1")

    config_path = root / "run-config.json"
    run_config = _load_object(config_path) if config_path.is_file() else None
    frozen_investigation = _load_object(root / "phase1/investigation.json")
    frozen_decision = _load_object(root / "phase2/decision.json")
    if frozen_investigation.get("schema_version") in {
        "investigation-v4",
        "investigation-v4.1",
    }:
        from .schemas_v4 import (
            DecisionV4,
            DecisionV41,
            InvestigationV4,
            InvestigationV41,
            TaxonomyReflectionV4,
        )

        is_v41 = frozen_investigation.get("schema_version") == "investigation-v4.1"
        investigation_model = InvestigationV41 if is_v41 else InvestigationV4
        decision_model = DecisionV41 if is_v41 else DecisionV4
        investigation_model.model_validate(frozen_investigation)
        decision_model.model_validate(frozen_decision)
        if state.get("investigation") != frozen_investigation:
            raise ValueError("v4 frozen investigation does not match state")
        if state.get("decision") != frozen_decision:
            raise ValueError("v4 frozen decision does not match state")
        if state.get("decision_sha256") != decision_digest:
            raise ValueError("v4 state decision hash does not match frozen Phase 2")
        target = frozen_investigation["episode_slug"]
        registry = state.get("evidence_registry")
        if not isinstance(registry, dict):
            raise ValueError("v4 evidence registry is invalid")
        for evidence_id, evidence in registry.items():
            if not isinstance(evidence, dict) or evidence.get("evidence_id") != evidence_id:
                raise ValueError("v4 evidence registry binding is invalid")
            if evidence.get("episode_slug") == target:
                raise ValueError("target appears in v4 evidence registry")
        cited = {
            evidence_id
            for rationale in frozen_investigation["rationales"]
            for evidence_id in [
                *rationale["wiki_evidence_ids"],
                *rationale["historical_evidence_ids"],
            ]
        }
        cited.update(
            evidence_id
            for question in frozen_investigation["questions"]
            for evidence_id in question["evidence_ids"]
        )
        cited.update(
            evidence_id
            for observation in frozen_investigation["unmapped_observations"]
            for evidence_id in observation["evidence_ids"]
        )
        if is_v41:
            cited.update(
                evidence_id
                for constraint in frozen_investigation["constraint_assessments"]
                for evidence_id in [
                    *constraint["wiki_evidence_ids"],
                    *constraint["historical_evidence_ids"],
                ]
            )
        if not cited <= set(registry):
            raise ValueError("v4 investigation cites inaccessible evidence")
        for response_path in root.glob("**/*-model-response.json"):
            response = _load_object(response_path)
            prompt = response.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError("v4 response artifact lacks exact prompt")
            if "accessible_episode_inventory" in prompt:
                raise ValueError("v4 prompt contains registry inventory noise")
        for artifact_path in root.glob("phase*/turn-*/historical-evidence.json"):
            for evidence in _load_array(artifact_path):
                if evidence.get("episode_slug") == target:
                    raise ValueError("target appears in v4 historical evidence")
        reflection_path = root / "reflection/taxonomy-proposals.json"
        if reflection_path.is_file():
            reflection = TaxonomyReflectionV4.model_validate(
                _load_object(reflection_path)
            )
            if reflection.decision_sha256 != decision_digest:
                raise ValueError("v4 taxonomy reflection does not bind frozen decision")
        return
    v3_contract = any(
        candidate == "v3" or candidate in {"investigation-v3", "decision-v3"}
        for candidate in (
            (run_config or {}).get("run", {}).get("contract_version"),
            frozen_investigation.get("schema_version"),
            frozen_decision.get("schema_version"),
            state.get("investigation", {}).get("schema_version"),
            state.get("decision", {}).get("schema_version"),
        )
    )
    precedents_enabled = bool(
        (run_config or {}).get("precedents", {}).get("enabled")
        or state.get("precedent_manifest_sha256")
    )
    if v3_contract and precedents_enabled:
        _verify_strict_v3_audit(root, state, run_config)

    target = state.get("investigation", {}).get("episode_slug")
    manifest_path = root / "precedent-manifest.filtered.json"
    if manifest_path.is_file():
        manifest = _load_object(manifest_path)
        supplied = manifest.get("sha256")
        payload = {key: value for key, value in manifest.items() if key != "sha256"}
        actual = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if supplied != actual:
            raise ValueError("precedent manifest hash mismatch")
        if manifest.get("target_episode_slug") != target:
            raise ValueError("precedent manifest target mismatch")
        accessible = manifest.get("accessible_episodes")
        if not isinstance(accessible, list) or any(
            row.get("episode_slug") == target for row in accessible if isinstance(row, dict)
        ):
            raise ValueError("target appears in filtered precedent manifest")
        if state.get("precedent_manifest_sha256") != supplied:
            raise ValueError("state precedent manifest hash mismatch")

    evidence_rows = [
        *state.get("phase1_historical_evidence", []),
        *state.get("phase2_historical_evidence", []),
    ]
    for phase in ("phase1", "phase2"):
        paths = sorted((root / phase).glob("turn-*/historical-evidence.json"))
        expected_phase = state.get(f"{phase}_historical_evidence", [])
        for path in paths:
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or any(row not in expected_phase for row in rows):
                raise ValueError("historical evidence artifact does not match state")
        if paths and json.loads(paths[-1].read_text(encoding="utf-8")) != expected_phase:
            raise ValueError("final historical evidence artifact is incomplete")
    read_rows = [
        *state.get("phase1_precedent_reads", []),
        *state.get("phase2_precedent_reads", []),
    ]
    artifact_read_rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("phase*/turn-*/precedent-reads.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("precedent read artifact is invalid")
        artifact_read_rows.extend(rows)
    if artifact_read_rows and artifact_read_rows != read_rows:
        raise ValueError("precedent read artifacts do not match recorded state")
    artifact_search_rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("phase*/turn-*/precedent-searches.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("precedent search artifact is invalid")
        artifact_search_rows.extend(rows)
        for row in rows:
            if any(
                isinstance(hit, dict) and hit.get("episode_slug") == target
                for hit in row.get("hits", [])
            ):
                raise ValueError("target appears in precedent search content")
    state_search_rows = [
        *state.get("phase1_precedent_searches", []),
        *state.get("phase2_precedent_searches", []),
    ]
    if artifact_search_rows and artifact_search_rows != state_search_rows:
        raise ValueError("precedent search artifacts do not match recorded state")
    opened_evidence: dict[str, list[dict[str, Any]]] = {}
    decisions: dict[str, str] = {}
    for read in read_rows:
        if not isinstance(read, dict):
            continue
        transcript = read.get("transcript")
        decision_bundle = read.get("decision")
        is_v4_bundle = isinstance(transcript, dict) and isinstance(
            decision_bundle, dict
        )
        if read.get("status") != "ok" and not is_v4_bundle:
            continue
        episode_slug = (
            transcript.get("episode_slug") if is_v4_bundle else read.get("episode_slug")
        )
        if episode_slug == target:
            raise ValueError("target appears in successful precedent read")
        nested = read.get("evidence")
        values = nested if isinstance(nested, list) else [nested] if isinstance(nested, dict) else []
        if isinstance(transcript, dict):
            transcript_evidence = transcript.get("evidence")
            transcript_values = (
                transcript_evidence
                if isinstance(transcript_evidence, list)
                else [transcript_evidence]
                if isinstance(transcript_evidence, dict)
                else []
            )
            nested_decision = (
                decision_bundle.get("decision")
                if isinstance(decision_bundle, dict)
                else None
            )
            decision_status = (
                nested_decision.get("status")
                if isinstance(nested_decision, dict)
                else None
            )
            for evidence in transcript_values:
                enriched = dict(evidence)
                if decision_status in {"In", "Out", "unobserved"}:
                    enriched["decision_status"] = decision_status
                values.append(enriched)
        if isinstance(decision_bundle, dict):
            decision_evidence = decision_bundle.get("evidence")
            decision_values = (
                decision_evidence
                if isinstance(decision_evidence, list)
                else [decision_evidence]
                if isinstance(decision_evidence, dict)
                else []
            )
            nested_decision = decision_bundle.get("decision")
            decision_status = (
                nested_decision.get("status")
                if isinstance(nested_decision, dict)
                else None
            )
            for evidence in decision_values:
                enriched = dict(evidence)
                if decision_status in {"In", "Out", "unobserved"}:
                    enriched["decision_status"] = decision_status
                enriched["evidence_role"] = "observed_decision"
                values.append(enriched)
        for evidence in values:
            opened_evidence.setdefault(evidence["evidence_id"], []).append(evidence)
        decision = read.get("decision")
        if isinstance(decision, dict) and decision.get("status") in {"In", "Out", "unobserved"}:
            decisions[str(read.get("episode_slug"))] = decision["status"]
    for evidence in evidence_rows:
        if not isinstance(evidence, dict):
            raise ValueError("historical evidence registry row is invalid")
        expected = _historical_identity(
            evidence.get("source_sha256", ""),
            evidence.get("turn_start"),
            evidence.get("turn_end"),
            evidence.get("text", ""),
        )
        if evidence.get("evidence_id") != expected:
            raise ValueError("historical evidence ID/text binding mismatch")
        if evidence.get("episode_slug") == target:
            raise ValueError("target appears in historical evidence")
        registry_binding = {
            key: value for key, value in evidence.items() if key != "query_id"
        }
        if not any(
            {
                key: value
                for key, value in opened.items()
                if key != "query_id"
            }
            == registry_binding
            for opened in opened_evidence.get(expected, [])
        ):
            raise ValueError("historical evidence does not resolve to an exact recorded read")

    decision = _load_object(root / "phase2/decision.json")
    if decision.get("investigation_sha256") != investigation_digest:
        raise ValueError("frozen decision does not bind frozen Phase 1")
    if decision.get("schema_version") == "decision-v3":
        by_id = {row["evidence_id"]: row for row in evidence_rows}
        for precedent in decision.get("decisive_precedents", []):
            slug = precedent.get("episode_slug")
            if slug == target:
                raise ValueError("target appears in decisive precedents")
            if decisions.get(slug) != precedent.get("observed_decision"):
                raise ValueError("decisive precedent decision is inconsistent")
            if any(by_id.get(evidence_id, {}).get("episode_slug") != slug for evidence_id in precedent.get("historical_evidence_ids", [])):
                raise ValueError("decisive precedent evidence is inconsistent")

    provider_kind = None
    if run_config is not None:
        provider_kind = run_config.get("provider", {}).get("kind")
    generation_paths = list(root.glob("phase*/turn-*/*-model-response.json"))
    if not generation_paths:
        raise ValueError("run contains no generation artifacts")
    for path in generation_paths:
        artifact = _load_object(path)
        metadata = artifact.get("raw_metadata")
        if not isinstance(metadata, dict) or not metadata.get("provider"):
            raise ValueError("generation artifact lacks provider metadata")
        if provider_kind == "openrouter" and (
            metadata.get("provider") != "openrouter"
            or not (metadata.get("model") or metadata.get("returned_model") or metadata.get("requested_model"))
        ):
            raise ValueError("OpenRouter generation artifact lacks provider/model metadata")

    usage = state.get("usage", {})
    usage_by_phase = state.get("usage_by_phase", {})
    if set(usage_by_phase) != {"phase1", "phase2"}:
        raise ValueError("usage_by_phase has invalid shape")
    for key in ("input_tokens", "cached_input_tokens", "output_tokens", "cost_usd"):
        combined = sum(float(usage_by_phase[phase].get(key, 0)) for phase in ("phase1", "phase2"))
        if abs(combined - float(usage.get(key, 0))) > 1e-9:
            raise ValueError("phase usage does not sum to total usage")
