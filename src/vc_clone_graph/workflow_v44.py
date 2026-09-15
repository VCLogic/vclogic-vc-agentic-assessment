"""Checkpointed, evidence-first Phase-1-only workflow for contract v4.4."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Protocol, TypedDict
from urllib.parse import urlsplit, urlunsplit

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from .adjudication_postprocess_v44 import (
    finalize_adjudication_v44,
    normalize_adjudication_payload_v44,
)
from .config import Phase1V44Settings
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
    validate_claim_map_runtime_v44,
)
from .precedents import PrecedentCorpus
from .portfolio_memory import FilteredPortfolioIndex
from .prompts_v44 import (
    claim_extraction_repair_v44_prompt,
    claim_extraction_v44_prompt,
    rationale_adjudication_repair_v44_prompt,
    rationale_adjudication_v44_prompt,
)
from .providers.base import (
    GenerationProvider,
    GenerationRequest,
    GenerationResult,
    embedding_identity,
)
from .retrieval import HybridWikiIndex
from .retrieval_v44 import (
    retrieve_claim_evidence_v44,
    retrieve_taxonomy_neighborhoods_v44,
)


_LOCAL_EMBEDDING_BACKENDS = {
    "ollama",
    "sentence_transformers",
    "test",
    "test-local",
}


class WorkflowSettingsV44Like(Protocol):
    episode_slug: str
    investor_name: str
    pitch: str
    taxonomy_records: tuple[dict[str, str], ...]
    run_root: Path
    contract_version: str
    execution_mode: str
    phase1_max_output_tokens: int | None
    phase1_reasoning_effort: str | None


def _pitch_evidence_index(pitch: str) -> tuple[dict[str, str], ...]:
    lines = [line.strip() for line in pitch.splitlines() if line.strip()]
    if not lines and pitch.strip():
        lines = [pitch.strip()]
    return tuple(
        {"evidence_id": f"P-{position:03d}", "text": line}
        for position, line in enumerate(lines, start=1)
    )


class WorkflowStateV44(TypedDict, total=False):
    contract_version: str
    episode_slug: str
    run_fingerprint: str
    run_thread_id: str
    call_records: list[dict[str, Any]]
    pending_call: dict[str, Any] | None
    pending_response: dict[str, Any] | None
    repair_context: dict[str, Any] | None
    phase1_iteration: int
    phase1_status: str
    phase2_status: str
    phase1_action: str
    claim_map: dict[str, Any]
    claim_map_sha256: str
    claim_retrieval: dict[str, Any]
    claim_retrieval_manifest: dict[str, Any]
    claim_retrieval_sha256: str
    evidence_registry: dict[str, dict[str, Any]]
    taxonomy_neighborhood: dict[str, Any]
    taxonomy_neighborhood_manifest: dict[str, Any]
    taxonomy_neighborhood_sha256: str
    adjudication: dict[str, Any]
    adjudication_sha256: str
    last_valid_adjudication: dict[str, Any]
    investigation: dict[str, Any]
    investigation_sha256: str
    phase1_findings: list[str]
    targeted_retrieval_ids: list[str]
    events: list[dict[str, Any]]
    usage: dict[str, int | float]
    usage_by_phase: dict[str, dict[str, int | float]]


class BoundSchemaValidationError(ValueError):
    """Expected provider-output failure against the exact runtime-bound schema."""


class _SequenceSet(set[str], Sequence[str]):
    """A set-valued targeted-retrieval boundary accepted by the retrieval API."""

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(set.copy(self)))

    def __getitem__(self, index: int | slice) -> str | list[str]:
        return sorted(set.copy(self))[index]


def _model_payload(value: BaseModel) -> dict[str, Any]:
    return value.model_dump(mode="json", warnings=False)


def _model_bytes(value: BaseModel) -> bytes:
    return json.dumps(
        _model_payload(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_value(value: Any) -> Any:
    """Return a JSON-safe deterministic value without inspecting arbitrary state."""
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="json", warnings=False))
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical mappings require string keys")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        rows = [_canonical_value(item) for item in value]
        return sorted(
            rows,
            key=lambda item: json.dumps(
                item, ensure_ascii=True, separators=(",", ":"), sort_keys=True
            ),
        )
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__qualname__}")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _canonical_value(value),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _before_atomic_temp_create() -> None:
    """Test seam immediately before descriptor-relative temporary creation."""


class _ArtifactStore:
    """One held run-root dirfd for all managed I/O and mutation."""

    def __init__(self, root_fd: int, display_root: Path) -> None:
        self.root_fd = root_fd
        self.display_root = display_root

    @classmethod
    def open(cls, root: Path) -> "_ArtifactStore":
        absolute = Path(os.path.abspath(os.fspath(root)))
        current_fd = os.open(
            absolute.anchor,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            for part in absolute.parts[1:]:
                try:
                    next_fd = os.open(
                        part,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_CLOEXEC
                        | os.O_NOFOLLOW,
                        dir_fd=current_fd,
                    )
                except FileNotFoundError:
                    os.mkdir(part, 0o700, dir_fd=current_fd)
                    os.fsync(current_fd)
                    next_fd = os.open(
                        part,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_CLOEXEC
                        | os.O_NOFOLLOW,
                        dir_fd=current_fd,
                    )
                os.close(current_fd)
                current_fd = next_fd
            store = cls(current_fd, absolute)
            store.verify_tree()
            return store
        except OSError as exc:
            os.close(current_fd)
            raise ValueError(
                "run root path contains a symlink or unsafe component"
            ) from exc
        except BaseException:
            os.close(current_fd)
            raise

    def close(self) -> None:
        if self.root_fd >= 0:
            os.close(self.root_fd)
            self.root_fd = -1

    def __enter__(self) -> "_ArtifactStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def _parts(relative: str) -> tuple[str, ...]:
        path = Path(relative)
        if path.is_absolute() or not path.parts or any(
            part in {"", ".", ".."} for part in path.parts
        ):
            raise ValueError("managed artifact path is not a safe relative path")
        return path.parts

    def open_dir(self, relative: str | None = None, *, create: bool = False) -> int:
        current_fd = os.dup(self.root_fd)
        try:
            for part in (() if relative is None else self._parts(relative)):
                try:
                    next_fd = os.open(
                        part,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_CLOEXEC
                        | os.O_NOFOLLOW,
                        dir_fd=current_fd,
                    )
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(part, 0o700, dir_fd=current_fd)
                    os.fsync(current_fd)
                    next_fd = os.open(
                        part,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_CLOEXEC
                        | os.O_NOFOLLOW,
                        dir_fd=current_fd,
                    )
                os.close(current_fd)
                current_fd = next_fd
            return current_fd
        except BaseException:
            os.close(current_fd)
            raise

    def _parent(self, relative: str, *, create: bool = False) -> tuple[int, str]:
        parts = self._parts(relative)
        parent = "/".join(parts[:-1])
        return self.open_dir(parent or None, create=create), parts[-1]

    def atomic_write(self, relative: str, raw: bytes) -> None:
        parent_fd, name = self._parent(relative, create=True)
        temporary = f".{name}.{secrets.token_hex(12)}.tmp"
        descriptor = -1
        try:
            _before_atomic_temp_create()
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(
                temporary,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                removed = False
                try:
                    os.unlink(temporary, dir_fd=parent_fd)
                    removed = True
                except FileNotFoundError:
                    pass
                if removed:
                    os.fsync(parent_fd)
            finally:
                os.close(parent_fd)

    def read_bytes(self, relative: str) -> bytes:
        parent_fd, name = self._parent(relative)
        descriptor = -1
        try:
            descriptor = os.open(
                name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd
            )
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("managed artifact is not a regular file")
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                return handle.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_fd)

    def read_json_object(self, relative: str) -> dict[str, Any]:
        try:
            value = json.loads(self.read_bytes(relative).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid local JSON artifact: {relative}") from exc
        if type(value) is not dict:
            raise ValueError(f"local JSON artifact must contain an object: {relative}")
        return value

    def exists(self, relative: str) -> bool:
        try:
            parent_fd, name = self._parent(relative)
        except FileNotFoundError:
            return False
        try:
            metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("managed artifact is a symlink")
            return True
        except FileNotFoundError:
            return False
        finally:
            os.close(parent_fd)

    def unlink(self, relative: str, *, missing_ok: bool = False) -> None:
        try:
            parent_fd, name = self._parent(relative)
        except FileNotFoundError:
            if missing_ok:
                return
            raise
        try:
            metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ValueError("managed unlink target is unsafe")
            os.unlink(name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileNotFoundError:
            if not missing_ok:
                raise
        finally:
            os.close(parent_fd)

    def list_files(self, relative: str | None = None) -> list[str]:
        try:
            starting_fd = self.open_dir(relative)
        except FileNotFoundError:
            return []
        prefix = "" if relative is None else relative.rstrip("/") + "/"
        files: list[str] = []

        def walk(directory_fd: int, current_prefix: str) -> None:
            for name in os.listdir(directory_fd):
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                nested = f"{current_prefix}{name}"
                if stat.S_ISLNK(metadata.st_mode):
                    raise ValueError(f"managed run artifact is a symlink: {nested}")
                if stat.S_ISDIR(metadata.st_mode):
                    child_fd = os.open(
                        name,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_CLOEXEC
                        | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                    try:
                        walk(child_fd, nested + "/")
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(metadata.st_mode):
                    files.append(nested)
                else:
                    raise ValueError(f"managed run artifact is unsafe: {nested}")

        try:
            walk(starting_fd, prefix)
        finally:
            os.close(starting_fd)
        return sorted(files)

    def verify_tree(self) -> None:
        self.list_files()


def _freeze_model(
    store: _ArtifactStore, root: str, name: str, value: BaseModel
) -> str:
    """Freeze the exact v4.4 canonical bytes used by contract hash bindings."""
    raw = _model_bytes(value)
    digest = sha256(raw).hexdigest()
    store.atomic_write(f"{root}/{name}.json", raw)
    store.atomic_write(f"{root}/{name}.sha256", (digest + "\n").encode("ascii"))
    return digest


def _event(state: WorkflowStateV44, kind: str, **payload: Any) -> list[dict[str, Any]]:
    events = list(state.get("events", []))
    return [
        *events,
        {"sequence": len(events) + 1, "kind": kind, **payload},
    ]


def _validation_errors(exc: Exception) -> list[str]:
    if isinstance(exc, ValidationError):
        errors = [
            f"{'.'.join(str(value) for value in row['loc'])}: {row['msg']}"
            for row in exc.errors(include_url=False)
        ]
    else:
        errors = [str(exc)]
    return [" ".join(value.split())[:512] or "invalid output" for value in errors[:24]]


def _validate_json_model(
    model: type[BaseModel],
    value: object,
    schema: dict[str, Any],
) -> BaseModel:
    if type(value) is not dict:
        raise BoundSchemaValidationError("provider parsed output must be an object")
    _validate_bound_schema(value, schema, schema, path="$")
    return model.model_validate_json(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    )


def _validate_bound_schema(
    value: object,
    schema: dict[str, Any],
    root: dict[str, Any],
    *,
    path: str,
) -> None:
    """Validate the strict provider-schema subset emitted by v4.4 contracts."""
    reference = schema.get("$ref")
    if reference is not None:
        prefix = "#/$defs/"
        if not isinstance(reference, str) or not reference.startswith(prefix):
            raise BoundSchemaValidationError(
                f"runtime schema has unsupported reference at {path}"
            )
        definition = root.get("$defs", {}).get(reference[len(prefix) :])
        if not isinstance(definition, dict):
            raise BoundSchemaValidationError(
                f"runtime schema has unresolved reference at {path}"
            )
        _validate_bound_schema(value, definition, root, path=path)
        return

    declared = schema.get("type")
    allowed_types = [declared] if isinstance(declared, str) else declared
    if allowed_types is not None:
        predicates = {
            "array": lambda item: type(item) is list,
            "boolean": lambda item: type(item) is bool,
            "integer": lambda item: type(item) is int,
            "null": lambda item: item is None,
            "number": lambda item: type(item) in {int, float},
            "object": lambda item: type(item) is dict,
            "string": lambda item: type(item) is str,
        }
        if not isinstance(allowed_types, list) or not any(
            kind in predicates and predicates[kind](value) for kind in allowed_types
        ):
            raise BoundSchemaValidationError(
                f"runtime schema type violation at {path}"
            )
        if value is None:
            return
    if "const" in schema and value != schema["const"]:
        raise BoundSchemaValidationError(f"runtime schema const violation at {path}")
    if "enum" in schema and value not in schema["enum"]:
        raise BoundSchemaValidationError(f"runtime schema enum violation at {path}")
    if type(value) is str and "pattern" in schema:
        if re.search(schema["pattern"], value) is None:
            raise BoundSchemaValidationError(
                f"runtime schema pattern violation at {path}"
            )
    if type(value) in {int, float}:
        if "minimum" in schema and value < schema["minimum"]:
            raise BoundSchemaValidationError(
                f"runtime schema minimum violation at {path}"
            )
        if "maximum" in schema and value > schema["maximum"]:
            raise BoundSchemaValidationError(
                f"runtime schema maximum violation at {path}"
            )
    if type(value) is list:
        if len(value) < schema.get("minItems", 0):
            raise BoundSchemaValidationError(
                f"runtime schema minItems violation at {path}"
            )
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise BoundSchemaValidationError(
                f"runtime schema maxItems violation at {path}"
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_bound_schema(
                    item, item_schema, root, path=f"{path}[{index}]"
                )
    if type(value) is dict:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            raise BoundSchemaValidationError(
                f"runtime schema required violation at {path}: {missing[0]}"
            )
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise BoundSchemaValidationError(
                    f"runtime schema additionalProperties violation at {path}: "
                    f"{extras[0]}"
                )
        for name, nested_schema in properties.items():
            if name in value and isinstance(nested_schema, dict):
                _validate_bound_schema(
                    value[name], nested_schema, root, path=f"{path}.{name}"
                )


def _state_model(model: type[BaseModel], value: dict[str, Any]) -> BaseModel:
    """Restore strict frozen models from JSON-shaped checkpoint state."""
    return model.model_validate_json(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    )


def _zero_usage() -> dict[str, int | float]:
    return {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }


def _require_local_embedder(embedder: object, *, context: str) -> None:
    identity = embedding_identity(embedder)
    backend = str(identity.get("backend") or "").casefold()
    if backend not in _LOCAL_EMBEDDING_BACKENDS:
        raise ValueError(f"{context} must use a local embedding backend")
    if backend == "ollama" and not identity.get("revision"):
        raise ValueError(
            f"{context} rejects unpinned Ollama embeddings; use a pinned local "
            "embedding model and revision"
        )
    if not identity.get("model") or not identity.get("revision"):
        raise ValueError(f"{context} embedding model and revision must be pinned")


def _validate_claim_map_runtime(
    claim_map: ClaimMapV44, pitch_ids: Sequence[str]
) -> None:
    try:
        validate_claim_map_runtime_v44(claim_map, pitch_ids)
    except ValueError as exc:
        raise BoundSchemaValidationError(str(exc)) from exc


def _add_usage(
    state: WorkflowStateV44, result: Any
) -> tuple[dict[str, int | float], dict[str, dict[str, int | float]]]:
    usage = {**_zero_usage(), **state.get("usage", {})}
    for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
        usage[key] = int(usage[key]) + int(getattr(result.usage, key, 0))
    usage["cost_usd"] = float(usage["cost_usd"]) + float(
        getattr(result.usage, "cost_usd", 0.0)
    )
    by_phase = {
        "phase1": {**_zero_usage()},
        "phase2": {**_zero_usage()},
    }
    for phase, values in state.get("usage_by_phase", {}).items():
        by_phase[phase] = {**_zero_usage(), **values}
    for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
        by_phase["phase1"][key] = int(by_phase["phase1"][key]) + int(
            getattr(result.usage, key, 0)
        )
    by_phase["phase1"]["cost_usd"] = float(
        by_phase["phase1"]["cost_usd"]
    ) + float(getattr(result.usage, "cost_usd", 0.0))
    return usage, by_phase


def _eligible_ids(
    manifest: ClaimRetrievalManifestV44,
) -> tuple[list[str], list[str]]:
    investor: set[str] = set()
    portfolio: set[str] = set()
    for bundle in manifest.claim_bundles:
        investor.update(
            row.evidence_id
            for row in (*bundle.wiki_evidence, *bundle.historical_evidence)
            if row.eligible
        )
        portfolio.update(
            row.evidence_id for row in bundle.portfolio_disclosures if row.eligible
        )
    return sorted(investor), sorted(portfolio)


def _targeted_ids_from_findings(
    claim_map: ClaimMapV44,
    adjudication: RationaleAdjudicationV44,
    findings: Sequence[str],
) -> list[str]:
    """Map eligible deterministic findings back to valid retrieval targets."""
    ordered_targets = [
        *(row.claim_id for row in claim_map.material_claims),
        *(row.claim_id for row in claim_map.adverse_claims),
        *(row.question_id for row in claim_map.unanswered_questions),
    ]
    valid_targets = set(ordered_targets)
    selected = {
        target_id
        for target_id in adjudication.requested_retrieval_ids
        if target_id in valid_targets
    }
    rows = [*adjudication.dispositions, *adjudication.unmapped_observations]
    for finding in findings:
        kind, separator, value = finding.partition(":")
        if not separator:
            continue
        if kind == "UNHANDLED_CLAIM" and value in valid_targets:
            selected.add(value)
            continue
        if kind == "LABEL_OUTSIDE_NEIGHBORHOOD":
            matched = [row for row in adjudication.dispositions if row.taxonomy_label == value]
        elif kind in {
            "PITCH_EVIDENCE_NOT_BOUND_TO_TARGET",
            "INVESTOR_EVIDENCE_NOT_RETRIEVED_FOR_TARGET",
            "PORTFOLIO_EVIDENCE_NOT_RETRIEVED_FOR_TARGET",
        }:
            matched = [
                row
                for row in rows
                if value
                in {
                    *row.pitch_evidence_ids,
                    *row.wiki_evidence_ids,
                    *row.historical_evidence_ids,
                    *row.portfolio_disclosure_ids,
                }
            ]
        else:
            matched = []
        for row in matched:
            selected.update(
                target_id
                for target_id in (*row.claim_ids, *row.question_ids)
                if target_id in valid_targets
            )
    return [target_id for target_id in ordered_targets if target_id in selected]


def _merge_retrieval_manifests(
    previous: ClaimRetrievalManifestV44,
    additional: ClaimRetrievalManifestV44,
) -> ClaimRetrievalManifestV44:
    """Merge partial retrieval without discarding prior target evidence or actions."""
    if (
        previous.episode_slug != additional.episode_slug
        or previous.claim_map_sha256 != additional.claim_map_sha256
    ):
        raise ValueError("partial retrieval manifest binding changed")

    registry: dict[str, dict[str, Any]] = {
        row.evidence_id: _model_payload(row) for row in previous.evidence_registry
    }
    for row in additional.evidence_registry:
        payload = _model_payload(row)
        prior = registry.setdefault(row.evidence_id, payload)
        if prior == payload:
            continue
        stable_fields = tuple(field for field in prior if field != "text")
        stable_identity = all(prior[field] == payload[field] for field in stable_fields)
        prior_text = prior["text"]
        additional_text = payload["text"]
        same_source_excerpt = stable_identity and (
            prior_text in additional_text or additional_text in prior_text
        )
        if not same_source_excerpt:
            raise ValueError(f"evidence identity changed on revisit: {row.evidence_id}")
        if len(additional_text) > len(prior_text):
            registry[row.evidence_id] = payload

    bundle_order = [row.target_id for row in previous.claim_bundles]
    previous_bundles = {row.target_id: row for row in previous.claim_bundles}
    additional_bundles = {row.target_id: row for row in additional.claim_bundles}
    for target_id in additional_bundles:
        if target_id not in previous_bundles:
            bundle_order.append(target_id)

    action_rows = {
        (source, row.action_id): row
        for source, manifest in (("old", previous), ("new", additional))
        for row in manifest.retrieval_actions
    }
    actions_by_target: dict[str, list[tuple[str, Any]]] = {
        target_id: [] for target_id in bundle_order
    }
    for source, manifest in (("old", previous), ("new", additional)):
        for bundle in manifest.claim_bundles:
            for action_id in bundle.retrieval_action_ids:
                actions_by_target[bundle.target_id].append(
                    (source, action_rows[(source, action_id)])
                )

    merged_actions: list[dict[str, Any]] = []
    bundle_action_ids: dict[str, list[str]] = {target: [] for target in bundle_order}
    for target_id in bundle_order:
        semantic: dict[tuple[Any, ...], dict[str, Any]] = {}
        for _source, action in actions_by_target[target_id]:
            key = (
                action.source_kind,
                action.action_kind,
                action.query_id,
                action.query,
                tuple(action.opened_episode_slugs),
            )
            payload = semantic.setdefault(
                key,
                {
                    **_model_payload(action),
                    "result_evidence_ids": [],
                    "warnings": [],
                },
            )
            payload["result_evidence_ids"] = list(
                dict.fromkeys(
                    [*payload["result_evidence_ids"], *action.result_evidence_ids]
                )
            )
            payload["warnings"] = list(
                dict.fromkeys([*payload["warnings"], *action.warnings])
            )
        for payload in semantic.values():
            action_id = f"A{len(merged_actions) + 1}"
            payload["action_id"] = action_id
            merged_actions.append(payload)
            bundle_action_ids[target_id].append(action_id)

    bundle_payloads: list[dict[str, Any]] = []
    for target_id in bundle_order:
        old = previous_bundles.get(target_id)
        new = additional_bundles.get(target_id)
        source = old or new
        assert source is not None
        payload = _model_payload(source)
        for field in (
            "wiki_evidence",
            "historical_evidence",
            "portfolio_disclosures",
        ):
            rows = [
                *(_model_payload(row) for row in getattr(old, field, ())),
                *(_model_payload(row) for row in getattr(new, field, ())),
            ]
            unique: dict[tuple[str, str], dict[str, Any]] = {}
            for row in rows:
                key = (row["evidence_id"], row["query_id"])
                prior = unique.setdefault(key, row)
                if row["eligible"] and not prior["eligible"]:
                    unique[key] = row
            payload[field] = list(unique.values())
        payload["warnings"] = list(
            dict.fromkeys(
                [
                    *(old.warnings if old else ()),
                    *(new.warnings if new else ()),
                ]
            )
        )
        payload["retrieval_action_ids"] = bundle_action_ids[target_id]
        bundle_payloads.append(payload)

    payload = {
        "schema_version": "claim-retrieval-manifest-v4.4",
        "episode_slug": previous.episode_slug,
        "claim_map_sha256": previous.claim_map_sha256,
        "claim_bundles": bundle_payloads,
        "target_episode_excluded": True,
        "warnings": list(dict.fromkeys([*previous.warnings, *additional.warnings])),
        "evidence_registry": list(registry.values()),
        "retrieval_actions": merged_actions,
    }
    return ClaimRetrievalManifestV44.model_validate_json(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    )


def _merge_neighborhoods(
    previous: TaxonomyNeighborhoodManifestV44,
    additional: TaxonomyNeighborhoodManifestV44,
) -> TaxonomyNeighborhoodManifestV44:
    if (
        previous.taxonomy_sha256 != additional.taxonomy_sha256
        or previous.embedding_model != additional.embedding_model
        or previous.embedding_revision != additional.embedding_revision
    ):
        raise ValueError("partial taxonomy neighborhood binding changed")
    order = [row.target_id for row in previous.claim_neighborhoods]
    old = {row.target_id: row for row in previous.claim_neighborhoods}
    new = {row.target_id: row for row in additional.claim_neighborhoods}
    for target_id in new:
        if target_id not in old:
            order.append(target_id)
    neighborhoods: list[dict[str, Any]] = []
    labels: list[str] = []
    for target_id in order:
        prior = old.get(target_id)
        fresh = new.get(target_id)
        source = prior or fresh
        assert source is not None
        merged_query = fresh.query if fresh is not None else source.query
        candidates: dict[str, dict[str, Any]] = {}
        for neighborhood in (prior, fresh):
            if neighborhood is None:
                continue
            for candidate in neighborhood.candidates:
                payload = _model_payload(candidate)
                payload["query"] = merged_query
                candidates.setdefault(candidate.taxonomy_label, payload)
        rows = list(candidates.values())
        rows.sort(key=lambda row: (row["rank"], row["taxonomy_label"]))
        neighborhoods.append(
            {"target_id": target_id, "query": merged_query, "candidates": rows}
        )
        for row in rows:
            if row["taxonomy_label"] not in labels:
                labels.append(row["taxonomy_label"])
    payload = {
        "schema_version": "taxonomy-neighborhood-manifest-v4.4",
        "taxonomy_sha256": previous.taxonomy_sha256,
        "embedding_model": previous.embedding_model,
        "embedding_revision": previous.embedding_revision,
        "claim_neighborhoods": neighborhoods,
        "ordered_labels": labels,
    }
    return TaxonomyNeighborhoodManifestV44.model_validate_json(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    )


_WORKFLOW_FINGERPRINT_FIELDS = (
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
)

_PROVIDER_FINGERPRINT_SCHEMA = "phase1-provider-fingerprint-v1"
_BUILTIN_PROVIDER_FIELDS: dict[str, tuple[str, ...]] = {
    "vc_clone_graph.providers.fake.FakeProvider": (),
    "vc_clone_graph.providers.fake.DemoFakeProvider": (),
    "vc_clone_graph.providers.openai.OpenAIProvider": (
        "model",
        "embedding_model",
        "max_output_tokens",
    ),
    "vc_clone_graph.providers.openrouter.OpenRouterProvider": (
        "model",
        "max_output_tokens",
        "require_parameters",
        "data_collection",
    ),
    "vc_clone_graph.providers.ollama.OllamaProvider": (
        "model",
        "embedding_model",
        "embedding_batch_size",
        "max_output_tokens",
        "enable_thinking",
        "context_window",
        "structured_output_mode",
    ),
}
_COMPOSITE_PROVIDER_CLASS = "vc_clone_graph.providers.base.CompositeProvider"
_FORBIDDEN_PROVIDER_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "auth_token",
    "token",
    "secret",
    "client_secret",
    "password",
    "authorization",
    "credential",
    "credentials",
    "bearer",
}


def _secret_like_provider_key(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    compact = normalized.replace("_", "")
    parts = set(normalized.split("_"))
    return (
        normalized in _FORBIDDEN_PROVIDER_KEYS
        or compact
        in {
            "apikey",
            "accesstoken",
            "refreshtoken",
            "authtoken",
            "clientsecret",
        }
        or bool(
            parts
            & {
                "token",
                "secret",
                "password",
                "authorization",
                "credential",
                "credentials",
                "bearer",
            }
        )
        or {"api", "key"} <= parts
    )


def _sanitized_base_url(value: object) -> str | None:
    if value is None:
        return None
    try:
        parsed = urlsplit(str(value))
        hostname = parsed.hostname
        if not parsed.scheme or hostname is None:
            return None
        host = hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))
    except (TypeError, ValueError):
        return None


def _safe_provider_payload(
    value: object,
    *,
    seen: set[int],
    key_path: tuple[str, ...] = (),
) -> Any:
    if value is None or type(value) in {str, int, float, bool}:
        if type(value) is float and not math.isfinite(value):
            raise ValueError("provider fingerprint numbers must be finite")
        looks_like_url = type(value) is str and "://" in value
        if type(value) is str and (
            (key_path and key_path[-1].casefold().endswith("url"))
            or looks_like_url
        ):
            sanitized = _sanitized_base_url(value)
            if sanitized is None:
                raise ValueError("provider fingerprint contains an invalid URL")
            return sanitized
        return value
    if type(value) not in {dict, list}:
        raise ValueError("provider fingerprint must be JSON-safe")
    identity = id(value)
    if identity in seen:
        raise ValueError("provider fingerprint contains a recursive value")
    seen.add(identity)
    try:
        if type(value) is list:
            return [
                _safe_provider_payload(item, seen=seen, key_path=key_path)
                for item in value
            ]
        assert isinstance(value, dict)
        result: dict[str, Any] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise ValueError("provider fingerprint keys must be strings")
            if _secret_like_provider_key(key):
                raise ValueError(
                    f"secret-like provider fingerprint key is forbidden: {key}"
                )
            result[key] = _safe_provider_payload(
                nested, seen=seen, key_path=(*key_path, key)
            )
        return result
    finally:
        seen.remove(identity)


def _builtin_provider_fingerprint(
    provider: object,
    class_name: str,
    *,
    seen_providers: set[int],
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "schema_version": _PROVIDER_FINGERPRINT_SCHEMA,
        "provider": class_name,
        "class": class_name,
    }
    if class_name == _COMPOSITE_PROVIDER_CLASS:
        generator = getattr(provider, "generator", None)
        if generator is None:
            raise ValueError("built-in composite provider has no generator")
        identity["generator"] = _safe_provider_identity(
            generator, seen_providers=seen_providers
        )
        return identity
    for field in _BUILTIN_PROVIDER_FIELDS[class_name]:
        value = getattr(provider, field, None)
        if value is None or type(value) in {str, int, float, bool}:
            identity[field] = value
        else:
            raise ValueError(f"built-in provider field is not safely inspectable: {field}")
    client = getattr(provider, "client", None)
    base_url = _sanitized_base_url(getattr(client, "base_url", None))
    if base_url is not None:
        identity["base_url"] = base_url
    return identity


def _safe_provider_identity(
    provider: object, *, seen_providers: set[int] | None = None
) -> dict[str, Any]:
    """Use exact built-in adapters or a versioned custom fingerprint contract."""
    active = seen_providers if seen_providers is not None else set()
    provider_id = id(provider)
    if provider_id in active:
        raise ValueError("provider fingerprint contains a recursive provider cycle")
    active.add(provider_id)
    try:
        class_name = (
            f"{provider.__class__.__module__}.{provider.__class__.__qualname__}"
        )
        if class_name in _BUILTIN_PROVIDER_FIELDS or class_name == _COMPOSITE_PROVIDER_CLASS:
            payload = _builtin_provider_fingerprint(
                provider, class_name, seen_providers=active
            )
        else:
            contract = getattr(provider, "phase1_fingerprint", None)
            if not callable(contract):
                raise ValueError(
                    "custom Phase 1 provider requires an explicit phase1_fingerprint"
                )
            payload = contract()
            if type(payload) is not dict:
                raise ValueError("phase1_fingerprint must return a JSON object")
            if "class" in payload:
                raise ValueError(
                    "provider fingerprint reserved key is forbidden: class"
                )
            payload = {**payload, "class": class_name}
        safe = _safe_provider_payload(payload, seen=set())
        assert isinstance(safe, dict)
        if safe.get("schema_version") != _PROVIDER_FINGERPRINT_SCHEMA:
            raise ValueError("provider fingerprint schema_version is invalid")
        if len(safe) < 2 or not safe.get("provider"):
            raise ValueError("provider fingerprint must be nonempty and named")
        return safe
    finally:
        active.remove(provider_id)


def _workflow_settings_identity(settings: WorkflowSettingsV44Like) -> dict[str, Any]:
    return {
        field: getattr(settings, field)
        for field in _WORKFLOW_FINGERPRINT_FIELDS
        if hasattr(settings, field)
    }


def _wiki_identity(index: HybridWikiIndex) -> dict[str, Any]:
    return {
        "root": index.wiki_root,
        "embedding": embedding_identity(index.embedder),
        "embedding_index": index.embedding_index,
        "require_complete_embeddings": index.require_complete_embeddings,
        "chunks": [
            {
                "chunk_id": row.chunk_id,
                "source_path": row.source_path,
                "source_sha256": row.source_sha256,
                "heading": row.heading,
                "text": row.text,
                "embedding": row.embedding,
            }
            for row in index.chunks
        ],
    }


def _precedent_identity(corpus: PrecedentCorpus | None) -> dict[str, Any] | None:
    if corpus is None:
        return None
    return {
        "target_episode_slug": corpus.target_slug,
        "filtered_manifest": corpus.filtered_manifest(),
        "embedding": embedding_identity(getattr(corpus, "embedder", None)),
        "embedding_index": getattr(corpus, "embedding_index", None),
    }


def _portfolio_identity(
    index: FilteredPortfolioIndex | None,
) -> dict[str, Any] | None:
    if index is None:
        return None
    filtered = index.filtered
    return {
        "filtered_manifest": filtered.manifest(),
        "disclosures": filtered.disclosures,
        "entities": index.entities,
        "embedding": embedding_identity(getattr(index, "embedder", None)),
    }


class VCDecisionWorkflowV44:
    """Isolated v4.4 graph. It contains no Phase 2 node or provider dependency."""

    def __init__(
        self,
        settings: WorkflowSettingsV44Like,
        index: HybridWikiIndex | None,
        provider: GenerationProvider | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        *,
        phase1_provider: GenerationProvider | None = None,
        v44_settings: Phase1V44Settings | None = None,
        precedent_corpus: PrecedentCorpus | None = None,
        portfolio_index: FilteredPortfolioIndex | None = None,
    ) -> None:
        if settings.contract_version != "v4.4":
            raise ValueError("VCDecisionWorkflowV44 requires contract_version v4.4")
        if settings.execution_mode != "phase1_only":
            raise ValueError("VCDecisionWorkflowV44 requires execution_mode phase1_only")
        if v44_settings is None:
            raise ValueError("v44_settings is required")
        if checkpointer is None:
            raise ValueError("checkpointer is required")
        selected_provider = phase1_provider or provider
        if selected_provider is None:
            raise ValueError("a Phase 1 provider is required")
        if index is None or not isinstance(index, HybridWikiIndex):
            raise ValueError("a local semantic HybridWikiIndex is required")
        if not index.chunks or any(row.embedding is None for row in index.chunks):
            raise ValueError("the wiki index must contain complete semantic embeddings")
        _require_local_embedder(index.embedder, context="the v4.4 wiki index")
        if precedent_corpus is not None:
            if precedent_corpus.target_slug != settings.episode_slug:
                raise ValueError("precedent corpus must be filtered to the target episode")
            if any(
                row.episode_slug == settings.episode_slug
                for row in precedent_corpus.list_episodes()
            ):
                raise ValueError("target episode remains accessible in precedent corpus")
            precedent_embedder = getattr(precedent_corpus, "embedder", None)
            if precedent_embedder is not None:
                _require_local_embedder(
                    precedent_embedder, context="the precedent corpus"
                )
        if portfolio_index is not None:
            filtered = getattr(portfolio_index, "filtered", None)
            if (
                filtered is None
                or filtered.target_episode_slug != settings.episode_slug
            ):
                raise ValueError("portfolio index must be filtered to the target episode")
            if any(
                row.source_episode_slug == settings.episode_slug
                for row in filtered.disclosures
            ):
                raise ValueError("target episode remains accessible in portfolio index")
            prefix = settings.episode_slug.split("-", 1)[0]
            target_number = int(prefix) if prefix.isdecimal() else None
            if filtered.target_episode_number != target_number or any(
                target_number is None
                or row.source_episode_number is None
                or row.source_episode_number >= target_number
                for row in filtered.disclosures
            ):
                raise ValueError("portfolio temporal filter is inconsistent")
            portfolio_embedder = getattr(portfolio_index, "embedder", None)
            if portfolio_embedder is not None:
                _require_local_embedder(
                    portfolio_embedder, context="the portfolio index"
                )

        self.settings = settings
        self.v44_settings = v44_settings
        self.index = index
        self.phase1_provider = selected_provider
        self.precedent_corpus = precedent_corpus
        self.portfolio_index = portfolio_index
        self.pitch_rows = _pitch_evidence_index(settings.pitch)
        if not self.pitch_rows:
            raise ValueError("v4.4 claim extraction requires nonempty pitch evidence")
        self.pitch_ids = tuple(row["evidence_id"] for row in self.pitch_rows)
        fingerprint_payload = {
            "schema_version": "phase1-v4.4-run-fingerprint-v1",
            "workflow_settings": _workflow_settings_identity(settings),
            "pitch": settings.pitch,
            "pitch_evidence": self.pitch_rows,
            "phase1_v44_settings": v44_settings,
            "wiki_index": _wiki_identity(index),
            "precedent": _precedent_identity(precedent_corpus),
            "portfolio": _portfolio_identity(portfolio_index),
            "provider": _safe_provider_identity(selected_provider),
        }
        self.run_fingerprint_payload = _canonical_value(fingerprint_payload)
        self.run_fingerprint = sha256(
            _canonical_json_bytes(self.run_fingerprint_payload)
        ).hexdigest()
        self.graph = self._build().compile(checkpointer=checkpointer)

    @property
    def _wal_root(self) -> str:
        return ".phase1-v44-wal"

    @staticmethod
    def _request_sha256(request: GenerationRequest) -> str:
        return sha256(
            _canonical_json_bytes(
                {
                    "phase": request.phase,
                    "prompt": request.prompt,
                    "schema": request.schema,
                    "max_output_tokens": request.max_output_tokens,
                    "reasoning_effort": request.reasoning_effort,
                }
            )
        ).hexdigest()

    @staticmethod
    def _wal_identity(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_fingerprint": payload["run_fingerprint"],
            "thread_id": payload["thread_id"],
            "relative_call_path": payload["relative_call_path"],
            "request_sha256": payload["request_sha256"],
        }

    def _load_wal_entries(self) -> dict[str, tuple[str, dict[str, Any]]]:
        entries: dict[str, tuple[str, dict[str, Any]]] = {}
        for relative in self._store.list_files(self._wal_root):
            name = relative.rsplit("/", 1)[-1]
            if re.fullmatch(r"[0-9a-f]{64}\.json", name) is None:
                raise ValueError("provider-call WAL set is invalid")
            payload = self._store.read_json_object(relative)
            wal_id = name.removesuffix(".json")
            if (
                payload.get("schema_version") != "phase1-provider-call-wal-v1"
                or payload.get("wal_id") != wal_id
                or payload.get("run_fingerprint") != self.run_fingerprint
                or type(payload.get("thread_id")) is not str
                or re.fullmatch(
                    r"phase1/(?:claim-extraction|adjudication/turn-[0-9]{2})/"
                    r"call-[0-9]{2}\.json",
                    str(payload.get("relative_call_path")),
                )
                is None
                or re.fullmatch(r"[0-9a-f]{64}", str(payload.get("request_sha256")))
                is None
                or payload.get("status")
                not in {"pending", "responded_uncheckpointed"}
            ):
                raise ValueError("provider-call WAL entry is invalid")
            expected_id = sha256(
                _canonical_json_bytes(self._wal_identity(payload))
            ).hexdigest()
            if expected_id != wal_id:
                raise ValueError("provider-call WAL identity mismatch")
            if payload["status"] == "responded_uncheckpointed" and type(
                payload.get("uncertain_usage")
            ) is not dict:
                raise ValueError("provider-call WAL uncertain usage is invalid")
            entries[wal_id] = (relative, payload)
        return entries

    @staticmethod
    def _call_records_by_wal(state: WorkflowStateV44) -> dict[str, dict[str, Any]]:
        records = state.get("call_records", [])
        if type(records) is not list:
            raise ValueError("checkpoint call record set is missing")
        result: dict[str, dict[str, Any]] = {}
        for record in records:
            if type(record) is not dict or type(record.get("wal_id")) is not str:
                raise ValueError("checkpoint call WAL binding is missing")
            wal_id = record["wal_id"]
            if wal_id in result:
                raise ValueError("checkpoint call WAL bindings are not unique")
            result[wal_id] = record
        return result

    def _validate_wal_against_state(
        self, state: WorkflowStateV44, *, settle: bool
    ) -> None:
        records = self._call_records_by_wal(state)
        for wal_id, (relative, payload) in self._load_wal_entries().items():
            record = records.get(wal_id)
            if record is None:
                status = payload["status"]
                cost = (payload.get("uncertain_usage") or {}).get("cost_usd")
                raise ValueError(
                    "unmatched provider-call WAL has an uncertain outcome/cost "
                    f"(status={status}, cost_usd={cost})"
                )
            if (
                record.get("relative_path") != payload["relative_call_path"]
                or record.get("request_sha256") != payload["request_sha256"]
            ):
                raise ValueError("provider-call WAL checkpoint binding mismatch")
            if settle:
                self._store.unlink(relative)

    def _begin_call_wal(
        self,
        state: WorkflowStateV44,
        *,
        request: GenerationRequest,
        relative_call_path: str,
    ) -> tuple[str, dict[str, Any]]:
        self._validate_wal_against_state(state, settle=False)
        request_digest = self._request_sha256(request)
        identity = {
            "run_fingerprint": self.run_fingerprint,
            "thread_id": state["run_thread_id"],
            "relative_call_path": relative_call_path,
            "request_sha256": request_digest,
        }
        wal_id = sha256(_canonical_json_bytes(identity)).hexdigest()
        relative = f"{self._wal_root}/{wal_id}.json"
        if self._store.exists(relative):
            raise ValueError("provider-call WAL identity is already pending")
        payload = {
            "schema_version": "phase1-provider-call-wal-v1",
            "wal_id": wal_id,
            **identity,
            "phase": request.phase,
            "status": "pending",
            "uncertain_usage": None,
        }
        self._store.atomic_write(relative, _canonical_json_bytes(payload))
        return relative, payload

    def _mark_wal_responded(
        self, relative: str, payload: dict[str, Any], result: GenerationResult
    ) -> dict[str, Any]:
        responded = {
            **payload,
            "status": "responded_uncheckpointed",
            "uncertain_usage": result.usage.model_dump(mode="json"),
        }
        self._store.atomic_write(relative, _canonical_json_bytes(responded))
        return responded

    def _record_call(
        self,
        relative_path: str,
        request: GenerationRequest,
        result: GenerationResult,
        *,
        wal_id: str,
        request_sha256: str,
    ) -> dict[str, Any]:
        payload = self._call_payload(request, result)
        raw = _canonical_json_bytes(payload)
        self._store.atomic_write(relative_path, raw)
        return {
            "relative_path": relative_path,
            "sha256": sha256(raw).hexdigest(),
            "payload": payload,
            "wal_id": wal_id,
            "request_sha256": request_sha256,
        }

    @staticmethod
    def _call_payload(
        request: GenerationRequest, result: GenerationResult
    ) -> dict[str, Any]:
        payload = _canonical_value(
            {
                "phase": request.phase,
                "prompt": request.prompt,
                "schema": request.schema,
                "raw": result.content,
                "parsed": result.parsed,
                "provider_metadata": result.raw_metadata,
                "usage": result.usage.model_dump(mode="json"),
                "cost_usd": result.usage.cost_usd,
                "elapsed_seconds": result.elapsed_seconds,
                "max_output_tokens": request.max_output_tokens,
                "reasoning_effort": request.reasoning_effort,
            }
        )
        assert isinstance(payload, dict)
        return payload

    @staticmethod
    def _request_payload(request: GenerationRequest) -> dict[str, Any]:
        return {
            "phase": request.phase,
            "prompt": request.prompt,
            "schema": request.schema,
            "max_output_tokens": request.max_output_tokens,
            "reasoning_effort": request.reasoning_effort,
        }

    @staticmethod
    def _request_from_payload(payload: dict[str, Any]) -> GenerationRequest:
        return GenerationRequest(
            phase=payload["phase"],
            prompt=payload["prompt"],
            schema=payload["schema"],
            max_output_tokens=payload["max_output_tokens"],
            reasoning_effort=payload["reasoning_effort"],
        )

    def _pending_call(
        self,
        state: WorkflowStateV44,
        *,
        kind: str,
        request: GenerationRequest,
        relative_path: str,
    ) -> dict[str, Any]:
        if (
            state.get("pending_call") is not None
            or state.get("pending_response") is not None
        ):
            raise ValueError("provider intent state is already occupied")
        request_payload = self._request_payload(request)
        request_digest = self._request_sha256(request)
        identity = {
            "schema_version": "phase1-provider-call-intent-v1",
            "kind": kind,
            "run_fingerprint": self.run_fingerprint,
            "thread_id": state["run_thread_id"],
            "relative_call_path": relative_path,
            "request_sha256": request_digest,
            "request": request_payload,
        }
        return {
            **identity,
            "intent_sha256": sha256(_canonical_json_bytes(identity)).hexdigest(),
        }

    def _validated_pending_call(self, state: WorkflowStateV44) -> dict[str, Any]:
        pending = state.get("pending_call")
        if type(pending) is not dict:
            raise ValueError("checkpoint pending provider intent is missing")
        try:
            identity = {
                key: pending[key]
                for key in (
                    "schema_version",
                    "kind",
                    "run_fingerprint",
                    "thread_id",
                    "relative_call_path",
                    "request_sha256",
                    "request",
                )
            }
        except KeyError as exc:
            raise ValueError(
                "checkpoint pending provider intent binding is incomplete"
            ) from exc
        if (
            identity["schema_version"] != "phase1-provider-call-intent-v1"
            or identity["run_fingerprint"] != self.run_fingerprint
            or identity["thread_id"] != state.get("run_thread_id")
            or sha256(_canonical_json_bytes(identity)).hexdigest()
            != pending.get("intent_sha256")
        ):
            raise ValueError("checkpoint pending provider intent binding is invalid")
        if type(identity["request"]) is not dict:
            raise ValueError("checkpoint pending provider request is invalid")
        try:
            request = self._request_from_payload(identity["request"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("checkpoint pending provider request is invalid") from exc
        if self._request_sha256(request) != identity["request_sha256"]:
            raise ValueError("checkpoint pending provider request hash is invalid")
        if (
            type(identity["kind"]) is not str
            or type(identity["relative_call_path"]) is not str
        ):
            raise ValueError("checkpoint pending provider call kind is invalid")
        expected = {
            "claim_primary": (
                "phase1_claim_extraction",
                r"phase1/claim-extraction/call-01\.json",
            ),
            "claim_repair": (
                "phase1_claim_extraction_repair",
                r"phase1/claim-extraction/call-02\.json",
            ),
            "adjudication_primary": (
                "phase1_adjudication",
                r"phase1/adjudication/turn-[0-9]{2}/call-01\.json",
            ),
            "adjudication_repair": (
                "phase1_adjudication_repair",
                r"phase1/adjudication/turn-[0-9]{2}/call-02\.json",
            ),
        }.get(identity["kind"])
        if (
            expected is None
            or request.phase != expected[0]
            or re.fullmatch(expected[1], identity["relative_call_path"]) is None
        ):
            raise ValueError("checkpoint pending provider call kind is invalid")
        return pending

    def _execute_pending_call(self, state: WorkflowStateV44) -> WorkflowStateV44:
        if state.get("pending_response") is not None:
            raise ValueError("pending provider intent already has a response")
        pending = self._validated_pending_call(state)
        request = self._request_from_payload(pending["request"])
        relative_path = pending["relative_call_path"]
        wal_path, wal = self._begin_call_wal(
            state, request=request, relative_call_path=relative_path
        )
        result = self.phase1_provider.generate(request)
        wal = self._mark_wal_responded(wal_path, wal, result)
        record = self._record_call(
            relative_path,
            request,
            result,
            wal_id=wal["wal_id"],
            request_sha256=wal["request_sha256"],
        )
        usage, by_phase = _add_usage(state, result)
        records = [*state.get("call_records", []), record]
        paths = [row["relative_path"] for row in records]
        if len(paths) != len(set(paths)):
            raise ValueError("provider call artifact path was reused")
        return {
            "pending_response": {
                "schema_version": "phase1-provider-call-response-v1",
                "intent_sha256": pending["intent_sha256"],
                "wal_id": wal["wal_id"],
                "request_sha256": wal["request_sha256"],
                "result": result.model_dump(mode="json"),
            },
            "usage": usage,
            "usage_by_phase": by_phase,
            "call_records": records,
        }

    def _pending_result(
        self, state: WorkflowStateV44
    ) -> tuple[dict[str, Any], GenerationRequest, GenerationResult]:
        pending = self._validated_pending_call(state)
        response = state.get("pending_response")
        if type(response) is not dict:
            raise ValueError("checkpoint pending provider response is missing")
        if (
            response.get("schema_version") != "phase1-provider-call-response-v1"
            or response.get("intent_sha256") != pending["intent_sha256"]
            or response.get("request_sha256") != pending["request_sha256"]
        ):
            raise ValueError("checkpoint pending provider response binding is invalid")
        result = GenerationResult.model_validate(response.get("result"))
        records = self._call_records_by_wal(state)
        record = records.get(response.get("wal_id"))
        expected_wal_id = sha256(
            _canonical_json_bytes(
                {
                    "run_fingerprint": pending["run_fingerprint"],
                    "thread_id": pending["thread_id"],
                    "relative_call_path": pending["relative_call_path"],
                    "request_sha256": pending["request_sha256"],
                }
            )
        ).hexdigest()
        if (
            record is None
            or response.get("wal_id") != expected_wal_id
            or record.get("relative_path") != pending["relative_call_path"]
            or record.get("request_sha256") != pending["request_sha256"]
            or record.get("payload") != self._call_payload(
                self._request_from_payload(pending["request"]), result
            )
        ):
            raise ValueError("checkpoint response has no exact call record binding")
        return pending, self._request_from_payload(pending["request"]), result

    def _validate_pending_resume(self, state: WorkflowStateV44) -> None:
        pending = state.get("pending_call")
        response = state.get("pending_response")
        if pending is None and response is None:
            return
        if pending is None:
            raise ValueError("checkpoint has a provider response without its intent")
        self._validated_pending_call(state)
        if response is None:
            raise ValueError(
                "checkpoint has pending provider intent with an uncertain outcome; "
                "refusing retry"
            )
        self._pending_result(state)

    @staticmethod
    def _cleared_pending() -> dict[str, Any]:
        return {"pending_call": None, "pending_response": None}

    def _claim_primary_intent(self, state: WorkflowStateV44) -> WorkflowStateV44:
        schema = claim_map_v44_json_schema(
            episode_slug=self.settings.episode_slug,
            pitch_evidence_ids=self.pitch_ids,
        )
        prompt = claim_extraction_v44_prompt(
            self.settings.investor_name,
            self.settings.episode_slug,
            self.pitch_rows,
        )
        request = GenerationRequest(
            phase="phase1_claim_extraction",
            prompt=prompt,
            schema=schema,
            max_output_tokens=self.settings.phase1_max_output_tokens,
            reasoning_effort=self.settings.phase1_reasoning_effort,
        )
        return {
            "pending_call": self._pending_call(
                state,
                kind="claim_primary",
                request=request,
                relative_path="phase1/claim-extraction/call-01.json",
            ),
            "pending_response": None,
            "phase1_action": "awaiting_call",
        }

    def _claim_repair_intent(self, state: WorkflowStateV44) -> WorkflowStateV44:
        context = state.get("repair_context")
        if type(context) is not dict or context.get("kind") != "claim":
            raise ValueError("claim repair context is missing")
        request = GenerationRequest(
            phase="phase1_claim_extraction_repair",
            prompt=claim_extraction_repair_v44_prompt(
                self.settings.investor_name,
                self.settings.episode_slug,
                self.pitch_rows,
                context["content"],
                context["errors"],
            ),
            schema=context["schema"],
            max_output_tokens=self.settings.phase1_max_output_tokens,
            reasoning_effort=self.settings.phase1_reasoning_effort,
        )
        return {
            "pending_call": self._pending_call(
                state,
                kind="claim_repair",
                request=request,
                relative_path="phase1/claim-extraction/call-02.json",
            ),
            "pending_response": None,
            "phase1_action": "awaiting_call",
        }

    def _claim_finalize(self, state: WorkflowStateV44) -> WorkflowStateV44:
        pending, request, result = self._pending_result(state)
        self._validate_wal_against_state(state, settle=True)
        kind = pending["kind"]
        if kind not in {"claim_primary", "claim_repair"}:
            raise ValueError("claim finalizer received another call kind")
        try:
            claim_map = _validate_json_model(ClaimMapV44, result.parsed, request.schema)
            assert isinstance(claim_map, ClaimMapV44)
            _validate_claim_map_runtime(claim_map, self.pitch_ids)
        except (ValidationError, BoundSchemaValidationError) as exc:
            if kind == "claim_primary":
                return {
                    **self._cleared_pending(),
                    "repair_context": {
                        "kind": "claim",
                        "content": result.content,
                        "errors": _validation_errors(exc),
                        "schema": request.schema,
                    },
                    "phase1_action": "claim_repair",
                }
            return {
                **self._cleared_pending(),
                "repair_context": None,
                "phase1_action": "failed",
                "phase1_status": "failed",
                "phase2_status": "not_run",
                "phase1_findings": [
                    f"CLAIM_EXTRACTION_INVALID:{value}"
                    for value in _validation_errors(exc)
                ],
                "events": _event(state, "phase1_claim_extraction", status="failed"),
            }
        digest = _freeze_model(
            self._store,
            "phase1",
            "claim-map",
            claim_map,
        )
        return {
            **self._cleared_pending(),
            "repair_context": None,
            "claim_map": _model_payload(claim_map),
            "claim_map_sha256": digest,
            "phase1_action": "continue",
            "events": _event(state, "phase1_claim_extraction", status="valid"),
        }

    def _retrieve_claims(self, state: WorkflowStateV44) -> WorkflowStateV44:
        self._validate_wal_against_state(state, settle=True)
        claim_map = _state_model(ClaimMapV44, state["claim_map"])
        assert isinstance(claim_map, ClaimMapV44)
        only_ids: Sequence[str] | None = None
        if state.get("phase1_iteration", 0) > 0:
            only_ids = _SequenceSet(state.get("targeted_retrieval_ids", []))
        retrieved = retrieve_claim_evidence_v44(
            claim_map=claim_map,
            wiki_index=self.index,
            precedent_corpus=self.precedent_corpus,
            portfolio_index=self.portfolio_index,
            target_episode_slug=self.settings.episode_slug,
            top_k=self.v44_settings.claim_retrieval_top_k,
            max_wiki_reads=self.v44_settings.max_wiki_reads_per_claim,
            max_precedent_reads=self.v44_settings.max_precedent_reads_per_claim,
            only_ids=only_ids,
        )
        if only_ids is not None:
            prior = _state_model(ClaimRetrievalManifestV44, state["claim_retrieval"])
            assert isinstance(prior, ClaimRetrievalManifestV44)
            retrieved = _merge_retrieval_manifests(prior, retrieved)
        digest = _freeze_model(
            self._store,
            "phase1",
            "claim-retrieval",
            retrieved,
        )
        registry = {
            row.evidence_id: _model_payload(row) for row in retrieved.evidence_registry
        }
        self._store.atomic_write(
            "phase1/evidence-registry.json",
            _canonical_json_bytes(list(registry.values())),
        )
        payload = _model_payload(retrieved)
        return {
            "claim_retrieval": payload,
            "claim_retrieval_manifest": payload,
            "claim_retrieval_sha256": digest,
            "evidence_registry": registry,
            "events": _event(
                state,
                "phase1_claim_retrieval",
                targeted_ids=list(only_ids or ()),
            ),
        }

    def _retrieve_taxonomy(self, state: WorkflowStateV44) -> WorkflowStateV44:
        self._validate_wal_against_state(state, settle=True)
        retrieval = _state_model(ClaimRetrievalManifestV44, state["claim_retrieval"])
        assert isinstance(retrieval, ClaimRetrievalManifestV44)
        only_ids = set(state.get("targeted_retrieval_ids", []))
        bundles = tuple(
            row
            for row in retrieval.claim_bundles
            if not only_ids
            or state.get("phase1_iteration", 0) == 0
            or row.target_id in only_ids
        )
        neighborhood = retrieve_taxonomy_neighborhoods_v44(
            claim_bundles=bundles,
            taxonomy=self.settings.taxonomy_records,
            embedder=self.index.embedder,
            top_k=self.v44_settings.taxonomy_top_k,
        )
        if state.get("phase1_iteration", 0) > 0:
            previous = _state_model(
                TaxonomyNeighborhoodManifestV44, state["taxonomy_neighborhood"]
            )
            assert isinstance(previous, TaxonomyNeighborhoodManifestV44)
            neighborhood = _merge_neighborhoods(previous, neighborhood)
        digest = _freeze_model(
            self._store,
            "phase1",
            "taxonomy-neighborhood",
            neighborhood,
        )
        payload = _model_payload(neighborhood)
        return {
            "taxonomy_neighborhood": payload,
            "taxonomy_neighborhood_manifest": payload,
            "taxonomy_neighborhood_sha256": digest,
            "events": _event(state, "phase1_taxonomy_retrieval"),
        }

    def _adjudication_primary_intent(
        self, state: WorkflowStateV44
    ) -> WorkflowStateV44:
        self._validate_wal_against_state(state, settle=True)
        iteration = state.get("phase1_iteration", 0) + 1
        claim_map = _state_model(ClaimMapV44, state["claim_map"])
        retrieval = _state_model(ClaimRetrievalManifestV44, state["claim_retrieval"])
        neighborhood = _state_model(
            TaxonomyNeighborhoodManifestV44, state["taxonomy_neighborhood"]
        )
        assert isinstance(claim_map, ClaimMapV44)
        assert isinstance(retrieval, ClaimRetrievalManifestV44)
        assert isinstance(neighborhood, TaxonomyNeighborhoodManifestV44)
        prior_payload = state.get("last_valid_adjudication")
        prior = (
            _state_model(RationaleAdjudicationV44, prior_payload)
            if prior_payload is not None
            else None
        )
        prior_findings = state.get("phase1_findings", []) if prior else []
        investor_ids, portfolio_ids = _eligible_ids(retrieval)
        claim_ids = [
            row.claim_id for row in (*claim_map.material_claims, *claim_map.adverse_claims)
        ]
        question_ids = [row.question_id for row in claim_map.unanswered_questions]
        schema = adjudication_v44_json_schema(
            episode_slug=self.settings.episode_slug,
            claim_ids=claim_ids,
            question_ids=question_ids,
            pitch_evidence_ids=self.pitch_ids,
            investor_evidence_ids=investor_ids,
            portfolio_disclosure_ids=portfolio_ids,
            taxonomy_labels=neighborhood.ordered_labels,
        )
        prompt = rationale_adjudication_v44_prompt(
            self.settings.investor_name,
            self.settings.episode_slug,
            claim_map,
            retrieval,
            neighborhood,
            prior,
            prior_findings,
        )
        request = GenerationRequest(
            phase="phase1_adjudication",
            prompt=prompt,
            schema=schema,
            max_output_tokens=self.settings.phase1_max_output_tokens,
            reasoning_effort=self.settings.phase1_reasoning_effort,
        )
        return {
            "phase1_iteration": iteration,
            "pending_call": self._pending_call(
                state,
                kind="adjudication_primary",
                request=request,
                relative_path=(
                    f"phase1/adjudication/turn-{iteration:02d}/call-01.json"
                ),
            ),
            "pending_response": None,
            "phase1_action": "awaiting_call",
        }

    def _adjudication_repair_intent(
        self, state: WorkflowStateV44
    ) -> WorkflowStateV44:
        context = state.get("repair_context")
        if type(context) is not dict or context.get("kind") != "adjudication":
            raise ValueError("adjudication repair context is missing")
        claim_map = _state_model(ClaimMapV44, state["claim_map"])
        retrieval = _state_model(ClaimRetrievalManifestV44, state["claim_retrieval"])
        neighborhood = _state_model(
            TaxonomyNeighborhoodManifestV44, state["taxonomy_neighborhood"]
        )
        assert isinstance(claim_map, ClaimMapV44)
        assert isinstance(retrieval, ClaimRetrievalManifestV44)
        assert isinstance(neighborhood, TaxonomyNeighborhoodManifestV44)
        prior_payload = state.get("last_valid_adjudication")
        prior = (
            _state_model(RationaleAdjudicationV44, prior_payload)
            if prior_payload is not None
            else None
        )
        prior_findings = state.get("phase1_findings", []) if prior else []
        request = GenerationRequest(
            phase="phase1_adjudication_repair",
            prompt=rationale_adjudication_repair_v44_prompt(
                self.settings.investor_name,
                self.settings.episode_slug,
                claim_map,
                retrieval,
                neighborhood,
                context["content"],
                context["errors"],
                prior,
                prior_findings,
            ),
            schema=context["schema"],
            max_output_tokens=self.settings.phase1_max_output_tokens,
            reasoning_effort=self.settings.phase1_reasoning_effort,
        )
        iteration = state["phase1_iteration"]
        return {
            "pending_call": self._pending_call(
                state,
                kind="adjudication_repair",
                request=request,
                relative_path=(
                    f"phase1/adjudication/turn-{iteration:02d}/call-02.json"
                ),
            ),
            "pending_response": None,
            "phase1_action": "awaiting_call",
        }

    def _adjudication_finalize(
        self, state: WorkflowStateV44
    ) -> WorkflowStateV44:
        pending, request, result = self._pending_result(state)
        self._validate_wal_against_state(state, settle=True)
        kind = pending["kind"]
        if kind not in {"adjudication_primary", "adjudication_repair"}:
            raise ValueError("adjudication finalizer received another call kind")
        iteration = state["phase1_iteration"]
        claim_map = _state_model(ClaimMapV44, state["claim_map"])
        retrieval = _state_model(ClaimRetrievalManifestV44, state["claim_retrieval"])
        neighborhood = _state_model(
            TaxonomyNeighborhoodManifestV44, state["taxonomy_neighborhood"]
        )
        assert isinstance(claim_map, ClaimMapV44)
        assert isinstance(retrieval, ClaimRetrievalManifestV44)
        assert isinstance(neighborhood, TaxonomyNeighborhoodManifestV44)
        prior_payload = state.get("last_valid_adjudication")
        prior = (
            _state_model(RationaleAdjudicationV44, prior_payload)
            if prior_payload is not None
            else None
        )
        prior_findings = state.get("phase1_findings", []) if prior else []
        investor_ids, portfolio_ids = _eligible_ids(retrieval)
        normalized_result, normalization_audit = (
            normalize_adjudication_payload_v44(result.parsed)
        )
        try:
            adjudication = _validate_json_model(
                RationaleAdjudicationV44, normalized_result, request.schema
            )
            assert isinstance(adjudication, RationaleAdjudicationV44)
        except (ValidationError, BoundSchemaValidationError) as exc:
            if kind == "adjudication_primary":
                return {
                    **self._cleared_pending(),
                    "repair_context": {
                        "kind": "adjudication",
                        "content": result.content,
                        "errors": _validation_errors(exc),
                        "schema": request.schema,
                    },
                    "phase1_action": "adjudication_repair",
                }
            if prior is None or prior.adjudication_status != "provisional":
                return {
                    **self._cleared_pending(),
                    "repair_context": None,
                    "phase1_action": "failed",
                    "phase1_status": "failed",
                    "phase2_status": "not_run",
                    "phase1_findings": [
                        f"ADJUDICATION_INVALID:{value}"
                        for value in _validation_errors(exc)
                    ],
                    "events": _event(state, "phase1_adjudication", status="failed"),
                }
            adjudication = prior
            findings = sorted(
                {
                    *prior.validator_findings,
                    *prior_findings,
                    "ADJUDICATION_REPAIR_FAILED_USING_PRIOR",
                }
            )
            return {
                **self._cleared_pending(),
                "repair_context": None,
                "adjudication": _model_payload(adjudication),
                "phase1_findings": findings,
                "phase1_action": "freeze_provisional",
                "events": _event(
                    state, "phase1_adjudication", status="prior_preserved"
                ),
            }

        adjudication, reuse_audit, mapping_audit = finalize_adjudication_v44(
            adjudication,
            neighborhood,
            retrieval,
        )
        unresolved_findings = {
            f"UNRESOLVED_CONSTRAINT_MAPPING:{constraint_id}"
            for constraint_id in mapping_audit.unresolved_constraint_ids
        }
        findings = sorted(
            {
                *adjudication.validator_findings,
                *unresolved_findings,
                *adjudication_findings_v44(
                    claim_map,
                    adjudication,
                    neighborhood,
                    pitch_evidence_ids=self.pitch_ids,
                    investor_evidence_ids=investor_ids,
                    portfolio_disclosure_ids=portfolio_ids,
                    retrieval_manifest=retrieval,
                ),
            }
        )
        requested = _targeted_ids_from_findings(
            claim_map, adjudication, findings
        )
        blocker_prefixes = (
            "DUPLICATE_",
            "INACCESSIBLE_",
            "INVESTOR_EVIDENCE_",
            "LABEL_OUTSIDE_",
            "PITCH_EVIDENCE_",
            "PORTFOLIO_EVIDENCE_",
            "UNRESOLVED_CONSTRAINT_MAPPING:",
            "UNKNOWN_",
        )
        structural_blockers = [
            value for value in findings if value.startswith(blocker_prefixes)
        ]
        if structural_blockers:
            if prior is None:
                action = "failed"
                status = "failed"
            else:
                adjudication = prior
                findings = sorted(
                    {
                        *prior_findings,
                        *structural_blockers,
                        "STRUCTURAL_ADJUDICATION_REJECTED_USING_PRIOR",
                    }
                )
                requested = []
                action = "freeze_provisional"
                status = "prior_preserved"
        elif requested and iteration <= self.v44_settings.max_revisits:
            action = "targeted_retrieval"
            status = "provisional"
        elif requested or adjudication.adjudication_status == "provisional" or findings:
            action = "freeze_provisional"
            status = "provisional"
        else:
            action = "freeze_accepted"
            status = "valid"
        payload = _model_payload(adjudication)
        result_state: WorkflowStateV44 = {
            **self._cleared_pending(),
            "repair_context": None,
            "adjudication": payload,
            "last_valid_adjudication": payload,
            "phase1_findings": findings,
            "phase1_action": action,
            "events": _event(
                state,
                "phase1_adjudication",
                status=status,
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
            ),
        }
        if requested:
            result_state["targeted_retrieval_ids"] = requested
        if action == "failed":
            result_state["phase1_status"] = "failed"
            result_state["phase2_status"] = "not_run"
        return result_state

    def _freeze(self, state: WorkflowStateV44) -> WorkflowStateV44:
        self._validate_wal_against_state(state, settle=True)
        claim_map = _state_model(ClaimMapV44, state["claim_map"])
        retrieval = _state_model(ClaimRetrievalManifestV44, state["claim_retrieval"])
        neighborhood = _state_model(
            TaxonomyNeighborhoodManifestV44, state["taxonomy_neighborhood"]
        )
        adjudication = _state_model(RationaleAdjudicationV44, state["adjudication"])
        assert isinstance(claim_map, ClaimMapV44)
        assert isinstance(retrieval, ClaimRetrievalManifestV44)
        assert isinstance(neighborhood, TaxonomyNeighborhoodManifestV44)
        assert isinstance(adjudication, RationaleAdjudicationV44)
        root = "phase1"
        claim_digest = _freeze_model(
            self._store, root, "claim-map", claim_map
        )
        retrieval_digest = _freeze_model(
            self._store, root, "claim-retrieval", retrieval
        )
        neighborhood_digest = _freeze_model(
            self._store,
            root,
            "taxonomy-neighborhood",
            neighborhood,
        )
        adjudication_digest = _freeze_model(
            self._store, root, "adjudication", adjudication
        )
        investigation = build_investigation_v44(
            episode_slug=self.settings.episode_slug,
            claim_map=claim_map,
            adjudication=adjudication,
            retrieval_manifest=retrieval,
            neighborhood=neighborhood,
            claim_map_sha256=claim_digest,
            adjudication_sha256=adjudication_digest,
            pitch_evidence_ids=self.pitch_ids,
        )
        if state.get("phase1_action") == "freeze_provisional":
            investigation = investigation.model_copy(
                update={
                    "investigation_status": "provisional",
                    "validator_findings": tuple(
                        sorted(
                            {
                                *investigation.validator_findings,
                                *state.get("phase1_findings", []),
                            }
                        )
                    ),
                }
            )
            assert isinstance(investigation, InvestigationV44)
        investigation_digest = _freeze_model(
            self._store,
            root,
            "investigation",
            investigation,
        )
        accepted = (
            state.get("phase1_action") == "freeze_accepted"
            and investigation.investigation_status == "accepted"
        )
        return {
            "claim_map_sha256": claim_digest,
            "claim_retrieval_sha256": retrieval_digest,
            "taxonomy_neighborhood_sha256": neighborhood_digest,
            "adjudication_sha256": adjudication_digest,
            "investigation": _model_payload(investigation),
            "investigation_sha256": investigation_digest,
            "phase1_status": "accepted" if accepted else "provisional",
            "phase2_status": "not_run",
            "events": _event(
                state,
                "phase1_freeze",
                status="accepted" if accepted else "provisional",
            ),
        }

    @staticmethod
    def _route_after_extraction(state: WorkflowStateV44) -> str:
        return state["phase1_action"]

    @staticmethod
    def _route_after_provider_call(state: WorkflowStateV44) -> str:
        pending = state.get("pending_call")
        if type(pending) is not dict:
            raise ValueError("provider execution produced no bound intent")
        kind = pending.get("kind")
        if kind in {"claim_primary", "claim_repair"}:
            return "claim"
        if kind in {"adjudication_primary", "adjudication_repair"}:
            return "adjudication"
        raise ValueError("provider execution produced an unknown call kind")

    @staticmethod
    def _route_after_adjudication(state: WorkflowStateV44) -> str:
        return state["phase1_action"]

    def _build(self) -> StateGraph:
        builder = StateGraph(WorkflowStateV44)
        builder.add_node("phase1_claim_intent", self._claim_primary_intent)
        builder.add_node("phase1_claim_repair_intent", self._claim_repair_intent)
        builder.add_node("phase1_provider_call", self._execute_pending_call)
        builder.add_node("phase1_claim_finalize", self._claim_finalize)
        builder.add_node("phase1_claim_retrieval", self._retrieve_claims)
        builder.add_node("phase1_taxonomy_retrieval", self._retrieve_taxonomy)
        builder.add_node(
            "phase1_adjudication_intent", self._adjudication_primary_intent
        )
        builder.add_node(
            "phase1_adjudication_repair_intent",
            self._adjudication_repair_intent,
        )
        builder.add_node("phase1_adjudication_finalize", self._adjudication_finalize)
        builder.add_node("phase1_freeze", self._freeze)
        builder.add_edge(START, "phase1_claim_intent")
        builder.add_edge("phase1_claim_intent", "phase1_provider_call")
        builder.add_edge("phase1_claim_repair_intent", "phase1_provider_call")
        builder.add_conditional_edges(
            "phase1_provider_call",
            self._route_after_provider_call,
            {
                "claim": "phase1_claim_finalize",
                "adjudication": "phase1_adjudication_finalize",
            },
        )
        builder.add_conditional_edges(
            "phase1_claim_finalize",
            self._route_after_extraction,
            {
                "continue": "phase1_claim_retrieval",
                "claim_repair": "phase1_claim_repair_intent",
                "failed": END,
            },
        )
        builder.add_edge("phase1_claim_retrieval", "phase1_taxonomy_retrieval")
        builder.add_edge("phase1_taxonomy_retrieval", "phase1_adjudication_intent")
        builder.add_edge("phase1_adjudication_intent", "phase1_provider_call")
        builder.add_edge(
            "phase1_adjudication_repair_intent", "phase1_provider_call"
        )
        builder.add_conditional_edges(
            "phase1_adjudication_finalize",
            self._route_after_adjudication,
            {
                "adjudication_repair": "phase1_adjudication_repair_intent",
                "targeted_retrieval": "phase1_claim_retrieval",
                "freeze_accepted": "phase1_freeze",
                "freeze_provisional": "phase1_freeze",
                "failed": END,
            },
        )
        builder.add_edge("phase1_freeze", END)
        return builder

    def _initial_state(self, thread_id: str) -> WorkflowStateV44:
        zero = _zero_usage()
        return {
            "contract_version": "v4.4",
            "episode_slug": self.settings.episode_slug,
            "run_fingerprint": self.run_fingerprint,
            "run_thread_id": thread_id,
            "call_records": [],
            "pending_call": None,
            "pending_response": None,
            "repair_context": None,
            "phase1_iteration": 0,
            "phase1_status": "running",
            "phase2_status": "not_run",
            "phase1_findings": [],
            "targeted_retrieval_ids": [],
            "evidence_registry": {},
            "events": [],
            "usage": dict(zero),
            "usage_by_phase": {
                "phase1": dict(zero),
                "phase2": dict(zero),
            },
        }

    @property
    def _owner_path(self) -> str:
        return "run-owner-v44.json"

    def _owner_payload(self, thread_id: str) -> dict[str, Any]:
        return {
            "schema_version": "phase1-v4.4-run-owner-v1",
            "thread_id": thread_id,
            "run_fingerprint": self.run_fingerprint,
            "fingerprint_payload": self.run_fingerprint_payload,
        }

    @contextmanager
    def _run_lock(self) -> Iterator[None]:
        descriptor = os.dup(self._store.root_fd)
        locked = False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
            self._store.verify_tree()
            yield
        finally:
            try:
                if locked:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _checkpoint_state(self, thread_id: str) -> WorkflowStateV44:
        snapshot = self.graph.get_state(
            {"configurable": {"thread_id": thread_id}}
        )
        values = getattr(snapshot, "values", None)
        return dict(values) if isinstance(values, dict) else {}

    def _establish_new_owner(self, thread_id: str) -> None:
        if self._store.exists(self._owner_path):
            try:
                owner = self._store.read_json_object(self._owner_path)
            except ValueError as exc:
                raise ValueError(
                    "run root has invalid existing owner metadata"
                ) from exc
            owner_thread = owner.get("thread_id")
            if owner_thread != thread_id:
                raise ValueError("run root is owned by another thread")
            raise ValueError("run root is already owned by a persisted run")
        self._store.atomic_write(
            self._owner_path,
            _canonical_json_bytes(self._owner_payload(thread_id)),
        )

    def _validate_or_rebuild_owner(
        self, thread_id: str, checkpoint: WorkflowStateV44
    ) -> None:
        if checkpoint.get("run_thread_id") != thread_id:
            raise ValueError("checkpoint thread ownership mismatch")
        expected = self._owner_payload(thread_id)
        try:
            owner = self._store.read_json_object(self._owner_path)
        except ValueError:
            owner = None
        if owner is None:
            self._store.atomic_write(
                self._owner_path, _canonical_json_bytes(expected)
            )
            return
        if owner != expected:
            raise ValueError("run root owner metadata mismatch")

    def _verify_owner_artifact(self, state: WorkflowStateV44) -> None:
        thread_id = state.get("run_thread_id")
        if type(thread_id) is not str or not thread_id:
            raise ValueError("checkpoint thread ownership is missing")
        owner = self._store.read_json_object(self._owner_path)
        if owner != self._owner_payload(thread_id):
            raise ValueError("run root owner metadata mismatch")

    def _confirm_checkpointed_calls(
        self, thread_id: str, result: WorkflowStateV44
    ) -> None:
        checkpoint = self._checkpoint_state(thread_id)
        if (
            checkpoint.get("run_fingerprint") != self.run_fingerprint
            or checkpoint.get("run_thread_id") != thread_id
            or checkpoint.get("call_records") != result.get("call_records")
            or checkpoint.get("pending_call") != result.get("pending_call")
            or checkpoint.get("pending_response") != result.get("pending_response")
        ):
            raise ValueError("completed graph state is not checkpoint-bound")
        self._validate_wal_against_state(checkpoint, settle=True)

    def _verify_call_artifacts(self, state: WorkflowStateV44) -> None:
        records = state.get("call_records")
        if type(records) is not list:
            raise ValueError("checkpoint call record set is missing")
        expected_paths: set[str] = set()
        expected_usage = _zero_usage()
        for record in records:
            if type(record) is not dict:
                raise ValueError("checkpoint call record is invalid")
            relative = record.get("relative_path")
            payload = record.get("payload")
            digest = record.get("sha256")
            wal_id = record.get("wal_id")
            request_digest = record.get("request_sha256")
            if (
                type(relative) is not str
                or type(payload) is not dict
                or type(digest) is not str
                or type(wal_id) is not str
                or re.fullmatch(r"[0-9a-f]{64}", wal_id) is None
                or type(request_digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", request_digest) is None
                or not re.fullmatch(
                    r"phase1/(?:claim-extraction|adjudication/turn-[0-9]{2})/"
                    r"call-[0-9]{2}\.json",
                    relative,
                )
            ):
                raise ValueError("checkpoint call record binding is invalid")
            if relative in expected_paths:
                raise ValueError("checkpoint call artifact paths are not unique")
            expected_paths.add(relative)
            raw = _canonical_json_bytes(payload)
            if sha256(raw).hexdigest() != digest:
                raise ValueError("checkpoint call record hash mismatch")
            if (
                not self._store.exists(relative)
                or self._store.read_bytes(relative) != raw
            ):
                self._store.atomic_write(relative, raw)
            usage = payload.get("usage")
            if type(usage) is not dict:
                raise ValueError("checkpoint call usage is invalid")
            try:
                for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
                    value = usage[key]
                    if type(value) is not int or value < 0:
                        raise ValueError
                    expected_usage[key] = int(expected_usage[key]) + value
                cost = usage["cost_usd"]
                if type(cost) not in {int, float} or cost < 0:
                    raise ValueError
                expected_usage["cost_usd"] = float(
                    expected_usage["cost_usd"]
                ) + float(cost)
            except (KeyError, ValueError) as exc:
                raise ValueError("checkpoint call usage is invalid") from exc
        actual_paths = {
            path
            for path in self._store.list_files("phase1")
            if re.search(r"/call-[0-9]{2}\.json$", path)
        }
        if actual_paths != expected_paths:
            raise ValueError("call artifact set mismatch")
        if state.get("usage") != expected_usage:
            raise ValueError("call usage mismatch")
        by_phase = state.get("usage_by_phase")
        if (
            type(by_phase) is not dict
            or by_phase.get("phase1") != expected_usage
            or by_phase.get("phase2") != _zero_usage()
        ):
            raise ValueError("call usage mismatch")

    def _ensure_terminal_artifacts(self, state: WorkflowStateV44) -> None:
        """Verify terminal checkpoint bindings, rebuilding only local artifacts."""
        self._verify_call_artifacts(state)
        if state.get("phase1_status") not in {"accepted", "provisional"}:
            return
        required = (
            "claim_map",
            "claim_retrieval",
            "taxonomy_neighborhood",
            "adjudication",
            "investigation",
        )
        missing = [name for name in required if name not in state]
        if missing:
            raise ValueError(
                "terminal checkpoint is missing required state: " + ", ".join(missing)
            )
        claim_map = _state_model(ClaimMapV44, state["claim_map"])
        retrieval = _state_model(
            ClaimRetrievalManifestV44, state["claim_retrieval"]
        )
        neighborhood = _state_model(
            TaxonomyNeighborhoodManifestV44, state["taxonomy_neighborhood"]
        )
        adjudication = _state_model(
            RationaleAdjudicationV44, state["adjudication"]
        )
        investigation = _state_model(InvestigationV44, state["investigation"])
        assert isinstance(claim_map, ClaimMapV44)
        assert isinstance(retrieval, ClaimRetrievalManifestV44)
        assert isinstance(neighborhood, TaxonomyNeighborhoodManifestV44)
        assert isinstance(adjudication, RationaleAdjudicationV44)
        assert isinstance(investigation, InvestigationV44)
        models = {
            "claim-map": (claim_map, "claim_map_sha256"),
            "claim-retrieval": (retrieval, "claim_retrieval_sha256"),
            "taxonomy-neighborhood": (
                neighborhood,
                "taxonomy_neighborhood_sha256",
            ),
            "adjudication": (adjudication, "adjudication_sha256"),
            "investigation": (investigation, "investigation_sha256"),
        }
        digests = {
            name: sha256(_model_bytes(model)).hexdigest()
            for name, (model, _field) in models.items()
        }
        for name, (_model, field) in models.items():
            if state.get(field) != digests[name]:
                raise ValueError(f"terminal checkpoint hash binding mismatch: {field}")
        expected_bindings = {
            "claim_map_sha256": digests["claim-map"],
            "claim_retrieval_sha256": digests["claim-retrieval"],
            "taxonomy_neighborhood_sha256": digests["taxonomy-neighborhood"],
            "adjudication_sha256": digests["adjudication"],
        }
        if any(
            getattr(investigation, field) != expected
            for field, expected in expected_bindings.items()
        ):
            raise ValueError("terminal investigation artifact bindings are inconsistent")
        if retrieval.claim_map_sha256 != digests["claim-map"]:
            raise ValueError("terminal retrieval does not bind the claim map")
        if any(
            value != self.settings.episode_slug
            for value in (
                claim_map.episode_slug,
                retrieval.episode_slug,
                adjudication.episode_slug,
                investigation.episode_slug,
            )
        ):
            raise ValueError("terminal checkpoint episode bindings are inconsistent")
        if _model_payload(investigation.claim_retrieval_manifest) != _model_payload(
            retrieval
        ) or _model_payload(
            investigation.taxonomy_neighborhood_manifest
        ) != _model_payload(neighborhood):
            raise ValueError("terminal investigation embeds stale retrieval manifests")
        if investigation.investigation_status != state["phase1_status"]:
            raise ValueError("terminal investigation and workflow statuses disagree")
        if state.get("claim_retrieval_manifest") != state["claim_retrieval"]:
            raise ValueError("terminal claim retrieval state aliases disagree")
        if state.get("taxonomy_neighborhood_manifest") != state[
            "taxonomy_neighborhood"
        ]:
            raise ValueError("terminal taxonomy neighborhood state aliases disagree")
        expected_registry = {
            row.evidence_id: _model_payload(row) for row in retrieval.evidence_registry
        }
        if state.get("evidence_registry") != expected_registry:
            raise ValueError("terminal evidence registry state is stale")

        for name, (model, _field) in models.items():
            json_path = f"phase1/{name}.json"
            sha_path = f"phase1/{name}.sha256"
            expected_bytes = _model_bytes(model)
            expected_digest = digests[name]
            try:
                valid = (
                    self._store.exists(json_path)
                    and self._store.exists(sha_path)
                    and self._store.read_bytes(json_path) == expected_bytes
                    and self._store.read_bytes(sha_path).decode("ascii").strip()
                    == expected_digest
                )
            except (OSError, UnicodeDecodeError):
                valid = False
            if not valid:
                _freeze_model(self._store, "phase1", name, model)
        registry = list(expected_registry.values())
        registry_path = "phase1/evidence-registry.json"
        try:
            registry_valid = json.loads(
                self._store.read_bytes(registry_path).decode("utf-8")
            ) == registry
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            registry_valid = False
        if not registry_valid:
            self._store.atomic_write(
                registry_path, _canonical_json_bytes(registry)
            )

    def _persist(self, result: WorkflowStateV44) -> WorkflowStateV44:
        if result.get("run_fingerprint") != self.run_fingerprint:
            raise ValueError("run fingerprint mismatch")
        if (
            result.get("pending_call") is not None
            or result.get("pending_response") is not None
        ):
            raise ValueError("terminal state retains a pending provider call")
        self._verify_owner_artifact(result)
        self._validate_wal_against_state(result, settle=False)
        if self._load_wal_entries():
            raise ValueError("provider-call WAL set is not settled")
        result["phase2_status"] = "not_run"
        self._ensure_terminal_artifacts(result)
        self._store.atomic_write("state.json", _canonical_json_bytes(result))
        return result

    def _prepare_new_run_artifacts(self) -> None:
        """Remove only artifacts owned by an earlier v4.4 invocation."""
        managed = set(self._store.list_files("phase1"))
        for relative in managed:
            if re.search(r"/call-[0-9]{2}\.json$", relative) or relative in {
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
            }:
                self._store.unlink(relative)
        self._store.unlink("state.json", missing_ok=True)

    def invoke(self, thread_id: str) -> WorkflowStateV44:
        if not thread_id:
            raise ValueError("thread_id is required")
        with _ArtifactStore.open(self.settings.run_root) as store:
            self._store = store
            with self._run_lock():
                if self._checkpoint_state(thread_id):
                    raise ValueError("checkpoint thread was already used")
                if self._load_wal_entries():
                    raise ValueError(
                        "unmatched provider-call WAL has an uncertain outcome/cost"
                    )
                self._establish_new_owner(thread_id)
                self._prepare_new_run_artifacts()
                result = self.graph.invoke(
                    self._initial_state(thread_id),
                    {"configurable": {"thread_id": thread_id}},
                )
                self._confirm_checkpointed_calls(thread_id, result)
                return self._persist(result)

    def resume(self, thread_id: str) -> WorkflowStateV44:
        if not thread_id:
            raise ValueError("thread_id is required")
        with _ArtifactStore.open(self.settings.run_root) as store:
            self._store = store
            with self._run_lock():
                checkpoint = self._checkpoint_state(thread_id)
                if not checkpoint:
                    raise ValueError("checkpoint thread does not exist")
                if checkpoint.get("run_fingerprint") != self.run_fingerprint:
                    raise ValueError("run fingerprint mismatch")
                if checkpoint.get("run_thread_id") != thread_id:
                    raise ValueError("checkpoint thread ownership mismatch")
                self._validate_pending_resume(checkpoint)
                self._validate_or_rebuild_owner(thread_id, checkpoint)
                self._verify_call_artifacts(checkpoint)
                self._validate_wal_against_state(checkpoint, settle=True)
                result = self.graph.invoke(
                    None, {"configurable": {"thread_id": thread_id}}
                )
                self._confirm_checkpointed_calls(thread_id, result)
                return self._persist(result)
