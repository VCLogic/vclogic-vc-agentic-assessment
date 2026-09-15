#!/usr/bin/env python3
"""Evaluate Phase 1 v4.4 broad/core views against the frozen v4.3 canary."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import csv
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
from statistics import fmean
import tempfile
from typing import Any, Callable, Mapping, Sequence

from vc_clone_graph.artifacts import verify_phase1_artifacts
from vc_clone_graph.phase1_evaluation import (
    Phase1Case,
    TaxonomyLabel,
    _prediction_rationales,
    _reference_rationales,
    evaluate_cases,
    score_set,
    sha256_file,
)


DEFAULT_REGISTRY = Path("evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json")
DEFAULT_REFERENCES = Path("evaluation/phase1_ground_truth_rationales")
DEFAULT_TAXONOMY = Path("../agentic-vc-clone-framework/taxonomy/codebook_v_final.json")
DEFAULT_MANIFEST = Path(
    "reports/evaluation/phase1-v43-diagnostic-2026-08-17/canary-manifest.json"
)
DEFAULT_RUNS = Path("outputs/phase1-v44-canary-2026-08-17/investors")
DEFAULT_OUTPUT = Path("reports/evaluation/phase1-v44-canary-2026-08-17")
DEFAULT_STATUS = DEFAULT_OUTPUT / "status.json"
SCIENTIFIC_STATUS = "development_canary_not_untouched_holdout"
FROZEN_CANARY_SCHEMA = "phase1-v43-canary-manifest-v1"
FROZEN_CANARY_INTERVENTION = "contrastive_within_family_mapping"
FROZEN_CANARY_SHA256 = "4caffcc057c17c227d6ef80d6a57fd2ea5bf4f97cc53275037c8c00dc0a76c45"
EXPECTED_OUTPUTS = (
    "metrics.json",
    "canonical-result.json",
    "broad-result.json",
    "core-result.json",
    "paired-cases.csv",
    "taxonomy-opportunity.csv",
    "process-diagnostics.csv",
    "evaluation.md",
)
FORBIDDEN_INFERENCE_KEYS = {
    "actual_decision",
    "actual_label",
    "ground_truth",
    "reference_rationale",
    "reference_rationales",
    "target_decision",
}


@dataclass(frozen=True)
class _RunSnapshot:
    exists: bool
    files: frozenset[str]
    directories: frozenset[str]
    json_objects: Mapping[str, Any]
    bytes_by_path: Mapping[str, bytes]
    hashes: Mapping[str, str]
    phase2_exists: bool


@dataclass(frozen=True)
class _CapturedFile:
    path: Path
    raw: bytes
    sha256: str


@dataclass(frozen=True)
class _CapturedGlob:
    base: Path
    pattern: str
    matches: tuple[Path, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _object(value: object, context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{context} must be an object")
    return value


def _array(value: object, context: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{context} must be a list")
    return value


def _load_json_object(path: Path, context: str) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), context)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {context}: {path}") from exc


def _capture_file(path: Path, context: str) -> _CapturedFile:
    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        current_fd = os.open(absolute.anchor, directory_flags)
    except OSError as exc:
        raise ValueError(f"cannot safely open {context}: {path}") from exc
    descriptor = -1
    try:
        for part in parts[1:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        descriptor = os.open(parts[-1], file_flags, dir_fd=current_fd)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{context} is not a regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            raw = stream.read()
    except OSError as exc:
        raise ValueError(f"cannot safely read {context}: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(current_fd)
    return _CapturedFile(absolute, raw, sha256(raw).hexdigest())


def _captured_json(captured: _CapturedFile, context: str) -> Any:
    try:
        return json.loads(captured.raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse captured {context}") from exc


def _recheck_captured_files(captured: Sequence[_CapturedFile]) -> None:
    for source in captured:
        observed = _capture_file(source.path, f"recheck {source.path}")
        if observed.raw != source.raw or observed.sha256 != source.sha256:
            raise ValueError(f"evaluation input mutated during run: {source.path}")


def _captured_binding(captured: _CapturedFile) -> dict[str, str]:
    return {"path": str(captured.path), "sha256": captured.sha256}


def _parse_taxonomy_capture(captured: _CapturedFile) -> dict[str, TaxonomyLabel]:
    rows = _array(_captured_json(captured, "taxonomy"), "taxonomy")
    taxonomy: dict[str, TaxonomyLabel] = {}
    for index, value in enumerate(rows):
        row = _object(value, f"taxonomy row {index}")
        label = row.get("label")
        definition = row.get("definition")
        parent = row.get("coarse_parent")
        if not all(type(item) is str and item for item in (label, definition, parent)):
            raise ValueError(f"malformed taxonomy row {index}")
        if label in taxonomy:
            raise ValueError(f"duplicate taxonomy label: {label}")
        taxonomy[label] = TaxonomyLabel(label, definition, parent)
    if len(taxonomy) != 44:
        raise ValueError(f"expected 44 taxonomy labels, found {len(taxonomy)}")
    return taxonomy


def _parse_canary_manifest_capture(
    captured: _CapturedFile,
) -> dict[tuple[str, str], dict[str, str]]:
    payload = _object(_captured_json(captured, "canary manifest"), "canary manifest")
    return _validate_canary_manifest(payload, captured.sha256)


def load_canary_manifest(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    captured = _capture_file(path, "canary manifest")
    payload = _object(_captured_json(captured, "canary manifest"), "canary manifest")
    return _validate_canary_manifest(payload, captured.sha256)


def _validate_canary_manifest(
    payload: Mapping[str, Any], observed_sha256: str
) -> dict[tuple[str, str], dict[str, str]]:
    if payload.get("schema") != FROZEN_CANARY_SCHEMA:
        raise ValueError("canary manifest schema is not the frozen v4.3 schema")
    if payload.get("scientific_status") != SCIENTIFIC_STATUS:
        raise ValueError("canary manifest scientific status is invalid")
    if payload.get("selected_intervention") != FROZEN_CANARY_INTERVENTION:
        raise ValueError("canary manifest selected intervention is invalid")
    rows = _array(payload.get("cases"), "canary manifest cases")
    if len(rows) != 18:
        raise ValueError("canary manifest must contain the frozen 18-case cohort")
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for index, value in enumerate(rows):
        row = _object(value, f"canary case {index}")
        required = ("vc_slug", "episode_slug", "actual_decision", "role")
        if any(type(row.get(key)) is not str or not row[key].strip() for key in required):
            raise ValueError(f"canary case {index} is incomplete")
        if row["actual_decision"] not in {"In", "Out"}:
            raise ValueError(f"canary manifest decision is invalid at case {index}")
        if row["role"] not in {"coverage", "precision", "control"}:
            raise ValueError(f"canary manifest role is invalid at case {index}")
        key = str(row["vc_slug"]), str(row["episode_slug"])
        if key in selected:
            raise ValueError(f"canary manifest contains duplicate case: {key}")
        selected[key] = {key_name: str(row[key_name]) for key_name in required}
    vc_counts = Counter(key[0] for key in selected)
    if len(vc_counts) != 6 or set(vc_counts.values()) != {3}:
        raise ValueError("canary manifest must contain six investors with three cases each")
    grouped_roles: dict[str, list[str]] = {}
    for (vc_slug, _), row in selected.items():
        grouped_roles.setdefault(vc_slug, []).append(row["role"])
    if any(
        roles != ["coverage", "precision", "control"]
        for roles in grouped_roles.values()
    ):
        raise ValueError("canary manifest investor role ordering is invalid")
    if observed_sha256 != FROZEN_CANARY_SHA256:
        raise ValueError("canary manifest composition/order differs from frozen v4.3")
    return selected


def case_input_bindings(
    cases: Sequence[Phase1Case], *, project_root: Path
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for case in cases:
        if case.artifact_path is None or case.reference_path is None:
            raise ValueError("canonical/reference per-case artifact path is missing")
        canonical = Path(case.artifact_path)
        reference = Path(case.reference_path)
        if not canonical.is_absolute():
            canonical = project_root / canonical
        if not reference.is_absolute():
            reference = project_root / reference
        sidecar = canonical.with_suffix(".sha256")
        if not canonical.is_file() or not sidecar.is_file() or not reference.is_file():
            raise ValueError("canonical/reference per-case artifact is missing")
        expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
        observed = sha256_file(canonical)
        if expected != observed:
            raise ValueError("canonical investigation hash mismatch")
        reference_payload = _load_json_object(reference, "automated reference record")
        if (
            reference_payload.get("vc_slug") != case.vc_slug
            or reference_payload.get("episode_slug") != case.episode_slug
        ):
            raise ValueError("automated reference record identity mismatch")
        bindings.append(
            {
                "vc_slug": case.vc_slug,
                "episode_slug": case.episode_slug,
                "canonical_investigation": {
                    "path": str(canonical.resolve()),
                    "sha256": observed,
                },
                "canonical_sidecar": {
                    "path": str(sidecar.resolve()),
                    "sha256": sha256_file(sidecar),
                },
                "automated_reference": {
                    "path": str(reference.resolve()),
                    "sha256": sha256_file(reference),
                },
            }
        )
    return bindings


def _capture_from_tree(
    root: Path, snapshot: _RunSnapshot, path: Path, context: str
) -> _CapturedFile:
    absolute_root = Path(os.path.abspath(os.fspath(root)))
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = absolute.relative_to(absolute_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{context} is outside its captured root") from exc
    try:
        raw = snapshot.bytes_by_path[relative]
        digest = snapshot.hashes[relative]
    except KeyError as exc:
        raise ValueError(f"{context} is missing from captured root") from exc
    return _CapturedFile(absolute, raw, digest)


def _rebuild_cases_from_captures(
    discovered: Sequence[Phase1Case],
    *,
    taxonomy: Mapping[str, TaxonomyLabel],
    project_root: Path,
    reference_root: Path,
    reference_snapshot: _RunSnapshot,
) -> tuple[list[Phase1Case], list[_CapturedFile], list[dict[str, Any]]]:
    rebuilt: list[Phase1Case] = []
    sources: list[_CapturedFile] = []
    bindings: list[dict[str, Any]] = []
    for case in discovered:
        if case.artifact_path is None or case.reference_path is None:
            raise ValueError("canonical/reference per-case artifact path is missing")
        canonical_path = Path(case.artifact_path)
        reference_path = Path(case.reference_path)
        if not canonical_path.is_absolute():
            canonical_path = project_root / canonical_path
        if not reference_path.is_absolute():
            reference_path = project_root / reference_path
        sidecar_path = canonical_path.with_suffix(".sha256")
        canonical = _capture_file(canonical_path, "canonical investigation")
        sidecar = _capture_file(sidecar_path, "canonical sidecar")
        reference = _capture_from_tree(
            reference_root,
            reference_snapshot,
            reference_path,
            "automated reference record",
        )
        try:
            expected = sidecar.raw.decode("utf-8").strip().split()[0]
        except (UnicodeDecodeError, IndexError) as exc:
            raise ValueError("canonical investigation sidecar is invalid") from exc
        if expected != canonical.sha256:
            raise ValueError("canonical investigation hash mismatch")
        predicted_payload = _object(
            _captured_json(canonical, "canonical investigation"),
            "canonical investigation",
        )
        reference_payload = _object(
            _captured_json(reference, "automated reference record"),
            "automated reference record",
        )
        if predicted_payload.get("episode_slug") != case.episode_slug:
            raise ValueError("canonical investigation episode mismatch")
        if reference_payload.get("schema") != "phase1-rationale-reference-v1":
            raise ValueError("automated reference schema is invalid")
        if (
            reference_payload.get("vc_slug") != case.vc_slug
            or reference_payload.get("episode_slug") != case.episode_slug
            or reference_payload.get("actual_decision") != case.actual_decision
        ):
            raise ValueError("automated reference identity/decision mismatch")
        rebuilt.append(
            replace(
                case,
                source_tier=str(reference_payload.get("source_tier")),
                source_format=str(reference_payload.get("source_format")),
                predicted=_prediction_rationales(
                    predicted_payload, taxonomy, str(canonical.path)
                ),
                reference=_reference_rationales(
                    reference_payload, taxonomy, str(reference.path)
                ),
                artifact_path=canonical.path,
                reference_path=reference.path,
            )
        )
        sources.extend((canonical, sidecar, reference))
        bindings.append(
            {
                "vc_slug": case.vc_slug,
                "episode_slug": case.episode_slug,
                "canonical_investigation": _captured_binding(canonical),
                "canonical_sidecar": _captured_binding(sidecar),
                "automated_reference": _captured_binding(reference),
            }
        )
    return rebuilt, sources, bindings


def _discover_cases_from_captured_registry(
    registry: _CapturedFile,
    *,
    selected: Mapping[tuple[str, str], Mapping[str, str]],
    reference_root: Path,
    reference_snapshot: _RunSnapshot,
) -> tuple[
    list[Phase1Case],
    list[_CapturedFile],
    dict[tuple[str, str], _CapturedFile],
    list[_CapturedGlob],
]:
    payload = _object(
        _captured_json(registry, "evaluation registry"), "evaluation registry"
    )
    if payload.get("schema") != "canonical-vc-evaluation-registry-v1":
        raise ValueError("unsupported evaluation registry schema")
    investors = _object(payload.get("investors"), "registry investors")
    selected_vcs = {key[0] for key in selected}
    if not selected_vcs <= set(investors):
        raise ValueError("canary contains an investor absent from registry")

    reference_paths: dict[tuple[str, str], Path] = {}
    for relative, value in reference_snapshot.json_objects.items():
        if not relative.startswith("records/") or type(value) is not dict:
            continue
        key = str(value.get("vc_slug")), str(value.get("episode_slug"))
        if key in reference_paths:
            raise ValueError(f"duplicate captured reference record: {key}")
        reference_paths[key] = reference_root / relative

    auxiliary: list[_CapturedFile] = []
    summaries_by_key: dict[tuple[str, str], _CapturedFile] = {}
    glob_captures: list[_CapturedGlob] = []
    cases: list[Phase1Case] = []
    registry_parent = registry.path.parent
    expected_reference_keys: set[tuple[str, str]] = set()
    for vc_slug in sorted(investors):
        investor = _object(investors[vc_slug], f"registry investor {vc_slug}")
        label_capture = _capture_file(
            registry_parent / str(investor.get("label_file")),
            f"registry label file {vc_slug}",
        )
        auxiliary.append(label_capture)
        label_rows = _array(
            _captured_json(label_capture, f"label file {vc_slug}"),
            f"label file {vc_slug}",
        )
        labels: dict[str, str] = {}
        for value in label_rows:
            row = _object(value, f"label row {vc_slug}")
            if row.get("evaluation_eligible") is not True:
                continue
            episode = row.get("episode_slug")
            decision = row.get("pitch_window_decision")
            if type(episode) is not str or decision not in {"In", "Out"}:
                raise ValueError(f"invalid eligible label for {vc_slug}")
            if episode in labels:
                raise ValueError(f"duplicate eligible label for {episode}")
            labels[episode] = str(decision)
        if len(labels) != int(investor.get("eligible_count", -1)):
            raise ValueError(f"eligible label count mismatch for {vc_slug}")

        artifacts: dict[str, _CapturedFile] = {}
        sources = _array(investor.get("sources"), f"registry sources {vc_slug}")
        for source_value in sources:
            source = _object(source_value, f"registry source {vc_slug}")
            kind = source.get("kind")
            paths: list[Path] = []
            if kind == "summary_files":
                values = _array(source.get("paths"), "summary_files paths")
                paths = [registry_parent / str(value) for value in values]
            elif kind == "summary_glob":
                pattern = source.get("path")
                if type(pattern) is not str or not pattern:
                    raise ValueError("summary_glob source requires path")
                matches = tuple(
                    sorted(
                        Path(os.path.abspath(os.fspath(path)))
                        for path in registry_parent.glob(pattern)
                    )
                )
                glob_captures.append(
                    _CapturedGlob(registry_parent, pattern, matches)
                )
                paths = list(matches)
            elif kind in {"batch_status", "recovery_status"}:
                status_relative = source.get("path")
                if type(status_relative) is not str or not status_relative:
                    raise ValueError(f"{kind} source requires path")
                status_capture = _capture_file(
                    registry_parent / status_relative, f"{kind} source"
                )
                auxiliary.append(status_capture)
                status_payload = _object(
                    _captured_json(status_capture, kind), kind
                )
                records = _array(
                    status_payload.get("completed_records"),
                    f"{kind} completed records",
                )
                for record_value in records:
                    record = _object(record_value, f"{kind} completed record")
                    if kind == "recovery_status" and record.get("vc") != source.get(
                        "vc_name"
                    ):
                        continue
                    artifact_root = record.get("artifact_root")
                    if type(artifact_root) is not str:
                        raise ValueError(f"{kind} record has no artifact_root")
                    root = Path(artifact_root)
                    if not root.is_absolute():
                        root = registry_parent.parent / root
                    paths.append(root / "summary.json")
            else:
                raise ValueError(f"unsupported artifact source kind: {kind!r}")

            source_rows: dict[str, _CapturedFile] = {}
            for path in paths:
                summary = _capture_file(path, f"canonical summary {vc_slug}")
                auxiliary.append(summary)
                summary_payload = _object(
                    _captured_json(summary, "canonical summary"),
                    "canonical summary",
                )
                decision = (
                    summary_payload.get("decision")
                    if type(summary_payload.get("decision")) is dict
                    else {}
                )
                episode = str(
                    summary_payload.get("episode_slug")
                    or decision.get("episode_slug")
                    or summary.path.parent.name
                )
                if episode in source_rows:
                    raise ValueError(
                        f"duplicate artifact at equal precedence for {episode}"
                    )
                source_rows[episode] = summary
            artifacts.update(source_rows)

        missing = sorted(set(labels) - set(artifacts))
        unexpected = sorted(set(artifacts) - set(labels))
        if missing:
            raise ValueError(
                f"missing canonical artifacts for {vc_slug}: {', '.join(missing)}"
            )
        if unexpected:
            raise ValueError(
                f"unexpected canonical artifacts for {vc_slug}: "
                f"{', '.join(unexpected)}"
            )
        expected_reference_keys.update((vc_slug, episode) for episode in labels)
        selected_episodes = {
            episode for investor_slug, episode in selected if investor_slug == vc_slug
        }
        for episode in sorted(selected_episodes):
            key = vc_slug, episode
            manifest_row = selected[key]
            if labels.get(episode) != manifest_row["actual_decision"]:
                raise ValueError(f"canary/label decision mismatch for {episode}")
            if key not in reference_paths:
                raise ValueError(f"captured reference is missing for {key}")
            summary = artifacts[episode]
            summaries_by_key[key] = summary
            cases.append(
                Phase1Case(
                    vc_slug=vc_slug,
                    vc_name=str(investor.get("display_name")),
                    episode_slug=episode,
                    actual_decision=manifest_row["actual_decision"],
                    source_tier="",
                    source_format="",
                    predicted=(),
                    reference=(),
                    artifact_path=summary.path.parent / "phase1/investigation.json",
                    reference_path=reference_paths[key],
                )
            )
    if set(reference_paths) != expected_reference_keys:
        missing = sorted(expected_reference_keys - set(reference_paths))
        unexpected = sorted(set(reference_paths) - expected_reference_keys)
        raise ValueError(
            f"reference coverage mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    return cases, auxiliary, summaries_by_key, glob_captures


def _recheck_captured_globs(captured: Sequence[_CapturedGlob]) -> None:
    for source in captured:
        observed = tuple(
            sorted(
                Path(os.path.abspath(os.fspath(path)))
                for path in source.base.glob(source.pattern)
            )
        )
        if observed != source.matches:
            raise ValueError(
                f"evaluation input glob inventory mutated: {source.pattern}"
            )


def _contains_forbidden_key(value: object) -> bool:
    if type(value) is dict:
        return any(
            str(key).casefold() in FORBIDDEN_INFERENCE_KEYS
            or _contains_forbidden_key(item)
            for key, item in value.items()
        )
    if type(value) is list:
        return any(_contains_forbidden_key(item) for item in value)
    if type(value) is str:
        if re.search(
            r'''(?i)["'](?:actual[_-]decision|target[_-]decision|'''
            r'''reference[_-]rationales?)["']\s*:''',
            value,
        ):
            return True
        normalized = " ".join(value.casefold().replace("_", " ").replace("-", " ").split())
        return any(
            marker in normalized
            for marker in (
                "actual target decision",
                "current target decision",
                "target outcome",
                "reference rationale",
                "evaluation report",
                "evaluation payload",
                "evaluation result",
                "evaluation score",
                "previous prediction",
                "ground truth decision",
                "ground truth outcome",
                "ground truth label",
                "gold decision",
                "gold outcome",
            )
        )
    return False


def _confined_tree_inventory(root_fd: int) -> tuple[frozenset[str], frozenset[str]]:
    directories: set[str] = set()
    files: set[str] = set()
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW

    def walk(directory_fd: int, prefix: str) -> None:
        for name in sorted(os.listdir(directory_fd)):
            relative = f"{prefix}{name}"
            try:
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise ValueError("non-usable snapshot inventory is inaccessible") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(
                    f"non-usable snapshot inventory contains a symlink: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
                try:
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise ValueError(
                        "non-usable snapshot directory is unsafe"
                    ) from exc
                try:
                    walk(child_fd, relative + "/")
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(metadata.st_mode):
                files.add(relative)
            else:
                raise ValueError(
                    f"non-usable snapshot inventory contains a special file: {relative}"
                )

    duplicate = os.dup(root_fd)
    try:
        walk(duplicate, "")
    finally:
        os.close(duplicate)
    return frozenset(directories), frozenset(files)


def _snapshot_nonusable_run(runs_root: Path, run_root: Path) -> _RunSnapshot:
    from vc_clone_graph.workflow_v44 import _ArtifactStore

    trusted = Path(os.path.abspath(os.fspath(runs_root)))
    target = Path(os.path.abspath(os.fspath(run_root)))
    if target == trusted or not target.is_relative_to(trusted):
        raise ValueError("non-usable run root is not confined beneath output root")
    parts = target.parts
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        current_fd = os.open(target.anchor, flags)
    except OSError as exc:
        raise ValueError("non-usable run root ancestry is unsafe") from exc
    try:
        for index, part in enumerate(parts[1:], start=1):
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if index != len(parts) - 1:
                    raise ValueError(
                        "non-usable run root has a missing ancestor"
                    )
                return _RunSnapshot(
                    False, frozenset(), frozenset(), {}, {}, {}, False
                )
            except OSError as exc:
                raise ValueError(
                    "non-usable run root ancestry contains a symlink or unsafe "
                    "component"
                ) from exc
            os.close(current_fd)
            current_fd = next_fd
        store = _ArtifactStore(current_fd, target)
        current_fd = -1
        with store:
            try:
                directories, paths = _confined_tree_inventory(store.root_fd)
                phase2_exists = "phase2" in directories or "phase2" in paths
            except (OSError, ValueError) as exc:
                raise ValueError(
                    "non-usable run root contains an unsafe artifact"
                ) from exc
            json_objects: dict[str, Any] = {}
            bytes_by_path: dict[str, bytes] = {}
            hashes: dict[str, str] = {}
            for relative in paths:
                try:
                    raw = store.read_bytes(relative)
                except (
                    OSError,
                    ValueError,
                ) as exc:
                    raise ValueError(
                        "suspicious partial v4.4 artifacts for non-usable case"
                    ) from exc
                bytes_by_path[relative] = raw
                hashes[relative] = sha256(raw).hexdigest()
                if relative.endswith(".json"):
                    try:
                        json_objects[relative] = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            "suspicious partial v4.4 artifacts for non-usable case"
                        ) from exc
            return _RunSnapshot(
                True,
                frozenset(paths),
                frozenset(directories),
                json_objects,
                bytes_by_path,
                hashes,
                phase2_exists,
            )
    finally:
        if current_fd >= 0:
            os.close(current_fd)


def _verify_nonusable_snapshot(
    snapshot: _RunSnapshot, verifier: Callable[[Path], None]
) -> None:
    from vc_clone_graph.workflow_v44 import _ArtifactStore

    if not snapshot.exists:
        raise ValueError("post-completion v4.4 snapshot is absent")

    def assert_exact_copy(store: Any) -> None:
        try:
            observed_directories, observed_files = _confined_tree_inventory(
                store.root_fd
            )
        except (OSError, ValueError) as exc:
            raise ValueError("materialized v4.4 snapshot inventory is unsafe") from exc
        if observed_directories != snapshot.directories:
            raise ValueError("materialized v4.4 snapshot directory inventory mismatch")
        if observed_files != snapshot.files:
            raise ValueError("materialized v4.4 snapshot file inventory mismatch")
        for relative in sorted(snapshot.files):
            try:
                raw = store.read_bytes(relative)
            except (OSError, ValueError) as exc:
                raise ValueError("materialized v4.4 snapshot is unsafe") from exc
            if raw != snapshot.bytes_by_path[relative]:
                raise ValueError("materialized v4.4 snapshot byte mismatch")

    with tempfile.TemporaryDirectory(prefix="phase1-v44-verify-") as directory:
        materialized = Path(directory) / "run"
        try:
            store_context = _ArtifactStore.open(materialized)
        except (OSError, ValueError) as exc:
            raise ValueError("cannot create private v4.4 verification root") from exc
        with store_context as store:
            for relative in sorted(
                snapshot.directories, key=lambda value: (value.count("/"), value)
            ):
                descriptor = store.open_dir(relative, create=True)
                os.close(descriptor)
            for relative in sorted(snapshot.files):
                store.atomic_write(relative, snapshot.bytes_by_path[relative])
            assert_exact_copy(store)
            verifier(materialized)
            assert_exact_copy(store)


def _after_nonusable_snapshot() -> None:
    """Test seam after the held-dirfd snapshot becomes authoritative."""


def _after_usable_snapshot() -> None:
    """Test seam after usable verification and snapshot capture."""


def _scan_nonusable_snapshot(snapshot: _RunSnapshot) -> None:
    for value in snapshot.json_objects.values():
        if _contains_forbidden_key(value):
            raise ValueError("v4.4 run contains a leaked target/evaluation label")


def _strict_nonnegative_int(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{context} must be a nonnegative integer")
    return value


def _strict_nonnegative_float(value: object, context: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{context} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{context} must be a finite nonnegative number")
    return result


def _runner_status_index(
    runner_status: Mapping[str, Any],
    selected: Mapping[tuple[str, str], Mapping[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    if runner_status.get("schema") != "phase1-v44-canary-status-v1":
        raise ValueError("v4.4 runner status schema is invalid")
    rows = _array(runner_status.get("cases"), "v4.4 runner status cases")
    if len(rows) != len(selected):
        raise ValueError("v4.4 runner status coverage differs from canary manifest")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    cumulative = 0.0
    selected_keys = list(selected)
    for position, value in enumerate(rows, start=1):
        row = _object(value, f"v4.4 runner status row {position}")
        key = str(row.get("vc_slug")), str(row.get("episode_slug"))
        if key != selected_keys[position - 1] or row.get("position") != position:
            raise ValueError("v4.4 runner status order/identity differs from manifest")
        manifest_row = selected[key]
        if (
            row.get("actual_decision") != manifest_row["actual_decision"]
            or row.get("role") != manifest_row["role"]
        ):
            raise ValueError("v4.4 runner status labels differ from manifest")
        if key in indexed:
            raise ValueError("v4.4 runner status contains duplicate cases")
        status = row.get("status")
        if type(status) is not str:
            raise ValueError("v4.4 runner case status is invalid")
        if status in {"completed", "completed_existing"}:
            category = str(row.get("phase1_status"))
            if category not in {"accepted", "provisional"}:
                raise ValueError("completed runner row lacks a usable Phase 1 status")
        elif status in {
            "failed",
            "failed_with_state",
            "failed_rate_ceiling_breach",
            "failed_cost_ceiling_exceeded",
            "resume_exhausted",
        }:
            category = "failed"
        elif status in {"not_started", "not_started_hard_ceiling"}:
            category = "not_started"
        else:
            raise ValueError("v4.4 runner status is incomplete or ambiguous")
        cost = _strict_nonnegative_float(row.get("cost_usd"), "runner cost_usd")
        cumulative += cost
        if not math.isclose(
            _strict_nonnegative_float(
                row.get("cumulative_cost_usd"), "runner cumulative_cost_usd"
            ),
            cumulative,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("v4.4 runner cumulative cost is inconsistent")
        for field in ("calls", "input_tokens", "output_tokens"):
            _strict_nonnegative_int(row.get(field), f"runner {field}")
        indexed[key] = {**row, "case_category": category}
    if not math.isclose(
        _strict_nonnegative_float(
            runner_status.get("cumulative_cost_usd"),
            "runner total cumulative_cost_usd",
        ),
        cumulative,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("v4.4 runner total cost is inconsistent")
    return indexed


def _labels(rows: object, context: str) -> set[str]:
    result: set[str] = set()
    for index, value in enumerate(_array(rows, context)):
        row = _object(value, f"{context} row {index}")
        label = row.get("taxonomy_label")
        if type(label) is not str or not label:
            raise ValueError(f"{context} contains an invalid taxonomy label")
        result.add(label)
    return result


def _rich_recall(result: Mapping[str, Any], subset: str) -> float:
    row = next(
        (
            value
            for value in result["rich_subset_rows"]
            if value["subset"] == subset
        ),
        None,
    )
    return float(row["recall"]) if row is not None else 0.0


def _vc_f1(result: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(row["value"]): float(row["micro_f1"])
        for row in result["scope_rows"]
        if row["dimension"] == "vc"
    }


def _method_by_role(
    cases: Sequence[Phase1Case],
    taxonomy: Mapping[str, TaxonomyLabel],
    selected: Mapping[tuple[str, str], Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    roles = sorted({row["role"] for row in selected.values()})
    result: dict[str, dict[str, Any]] = {}
    for role in roles:
        scoped = [
            case
            for case in cases
            if selected[(case.vc_slug, case.episode_slug)]["role"] == role
        ]
        if scoped:
            result[role] = evaluate_cases(scoped, taxonomy)["overall"]
    return result


def _paired_label_change(
    before: Sequence[Phase1Case],
    after: Sequence[Phase1Case],
    taxonomy: Mapping[str, TaxonomyLabel],
) -> dict[str, int]:
    improved = worsened = 0
    for old, new in zip(before, after, strict=True):
        old_labels = {row.label for row in old.predicted}
        new_labels = {row.label for row in new.predicted}
        reference = {row.label for row in old.reference}
        for label in taxonomy:
            old_correct = (label in old_labels) == (label in reference)
            new_correct = (label in new_labels) == (label in reference)
            improved += int(new_correct and not old_correct)
            worsened += int(old_correct and not new_correct)
    return {
        "improved": improved,
        "worsened": worsened,
        "discordant": improved + worsened,
    }


def promotion_criteria(
    *,
    usable_cases: int,
    canonical: Mapping[str, float],
    broad: Mapping[str, float],
    core: Mapping[str, float],
    canonical_primary_explicit_recall: float,
    broad_primary_explicit_recall: float,
    canonical_vc_f1: Mapping[str, float],
    core_vc_f1: Mapping[str, float],
) -> dict[str, bool]:
    if set(canonical_vc_f1) != set(core_vc_f1):
        raise ValueError("canonical and core investor scopes do not match")
    preserved = sum(
        float(core_vc_f1[vc]) >= float(canonical_vc_f1[vc])
        for vc in canonical_vc_f1
    )
    return {
        "all_18_cases_usable": usable_cases == 18,
        "broad_exact_recall_at_least_canonical": (
            float(broad["micro_recall"]) >= float(canonical["micro_recall"])
        ),
        "broad_primary_explicit_recall_loss_at_most_0_03": (
            broad_primary_explicit_recall
            >= canonical_primary_explicit_recall - 0.03
        ),
        "core_exact_f1_gain_at_least_0_03": (
            float(core["micro_f1"]) - float(canonical["micro_f1"])
            >= 0.03 - 1e-12
        ),
        "core_family_f1_not_reduced": (
            float(core["family_micro_f1"])
            >= float(canonical["family_micro_f1"])
        ),
        "core_precision_exceeds_canonical": (
            float(core["micro_precision"]) > float(canonical["micro_precision"])
        ),
        "at_least_four_of_six_vcs_preserve_core_f1": (
            len(canonical_vc_f1) == 6 and preserved >= 4
        ),
        "dual_view_rationale_count_bounds": (
            float(broad["average_predicted_size"])
            <= 2 * float(canonical["average_predicted_size"])
            and float(core["average_predicted_size"])
            < float(broad["average_predicted_size"])
        ),
    }


def _first_adjudication_labels(state: Mapping[str, Any]) -> set[str]:
    iteration = _strict_nonnegative_int(
        state.get("phase1_iteration"), "phase1_iteration"
    )
    if iteration < 1:
        raise ValueError("v4.4 adjudication provenance is ambiguous")
    calls: dict[str, dict[str, Any]] = {}
    for value in _array(state.get("call_records"), "v4.4 call records"):
        record = _object(value, "v4.4 call record")
        relative = record.get("relative_path")
        payload = _object(record.get("payload"), "v4.4 call payload")
        phase = payload.get("phase")
        if phase not in {"phase1_adjudication", "phase1_adjudication_repair"}:
            continue
        if type(relative) is not str or relative in calls:
            raise ValueError("v4.4 adjudication provenance is ambiguous")
        calls[relative] = payload
    events = [
        event
        for event in _array(state.get("events"), "v4.4 events")
        if type(event) is dict and event.get("kind") == "phase1_adjudication"
    ]
    if len(events) != iteration:
        raise ValueError("v4.4 adjudication provenance is ambiguous")
    expected_paths: set[str] = set()
    first_labels: set[str] | None = None
    for turn in range(1, iteration + 1):
        primary_path = f"phase1/adjudication/turn-{turn:02d}/call-01.json"
        repair_path = f"phase1/adjudication/turn-{turn:02d}/call-02.json"
        primary = calls.get(primary_path)
        repair = calls.get(repair_path)
        if primary is None or primary.get("phase") != "phase1_adjudication":
            raise ValueError("v4.4 adjudication provenance is ambiguous")
        expected_paths.add(primary_path)
        if repair is not None:
            if repair.get("phase") != "phase1_adjudication_repair":
                raise ValueError("v4.4 adjudication provenance is ambiguous")
            expected_paths.add(repair_path)
        event = events[turn - 1]
        status = event.get("status")
        if status not in {"valid", "provisional", "prior_preserved"}:
            raise ValueError("v4.4 adjudication provenance is ambiguous")
        if status == "prior_preserved":
            if turn == 1:
                raise ValueError("v4.4 adjudication provenance is ambiguous")
            continue
        accepted = repair if repair is not None else primary
        parsed = accepted.get("parsed")
        if type(parsed) is not dict or type(parsed.get("dispositions")) is not list:
            raise ValueError("v4.4 adjudication provenance is ambiguous")
        labels: set[str] = set()
        for row_value in parsed["dispositions"]:
            if type(row_value) is not dict:
                raise ValueError("v4.4 adjudication provenance is ambiguous")
            if row_value.get("disposition") not in {"core", "candidate"}:
                continue
            label = row_value.get("taxonomy_label")
            if type(label) is not str or not label:
                raise ValueError("v4.4 adjudication provenance is ambiguous")
            labels.add(label)
        if first_labels is None:
            first_labels = labels
    if set(calls) != expected_paths or first_labels is None:
        raise ValueError("v4.4 adjudication provenance is ambiguous")
    return first_labels


def _process_row(
    *,
    case: Phase1Case,
    role: str,
    investigation: Mapping[str, Any],
    state: Mapping[str, Any],
    neighborhood_labels: set[str],
    broad_labels: set[str],
) -> dict[str, Any]:
    usage = _object(state.get("usage"), "v4.4 usage")
    call_records = _array(state.get("call_records"), "v4.4 call records")
    phases: list[str] = []
    for value in call_records:
        payload = _object(_object(value, "v4.4 call record").get("payload"), "call payload")
        phase = payload.get("phase")
        if type(phase) is not str:
            raise ValueError("v4.4 call record phase is invalid")
        phases.append(phase)
    status = state.get("phase1_status")
    iteration = _strict_nonnegative_int(state.get("phase1_iteration"), "phase1_iteration")
    first_labels = _first_adjudication_labels(state)
    material = _array(investigation.get("material_claims"), "material claims")
    adverse = _array(investigation.get("adverse_claims"), "adverse claims")
    activated_instances = _array(
        investigation.get("candidate_rationales"), "candidate rationales"
    )
    adjudication_events = [
        _object(value, "v4.4 adjudication event")
        for value in _array(state.get("events"), "v4.4 events")
        if type(value) is dict and value.get("kind") == "phase1_adjudication"
    ]
    exact_duplicates_removed = 0
    cross_target_reuse_count = 0
    mapping_revision_count = 0
    unresolved_mapping_count = 0
    for event in adjudication_events:
        exact_duplicates_removed += _strict_nonnegative_int(
            event.get("exact_duplicate_dispositions_removed", 0),
            "exact_duplicate_dispositions_removed",
        )
        reuse_rows = _array(
            event.get("cross_target_evidence_reuse", []),
            "cross-target evidence reuse",
        )
        mapping_rows = _array(
            event.get("constraint_mapping_revisions", []),
            "constraint mapping revisions",
        )
        cross_target_reuse_count += len(reuse_rows)
        mapping_revision_count += len(mapping_rows)
        for value in mapping_rows:
            mapping = _object(value, "constraint mapping revision")
            recomputed = _array(
                mapping.get("recomputed_mapped_ids"),
                "recomputed constraint mappings",
            )
            unresolved_mapping_count += int(not recomputed)
    reference = {row.label for row in case.reference}
    return {
        "vc_slug": case.vc_slug,
        "episode_slug": case.episode_slug,
        "role": role,
        "case_category": status,
        "phase1_status": status,
        "material_claims": len(material),
        "adverse_claims": len(adverse),
        "claim_coverage_rows": len(
            _array(investigation.get("claim_coverage"), "claim coverage")
        ),
        "unanswered_questions": len(
            _array(investigation.get("unanswered_questions"), "unanswered questions")
        ),
        "question_only_dispositions": len(
            _array(
                investigation.get("question_only_dispositions"),
                "question-only dispositions",
            )
        ),
        "taxonomy_neighborhood_size": len(neighborhood_labels),
        "reference_in_neighborhood": len(reference & neighborhood_labels),
        "reference_absent_from_neighborhood": len(reference - neighborhood_labels),
        "phase1_iteration": iteration,
        "revisit_triggered": int(iteration > 1),
        "revisit_candidate_labels_added": len(broad_labels - first_labels)
        if iteration > 1
        else 0,
        "revisit_candidate_labels_removed": len(first_labels - broad_labels)
        if iteration > 1
        else 0,
        "activated_instance_count": len(activated_instances),
        "unique_activated_label_count": len(broad_labels),
        "exact_duplicate_dispositions_removed": exact_duplicates_removed,
        "cross_target_evidence_reuse_count": cross_target_reuse_count,
        "constraint_mapping_revision_count": mapping_revision_count,
        "unresolved_constraint_mapping_count": unresolved_mapping_count,
        "calls": len(call_records),
        "claim_calls": sum("claim_extraction" in phase for phase in phases),
        "adjudication_calls": sum("adjudication" in phase for phase in phases),
        "repair_calls": sum(phase.endswith("_repair") for phase in phases),
        "input_tokens": _strict_nonnegative_int(
            usage.get("input_tokens"), "input_tokens"
        ),
        "cached_input_tokens": _strict_nonnegative_int(
            usage.get("cached_input_tokens"), "cached_input_tokens"
        ),
        "output_tokens": _strict_nonnegative_int(
            usage.get("output_tokens"), "output_tokens"
        ),
        "cost_usd": _strict_nonnegative_float(usage.get("cost_usd"), "cost_usd"),
    }


def _inactive_process_row(
    case: Phase1Case,
    role: str,
    status_row: Mapping[str, Any],
    *,
    cached_input_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "vc_slug": case.vc_slug,
        "episode_slug": case.episode_slug,
        "role": role,
        "case_category": status_row["case_category"],
        "phase1_status": status_row.get("phase1_status", ""),
        "material_claims": 0,
        "adverse_claims": 0,
        "claim_coverage_rows": 0,
        "unanswered_questions": 0,
        "question_only_dispositions": 0,
        "taxonomy_neighborhood_size": 0,
        "reference_in_neighborhood": 0,
        "reference_absent_from_neighborhood": 0,
        "phase1_iteration": 0,
        "revisit_triggered": 0,
        "revisit_candidate_labels_added": 0,
        "revisit_candidate_labels_removed": 0,
        "activated_instance_count": 0,
        "unique_activated_label_count": 0,
        "exact_duplicate_dispositions_removed": 0,
        "cross_target_evidence_reuse_count": 0,
        "constraint_mapping_revision_count": 0,
        "unresolved_constraint_mapping_count": 0,
        "calls": int(status_row["calls"]),
        "claim_calls": 0,
        "adjudication_calls": 0,
        "repair_calls": 0,
        "input_tokens": int(status_row["input_tokens"]),
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": int(status_row["output_tokens"]),
        "cost_usd": float(status_row["cost_usd"]),
    }


def _secure_failed_accounting(
    state: Mapping[str, Any], *, snapshot: _RunSnapshot
) -> Any:
    from scripts.run_phase1_v44_canary import (
        CALL_PHASES,
        CaseAccounting,
        _finite_nonnegative,
        _usage,
    )
    if (
        state.get("pending_call") is not None
        or state.get("pending_response") is not None
    ):
        raise ValueError("evaluation pending provider state cannot be reconciled")
    records = _array(state.get("call_records"), "non-usable call records")
    paths: dict[str, str] = {}
    records_by_wal: dict[str, Mapping[str, Any]] = {}
    totals: dict[str, int | float] = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    inventory = snapshot.files
    with nullcontext(snapshot) as artifacts:
        for value in records:
            record = _object(value, "non-usable call record")
            payload = _object(record.get("payload"), "call payload")
            relative = record.get("relative_path")
            digest = record.get("sha256")
            wal_id = record.get("wal_id")
            request_digest = record.get("request_sha256")
            phase = payload.get("phase")
            if (
                type(relative) is not str
                or type(digest) is not str
                or type(wal_id) is not str
                or re.fullmatch(r"[0-9a-f]{64}", wal_id) is None
                or type(request_digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", request_digest) is None
                or re.fullmatch(
                    r"phase1/(?:claim-extraction|adjudication/turn-[0-9]{2})/"
                    r"call-[0-9]{2}\.json",
                    relative,
                )
                is None
                or relative in paths
                or wal_id in records_by_wal
                or phase not in CALL_PHASES
            ):
                raise ValueError(
                    "evaluation provider call ledger cannot be reconciled"
                )
            try:
                raw = artifacts.bytes_by_path[relative]
                persisted_payload = json.loads(raw)
            except KeyError as exc:
                raise ValueError(
                    "evaluation provider call artifact is unsafe or cannot be "
                    "reconciled"
                ) from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "evaluation provider call artifact cannot be reconciled"
                ) from exc
            call_usage = _usage(payload.get("usage"), context="evaluation call")
            if (
                _finite_nonnegative(
                    payload.get("cost_usd"), field="evaluation call cost_usd"
                )
                != call_usage["cost_usd"]
            ):
                raise ValueError(
                    "evaluation provider call cost cannot be reconciled"
                )
            if sha256(raw).hexdigest() != digest or persisted_payload != payload:
                raise ValueError(
                    "evaluation provider call ledger cannot be reconciled"
                )
            paths[relative] = phase
            records_by_wal[wal_id] = record
            for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
                totals[key] = int(totals[key]) + int(call_usage[key])
            totals["cost_usd"] = float(totals["cost_usd"]) + float(
                call_usage["cost_usd"]
            )

        reported = _usage(state.get("usage"), context="evaluation")
        for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
            if reported[key] != totals[key]:
                raise ValueError("evaluation provider usage cannot be reconciled")
        if float(reported["cost_usd"]) != float(totals["cost_usd"]):
            raise ValueError("evaluation provider cost cannot be reconciled")

        wal_paths = [
            path for path in inventory if path.startswith(".phase1-v44-wal/")
        ]
        for relative in wal_paths:
            match = re.fullmatch(
                r"\.phase1-v44-wal/([0-9a-f]{64})\.json", relative
            )
            if match is None:
                raise ValueError("evaluation provider WAL artifact is unsafe")
            try:
                wal = artifacts.json_objects[relative]
            except KeyError as exc:
                raise ValueError("evaluation provider WAL artifact is unsafe") from exc
            record = records_by_wal.get(match.group(1))
            if (
                type(wal) is not dict
                or wal.get("schema_version") != "phase1-provider-call-wal-v1"
                or wal.get("wal_id") != match.group(1)
                or wal.get("status") != "responded_uncheckpointed"
                or record is None
            ):
                raise ValueError("evaluation provider WAL cannot be reconciled")
            wal_usage = _usage(
                wal.get("uncertain_usage"), context="evaluation WAL"
            )
            record_usage = _usage(
                _object(record.get("payload"), "call payload").get("usage"),
                context="evaluation call",
            )
            if wal_usage != record_usage:
                raise ValueError("evaluation provider WAL cost cannot be reconciled")

    return CaseAccounting(
        cost_usd=float(reported["cost_usd"]),
        input_tokens=int(reported["input_tokens"]),
        cached_input_tokens=int(reported["cached_input_tokens"]),
        output_tokens=int(reported["output_tokens"]),
        calls=len(records),
        phase1_status=str(state.get("phase1_status", "")),
        phase2_status=str(state.get("phase2_status", "")),
        source="evaluation",
        call_phases=tuple(paths.values()),
    )


def _reconcile_state_accounting(
    state: Mapping[str, Any],
    status_row: Mapping[str, Any],
    *,
    snapshot: _RunSnapshot,
) -> int:
    account = _secure_failed_accounting(state, snapshot=snapshot)
    records = _array(state.get("call_records"), "non-usable call records")
    paths: dict[str, str] = {}
    for value in records:
        record = _object(value, "non-usable call record")
        relative = str(record.get("relative_path"))
        phase = str(_object(record.get("payload"), "call payload").get("phase"))
        if relative in paths:
            raise ValueError("evaluation provider call topology cannot be reconciled")
        paths[relative] = phase
    expected: dict[str, str] = {}
    claim_primary = "phase1/claim-extraction/call-01.json"
    claim_repair = "phase1/claim-extraction/call-02.json"
    if claim_primary in paths:
        expected[claim_primary] = "phase1_claim_extraction"
    if claim_repair in paths:
        if claim_primary not in paths:
            raise ValueError("evaluation provider call topology cannot be reconciled")
        expected[claim_repair] = "phase1_claim_extraction_repair"
    turns: set[int] = set()
    for relative in paths:
        match = re.fullmatch(
            r"phase1/adjudication/turn-([0-9]{2})/call-([0-9]{2})\.json",
            relative,
        )
        if match is not None:
            turns.add(int(match.group(1)))
    if turns and turns != set(range(1, max(turns) + 1)):
        raise ValueError("evaluation provider call topology cannot be reconciled")
    iteration = _strict_nonnegative_int(
        state.get("phase1_iteration"), "failed phase1_iteration"
    )
    if turns and max(turns) > iteration:
        raise ValueError("evaluation provider call topology cannot be reconciled")
    for turn in sorted(turns):
        primary = f"phase1/adjudication/turn-{turn:02d}/call-01.json"
        repair = f"phase1/adjudication/turn-{turn:02d}/call-02.json"
        if primary not in paths:
            raise ValueError("evaluation provider call topology cannot be reconciled")
        expected[primary] = "phase1_adjudication"
        if repair in paths:
            expected[repair] = "phase1_adjudication_repair"
    if paths != expected:
        raise ValueError("evaluation provider call topology cannot be reconciled")
    if (
        account.calls != status_row["calls"]
        or account.input_tokens != status_row["input_tokens"]
        or account.output_tokens != status_row["output_tokens"]
        or not math.isclose(
            account.cost_usd,
            float(status_row["cost_usd"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("runner non-usable accounting is inconsistent")
    return account.cached_input_tokens


def _status_state_path(status_row: Mapping[str, Any], state_path: Path) -> None:
    expected = Path(str(status_row.get("state_path") or ""))
    if not expected.is_absolute():
        expected = Path.cwd() / expected
    expected = Path(os.path.abspath(os.fspath(expected)))
    observed = Path(os.path.abspath(os.fspath(state_path)))
    if expected != observed:
        raise ValueError("runner failure state path is inconsistent")


def _zero_status_accounting(status_row: Mapping[str, Any]) -> bool:
    return all(
        float(status_row[field]) == 0.0
        for field in ("calls", "input_tokens", "output_tokens", "cost_usd")
    )


def evaluate_verified_cases(
    *,
    canonical_cases: Sequence[Phase1Case],
    taxonomy: Mapping[str, TaxonomyLabel],
    selected: Mapping[tuple[str, str], Mapping[str, str]],
    runner_status: Mapping[str, Any],
    runs_root: Path,
    verifier: Callable[[Path], None] = verify_phase1_artifacts,
) -> dict[str, Any]:
    if not canonical_cases:
        raise ValueError("canonical canary is empty")
    canonical_by_key: dict[tuple[str, str], Phase1Case] = {}
    for case in canonical_cases:
        key = case.vc_slug, case.episode_slug
        if key in canonical_by_key:
            raise ValueError("canonical cases are ambiguous")
        canonical_by_key[key] = case
    if set(canonical_by_key) != set(selected):
        raise ValueError("canonical and canary case coverage does not match")
    canonical = [canonical_by_key[key] for key in selected]
    status_by_key = _runner_status_index(runner_status, selected)
    canonical_usable: list[Phase1Case] = []
    broad_cases: list[Phase1Case] = []
    core_cases: list[Phase1Case] = []
    paired_rows: list[dict[str, Any]] = []
    taxonomy_rows: list[dict[str, Any]] = []
    process_rows: list[dict[str, Any]] = []
    verified_runs: list[dict[str, str]] = []

    for case in canonical:
        key = case.vc_slug, case.episode_slug
        manifest_row = selected[key]
        if manifest_row["actual_decision"] != case.actual_decision:
            raise ValueError(f"canary decision mismatch for {case.episode_slug}")
        run_root = runs_root / case.vc_slug / case.episode_slug
        investigation_path = run_root / "phase1/investigation.json"
        state_path = run_root / "state.json"
        status_row = status_by_key[key]
        category = status_row["case_category"]
        if category in {"failed", "not_started"}:
            post_completion_failure = status_row["status"] in {
                "failed_rate_ceiling_breach",
                "failed_cost_ceiling_exceeded",
            }
            snapshot = _snapshot_nonusable_run(runs_root, run_root)
            if post_completion_failure:
                try:
                    _verify_nonusable_snapshot(snapshot, verifier)
                except Exception as exc:
                    raise ValueError(
                        f"v4.4 verification failed for "
                        f"{case.vc_slug}/{case.episode_slug}"
                    ) from exc
            _after_nonusable_snapshot()
            has_files = bool(snapshot.files)
            if snapshot.phase2_exists:
                raise ValueError("suspicious partial v4.4 artifacts for non-usable case")
            if "phase1/investigation.json" in snapshot.json_objects:
                if (
                    category != "failed"
                    or not post_completion_failure
                    or "state.json" not in snapshot.json_objects
                ):
                    raise ValueError(
                        "suspicious partial v4.4 artifacts for non-usable case"
                    )
                investigation = _object(
                    snapshot.json_objects["phase1/investigation.json"],
                    "non-usable terminal v4.4 investigation",
                )
                state = _object(
                    snapshot.json_objects["state.json"],
                    "non-usable terminal v4.4 state",
                )
                _scan_nonusable_snapshot(snapshot)
                if (
                    investigation.get("schema_version") != "investigation-v4.4"
                    or investigation.get("episode_slug") != case.episode_slug
                    or state.get("contract_version") != "v4.4"
                    or state.get("episode_slug") != case.episode_slug
                    or state.get("phase1_status") not in {"accepted", "provisional"}
                    or state.get("phase1_status") != status_row.get("phase1_status")
                    or state.get("phase2_status") != "not_run"
                ):
                    raise ValueError(
                        "suspicious partial v4.4 artifacts for non-usable case"
                    )
                _status_state_path(status_row, state_path)
                _reconcile_state_accounting(
                    state, status_row, snapshot=snapshot
                )
                broad_labels = _labels(
                    investigation.get("candidate_rationales"),
                    "non-usable terminal candidate rationales",
                )
                neighborhood = _object(
                    investigation.get("taxonomy_neighborhood_manifest"),
                    "non-usable terminal taxonomy neighborhood",
                )
                ordered = _array(
                    neighborhood.get("ordered_labels"),
                    "non-usable terminal ordered taxonomy labels",
                )
                if not all(type(label) is str and label in taxonomy for label in ordered):
                    raise ValueError("taxonomy neighborhood contains an invalid label")
                row = _process_row(
                    case=case,
                    role=manifest_row["role"],
                    investigation=investigation,
                    state=state,
                    neighborhood_labels=set(ordered),
                    broad_labels=broad_labels,
                )
                row["case_category"] = "failed"
                process_rows.append(row)
                verified_runs.append(
                    {
                        "vc_slug": case.vc_slug,
                        "episode_slug": case.episode_slug,
                        "status": str(status_row["status"]),
                        "investigation_sha256": snapshot.hashes[
                            "phase1/investigation.json"
                        ],
                        "state_sha256": snapshot.hashes["state.json"],
                    }
                )
                continue
            cached = 0
            if category == "not_started":
                if has_files or status_row.get("state_path"):
                    raise ValueError(
                        "suspicious partial v4.4 artifacts for non-usable case"
                    )
                if not _zero_status_accounting(status_row):
                    raise ValueError(
                        "non-usable case without state has nonzero accounting"
                    )
            elif "state.json" in snapshot.json_objects:
                state = _object(
                    snapshot.json_objects["state.json"], "failed v4.4 state"
                )
                _scan_nonusable_snapshot(snapshot)
                if (
                    state.get("contract_version") != "v4.4"
                    or state.get("episode_slug") != case.episode_slug
                    or state.get("phase1_status") in {"accepted", "provisional"}
                    or state.get("phase2_status") != "not_run"
                ):
                    raise ValueError(
                        "suspicious partial v4.4 artifacts for non-usable case"
                    )
                _status_state_path(status_row, state_path)
                cached = _reconcile_state_accounting(
                    state, status_row, snapshot=snapshot
                )
                verified_runs.append(
                    {
                        "vc_slug": case.vc_slug,
                        "episode_slug": case.episode_slug,
                        "status": "failed",
                        "state_sha256": snapshot.hashes["state.json"],
                    }
                )
            elif has_files or status_row.get("state_path"):
                raise ValueError("suspicious partial v4.4 artifacts for non-usable case")
            elif not _zero_status_accounting(status_row):
                raise ValueError("runner failure without state has nonzero accounting")
            process_rows.append(
                _inactive_process_row(
                    case,
                    manifest_row["role"],
                    status_row,
                    cached_input_tokens=cached,
                )
            )
            continue
        snapshot = _snapshot_nonusable_run(runs_root, run_root)
        if "phase1/investigation.json" not in snapshot.json_objects:
            raise ValueError(f"missing v4.4 investigation: {investigation_path}")
        if "state.json" not in snapshot.json_objects:
            raise ValueError(f"missing v4.4 state: {state_path}")
        try:
            _verify_nonusable_snapshot(snapshot, verifier)
        except Exception as exc:
            raise ValueError(
                f"v4.4 verification failed for {case.vc_slug}/{case.episode_slug}"
            ) from exc
        _after_usable_snapshot()
        investigation = _object(
            snapshot.json_objects["phase1/investigation.json"],
            "v4.4 investigation",
        )
        state = _object(snapshot.json_objects["state.json"], "v4.4 state")
        if _contains_forbidden_key(investigation) or _contains_forbidden_key(state):
            raise ValueError("v4.4 run contains a leaked target/evaluation label")
        if (
            investigation.get("schema_version") != "investigation-v4.4"
            or state.get("contract_version") != "v4.4"
        ):
            raise ValueError("mixed or unsupported v4.4 contract")
        if (
            investigation.get("episode_slug") != case.episode_slug
            or state.get("episode_slug") != case.episode_slug
        ):
            raise ValueError("v4.4 episode binding mismatch")
        if state.get("phase1_status") not in {"accepted", "provisional"}:
            raise ValueError("v4.4 case is not usable")
        if state.get("phase2_status") != "not_run":
            raise ValueError("v4.4 evaluator rejects Phase 2 artifacts")
        if category != state.get("phase1_status"):
            raise ValueError("runner usable status is inconsistent with frozen state")
        _status_state_path(status_row, state_path)
        candidates = _array(
            investigation.get("candidate_rationales"), "candidate rationales"
        )
        cores = _array(investigation.get("core_rationales"), "core rationales")
        compatibility = _array(investigation.get("rationales"), "rationales")
        if compatibility != cores:
            raise ValueError("v4.4 compatibility rationales do not equal core_rationales")
        broad_payload = deepcopy(investigation)
        broad_payload["rationales"] = deepcopy(candidates)
        broad_predicted = _prediction_rationales(
            broad_payload, taxonomy, str(investigation_path) + "#candidate_rationales"
        )
        core_predicted = _prediction_rationales(
            investigation, taxonomy, str(investigation_path) + "#core_rationales"
        )
        broad_case = replace(
            case,
            predicted=broad_predicted,
            artifact_path=investigation_path,
        )
        core_case = replace(
            case,
            predicted=core_predicted,
            artifact_path=investigation_path,
        )
        broad_cases.append(broad_case)
        core_cases.append(core_case)
        canonical_usable.append(case)

        neighborhood = _object(
            investigation.get("taxonomy_neighborhood_manifest"),
            "taxonomy neighborhood manifest",
        )
        ordered_labels = _array(
            neighborhood.get("ordered_labels"), "ordered taxonomy labels"
        )
        if not all(type(label) is str and label in taxonomy for label in ordered_labels):
            raise ValueError("taxonomy neighborhood contains an invalid label")
        neighborhood_labels = set(ordered_labels)
        broad_labels = {row.label for row in broad_predicted}
        core_labels = {row.label for row in core_predicted}
        reference_labels = {row.label for row in case.reference}
        for reference in case.reference:
            if reference.label not in neighborhood_labels:
                stage = "retrieval_omission"
            elif reference.label not in broad_labels:
                stage = "adjudication_omission"
            elif reference.label not in core_labels:
                stage = "core_filtering"
            else:
                stage = "recovered_core"
            taxonomy_rows.append(
                {
                    "vc_slug": case.vc_slug,
                    "episode_slug": case.episode_slug,
                    "role": manifest_row["role"],
                    "reference_label": reference.label,
                    "reference_salience": reference.salience,
                    "reference_decision_link": reference.decision_link,
                    "in_neighborhood": reference.label in neighborhood_labels,
                    "in_broad": reference.label in broad_labels,
                    "in_core": reference.label in core_labels,
                    "error_stage": stage,
                }
            )
        process_row = _process_row(
            case=case,
            role=manifest_row["role"],
            investigation=investigation,
            state=state,
            neighborhood_labels=neighborhood_labels,
            broad_labels=broad_labels,
        )
        if (
            process_row["calls"] != status_row["calls"]
            or process_row["input_tokens"] != status_row["input_tokens"]
            or process_row["output_tokens"] != status_row["output_tokens"]
            or not math.isclose(
                float(process_row["cost_usd"]),
                float(status_row["cost_usd"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("runner usable accounting is inconsistent with frozen state")
        process_rows.append(process_row)
        canonical_score = score_set(
            {row.label for row in case.predicted}, reference_labels
        )
        broad_score = score_set(broad_labels, reference_labels)
        core_score = score_set(core_labels, reference_labels)
        paired_rows.append(
            {
                "vc_slug": case.vc_slug,
                "episode_slug": case.episode_slug,
                "actual_decision": case.actual_decision,
                "role": manifest_row["role"],
                "canonical_precision": canonical_score["precision"],
                "canonical_recall": canonical_score["recall"],
                "canonical_f1": canonical_score["f1"],
                "broad_precision": broad_score["precision"],
                "broad_recall": broad_score["recall"],
                "broad_f1": broad_score["f1"],
                "core_precision": core_score["precision"],
                "core_recall": core_score["recall"],
                "core_f1": core_score["f1"],
                "broad_f1_delta_vs_canonical": float(broad_score["f1"])
                - float(canonical_score["f1"]),
                "core_f1_delta_vs_canonical": float(core_score["f1"])
                - float(canonical_score["f1"]),
                "core_f1_delta_vs_broad": float(core_score["f1"])
                - float(broad_score["f1"]),
                "canonical_count": len({row.label for row in case.predicted}),
                "broad_count": len(broad_labels),
                "core_count": len(core_labels),
                "broad_correct_added_vs_canonical": len(
                    (broad_labels & reference_labels)
                    - ({row.label for row in case.predicted} & reference_labels)
                ),
                "broad_correct_lost_vs_canonical": len(
                    ({row.label for row in case.predicted} & reference_labels)
                    - (broad_labels & reference_labels)
                ),
                "core_correct_lost_vs_broad": len(
                    (broad_labels & reference_labels) - core_labels
                ),
            }
        )
        verified_runs.append(
            {
                "vc_slug": case.vc_slug,
                "episode_slug": case.episode_slug,
                "status": str(state["phase1_status"]),
                "investigation_sha256": snapshot.hashes[
                    "phase1/investigation.json"
                ],
                "state_sha256": snapshot.hashes["state.json"],
            }
        )

    if not canonical_usable:
        raise ValueError("v4.4 canary has no strictly verified usable cases to score")
    canonical_result = evaluate_cases(canonical_usable, taxonomy)
    broad_result = evaluate_cases(broad_cases, taxonomy)
    core_result = evaluate_cases(core_cases, taxonomy)
    canonical_overall = canonical_result["overall"]
    broad_overall = broad_result["overall"]
    core_overall = core_result["overall"]
    opportunity_counts = Counter(row["error_stage"] for row in taxonomy_rows)
    reference_total = len(taxonomy_rows)
    in_neighborhood = reference_total - opportunity_counts["retrieval_omission"]
    canonical_primary = _rich_recall(canonical_result, "primary_and_explicit")
    broad_primary = _rich_recall(broad_result, "primary_and_explicit")
    criteria = promotion_criteria(
        usable_cases=len(core_cases),
        canonical=canonical_overall,
        broad=broad_overall,
        core=core_overall,
        canonical_primary_explicit_recall=canonical_primary,
        broad_primary_explicit_recall=broad_primary,
        canonical_vc_f1=_vc_f1(canonical_result),
        core_vc_f1=_vc_f1(core_result),
    )
    usage = {
        "calls": sum(int(row["calls"]) for row in process_rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in process_rows),
        "cached_input_tokens": sum(
            int(row["cached_input_tokens"]) for row in process_rows
        ),
        "output_tokens": sum(int(row["output_tokens"]) for row in process_rows),
        "total_cost_usd": sum(float(row["cost_usd"]) for row in process_rows),
        "average_cost_usd": fmean(float(row["cost_usd"]) for row in process_rows),
    }
    process = {
        "accepted": sum(row["case_category"] == "accepted" for row in process_rows),
        "provisional": sum(
            row["case_category"] == "provisional" for row in process_rows
        ),
        "failed": sum(row["case_category"] == "failed" for row in process_rows),
        "not_started": sum(
            row["case_category"] == "not_started" for row in process_rows
        ),
        "revisits_triggered": sum(int(row["revisit_triggered"]) for row in process_rows),
        "question_only_dispositions": sum(
            int(row["question_only_dispositions"]) for row in process_rows
        ),
        "material_claims": sum(int(row["material_claims"]) for row in process_rows),
        "adverse_claims": sum(int(row["adverse_claims"]) for row in process_rows),
        "claim_coverage_rows": sum(
            int(row["claim_coverage_rows"]) for row in process_rows
        ),
        "revisit_candidate_labels_added": sum(
            int(row["revisit_candidate_labels_added"]) for row in process_rows
        ),
        "revisit_candidate_labels_removed": sum(
            int(row["revisit_candidate_labels_removed"]) for row in process_rows
        ),
        "activated_instances": sum(
            int(row["activated_instance_count"]) for row in process_rows
        ),
        "unique_activated_labels": sum(
            int(row["unique_activated_label_count"]) for row in process_rows
        ),
        "exact_duplicate_dispositions_removed": sum(
            int(row["exact_duplicate_dispositions_removed"])
            for row in process_rows
        ),
        "cross_target_evidence_reuse": sum(
            int(row["cross_target_evidence_reuse_count"])
            for row in process_rows
        ),
        "constraint_mapping_revisions": sum(
            int(row["constraint_mapping_revision_count"])
            for row in process_rows
        ),
        "unresolved_constraint_mappings": sum(
            int(row["unresolved_constraint_mapping_count"])
            for row in process_rows
        ),
    }
    metrics = {
        "schema": "phase1-v44-canary-evaluation-v1",
        "scientific_status": SCIENTIFIC_STATUS,
        "case_count": len(core_cases),
        "canonical": canonical_overall,
        "broad": broad_overall,
        "core": core_overall,
        "primary_explicit_recall": {
            "canonical": canonical_primary,
            "broad": broad_primary,
            "core": _rich_recall(core_result, "primary_and_explicit"),
        },
        "taxonomy_opportunity": {
            "reference_total": reference_total,
            "reference_in_neighborhood": in_neighborhood,
            "reference_absent_from_neighborhood": opportunity_counts[
                "retrieval_omission"
            ],
            "neighborhood_recall": in_neighborhood / reference_total
            if reference_total
            else 1.0,
        },
        "adjudication": {
            "available_but_not_candidate": opportunity_counts[
                "adjudication_omission"
            ],
            "candidate_but_not_core": opportunity_counts["core_filtering"],
            "recovered_in_core": opportunity_counts["recovered_core"],
        },
        "paired_label_changes": {
            "canonical_to_broad": _paired_label_change(
                canonical_usable, broad_cases, taxonomy
            ),
            "canonical_to_core": _paired_label_change(
                canonical_usable, core_cases, taxonomy
            ),
            "broad_to_core": _paired_label_change(broad_cases, core_cases, taxonomy),
            "interpretation": {
                "canonical_to_broad_and_core": "includes fresh model-run variability",
                "broad_to_core": "isolates the two frozen views within the same run",
            },
        },
        "by_role": {
            "canonical": _method_by_role(canonical_usable, taxonomy, selected),
            "broad": _method_by_role(broad_cases, taxonomy, selected),
            "core": _method_by_role(core_cases, taxonomy, selected),
        },
        "per_vc_f1": {
            vc: {
                "canonical": _vc_f1(canonical_result)[vc],
                "broad": _vc_f1(broad_result)[vc],
                "core": _vc_f1(core_result)[vc],
            }
            for vc in _vc_f1(canonical_result)
        },
        "process": process,
        "usage": usage,
        "criteria": criteria,
        "recommendation": (
            "promote_to_larger_paired_development_run"
            if all(criteria.values())
            else "retain_canonical_v4_v41"
        ),
        "verified_runs": verified_runs,
    }
    return {
        "metrics": metrics,
        "canonical_result": canonical_result,
        "broad_result": broad_result,
        "core_result": core_result,
        "paired_rows": paired_rows,
        "taxonomy_rows": taxonomy_rows,
        "process_rows": process_rows,
        "scored_keys": [
            [case.vc_slug, case.episode_slug] for case in canonical_usable
        ],
    }


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with open(descriptor, "wb", closefd=True) as stream:
            stream.write(content)
            stream.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: object) -> None:
    _atomic_write(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty required CSV: {path.name}")
    fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with open(descriptor, "w", encoding="utf-8", newline="", closefd=True) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _evaluation_markdown(metrics: Mapping[str, Any]) -> str:
    canonical = metrics["canonical"]
    broad = metrics["broad"]
    core = metrics["core"]
    usage = metrics["usage"]
    lines = [
        "# Phase 1 v4.4 Paired Canary",
        "",
        "> Development canary using automated transcript-derived rationale references; not an untouched holdout.",
        "",
        f"Usable cases: **{metrics['case_count']}/18**. Exact provider cost: **${usage['total_cost_usd']:.6f}**.",
        "",
        "| Method | Precision | Recall | F1 | Family F1 | Avg rationales |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Canonical v4/v4.1 | {canonical['micro_precision']:.3f} | {canonical['micro_recall']:.3f} | {canonical['micro_f1']:.3f} | {canonical['family_micro_f1']:.3f} | {canonical['average_predicted_size']:.2f} |",
        f"| v4.4 broad candidates | {broad['micro_precision']:.3f} | {broad['micro_recall']:.3f} | {broad['micro_f1']:.3f} | {broad['family_micro_f1']:.3f} | {broad['average_predicted_size']:.2f} |",
        f"| v4.4 evidence-supported core | {core['micro_precision']:.3f} | {core['micro_recall']:.3f} | {core['micro_f1']:.3f} | {core['family_micro_f1']:.3f} | {core['average_predicted_size']:.2f} |",
        "",
        "Canonical-to-v4.4 changes include fresh model-run variability. Broad-to-core changes compare two frozen views from the same run.",
        "",
        f"Recommendation: **{metrics['recommendation']}**.",
        "",
        "## Promotion criteria",
        "",
        *[
            f"- {'PASS' if passed else 'FAIL'} — `{name}`"
            for name, passed in metrics["criteria"].items()
        ],
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    output: Path,
    report: Mapping[str, Any],
    *,
    inputs: Mapping[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "metrics.json", report["metrics"])
    _write_json(output / "canonical-result.json", report["canonical_result"])
    _write_json(output / "broad-result.json", report["broad_result"])
    _write_json(output / "core-result.json", report["core_result"])
    _write_csv(output / "paired-cases.csv", report["paired_rows"])
    _write_csv(output / "taxonomy-opportunity.csv", report["taxonomy_rows"])
    _write_csv(output / "process-diagnostics.csv", report["process_rows"])
    _atomic_write(
        output / "evaluation.md", _evaluation_markdown(report["metrics"]).encode("utf-8")
    )
    output_hashes = {
        filename: sha256_file(output / filename) for filename in EXPECTED_OUTPUTS
    }
    manifest = {
        "schema": "phase1-v44-canary-evaluation-manifest-v1",
        "scientific_status": SCIENTIFIC_STATUS,
        "inputs": dict(inputs),
        "verified_runs": report["metrics"]["verified_runs"],
        "outputs": output_hashes,
        "canonical_modified": False,
        "recommendation": report["metrics"]["recommendation"],
    }
    _write_json(output / "manifest.json", manifest)


def _input_binding(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"required evaluation input is missing: {path}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _before_report_source_recheck() -> None:
    """Test seam immediately before captured report sources are rechecked."""


def _recheck_tree_snapshot(root: Path, expected: _RunSnapshot) -> None:
    observed = _snapshot_nonusable_run(root.parent, root)
    if (
        observed.exists != expected.exists
        or observed.directories != expected.directories
        or observed.files != expected.files
        or observed.hashes != expected.hashes
    ):
        raise ValueError(f"evaluation input tree mutated during run: {root}")


def _commit_staged_report(staging: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for filename in (*EXPECTED_OUTPUTS, "manifest.json"):
        _atomic_write(output / filename, (staging / filename).read_bytes())


def _publish_captured_report(
    output: Path,
    report: Mapping[str, Any],
    *,
    inputs: Mapping[str, Any],
    captured_sources: Sequence[_CapturedFile],
    tree_sources: Sequence[tuple[Path, _RunSnapshot]],
    glob_sources: Sequence[_CapturedGlob] = (),
) -> None:
    output = Path(os.path.abspath(os.fspath(output)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.staging-"
    ) as directory:
        staging = Path(directory)
        write_outputs(staging, report, inputs=inputs)
        _before_report_source_recheck()
        _recheck_captured_files(captured_sources)
        for root, snapshot in tree_sources:
            _recheck_tree_snapshot(root, snapshot)
        _recheck_captured_globs(glob_sources)
        _commit_staged_report(staging, output)


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path.cwd()
    registry_capture = _capture_file(args.registry, "evaluation registry")
    taxonomy_capture = _capture_file(args.taxonomy, "taxonomy")
    canary_capture = _capture_file(args.manifest, "canary manifest")
    status_capture = _capture_file(args.status, "runner status")
    reference_root = Path(os.path.abspath(os.fspath(args.references)))
    reference_snapshot = _snapshot_nonusable_run(
        reference_root.parent, reference_root
    )
    reference_manifest = _capture_from_tree(
        reference_root,
        reference_snapshot,
        reference_root / "manifest.json",
        "reference manifest",
    )
    reference_manifest_payload = _object(
        _captured_json(reference_manifest, "reference manifest"),
        "reference manifest",
    )
    if reference_manifest_payload.get("missing_count") != 0:
        raise ValueError("reference manifest reports missing cases")
    selected = _parse_canary_manifest_capture(canary_capture)
    taxonomy = _parse_taxonomy_capture(taxonomy_capture)
    runner_status = _object(
        _captured_json(status_capture, "v4.4 runner status"),
        "v4.4 runner status",
    )

    discovered, auxiliary_sources, summaries_by_key, glob_sources = (
        _discover_cases_from_captured_registry(
            registry_capture,
            selected=selected,
            reference_root=reference_root,
            reference_snapshot=reference_snapshot,
        )
    )
    canonical, case_sources, case_bindings = _rebuild_cases_from_captures(
        discovered,
        taxonomy=taxonomy,
        project_root=project_root,
        reference_root=reference_root,
        reference_snapshot=reference_snapshot,
    )
    for case in canonical:
        if case.artifact_path is None:
            raise ValueError("canonical artifact path is missing")
        captured = next(
            source
            for source in case_sources
            if source.path == Path(case.artifact_path)
        )
        schema = _object(
            _captured_json(captured, "canonical investigation"),
            "canonical investigation",
        ).get("schema_version")
        if schema not in {"investigation-v4", "investigation-v4.1"}:
            raise ValueError("canonical registry contains a mixed contract")
        summary = summaries_by_key[(case.vc_slug, case.episode_slug)]
        summary_payload = _object(
            _captured_json(summary, "canonical summary manifest"),
            "canonical summary manifest",
        )
        decision = (
            summary_payload.get("decision")
            if type(summary_payload.get("decision")) is dict
            else {}
        )
        summary_episode = (
            summary_payload.get("episode_slug")
            or decision.get("episode_slug")
            or summary.path.parent.name
        )
        if summary_episode != case.episode_slug:
            raise ValueError("canonical summary episode mismatch")
        binding = next(
            value
            for value in case_bindings
            if value["vc_slug"] == case.vc_slug
            and value["episode_slug"] == case.episode_slug
        )
        binding["canonical_summary"] = _captured_binding(summary)

    captured_sources = [
        registry_capture,
        taxonomy_capture,
        canary_capture,
        status_capture,
        reference_manifest,
        *case_sources,
        *auxiliary_sources,
    ]
    _recheck_captured_files(captured_sources)
    _recheck_tree_snapshot(reference_root, reference_snapshot)
    _recheck_captured_globs(glob_sources)
    report = evaluate_verified_cases(
        canonical_cases=canonical,
        taxonomy=taxonomy,
        selected=selected,
        runner_status=runner_status,
        runs_root=args.runs,
    )
    scored_keys = {tuple(value) for value in report["scored_keys"]}
    scored_bindings = [
        binding
        for binding in case_bindings
        if (binding["vc_slug"], binding["episode_slug"]) in scored_keys
    ]
    inputs = {
        "registry": _captured_binding(registry_capture),
        "references_manifest": _captured_binding(reference_manifest),
        "taxonomy": _captured_binding(taxonomy_capture),
        "canary_manifest": _captured_binding(canary_capture),
        "runner_status": _captured_binding(status_capture),
        "registry_source_manifests": [
            _captured_binding(source) for source in auxiliary_sources
        ],
        "scored_case_artifacts": scored_bindings,
    }
    _publish_captured_report(
        args.output,
        report,
        inputs=inputs,
        captured_sources=captured_sources,
        tree_sources=((reference_root, reference_snapshot),),
        glob_sources=glob_sources,
    )
    return report


def main() -> None:
    args = parse_args()
    report = run(args)
    metrics = report["metrics"]
    broad = metrics["broad"]
    core = metrics["core"]
    print(
        f"v4.4 development canary n={metrics['case_count']} "
        f"broad P={broad['micro_precision']:.3f} R={broad['micro_recall']:.3f} "
        f"F1={broad['micro_f1']:.3f}; core P={core['micro_precision']:.3f} "
        f"R={core['micro_recall']:.3f} F1={core['micro_f1']:.3f}; "
        f"recommendation={metrics['recommendation']} "
        f"cost=${metrics['usage']['total_cost_usd']:.6f}"
    )


if __name__ == "__main__":
    main()
