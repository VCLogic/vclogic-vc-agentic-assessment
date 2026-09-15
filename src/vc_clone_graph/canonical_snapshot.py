"""Build a self-contained canonical evaluation artifact snapshot."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

from vc_clone_graph.evaluation import (
    EvaluationRegistry,
    PredictionRow,
    load_canonical_predictions,
    load_registry,
    prediction_from_summary,
)


SNAPSHOT_SCHEMA = "canonical-artifact-snapshot-v1"
SNAPSHOT_BUILDER_VERSION = "1.0.0"
DIRECTORY_DIGEST_ALGORITHM = "sha256-path-size-content-v1"
RUNTIME_TO_CANONICAL_SLUG = {
    "charles-hudson-precursor-ventures": "charles-hudson",
    "cyan-banister-long-journey-ventures": "cyan-banister",
    "elizabeth-yin-hustle-fund": "elizabeth-yin",
    "jesse-middleton-flybridge": "jesse-middleton",
    "jillian-manus-structure-capital": "jillian-manus",
    "phil-nadel": "phil-nadel",
}
PROVENANCE_FIELDS = (
    "vc_slug",
    "vc_name",
    "episode_slug",
    "actual_decision",
    "selection_tier",
    "source_artifact_path",
    "snapshot_artifact_path",
    "contract_version",
    "predicted_decision",
    "investment_likelihood",
    "decision_confidence",
    "review_priority_score",
    "phase1_status",
    "phase1_iterations",
    "phase2_status",
    "phase2_iterations",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cost_usd",
    "source_directory_digest",
    "copied_directory_digest",
    "file_count",
    "byte_count",
)


@dataclass(frozen=True)
class DirectoryDigest:
    sha256: str
    file_count: int
    byte_count: int


@dataclass(frozen=True)
class SelectedArtifact:
    vc_slug: str
    vc_name: str
    episode_slug: str
    actual_decision: str
    selection_tier: str
    source_root: Path
    prediction: PredictionRow
    override_record: Mapping[str, Any] | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(root: Path) -> DirectoryDigest:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"artifact root is not a directory: {root}")
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        content_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        byte_count += size
    return DirectoryDigest(digest.hexdigest(), file_count, byte_count)


def _object(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _read_json(path: Path, context: str) -> dict[str, Any]:
    return _object(json.loads(path.read_text(encoding="utf-8")), context)


def _resolve_artifact_root(raw: str, project_root: Path) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else project_root / path).resolve()


def _validate_artifact_identity(
    root: Path,
    vc_slug: str,
    episode_slug: str,
    runtime_to_canonical: Mapping[str, str],
) -> None:
    summary_path = root / "summary.json"
    investigation_path = root / "phase1" / "investigation.json"
    sidecar_path = root / "phase1" / "investigation.sha256"
    run_config_path = root / "run-config.json"
    for path in (summary_path, investigation_path, sidecar_path, run_config_path):
        if not path.is_file():
            raise ValueError(
                f"missing required artifact for {vc_slug}/{episode_slug}: {path}"
            )

    summary = _read_json(summary_path, f"summary {summary_path}")
    investigation = _read_json(investigation_path, f"investigation {investigation_path}")
    run_config = _read_json(run_config_path, f"run config {run_config_path}")
    if summary.get("episode_slug") != episode_slug:
        raise ValueError(
            f"summary episode mismatch for {vc_slug}/{episode_slug}: {summary_path}"
        )
    if investigation.get("episode_slug") != episode_slug:
        raise ValueError(
            f"Phase 1 episode mismatch for {vc_slug}/{episode_slug}: {investigation_path}"
        )
    expected = sidecar_path.read_text(encoding="utf-8").strip().split()[0]
    observed = sha256_file(investigation_path)
    if expected != observed:
        raise ValueError(
            f"Phase 1 hash mismatch for {vc_slug}/{episode_slug}: {investigation_path}"
        )

    run = _object(run_config.get("run"), f"run config run {run_config_path}")
    runtime_slug = str(run.get("vc_slug"))
    observed_vc = runtime_to_canonical.get(runtime_slug, runtime_slug)
    if observed_vc != vc_slug:
        raise ValueError(
            f"investor identity mismatch for {vc_slug}/{episode_slug}: {runtime_slug!r}"
        )
    if run.get("episode_slug") != episode_slug:
        raise ValueError(
            f"run-config episode mismatch for {vc_slug}/{episode_slug}: {run_config_path}"
        )


def _replacement_prediction(
    registry: EvaluationRegistry,
    vc_slug: str,
    actual_decision: str,
    source_root: Path,
) -> PredictionRow:
    return prediction_from_summary(
        registry.investors[vc_slug],
        actual_decision,
        source_root / "summary.json",
    )


def select_snapshot_sources(
    registry_path: Path,
    override_status_path: Path,
    *,
    runtime_to_canonical: Mapping[str, str] = RUNTIME_TO_CANONICAL_SLUG,
) -> list[SelectedArtifact]:
    registry = load_registry(registry_path)
    canonical_rows = load_canonical_predictions(registry)
    canonical = {(row.vc_slug, row.episode_slug): row for row in canonical_rows}
    if len(canonical) != len(canonical_rows):
        raise ValueError("canonical registry resolved duplicate VC/episode keys")

    status_path = override_status_path.resolve()
    status = _read_json(status_path, f"override status {status_path}")
    completed = status.get("completed")
    if not isinstance(completed, list):
        raise ValueError(f"override status has no completed list: {status_path}")
    project_root = registry.path.parent.parent
    overrides: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for index, value in enumerate(completed):
        record = _object(value, f"override record {index}")
        runtime_slug = record.get("vc_slug")
        vc_slug = runtime_to_canonical.get(str(runtime_slug))
        if vc_slug is None:
            raise ValueError(f"unknown override VC slug: {runtime_slug!r}")
        episode_slug = record.get("episode_slug")
        if not isinstance(episode_slug, str):
            raise ValueError(f"override has no episode slug: record {index}")
        key = (vc_slug, episode_slug)
        if key not in canonical:
            raise ValueError(f"unexpected override key: {key}")
        if key in overrides:
            raise ValueError(f"duplicate override key: {key}")
        if record.get("verification") != "verified":
            raise ValueError(f"override is not verified: {key}")
        if record.get("predicted") not in {"In", "Out"}:
            raise ValueError(f"override has no usable decision: {key}")
        if record.get("actual") != canonical[key].actual_decision:
            raise ValueError(f"override actual label mismatch: {key}")
        raw_root = record.get("artifact_root")
        if not isinstance(raw_root, str):
            raise ValueError(f"override has no artifact_root: {key}")
        overrides[key] = (_resolve_artifact_root(raw_root, project_root), record)

    selected: list[SelectedArtifact] = []
    for key, prior in sorted(canonical.items()):
        spec = registry.investors[prior.vc_slug]
        if key in overrides:
            source_root, record = overrides[key]
            replacement = _replacement_prediction(
                registry, prior.vc_slug, prior.actual_decision, source_root
            )
            if replacement.predicted_decision != record["predicted"]:
                raise ValueError(f"override status/summary decision mismatch: {key}")
            tier = "portfolio_v41_override"
        else:
            source_root = prior.artifact_path.parent
            record = None
            replacement = prior
            tier = "prior_canonical"
        _validate_artifact_identity(
            source_root, key[0], key[1], runtime_to_canonical
        )
        selected.append(
            SelectedArtifact(
                vc_slug=key[0],
                vc_name=spec.display_name,
                episode_slug=key[1],
                actual_decision=prior.actual_decision,
                selection_tier=tier,
                source_root=source_root,
                prediction=replacement,
                override_record=record,
            )
        )
    return selected


def _count_by(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    materialized = list(rows)
    values = {str(row.get(field)) for row in materialized}
    return {
        value: sum(str(row.get(field)) == value for row in materialized)
        for value in sorted(values)
    }


def _summary_metadata(selected: SelectedArtifact) -> dict[str, Any]:
    summary = _read_json(
        selected.source_root / "summary.json",
        f"selected summary {selected.source_root / 'summary.json'}",
    )
    usage = summary.get("usage") if isinstance(summary.get("usage"), dict) else {}
    return {
        "contract_version": summary.get("contract_version"),
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cached_input_tokens": int(usage.get("cached_input_tokens", 0) or 0),
        "cost_usd": float(usage.get("cost_usd", 0.0) or 0.0),
    }


def _episode_row(
    selected: SelectedArtifact,
    source_digest: DirectoryDigest,
    copied_digest: DirectoryDigest,
) -> dict[str, Any]:
    prediction = selected.prediction
    relative = Path("investors") / selected.vc_slug / selected.episode_slug
    return {
        "vc_slug": selected.vc_slug,
        "vc_name": selected.vc_name,
        "episode_slug": selected.episode_slug,
        "actual_decision": selected.actual_decision,
        "selection_tier": selected.selection_tier,
        "source_artifact_path": str(selected.source_root),
        "snapshot_artifact_path": relative.as_posix(),
        **_summary_metadata(selected),
        "predicted_decision": prediction.predicted_decision,
        "investment_likelihood": prediction.investment_likelihood,
        "decision_confidence": prediction.decision_confidence,
        "review_priority_score": prediction.review_priority_score,
        "phase1_status": prediction.phase1_status,
        "phase1_iterations": prediction.phase1_iterations,
        "phase2_status": prediction.phase2_status,
        "phase2_iterations": prediction.phase2_iterations,
        "source_directory_digest": source_digest.sha256,
        "copied_directory_digest": copied_digest.sha256,
        "file_count": copied_digest.file_count,
        "byte_count": copied_digest.byte_count,
    }


def build_snapshot(
    registry_path: Path,
    override_status_path: Path,
    destination: Path,
    *,
    runtime_to_canonical: Mapping[str, str] = RUNTIME_TO_CANONICAL_SLUG,
    expected_total: int,
    expected_overrides: int,
) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    override_status_path = override_status_path.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")

    selected = select_snapshot_sources(
        registry_path,
        override_status_path,
        runtime_to_canonical=runtime_to_canonical,
    )
    override_count = sum(
        row.selection_tier == "portfolio_v41_override" for row in selected
    )
    if len(selected) != expected_total:
        raise ValueError(
            f"expected {expected_total} total artifacts, found {len(selected)}"
        )
    if override_count != expected_overrides:
        raise ValueError(
            f"expected {expected_overrides} overrides, found {override_count}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-", dir=destination.parent
        )
    )
    rows: list[dict[str, Any]] = []
    try:
        for selected_row in selected:
            source_before = directory_digest(selected_row.source_root)
            copied_root = (
                temporary
                / "investors"
                / selected_row.vc_slug
                / selected_row.episode_slug
            )
            shutil.copytree(
                selected_row.source_root, copied_root, copy_function=shutil.copy2
            )
            source_after = directory_digest(selected_row.source_root)
            copied = directory_digest(copied_root)
            if source_before != source_after:
                raise ValueError(
                    f"source changed during copy: {selected_row.source_root}"
                )
            if source_after != copied:
                raise ValueError(
                    "copy digest mismatch: "
                    f"{selected_row.vc_slug}/{selected_row.episode_slug}"
                )
            rows.append(_episode_row(selected_row, source_after, copied))

        ordered_row_bytes = json.dumps(
            rows, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        counts = {
            "total": len(rows),
            "selection_tier": {
                "portfolio_v41_override": override_count,
                "prior_canonical": len(rows) - override_count,
            },
            "by_vc": _count_by(rows, "vc_slug"),
            "contract_version": _count_by(rows, "contract_version"),
            "actual_decision": _count_by(rows, "actual_decision"),
            "predicted_decision": _count_by(rows, "predicted_decision"),
            "phase1_status": _count_by(rows, "phase1_status"),
            "phase2_status": _count_by(rows, "phase2_status"),
        }
        manifest = {
            "schema": SNAPSHOT_SCHEMA,
            "builder_version": SNAPSHOT_BUILDER_VERSION,
            "directory_digest_algorithm": DIRECTORY_DIGEST_ALGORITHM,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_registry": str(registry_path),
            "source_registry_sha256": sha256_file(registry_path),
            "override_status": str(override_status_path),
            "override_status_sha256": sha256_file(override_status_path),
            "counts": counts,
            "totals": {
                "file_count": sum(row["file_count"] for row in rows),
                "byte_count": sum(row["byte_count"] for row in rows),
                "input_tokens": sum(row["input_tokens"] for row in rows),
                "output_tokens": sum(row["output_tokens"] for row in rows),
                "cached_input_tokens": sum(
                    row["cached_input_tokens"] for row in rows
                ),
                "cost_usd": sum(row["cost_usd"] for row in rows),
            },
            "episode_rows_sha256": hashlib.sha256(ordered_row_bytes).hexdigest(),
            "episode_rows": rows,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (temporary / "sources.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=PROVENANCE_FIELDS)
            writer.writeheader()
            writer.writerows(
                {key: row.get(key) for key in PROVENANCE_FIELDS} for row in rows
            )
        temporary.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def write_snapshot_registry(
    source_registry_path: Path,
    snapshot_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    source_registry_path = source_registry_path.resolve()
    snapshot_root = snapshot_root.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"registry already exists: {output_path}")

    source_registry = load_registry(source_registry_path)
    payload = _read_json(source_registry_path, f"registry {source_registry_path}")
    investors = _object(payload.get("investors"), "registry investors")
    for vc_slug, raw in investors.items():
        investor = _object(raw, f"registry investor {vc_slug}")
        label_path = source_registry.investors[vc_slug].label_file
        investor["label_file"] = Path(
            os.path.relpath(label_path, start=output_path.parent)
        ).as_posix()
        pattern = snapshot_root / "investors" / vc_slug / "*" / "summary.json"
        relative_pattern = Path(
            os.path.relpath(pattern, start=output_path.parent)
        ).as_posix()
        investor["sources"] = [
            {"kind": "summary_glob", "path": relative_pattern}
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    try:
        generated = load_registry(output_path)
        rows = load_canonical_predictions(generated)
        expected = sum(
            spec.eligible_count for spec in generated.investors.values()
        )
        if len(rows) != expected:
            raise ValueError(
                f"generated registry resolved {len(rows)} rows, expected {expected}"
            )
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return payload
