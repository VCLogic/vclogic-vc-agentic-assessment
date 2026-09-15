"""Freeze completed live-batch records for reproducible interim calibration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InterimSnapshot:
    """Paths and label counts for one immutable completed-record snapshot."""

    status_path: Path
    labels_path: Path
    episode_count: int
    in_count: int
    out_count: int


def _write_json_atomic(path: Path, value: Any) -> None:
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    temporary = candidate.with_name(candidate.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(candidate)


def freeze_completed_snapshot(
    source_status_path: Path,
    source_labels_path: Path,
    output_root: Path,
) -> InterimSnapshot:
    """Copy completed records and their audited labels without mutating live inputs."""
    status_source = Path(source_status_path)
    labels_source = Path(source_labels_path)
    status_raw = status_source.read_bytes()
    labels_raw = labels_source.read_bytes()
    status = json.loads(status_raw)
    labels = json.loads(labels_raw)
    if not isinstance(status, dict):
        raise ValueError("batch status must be a JSON object")
    if not isinstance(labels, list):
        raise ValueError("label file must be a JSON array")
    completed = status.get("completed_records")
    if not isinstance(completed, list) or not completed:
        raise ValueError("batch status has no completed records")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_record in completed:
        if not isinstance(source_record, dict):
            raise ValueError("completed batch record must be an object")
        record = deepcopy(source_record)
        slug = str(record.get("episode_slug", ""))
        if not slug or slug in seen:
            raise ValueError(f"duplicate or missing completed episode slug: {slug!r}")
        if record.get("status") != "completed":
            raise ValueError(f"non-completed record in completed population: {slug}")
        seen.add(slug)
        summary = record.get("summary", {})
        if isinstance(summary, dict):
            for field in (
                "phase1_status",
                "phase1_findings",
                "phase2_status",
                "phase2_findings",
            ):
                if field not in record and field in summary:
                    record[field] = deepcopy(summary[field])
        normalized.append(record)

    eligible_by_slug: dict[str, dict[str, Any]] = {}
    for source_row in labels:
        if not isinstance(source_row, dict) or source_row.get("evaluation_eligible") is not True:
            continue
        slug = str(source_row.get("episode_slug", ""))
        target = source_row.get("pitch_window_decision")
        if not slug or target not in {"In", "Out"} or slug in eligible_by_slug:
            raise ValueError(f"invalid or duplicate eligible label: {slug!r}")
        eligible_by_slug[slug] = deepcopy(source_row)
    missing = seen - set(eligible_by_slug)
    if missing:
        raise ValueError("completed episodes lack audited labels: " + ", ".join(sorted(missing)))
    frozen_labels = [eligible_by_slug[record["episode_slug"]] for record in normalized]

    root = Path(output_root)
    frozen_status = {
        **deepcopy(status),
        "completed_records": normalized,
        "failed_records": [],
        "progress": {"processed": len(normalized), "total": len(normalized)},
    }
    status_path = root / "batch-status.snapshot.json"
    labels_path = root / "labels.snapshot.json"
    _write_json_atomic(status_path, frozen_status)
    _write_json_atomic(labels_path, frozen_labels)
    in_count = sum(row["pitch_window_decision"] == "In" for row in frozen_labels)
    provenance = {
        "schema": "interim-calibration-snapshot-v1",
        "source_status_path": str(status_source.resolve()),
        "source_status_sha256": sha256(status_raw).hexdigest(),
        "source_labels_path": str(labels_source.resolve()),
        "source_labels_sha256": sha256(labels_raw).hexdigest(),
        "episode_count": len(normalized),
        "in_count": in_count,
        "out_count": len(normalized) - in_count,
        "episode_slugs": [record["episode_slug"] for record in normalized],
    }
    _write_json_atomic(root / "snapshot-provenance.json", provenance)
    return InterimSnapshot(
        status_path=status_path,
        labels_path=labels_path,
        episode_count=len(normalized),
        in_count=in_count,
        out_count=len(normalized) - in_count,
    )
