"""Read the saved artifacts of one assessment run directory."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    vc_slug: str
    episode_slug: str
    investor_name: str
    pitch_text: str
    phase1: dict[str, Any]
    phase2: dict[str, Any] | None
    policy_evidence: tuple[dict[str, Any], ...]
    taxonomy: tuple[dict[str, Any], ...]
    hashes: dict[str, str]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return value


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _resolve(workspace: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def load_taxonomy(path: Path) -> tuple[dict[str, Any], ...]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not all("label" in r for r in records):
        raise ValueError(f"taxonomy must be a list of labelled records: {path}")
    return tuple(records)


def load_run(
    run_dir: Path,
    *,
    workspace: Path = Path("."),
    taxonomy_path: Path | None = None,
    require_phase2: bool = False,
) -> RunArtifacts:
    """Load one run. Pitch and taxonomy locations come from the run's own records."""
    run_dir = Path(run_dir)
    phase1_path = run_dir / "phase1" / "investigation.json"
    phase1 = _read(phase1_path)
    hashes = {"phase1": _digest(phase1_path)}

    phase2_path = run_dir / "phase2" / "decision.json"
    phase2 = _read(phase2_path) if phase2_path.is_file() else None
    if phase2 is not None:
        hashes["phase2"] = _digest(phase2_path)
    elif require_phase2:
        raise ValueError(f"run has no Phase 2 decision: {run_dir}")

    run_config = _read(run_dir / "run-config.json")["run"]
    if run_config.get("contract_version") == "v4.4":
        raise ValueError("v4.4 run inventories forbid extra artifacts; use a v4.1-family run")

    provenance = _read(run_dir / "input-provenance.json")
    pitch_path = _resolve(workspace, provenance["pitch_path"])
    hashes["pitch"] = _digest(pitch_path)
    if hashes["pitch"] != provenance.get("pitch_sha256"):
        raise ValueError(f"pitch changed since the run: {pitch_path}")

    if taxonomy_path is None:
        taxonomy_path = _resolve(workspace, run_config["input_root"]) / run_config["taxonomy_path"]
    hashes["taxonomy"] = _digest(taxonomy_path)

    policy_path = run_dir / "assessment-policy.json"
    policy: dict[str, Any] = {}
    if policy_path.is_file():
        policy = _read(policy_path)
        hashes["policy"] = _digest(policy_path)
    investor_name = (policy.get("target_context") or {}).get("investor_name") or run_config["vc_slug"]

    return RunArtifacts(
        run_dir=run_dir,
        vc_slug=run_config["vc_slug"],
        episode_slug=run_config["episode_slug"],
        investor_name=investor_name,
        pitch_text=pitch_path.read_text(encoding="utf-8"),
        phase1=phase1,
        phase2=phase2,
        policy_evidence=tuple(policy.get("evidence") or ()),
        taxonomy=load_taxonomy(taxonomy_path),
        hashes=hashes,
    )
