"""Prepare and execute canonical assessments for previously unseen pitches."""

from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Literal, Sequence

from .config import RunConfig, load_config
from .firewall import safe_slug, verify_package
from .manifest_builder import build_episode_manifest
from .rehearsal_config import RehearsalConfig
from .rehearsal_grounding import (
    CanonicalRehearsalBaseline,
    load_canonical_baseline_from_run,
)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"trusted live-package source is not a directory: {source}")
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"trusted live-package source contains symlink: {path}")
        if path.is_file():
            _copy_file(path, destination / path.relative_to(source))


def _pitch_text(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("pitch text must not be empty")
    return text + "\n"


def prepare_live_package(
    *,
    source_root: Path,
    destination_root: Path,
    vc_slug: str,
    episode_slug: str,
    pitch: str,
    target_company_aliases: Sequence[str] = (),
) -> Path:
    """Create an immutable, manifest-bound package overlay for one live pitch."""
    source = Path(source_root).resolve(strict=True)
    destination = Path(destination_root).resolve(strict=False)
    vc = safe_slug(vc_slug, "vc_slug")
    episode = safe_slug(episode_slug, "episode_slug")
    normalized_pitch = _pitch_text(pitch)
    pitch_digest = sha256(normalized_pitch.encode("utf-8")).hexdigest()
    pitch_path = destination / f"data/investors/{vc}/pitches/{episode}.txt"

    if destination.exists():
        if not pitch_path.is_file():
            raise ValueError("existing live package is incomplete")
        if sha256(pitch_path.read_bytes()).hexdigest() != pitch_digest:
            raise ValueError("existing live package contains a different pitch")
        verify_package(destination, vc, episode)
        return destination

    destination.mkdir(parents=True)
    required_files = (
        (source / f"investors/{vc}.toml", destination / f"investors/{vc}.toml"),
        (
            source / "taxonomy/codebook_v_final.json",
            destination / "taxonomy/codebook_v_final.json",
        ),
    )
    for original, copied in required_files:
        if not original.is_file() or original.is_symlink():
            raise ValueError(f"trusted live-package source is missing: {original}")
        _copy_file(original, copied)
    _copy_tree(source / f"wiki/{vc}", destination / f"wiki/{vc}")
    _copy_tree(source / "indexes", destination / "indexes")
    _copy_tree(
        source / f"data/investors/{vc}/precedents",
        destination / f"data/investors/{vc}/precedents",
    )
    _copy_tree(
        source / f"data/investors/{vc}/portfolio-memory",
        destination / f"data/investors/{vc}/portfolio-memory",
    )

    pitch_path.parent.mkdir(parents=True, exist_ok=True)
    pitch_path.write_text(normalized_pitch, encoding="utf-8")
    audit_path = destination / f"data/investors/{vc}/audits/{episode}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                "schema": "pitch-leakage-audit-v1",
                "status": "audited",
                "audit_source": "live-founder-submission",
                "vc_slug": vc,
                "episode_slug": episode,
                "pitch_path": pitch_path.relative_to(destination).as_posix(),
                "pitch_sha256": pitch_digest,
                "target_company_aliases": [
                    str(alias).strip()
                    for alias in target_company_aliases
                    if str(alias).strip()
                ],
                "leakage_checklist": {
                    "actual_label_in_package": False,
                    "complete_transcript_in_package": False,
                    "founder_decision_acknowledgement_retained": False,
                    "investor_evaluation_retained": False,
                    "narrator_evaluation_retained": False,
                    "post_decision_material_retained": False,
                    "prior_episode_inputs_in_package": False,
                    "target_investor_identity_in_pitch": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    build_episode_manifest(destination, vc, episode)
    verify_package(destination, vc, episode)
    return destination


def _relative(workspace: Path, path: Path, *, label: str) -> str:
    try:
        return path.resolve(strict=False).relative_to(workspace.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the workspace") from exc


def derive_live_run_config(
    *,
    template: RunConfig,
    workspace: Path,
    session_root: Path,
    vc_slug: str,
    episode_slug: str,
    contract_version: Literal["v4", "v4.1"],
) -> RunConfig:
    """Bind a canonical template to one session-owned live run."""
    root = Path(workspace).resolve()
    session = Path(session_root).resolve(strict=False)
    bootstrap = session / "canonical-bootstrap"
    run = template.run.model_copy(
        update={
            "vc_slug": safe_slug(vc_slug, "vc_slug"),
            "episode_slug": safe_slug(episode_slug, "episode_slug"),
            "input_root": _relative(root, bootstrap / "inputs", label="live input"),
            "output_root": _relative(root, bootstrap / "runs", label="live output"),
            "checkpoint_path": _relative(
                root,
                bootstrap / "checkpoints/canonical.sqlite",
                label="canonical checkpoint",
            ),
            "contract_version": contract_version,
            "mode": "full",
        }
    )
    derived = template.model_copy(update={"run": run})
    derived._workspace = root
    return derived


def _bootstrap_record(
    path: Path,
    *,
    template_path: Path,
    config: RunConfig,
    baseline: CanonicalRehearsalBaseline | Any,
    pitch_sha256: str,
) -> None:
    state = getattr(baseline, "phase1_state", {})
    usage = state.get("usage", {}) if isinstance(state, dict) else {}
    payload = {
        "schema": "live-canonical-rehearsal-bootstrap-v1",
        "status": "complete",
        "template_path": str(template_path),
        "template_sha256": sha256(template_path.read_bytes()).hexdigest(),
        "derived_config_sha256": sha256(
            json.dumps(
                config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "pitch_sha256": pitch_sha256,
        "investigation_sha256": getattr(baseline, "investigation_sha256", None),
        "decision_sha256": getattr(baseline, "decision_sha256", None),
        "usage": usage,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_or_load_canonical_baseline(
    *,
    workspace: Path,
    rehearsal_config: RehearsalConfig,
    session_root: Path,
    vc_slug: str,
    episode_slug: str,
    pitch: str,
    target_company_aliases: Sequence[str] = (),
    runner: Callable[..., None] | None = None,
) -> CanonicalRehearsalBaseline:
    """Build exactly once, verify, and return a live canonical baseline."""
    root = Path(workspace).resolve()
    session = Path(session_root).resolve(strict=False)
    configured_template = rehearsal_config.classification.canonical_config_path
    if configured_template is None:
        raise ValueError("grounded live rehearsal lacks canonical_config_path")
    template_path = (root / configured_template).resolve(strict=True)
    if not template_path.is_relative_to(root):
        raise ValueError("canonical configuration escapes the workspace")
    template = load_config(template_path)
    prepared = prepare_live_package(
        source_root=root / rehearsal_config.rehearsal.input_root,
        destination_root=session / "canonical-bootstrap/inputs",
        vc_slug=vc_slug,
        episode_slug=episode_slug,
        pitch=pitch,
        target_company_aliases=target_company_aliases,
    )
    package = verify_package(prepared, vc_slug, episode_slug)
    pitch_digest = sha256(package.pitch.read_bytes()).hexdigest()
    derived = derive_live_run_config(
        template=template,
        workspace=root,
        session_root=session,
        vc_slug=vc_slug,
        episode_slug=episode_slug,
        contract_version=rehearsal_config.classification.live_contract_version,
    )
    run_root = root / derived.run.output_root / episode_slug
    record_path = session / "canonical-bootstrap/bootstrap.json"
    if not record_path.is_file():
        canonical_runner = runner
        capture_stdout = canonical_runner is None
        if canonical_runner is None:
            from .cli import command_run

            canonical_runner = command_run
        checkpoint = root / derived.run.checkpoint_path
        try:
            if capture_stdout:
                with redirect_stdout(StringIO()):
                    canonical_runner(derived, resume=checkpoint.is_file())
            else:
                canonical_runner(derived, resume=checkpoint.is_file())
        except Exception as exc:
            failure_path = session / "canonical-bootstrap/bootstrap-failure.json"
            failure_path.write_text(
                json.dumps(
                    {
                        "schema": "live-canonical-rehearsal-bootstrap-failure-v1",
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "checkpoint_available": checkpoint.is_file(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            raise
    baseline = load_canonical_baseline_from_run(
        workspace=root,
        run_root=run_root,
        canonical_vc_slug=vc_slug,
        expected_run_vc_slug=vc_slug,
        expected_episode_slug=episode_slug,
        expected_pitch_sha256=pitch_digest,
    )
    if not record_path.is_file():
        _bootstrap_record(
            record_path,
            template_path=template_path,
            config=derived,
            baseline=baseline,
            pitch_sha256=pitch_digest,
        )
    return baseline


def import_canonical_baseline(
    *,
    workspace: Path,
    source_run: Path,
    session_root: Path,
    expected_vc_slug: str,
    expected_pitch_sha256: str,
) -> CanonicalRehearsalBaseline:
    """Verify and copy an existing run into session-owned durable storage."""
    root = Path(workspace).resolve()
    source = Path(source_run).resolve(strict=True)
    if not source.is_relative_to(root):
        raise ValueError("canonical baseline must be inside the workspace")
    run_config = json.loads((source / "run-config.json").read_text(encoding="utf-8"))
    run_identity = run_config.get("run")
    if not isinstance(run_identity, dict) or not isinstance(
        run_identity.get("episode_slug"), str
    ):
        raise ValueError("canonical baseline run identity is invalid")
    episode_slug = str(run_identity["episode_slug"])
    load_canonical_baseline_from_run(
        workspace=root,
        run_root=source,
        canonical_vc_slug=expected_vc_slug,
        expected_run_vc_slug=expected_vc_slug,
        expected_episode_slug=episode_slug,
        expected_pitch_sha256=expected_pitch_sha256,
    )
    destination = Path(session_root).resolve() / "canonical-bootstrap/imported-run"
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("imported canonical baseline path is invalid")
    else:
        shutil.copytree(source, destination)
    return load_canonical_baseline_from_run(
        workspace=root,
        run_root=destination,
        canonical_vc_slug=expected_vc_slug,
        expected_run_vc_slug=expected_vc_slug,
        expected_episode_slug=episode_slug,
        expected_pitch_sha256=expected_pitch_sha256,
    )
