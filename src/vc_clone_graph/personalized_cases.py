"""Verified population assembly for personalized multi-investor models."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import tomllib
from typing import Mapping, Sequence

from .artifacts import read_verified_frozen
from .firewall import (
    resolve_manifest_path,
    safe_slug,
    sha256_file,
    validate_package_path,
    verify_package,
)
from .phase1_evaluation import Phase1Case, load_phase1_cases
from .phase2_calibration_evaluation import (
    Phase2CalibrationRecord,
    load_phase2_records,
    validate_phase2_payloads,
)
from .posthoc_semantic_models import (
    extract_phase2_reasoning_view,
    extract_reasoning_views,
)
from .standalone_pitch_models import extract_phase1_rationale_features


@dataclass(frozen=True)
class InvestorInput:
    vc_input_slug: str
    display_name: str


@dataclass(frozen=True)
class ArtifactViews:
    episode_slug: str
    phase1_text: str
    phase2_text: str
    phase1_features: Mapping[str, float]
    phase2_features: Mapping[str, float]
    phase1_sha256: str
    phase2_sha256: str


@dataclass(frozen=True)
class PackageViews:
    vc_input_slug: str
    episode_slug: str
    pitch_text: str
    wiki_text: str
    pitch_sha256: str
    wiki_sha256: str


@dataclass(frozen=True)
class PersonalizedCase:
    vc_slug: str
    vc_input_slug: str
    vc_name: str
    episode_slug: str
    group: str
    target: int
    pitch_text: str
    phase1_text: str
    phase2_text: str
    phase1_features: Mapping[str, float]
    phase2_features: Mapping[str, float]
    wiki_text: str
    actual_rationale_targets: Mapping[str, int]
    actual_rationale_available: bool
    actual_rationale_source_tier: str
    actual_rationale_source_format: str
    source_hashes: Mapping[str, str]


def render_wiki(path: Path) -> tuple[str, str]:
    """Render Markdown wiki files in stable, source-visible order."""
    root = Path(path)
    files = sorted(
        (item for item in root.rglob("*.md") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError(f"investor wiki contains no Markdown files: {root}")
    blocks = []
    for item in files:
        relative = item.relative_to(root).as_posix()
        blocks.append(f"----- SOURCE: {relative} -----\n{item.read_text(encoding='utf-8').strip()}")
    text = "\n\n".join(blocks).strip() + "\n"
    return text, sha256(text.encode("utf-8")).hexdigest()


def assemble_personalized_cases(
    records: Sequence[Phase2CalibrationRecord],
    *,
    phase1_by_key: Mapping[tuple[str, str], Phase1Case],
    investor_by_name: Mapping[str, InvestorInput],
    artifact_by_key: Mapping[tuple[str, str], ArtifactViews],
    package_by_key: Mapping[tuple[str, str], PackageViews],
) -> list[PersonalizedCase]:
    """Join already verified inputs while enforcing identity and hash alignment."""
    result: list[PersonalizedCase] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (record.vc_slug, record.episode_slug)
        if key in seen:
            raise ValueError(f"duplicate personalized case: {key}")
        seen.add(key)
        rationale = phase1_by_key.get(key)
        artifact = artifact_by_key.get(key)
        package = package_by_key.get(key)
        investor = investor_by_name.get(record.vc_name)
        if investor is None:
            raise ValueError(f"investor profile is missing: {record.vc_name}")
        if rationale is None or artifact is None or package is None:
            raise ValueError(f"missing aligned source for investor episode: {key}")
        if rationale.actual_decision != ("In" if record.target else "Out"):
            raise ValueError(f"actual decision mismatch: {key}")
        if rationale.vc_name != record.vc_name:
            raise ValueError(f"investor identity mismatch: {key}")
        if any(
            value != record.episode_slug
            for value in (rationale.episode_slug, artifact.episode_slug, package.episode_slug)
        ):
            raise ValueError(f"episode identity mismatch: {key}")
        if package.vc_input_slug != investor.vc_input_slug:
            raise ValueError(f"investor package mismatch: {key}")
        if artifact.phase1_sha256 != record.phase1_sha256:
            raise ValueError(f"Phase 1 hash mismatch: {key}")
        if artifact.phase2_sha256 != record.phase2_sha256:
            raise ValueError(f"Phase 2 hash mismatch: {key}")

        result.append(PersonalizedCase(
            vc_slug=record.vc_slug,
            vc_input_slug=investor.vc_input_slug,
            vc_name=record.vc_name,
            episode_slug=record.episode_slug,
            group=record.group,
            target=record.target,
            pitch_text=package.pitch_text,
            phase1_text=artifact.phase1_text,
            phase2_text=artifact.phase2_text,
            phase1_features=dict(artifact.phase1_features),
            phase2_features=dict(artifact.phase2_features),
            wiki_text=package.wiki_text,
            actual_rationale_targets={item.label: 1 for item in rationale.reference},
            actual_rationale_available=True,
            actual_rationale_source_tier=rationale.source_tier,
            actual_rationale_source_format=rationale.source_format,
            source_hashes={
                "pitch": package.pitch_sha256,
                "wiki": package.wiki_sha256,
                "phase1": artifact.phase1_sha256,
                "phase2": artifact.phase2_sha256,
            },
        ))
    return sorted(result, key=lambda row: (row.vc_slug, row.episode_slug))


def load_investor_inputs(input_root: Path) -> dict[str, InvestorInput]:
    """Load exact display-name to inference-package mappings."""
    result: dict[str, InvestorInput] = {}
    for path in sorted((Path(input_root) / "investors").glob("*.toml")):
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        display_name = payload.get("display_name")
        vc_input_slug = payload.get("vc_slug")
        if not isinstance(display_name, str) or not isinstance(vc_input_slug, str):
            raise ValueError(f"invalid investor profile: {path}")
        if display_name in result:
            raise ValueError(f"duplicate investor display name: {display_name}")
        result[display_name] = InvestorInput(vc_input_slug, display_name)
    return result


def verify_episode_pitch_assets(
    input_root: Path, vc_slug: str, episode_slug: str
) -> tuple[Path, str]:
    """Verify episode-specific assets after shared investor assets were verified."""
    root = Path(input_root).resolve(strict=True)
    vc = safe_slug(vc_slug, "vc_slug")
    episode = safe_slug(episode_slug, "episode_slug")
    pitch = root / f"data/investors/{vc}/pitches/{episode}.txt"
    audit = root / f"data/investors/{vc}/audits/{episode}.json"
    manifest = resolve_manifest_path(root, vc, episode)
    for path in (pitch, audit, manifest):
        if not path.is_file():
            raise ValueError(f"required episode file is missing: {path}")
        validate_package_path(root, path)
    try:
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        audit_data = json.loads(audit.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("episode manifest or audit is invalid") from exc
    if (
        manifest_data.get("vc_slug") != vc
        or manifest_data.get("episode_slug") != episode
    ):
        raise ValueError("episode manifest identity mismatch")
    rows = manifest_data.get("files")
    if not isinstance(rows, list):
        raise ValueError("episode manifest files are invalid")
    hashes: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"destination", "sha256"}:
            raise ValueError("episode manifest row is invalid")
        destination, digest = row["destination"], row["sha256"]
        if not isinstance(destination, str) or not isinstance(digest, str):
            raise ValueError("episode manifest row values are invalid")
        if destination in hashes:
            raise ValueError("episode manifest paths are not unique")
        hashes[destination] = digest
    pitch_relative = pitch.relative_to(root).as_posix()
    audit_relative = audit.relative_to(root).as_posix()
    pitch_digest = sha256_file(pitch)
    if hashes.get(pitch_relative) != pitch_digest:
        raise ValueError("episode pitch hash mismatch")
    if hashes.get(audit_relative) != sha256_file(audit):
        raise ValueError("episode audit hash mismatch")
    if (
        audit_data.get("status") not in {"approved", "audited"}
        or audit_data.get("vc_slug") != vc
        or audit_data.get("episode_slug") != episode
        or audit_data.get("pitch_sha256") != pitch_digest
    ):
        raise ValueError("episode audit identity or pitch hash mismatch")
    checklist = audit_data.get("leakage_checklist")
    if not isinstance(checklist, dict) or any(value is not False for value in checklist.values()):
        raise ValueError("episode pitch audit is not leakage-clean")
    return pitch.resolve(), pitch_digest


def _load_artifact_views(
    records: Sequence[Phase2CalibrationRecord],
) -> dict[tuple[str, str], ArtifactViews]:
    result = {}
    for record in records:
        root = Path(record.artifact_root)
        phase1_raw, phase1_digest = read_verified_frozen(
            root / "phase1/investigation.json", root / "phase1/investigation.sha256"
        )
        phase2_raw, phase2_digest = read_verified_frozen(
            root / "phase2/decision.json", root / "phase2/decision.sha256"
        )
        investigation, decision = validate_phase2_payloads(phase1_raw, phase2_raw)
        views = extract_reasoning_views(investigation, decision)
        result[(record.vc_slug, record.episode_slug)] = ArtifactViews(
            episode_slug=record.episode_slug,
            phase1_text=views["phase1_rationales"],
            phase2_text=extract_phase2_reasoning_view(decision),
            phase1_features=extract_phase1_rationale_features(investigation),
            phase2_features=dict(record.phase2_features),
            phase1_sha256=phase1_digest,
            phase2_sha256=phase2_digest,
        )
    return result


def _load_package_views(
    records: Sequence[Phase2CalibrationRecord],
    input_root: Path,
    investor_by_name: Mapping[str, InvestorInput],
) -> dict[tuple[str, str], PackageViews]:
    wiki_cache: dict[str, tuple[str, str]] = {}
    verified_shared: dict[str, Path] = {}
    result = {}
    for record in records:
        investor = investor_by_name.get(record.vc_name)
        if investor is None:
            raise ValueError(f"investor profile is missing: {record.vc_name}")
        if investor.vc_input_slug not in verified_shared:
            package = verify_package(input_root, investor.vc_input_slug, record.episode_slug)
            verified_shared[investor.vc_input_slug] = package.wiki
            pitch_path, pitch_digest = package.pitch, sha256_file(package.pitch)
        else:
            pitch_path, pitch_digest = verify_episode_pitch_assets(
                input_root, investor.vc_input_slug, record.episode_slug
            )
        if investor.vc_input_slug not in wiki_cache:
            wiki_cache[investor.vc_input_slug] = render_wiki(
                verified_shared[investor.vc_input_slug]
            )
        wiki_text, wiki_digest = wiki_cache[investor.vc_input_slug]
        result[(record.vc_slug, record.episode_slug)] = PackageViews(
            vc_input_slug=investor.vc_input_slug,
            episode_slug=record.episode_slug,
            pitch_text=pitch_path.read_text(encoding="utf-8"),
            wiki_text=wiki_text,
            pitch_sha256=pitch_digest,
            wiki_sha256=wiki_digest,
        )
    return result


def load_personalized_cases(
    project_root: Path,
    registry_path: Path,
    input_root: Path,
    reference_root: Path,
    taxonomy_path: Path,
) -> list[PersonalizedCase]:
    """Load the complete canonical personalized decision population."""
    records = load_phase2_records(registry_path)
    phase1_cases, _ = load_phase1_cases(
        project_root, registry_path, reference_root, taxonomy_path
    )
    phase1_by_key = {(row.vc_slug, row.episode_slug): row for row in phase1_cases}
    investors = load_investor_inputs(input_root)
    cases = assemble_personalized_cases(
        records,
        phase1_by_key=phase1_by_key,
        investor_by_name=investors,
        artifact_by_key=_load_artifact_views(records),
        package_by_key=_load_package_views(records, input_root, investors),
    )
    if len(cases) != 301 or sum(row.target for row in cases) != 86:
        raise ValueError("canonical personalized population must contain 301 cases and 86 Ins")
    if len({row.episode_slug for row in cases}) != 136:
        raise ValueError("canonical personalized population must contain 136 episodes")
    if len({row.vc_slug for row in cases}) != 6:
        raise ValueError("canonical personalized population must contain six investors")
    return cases
