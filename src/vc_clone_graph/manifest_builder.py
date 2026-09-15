"""Build deterministic episode manifests from already-audited inference assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .firewall import (
    InputBoundaryError,
    safe_slug,
    sha256_file,
    validate_package_path,
    validate_portfolio_memory,
    validate_precedent_corpus,
)


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise InputBoundaryError(f"manifest path contains symlink: {path}")


def _validate_output_path(package_root: Path, subtree: Path, target: Path) -> None:
    try:
        target.relative_to(subtree)
        target.resolve(strict=False).relative_to(package_root)
    except ValueError as exc:
        raise InputBoundaryError("manifest output path escapes package root") from exc
    current = package_root
    for part in target.relative_to(package_root).parts:
        current = current / part
        if current.is_symlink():
            raise InputBoundaryError("manifest output path contains symlink")


def build_episode_manifest(root: Path, vc_slug: str, episode_slug: str) -> Path:
    requested_root = Path(root)
    _reject_symlink_components(requested_root)
    try:
        package_root = requested_root.resolve(strict=True)
    except OSError as exc:
        raise InputBoundaryError("manifest package root does not exist") from exc
    if not package_root.is_dir():
        raise InputBoundaryError("manifest package root is not a directory")
    vc_slug = safe_slug(vc_slug, "vc_slug")
    episode_slug = safe_slug(episode_slug, "episode_slug")
    registry = package_root / "investors" / f"{vc_slug}.toml"
    taxonomy = package_root / "taxonomy" / "codebook_v_final.json"
    pitch = (
        package_root
        / "data"
        / "investors"
        / vc_slug
        / "pitches"
        / f"{episode_slug}.txt"
    )
    audit = (
        package_root
        / "data"
        / "investors"
        / vc_slug
        / "audits"
        / f"{episode_slug}.json"
    )
    wiki = package_root / "wiki" / vc_slug
    precedents = package_root / "data" / "investors" / vc_slug / "precedents"
    portfolio_memory = (
        package_root / "data" / "investors" / vc_slug / "portfolio-memory"
    )
    candidates = (
        (registry, package_root / "investors"),
        (taxonomy, package_root / "taxonomy"),
        (pitch, package_root / "data" / "investors" / vc_slug / "pitches"),
        (audit, package_root / "data" / "investors" / vc_slug / "audits"),
    )
    for path, subtree in candidates:
        _validate_output_path(package_root, subtree, path)
        if not path.is_file():
            raise InputBoundaryError(
                f"manifest source file is missing: {path.relative_to(package_root)}"
            )
        validate_package_path(package_root, path)
    _validate_output_path(
        package_root,
        package_root / "data" / "investors" / vc_slug,
        precedents,
    )
    _validate_output_path(
        package_root,
        package_root / "data" / "investors" / vc_slug,
        portfolio_memory,
    )
    _validate_output_path(package_root, package_root / "wiki", wiki)
    if not wiki.is_dir():
        raise InputBoundaryError(f"manifest source directory is missing: wiki/{vc_slug}")
    validate_package_path(package_root, wiki)
    wiki_paths = sorted(wiki.rglob("*"))
    for path in wiki_paths:
        validate_package_path(package_root, path)
    precedent_files: tuple[Path, ...] = ()
    if precedents.exists() or precedents.is_symlink():
        validate_package_path(package_root, precedents)
        precedent_files = validate_precedent_corpus(precedents)
    portfolio_files: tuple[Path, ...] = ()
    if portfolio_memory.exists() or portfolio_memory.is_symlink():
        validate_package_path(package_root, portfolio_memory)
        portfolio_files = validate_portfolio_memory(portfolio_memory, vc_slug)
    shared = [
        registry,
        taxonomy,
        *(path for path in wiki_paths if path.is_file()),
        *precedent_files,
        *portfolio_files,
    ]
    files = [*shared, pitch, audit]
    for path in files:
        validate_package_path(package_root, path)
    rows = [
        {
            "destination": path.relative_to(package_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    manifest_root = (
        package_root
        / "data"
        / "investors"
        / vc_slug
        / "manifests"
    )
    target = manifest_root / f"{episode_slug}.json"
    _validate_output_path(package_root, manifest_root, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema": "langgraph-vc-clone-source-manifest-v1",
                "vc_slug": vc_slug,
                "episode_slug": episode_slug,
                "files": sorted(rows, key=lambda row: row["destination"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--vc", required=True)
    parser.add_argument("--episode", required=True)
    args = parser.parse_args()
    print(build_episode_manifest(args.root, args.vc, args.episode))


if __name__ == "__main__":
    main()
