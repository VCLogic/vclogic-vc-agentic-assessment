"""Copy a minimal, hash-bound inference package from the source project."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

from .firewall import (
    InputBoundaryError,
    resolve_manifest_path,
    safe_manifest_relative,
    safe_slug,
    validate_portfolio_memory,
    validate_precedent_corpus,
    verify_package,
)


def _validate_selected_source_path(source_root: Path, path: Path) -> None:
    """Require a selected source path to stay in-root without symlink hops."""
    try:
        relative = path.relative_to(source_root)
    except ValueError as exc:
        raise InputBoundaryError("selected source path escapes source root") from exc
    current = source_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise InputBoundaryError(
                f"selected source path contains symlink: {relative}"
            )
    try:
        path.resolve(strict=True).relative_to(source_root)
    except (OSError, ValueError) as exc:
        raise InputBoundaryError("selected source path escapes source root") from exc


def _before_copy_selected(relative: str, source_path: Path) -> None:
    """Deterministic test seam immediately before the no-follow source open."""


def _open_selected_descriptor(source_root_fd: int, relative: str) -> int:
    """Open a safe relative file by no-follow traversal from source_root_fd."""
    if (
        type(relative) is not str
        or not relative
        or relative.startswith("/")
        or "\\" in relative
    ):
        raise InputBoundaryError("selected source descriptor path is unsafe")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise InputBoundaryError("selected source descriptor path is unsafe")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    opened_directories: list[int] = []
    parent_fd = source_root_fd
    try:
        for component in parts[:-1]:
            parent_fd = os.open(
                component,
                directory_flags,
                dir_fd=parent_fd,
            )
            opened_directories.append(parent_fd)
            if not stat.S_ISDIR(os.fstat(parent_fd).st_mode):
                raise InputBoundaryError(
                    "selected source descriptor ancestor is not a directory"
                )
        return os.open(parts[-1], file_flags, dir_fd=parent_fd)
    except OSError as exc:
        raise InputBoundaryError(
            f"selected source descriptor traversal failed: {relative}"
        ) from exc
    finally:
        for descriptor in reversed(opened_directories):
            os.close(descriptor)


def _copy_selected_file(
    source_root_fd: int,
    relative: str,
    target: Path,
    expected: str | None,
) -> str:
    """Copy and hash the bytes read from one no-follow regular-file descriptor."""
    descriptor = _open_selected_descriptor(source_root_fd, relative)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise InputBoundaryError(f"selected source is not regular: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256()
        source_stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        with source_stream, target.open("xb") as target_stream:
            for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                digest.update(chunk)
                target_stream.write(chunk)
        actual = digest.hexdigest()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if expected is not None and actual != expected:
        raise InputBoundaryError(f"source hash mismatch: {relative}")
    return actual


def import_assets(source: Path, destination: Path, vc_slug: str, episode_slug: str) -> Path:
    vc_slug = safe_slug(vc_slug, "vc_slug")
    episode_slug = safe_slug(episode_slug, "episode_slug")
    requested_source = Path(source)
    if requested_source.is_symlink():
        raise InputBoundaryError("source root cannot be a symlink")
    source_root = requested_source.resolve()
    destination_root = Path(destination).resolve()
    if destination_root.exists():
        raise FileExistsError(f"destination already exists: {destination_root}")
    source_manifest = resolve_manifest_path(source_root, vc_slug, episode_slug)
    _validate_selected_source_path(source_root, source_manifest)
    try:
        manifest_data = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputBoundaryError("source manifest is invalid") from exc
    raw_rows = manifest_data.get("files")
    if type(raw_rows) is not list:
        raise InputBoundaryError("source manifest files are invalid")
    source_hashes: dict[str, str | None] = {}
    for row in raw_rows:
        if type(row) is not dict or set(row) != {"destination", "sha256"}:
            raise InputBoundaryError("source manifest row is invalid")
        relative = safe_manifest_relative(row["destination"]).as_posix()
        digest = row["sha256"]
        if relative in source_hashes:
            raise InputBoundaryError("source manifest paths are not unique")
        source_hashes[relative] = digest
    registry_relative = f"investors/{vc_slug}.toml"
    manifest_required = {
        f"data/investors/{vc_slug}/pitches/{episode_slug}.txt",
        f"data/investors/{vc_slug}/audits/{episode_slug}.json",
        "taxonomy/codebook_v_final.json",
    }
    registry_path = source_root / registry_relative
    _validate_selected_source_path(source_root, registry_path)
    if not registry_path.is_file():
        raise InputBoundaryError("source investor registry is missing")
    source_hashes.setdefault(registry_relative, None)
    required = manifest_required | {registry_relative}
    wiki_prefix = f"wiki/{vc_slug}/"
    source_precedents = source_root / f"data/investors/{vc_slug}/precedents"
    precedent_prefix = f"data/investors/{vc_slug}/precedents/"
    precedent_relatives: set[str] = set()
    if source_precedents.exists() or source_precedents.is_symlink():
        _validate_selected_source_path(source_root, source_precedents)
        for path in source_precedents.rglob("*"):
            _validate_selected_source_path(source_root, path)
        precedent_relatives = {
            path.relative_to(source_root).as_posix()
            for path in validate_precedent_corpus(source_precedents)
        }
        if not precedent_relatives <= source_hashes.keys():
            raise InputBoundaryError("precedent corpus files are not manifest-bound")
    elif any(relative.startswith(precedent_prefix) for relative in source_hashes):
        raise InputBoundaryError("source manifest references an absent precedent corpus")
    source_portfolio = source_root / f"data/investors/{vc_slug}/portfolio-memory"
    portfolio_prefix = f"data/investors/{vc_slug}/portfolio-memory/"
    portfolio_relatives: set[str] = set()
    if source_portfolio.exists() or source_portfolio.is_symlink():
        _validate_selected_source_path(source_root, source_portfolio)
        for path in source_portfolio.rglob("*"):
            _validate_selected_source_path(source_root, path)
        portfolio_relatives = {
            path.relative_to(source_root).as_posix()
            for path in validate_portfolio_memory(source_portfolio, vc_slug)
        }
        if not portfolio_relatives <= source_hashes.keys():
            raise InputBoundaryError("portfolio memory files are not manifest-bound")
    elif any(relative.startswith(portfolio_prefix) for relative in source_hashes):
        raise InputBoundaryError("source manifest references absent portfolio memory")
    selected = required | {
        relative for relative in source_hashes if relative.startswith(wiki_prefix)
    } | precedent_relatives | portfolio_relatives
    if not manifest_required <= source_hashes.keys() or not any(
        relative.startswith(wiki_prefix) for relative in source_hashes
    ):
        raise InputBoundaryError("source manifest lacks required inference assets")

    verified_selected: list[tuple[str, Path, str | None]] = []
    for relative in sorted(selected):
        source_path = source_root / relative
        _validate_selected_source_path(source_root, source_path)
        expected = source_hashes[relative]
        if not source_path.is_file():
            raise InputBoundaryError(f"source hash mismatch: {relative}")
        verified_selected.append((relative, source_path, expected))

    root_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        source_root_fd = os.open(source_root, root_flags)
    except OSError as exc:
        raise InputBoundaryError("source root descriptor open failed") from exc
    try:
        root_is_directory = stat.S_ISDIR(os.fstat(source_root_fd).st_mode)
    except OSError as exc:
        os.close(source_root_fd)
        raise InputBoundaryError("source root descriptor stat failed") from exc
    if not root_is_directory:
        os.close(source_root_fd)
        raise InputBoundaryError("source root descriptor is not a directory")
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination_root.name}.staging-",
                dir=destination_root.parent,
            )
        )
        rows: list[dict[str, str]] = []
        for relative, source_path, expected in verified_selected:
            _before_copy_selected(relative, source_path)
            actual = _copy_selected_file(
                source_root_fd,
                relative,
                staging / relative,
                expected,
            )
            bound_digest = expected if expected is not None else actual
            rows.append({"destination": relative, "sha256": bound_digest})

        target_manifest = (
            staging / f"data/investors/{vc_slug}/manifests/{episode_slug}.json"
        )
        target_manifest.parent.mkdir(parents=True, exist_ok=True)
        target_manifest.write_text(
            json.dumps(
                {
                    "schema": "langgraph-vc-clone-source-manifest-v1",
                    "vc_slug": vc_slug,
                    "episode_slug": episode_slug,
                    "files": rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        verify_package(staging, vc_slug, episode_slug)
        staging.rename(destination_root)
    except Exception:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        os.close(source_root_fd)
    return destination_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--vc", required=True)
    parser.add_argument("--episode", required=True)
    args = parser.parse_args()
    imported = import_assets(args.source, args.destination, args.vc, args.episode)
    print(imported)


if __name__ == "__main__":
    main()
