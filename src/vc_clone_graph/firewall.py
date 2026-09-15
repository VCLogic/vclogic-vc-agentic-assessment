"""Fail-closed verification of investor-scoped inference assets."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import tomllib

from .precedents import PrecedentCorpus, PrecedentEpisode
from .portfolio_memory import PortfolioEntity, PortfolioMemoryCorpus


_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SHA = re.compile(r"[0-9a-f]{64}")
_PRECEDENT_INDEX = "precedents-index.json"
_PORTFOLIO_FILES = {
    "company-index.json",
    "corpus-manifest.json",
    "disclosure-events.jsonl",
    "embedding-index.json",
    "extraction-audit.jsonl",
}
_FORBIDDEN = {
    "actual_decision.json",
    "complete-transcript.json",
    "episodes",
    "labels",
    "predictions",
    "reference_rationales",
    "reports",
}


class InputBoundaryError(ValueError):
    """Inference input violates the leakage or integrity boundary."""


@dataclass(frozen=True)
class VerifiedPackage:
    root: Path
    registry: Path
    pitch: Path
    audit: Path
    manifest: Path
    wiki: Path
    taxonomy: Path
    target_company_aliases: tuple[str, ...]
    precedents: Path | None = None
    precedent_manifest: Path | None = None
    portfolio_memory: Path | None = None
    portfolio_manifest: Path | None = None


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str, label: str) -> str:
    if type(value) is not str or _SLUG.fullmatch(value) is None:
        raise InputBoundaryError(f"invalid {label}")
    return value


def safe_slug(value: str, label: str = "slug") -> str:
    """Validate a generic package slug using the inference boundary grammar."""
    return _slug(value, label)


def _safe_relative(value: str) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value:
        raise InputBoundaryError("manifest path is unsafe")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise InputBoundaryError("manifest path is unsafe")
    return path


def safe_manifest_relative(value: str) -> PurePosixPath:
    """Return a validated package-relative manifest destination."""
    return _safe_relative(value)


def _target_company_aliases(
    value: object, vc_slug: str, episode_slug: str
) -> tuple[str, ...]:
    if type(value) is not list or not 1 <= len(value) <= 8:
        raise InputBoundaryError("target company aliases must contain one to eight values")
    aliases: list[str] = []
    seen: set[str] = set()
    for raw in value:
        if type(raw) is not str:
            raise InputBoundaryError("target company aliases must be strings")
        alias = raw.strip()
        if not alias or len(alias) > 120:
            raise InputBoundaryError("target company aliases must be 1 to 120 characters")
        if not alias.isprintable():
            raise InputBoundaryError("target company aliases must be printable")
        folded = alias.casefold()
        if folded in seen:
            raise InputBoundaryError("target company aliases must be unique")
        if folded in {vc_slug.casefold(), episode_slug.casefold()}:
            raise InputBoundaryError("target company alias cannot equal a package slug")
        seen.add(folded)
        aliases.append(alias)
    return tuple(aliases)


def _reject_forbidden_tree(root: Path) -> None:
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise InputBoundaryError(f"package tree contains symlink: {relative}")
        if any(part.lower() in _FORBIDDEN for part in relative.parts):
            raise InputBoundaryError(f"forbidden inference path: {relative}")


def _validate_package_path(package_root: Path, path: Path) -> None:
    """Reject symlink components and paths resolving outside the package."""
    try:
        relative = path.relative_to(package_root)
    except ValueError as exc:
        raise InputBoundaryError("package asset path escapes input root") from exc
    current = package_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise InputBoundaryError(f"package asset path contains symlink: {relative}")
    try:
        path.resolve(strict=True).relative_to(package_root)
    except (OSError, ValueError) as exc:
        raise InputBoundaryError("package asset path escapes input root") from exc


def validate_package_path(package_root: Path, path: Path) -> None:
    """Validate an existing package path's containment and components."""
    _validate_package_path(package_root, path)


def _expected_chunk_rows(
    records: list[PrecedentEpisode],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        for start in range(0, len(record.turns), 8):
            turns = record.turns[start : start + 8]
            rows.append(
                {
                    "chunk_id": "H-"
                    + sha256(f"{record.episode_slug}:{start}".encode()).hexdigest()[
                        :20
                    ],
                    "episode_slug": record.episode_slug,
                    "source_sha256": record.source_sha256,
                    "text": "\n".join(
                        f"{turn.speaker}: {turn.text}" for turn in turns
                    ),
                    "turn_end": start + len(turns) - 1,
                    "turn_start": start,
                }
            )
    return rows


def _validate_chunk_export(path: Path, records: list[PrecedentEpisode]) -> None:
    expected = _expected_chunk_rows(records)
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    expected_by_position = {
        (row["episode_slug"], row["turn_start"]): row for row in expected
    }
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise InputBoundaryError("precedent chunk export is unreadable") from exc
    fields = {
        "episode_slug",
        "chunk_id",
        "source_sha256",
        "turn_start",
        "turn_end",
        "text",
    }
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputBoundaryError(
                "precedent chunk export contains invalid JSON"
            ) from exc
        if (
            type(row) is not dict
            or set(row) != fields
            or type(row["episode_slug"]) is not str
            or type(row["chunk_id"]) is not str
            or type(row["source_sha256"]) is not str
            or type(row["turn_start"]) is not int
            or type(row["turn_end"]) is not int
            or type(row["text"]) is not str
        ):
            raise InputBoundaryError("precedent chunk export row is invalid")
        chunk_id = row["chunk_id"]
        if chunk_id in seen_ids:
            raise InputBoundaryError("precedent chunk IDs are not unique")
        seen_ids.add(chunk_id)
        canonical = expected_by_position.get(
            (row["episode_slug"], row["turn_start"])
        )
        if canonical is None or row != canonical:
            raise InputBoundaryError("precedent chunk export is inconsistent")
        rows.append(row)
    if rows != expected:
        raise InputBoundaryError("precedent chunk export is incomplete or out of order")


def validate_precedent_corpus(root: Path) -> tuple[Path, ...]:
    """Validate the closed corpus layout and all canonical source bindings."""
    corpus_root = Path(root)
    if corpus_root.is_symlink() or not corpus_root.is_dir():
        raise InputBoundaryError("precedent corpus root is invalid or a symlink")

    required_files = {"corpus-manifest.json", "chunks.jsonl"}
    allowed_root_files = required_files | {_PRECEDENT_INDEX}
    required_directories = {"records", "sources"}
    files: list[Path] = []
    directories: set[str] = set()
    for path in corpus_root.rglob("*"):
        if path.is_symlink():
            raise InputBoundaryError("precedent corpus cannot contain symlinks")
        relative = path.relative_to(corpus_root)
        if path.is_dir():
            if len(relative.parts) != 1 or relative.name not in required_directories:
                raise InputBoundaryError(f"unexpected precedent directory: {relative}")
            directories.add(relative.name)
            continue
        if not path.is_file():
            raise InputBoundaryError(f"unexpected precedent path: {relative}")
        if len(relative.parts) == 1:
            if relative.name not in allowed_root_files:
                raise InputBoundaryError(f"unexpected precedent file: {relative}")
        elif (
            len(relative.parts) != 2
            or relative.parts[0] not in required_directories
            or relative.suffix != ".json"
            or _SLUG.fullmatch(relative.stem) is None
        ):
            raise InputBoundaryError(f"unexpected precedent file: {relative}")
        files.append(path)

    root_file_names = {path.name for path in files if path.parent == corpus_root}
    if not required_files <= root_file_names or directories != required_directories:
        raise InputBoundaryError("precedent corpus is incomplete")

    try:
        PrecedentCorpus.build(corpus_root)
        index = corpus_root / _PRECEDENT_INDEX
        if index.is_file():
            PrecedentCorpus.load(index, corpus_root)
    except (OSError, TypeError, ValueError) as exc:
        raise InputBoundaryError(f"precedent corpus is invalid: {exc}") from exc

    manifest_path = corpus_root / "corpus-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputBoundaryError("precedent corpus manifest is invalid") from exc
    if set(manifest) != {"records", "schema"}:
        raise InputBoundaryError("precedent corpus manifest structure is invalid")
    expected_records: set[str] = set()
    expected_sources: set[str] = set()
    records: list[PrecedentEpisode] = []
    for entry in manifest["records"]:
        if type(entry) is not dict or set(entry) != {
            "decision_status",
            "episode_slug",
            "record_path",
            "record_sha256",
            "source_path",
            "source_sha256",
        }:
            raise InputBoundaryError("precedent corpus manifest record is invalid")
        slug = entry["episode_slug"]
        if type(slug) is not str or _SLUG.fullmatch(slug) is None:
            raise InputBoundaryError("precedent corpus manifest slug is invalid")
        expected_record = f"records/{slug}.json"
        expected_source = f"sources/{slug}.json"
        if (
            entry["record_path"] != expected_record
            or entry["source_path"] != expected_source
        ):
            raise InputBoundaryError("precedent corpus manifest path is invalid")
        expected_records.add(expected_record)
        expected_sources.add(expected_source)
        try:
            records.append(
                PrecedentEpisode.model_validate_json(
                    (corpus_root / expected_record).read_bytes()
                )
            )
        except (OSError, ValueError) as exc:
            raise InputBoundaryError("precedent corpus record is invalid") from exc
    actual_records = {
        path.relative_to(corpus_root).as_posix()
        for path in files
        if path.parent == corpus_root / "records"
    }
    actual_sources = {
        path.relative_to(corpus_root).as_posix()
        for path in files
        if path.parent == corpus_root / "sources"
    }
    if actual_records != expected_records or actual_sources != expected_sources:
        raise InputBoundaryError("precedent corpus files do not match its manifest")
    _validate_chunk_export(corpus_root / "chunks.jsonl", records)
    return tuple(sorted(files))


def validate_portfolio_memory(root: Path, vc_slug: str) -> tuple[Path, ...]:
    """Validate a closed, source-bound temporal portfolio-memory package."""
    corpus_root = Path(root)
    if corpus_root.is_symlink() or not corpus_root.is_dir():
        raise InputBoundaryError("portfolio memory root is invalid or a symlink")
    files: list[Path] = []
    for path in corpus_root.rglob("*"):
        if path.is_symlink():
            raise InputBoundaryError("portfolio memory cannot contain symlinks")
        if path.is_dir() or path.parent != corpus_root or path.name not in _PORTFOLIO_FILES:
            raise InputBoundaryError(f"unexpected portfolio memory path: {path.relative_to(corpus_root)}")
        files.append(path)
    if {path.name for path in files} != _PORTFOLIO_FILES:
        raise InputBoundaryError("portfolio memory is incomplete")
    events_path = corpus_root / "disclosure-events.jsonl"
    try:
        corpus = PortfolioMemoryCorpus.load_events(vc_slug, events_path)
        entities_raw = json.loads((corpus_root / "company-index.json").read_text(encoding="utf-8"))
        if type(entities_raw) is not list:
            raise ValueError("company index is not a list")
        for row in entities_raw:
            PortfolioEntity.model_validate(row)
        index = json.loads((corpus_root / "embedding-index.json").read_text(encoding="utf-8"))
        if index.get("schema") != "portfolio-memory-index-v1" or index.get("vc_slug") != vc_slug:
            raise ValueError("embedding index identity is invalid")
        if index.get("corpus_sha256") != sha256_file(events_path):
            raise ValueError("embedding index corpus hash mismatch")
        embeddings = index.get("event_embeddings")
        if type(embeddings) is not dict or set(embeddings) != {
            row.disclosure_id for row in corpus.disclosures
        }:
            raise ValueError("embedding coverage is invalid")
        for line in (corpus_root / "extraction-audit.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip() and type(json.loads(line)) is not dict:
                raise ValueError("audit row is invalid")
        manifest = json.loads((corpus_root / "corpus-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise InputBoundaryError(f"portfolio memory is invalid: {exc}") from exc
    required_manifest = {
        "schema", "vc_slug", "accepted_count", "named_count", "anonymous_count",
        "rejected_count", "candidate_unextracted_count", "files", "embedding",
    }
    if set(manifest) != required_manifest or manifest.get("schema") != "portfolio-memory-corpus-manifest-v1":
        raise InputBoundaryError("portfolio memory manifest structure is invalid")
    if manifest.get("vc_slug") != vc_slug or manifest.get("accepted_count") != len(corpus.disclosures):
        raise InputBoundaryError("portfolio memory manifest identity or count is invalid")
    expected_hashes = {
        name: sha256_file(corpus_root / name)
        for name in _PORTFOLIO_FILES - {"corpus-manifest.json"}
    }
    if manifest.get("files") != expected_hashes:
        raise InputBoundaryError("portfolio memory files do not match its manifest")
    return tuple(sorted(files))


def resolve_manifest_path(root: Path, vc_slug: str, episode_slug: str) -> Path:
    """Prefer an episode-scoped manifest while supporting legacy packages."""
    investor_root = Path(root) / "data" / "investors" / vc_slug
    scoped = investor_root / "manifests" / f"{episode_slug}.json"
    if scoped.is_file():
        return scoped
    return investor_root / "source-manifest.json"


def verify_package(
    root: Path,
    vc_slug: str,
    episode_slug: str,
    *,
    taxonomy_path: str = "taxonomy/codebook_v_final.json",
) -> VerifiedPackage:
    """Verify paths, audit status, manifest membership, and every file hash."""
    requested_root = Path(root)
    if requested_root.is_symlink():
        raise InputBoundaryError("input root cannot be a symlink")
    package_root = requested_root.resolve()
    vc = _slug(vc_slug, "vc_slug")
    episode = _slug(episode_slug, "episode_slug")
    if not package_root.is_dir():
        raise InputBoundaryError("input root does not exist")
    _reject_forbidden_tree(package_root)

    registry = package_root / "investors" / f"{vc}.toml"
    pitch = package_root / f"data/investors/{vc}/pitches/{episode}.txt"
    audit = package_root / f"data/investors/{vc}/audits/{episode}.json"
    manifest = resolve_manifest_path(package_root, vc, episode)
    taxonomy_relative = _safe_relative(taxonomy_path)
    if taxonomy_relative.parts[0] != "taxonomy" or taxonomy_relative.suffix != ".json":
        raise InputBoundaryError("taxonomy path must select a JSON file under taxonomy/")
    taxonomy = package_root / taxonomy_relative
    required = (registry, pitch, audit, manifest, taxonomy)
    if any(not path.is_file() for path in required):
        raise InputBoundaryError("required inference file is missing")
    for path in required:
        _validate_package_path(package_root, path)

    try:
        registry_data = tomllib.loads(registry.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise InputBoundaryError("investor registry is invalid") from exc
    expected_wiki = f"wiki/{vc}"
    if registry_data.get("vc_slug") != vc or registry_data.get("wiki_path") != expected_wiki:
        raise InputBoundaryError("investor registry boundary is invalid")
    wiki = package_root / expected_wiki
    if not wiki.is_dir() or not any(path.is_file() for path in wiki.rglob("*")):
        raise InputBoundaryError("investor wiki is missing")
    _validate_package_path(package_root, wiki)

    try:
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        audit_data = json.loads(audit.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputBoundaryError("manifest or audit is invalid") from exc
    if manifest_data.get("vc_slug") != vc or manifest_data.get("episode_slug") != episode:
        raise InputBoundaryError("manifest identity does not match run")
    rows = manifest_data.get("files")
    if type(rows) is not list or not rows:
        raise InputBoundaryError("manifest files are invalid")
    destinations: set[str] = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"destination", "sha256"}:
            raise InputBoundaryError("manifest row is invalid")
        relative = _safe_relative(row["destination"])
        digest = row["sha256"]
        if type(digest) is not str or _SHA.fullmatch(digest) is None:
            raise InputBoundaryError("manifest hash is invalid")
        destination = relative.as_posix()
        if destination in destinations:
            raise InputBoundaryError("manifest paths are not unique")
        destinations.add(destination)
        file_path = package_root / relative
        if not file_path.is_file():
            raise InputBoundaryError(f"manifest hash mismatch: {destination}")
        _validate_package_path(package_root, file_path)
        if sha256_file(file_path) != digest:
            raise InputBoundaryError(f"manifest hash mismatch: {destination}")

    required_destinations = {
        registry.relative_to(package_root).as_posix(),
        pitch.relative_to(package_root).as_posix(),
        audit.relative_to(package_root).as_posix(),
        taxonomy.relative_to(package_root).as_posix(),
    }
    wiki_destinations = {
        path.relative_to(package_root).as_posix()
        for path in wiki.rglob("*")
        if path.is_file()
    }
    if not required_destinations <= destinations or not wiki_destinations <= destinations:
        raise InputBoundaryError("required files are not manifest-bound")

    if audit_data.get("status") not in {"approved", "audited"}:
        raise InputBoundaryError("pitch audit is not approved")
    if audit_data.get("vc_slug") != vc or audit_data.get("episode_slug") != episode:
        raise InputBoundaryError("pitch audit identity does not match run")
    if audit_data.get("pitch_sha256") != sha256_file(pitch):
        raise InputBoundaryError("pitch audit hash does not match")
    checklist = audit_data.get("leakage_checklist")
    if type(checklist) is not dict or any(value is not False for value in checklist.values()):
        raise InputBoundaryError("pitch audit leakage checklist is not clean")

    aliases = _target_company_aliases(
        audit_data.get("target_company_aliases"), vc, episode
    )

    precedents = package_root / f"data/investors/{vc}/precedents"
    precedent_manifest: Path | None = None
    if precedents.exists() or precedents.is_symlink():
        _validate_package_path(package_root, precedents)
        precedent_files = validate_precedent_corpus(precedents)
        precedent_destinations = {
            path.relative_to(package_root).as_posix() for path in precedent_files
        }
        if not precedent_destinations <= destinations:
            raise InputBoundaryError("precedent corpus files are not manifest-bound")
        precedent_manifest = precedents / "corpus-manifest.json"
    else:
        precedents = None

    portfolio_memory = package_root / f"data/investors/{vc}/portfolio-memory"
    portfolio_manifest: Path | None = None
    if portfolio_memory.exists() or portfolio_memory.is_symlink():
        _validate_package_path(package_root, portfolio_memory)
        portfolio_files = validate_portfolio_memory(portfolio_memory, vc)
        portfolio_destinations = {
            path.relative_to(package_root).as_posix() for path in portfolio_files
        }
        if not portfolio_destinations <= destinations:
            raise InputBoundaryError("portfolio memory files are not manifest-bound")
        portfolio_manifest = portfolio_memory / "corpus-manifest.json"
    else:
        portfolio_memory = None

    return VerifiedPackage(
        package_root,
        registry,
        pitch,
        audit,
        manifest,
        wiki,
        taxonomy,
        aliases,
        precedents,
        precedent_manifest,
        portfolio_memory,
        portfolio_manifest,
    )
