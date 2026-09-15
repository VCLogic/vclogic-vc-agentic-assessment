"""Immutable, auditable artifact storage for founder rehearsal sessions."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any


_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SESSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_KEYS = {
    "authorization",
    "api_key",
    "api-key",
    "openai_api_key",
    "openrouter_api_key",
}


class PendingAnswerConflict(ValueError):
    """A different answer is already accepted for the active question."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _contains_secret(value: object) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).casefold() in _SECRET_KEYS:
                return True
            if _contains_secret(nested):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_secret(item) for item in value)
    return isinstance(value, str) and value.casefold().startswith("bearer ")


class RehearsalArtifactStore:
    """Persist accepted state separately from raw calls and resumable findings."""

    def __init__(self, session_root: Path) -> None:
        self.session_root = Path(session_root).resolve()
        self.pitch_path = self.session_root / "pitch.txt"
        self.manifest_path = self.session_root / "manifest.json"
        self.index_path = self.session_root / "artifact-index.json"

    @classmethod
    def create(
        cls,
        output_root: Path,
        vc_slug: str,
        session_id: str,
        pitch: str,
    ) -> "RehearsalArtifactStore":
        if not _SLUG.fullmatch(vc_slug):
            raise ValueError("vc_slug is invalid")
        if not _SESSION.fullmatch(session_id):
            raise ValueError("session_id is invalid")
        if not pitch.strip():
            raise ValueError("pitch must not be empty")
        session_root = Path(output_root).resolve() / vc_slug / session_id
        session_root.mkdir(parents=True, exist_ok=False)
        store = cls(session_root)
        store.write_pitch_once(pitch)
        store._atomic_json(
            store.manifest_path,
            {
                "schema_version": "founder-rehearsal-session-v1",
                "session_id": session_id,
                "vc_slug": vc_slug,
                "created_at": _utc_now(),
                "pitch_sha256": _sha256_path(store.pitch_path),
            },
        )
        store._atomic_json(store.index_path, {"artifacts": {}})
        return store

    @classmethod
    def open(cls, session_root: Path) -> "RehearsalArtifactStore":
        store = cls(session_root)
        if not store.manifest_path.is_file() or not store.pitch_path.is_file():
            raise ValueError("session is missing its manifest or immutable pitch")
        return store

    def _safe_path(self, relative: str) -> Path:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or not relative or "\\" in relative:
            raise ValueError("artifact path must be safe and relative")
        candidate = (self.session_root / pure).resolve()
        if not candidate.is_relative_to(self.session_root):
            raise ValueError("artifact path must remain relative to the session")
        return candidate

    @staticmethod
    def _atomic_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_canonical_bytes(value))
                stream.flush()
                os.fsync(stream.fileno())
            Path(temporary).replace(path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def write_pitch_once(self, pitch: str) -> None:
        self.pitch_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.pitch_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(pitch)
            stream.flush()
            os.fsync(stream.fileno())

    def _record_digest(self, path: Path) -> None:
        relative = path.relative_to(self.session_root).as_posix()
        if relative in {"manifest.json", "artifact-index.json", "pitch.txt"}:
            return
        payload = (
            json.loads(self.index_path.read_text(encoding="utf-8"))
            if self.index_path.is_file()
            else {"artifacts": {}}
        )
        artifacts = payload.setdefault("artifacts", {})
        artifacts[relative] = _sha256_path(path)
        self._atomic_json(self.index_path, payload)

    def append_answer(self, answer_id: str, payload: dict[str, object]) -> Path:
        if payload.get("answer_id") != answer_id:
            raise ValueError("answer payload identity does not match answer_id")
        try:
            number = int(answer_id.removeprefix("A-"))
        except ValueError as exc:
            raise ValueError("answer_id is invalid") from exc
        path = self._safe_path(f"turns/turn-{number:02d}/answer.json")
        if path.exists():
            raise FileExistsError(path)
        self._atomic_json(path, payload)
        self._record_digest(path)
        return path

    @staticmethod
    def answer_digest(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()

    def accept_answer_submission(
        self, question_id: str, text: str
    ) -> dict[str, Any]:
        """Persist the exact founder text before asynchronous processing begins."""
        if not question_id or not text.strip():
            raise ValueError("question_id and answer text are required")
        relative = "pending-answer-submission.json"
        path = self._safe_path(relative)
        digest = self.answer_digest(text)
        if path.is_file():
            current = self.read_json(relative)
            if current.get("status") != "processed":
                if (
                    current.get("question_id") == question_id
                    and current.get("text_verbatim") == text
                    and current.get("submission_sha256") == digest
                ):
                    return current
                raise PendingAnswerConflict(
                    "a different answer is already pending for this question"
                )
        payload: dict[str, Any] = {
            "question_id": question_id,
            "text_verbatim": text,
            "submission_sha256": digest,
            "status": "accepted_for_processing",
            "accepted_at": _utc_now(),
        }
        self.write_accepted(relative, payload)
        return payload

    def mark_answer_submission_processed(self, answer_id: str) -> dict[str, Any]:
        relative = "pending-answer-submission.json"
        current = self.read_json(relative)
        payload = {
            **current,
            "status": "processed",
            "answer_id": answer_id,
            "processed_at": _utc_now(),
        }
        self.write_accepted(relative, payload)
        return payload

    def verify_answer_submission(
        self,
        pending: dict[str, Any],
        accepted: dict[str, Any],
    ) -> bool:
        text = accepted.get("text_verbatim")
        if not isinstance(text, str):
            return False
        digest = self.answer_digest(text)
        return (
            accepted.get("question_id") == pending.get("question_id")
            and text == pending.get("text_verbatim")
            and digest == pending.get("submission_sha256")
            and accepted.get("submission_sha256") == digest
        )

    def write_call(
        self,
        phase: str,
        sequence: int,
        request: dict[str, object],
        result: dict[str, object],
    ) -> Path:
        if _contains_secret(request) or _contains_secret(result):
            raise ValueError("secret material must not be stored in call artifacts")
        path = self._safe_path(f"calls/{phase}/call-{sequence:03d}.json")
        self._atomic_json(path, {"phase": phase, "request": request, "result": result})
        self._record_digest(path)
        return path

    def write_failed_call(
        self,
        phase: str,
        sequence: int,
        result: dict[str, object],
        finding: str,
    ) -> Path:
        if _contains_secret(result):
            raise ValueError("secret material must not be stored in findings")
        path = self._safe_path(f"findings/{phase}-{sequence:03d}.json")
        self._atomic_json(
            path,
            {
                "phase": phase,
                "sequence": sequence,
                "finding": finding,
                "result": result,
                "recorded_at": _utc_now(),
            },
        )
        self._record_digest(path)
        return path

    def write_accepted(self, relative: str, payload: dict[str, object]) -> Path:
        if _contains_secret(payload):
            raise ValueError("secret material must not be stored in accepted artifacts")
        path = self._safe_path(relative)
        self._atomic_json(path, payload)
        self._record_digest(path)
        return path

    def read_json(self, relative: str) -> dict[str, Any]:
        value = json.loads(self._safe_path(relative).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"artifact is not a JSON object: {relative}")
        return value

    def verify(self) -> dict[str, object]:
        manifest = self.read_json("manifest.json")
        if manifest.get("schema_version") != "founder-rehearsal-session-v1":
            raise ValueError("session manifest schema is invalid")
        if _sha256_path(self.pitch_path) != manifest.get("pitch_sha256"):
            raise ValueError("immutable pitch digest does not match the manifest")
        index = self.read_json("artifact-index.json")
        artifacts = index.get("artifacts", {})
        if not isinstance(artifacts, dict):
            raise ValueError("artifact index is invalid")
        for relative, digest in artifacts.items():
            path = self._safe_path(str(relative))
            if not path.is_file() or _sha256_path(path) != digest:
                raise ValueError(f"artifact digest mismatch: {relative}")
        return {
            "session_id": manifest["session_id"],
            "vc_slug": manifest["vc_slug"],
            "pitch_sha256": manifest["pitch_sha256"],
            "artifact_count": len(artifacts),
        }
