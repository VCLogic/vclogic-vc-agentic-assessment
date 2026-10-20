#!/usr/bin/env python3
"""Structured generation adapter for a logged-in Codex CLI session."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
import re
import subprocess
import tempfile
from threading import Event
from typing import Any

from vc_clone_graph.providers.base import GenerationRequest, GenerationResult, Usage


_SECRET = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{16,}|(?:api[_-]?key|authorization|token)\s*[:=]\s*\S+)"
)
CODEX_UNAVAILABLE = Event()
_TERMINAL_FAILURE = re.compile(
    r"(?i)(?:usage limit|usage cap|not logged in|authentication (?:failed|required)|unauthorized)"
)


class CodexProvider:
    """Run structured prompts through ``codex exec`` and ChatGPT login."""

    def __init__(
        self,
        extraction_model: str = "gpt-5.6-luna",
        review_model: str = "gpt-5.6-sol",
    ) -> None:
        self.extraction_model = extraction_model
        self.review_model = review_model
        # Existing callers close ``provider.client`` during cleanup.
        self.client = self

    def close(self) -> None:
        pass

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if CODEX_UNAVAILABLE.is_set():
            raise RuntimeError("codex exec is unavailable for this batch")
        started = perf_counter()
        model = (
            self.extraction_model
            if request.phase == "rationale_extract"
            else self.review_model
        )
        prompt = (
            f"{request.prompt}\n\n"
            "Do not use tools. Return only the structured answer required by the "
            "provided JSON schema."
        )

        with tempfile.TemporaryDirectory(prefix="pitchshow-codex-") as directory:
            workdir = Path(directory)
            schema_path = workdir / "schema.json"
            output_path = workdir / "last-message.json"
            schema_path.write_text(
                json.dumps(request.schema, ensure_ascii=True), encoding="utf-8"
            )
            args = [
                "codex",
                "exec",
                "--ignore-user-config",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--model",
                model,
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "--json",
                "-",
            ]
            if request.reasoning_effort is not None:
                args[2:2] = [
                    "-c",
                    f'model_reasoning_effort="{request.reasoning_effort}"',
                ]
            try:
                completed = subprocess.run(
                    args,
                    input=prompt,
                    cwd=str(workdir),
                    timeout=900,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                detail = _safe_diagnostic(str(exc))
                raise RuntimeError(f"codex exec could not complete: {detail}") from exc

            if completed.returncode != 0:
                diagnostic = "\n".join(
                    part for part in (completed.stderr, completed.stdout) if part
                )
                if _TERMINAL_FAILURE.search(diagnostic):
                    CODEX_UNAVAILABLE.set()
                detail = _safe_diagnostic(diagnostic)
                raise RuntimeError(
                    f"codex exec failed with exit code {completed.returncode}: {detail}"
                )
            if not output_path.is_file():
                raise RuntimeError("codex exec completed without a structured output file")

            content = output_path.read_text(encoding="utf-8")
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise RuntimeError("codex exec output was not valid JSON") from exc
            if type(parsed) is not dict:
                raise RuntimeError("codex exec output was not a JSON object")

            usage = _usage_from_jsonl(completed.stdout)

        return GenerationResult(
            parsed=parsed,
            content=content,
            usage=usage,
            elapsed_seconds=perf_counter() - started,
            raw_metadata={
                "backend": "codex-cli",
                "model": model,
                "cost_unknown": True,
            },
        )


def _usage_from_jsonl(stdout: str) -> Usage:
    latest: dict[str, Any] = {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and isinstance(event.get("usage"), dict):
            latest = event["usage"]
    return Usage(
        input_tokens=_nonnegative_int(latest.get("input_tokens")),
        cached_input_tokens=_nonnegative_int(latest.get("cached_input_tokens")),
        output_tokens=_nonnegative_int(latest.get("output_tokens")),
    )


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _safe_diagnostic(value: str) -> str:
    cleaned = _SECRET.sub("[redacted]", value).replace("\n", " ").strip()
    return (cleaned or "no diagnostic output")[:600]
