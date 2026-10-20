"""Structured generation through a logged-in Codex CLI session."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import tempfile
from threading import Event
from time import perf_counter
from typing import Any

from jsonschema import ValidationError, validate

from .base import GenerationRequest, GenerationResult, Usage


_SECRET = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{16,}|(?:api[_-]?key|authorization|token)\s*[:=]\s*\S+)"
)
CODEX_UNAVAILABLE = Event()
_TERMINAL_FAILURE = re.compile(
    r"(?i)(?:usage limit|usage cap|not logged in|authentication (?:failed|required)|unauthorized)"
)
_OUTPUT_CAP_FAILURE = re.compile(r"(?i)max_output_tokens")


class CodexProvider:
    """Run schema-bounded prompts through ``codex exec`` and ChatGPT login."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.client = self

    def close(self) -> None:
        pass

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if CODEX_UNAVAILABLE.is_set():
            raise RuntimeError("codex exec is unavailable for this batch")
        started = perf_counter()
        prompt = (
            f"{request.prompt}\n\n"
            "Do not use tools. Return only the structured answer required by the "
            "provided JSON schema."
        )

        with tempfile.TemporaryDirectory(prefix="assessment-codex-") as directory:
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
                self.model,
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
            fallback_reason = None
            for attempt in range(2):
                attempt_args = list(args)
                attempt_prompt = prompt
                if attempt:
                    output_path.unlink(missing_ok=True)
                    schema_index = attempt_args.index("--output-schema")
                    del attempt_args[schema_index : schema_index + 2]
                    attempt_prompt = (
                        f"{prompt}\n\n"
                        "Schema-in-prompt fallback: return one JSON object that validates "
                        "against this exact JSON schema. Emit no markdown or commentary.\n"
                        f"{json.dumps(request.schema, ensure_ascii=True)}"
                    )
                try:
                    completed = subprocess.run(
                        attempt_args,
                        input=attempt_prompt,
                        cwd=str(workdir),
                        timeout=900,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    detail = _safe_diagnostic(str(exc))
                    raise RuntimeError(
                        f"codex exec could not complete: {detail}"
                    ) from exc

                if completed.returncode == 0:
                    break
                diagnostic = "\n".join(
                    part for part in (completed.stderr, completed.stdout) if part
                )
                if _TERMINAL_FAILURE.search(diagnostic):
                    CODEX_UNAVAILABLE.set()
                if attempt == 0 and _OUTPUT_CAP_FAILURE.search(diagnostic):
                    fallback_reason = "max_output_tokens"
                    continue
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
            try:
                validate(instance=parsed, schema=request.schema)
            except ValidationError as exc:
                raise RuntimeError(
                    "codex exec output did not match requested schema: "
                    f"{_safe_diagnostic(exc.message)}"
                ) from exc
            usage = _usage_from_jsonl(completed.stdout)

        return GenerationResult(
            parsed=parsed,
            content=content,
            usage=usage,
            elapsed_seconds=perf_counter() - started,
            raw_metadata={
                "provider": "codex",
                "backend": "codex-cli",
                "model": self.model,
                "cost_unknown": True,
                "structured_output_mode": (
                    "schema-in-prompt-fallback" if fallback_reason else "cli-schema"
                ),
                "fallback_reason": fallback_reason,
                "requested_max_output_tokens": request.max_output_tokens,
                "output_budget_enforced": False,
                "failed_attempt_usage_unavailable": fallback_reason is not None,
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
    return (cleaned or "no diagnostic output")[-600:]
