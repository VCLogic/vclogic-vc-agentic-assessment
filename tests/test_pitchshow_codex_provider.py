import json
import subprocess
from pathlib import Path

import pytest

from scripts.pitchshow_codex_provider import CODEX_UNAVAILABLE, CodexProvider
from vc_clone_graph.providers.base import GenerationRequest


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def test_generate_uses_isolated_codex_exec_and_parses_structured_result(monkeypatch) -> None:
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        schema_path = Path(args[args.index("--output-schema") + 1])
        output_path = Path(args[args.index("--output-last-message") + 1])
        assert json.loads(schema_path.read_text()) == SCHEMA
        output_path.write_text('{"answer":"yes"}')
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=(
                '{"type":"turn.completed","usage":{"input_tokens":12,'
                '"cached_input_tokens":3,"output_tokens":4}}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CodexProvider()
    result = provider.generate(
        GenerationRequest(
            "rationale_extract", "Read this deck", SCHEMA, reasoning_effort="high"
        )
    )

    assert result.parsed == {"answer": "yes"}
    assert result.content == '{"answer":"yes"}'
    assert result.usage.input_tokens == 12
    assert result.usage.cached_input_tokens == 3
    assert result.usage.output_tokens == 4
    assert result.raw_metadata == {
        "backend": "codex-cli",
        "model": "gpt-5.6-luna",
        "cost_unknown": True,
    }
    assert seen["args"][:2] == ["codex", "exec"]
    assert "--ignore-user-config" in seen["args"]
    assert "--ephemeral" in seen["args"]
    assert "--skip-git-repo-check" in seen["args"]
    assert seen["args"][seen["args"].index("--sandbox") + 1] == "read-only"
    assert seen["args"][seen["args"].index("--model") + 1] == "gpt-5.6-luna"
    assert seen["args"][seen["args"].index("-c") + 1] == (
        'model_reasoning_effort="high"'
    )
    assert "--json" in seen["args"]
    assert seen["kwargs"]["timeout"] == 900
    assert seen["kwargs"]["capture_output"] is True
    assert seen["kwargs"]["text"] is True
    assert seen["kwargs"]["cwd"] != str(Path.cwd())
    assert seen["kwargs"]["input"].startswith("Read this deck")
    assert "Do not use tools" in seen["kwargs"]["input"]


def test_generate_selects_review_model_for_other_phases(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        Path(args[args.index("--output-last-message") + 1]).write_text(
            '{"answer":"reviewed"}'
        )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CodexProvider(extraction_model="extract", review_model="review")
    result = provider.generate(GenerationRequest("evaluation_review", "Review", SCHEMA))

    assert result.raw_metadata["model"] == "review"


def test_generate_reports_bounded_redacted_failure(monkeypatch) -> None:
    secret = "sk-" + "x" * 80

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(
            args, 7, stdout="", stderr=("failure " + secret + " noisy" * 500)
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as exc_info:
        CodexProvider().generate(GenerationRequest("review", "Review", SCHEMA))

    message = str(exc_info.value)
    assert "exit code 7" in message
    assert secret not in message
    assert len(message) < 800


def test_auth_or_usage_exhaustion_stops_later_calls(monkeypatch) -> None:
    CODEX_UNAVAILABLE.clear()
    calls = 0

    def fake_run(args, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            args,
            1,
            stdout="You have reached your usage limit",
            stderr="unrelated warning",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = CodexProvider()
    with pytest.raises(RuntimeError, match="usage limit"):
        provider.generate(GenerationRequest("review", "Review", SCHEMA))
    with pytest.raises(RuntimeError, match="unavailable for this batch"):
        provider.generate(GenerationRequest("review", "Review", SCHEMA))
    assert calls == 1
    CODEX_UNAVAILABLE.clear()


def test_generate_rejects_non_object_output(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        Path(args[args.index("--output-last-message") + 1]).write_text("[]")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="JSON object"):
        CodexProvider().generate(GenerationRequest("review", "Review", SCHEMA))


def test_client_close_compatibility() -> None:
    provider = CodexProvider()
    assert provider.client is provider
    assert provider.close() is None
