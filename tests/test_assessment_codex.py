import json
import subprocess
from pathlib import Path

import pytest

from vc_clone_graph import cli
from vc_clone_graph.cli import _generation_provider
from vc_clone_graph.config import RunConfig
from vc_clone_graph.providers.base import GenerationRequest
from vc_clone_graph.providers.codex import CodexProvider


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "run": {
                "vc_slug": "jenny-fielding-everywhere-ventures",
                "episode_slug": "case-one",
                "input_root": "inputs",
                "output_root": "outputs/codex",
                "checkpoint_path": "outputs/codex/checkpoints.sqlite",
            },
            "provider": {"kind": "codex", "model": "openai/gpt-5.6-sol"},
            "phase1": {
                "min_iterations": 1,
                "max_iterations": 2,
                "reasoning_effort": "high",
            },
            "phase2": {"min_iterations": 1, "max_iterations": 2},
            "retrieval": {"top_k": 5, "max_exact_reads": 16},
        }
    )


def test_codex_config_allows_reasoning_and_factory_never_builds_openrouter(
    monkeypatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("OpenRouter must not be constructed for Codex")

    monkeypatch.setattr(cli, "OpenRouterProvider", forbidden)
    provider = _generation_provider(_config())

    assert isinstance(provider, CodexProvider)
    assert provider.model == "gpt-5.6-sol"


def test_codex_generation_uses_schema_stdin_reasoning_and_metadata(monkeypatch) -> None:
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        schema_path = Path(args[args.index("--output-schema") + 1])
        assert json.loads(schema_path.read_text(encoding="utf-8")) == SCHEMA
        Path(args[args.index("--output-last-message") + 1]).write_text(
            '{"answer":"yes"}', encoding="utf-8"
        )
        return subprocess.CompletedProcess(
            args,
            0,
            stdout='{"usage":{"input_tokens":12,"output_tokens":4}}\n',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = CodexProvider("gpt-5.6-sol").generate(
        GenerationRequest("assessment", "Assess", SCHEMA, reasoning_effort="high")
    )

    assert result.parsed == {"answer": "yes"}
    assert result.usage.input_tokens == 12
    assert result.raw_metadata == {
        "provider": "codex",
        "backend": "codex-cli",
        "model": "gpt-5.6-sol",
        "cost_unknown": True,
        "structured_output_mode": "cli-schema",
        "fallback_reason": None,
        "requested_max_output_tokens": None,
        "output_budget_enforced": False,
        "failed_attempt_usage_unavailable": False,
    }
    assert seen["args"][:2] == ["codex", "exec"]
    assert seen["args"][seen["args"].index("-c") + 1] == (
        'model_reasoning_effort="high"'
    )
    assert seen["args"][-1] == "-"
    assert seen["kwargs"]["input"].startswith("Assess")


def test_codex_retries_output_cap_once_with_schema_in_prompt(monkeypatch) -> None:
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            return subprocess.CompletedProcess(
                args,
                1,
                stdout='{"type":"error","message":"Incomplete response returned, reason: max_output_tokens"}\n',
                stderr="",
            )
        assert "--output-schema" not in args
        assert json.dumps(SCHEMA, ensure_ascii=True) in kwargs["input"]
        Path(args[args.index("--output-last-message") + 1]).write_text(
            '{"answer":"fallback"}', encoding="utf-8"
        )
        return subprocess.CompletedProcess(
            args,
            0,
            stdout='{"usage":{"input_tokens":20,"output_tokens":8}}\n',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = CodexProvider("gpt-5.6-sol").generate(
        GenerationRequest(
            "assessment", "Assess", SCHEMA, max_output_tokens=16384,
            reasoning_effort="high",
        )
    )

    assert len(calls) == 2
    assert "--output-schema" in calls[0][0]
    assert result.parsed == {"answer": "fallback"}
    assert result.usage.output_tokens == 8
    assert result.raw_metadata["structured_output_mode"] == "schema-in-prompt-fallback"
    assert result.raw_metadata["fallback_reason"] == "max_output_tokens"
    assert result.raw_metadata["requested_max_output_tokens"] == 16384
    assert result.raw_metadata["output_budget_enforced"] is False
    assert result.raw_metadata["failed_attempt_usage_unavailable"] is True


def test_codex_rejects_json_that_does_not_match_schema(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        Path(args[args.index("--output-last-message") + 1]).write_text(
            '{"wrong":"value"}', encoding="utf-8"
        )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="did not match requested schema"):
        CodexProvider("gpt-5.6-sol").generate(
            GenerationRequest("assessment", "Assess", SCHEMA)
        )


def test_codex_rejects_invalid_schema_in_prompt_fallback(monkeypatch) -> None:
    calls = 0

    def fake_run(args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(
                args, 1, stdout="max_output_tokens", stderr=""
            )
        Path(args[args.index("--output-last-message") + 1]).write_text(
            '{"wrong":"value"}', encoding="utf-8"
        )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="did not match requested schema"):
        CodexProvider("gpt-5.6-sol").generate(
            GenerationRequest("assessment", "Assess", SCHEMA)
        )
    assert calls == 2


def test_codex_does_not_retry_other_failures(monkeypatch) -> None:
    calls = 0

    def fake_run(args, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            args, 1, stdout='{"type":"error","message":"server error"}', stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="server error"):
        CodexProvider("gpt-5.6-sol").generate(
            GenerationRequest("assessment", "Assess", SCHEMA)
        )
    assert calls == 1
