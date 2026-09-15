import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vc_clone_graph.rehearsal_artifacts import RehearsalArtifactStore
from vc_clone_graph.rehearsal_cli import (
    _parser,
    command_start,
    locate_session,
    markdown_report,
)
from vc_clone_graph.rehearsal_config import load_rehearsal_config


ROOT = Path(__file__).resolve().parents[1]


def test_help_exposes_session_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vc_clone_graph.rehearsal_cli", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    for command in (
        "start",
        "answer",
        "finish",
        "retry",
        "report",
        "compare",
        "verify",
        "investors",
    ):
        assert command in result.stdout


def test_start_requires_exactly_one_pitch_source() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vc_clone_graph.rehearsal_cli",
            "start",
            "--config",
            "missing.toml",
            "--vc",
            "charles-hudson-precursor-ventures",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--pitch" in result.stderr and "--text" in result.stderr


def test_start_exposes_explicit_canonical_baseline_options() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "start",
                "--config",
                "config.toml",
                "--vc",
                "test-investor",
                "--text",
                "Pitch",
                "--build-canonical-baseline",
                "--canonical-baseline",
                "run",
            ]
        )

    args = _parser().parse_args(
        [
            "start",
            "--config",
            "config.toml",
            "--vc",
            "test-investor",
            "--text",
            "Pitch",
            "--build-canonical-baseline",
        ]
    )
    assert args.build_canonical_baseline is True
    assert args.canonical_baseline is None


def test_locate_session_requires_unique_session_id(tmp_path: Path) -> None:
    RehearsalArtifactStore.create(tmp_path, "investor-one", "same", "Pitch")
    RehearsalArtifactStore.create(tmp_path, "investor-two", "same", "Pitch")

    with pytest.raises(ValueError, match="ambiguous"):
        locate_session(tmp_path, "same")


def test_markdown_report_separates_judgment_and_coaching() -> None:
    report = {
        "disclosure": "Simulation only.",
        "initial_assessment": {
            "decision": "Out",
            "investment_likelihood": 0.3,
            "decision_confidence": 0.6,
        },
        "final_assessment": {
            "decision": "In",
            "investment_likelihood": 0.7,
            "decision_confidence": 0.65,
            "decision_justification": "The answer resolved retention.",
            "unresolved_uncertainties": ["Scale remains unverified."],
            "reversal_conditions": ["Retention cannot be substantiated."],
        },
        "pitch_improvement_suggestions": ["Lead with retention."],
        "founder_reflection_prompts": ["What substantiates retention?"],
        "usage": {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01},
    }

    rendered = markdown_report(report)

    assert "## Simulated investor judgment" in rendered
    assert "## Founder coaching" in rendered
    assert rendered.index("Simulated investor judgment") < rendered.index(
        "Founder coaching"
    )


def test_project_registers_independent_rehearsal_entry_point() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'vc-clone-rehearsal = "vc_clone_graph.rehearsal_cli:main"' in text


@pytest.mark.parametrize("separate_workspace", [False, True])
def test_grounded_start_builds_baseline_then_starts_rehearsal(
    tmp_path: Path, monkeypatch, capsys, separate_workspace
) -> None:
    config = load_rehearsal_config(ROOT / "configs/rehearsal-charles-v41-grounded.toml")
    payload = config.model_dump(mode="python")
    payload["rehearsal"]["output_root"] = "outputs/rehearsals"
    payload["rehearsal"]["checkpoint_path"] = "outputs/checkpoints.sqlite"
    config = type(config).model_validate(payload)
    if separate_workspace:
        config._workspace = tmp_path
    run_root = tmp_path / "canonical-run"
    run_root.mkdir()
    (run_root / "input-provenance.json").write_text(
        json.dumps({"pitch_sha256": "digest"}), encoding="utf-8"
    )
    baseline = SimpleNamespace(
        run_root=run_root,
        episode_slug="live-episode",
    )
    build_calls = []

    def build(**kwargs):
        build_calls.append(kwargs)
        return baseline

    runtime = SimpleNamespace(
        investor=SimpleNamespace(display_name="Charles Hudson"),
        classifier_resolution=SimpleNamespace(
            status="fallback", artifact=None, reason="test"
        ),
    )
    workflow = SimpleNamespace(
        start=lambda session_id, **kwargs: {"status": "awaiting_answer"}
    )
    launch_directory = tmp_path / "web" if separate_workspace else tmp_path
    launch_directory.mkdir(exist_ok=True)
    monkeypatch.chdir(launch_directory)
    monkeypatch.setattr(
        "vc_clone_graph.rehearsal_cli.build_or_load_canonical_baseline", build
    )
    monkeypatch.setattr(
        "vc_clone_graph.rehearsal_cli.resolve_investor", lambda *args, **kwargs: runtime
    )
    monkeypatch.setattr(
        "vc_clone_graph.rehearsal_cli._build_workflow",
        lambda *args, **kwargs: workflow,
    )
    args = argparse.Namespace(
        pitch=None,
        text="Founder pitch",
        session="live-session",
        company=["TargetCo"],
        vc="charles-hudson-precursor-ventures",
        build_canonical_baseline=True,
        canonical_baseline=None,
        rehearsal_depth="quick",
    )

    command_start(config, args)

    assert len(build_calls) == 1
    assert build_calls[0]["workspace"] == tmp_path
    assert Path.cwd() == launch_directory
    assert build_calls[0]["pitch"] == "Founder pitch\n"
    metadata = json.loads(
        (
            tmp_path
            / "outputs/rehearsals/charles-hudson-precursor-ventures/live-session/session-config.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["rehearsal_depth"] == "quick"
    assert metadata["max_questions"] == 3
    assert json.loads(capsys.readouterr().out)["status"] == "awaiting_answer"
