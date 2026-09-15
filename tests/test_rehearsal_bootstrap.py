from __future__ import annotations

import json
from pathlib import Path

import pytest

from vc_clone_graph.config import load_config
from vc_clone_graph.firewall import verify_package
from vc_clone_graph.rehearsal_bootstrap import (
    build_or_load_canonical_baseline,
    derive_live_run_config,
    prepare_live_package,
)
from vc_clone_graph.rehearsal_config import RehearsalConfig


ROOT = Path(__file__).resolve().parents[1]


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    vc = "test-investor"
    (root / "investors").mkdir(parents=True)
    (root / "investors" / f"{vc}.toml").write_text(
        'vc_slug = "test-investor"\n'
        'display_name = "Test Investor"\n'
        'firm = "Test Fund"\n'
        'role = "Partner"\n'
        'wiki_path = "wiki/test-investor"\n',
        encoding="utf-8",
    )
    (root / "taxonomy").mkdir()
    (root / "taxonomy/codebook_v_final.json").write_text(
        json.dumps([{"label": "founder_execution", "definition": "Execution"}]),
        encoding="utf-8",
    )
    (root / f"wiki/{vc}").mkdir(parents=True)
    (root / f"wiki/{vc}/profile.md").write_text("# Investor\nEvidence.", encoding="utf-8")
    (root / "indexes").mkdir()
    (root / f"indexes/{vc}.json").write_text("{}", encoding="utf-8")
    return root


def test_prepare_live_package_is_verified_and_manifest_bound(tmp_path: Path) -> None:
    source = _source_root(tmp_path)
    destination = tmp_path / "session/canonical-bootstrap/inputs"

    prepared = prepare_live_package(
        source_root=source,
        destination_root=destination,
        vc_slug="test-investor",
        episode_slug="live-session-abc",
        pitch="Founder pitch",
        target_company_aliases=("TargetCo",),
    )
    package = verify_package(prepared, "test-investor", "live-session-abc")
    audit = json.loads(package.audit.read_text(encoding="utf-8"))

    assert package.pitch.read_text(encoding="utf-8") == "Founder pitch\n"
    assert audit["audit_source"] == "live-founder-submission"
    assert audit["status"] == "audited"
    assert all(value is False for value in audit["leakage_checklist"].values())
    assert audit["target_company_aliases"] == ["TargetCo"]


def test_prepare_live_package_is_idempotent_but_rejects_changed_pitch(
    tmp_path: Path,
) -> None:
    source = _source_root(tmp_path)
    destination = tmp_path / "session/canonical-bootstrap/inputs"
    kwargs = {
        "source_root": source,
        "destination_root": destination,
        "vc_slug": "test-investor",
        "episode_slug": "live-session-abc",
        "target_company_aliases": ("TargetCo",),
    }

    first = prepare_live_package(pitch="Founder pitch", **kwargs)
    second = prepare_live_package(pitch="Founder pitch", **kwargs)

    assert second == first
    with pytest.raises(ValueError, match="different pitch"):
        prepare_live_package(pitch="Changed pitch", **kwargs)


def test_derive_live_run_config_isolates_the_canonical_run(tmp_path: Path) -> None:
    template = load_config(ROOT / "configs/openrouter-luna-charles-v4.toml")
    workspace = tmp_path.resolve()
    session = workspace / "outputs/rehearsals/test-investor/session-one"

    derived = derive_live_run_config(
        template=template,
        workspace=workspace,
        session_root=session,
        vc_slug="test-investor",
        episode_slug="live-session-one",
        contract_version="v4.1",
    )

    assert derived.run.vc_slug == "test-investor"
    assert derived.run.episode_slug == "live-session-one"
    assert derived.run.input_root.endswith("canonical-bootstrap/inputs")
    assert derived.run.output_root.endswith("canonical-bootstrap/runs")
    assert derived.run.contract_version == "v4.1"
    assert derived.run.mode == "full"


def test_build_or_load_runs_once_and_reuses_verified_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    source = _source_root(tmp_path)
    template_path = tmp_path / "canonical.toml"
    template_path.write_text(
        (ROOT / "configs/openrouter-luna-charles-v4.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    rehearsal_payload = {
        "rehearsal": {
            "input_root": str(source.relative_to(tmp_path)),
            "output_root": "outputs/rehearsals",
            "checkpoint_path": "outputs/rehearsals/checkpoints.sqlite",
            "max_questions": 4,
        },
        "provider": {
            "kind": "openrouter",
            "model": "openai/gpt-5.6-luna",
            "api_key_env": "OPENROUTER_API_KEY",
        },
        "embedding": {
            "kind": "sentence_transformers",
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "revision": "revision",
        },
        "retrieval": {"top_k": 5, "max_exact_reads": 12},
        "classification": {
            "mode": "v41_grounded",
            "canonical_config_path": template_path.name,
            "live_contract_version": "v4.1",
            "maximum_questions": 4,
        },
    }
    rehearsal = RehearsalConfig.model_validate(rehearsal_payload)
    calls = []
    expected = object()

    def runner(config, *, resume=False):
        calls.append((config, resume))
        run_root = tmp_path / config.run.output_root / config.run.episode_slug
        run_root.mkdir(parents=True)
        (run_root / "summary.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "vc_clone_graph.rehearsal_bootstrap.load_canonical_baseline_from_run",
        lambda **kwargs: expected,
    )
    kwargs = {
        "workspace": tmp_path,
        "rehearsal_config": rehearsal,
        "session_root": tmp_path / "outputs/rehearsals/test-investor/session-one",
        "vc_slug": "test-investor",
        "episode_slug": "live-session-one",
        "pitch": "Founder pitch",
        "target_company_aliases": ("TargetCo",),
        "runner": runner,
    }

    assert build_or_load_canonical_baseline(**kwargs) is expected
    assert build_or_load_canonical_baseline(**kwargs) is expected
    assert len(calls) == 1


def test_bootstrap_failure_is_recorded_and_resumes_from_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    source = _source_root(tmp_path)
    template_path = tmp_path / "canonical.toml"
    template_path.write_text(
        (ROOT / "configs/openrouter-luna-charles-v4.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    rehearsal = RehearsalConfig.model_validate(
        {
            "rehearsal": {
                "input_root": "source",
                "output_root": "outputs/rehearsals",
                "checkpoint_path": "outputs/rehearsals/checkpoints.sqlite",
                "max_questions": 4,
            },
            "provider": {
                "kind": "openrouter",
                "model": "openai/gpt-5.6-luna",
            },
            "embedding": {
                "kind": "sentence_transformers",
                "model": "nomic-ai/nomic-embed-text-v1.5",
                "revision": "revision",
            },
            "retrieval": {"top_k": 5, "max_exact_reads": 12},
            "classification": {
                "mode": "v41_grounded",
                "canonical_config_path": template_path.name,
                "maximum_questions": 4,
            },
        }
    )
    calls = []
    expected = object()

    def runner(config, *, resume=False):
        calls.append(resume)
        checkpoint = tmp_path / config.run.checkpoint_path
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("checkpoint", encoding="utf-8")
        if len(calls) == 1:
            raise RuntimeError("provider interrupted")

    monkeypatch.setattr(
        "vc_clone_graph.rehearsal_bootstrap.load_canonical_baseline_from_run",
        lambda **kwargs: expected,
    )
    kwargs = {
        "workspace": tmp_path,
        "rehearsal_config": rehearsal,
        "session_root": tmp_path / "outputs/rehearsals/test-investor/session-two",
        "vc_slug": "test-investor",
        "episode_slug": "live-session-two",
        "pitch": "Founder pitch",
        "target_company_aliases": ("TargetCo",),
        "runner": runner,
    }

    with pytest.raises(RuntimeError, match="provider interrupted"):
        build_or_load_canonical_baseline(**kwargs)
    failure = kwargs["session_root"] / "canonical-bootstrap/bootstrap-failure.json"
    assert json.loads(failure.read_text(encoding="utf-8"))["status"] == "failed"

    assert build_or_load_canonical_baseline(**kwargs) is expected
    assert calls == [False, True]


def test_default_canonical_runner_does_not_pollute_rehearsal_stdout(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _source_root(tmp_path)
    template_path = tmp_path / "canonical.toml"
    template_path.write_text(
        (ROOT / "configs/openrouter-luna-charles-v4.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    rehearsal = RehearsalConfig.model_validate(
        {
            "rehearsal": {
                "input_root": "source",
                "output_root": "outputs/rehearsals",
                "checkpoint_path": "outputs/rehearsals/checkpoints.sqlite",
                "max_questions": 4,
            },
            "provider": {"kind": "openrouter", "model": "model"},
            "embedding": {
                "kind": "sentence_transformers",
                "model": "embedding",
                "revision": "revision",
            },
            "retrieval": {"top_k": 5, "max_exact_reads": 12},
            "classification": {
                "mode": "v41_grounded",
                "canonical_config_path": template_path.name,
                "maximum_questions": 4,
            },
        }
    )

    def command_run(config, *, resume=False):
        print('{"canonical": "summary"}')

    monkeypatch.setattr("vc_clone_graph.cli.command_run", command_run)
    monkeypatch.setattr(
        "vc_clone_graph.rehearsal_bootstrap.load_canonical_baseline_from_run",
        lambda **kwargs: object(),
    )

    build_or_load_canonical_baseline(
        workspace=tmp_path,
        rehearsal_config=rehearsal,
        session_root=tmp_path / "outputs/rehearsals/test-investor/session-three",
        vc_slug="test-investor",
        episode_slug="live-session-three",
        pitch="Founder pitch",
        target_company_aliases=("TargetCo",),
    )

    assert capsys.readouterr().out == ""
