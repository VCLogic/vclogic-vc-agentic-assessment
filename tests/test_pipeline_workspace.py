from pathlib import Path

from vc_clone_graph.rehearsal_cli import _checkpoint, command_investors
from vc_clone_graph.rehearsal_config import load_rehearsal_config

ROOT = Path(__file__).resolve().parents[1]


def test_rehearsal_paths_use_explicit_workspace_without_changing_cwd(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config = load_rehearsal_config(
        ROOT / 'configs/rehearsal-charles-v41-grounded.toml', workspace=ROOT
    )
    command_investors(config)
    assert 'charles-hudson' in capsys.readouterr().out
    assert _checkpoint(config) == ROOT / config.rehearsal.checkpoint_path
    assert config.resolve_path(config.classification.registry_path) == ROOT / config.classification.registry_path
    assert Path.cwd() == tmp_path
    assert '_workspace' not in config.model_dump()


def test_canonical_runner_uses_pipeline_workspace(tmp_path, monkeypatch):
    from vc_clone_graph import cli
    from vc_clone_graph.config import load_config
    from vc_clone_graph.rehearsal_bootstrap import derive_live_run_config

    config = derive_live_run_config(
        template=load_config(ROOT / 'configs/fake-elizabeth-thoras.toml'),
        workspace=tmp_path,
        session_root=tmp_path / 'outputs/session',
        vc_slug='elizabeth-yin-hustle-fund',
        episode_slug='live-test',
        contract_version='v4',
    )
    # Exercise the real command's boundary, stopping before any model operation.
    class BoundaryReached(Exception):
        pass
    def verify(input_root, *args, **kwargs):
        assert input_root.resolve() == tmp_path / 'outputs/session/canonical-bootstrap/inputs'
        raise BoundaryReached
    monkeypatch.setattr(cli, 'verify_package', verify)
    launch = tmp_path / 'web'
    launch.mkdir()
    monkeypatch.chdir(launch)
    import pytest
    with pytest.raises(BoundaryReached):
        cli.command_run(config)
    assert cli._index_path(config) == tmp_path / 'outputs/session/canonical-bootstrap/inputs/indexes/elizabeth-yin-hustle-fund.json'
    assert Path.cwd() == launch
