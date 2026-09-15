import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "evaluate_rehearsal_questions.py"
    spec = importlib.util.spec_from_file_location("evaluate_rehearsal_questions", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discovers_only_complete_historical_sessions(tmp_path: Path) -> None:
    complete = tmp_path / "vc" / "complete"
    complete.mkdir(parents=True)
    (complete / "session-config.json").write_text("{}")
    (complete / "historical-canary.json").write_text("{}")
    turn = complete / "turns" / "turn-01"
    turn.mkdir(parents=True)
    (turn / "question.json").write_text("{}")
    incomplete = tmp_path / "vc" / "incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "session-config.json").write_text("{}")

    assert _module().discover_session_roots(tmp_path) == (complete,)


def test_cli_source_never_invokes_generation_provider() -> None:
    source = (ROOT / "scripts" / "evaluate_rehearsal_questions.py").read_text()
    assert ".generate(" not in source
    assert "_generation_provider" not in source
