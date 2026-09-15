from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_phase1_v43_canary import existing_case_mode


ROOT = Path(__file__).resolve().parents[1]


def test_v43_canary_runner_exposes_cost_and_case_controls() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_phase1_v43_canary.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    for option in (
        "--manifest", "--config-dir", "--output", "--cost-ceiling",
        "--per-case-reserve", "--limit", "--dry-run",
    ):
        assert option in result.stdout


def test_v43_canary_runner_skips_frozen_and_resumes_checkpoint(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    checkpoint = tmp_path / "checkpoint.sqlite"
    assert existing_case_mode(state, checkpoint) == "run"
    checkpoint.write_bytes(b"sqlite")
    assert existing_case_mode(state, checkpoint) == "resume"
    state.write_text('{"phase1_status":"provisional","phase2_status":"not_run"}')
    assert existing_case_mode(state, checkpoint) == "skip"


def test_v43_mapping_retry_exposes_bounded_repair_controls() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/retry_phase1_v43_mapping.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--run-root" in result.stdout
    assert "--max-attempts" in result.stdout
