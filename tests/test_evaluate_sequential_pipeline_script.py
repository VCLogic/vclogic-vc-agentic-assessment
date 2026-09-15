from __future__ import annotations

import subprocess
import sys


def test_sequential_cli_exposes_canonical_and_resume_options() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_sequential_pipeline.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--registry" in result.stdout
    assert "--references" in result.stdout
    assert "--taxonomy" in result.stdout
    assert "--output" in result.stdout
    assert "--resume-analysis" in result.stdout
    assert "--benchmark-output" in result.stdout
    assert "--jobs" in result.stdout
