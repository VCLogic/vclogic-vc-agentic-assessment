from __future__ import annotations

import subprocess
import sys


def test_phase1_v43_diagnostics_cli_exposes_reproducibility_options() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/analyze_phase1_v43_errors.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    for option in (
        "--registry",
        "--references",
        "--taxonomy",
        "--calibration-predictions",
        "--output",
        "--neighbor-threshold",
        "--canary-per-vc",
    ):
        assert option in result.stdout
