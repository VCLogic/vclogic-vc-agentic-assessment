from __future__ import annotations

import subprocess
import sys


def test_report_cli_exposes_config_and_family_selection() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/report_personalized_models.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--config" in result.stdout
    assert "--families" in result.stdout
