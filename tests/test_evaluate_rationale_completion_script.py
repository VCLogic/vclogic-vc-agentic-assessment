from __future__ import annotations

import subprocess
import sys


def test_rationale_completion_cli_exposes_reproducibility_options() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_rationale_completion.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    for option in (
        "--registry",
        "--references",
        "--taxonomy",
        "--output",
        "--vc-slug",
        "--max-hypotheses",
        "--seed",
    ):
        assert option in result.stdout
