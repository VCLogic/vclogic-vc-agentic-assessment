from __future__ import annotations

import subprocess
import sys


def test_actual_rationale_cli_exposes_reproducibility_options() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_actual_rationale_models.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    for option in (
        "--registry", "--references", "--taxonomy", "--output",
        "--benchmark-output", "--jobs", "--inner-splits",
        "--bootstrap-iterations", "--permutation-iterations",
        "--review-budgets", "--seed", "--resume-analysis",
    ):
        assert option in result.stdout
