from __future__ import annotations

import subprocess
import sys


def test_phase1_calibration_cli_exposes_reproducibility_options() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_phase1_calibration.py", "--help"],
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
        "--benchmark-output",
        "--embedding-model",
        "--embedding-revision",
        "--embedding-device",
        "--embedding-batch-size",
        "--jobs",
        "--inner-splits",
        "--pca-components",
        "--bootstrap-iterations",
        "--permutation-iterations",
        "--review-budgets",
        "--seed",
        "--resume-analysis",
    ):
        assert option in result.stdout
