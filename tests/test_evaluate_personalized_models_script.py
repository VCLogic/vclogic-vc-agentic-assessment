from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def test_personalized_cli_exposes_stages_resume_and_canary_limit() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_personalized_models.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--stage" in result.stdout
    assert "--resume" in result.stdout
    assert "--max-folds" in result.stdout
    assert "--dry-run" in result.stdout
    assert "prepare" in result.stdout
    assert "tabpfn" in result.stdout


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "outputs/canonical-v4-v41-portfolio-2026-08-15").is_dir(),
    reason="requires archived canonical assessment outputs",
)
def test_prepare_dry_run_verifies_canonical_population_without_model_calls() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_personalized_models.py",
            "--config",
            "configs/evaluation/multimodal-personalized-v1.toml",
            "--stage",
            "prepare",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "cases=301" in result.stdout
    assert "ins=86" in result.stdout
    assert "episodes=136" in result.stdout
    assert "model_calls=0" in result.stdout
