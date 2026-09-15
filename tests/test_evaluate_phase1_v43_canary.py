from __future__ import annotations

import subprocess
import sys
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_evaluator_module():
    path = ROOT / "scripts/evaluate_phase1_v43_canary.py"
    spec = importlib.util.spec_from_file_location("evaluate_phase1_v43_canary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v43_canary_evaluator_exposes_paired_inputs_and_output() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/evaluate_phase1_v43_canary.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    for option in (
        "--registry", "--references", "--taxonomy", "--manifest",
        "--runs", "--status", "--output",
    ):
        assert option in result.stdout


def test_mapping_effect_summary_compares_same_run_before_and_after_mapping() -> None:
    module = _load_evaluator_module()
    before = {
        "micro_precision": 0.25,
        "micro_recall": 0.50,
        "micro_f1": 1 / 3,
        "family_micro_f1": 0.70,
        "average_predicted_size": 18.0,
    }
    after = {
        "micro_precision": 0.30,
        "micro_recall": 0.45,
        "micro_f1": 0.36,
        "family_micro_f1": 0.72,
        "average_predicted_size": 16.0,
    }

    effect = module._mapping_effect_summary(before, after)

    assert effect == pytest.approx({
        "micro_precision": 0.05,
        "micro_recall": -0.05,
        "micro_f1": 0.36 - (1 / 3),
        "family_micro_f1": 0.02,
        "average_predicted_size": -2.0,
    })
