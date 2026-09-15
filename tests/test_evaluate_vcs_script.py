from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_vcs.py"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _registry(tmp_path: Path) -> Path:
    _write_json(
        tmp_path / "labels.json",
        [
            {
                "episode_slug": "1-one",
                "pitch_window_decision": "In",
                "evaluation_eligible": True,
            },
            {
                "episode_slug": "2-two",
                "pitch_window_decision": "Out",
                "evaluation_eligible": True,
            },
        ],
    )
    for slug, decision, likelihood in (("1-one", "In", 0.8), ("2-two", "Out", 0.2)):
        _write_json(
            tmp_path / "runs" / slug / "summary.json",
            {
                "episode_slug": slug,
                "decision": {
                    "decision": decision,
                    "investment_likelihood": likelihood,
                    "decision_confidence": 0.8,
                    "review_priority_score": 0.9,
                },
                "phase1_status": "accepted",
                "phase1_iterations": 1,
                "phase2_status": "accepted",
                "phase2_iterations": 1,
                "usage": {"cost_usd": 0.01},
            },
        )
    return _write_json(
        tmp_path / "registry.json",
        {
            "schema": "canonical-vc-evaluation-registry-v1",
            "investors": {
                "example-vc": {
                    "display_name": "Example VC",
                    "label_file": "labels.json",
                    "eligible_count": 2,
                    "sources": [
                        {"kind": "summary_glob", "path": "runs/*/summary.json"}
                    ],
                }
            },
        },
    )


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_lists_canonical_vcs(tmp_path: Path) -> None:
    result = _run("--registry", str(_registry(tmp_path)), "--list-vcs")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "example-vc\tExample VC\t2"


def test_cli_evaluates_selected_vc_and_writes_all_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "report"
    result = _run(
        "--registry",
        str(_registry(tmp_path)),
        "--vc",
        "example-vc",
        "--top-k",
        "1,2",
        "--format",
        "both",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    assert "# Canonical VC Classification and Ranking Evaluation" in result.stdout
    assert "| Example VC | 2 | 1 / 1 | 1 | 0 | 1 | 0 |" in result.stdout
    assert {path.name for path in output_dir.iterdir()} == {
        "classification_metrics.csv",
        "ranking_metrics.csv",
        "predictions.csv",
        "evaluation.md",
    }


def test_cli_requires_output_dir_for_csv(tmp_path: Path) -> None:
    result = _run(
        "--registry",
        str(_registry(tmp_path)),
        "--all",
        "--format",
        "csv",
    )

    assert result.returncode != 0
    assert "--output-dir is required" in result.stderr
