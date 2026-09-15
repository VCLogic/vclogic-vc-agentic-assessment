import json
from pathlib import Path

from vc_clone_graph.rehearsal_mode_evaluation import (
    evaluate_replay_files,
    write_evaluation_outputs,
)


def _replay(path: Path, *, episode: str, actual: str, final_score: float) -> Path:
    direct = "In" if final_score >= 0.5 else "Out"
    path.write_text(
        json.dumps(
            {
                "schema_version": "historical-rehearsal-replay-v2",
                "episode_slug": episode,
                "actual_decision": actual,
                "status": "complete",
                "predicted_decision": direct,
                "investment_likelihood": final_score,
                "replay_turns": [{"turn": 1, "accepted": True}],
                "total_usage": {"cost_usd": 0.02},
                "classification": {
                    "status": "available",
                    "artifact_id": f"held-{episode}",
                    "snapshots": [
                        {
                            "stage": "initial",
                            "probability_in": 1 - final_score,
                            "predicted_decision": "Out" if direct == "In" else "In",
                        },
                        {
                            "stage": "final",
                            "probability_in": final_score,
                            "predicted_decision": direct,
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_evaluator_separates_static_final_and_direct_endpoints(tmp_path: Path) -> None:
    paths = [
        _replay(tmp_path / "in.json", episode="1-in", actual="In", final_score=0.8),
        _replay(tmp_path / "out.json", episode="2-out", actual="Out", final_score=0.2),
    ]

    result = evaluate_replay_files({"classification_informed": paths})

    methods = {row.method: row for row in result.metrics}
    assert methods["classification_informed_classifier"].balanced_accuracy == 1.0
    assert methods["classification_informed_direct"].balanced_accuracy == 1.0
    assert methods["static_classifier"].balanced_accuracy == 0.0
    assert result.audit.fallback_count == 0
    assert result.audit.total_questions == 2
    assert result.audit.total_cost_usd == 0.04


def test_evaluator_writes_csv_and_markdown(tmp_path: Path) -> None:
    paths = [
        _replay(tmp_path / "in.json", episode="1-in", actual="In", final_score=0.8),
        _replay(tmp_path / "out.json", episode="2-out", actual="Out", final_score=0.2),
    ]
    result = evaluate_replay_files({"classification_informed": paths})

    outputs = write_evaluation_outputs(tmp_path / "report", result)

    assert outputs["metrics_csv"].is_file()
    assert outputs["predictions_csv"].is_file()
    assert b"\r\n" not in outputs["metrics_csv"].read_bytes()
    assert b"\r\n" not in outputs["predictions_csv"].read_bytes()
    assert "classification_informed_classifier" in outputs["report_md"].read_text()


def test_evaluator_adds_canonical_baseline_and_grounded_final(tmp_path: Path) -> None:
    root = tmp_path / "session"
    root.mkdir()
    replay = _replay(
        root / "replay-evaluation.json",
        episode="1-grounded",
        actual="In",
        final_score=0.7,
    )
    (root / "founder-report.json").write_text(
        json.dumps({
            "grounded": {
                "baseline": {
                    "decision": "Out",
                    "investment_likelihood": 0.2,
                    "phase2": {"sha256": "a" * 64},
                }
            }
        }),
        encoding="utf-8",
    )

    result = evaluate_replay_files({"v41_grounded": [replay]})

    methods = {row.method for row in result.predictions}
    assert "canonical_v41_baseline" in methods
    assert "v41_grounded_final" in methods
