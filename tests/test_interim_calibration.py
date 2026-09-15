from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import scripts.analyze_current_batch_calibration as command

from vc_clone_graph.interim_calibration import freeze_completed_snapshot


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    status_path = tmp_path / "live-status.json"
    status_path.write_text(
        json.dumps(
            {
                "schema": "round-robin-batch-v1",
                "completed_records": [
                    {
                        "episode_slug": "1-in",
                        "actual_decision": "In",
                        "status": "completed",
                        "artifact_root": "runs/1-in",
                        "summary": {
                            "phase1_status": "provisional",
                            "phase1_findings": ["SEARCH_BUDGET_EXHAUSTED"],
                            "phase2_status": "accepted",
                            "phase2_findings": [],
                        },
                    },
                    {
                        "episode_slug": "2-out",
                        "actual_decision": "Out",
                        "status": "completed",
                        "artifact_root": "runs/2-out",
                        "summary": {
                            "phase1_status": "accepted",
                            "phase1_findings": [],
                            "phase2_status": "provisional",
                            "phase2_findings": ["MINOR_WARNING"],
                        },
                    },
                ],
                "failed_records": [{"episode_slug": "3-failed"}],
                "episode_order": ["1-in", "2-out", "3-failed", "4-pending"],
            }
        ),
        encoding="utf-8",
    )
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            [
                {
                    "episode_slug": "1-in",
                    "evaluation_eligible": True,
                    "pitch_window_decision": "In",
                },
                {
                    "episode_slug": "2-out",
                    "evaluation_eligible": True,
                    "pitch_window_decision": "Out",
                },
                {
                    "episode_slug": "3-failed",
                    "evaluation_eligible": True,
                    "pitch_window_decision": "In",
                },
                {
                    "episode_slug": "4-pending",
                    "evaluation_eligible": True,
                    "pitch_window_decision": "Out",
                },
            ]
        ),
        encoding="utf-8",
    )
    return status_path, labels_path


def test_freeze_completed_snapshot_is_immutable_and_loadable(tmp_path: Path) -> None:
    status_path, labels_path = _write_inputs(tmp_path)
    status_before = status_path.read_bytes()
    labels_before = labels_path.read_bytes()

    snapshot = freeze_completed_snapshot(
        status_path, labels_path, tmp_path / "snapshot"
    )

    assert status_path.read_bytes() == status_before
    assert labels_path.read_bytes() == labels_before
    assert snapshot.episode_count == 2
    assert snapshot.in_count == 1
    assert snapshot.out_count == 1
    frozen_status = json.loads(snapshot.status_path.read_text(encoding="utf-8"))
    assert frozen_status["progress"] == {"processed": 2, "total": 2}
    assert [row["episode_slug"] for row in frozen_status["completed_records"]] == [
        "1-in",
        "2-out",
    ]
    assert frozen_status["completed_records"][0]["phase1_status"] == "provisional"
    assert frozen_status["completed_records"][1]["phase2_findings"] == [
        "MINOR_WARNING"
    ]
    frozen_labels = json.loads(snapshot.labels_path.read_text(encoding="utf-8"))
    assert [row["episode_slug"] for row in frozen_labels] == ["1-in", "2-out"]
    provenance = json.loads(
        (tmp_path / "snapshot/snapshot-provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["source_status_sha256"] == sha256(status_before).hexdigest()
    assert provenance["source_labels_sha256"] == sha256(labels_before).hexdigest()


def test_command_runs_only_approved_calibration_methods(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}
    snapshot = command.InterimSnapshot(
        status_path=tmp_path / "snapshot-status.json",
        labels_path=tmp_path / "snapshot-labels.json",
        episode_count=20,
        in_count=10,
        out_count=10,
    )
    monkeypatch.setattr(command, "freeze_completed_snapshot", lambda *args: snapshot)
    monkeypatch.setattr(command, "load_frozen_population", lambda *args: ["records"])
    monkeypatch.setattr(
        command,
        "make_repeated_stratified_folds",
        lambda records, repeats, splits: ["fold-map"],
    )

    def fake_write(records, output_root, **kwargs):
        captured["records"] = records
        captured["output_root"] = output_root
        captured.update(kwargs)
        return {
            "dataset": {"episodes": 20, "ins": 10, "outs": 10},
            "evaluation": "repeated_stratified",
            "outer_protocol": {"repeats": 5},
        }

    monkeypatch.setattr(command, "write_advanced_analysis", fake_write)

    result = command.run_analysis(
        status_path=tmp_path / "live.json",
        labels_path=tmp_path / "labels.json",
        output_root=tmp_path / "analysis",
        repeats=5,
        splits=5,
        bootstrap_iterations=100,
    )

    assert command.METHODS == (
        "legacy_phase1",
        "phase2",
        "legacy_plus_phase2",
        "majority_out",
        "raw_any_check",
        "raw_standard_check",
        "raw_ranking_score",
    )
    assert captured["methods"] == command.METHODS
    assert captured["bootstrap_iterations"] == 100
    assert captured["fold_maps"] == ["fold-map"]
    assert result["dataset"]["episodes"] == 20
