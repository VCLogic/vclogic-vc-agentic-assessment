from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from tests.test_canonical_snapshot import RUNTIME_MAP, _artifact, _registry, _status
from vc_clone_graph.canonical_snapshot import build_snapshot, write_snapshot_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "compare_canonical_registries.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_reports_only_changed_artifacts_and_metric_deltas(tmp_path: Path) -> None:
    old_registry, _, _ = _registry(tmp_path)
    replacement = _artifact(
        tmp_path / "new" / "1-one",
        episode="1-one",
        decision="In",
        likelihood=0.85,
        contract="v4.1",
    )
    assert replacement.is_dir()
    new_payload = json.loads(old_registry.read_text())
    new_payload["investors"]["example-vc"]["sources"].append(
        {"kind": "summary_glob", "path": "new/*/summary.json"}
    )
    new_registry = tmp_path / "new-registry.json"
    new_registry.write_text(json.dumps(new_payload), encoding="utf-8")
    output = tmp_path / "comparison"

    result = _run(
        "--old-registry",
        str(old_registry),
        "--new-registry",
        str(new_registry),
        "--output-dir",
        str(output),
        "--expected-changed",
        "1",
    )

    assert result.returncode == 0, result.stderr
    assert "Changed artifacts: 1 / 2" in result.stdout
    report = (output / "comparison.md").read_text()
    assert "Only changed rows are differences; labels and population are identical." in report
    with (output / "changed_predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["vc_slug"], row["episode_slug"]) for row in rows] == [
        ("example-vc", "1-one")
    ]
    assert rows[0]["old_decision"] == "Out"
    assert rows[0]["new_decision"] == "In"
    assert rows[0]["old_rank"] == "2"
    assert rows[0]["new_rank"] == "1"


def test_cli_fails_closed_when_expected_changed_count_is_wrong(tmp_path: Path) -> None:
    registry, _, _ = _registry(tmp_path)
    output = tmp_path / "comparison"

    result = _run(
        "--old-registry",
        str(registry),
        "--new-registry",
        str(registry),
        "--output-dir",
        str(output),
        "--expected-changed",
        "1",
    )

    assert result.returncode != 0
    assert "expected 1 changed artifacts, found 0" in result.stderr
    assert not output.exists()


def test_cli_does_not_count_identical_physical_copies_as_changed(tmp_path: Path) -> None:
    registry, _, _ = _registry(tmp_path)
    status = _status(tmp_path / "status.json", [])
    snapshot = tmp_path / "snapshot"
    build_snapshot(
        registry,
        status,
        snapshot,
        runtime_to_canonical=RUNTIME_MAP,
        expected_total=2,
        expected_overrides=0,
    )
    copied_registry = tmp_path / "copied-registry.json"
    write_snapshot_registry(registry, snapshot, copied_registry)
    output = tmp_path / "comparison"

    result = _run(
        "--old-registry",
        str(registry),
        "--new-registry",
        str(copied_registry),
        "--output-dir",
        str(output),
        "--expected-changed",
        "0",
    )

    assert result.returncode == 0, result.stderr
    assert "Changed artifacts: 0 / 2" in result.stdout
