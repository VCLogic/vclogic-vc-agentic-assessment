from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from tests.test_canonical_snapshot import _registry, _status


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_canonical_snapshot.py"


def test_cli_builds_snapshot_and_registry(tmp_path: Path) -> None:
    registry, _, _ = _registry(tmp_path)
    registry_payload = json.loads(registry.read_text())
    registry_payload["investors"]["phil-nadel"] = registry_payload["investors"].pop(
        "example-vc"
    )
    registry.write_text(json.dumps(registry_payload), encoding="utf-8")
    for run_config in (tmp_path / "old").glob("*/run-config.json"):
        payload = json.loads(run_config.read_text())
        payload["run"]["vc_slug"] = "phil-nadel"
        run_config.write_text(json.dumps(payload), encoding="utf-8")
    status = _status(tmp_path / "status.json", [])
    destination = tmp_path / "snapshot"
    output_registry = tmp_path / "generated-registry.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--registry",
            str(registry),
            "--override-status",
            str(status),
            "--destination",
            str(destination),
            "--output-registry",
            str(output_registry),
            "--expected-total",
            "2",
            "--expected-overrides",
            "0",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Published 2 artifacts" in result.stdout
    assert "0 overrides, 2 retained" in result.stdout
    assert destination.is_dir()
    assert output_registry.is_file()
