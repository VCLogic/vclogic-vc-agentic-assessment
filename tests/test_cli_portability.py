import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_cli_help_does_not_require_unix_file_locking():
    source = Path(__file__).resolve().parents[1] / "src"
    code = (
        f"import sys; sys.path.insert(0, {str(source)!r}); "
        "sys.modules['fcntl'] = None; "
        "from vc_clone_graph.cli import main; main(['--help'])"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "preflight" in result.stdout


@pytest.mark.skipif(os.name != "nt", reason="Windows platform boundary")
def test_v44_store_rejects_windows_before_creating_artifacts(tmp_path):
    from vc_clone_graph.workflow_v44 import _ArtifactStore

    root = tmp_path / "run"
    with pytest.raises(ValueError, match="POSIX"):
        _ArtifactStore.open(root)
    assert not root.exists()
