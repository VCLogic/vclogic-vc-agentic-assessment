from pathlib import Path

from vc_clone_graph.rehearsal_config import load_rehearsal_config
from vc_clone_graph.rehearsal_runtime import _classifier_resolution, list_investors


ROOT = Path(__file__).resolve().parents[1]
VC = "charles-hudson-precursor-ventures"


def test_historical_classifier_excludes_the_current_episode() -> None:
    config = load_rehearsal_config(ROOT / "configs/rehearsal/openrouter-luna.toml")
    profile = next(
        row for row in list_investors(ROOT / "inputs") if row.vc_slug == VC
    )

    resolution = _classifier_resolution(
        config,
        profile,
        excluded_episode_slug="18-rowvigor",
    )

    assert resolution.status == "available"
    assert resolution.artifact is not None
    assert resolution.artifact.training_context == "leave_one_episode_out"
    assert resolution.artifact.excluded_episode_slug == "18-rowvigor"
    assert "18-rowvigor" not in resolution.artifact.training_episode_slugs
