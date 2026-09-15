from __future__ import annotations

from pathlib import Path
import tomllib

from scripts.run_v5_phase2_canaries import CANARIES, prepare_config


def test_v5_canary_panel_is_one_in_and_one_out_per_vc() -> None:
    vcs = {row.vc_slug for row in CANARIES}
    assert len(CANARIES) == 12
    assert all(
        sorted(row.actual_decision for row in CANARIES if row.vc_slug == vc)
        == ["In", "Out"]
        for vc in vcs
    )


def test_prepare_config_retains_source_retrieval_and_sets_v5_phase2(tmp_path: Path) -> None:
    root = tmp_path
    canary = CANARIES[0]
    source = (
        root / "outputs/canonical-v4-v41-portfolio-2026-08-15/investors"
        / canary.vc_slug / canary.episode_slug
    )
    source.mkdir(parents=True)
    (source / "run-config.json").write_text(
        '{"run":{"vc_slug":"x","episode_slug":"y","input_root":"inputs",'
        '"output_root":"old","checkpoint_path":"old.sqlite","taxonomy_path":"taxonomy/x.json",'
        '"contract_version":"v4"},"provider":{"kind":"openrouter","model":"old"},'
        '"phase1":{"min_iterations":1,"max_iterations":4},'
        '"phase2":{"min_iterations":1,"max_iterations":4},'
        '"retrieval":{"top_k":6,"max_exact_reads":12},'
        '"precedents":{"enabled":true,"selection_policy":"semantic"}}',
        encoding="utf-8",
    )

    path = prepare_config(root, canary, Path("generated"))
    config = tomllib.loads(path.read_text(encoding="utf-8"))

    assert config["run"]["contract_version"] == "v5"
    assert config["provider"]["model"] == "openai/gpt-5.6-luna"
    assert config["phase2"]["max_iterations"] == 2
    assert config["retrieval"]["top_k"] == 6
    assert config["precedents"]["selection_policy"] == "semantic"
