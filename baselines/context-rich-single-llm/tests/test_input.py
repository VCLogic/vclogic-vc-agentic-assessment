from pathlib import Path

import pytest

from config import load_baseline_config
from input import assemble_context, select_unique_precedents


class Hit:
    def __init__(self, slug, score):
        self.episode_slug = slug
        self.score = score


def test_unique_selection_excludes_target_and_deduplicates():
    hits = [Hit("target", 1), Hit("1-a", .9), Hit("1-a", .8), Hit("2-b", .7), Hit("3-c", .6), Hit("4-d", .5), Hit("5-e", .4)]
    selected = select_unique_precedents(hits, "target", 5)
    assert [row.episode_slug for row in selected] == ["1-a", "2-b", "3-c", "4-d", "5-e"]


def test_real_context_is_complete_target_excluded_and_deterministic():
    config = load_baseline_config(Path("baselines/context-rich-single-llm/config.toml"))
    first, first_manifest = assemble_context(config, "18-rowvigor")
    second, second_manifest = assemble_context(config, "18-rowvigor")
    assert first == second and first_manifest == second_manifest
    assert len(first.precedents) == 5
    assert len({row.episode_slug for row in first.precedents}) == 5
    assert all(row.episode_slug != "18-rowvigor" for row in first.precedents)
    assert first.current_pitch
    assert len(first.taxonomy) >= 40
    assert {row.path for row in first.wiki} == set(first_manifest["wiki_sha256s"])
    assert first_manifest["target_episode_slug"] == "18-rowvigor"
    assert first_manifest["embedding"]["complete"] is True
    assert "actual_label" not in first_manifest


def test_config_rejects_non_five_retrieval(tmp_path):
    source = Path("baselines/context-rich-single-llm/config.toml").read_text()
    path = tmp_path / "bad.toml"
    path.write_text(source.replace("top_k = 5", "top_k = 4"))
    with pytest.raises(ValueError, match="exactly five"):
        load_baseline_config(path)
