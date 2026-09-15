from __future__ import annotations

import json
from pathlib import Path

from vc_clone_graph.personalized_cases import PersonalizedCase
from vc_clone_graph.personalized_folds import make_episode_folds, write_fold_manifest


def case(vc: str, episode: str, target: int) -> PersonalizedCase:
    return PersonalizedCase(
        vc_slug=vc,
        vc_input_slug=f"{vc}-fund",
        vc_name=vc.title(),
        episode_slug=episode,
        group=episode,
        target=target,
        pitch_text="pitch",
        phase1_text="phase one",
        phase2_text="phase two",
        phase1_features={},
        phase2_features={},
        wiki_text="wiki",
        actual_rationale_targets={},
        actual_rationale_available=True,
        actual_rationale_source_tier="test",
        actual_rationale_source_format="test",
        source_hashes={"pitch": "a" * 64},
    )


def population() -> list[PersonalizedCase]:
    rows = []
    for index in range(8):
        episode = f"{index}-episode"
        target = index % 2
        rows.append(case("alpha", episode, target))
        if index in {2, 5}:
            rows.append(case("beta", episode, 1 - target))
    return rows


def test_outer_fold_holds_every_row_of_episode_together() -> None:
    cases = population()
    folds = make_episode_folds(cases, inner_splits=3, seed=17)

    assert len(folds) == 8
    assert {fold.held_episode for fold in folds} == {
        f"{index}-episode" for index in range(8)
    }
    for fold in folds:
        assert {cases[index].episode_slug for index in fold.test_indices} == {
            fold.held_episode
        }
        assert fold.held_episode not in {
            cases[index].episode_slug for index in fold.train_indices
        }
        for inner_train, inner_valid in fold.inner_splits:
            train_groups = {cases[index].episode_slug for index in inner_train}
            valid_groups = {cases[index].episode_slug for index in inner_valid}
            assert fold.held_episode not in train_groups | valid_groups
            assert train_groups.isdisjoint(valid_groups)
            assert {cases[index].target for index in inner_train} == {0, 1}


def test_fold_manifest_is_deterministic_and_source_bound(tmp_path: Path) -> None:
    cases = population()
    folds = make_episode_folds(cases, inner_splits=3, seed=17)

    first = write_fold_manifest(
        tmp_path / "first.json", cases, folds, source_registry_sha256="f" * 64
    )
    second = write_fold_manifest(
        tmp_path / "second.json", cases, folds, source_registry_sha256="f" * 64
    )

    assert first == second
    assert len(first) == 64
    assert (tmp_path / "first.sha256").read_text().strip() == first
    payload = json.loads((tmp_path / "first.json").read_text())
    assert payload["episode_count"] == 8
    assert payload["case_count"] == 10
    assert payload["source_registry_sha256"] == "f" * 64
    assert payload["folds"][2]["held_episode"] == "2-episode"
