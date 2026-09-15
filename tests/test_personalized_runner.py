from __future__ import annotations

from pathlib import Path

import pytest

from vc_clone_graph.personalized_folds import EpisodeFold
from vc_clone_graph.personalized_runner import (
    FoldRunResult,
    implementation_digest,
    model_family_for_stage,
    run_checkpointed_stage,
)


def folds() -> tuple[EpisodeFold, ...]:
    return (
        EpisodeFold("1-first", (1,), (0,), (((1,), ()),)),
        EpisodeFold("2-second", (0,), (1,), (((0,), ()),)),
    )


def test_stage_retries_once_persists_and_resumes(tmp_path: Path) -> None:
    attempts: dict[tuple[str, str], int] = {}

    def run_fold(condition: str, fold: EpisodeFold) -> list[dict[str, object]]:
        key = (condition, fold.held_episode)
        attempts[key] = attempts.get(key, 0) + 1
        if key == ("all", "1-first") and attempts[key] == 1:
            raise RuntimeError("transient")
        return [{"condition": condition, "episode_slug": fold.held_episode, "score": 0.5}]

    first = run_checkpointed_stage(
        family="tabpfn", conditions=("all",), folds=folds(), output=tmp_path,
        source_manifest={"registry_sha256": "a" * 64}, run_fold=run_fold,
        max_retries=1, resume=True,
    )
    second = run_checkpointed_stage(
        family="tabpfn", conditions=("all",), folds=folds(), output=tmp_path,
        source_manifest={"registry_sha256": "a" * 64}, run_fold=run_fold,
        max_retries=1, resume=True,
    )

    assert len(first) == len(second) == 2
    assert attempts[("all", "1-first")] == 2
    assert attempts[("all", "2-second")] == 1
    assert all(row["status"] == "complete" for row in first)
    assert (tmp_path / "tabpfn/all/1-first.json").is_file()


def test_stage_keeps_failed_fold_visible_after_second_failure(tmp_path: Path) -> None:
    def fail(_condition: str, _fold: EpisodeFold) -> list[dict[str, object]]:
        raise RuntimeError("persistent")

    rows = run_checkpointed_stage(
        family="tabpfn", conditions=("all",), folds=folds()[:1], output=tmp_path,
        source_manifest={"registry_sha256": "a" * 64}, run_fold=fail,
        max_retries=1, resume=False,
    )

    assert rows[0]["status"] == "failed"
    assert rows[0]["attempts"] == 2
    assert "persistent" in rows[0]["error"]


def test_resume_rejects_source_manifest_mismatch(tmp_path: Path) -> None:
    run_checkpointed_stage(
        family="tabpfn", conditions=("all",), folds=folds()[:1], output=tmp_path,
        source_manifest={"registry_sha256": "a" * 64},
        run_fold=lambda condition, fold: [], max_retries=0, resume=True,
    )
    with pytest.raises(ValueError, match="source manifest"):
        run_checkpointed_stage(
            family="tabpfn", conditions=("all",), folds=folds()[:1], output=tmp_path,
            source_manifest={"registry_sha256": "b" * 64},
            run_fold=lambda condition, fold: [], max_retries=0, resume=True,
        )


def test_dry_run_does_not_call_model_or_write_checkpoints(tmp_path: Path) -> None:
    called = False

    def forbidden(_condition: str, _fold: EpisodeFold) -> list[dict[str, object]]:
        nonlocal called
        called = True
        return []

    rows = run_checkpointed_stage(
        family="tabpfn", conditions=("all",), folds=folds(), output=tmp_path,
        source_manifest={"registry_sha256": "a" * 64}, run_fold=forbidden,
        max_retries=1, resume=True, dry_run=True,
    )

    assert called is False
    assert len(rows) == 2
    assert all(row["status"] == "planned" for row in rows)
    assert not list(tmp_path.rglob("*.json"))


def test_model_stage_dispatch_preserves_distinct_families() -> None:
    assert model_family_for_stage("tabpfn") == "tabpfn"
    assert model_family_for_stage("setfit") == "setfit"
    with pytest.raises(ValueError, match="not a directly fitted"):
        model_family_for_stage("prepare")


def test_stage_persists_an_explicit_conditional_skip(tmp_path: Path) -> None:
    rows = run_checkpointed_stage(
        family="ensemble", conditions=("all",), folds=folds()[:1], output=tmp_path,
        source_manifest={"registry_sha256": "a" * 64},
        run_fold=lambda condition, fold: FoldRunResult(
            status="skipped", predictions=(), reason="components not complementary"
        ),
        max_retries=0, resume=False,
    )

    assert rows[0]["status"] == "skipped"
    assert rows[0]["reason"] == "components not complementary"
    assert rows[0]["error"] is None


def test_implementation_digest_binds_paths_and_contents(tmp_path: Path) -> None:
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")

    original = implementation_digest((first, second), root=tmp_path)
    assert original == implementation_digest((second, first), root=tmp_path)
    second.write_text("changed", encoding="utf-8")
    assert original != implementation_digest((first, second), root=tmp_path)
