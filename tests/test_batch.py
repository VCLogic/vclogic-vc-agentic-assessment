from __future__ import annotations

import json
from pathlib import Path

import pytest

from vc_clone_graph.batch import (
    discover_observed_pitch_rows,
    load_cohort,
    round_robin_order,
    run_with_retries,
    scoped_config,
)
from vc_clone_graph.config import RunConfig


def base_config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "run": {
                "vc_slug": "charles-hudson-precursor-ventures",
                "episode_slug": "18-rowvigor",
                "input_root": "inputs",
                "output_root": "outputs/base",
                "checkpoint_path": "outputs/base/checkpoints/18-rowvigor.sqlite",
                "taxonomy_path": "taxonomy/codebook_v2.json",
                "contract_version": "v2",
            },
            "provider": {"kind": "fake", "model": "fake"},
            "phase1": {"min_iterations": 1, "max_iterations": 3},
            "phase2": {"min_iterations": 1, "max_iterations": 2},
            "retrieval": {"top_k": 5, "max_exact_reads": 20},
        }
    )


def test_scoped_config_isolates_each_episode_attempt() -> None:
    config = scoped_config(
        base_config(),
        "61-can-a-zebra-survive-in-a-unicorn-world",
        "outputs/charles-batch",
        attempt=2,
    )

    assert config.run.episode_slug == "61-can-a-zebra-survive-in-a-unicorn-world"
    assert config.run.output_root == "outputs/charles-batch/attempt-2"
    assert config.run.checkpoint_path == (
        "outputs/charles-batch/attempt-2/checkpoints/"
        "61-can-a-zebra-survive-in-a-unicorn-world.sqlite"
    )


def test_run_with_retries_stops_at_first_usable_decision() -> None:
    calls: list[int] = []

    def attempt(number: int) -> dict:
        calls.append(number)
        if number == 1:
            return {"phase1_status": "failed", "decision": {}}
        return {
            "phase1_status": "accepted",
            "phase2_status": "provisional",
            "decision": {"decision": "In"},
        }

    result = run_with_retries(attempt, max_attempts=2)

    assert calls == [1, 2]
    assert result["status"] == "completed"
    assert result["selected_attempt"] == 2
    assert result["summary"]["decision"]["decision"] == "In"


def test_run_with_retries_marks_episode_failed_after_second_failure() -> None:
    result = run_with_retries(
        lambda number: {"phase1_status": "failed", "decision": {}},
        max_attempts=2,
    )

    assert result["status"] == "failed"
    assert result["selected_attempt"] is None
    assert len(result["attempt_summaries"]) == 2


def test_run_with_retries_accepts_frozen_phase1_only_summary() -> None:
    result = run_with_retries(
        lambda number: {
            "phase1_status": "provisional",
            "phase2_status": "not_run",
            "decision": {},
        },
        max_attempts=2,
    )

    assert result["status"] == "completed"
    assert result["selected_attempt"] == 1


def test_load_cohort_keeps_execution_slugs_separate_from_exclusions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cohort.json"
    path.write_text(
        json.dumps(
            {
                "schema": "vc-clone-batch-v1",
                "episodes": ["18-rowvigor", "61-can-a-zebra-survive-in-a-unicorn-world"],
                "excluded": [
                    {"episode_slug": "91-if-we-dont-get-the-money-by-friday", "reason": "unobserved"}
                ],
            }
        ),
        encoding="utf-8",
    )

    cohort = load_cohort(path)

    assert cohort["episodes"] == (
        "18-rowvigor",
        "61-can-a-zebra-survive-in-a-unicorn-world",
    )
    assert cohort["excluded"][0]["episode_slug"] == "91-if-we-dont-get-the-money-by-friday"


def test_round_robin_order_alternates_ins_and_outs_then_finishes_outs() -> None:
    rows = [
        ("24-second-out", "Out"),
        ("18-first-in", "In"),
        ("22-first-out", "Out"),
        ("20-second-in", "In"),
        ("26-third-out", "Out"),
    ]

    assert round_robin_order(rows) == (
        "18-first-in",
        "22-first-out",
        "20-second-in",
        "24-second-out",
        "26-third-out",
    )


def test_round_robin_order_rejects_unobserved_and_duplicate_slugs() -> None:
    with pytest.raises(ValueError, match="observed In or Out"):
        round_robin_order((("18-unobserved", "unobserved"),))
    with pytest.raises(ValueError, match="duplicate episode slug"):
        round_robin_order((("18-same", "In"), ("18-same", "Out")))


def test_discover_observed_pitch_rows_joins_pitch_files_to_audited_records(
    tmp_path: Path,
) -> None:
    pitches = tmp_path / "pitches"
    records = tmp_path / "records"
    pitches.mkdir()
    records.mkdir()
    (pitches / "18-first.txt").write_text("pitch", encoding="utf-8")
    (pitches / "20-second.txt").write_text("pitch", encoding="utf-8")
    (records / "18-first.json").write_text(
        json.dumps({"episode_slug": "18-first", "decision": {"status": "Out"}}),
        encoding="utf-8",
    )
    (records / "20-second.json").write_text(
        json.dumps({"episode_slug": "20-second", "decision": {"status": "In"}}),
        encoding="utf-8",
    )

    assert discover_observed_pitch_rows(pitches, records) == (
        ("18-first", "Out"),
        ("20-second", "In"),
    )
