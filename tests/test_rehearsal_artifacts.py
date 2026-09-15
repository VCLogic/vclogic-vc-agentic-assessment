import json
from pathlib import Path

import pytest

from vc_clone_graph.rehearsal_artifacts import (
    PendingAnswerConflict,
    RehearsalArtifactStore,
)


def test_pitch_is_immutable_and_answers_are_separate(tmp_path: Path) -> None:
    store = RehearsalArtifactStore.create(
        tmp_path,
        "charles-hudson-precursor-ventures",
        "session-1",
        "Original pitch",
    )
    store.append_answer(
        "A-001",
        {
            "answer_id": "A-001",
            "question_id": "Q-001",
            "text_verbatim": "New answer",
            "received_at": "2026-08-22T00:00:00Z",
        },
    )

    assert store.pitch_path.read_text(encoding="utf-8") == "Original pitch"
    assert "New answer" not in store.pitch_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        store.write_pitch_once("Changed pitch")


def test_failed_call_does_not_replace_accepted_state(tmp_path: Path) -> None:
    store = RehearsalArtifactStore.create(
        tmp_path, "charles-hudson-precursor-ventures", "session-1", "Pitch"
    )
    store.write_accepted("initial-assessment.json", {"decision": "Out"})
    store.write_failed_call(
        "select-question", 1, {"content": "bad"}, "schema invalid"
    )

    assert store.read_json("initial-assessment.json")["decision"] == "Out"
    finding = store.read_json("findings/select-question-001.json")
    assert finding["finding"] == "schema invalid"


def test_verify_detects_pitch_tampering(tmp_path: Path) -> None:
    store = RehearsalArtifactStore.create(
        tmp_path, "charles-hudson-precursor-ventures", "session-1", "Pitch"
    )
    store.pitch_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="pitch digest"):
        store.verify()


def test_store_rejects_path_escape_and_secret_call_metadata(tmp_path: Path) -> None:
    store = RehearsalArtifactStore.create(
        tmp_path, "charles-hudson-precursor-ventures", "session-1", "Pitch"
    )

    with pytest.raises(ValueError, match="relative"):
        store.write_accepted("../outside.json", {"value": 1})
    with pytest.raises(ValueError, match="secret"):
        store.write_call(
            "initial-assessment",
            1,
            {"authorization": "Bearer forbidden"},
            {"content": "{}"},
        )


def test_open_and_verify_returns_manifest_summary(tmp_path: Path) -> None:
    created = RehearsalArtifactStore.create(
        tmp_path, "charles-hudson-precursor-ventures", "session-1", "Pitch"
    )
    opened = RehearsalArtifactStore.open(created.session_root)

    summary = opened.verify()

    assert summary["session_id"] == "session-1"
    assert summary["vc_slug"] == "charles-hudson-precursor-ventures"
    assert json.loads(opened.manifest_path.read_text())["schema_version"] == (
        "founder-rehearsal-session-v1"
    )


def test_pending_answer_submission_is_exact_and_idempotent(tmp_path: Path) -> None:
    store = RehearsalArtifactStore.create(
        tmp_path, "charles-hudson-precursor-ventures", "session-1", "Pitch"
    )

    first = store.accept_answer_submission("Q-001", "Exact founder answer")
    second = store.accept_answer_submission("Q-001", "Exact founder answer")

    assert second == first
    assert first["text_verbatim"] == "Exact founder answer"
    assert len(first["submission_sha256"]) == 64


def test_pending_answer_rejects_different_text_for_the_same_question(
    tmp_path: Path,
) -> None:
    store = RehearsalArtifactStore.create(
        tmp_path, "charles-hudson-precursor-ventures", "session-1", "Pitch"
    )
    store.accept_answer_submission("Q-001", "First answer")

    with pytest.raises(PendingAnswerConflict):
        store.accept_answer_submission("Q-001", "Different answer")


def test_processed_submission_allows_the_next_question(tmp_path: Path) -> None:
    store = RehearsalArtifactStore.create(
        tmp_path, "charles-hudson-precursor-ventures", "session-1", "Pitch"
    )
    pending = store.accept_answer_submission("Q-001", "First answer")
    store.mark_answer_submission_processed("A-001")

    next_submission = store.accept_answer_submission("Q-002", "Second answer")

    assert next_submission["question_id"] == "Q-002"
    assert next_submission["submission_sha256"] != pending["submission_sha256"]


def test_submission_verification_detects_stale_answer_text(tmp_path: Path) -> None:
    store = RehearsalArtifactStore.create(
        tmp_path, "charles-hudson-precursor-ventures", "session-1", "Pitch"
    )
    pending = store.accept_answer_submission("Q-001", "Exact answer")

    assert store.verify_answer_submission(
        pending,
        {
            "question_id": "Q-001",
            "text_verbatim": "Exact answer",
            "submission_sha256": pending["submission_sha256"],
        },
    )
    assert not store.verify_answer_submission(
        pending,
        {
            "question_id": "Q-001",
            "text_verbatim": "Stale answer",
            "submission_sha256": pending["submission_sha256"],
        },
    )
