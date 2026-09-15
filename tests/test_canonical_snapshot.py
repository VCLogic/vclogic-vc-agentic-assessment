from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vc_clone_graph.canonical_snapshot import (
    build_snapshot,
    directory_digest,
    select_snapshot_sources,
    write_snapshot_registry,
)
from vc_clone_graph.evaluation import load_canonical_predictions, load_registry


RUNTIME_MAP = {"example-vc-runtime": "example-vc"}


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_json(path: Path, payload: object) -> Path:
    return _write_text(path, json.dumps(payload, indent=2) + "\n")


def _artifact(
    root: Path,
    *,
    episode: str,
    decision: str = "Out",
    likelihood: float = 0.2,
    contract: str = "v4",
    runtime_slug: str = "example-vc-runtime",
    marker: str = "trace",
) -> Path:
    _write_json(
        root / "summary.json",
        {
            "contract_version": contract,
            "episode_slug": episode,
            "decision": {
                "decision": decision,
                "investment_likelihood": likelihood,
                "decision_confidence": 0.75,
                "review_priority_score": 0.65,
            },
            "phase1_status": "accepted",
            "phase1_iterations": 2,
            "phase2_status": "provisional",
            "phase2_iterations": 3,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cached_input_tokens": 10,
                "cost_usd": 0.01,
            },
        },
    )
    investigation = _write_json(
        root / "phase1" / "investigation.json",
        {"schema_version": "investigation-v4", "episode_slug": episode, "rationales": []},
    )
    _write_text(
        root / "phase1" / "investigation.sha256",
        hashlib.sha256(investigation.read_bytes()).hexdigest() + "\n",
    )
    _write_json(
        root / "run-config.json",
        {"run": {"vc_slug": runtime_slug, "episode_slug": episode}},
    )
    _write_text(root / "trace" / "turn-1.txt", marker)
    return root


def _registry(tmp_path: Path) -> tuple[Path, Path, Path]:
    old_one = _artifact(
        tmp_path / "old" / "1-one",
        episode="1-one",
        decision="Out",
        likelihood=0.1,
        marker="old one",
    )
    old_two = _artifact(
        tmp_path / "old" / "2-two",
        episode="2-two",
        decision="Out",
        likelihood=0.3,
        marker="old two",
    )
    _write_json(
        tmp_path / "labels.json",
        [
            {
                "episode_slug": "1-one",
                "pitch_window_decision": "In",
                "evaluation_eligible": True,
            },
            {
                "episode_slug": "2-two",
                "pitch_window_decision": "Out",
                "evaluation_eligible": True,
            },
        ],
    )
    registry = _write_json(
        tmp_path / "registry.json",
        {
            "schema": "canonical-vc-evaluation-registry-v1",
            "investors": {
                "example-vc": {
                    "display_name": "Example VC",
                    "label_file": "labels.json",
                    "eligible_count": 2,
                    "sources": [
                        {"kind": "summary_glob", "path": "old/*/summary.json"}
                    ],
                }
            },
        },
    )
    return registry, old_one, old_two


def _status(path: Path, rows: list[dict[str, object]]) -> Path:
    return _write_json(path, {"completed": rows, "failed": [], "excluded": []})


def _override_row(root: Path, **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "vc_slug": "example-vc-runtime",
        "episode_slug": "1-one",
        "actual": "In",
        "predicted": "In",
        "verification": "verified",
        "artifact_root": str(root),
    }
    row.update(changes)
    return row


def test_directory_digest_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_text(root / "z.txt", "last")
    _write_text(root / "nested" / "a.txt", "first")

    first = directory_digest(root)
    second = directory_digest(root)

    assert first == second
    assert first.file_count == 2
    assert first.byte_count == len("last") + len("first")
    assert len(first.sha256) == 64

    _write_text(root / "nested" / "a.txt", "changed")
    assert directory_digest(root).sha256 != first.sha256


def test_directory_digest_includes_relative_paths(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_text(left / "a.txt", "same")
    _write_text(right / "b.txt", "same")

    assert directory_digest(left).sha256 != directory_digest(right).sha256


def test_override_replaces_match_and_other_canonical_row_is_retained(tmp_path: Path) -> None:
    registry, _, old_two = _registry(tmp_path)
    replacement = _artifact(
        tmp_path / "override" / "1-one",
        episode="1-one",
        decision="In",
        likelihood=0.82,
        contract="v4.1",
        marker="replacement",
    )
    status = _status(tmp_path / "status.json", [_override_row(replacement)])

    selected = select_snapshot_sources(
        registry, status, runtime_to_canonical=RUNTIME_MAP
    )

    assert [(row.vc_slug, row.episode_slug, row.selection_tier) for row in selected] == [
        ("example-vc", "1-one", "portfolio_v41_override"),
        ("example-vc", "2-two", "prior_canonical"),
    ]
    assert selected[0].source_root == replacement.resolve()
    assert selected[0].prediction.predicted_decision == "In"
    assert selected[0].prediction.investment_likelihood == pytest.approx(0.82)
    assert selected[1].source_root == old_two.resolve()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"verification": "unverified"}, "not verified"),
        ({"episode_slug": "9-unexpected"}, "unexpected override"),
        ({"predicted": "Maybe"}, "usable decision"),
        ({"actual": "Out"}, "actual label mismatch"),
    ],
)
def test_invalid_override_is_rejected(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    registry, _, _ = _registry(tmp_path)
    replacement = _artifact(tmp_path / "replacement", episode="1-one")
    status = _status(tmp_path / "status.json", [_override_row(replacement, **changes)])

    with pytest.raises(ValueError, match=message):
        select_snapshot_sources(registry, status, runtime_to_canonical=RUNTIME_MAP)


def test_duplicate_override_key_is_rejected(tmp_path: Path) -> None:
    registry, _, _ = _registry(tmp_path)
    replacement = _artifact(tmp_path / "replacement", episode="1-one")
    row = _override_row(replacement)
    status = _status(tmp_path / "status.json", [row, dict(row)])

    with pytest.raises(ValueError, match="duplicate override"):
        select_snapshot_sources(registry, status, runtime_to_canonical=RUNTIME_MAP)


def test_artifact_investor_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    registry, _, _ = _registry(tmp_path)
    replacement = _artifact(
        tmp_path / "replacement",
        episode="1-one",
        decision="In",
        runtime_slug="different-vc-runtime",
    )
    status = _status(tmp_path / "status.json", [_override_row(replacement)])

    with pytest.raises(ValueError, match="investor identity mismatch"):
        select_snapshot_sources(
            registry,
            status,
            runtime_to_canonical={
                "example-vc-runtime": "example-vc",
                "different-vc-runtime": "different-vc",
            },
        )


def test_build_snapshot_copies_full_trees_and_writes_provenance(tmp_path: Path) -> None:
    registry, old_one, _ = _registry(tmp_path)
    old_digest = directory_digest(old_one)
    replacement = _artifact(
        tmp_path / "override" / "1-one",
        episode="1-one",
        decision="In",
        likelihood=0.82,
        contract="v4.1",
        marker="complete replacement trace",
    )
    status = _status(tmp_path / "status.json", [_override_row(replacement)])
    destination = tmp_path / "snapshot"

    manifest = build_snapshot(
        registry,
        status,
        destination,
        runtime_to_canonical=RUNTIME_MAP,
        expected_total=2,
        expected_overrides=1,
    )

    copied = destination / "investors" / "example-vc" / "1-one"
    assert (copied / "trace" / "turn-1.txt").read_text() == "complete replacement trace"
    assert manifest["counts"]["total"] == 2
    assert manifest["counts"]["selection_tier"] == {
        "portfolio_v41_override": 1,
        "prior_canonical": 1,
    }
    stored = json.loads((destination / "manifest.json").read_text())
    assert stored["episode_rows"][0]["source_directory_digest"] == stored[
        "episode_rows"
    ][0]["copied_directory_digest"]
    assert len((destination / "sources.csv").read_text().splitlines()) == 3
    assert directory_digest(old_one) == old_digest


def test_build_snapshot_refuses_existing_destination(tmp_path: Path) -> None:
    registry, _, _ = _registry(tmp_path)
    status = _status(tmp_path / "status.json", [])
    destination = tmp_path / "snapshot"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="destination already exists"):
        build_snapshot(
            registry,
            status,
            destination,
            runtime_to_canonical=RUNTIME_MAP,
            expected_total=2,
            expected_overrides=0,
        )


def test_wrong_strict_count_does_not_publish_or_leave_temporary_tree(
    tmp_path: Path,
) -> None:
    registry, _, _ = _registry(tmp_path)
    status = _status(tmp_path / "status.json", [])
    destination = tmp_path / "snapshot"

    with pytest.raises(ValueError, match="expected 3 total"):
        build_snapshot(
            registry,
            status,
            destination,
            runtime_to_canonical=RUNTIME_MAP,
            expected_total=3,
            expected_overrides=0,
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".snapshot.tmp-*"))


def test_generated_registry_loads_exact_snapshot_population(tmp_path: Path) -> None:
    registry, _, _ = _registry(tmp_path)
    status = _status(tmp_path / "status.json", [])
    destination = tmp_path / "snapshot"
    build_snapshot(
        registry,
        status,
        destination,
        runtime_to_canonical=RUNTIME_MAP,
        expected_total=2,
        expected_overrides=0,
    )
    generated = tmp_path / "evaluation" / "generated-registry.json"

    write_snapshot_registry(registry, destination, generated)

    rows = load_canonical_predictions(load_registry(generated))
    assert [(row.vc_slug, row.episode_slug) for row in rows] == [
        ("example-vc", "1-one"),
        ("example-vc", "2-two"),
    ]
    assert all(destination.resolve() in row.artifact_path.resolve().parents for row in rows)


def test_registry_writer_refuses_overwrite(tmp_path: Path) -> None:
    registry, _, _ = _registry(tmp_path)
    output = _write_text(tmp_path / "existing.json", "keep me")

    with pytest.raises(FileExistsError, match="registry already exists"):
        write_snapshot_registry(registry, tmp_path / "snapshot", output)

    assert output.read_text() == "keep me"
