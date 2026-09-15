from __future__ import annotations

import json
from pathlib import Path
import csv

import pytest

from vc_clone_graph.evaluation import (
    PredictionRow,
    build_evaluation,
    classification_metrics,
    load_canonical_predictions,
    load_canonical_artifacts,
    load_registry,
    rank_predictions,
    ranking_metrics,
    render_markdown,
    write_evaluation_outputs,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(
    path: Path,
    *,
    episode_slug: str,
    decision: str,
    likelihood: float,
) -> Path:
    return _write_json(
        path,
        {
            "episode_slug": episode_slug,
            "decision": {
                "decision": decision,
                "investment_likelihood": likelihood,
                "decision_confidence": 0.75,
                "review_priority_score": 0.9,
            },
            "phase1_status": "accepted",
            "phase1_iterations": 1,
            "phase2_status": "provisional",
            "phase2_iterations": 2,
            "usage": {"cost_usd": 0.01},
        },
    )


def _row(
    episode_slug: str,
    actual: str,
    predicted: str,
    likelihood: float,
) -> PredictionRow:
    return PredictionRow(
        vc_slug="example-vc",
        vc_name="Example VC",
        episode_slug=episode_slug,
        actual_decision=actual,
        predicted_decision=predicted,
        investment_likelihood=likelihood,
        decision_confidence=0.75,
        review_priority_score=0.9,
        phase1_status="accepted",
        phase1_iterations=1,
        phase2_status="accepted",
        phase2_iterations=1,
        cost_usd=0.01,
        artifact_path=Path(f"/{episode_slug}/summary.json"),
    )


def test_later_registry_source_replaces_an_earlier_artifact(tmp_path: Path) -> None:
    labels = _write_json(
        tmp_path / "labels.json",
        [
            {
                "episode_slug": "1-one",
                "pitch_window_decision": "In",
                "evaluation_eligible": True,
            }
        ],
    )
    old = _summary(
        tmp_path / "old" / "1-one" / "summary.json",
        episode_slug="1-one",
        decision="Out",
        likelihood=0.2,
    )
    new = _summary(
        tmp_path / "new" / "1-one" / "summary.json",
        episode_slug="1-one",
        decision="In",
        likelihood=0.7,
    )
    registry_path = _write_json(
        tmp_path / "registry.json",
        {
            "schema": "canonical-vc-evaluation-registry-v1",
            "investors": {
                "example-vc": {
                    "display_name": "Example VC",
                    "label_file": str(labels.relative_to(tmp_path)),
                    "eligible_count": 1,
                    "sources": [
                        {"kind": "summary_files", "paths": [str(old.relative_to(tmp_path))]},
                        {"kind": "summary_files", "paths": [str(new.relative_to(tmp_path))]},
                    ],
                }
            },
        },
    )

    registry = load_registry(registry_path)
    rows = load_canonical_predictions(registry, ["example-vc"])

    assert len(rows) == 1
    assert rows[0].episode_slug == "1-one"
    assert rows[0].actual_decision == "In"
    assert rows[0].predicted_decision == "In"
    assert rows[0].investment_likelihood == pytest.approx(0.7)
    assert rows[0].artifact_path == new


def test_artifact_loader_accepts_phase1_only_summary_without_decision(tmp_path: Path) -> None:
    labels = _write_json(
        tmp_path / "labels.json",
        [{
            "episode_slug": "1-one",
            "pitch_window_decision": "In",
            "evaluation_eligible": True,
        }],
    )
    summary = _write_json(
        tmp_path / "runs/1-one/summary.json",
        {
            "episode_slug": "1-one",
            "phase1_status": "provisional",
            "phase2_status": "not_run",
            "decision": {},
        },
    )
    registry_path = _write_json(
        tmp_path / "registry.json",
        {
            "schema": "canonical-vc-evaluation-registry-v1",
            "investors": {
                "example-vc": {
                    "display_name": "Example VC",
                    "label_file": str(labels.relative_to(tmp_path)),
                    "eligible_count": 1,
                    "sources": [{
                        "kind": "summary_files",
                        "paths": [str(summary.relative_to(tmp_path))],
                    }],
                }
            },
        },
    )

    rows = load_canonical_artifacts(load_registry(registry_path))

    assert len(rows) == 1
    assert rows[0].episode_slug == "1-one"
    assert rows[0].actual_decision == "In"
    assert rows[0].artifact_path == summary


def test_batch_status_and_filtered_recovery_status_supply_artifacts(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "labels.json",
        [
            {
                "episode_slug": slug,
                "pitch_window_decision": actual,
                "evaluation_eligible": True,
            }
            for slug, actual in (("1-one", "Out"), ("2-two", "In"))
        ],
    )
    old = _summary(
        tmp_path / "runs" / "1-one" / "summary.json",
        episode_slug="1-one",
        decision="Out",
        likelihood=0.1,
    )
    recovered = _summary(
        tmp_path / "recovery" / "2-two" / "summary.json",
        episode_slug="2-two",
        decision="In",
        likelihood=0.8,
    )
    _write_json(
        tmp_path / "batch.json",
        {"completed_records": [{"episode_slug": "1-one", "artifact_root": str(old.parent)}]},
    )
    _write_json(
        tmp_path / "recovery.json",
        {
            "completed_records": [
                {
                    "vc": "Example VC",
                    "episode_slug": "2-two",
                    "artifact_root": str(recovered.parent),
                },
                {
                    "vc": "Different VC",
                    "episode_slug": "ignored",
                    "artifact_root": str(tmp_path / "ignored"),
                },
            ]
        },
    )
    registry_path = _write_json(
        tmp_path / "registry.json",
        {
            "schema": "canonical-vc-evaluation-registry-v1",
            "investors": {
                "example-vc": {
                    "display_name": "Example VC",
                    "label_file": "labels.json",
                    "eligible_count": 2,
                    "sources": [
                        {"kind": "batch_status", "path": "batch.json"},
                        {
                            "kind": "recovery_status",
                            "path": "recovery.json",
                            "vc_name": "Example VC",
                        },
                    ],
                }
            },
        },
    )

    rows = load_canonical_predictions(load_registry(registry_path))

    assert [(row.episode_slug, row.predicted_decision) for row in rows] == [
        ("1-one", "Out"),
        ("2-two", "In"),
    ]


def test_summary_glob_is_strict_about_missing_eligible_episodes(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "labels.json",
        [
            {
                "episode_slug": "1-one",
                "pitch_window_decision": "In",
                "evaluation_eligible": True,
            }
        ],
    )
    registry_path = _write_json(
        tmp_path / "registry.json",
        {
            "schema": "canonical-vc-evaluation-registry-v1",
            "investors": {
                "example-vc": {
                    "display_name": "Example VC",
                    "label_file": "labels.json",
                    "eligible_count": 1,
                    "sources": [
                        {"kind": "summary_glob", "path": "runs/*/summary.json"}
                    ],
                }
            },
        },
    )

    with pytest.raises(ValueError, match="missing canonical artifacts.*1-one"):
        load_canonical_predictions(load_registry(registry_path))


def test_classification_and_ranking_metrics_are_computed_from_phase2_fields() -> None:
    rows = [
        _row("1-first", "In", "In", 0.9),
        _row("2-second", "In", "Out", 0.8),
        _row("3-third", "Out", "In", 0.7),
        _row("4-fourth", "Out", "Out", 0.1),
    ]

    classification = classification_metrics(rows)
    ranking = ranking_metrics(rows, top_ks=(1, 3, 10))

    assert classification == {
        "n": 4,
        "in_count": 2,
        "out_count": 2,
        "tp": 1,
        "fp": 1,
        "tn": 1,
        "fn": 1,
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "in_precision": 0.5,
        "in_recall": 0.5,
        "in_f1": 0.5,
        "specificity": 0.5,
        "prevalence": 0.5,
    }
    assert ranking["average_precision"] == pytest.approx(1.0)
    assert ranking["roc_auc"] == pytest.approx(1.0)
    assert ranking["budgets"] == [
        {"requested_k": 1, "reviewed": 1, "hits": 1, "precision": 1.0, "recall": 0.5},
        {
            "requested_k": 3,
            "reviewed": 3,
            "hits": 2,
            "precision": pytest.approx(2 / 3),
            "recall": 1.0,
        },
        {"requested_k": 10, "reviewed": 4, "hits": 2, "precision": 0.5, "recall": 1.0},
    ]


def test_ranking_ties_are_broken_by_episode_slug() -> None:
    rows = [
        _row("2-later", "Out", "In", 0.6),
        _row("1-earlier", "In", "In", 0.6),
    ]

    ranked = rank_predictions(rows)

    assert [(rank, row.episode_slug) for rank, row in ranked] == [
        (1, "1-earlier"),
        (2, "2-later"),
    ]


def test_markdown_and_csv_outputs_have_stable_public_schemas(tmp_path: Path) -> None:
    rows = [
        _row("1-first", "In", "In", 0.9),
        _row("2-second", "Out", "Out", 0.1),
    ]
    result = build_evaluation(rows, top_ks=(1, 2))

    markdown = render_markdown(result)
    paths = write_evaluation_outputs(result, rows, tmp_path)

    assert "| Investor | N | In / Out | TP | FP | TN | FN |" in markdown
    assert "## Ranking at review budgets" in markdown
    assert set(paths) == {
        "classification_metrics",
        "ranking_metrics",
        "predictions",
        "markdown",
    }
    with paths["classification_metrics"].open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == [
            "scope",
            "vc_slug",
            "vc_name",
            "n",
            "in_count",
            "out_count",
            "tp",
            "fp",
            "tn",
            "fn",
            "accuracy",
            "balanced_accuracy",
            "in_precision",
            "in_recall",
            "in_f1",
            "specificity",
            "prevalence",
            "phase1_accepted",
            "phase1_provisional",
            "phase2_accepted",
            "phase2_provisional",
            "total_cost_usd",
        ]
    with paths["ranking_metrics"].open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == [
            "scope",
            "vc_slug",
            "vc_name",
            "average_precision",
            "roc_auc",
            "requested_k",
            "reviewed",
            "hits",
            "precision_at_k",
            "recall_at_k",
        ]
    with paths["predictions"].open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == [
            "vc_slug",
            "vc_name",
            "rank",
            "episode_slug",
            "actual_decision",
            "predicted_decision",
            "correct",
            "investment_likelihood",
            "decision_confidence",
            "review_priority_score",
            "phase1_status",
            "phase1_iterations",
            "phase2_status",
            "phase2_iterations",
            "cost_usd",
            "artifact_path",
        ]
    assert paths["markdown"].read_text(encoding="utf-8") == markdown
