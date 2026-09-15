from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_schemas_v5 import payload
from vc_clone_graph.phase2_calibration_evaluation import Phase2CalibrationRecord
from vc_clone_graph.structured_fusion_features import (
    FEATURE_CONDITIONS,
    build_structured_fusion_cases,
    feature_view,
)


def record() -> Phase2CalibrationRecord:
    phase2 = {
        "investment_likelihood": 0.62,
        "decision_confidence": 0.7,
        "review_priority_score": 0.84,
        "signed_decision_confidence": 0.7,
        "decision_in": 1.0,
        "phase2_provisional": 0.0,
        "vc__charles": 1.0,
    }
    semantic = {
        **phase2,
        "rationale__founder_execution__signed_confidence": 0.8,
        "rationale__founder_execution__direction__positive": 1.0,
        "constraint__portfolio_conflict__possible__material": 1.0,
        "conflict_count": 1.0,
        "vc_rationale__charles__founder_execution__signed_confidence": 0.8,
    }
    return Phase2CalibrationRecord(
        vc_slug="charles", vc_name="Charles", episode_slug="18-rowvigor",
        group="18-rowvigor", target=1, raw_decision=1, raw_likelihood=0.62,
        phase2_features=phase2, combined_features={}, semantic_features=semantic,
        artifact_root="runs/charles/18-rowvigor", phase1_sha256="a" * 64,
        phase2_sha256="b" * 64,
    )


def write_v5(root: Path, *, episode_slug: str = "18-rowvigor") -> None:
    candidate = payload()
    candidate["episode_slug"] = episode_slug
    candidate["rationales"][0]["associated_rationales"] = []
    candidate["episode_level_associations"] = [{
        "taxonomy_label": "market_size_assessment",
        "posterior_probability": 0.75,
        "support": 4,
        "lift": 2.0,
        "antecedent_labels": ["founder_execution"],
        "status": "hypothesis_only",
    }]
    path = root / "investors/charles" / episode_slug / "phase1/investigation.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(candidate), encoding="utf-8")


def test_builds_five_deployable_feature_conditions(tmp_path: Path) -> None:
    write_v5(tmp_path)
    cases = build_structured_fusion_cases(
        [record()], tmp_path,
        families={
            "founder_execution": "founding_team",
            "market_size_assessment": "market_opportunity",
        },
    )

    assert tuple(FEATURE_CONDITIONS) == (
        "phase1", "phase2", "phase1_assoc", "phase1_phase2", "full"
    )
    case = cases[0]
    assert case.association_training_episode_slugs == (
        "20-harper-wilde", "22-lunar"
    )
    assert feature_view(case, "phase1")[
        "p1__rationale__founder_execution__signed_confidence"
    ] == pytest.approx(0.8)
    assert feature_view(case, "phase1_assoc")[
        "assoc__market_size_assessment__max_probability"
    ] == pytest.approx(0.75)
    assert feature_view(case, "phase2")["p2__investment_likelihood"] == 0.62
    assert "interaction__signed_rationale_x_likelihood" in feature_view(case, "full")
    assert not any(
        "target" in name or "actual" in name or name.startswith("vc__")
        for condition in FEATURE_CONDITIONS
        for name in feature_view(case, condition)
    )


def test_rejects_misaligned_or_leaking_v5_artifact(tmp_path: Path) -> None:
    write_v5(tmp_path, episode_slug="19-other")
    with pytest.raises(FileNotFoundError):
        build_structured_fusion_cases(
            [record()], tmp_path,
            families={"founder_execution": "founding_team"},
        )

    write_v5(tmp_path)
    path = tmp_path / "investors/charles/18-rowvigor/phase1/investigation.json"
    candidate = json.loads(path.read_text(encoding="utf-8"))
    candidate["association_training_episode_slugs"].append("18-rowvigor")
    path.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(ValueError, match="target episode"):
        build_structured_fusion_cases(
            [record()], tmp_path,
            families={
                "founder_execution": "founding_team",
                "market_size_assessment": "market_opportunity",
            },
        )
