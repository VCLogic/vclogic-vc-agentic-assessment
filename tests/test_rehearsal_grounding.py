from __future__ import annotations

from pathlib import Path
import json
import shutil

import pytest

from vc_clone_graph.rehearsal_grounding import (
    baseline_initial_assessment,
    baseline_summary,
    load_canonical_baseline,
    load_canonical_baseline_from_run,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json"
RUN_ROOT = (
    ROOT
    / "outputs/canonical-v4-v41-portfolio-2026-08-15/investors/charles-hudson"
    / "39-this-pitch-is-damn-near-perfect"
)


@pytest.mark.skipif(not (RUN_ROOT / "summary.json").is_file(), reason="requires archived canonical assessment outputs")
def test_loads_verified_baseline_directly_from_run() -> None:
    provenance = json.loads(
        (RUN_ROOT / "input-provenance.json").read_text(encoding="utf-8")
    )

    baseline = load_canonical_baseline_from_run(
        workspace=ROOT,
        run_root=RUN_ROOT,
        canonical_vc_slug="charles-hudson",
        expected_run_vc_slug="charles-hudson-precursor-ventures",
        expected_episode_slug="39-this-pitch-is-damn-near-perfect",
        expected_pitch_sha256=provenance["pitch_sha256"],
    )

    assert baseline.run_root == RUN_ROOT.resolve()
    assert baseline.decision.decision == "In"


@pytest.mark.skipif(not (RUN_ROOT / "summary.json").is_file(), reason="requires archived canonical assessment outputs")
def test_direct_loader_rejects_pitch_digest_mismatch() -> None:
    with pytest.raises(ValueError, match="pitch digest mismatch"):
        load_canonical_baseline_from_run(
            workspace=ROOT,
            run_root=RUN_ROOT,
            canonical_vc_slug="charles-hudson",
            expected_run_vc_slug="charles-hudson-precursor-ventures",
            expected_episode_slug="39-this-pitch-is-damn-near-perfect",
            expected_pitch_sha256="0" * 64,
        )


@pytest.mark.skipif(not (RUN_ROOT / "summary.json").is_file(), reason="requires archived canonical assessment outputs")
def test_loads_exact_historical_v4_baseline() -> None:
    baseline = load_canonical_baseline(
        workspace=ROOT,
        registry_path=REGISTRY,
        canonical_vc_slug="charles-hudson",
        episode_slug="39-this-pitch-is-damn-near-perfect",
    )

    assert baseline.contract_version == "v4"
    assert baseline.phase1_status == "accepted"
    assert baseline.phase2_status == "accepted"
    assert baseline.decision.decision == "In"
    assert baseline.decision.investment_likelihood == pytest.approx(0.63)
    assert baseline.investigation.episode_slug == baseline.episode_slug
    assert baseline.decision.investigation_sha256 == baseline.investigation_sha256


@pytest.mark.skipif(not (RUN_ROOT.parent / "18-rowvigor/summary.json").is_file(), reason="requires archived canonical assessment outputs")
def test_loads_provisional_baseline_without_discarding_decision() -> None:
    baseline = load_canonical_baseline(
        workspace=ROOT,
        registry_path=REGISTRY,
        canonical_vc_slug="charles-hudson",
        episode_slug="18-rowvigor",
    )

    assert baseline.phase1_status == "provisional"
    assert baseline.phase2_status == "provisional"
    assert baseline.decision.decision == "Out"


@pytest.mark.skipif(not (RUN_ROOT / "summary.json").is_file(), reason="requires archived canonical assessment outputs")
def test_adapts_baseline_without_a_model_call() -> None:
    baseline = load_canonical_baseline(
        workspace=ROOT,
        registry_path=REGISTRY,
        canonical_vc_slug="charles-hudson",
        episode_slug="39-this-pitch-is-damn-near-perfect",
    )

    assessment = baseline_initial_assessment(baseline)
    summary = baseline_summary(baseline, workspace=ROOT)

    assert assessment.decision == "In"
    assert assessment.investment_likelihood == pytest.approx(0.63)
    assert {row.rationale_id for row in assessment.rationales} == {
        row.rationale_id for row in baseline.investigation.rationales
    }
    assert assessment.candidate_questions == ()
    assert summary.phase1.sha256 == baseline.investigation_sha256
    assert summary.phase2.sha256 == baseline.decision_sha256
    assert summary.phase1.path.endswith("phase1/investigation.json")


@pytest.mark.skipif(not (RUN_ROOT / "summary.json").is_file(), reason="requires archived canonical assessment outputs")
def test_rejects_baseline_with_mismatched_decision_hash(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_root = workspace / "outputs/run/investors/charles-hudson/episode-one"
    source = (
        ROOT
        / "outputs/canonical-v4-v41-portfolio-2026-08-15/investors/charles-hudson"
        / "39-this-pitch-is-damn-near-perfect"
    )
    shutil.copytree(source, run_root)
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    summary["episode_slug"] = "episode-one"
    (run_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    state["investigation"]["episode_slug"] = "episode-one"
    state["decision"]["episode_slug"] = "episode-one"
    (run_root / "state.json").write_text(json.dumps(state), encoding="utf-8")
    investigation = json.loads(
        (run_root / "phase1/investigation.json").read_text(encoding="utf-8")
    )
    investigation["episode_slug"] = "episode-one"
    (run_root / "phase1/investigation.json").write_text(
        json.dumps(investigation), encoding="utf-8"
    )
    decision = json.loads(
        (run_root / "phase2/decision.json").read_text(encoding="utf-8")
    )
    decision["episode_slug"] = "episode-one"
    (run_root / "phase2/decision.json").write_text(
        json.dumps(decision), encoding="utf-8"
    )
    evaluation = workspace / "evaluation"
    evaluation.mkdir(parents=True)
    registry = evaluation / "canonical.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "canonical-vc-evaluation-registry-v1",
                "investors": {
                    "charles-hudson": {
                        "display_name": "Charles Hudson",
                        "label_file": "labels.json",
                        "eligible_count": 1,
                        "sources": [
                            {
                                "kind": "summary_glob",
                                "path": "../outputs/run/investors/charles-hudson/*/summary.json",
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frozen artifact hash mismatch"):
        load_canonical_baseline(
            workspace=workspace,
            registry_path=registry,
            canonical_vc_slug="charles-hudson",
            episode_slug="episode-one",
        )


def test_requires_exactly_one_episode() -> None:
    with pytest.raises(ValueError, match="exactly one canonical baseline"):
        load_canonical_baseline(
            workspace=ROOT,
            registry_path=REGISTRY,
            canonical_vc_slug="charles-hudson",
            episode_slug="does-not-exist",
        )
