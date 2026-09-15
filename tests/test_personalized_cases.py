from __future__ import annotations

import json
from pathlib import Path

import pytest

from vc_clone_graph.personalized_cases import (
    ArtifactViews,
    InvestorInput,
    PackageViews,
    assemble_personalized_cases,
    render_wiki,
    verify_episode_pitch_assets,
)
from vc_clone_graph.phase1_evaluation import (
    Phase1Case,
    PredictedRationale,
    ReferenceRationale,
)
from vc_clone_graph.phase2_calibration_evaluation import Phase2CalibrationRecord
from vc_clone_graph.posthoc_semantic_models import extract_phase2_reasoning_view


def phase2_record(*, vc: str = "alpha", episode: str = "1-example") -> Phase2CalibrationRecord:
    return Phase2CalibrationRecord(
        vc_slug=vc,
        vc_name="Alpha Investor",
        episode_slug=episode,
        group=episode,
        target=1,
        raw_decision=0,
        raw_likelihood=0.4,
        phase2_features={"decision_in": 0.0, "investment_likelihood": 0.4},
        combined_features={"rationale_count": 1.0},
        semantic_features={"rationale__market__signed_confidence": 0.7},
        artifact_root=f"outputs/{vc}/{episode}",
        phase1_sha256="a" * 64,
        phase2_sha256="b" * 64,
    )


def phase1_case(*, vc: str = "alpha", episode: str = "1-example") -> Phase1Case:
    return Phase1Case(
        vc_slug=vc,
        vc_name="Alpha Investor",
        episode_slug=episode,
        actual_decision="In",
        source_tier="newly_extracted",
        source_format="transcript-observed-rationales-v1",
        predicted=(PredictedRationale(
            label="market", direction="positive", salience="primary",
            confidence=0.7, justification="Large market.",
        ),),
        reference=(ReferenceRationale(
            label="market", direction="positive", salience="primary",
            confidence=0.9, activation="explicit", utterance_type="assessment",
            decision_link="supports", evidence=("A large market matters.",),
        ),),
        artifact_path=None,
        reference_path=None,
    )


def artifact_views(*, episode: str = "1-example") -> ArtifactViews:
    return ArtifactViews(
        episode_slug=episode,
        phase1_text="rationale market positive",
        phase2_text="decision Out likelihood 0.4",
        phase1_features={"rationale__market__signed_confidence": 0.7},
        phase2_features={"decision_in": 0.0, "investment_likelihood": 0.4},
        phase1_sha256="a" * 64,
        phase2_sha256="b" * 64,
    )


def package_views(*, episode: str = "1-example") -> PackageViews:
    return PackageViews(
        vc_input_slug="alpha-fund",
        episode_slug=episode,
        pitch_text="Founder: We have ten paying customers.",
        wiki_text="# thesis\nWe invest early.",
        pitch_sha256="c" * 64,
        wiki_sha256="d" * 64,
    )


def test_assembles_aligned_case_without_exposing_reference_text() -> None:
    record = phase2_record()
    result = assemble_personalized_cases(
        [record],
        phase1_by_key={("alpha", "1-example"): phase1_case()},
        investor_by_name={
            "Alpha Investor": InvestorInput(
                vc_input_slug="alpha-fund", display_name="Alpha Investor"
            )
        },
        artifact_by_key={("alpha", "1-example"): artifact_views()},
        package_by_key={("alpha", "1-example"): package_views()},
    )

    assert len(result) == 1
    case = result[0]
    assert case.vc_slug == "alpha"
    assert case.vc_input_slug == "alpha-fund"
    assert case.target == 1
    assert case.pitch_text.startswith("Founder:")
    assert case.actual_rationale_targets == {"market": 1}
    assert case.actual_rationale_available is True
    assert "A large market matters" not in case.phase1_text
    assert case.source_hashes["pitch"] == "c" * 64


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("decision", "decision"),
        ("phase1_hash", "Phase 1"),
        ("episode", "episode"),
        ("profile", "investor"),
    ],
)
def test_alignment_fails_closed(change: str, message: str) -> None:
    record = phase2_record()
    rationale = phase1_case()
    artifact = artifact_views()
    package = package_views()
    profiles = {"Alpha Investor": InvestorInput("alpha-fund", "Alpha Investor")}
    if change == "decision":
        rationale = Phase1Case(**{**rationale.__dict__, "actual_decision": "Out"})
    elif change == "phase1_hash":
        artifact = ArtifactViews(**{**artifact.__dict__, "phase1_sha256": "f" * 64})
    elif change == "episode":
        package = package_views(episode="2-wrong")
    else:
        profiles = {}

    with pytest.raises(ValueError, match=message):
        assemble_personalized_cases(
            [record],
            phase1_by_key={("alpha", "1-example"): rationale},
            investor_by_name=profiles,
            artifact_by_key={("alpha", "1-example"): artifact},
            package_by_key={("alpha", "1-example"): package},
        )


def test_duplicate_investor_episode_is_rejected() -> None:
    record = phase2_record()
    with pytest.raises(ValueError, match="duplicate"):
        assemble_personalized_cases(
            [record, record],
            phase1_by_key={("alpha", "1-example"): phase1_case()},
            investor_by_name={"Alpha Investor": InvestorInput("alpha-fund", "Alpha Investor")},
            artifact_by_key={("alpha", "1-example"): artifact_views()},
            package_by_key={("alpha", "1-example"): package_views()},
        )


def test_render_wiki_is_sorted_and_source_linked(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    (wiki / "evidence").mkdir(parents=True)
    (wiki / "z.md").write_text("Last", encoding="utf-8")
    (wiki / "a.md").write_text("First", encoding="utf-8")
    (wiki / "evidence/b.md").write_text("Middle", encoding="utf-8")
    (wiki / "index.json").write_text("{}", encoding="utf-8")

    text, digest = render_wiki(wiki)

    assert text.index("a.md") < text.index("evidence/b.md") < text.index("z.md")
    assert "index.json" not in text
    assert len(digest) == 64


def test_phase2_reasoning_view_excludes_phase1_and_preserves_decision_fields() -> None:
    decision = {
        "decision": "Out",
        "decision_path": "standard_check",
        "investment_likelihood": 0.4,
        "decision_confidence": 0.7,
        "decision_justification": "Market evidence is weak.",
        "evidence_basis": [],
        "strongest_opposing_case": {"argument": "Strong founder", "response": "Not enough"},
        "founder_exception_assessment": "Not activated",
        "sufficiency_assessment": "Enough for pitch decision",
        "reversal_conditions": ["More traction"],
    }

    text = extract_phase2_reasoning_view(decision)

    assert "agent decision: Out" in text
    assert "investment likelihood: 0.4" in text
    assert "Market evidence is weak" in text


def test_fast_episode_verification_hashes_pitch_and_requires_clean_audit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "inputs"
    vc = "alpha-fund"
    episode = "2-second"
    pitch = root / f"data/investors/{vc}/pitches/{episode}.txt"
    audit = root / f"data/investors/{vc}/audits/{episode}.json"
    manifest = root / f"data/investors/{vc}/manifests/{episode}.json"
    pitch.parent.mkdir(parents=True)
    audit.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    pitch.write_text("Founder: safe pitch", encoding="utf-8")
    from hashlib import sha256
    pitch_digest = sha256(pitch.read_bytes()).hexdigest()
    audit.write_text(json.dumps({
        "status": "audited",
        "vc_slug": vc,
        "episode_slug": episode,
        "pitch_sha256": pitch_digest,
        "leakage_checklist": {"actual_label_in_package": False},
    }), encoding="utf-8")
    audit_digest = sha256(audit.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({
        "vc_slug": vc,
        "episode_slug": episode,
        "files": [
            {"destination": pitch.relative_to(root).as_posix(), "sha256": pitch_digest},
            {"destination": audit.relative_to(root).as_posix(), "sha256": audit_digest},
        ],
    }), encoding="utf-8")

    verified = verify_episode_pitch_assets(root, vc, episode)

    assert verified == (pitch.resolve(), pitch_digest)
    pitch.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        verify_episode_pitch_assets(root, vc, episode)
