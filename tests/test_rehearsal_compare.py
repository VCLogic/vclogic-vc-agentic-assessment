from pathlib import Path

import pytest

from vc_clone_graph.rehearsal_artifacts import RehearsalArtifactStore
from vc_clone_graph.rehearsal_compare import compare_sessions, comparison_csv


def _rationale() -> dict:
    return {
        "rationale_id": "R-001",
        "taxonomy_label": "founder_execution",
        "direction": "positive",
        "salience": "primary",
        "confidence": 0.8,
        "assessment": "Execution is promising.",
        "evidence_refs": [
            {
                "evidence_id": "P-001",
                "source_kind": "pitch",
                "source_path": "pitch.txt",
                "excerpt": "Five pilots.",
            }
        ],
    }


def completed_session(
    root: Path,
    vc: str,
    *,
    decision: str = "In",
    pitch: str = "Same pitch",
    questions: tuple[str, ...] = ("What is retention?",),
) -> Path:
    store = RehearsalArtifactStore.create(root, vc, f"session-{vc}", pitch)
    initial = {
        "decision": "Out",
        "investment_likelihood": 0.4,
        "decision_confidence": 0.6,
        "rationales": [_rationale()],
        "unresolved_questions": ["Retention unknown."],
        "candidate_questions": [],
    }
    final = {
        "decision": decision,
        "investment_likelihood": 0.7 if decision == "In" else 0.2,
        "decision_confidence": 0.7,
        "strongest_positive_rationale_ids": ["R-001"],
        "decisive_concern_rationale_ids": [],
        "unresolved_uncertainties": [],
        "reversal_conditions": ["Evidence changes."],
        "rationale_state": [_rationale()],
        "decision_justification": "The simulated bar is evaluated.",
    }
    store.write_accepted(
        "founder-report.json",
        {
            "session_id": f"session-{vc}",
            "vc_slug": vc,
            "disclosure": "Simulation based on public evidence.",
            "initial_assessment": initial,
            "final_assessment": final,
            "material_answer_ids": [],
            "pitch_improvement_suggestions": ["Quantify retention."],
            "founder_reflection_prompts": [],
            "usage": {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01},
        },
    )
    store.write_accepted(
        "state.json",
        {
            "status": "complete",
            "answers": [
                {"question": question, "text_verbatim": "Founder answer"}
                for question in questions
            ],
        },
    )
    return store.session_root


def test_comparison_requires_two_completed_sessions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="two completed"):
        compare_sessions([completed_session(tmp_path, "charles")])


def test_comparison_preserves_independent_decisions(tmp_path: Path) -> None:
    charles = completed_session(tmp_path, "charles", decision="Out")
    yin = completed_session(
        tmp_path, "yin", decision="In", questions=("How will you distribute?",)
    )

    result = compare_sessions([charles, yin])

    assert [row.decision for row in result.rows] == ["Out", "In"]
    assert result.shared_state_used is False
    assert result.rows[0].questions != result.rows[1].questions


def test_comparison_rejects_different_pitches(tmp_path: Path) -> None:
    first = completed_session(tmp_path, "charles", pitch="Pitch one")
    second = completed_session(tmp_path, "yin", pitch="Pitch two")

    with pytest.raises(ValueError, match="same pitch"):
        compare_sessions([first, second])


def test_comparison_csv_contains_vc_and_decision(tmp_path: Path) -> None:
    result = compare_sessions(
        [completed_session(tmp_path, "charles"), completed_session(tmp_path, "yin")]
    )

    rendered = comparison_csv(result)

    assert "vc_slug,decision" in rendered
    assert "charles,In" in rendered
