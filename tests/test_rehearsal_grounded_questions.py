from types import SimpleNamespace

from vc_clone_graph.rehearsal_grounded_questions import (
    GroundedQuestionCandidate,
    grounded_question_candidates,
    rank_grounded_questions,
)
from vc_clone_graph.rehearsal_schemas import EvidenceRef, RationaleState


def _rationale(
    rationale_id: str,
    label: str,
    confidence: float,
    *,
    salience: str = "primary",
) -> RationaleState:
    return RationaleState(
        rationale_id=rationale_id,
        taxonomy_label=label,
        direction="negative",
        salience=salience,
        confidence=confidence,
        assessment=f"Assessment of {label} and retention.",
        evidence_refs=(
            EvidenceRef(
                evidence_id="P-001",
                source_kind="pitch",
                excerpt="Retention has not yet been measured.",
            ),
        ),
    )


def _baseline(*, controlling: tuple[str, ...], searchable: tuple[str, ...] = ()):
    return SimpleNamespace(
        investigation=SimpleNamespace(
            searchable_questions=list(searchable),
            diligence_questions=[],
            conflicts=[],
            information_sufficient=False,
        ),
        decision=SimpleNamespace(
            controlling_rationale_ids=list(controlling),
            searchable_questions=[],
            diligence_questions=[],
            reversal_conditions=[],
            missing_considerations=[],
        ),
    )


def test_controlling_unresolved_rationale_ranks_first() -> None:
    candidates = grounded_question_candidates(
        baseline=_baseline(
            controlling=("R2",),
            searchable=("What is six-month subscriber retention?",),
        ),
        rationales=(
            _rationale("R1", "market_size_assessment", 0.4),
            _rationale("R2", "traction_repeatability_concern", 0.55),
        ),
        asked_labels=(),
    )

    assert candidates[0].rationale_id == "R2"
    assert candidates[0].origin == "controlling_unresolved"
    assert "retention" in candidates[0].evidence_gap.lower()


def test_classifier_cannot_create_unsupported_candidate() -> None:
    candidates = grounded_question_candidates(
        baseline=_baseline(controlling=("R2",)),
        rationales=(_rationale("R2", "market_size_assessment", 0.7),),
        asked_labels=(),
    )
    ranked = rank_grounded_questions(
        candidates,
        classifier_priorities={
            "corporate_structure_constraint": 0.9,
            "market_size_assessment": 0.1,
        },
    )

    assert {row.taxonomy_label for row in ranked} == {"market_size_assessment"}


def test_classifier_only_breaks_near_ties() -> None:
    clear_winner = GroundedQuestionCandidate(
        rationale_id="R1",
        taxonomy_label="traction",
        evidence_gap="What is retention?",
        origin="controlling_unresolved",
        decision_criticality=1.0,
        information_gap=1.0,
        answerability=1.0,
        investor_relevance=1.0,
        novelty=1.0,
        source_references=("R1",),
    )
    lower = GroundedQuestionCandidate(
        rationale_id="R2",
        taxonomy_label="market",
        evidence_gap="What is the reachable market?",
        origin="low_confidence_primary",
        decision_criticality=0.6,
        information_gap=0.6,
        answerability=0.6,
        investor_relevance=0.6,
        novelty=0.6,
        source_references=("R2",),
    )
    ranked = rank_grounded_questions(
        (clear_winner, lower),
        classifier_priorities={"market": 1.0, "traction": 0.0},
    )

    assert ranked[0].rationale_id == "R1"
    assert ranked[0].primary_score > ranked[1].primary_score


def test_investor_distinctiveness_breaks_a_primary_score_near_tie() -> None:
    generic = GroundedQuestionCandidate(
        rationale_id="R1",
        taxonomy_label="unit_economics_assessment",
        evidence_gap="What is gross margin?",
        origin="controlling_unresolved",
        decision_criticality=0.9,
        information_gap=0.8,
        answerability=0.9,
        investor_relevance=0.8,
        investor_distinctiveness=0.1,
        novelty=1.0,
        source_references=("R1",),
    )
    distinctive = GroundedQuestionCandidate(
        rationale_id="R2",
        taxonomy_label="founder_investor_vision_alignment",
        evidence_gap="What are you building this to become?",
        origin="controlling_unresolved",
        decision_criticality=0.9,
        information_gap=0.8,
        answerability=0.9,
        investor_relevance=0.8,
        investor_distinctiveness=1.0,
        novelty=1.0,
        source_references=("R2",),
    )

    ranked = rank_grounded_questions(
        (generic, distinctive), classifier_priorities={}
    )

    assert ranked[0].taxonomy_label == "founder_investor_vision_alignment"


def test_distinctiveness_cannot_override_material_decision_criticality() -> None:
    decisive = GroundedQuestionCandidate(
        rationale_id="R1",
        taxonomy_label="portfolio_conflict",
        evidence_gap="Does the company overlap an existing investment?",
        origin="controlling_unresolved",
        decision_criticality=1.0,
        information_gap=1.0,
        answerability=0.9,
        investor_relevance=1.0,
        investor_distinctiveness=0.0,
        novelty=1.0,
        source_references=("R1",),
    )
    distinctive = GroundedQuestionCandidate(
        rationale_id="R2",
        taxonomy_label="founder_investor_vision_alignment",
        evidence_gap="What are you building this to become?",
        origin="low_confidence_primary",
        decision_criticality=0.6,
        information_gap=0.6,
        answerability=0.9,
        investor_relevance=0.7,
        investor_distinctiveness=1.0,
        novelty=1.0,
        source_references=("R2",),
    )

    ranked = rank_grounded_questions(
        (distinctive, decisive), classifier_priorities={}
    )

    assert ranked[0].taxonomy_label == "portfolio_conflict"


def test_asked_label_is_not_asked_again() -> None:
    candidates = grounded_question_candidates(
        baseline=_baseline(
            controlling=("R1",),
            searchable=("What is retention?",),
        ),
        rationales=(_rationale("R1", "traction", 0.5),),
        asked_labels=("traction",),
    )

    assert candidates == ()


def test_covered_evidence_gap_is_not_asked_again_under_another_label() -> None:
    rationale = _rationale("R1", "traction_repeatability_concern", 0.5).model_copy(
        update={"assessment": "Outbound conversion repeatability remains unknown."}
    )
    candidates = grounded_question_candidates(
        baseline=_baseline(
            controlling=("R1",),
            searchable=(
                "When outbound is repeated with similar agencies, what conversion pattern appears?",
            ),
        ),
        rationales=(rationale,),
        asked_labels=(),
        covered_evidence_gap_keys=("acquisition_repeatability",),
    )

    assert candidates == ()
