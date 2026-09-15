"""Question candidates derived only from a verified canonical decision state."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Literal, Mapping, Sequence

from .rehearsal_grounding import CanonicalRehearsalBaseline
from .rehearsal_question_policy import order_with_bounded_tiebreak
from .question_memory import question_evidence_gap_key
from .rehearsal_schemas import RationaleState


QuestionOrigin = Literal[
    "controlling_unresolved",
    "constraint",
    "low_confidence_primary",
    "conflict",
    "reversal_condition",
    "missing_consideration",
]


@dataclass(frozen=True)
class GroundedQuestionCandidate:
    rationale_id: str
    taxonomy_label: str
    evidence_gap: str
    origin: QuestionOrigin
    decision_criticality: float
    information_gap: float
    answerability: float
    investor_relevance: float
    novelty: float
    source_references: tuple[str, ...]
    investor_distinctiveness: float = 0.0
    primary_score: float = 0.0
    classifier_tiebreaker_score: float = 0.0

    def __post_init__(self) -> None:
        for field in (
            "decision_criticality",
            "information_gap",
            "answerability",
            "investor_relevance",
            "novelty",
            "investor_distinctiveness",
            "primary_score",
            "classifier_tiebreaker_score",
        ):
            value = getattr(self, field)
            if not 0 <= value <= 1:
                raise ValueError(f"{field} must be between zero and one")


_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "does",
    "for", "from", "has", "have", "how", "in", "is", "it", "of", "on",
    "or", "that", "the", "their", "this", "to", "what", "which", "with",
}


def _terms(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _STOP}


def _question_for(rationale: RationaleState, questions: Sequence[str]) -> str:
    rationale_terms = _terms(
        f"{rationale.taxonomy_label.replace('_', ' ')} {rationale.assessment}"
    )
    ranked = sorted(
        questions,
        key=lambda question: (
            -len(rationale_terms & _terms(question)),
            questions.index(question),
        ),
    )
    if ranked and rationale_terms & _terms(ranked[0]):
        return ranked[0]
    return (
        "What additional founder evidence would resolve the uncertainty in "
        f"{rationale.taxonomy_label.replace('_', ' ')}?"
    )


def grounded_question_candidates(
    *,
    baseline: CanonicalRehearsalBaseline,
    rationales: Sequence[RationaleState],
    asked_labels: Sequence[str],
    covered_evidence_gap_keys: Sequence[str] = (),
) -> tuple[GroundedQuestionCandidate, ...]:
    """Derive candidates from existing rationale IDs, never the full taxonomy."""
    asked = set(asked_labels)
    covered = set(covered_evidence_gap_keys)
    controlling = set(baseline.decision.controlling_rationale_ids)
    questions = tuple(
        dict.fromkeys(
            (
                *baseline.investigation.searchable_questions,
                *baseline.investigation.diligence_questions,
                *baseline.decision.searchable_questions,
                *baseline.decision.diligence_questions,
            )
        )
    )
    candidates: list[GroundedQuestionCandidate] = []
    for rationale in rationales:
        if rationale.taxonomy_label in asked:
            continue
        if rationale.rationale_id in controlling:
            origin: QuestionOrigin = "controlling_unresolved"
            criticality = 1.0
        elif rationale.salience == "primary" and rationale.confidence < 0.7:
            origin = "low_confidence_primary"
            criticality = 0.72
        else:
            continue
        gap = _question_for(rationale, questions)
        if question_evidence_gap_key(gap, (rationale.taxonomy_label,)) in covered:
            continue
        investor_evidence_ids = tuple(
            evidence.evidence_id
            for evidence in rationale.evidence_refs
            if evidence.source_kind in {"wiki", "precedent", "portfolio"}
        )
        candidates.append(
            GroundedQuestionCandidate(
                rationale_id=rationale.rationale_id,
                taxonomy_label=rationale.taxonomy_label,
                evidence_gap=gap,
                origin=origin,
                decision_criticality=criticality,
                information_gap=max(0.35, 1.0 - rationale.confidence),
                answerability=0.9 if gap in questions else 0.7,
                investor_relevance=1.0 if rationale.salience == "primary" else 0.7,
                novelty=1.0,
                source_references=(
                    rationale.rationale_id,
                    *investor_evidence_ids,
                ),
                investor_distinctiveness=min(
                    1.0,
                    len(investor_evidence_ids) / 2.0,
                ),
            )
        )
    return rank_grounded_questions(candidates, classifier_priorities={})


def _primary_score(candidate: GroundedQuestionCandidate) -> float:
    return (
        0.33 * candidate.decision_criticality
        + 0.24 * candidate.information_gap
        + 0.18 * candidate.answerability
        + 0.12 * candidate.investor_relevance
        + 0.08 * candidate.investor_distinctiveness
        + 0.05 * candidate.novelty
    )


def rank_grounded_questions(
    candidates: Sequence[GroundedQuestionCandidate],
    *,
    classifier_priorities: Mapping[str, float],
    tie_tolerance: float = 0.05,
) -> tuple[GroundedQuestionCandidate, ...]:
    """Rank grounded gaps; classifier information is a bounded tie-breaker only."""
    scored = tuple(
        replace(
            candidate,
            primary_score=_primary_score(candidate),
            classifier_tiebreaker_score=max(
                0.0,
                min(1.0, float(classifier_priorities.get(candidate.taxonomy_label, 0.0))),
            ),
        )
        for candidate in candidates
    )
    return order_with_bounded_tiebreak(
        scored,
        primary_score=lambda row: row.primary_score,
        tiebreaker_score=lambda row: row.classifier_tiebreaker_score,
        stable_key=lambda row: row.rationale_id,
        tolerance=tie_tolerance,
    )
