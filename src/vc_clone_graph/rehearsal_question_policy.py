"""Private counterfactual policy for selecting founder-rehearsal questions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence, TypeVar

from .rehearsal_classifier import LinearClassifierArtifact
from .rehearsal_features import counterfactual_feature_maps


_RankedT = TypeVar("_RankedT")


def order_with_bounded_tiebreak(
    rows: Sequence[_RankedT],
    *,
    primary_score: Callable[[_RankedT], float],
    tiebreaker_score: Callable[[_RankedT], float],
    stable_key: Callable[[_RankedT], str],
    tolerance: float = 0.05,
) -> tuple[_RankedT, ...]:
    """Apply a secondary ordering only inside primary-score near-tie bands."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    ordered = sorted(rows, key=lambda row: (-primary_score(row), stable_key(row)))
    result: list[_RankedT] = []
    index = 0
    while index < len(ordered):
        band_top = primary_score(ordered[index])
        end = index + 1
        while end < len(ordered) and (
            band_top - primary_score(ordered[end]) <= tolerance
        ):
            end += 1
        result.extend(
            sorted(
                ordered[index:end],
                key=lambda row: (-tiebreaker_score(row), stable_key(row)),
            )
        )
        index = end
    return tuple(result)


@dataclass(frozen=True)
class QuestionCandidate:
    rationale_label: str
    evidence_gap: str
    question_archetype: str
    answerability: float = 1.0
    investor_relevance: float = 1.0
    novelty: float = 1.0

    def __post_init__(self) -> None:
        for name in ("answerability", "investor_relevance", "novelty"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True)
class QuestionPriority:
    rationale_label: str
    evidence_gap: str
    question_archetype: str
    current_probability: float
    positive_probability: float
    negative_probability: float
    probability_impact: float
    question_score: float
    below_impact_threshold: bool

    def public_guidance(self) -> dict[str, str]:
        """Return only interview-safe guidance for question generation."""
        return {
            "rationale_label": self.rationale_label,
            "evidence_gap": self.evidence_gap,
            "question_archetype": self.question_archetype,
            "impact_band": (
                "low"
                if self.probability_impact < 0.05
                else "medium"
                if self.probability_impact < 0.15
                else "high"
            ),
        }


@dataclass(frozen=True)
class InterviewContinuation:
    continue_interview: bool
    reason: Literal[
        "minimum_diagnostic_question",
        "founder_requested",
        "question_cap",
        "information_sufficient",
        "low_expected_impact",
        "material_question_available",
    ]


def rank_counterfactual_questions(
    artifact: LinearClassifierArtifact,
    base_features: Mapping[str, float],
    candidates: Sequence[QuestionCandidate],
    *,
    minimum_probability_impact: float = 0.05,
) -> tuple[QuestionPriority, ...]:
    """Rank unresolved rationales by bounded, weighted prediction movement."""
    current = artifact.score(base_features).probability_in
    priorities: list[QuestionPriority] = []
    for candidate in candidates:
        positive_features, negative_features = counterfactual_feature_maps(
            base_features, candidate.rationale_label
        )
        positive = artifact.score(positive_features).probability_in
        negative = artifact.score(negative_features).probability_in
        impact = max(abs(positive - current), abs(negative - current))
        score = (
            impact
            * candidate.answerability
            * candidate.investor_relevance
            * candidate.novelty
        )
        priorities.append(
            QuestionPriority(
                rationale_label=candidate.rationale_label,
                evidence_gap=candidate.evidence_gap,
                question_archetype=candidate.question_archetype,
                current_probability=current,
                positive_probability=positive,
                negative_probability=negative,
                probability_impact=impact,
                question_score=score,
                below_impact_threshold=impact < minimum_probability_impact,
            )
        )
    return tuple(
        sorted(
            priorities,
            key=lambda row: (-row.question_score, row.rationale_label),
        )
    )


def should_continue_interview(
    *,
    question_count: int,
    minimum_questions: int,
    maximum_questions: int,
    founder_requested_finish: bool,
    information_sufficient: bool,
    highest_probability_impact: float,
    minimum_probability_impact: float,
) -> InterviewContinuation:
    if founder_requested_finish:
        return InterviewContinuation(False, "founder_requested")
    if question_count >= maximum_questions:
        return InterviewContinuation(False, "question_cap")
    if question_count < minimum_questions:
        return InterviewContinuation(True, "minimum_diagnostic_question")
    if information_sufficient:
        return InterviewContinuation(False, "information_sufficient")
    if highest_probability_impact < minimum_probability_impact:
        return InterviewContinuation(False, "low_expected_impact")
    return InterviewContinuation(True, "material_question_available")
