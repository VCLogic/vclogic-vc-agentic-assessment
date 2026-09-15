"""Deterministic safeguards against already-answered rehearsal questions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Sequence


CoverageReason = Literal[
    "answer_already_present",
    "prior_answer_already_present",
    "unresolved_information_request",
]


@dataclass(frozen=True)
class EvidenceStatement:
    evidence_id: str
    text: str
    rationale_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuestionCoverage:
    redundant: bool
    reason: CoverageReason
    supporting_evidence_ids: tuple[str, ...] = ()


_WORD = re.compile(r"[a-z0-9]+")
_POLAR_OPENERS = {
    "am", "are", "can", "could", "did", "do", "does", "has", "have",
    "is", "may", "should", "was", "were", "will", "would",
}
_NON_ANSWER_MARKERS = {
    "not", "none", "unknown", "unavailable", "unproven", "unvalidated",
    "unverified", "neither", "never", "no",
}
_STOP = {
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "being",
    "beyond", "by", "did", "do", "does", "for", "from", "has", "have",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "what", "when", "which", "who", "with", "your",
}
_ALIASES = {
    "agreed": "payment",
    "agree": "payment",
    "pay": "payment",
    "paying": "payment",
    "paid": "payment",
    "willingness": "payment",
    "validated": "proof",
    "validation": "proof",
    "verified": "proof",
    "verification": "proof",
    "demonstrated": "proof",
    "proven": "proof",
    "expansion": "module",
}


def _stem(word: str) -> str:
    if len(word) > 5 and word.endswith("ies"):
        return f"{word[:-3]}y"
    for suffix in ("ments", "ment", "ations", "ation", "ing", "ers", "ed", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw in _WORD.findall(text.casefold()):
        if raw in _STOP:
            continue
        normalized = _ALIASES.get(raw, raw)
        terms.add(_stem(normalized))
    return terms


def _normalized(text: str) -> str:
    return " ".join(_WORD.findall(text.casefold()))


def _is_polar_question(question: str) -> bool:
    words = _WORD.findall(question.casefold())
    return bool(words and words[0] in _POLAR_OPENERS)


def _is_non_answer(text: str) -> bool:
    words = set(_WORD.findall(text.casefold()))
    return bool(words & _NON_ANSWER_MARKERS) or "not yet" in text.casefold()


def _same_fact(
    question_terms: set[str],
    statement: EvidenceStatement,
    rationale_labels: set[str],
) -> bool:
    statement_terms = _terms(statement.text)
    overlap = question_terms & statement_terms
    if len(overlap) < 2:
        return False
    aligned = bool(rationale_labels & set(statement.rationale_labels))
    denominator = min(len(question_terms), len(statement_terms)) or 1
    return len(overlap) / denominator >= (0.28 if aligned else 0.5)


def assess_question_coverage(
    *,
    question: str,
    rationale_labels: Sequence[str],
    evidence: Sequence[EvidenceStatement],
    prior_answers: Sequence[EvidenceStatement],
) -> QuestionCoverage:
    """Determine whether asking would only recover an answer already supplied.

    A pitch's explicit absence of proof answers a yes/no question but does not
    prevent a founder from supplying a newly measured value in response to an
    open information request.
    """
    question_terms = _terms(question)
    labels = set(rationale_labels)
    for statement in prior_answers:
        if _same_fact(question_terms, statement, labels):
            repeated_question = _normalized(question) in _normalized(statement.text)
            if (
                _is_non_answer(statement.text)
                and not _is_polar_question(question)
                and not repeated_question
            ):
                continue
            return QuestionCoverage(
                redundant=True,
                reason="prior_answer_already_present",
                supporting_evidence_ids=(statement.evidence_id,),
            )
    for statement in evidence:
        if not _same_fact(question_terms, statement, labels):
            continue
        if _is_non_answer(statement.text) and not _is_polar_question(question):
            continue
        return QuestionCoverage(
            redundant=True,
            reason="answer_already_present",
            supporting_evidence_ids=(statement.evidence_id,),
        )
    return QuestionCoverage(
        redundant=False,
        reason="unresolved_information_request",
    )
