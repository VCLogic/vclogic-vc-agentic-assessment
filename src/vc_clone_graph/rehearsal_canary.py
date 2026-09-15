"""Leakage-safe construction and scoring for historical rehearsal canaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .precedents import PrecedentCorpus, PrecedentEpisode
from .providers.base import embed_queries


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ObservedQuestion(FrozenModel):
    turn_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    rationale_labels: tuple[str, ...]


class HistoricalRehearsalCanary(FrozenModel):
    schema_version: str = "founder-rehearsal-canary-v1"
    episode_slug: str
    target_vc: str
    pitch_text: str = Field(min_length=1)
    observed_questions: tuple[ObservedQuestion, ...]
    founder_answers: tuple[str, ...]


class QuestionMetrics(FrozenModel):
    semantic_similarity: float = Field(ge=0, le=1)
    rationale_precision: float = Field(ge=0, le=1)
    rationale_recall: float = Field(ge=0, le=1)
    rationale_f1: float = Field(ge=0, le=1)


class RationaleMetrics(FrozenModel):
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    direction_accuracy: float = Field(ge=0, le=1)
    salience_accuracy: float = Field(ge=0, le=1)
    matched_labels: tuple[str, ...]
    missed_labels: tuple[str, ...]
    extra_labels: tuple[str, ...]


class CanaryOperationalMetrics(FrozenModel):
    information_gain: float = Field(ge=0, le=1)
    decision_correct: bool
    schema_repairs: int = Field(ge=0)
    call_count: int = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)


def build_canary(
    episode: PrecedentEpisode,
    *,
    target_vc: str,
    pitch_text: str,
    founder_speakers: Set[str],
    investor_speakers: Set[str],
    observed_question_labels: Mapping[int, Set[str]] | None = None,
) -> HistoricalRehearsalCanary:
    """Extract only target questions and their founder answers, never the outcome."""
    if not pitch_text.strip():
        raise ValueError("canary pitch text must not be empty")
    label_map = observed_question_labels or {}
    target_names = {
        target_vc.casefold().strip(),
        *(alias.casefold().strip() for alias in episode.investor_aliases),
    }
    founder_names = {name.casefold().strip() for name in founder_speakers}
    investor_names = {name.casefold().strip() for name in investor_speakers}
    decision_turns = {
        turn
        for evidence in episode.decision.evidence
        for turn in range(evidence.turn_start, evidence.turn_end + 1)
    }
    questions: list[ObservedQuestion] = []
    answers: list[str] = []
    for position, turn in enumerate(episode.turns):
        if (
            turn.turn_index in decision_turns
            or turn.speaker.casefold().strip() not in target_names
            or "?" not in turn.text
        ):
            continue
        founder_answer: str | None = None
        for candidate in episode.turns[position + 1 :]:
            speaker = candidate.speaker.casefold().strip()
            if speaker in investor_names:
                break
            if speaker in founder_names:
                founder_answer = candidate.text
                break
        if founder_answer is None:
            continue
        questions.append(
            ObservedQuestion(
                turn_index=turn.turn_index,
                text=turn.text,
                rationale_labels=tuple(sorted(label_map.get(turn.turn_index, set()))),
            )
        )
        answers.append(founder_answer)
    return HistoricalRehearsalCanary(
        episode_slug=episode.episode_slug,
        target_vc=target_vc,
        pitch_text=pitch_text,
        observed_questions=tuple(questions),
        founder_answers=tuple(answers),
    )


def _prf(predicted: set[str], observed: set[str]) -> tuple[float, float, float]:
    true_positive = len(predicted & observed)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(observed) if observed else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def score_question(
    *,
    generated: str,
    generated_labels: Set[str],
    observed: str,
    observed_labels: Set[str],
    embedder: Any,
) -> QuestionMetrics:
    vectors = np.asarray(embed_queries(embedder, [generated, observed]), dtype=float)
    if vectors.shape[0] != 2 or vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("question embedding output is invalid")
    denominator = float(np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))
    cosine = float(vectors[0] @ vectors[1] / denominator) if denominator else 0.0
    precision, recall, f1 = _prf(set(generated_labels), set(observed_labels))
    return QuestionMetrics(
        semantic_similarity=min(1.0, max(0.0, cosine)),
        rationale_precision=precision,
        rationale_recall=recall,
        rationale_f1=f1,
    )


def score_rationales(
    *,
    predicted: Sequence[Mapping[str, Any]],
    observed: Sequence[Mapping[str, Any]],
) -> RationaleMetrics:
    predicted_by_label = {
        str(row["taxonomy_label"]): row for row in predicted if row.get("taxonomy_label")
    }
    observed_by_label = {
        str(row["taxonomy_label"]): row for row in observed if row.get("taxonomy_label")
    }
    predicted_labels = set(predicted_by_label)
    observed_labels = set(observed_by_label)
    matched = predicted_labels & observed_labels
    precision, recall, f1 = _prf(predicted_labels, observed_labels)
    direction = (
        sum(
            predicted_by_label[label].get("direction")
            == observed_by_label[label].get("direction")
            for label in matched
        )
        / len(matched)
        if matched
        else 0.0
    )
    salience = (
        sum(
            predicted_by_label[label].get("salience")
            == observed_by_label[label].get("salience")
            for label in matched
        )
        / len(matched)
        if matched
        else 0.0
    )
    return RationaleMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        direction_accuracy=direction,
        salience_accuracy=salience,
        matched_labels=tuple(sorted(matched)),
        missed_labels=tuple(sorted(observed_labels - predicted_labels)),
        extra_labels=tuple(sorted(predicted_labels - observed_labels)),
    )


def operational_metrics(
    *,
    initial_likelihood: float,
    final_likelihood: float,
    predicted_decision: str,
    observed_decision: str,
    schema_repairs: int,
    call_count: int,
    latency_seconds: float,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> CanaryOperationalMetrics:
    return CanaryOperationalMetrics(
        information_gain=min(1.0, abs(final_likelihood - initial_likelihood)),
        decision_correct=predicted_decision == observed_decision,
        schema_repairs=schema_repairs,
        call_count=call_count,
        latency_seconds=latency_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


def canary_precedents(
    corpus: PrecedentCorpus, target_episode_slug: str
) -> PrecedentCorpus:
    """Return a target-excluded corpus and verify the firewall invariant."""
    filtered = corpus.for_target(target_episode_slug)
    if any(
        row.episode_slug == target_episode_slug for row in filtered.list_episodes()
    ):
        raise ValueError("target episode remains accessible in canary precedents")
    return filtered
