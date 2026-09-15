"""Local behavioral-fidelity and utility metrics for rehearsal questions."""

from __future__ import annotations

import re
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .providers.base import EmbeddingProvider
from .question_memory import atomic_question_findings
from .rehearsal_canary import ObservedQuestion, score_question


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QuestionFidelityRecord(FrozenModel):
    question_id: str
    generated_text: str
    generated_labels: tuple[str, ...]
    nearest_observed_turn: int | None
    nearest_observed_text: str | None
    semantic_similarity: float | None = Field(default=None, ge=0, le=1)
    rationale_precision: float | None = Field(default=None, ge=0, le=1)
    rationale_recall: float | None = Field(default=None, ge=0, le=1)
    rationale_f1: float | None = Field(default=None, ge=0, le=1)
    coarse_dimension_agreement: bool | None
    same_rationale_covered: bool | None
    atomic: bool
    atomicity_findings: tuple[str, ...]
    word_count: int = Field(ge=1)
    redundancy: float = Field(ge=0, le=1)
    redundant: bool


class BehavioralFidelityMetrics(FrozenModel):
    mean_semantic_similarity: float | None = Field(default=None, ge=0, le=1)
    mean_rationale_f1: float | None = Field(default=None, ge=0, le=1)
    observed_rationale_coverage: float | None = Field(default=None, ge=0, le=1)
    generated_rationale_precision: float | None = Field(default=None, ge=0, le=1)
    coarse_dimension_agreement_rate: float | None = Field(default=None, ge=0, le=1)
    ordering_agreement: float | None = Field(default=None, ge=0, le=1)


class QuestionUtilityMetrics(FrozenModel):
    atomic_rate: float = Field(ge=0, le=1)
    length_compliance_rate: float = Field(ge=0, le=1)
    nonredundant_rate: float = Field(ge=0, le=1)
    mean_redundancy: float = Field(ge=0, le=1)
    unique_rationale_dimensions: int = Field(ge=0)


class EpisodeQuestionEvaluation(FrozenModel):
    episode_slug: str
    session_id: str | None = None
    replay_mode: str | None = None
    question_count: int = Field(ge=0)
    observed_question_count: int = Field(ge=0)
    records: tuple[QuestionFidelityRecord, ...]
    behavioral_fidelity: BehavioralFidelityMetrics
    question_utility: QuestionUtilityMetrics


_TOKEN = re.compile(r"[a-z0-9]+")


def _jaccard(left: str, right: str) -> float:
    left_tokens = set(_TOKEN.findall(left.casefold()))
    right_tokens = set(_TOKEN.findall(right.casefold()))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score_generated_question(
    *,
    question_id: str,
    generated_text: str,
    generated_labels: Sequence[str],
    observed: Sequence[ObservedQuestion],
    embedder: EmbeddingProvider,
    prior_generated: Sequence[str],
    taxonomy_parents: Mapping[str, str],
) -> QuestionFidelityRecord:
    """Score one generated question against decision-blind observed questions."""
    labels = tuple(dict.fromkeys(generated_labels))
    candidates: list[tuple[float, float, int, ObservedQuestion, object]] = []
    for row in observed:
        metrics = score_question(
            generated=generated_text,
            generated_labels=set(labels),
            observed=row.text,
            observed_labels=set(row.rationale_labels),
            embedder=embedder,
        )
        candidates.append(
            (
                metrics.semantic_similarity,
                metrics.rationale_f1,
                -row.turn_index,
                row,
                metrics,
            )
        )
    candidates.sort(key=lambda item: item[:3], reverse=True)
    nearest = candidates[0] if candidates else None
    generated_parents = {
        taxonomy_parents[label] for label in labels if label in taxonomy_parents
    }
    if nearest is not None:
        row = nearest[3]
        metrics = nearest[4]
        observed_parents = {
            taxonomy_parents[label]
            for label in row.rationale_labels
            if label in taxonomy_parents
        }
        coarse_agreement = bool(generated_parents & observed_parents)
        same_rationale = bool(set(labels) & set(row.rationale_labels))
    else:
        row = None
        metrics = None
        coarse_agreement = None
        same_rationale = None
    redundancy = max(
        (_jaccard(generated_text, prior) for prior in prior_generated), default=0.0
    )
    findings = atomic_question_findings(generated_text)
    return QuestionFidelityRecord(
        question_id=question_id,
        generated_text=generated_text,
        generated_labels=labels,
        nearest_observed_turn=row.turn_index if row else None,
        nearest_observed_text=row.text if row else None,
        semantic_similarity=metrics.semantic_similarity if metrics else None,
        rationale_precision=metrics.rationale_precision if metrics else None,
        rationale_recall=metrics.rationale_recall if metrics else None,
        rationale_f1=metrics.rationale_f1 if metrics else None,
        coarse_dimension_agreement=coarse_agreement,
        same_rationale_covered=same_rationale,
        atomic=not findings,
        atomicity_findings=findings,
        word_count=len(generated_text.split()),
        redundancy=redundancy,
        redundant=redundancy >= 0.8,
    )


def _ordering_agreement(records: Sequence[QuestionFidelityRecord]) -> float | None:
    turns = [
        row.nearest_observed_turn
        for row in records
        if row.nearest_observed_turn is not None
    ]
    if len(turns) < 2:
        return None
    comparable = 0
    concordant = 0
    for left in range(len(turns)):
        for right in range(left + 1, len(turns)):
            if turns[left] == turns[right]:
                continue
            comparable += 1
            concordant += int(turns[left] < turns[right])
    return concordant / comparable if comparable else None


def aggregate_episode(
    *,
    episode_slug: str,
    records: Sequence[QuestionFidelityRecord],
    observed_questions: Sequence[ObservedQuestion],
    session_id: str | None = None,
    replay_mode: str | None = None,
) -> EpisodeQuestionEvaluation:
    """Aggregate behavioral fidelity separately from deterministic utility."""
    rows = tuple(records)
    observed_labels = {
        label for question in observed_questions for label in question.rationale_labels
    }
    generated_labels = {label for row in rows for label in row.generated_labels}
    covered = observed_labels & generated_labels
    semantic = [
        row.semantic_similarity for row in rows if row.semantic_similarity is not None
    ]
    rationale_f1 = [row.rationale_f1 for row in rows if row.rationale_f1 is not None]
    coarse = [
        row.coarse_dimension_agreement
        for row in rows
        if row.coarse_dimension_agreement is not None
    ]
    count = len(rows)
    behavioral = BehavioralFidelityMetrics(
        mean_semantic_similarity=_mean(semantic),
        mean_rationale_f1=_mean(rationale_f1),
        observed_rationale_coverage=(
            len(covered) / len(observed_labels) if observed_labels else None
        ),
        generated_rationale_precision=(
            len(covered) / len(generated_labels) if generated_labels else None
        ),
        coarse_dimension_agreement_rate=(
            sum(bool(value) for value in coarse) / len(coarse) if coarse else None
        ),
        ordering_agreement=_ordering_agreement(rows),
    )
    utility = QuestionUtilityMetrics(
        atomic_rate=sum(row.atomic for row in rows) / count if count else 0.0,
        length_compliance_rate=(
            sum(row.word_count <= 35 for row in rows) / count if count else 0.0
        ),
        nonredundant_rate=(
            sum(not row.redundant for row in rows) / count if count else 0.0
        ),
        mean_redundancy=(
            sum(row.redundancy for row in rows) / count if count else 0.0
        ),
        unique_rationale_dimensions=len(generated_labels),
    )
    return EpisodeQuestionEvaluation(
        episode_slug=episode_slug,
        session_id=session_id,
        replay_mode=replay_mode,
        question_count=count,
        observed_question_count=len(observed_questions),
        records=rows,
        behavioral_fidelity=behavioral,
        question_utility=utility,
    )


def write_evaluation_outputs(
    *,
    evaluations: Sequence[EpisodeQuestionEvaluation],
    ablations: Sequence[Mapping[str, object]],
    output_root: Path,
) -> None:
    """Write deterministic machine-readable and business-readable evaluation outputs."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "question-records.jsonl").open("w", encoding="utf-8") as stream:
        for evaluation in evaluations:
            for record in evaluation.records:
                payload = {
                    "episode_slug": evaluation.episode_slug,
                    **record.model_dump(mode="json"),
                }
                stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    csv_fields = (
        "episode_slug",
        "session_id",
        "replay_mode",
        "question_count",
        "observed_question_count",
        "mean_semantic_similarity",
        "mean_rationale_f1",
        "observed_rationale_coverage",
        "atomic_rate",
        "nonredundant_rate",
    )
    with (root / "episode-summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in evaluations:
            writer.writerow(
                {
                    "episode_slug": row.episode_slug,
                    "session_id": row.session_id,
                    "replay_mode": row.replay_mode,
                    "question_count": row.question_count,
                    "observed_question_count": row.observed_question_count,
                    "mean_semantic_similarity": row.behavioral_fidelity.mean_semantic_similarity,
                    "mean_rationale_f1": row.behavioral_fidelity.mean_rationale_f1,
                    "observed_rationale_coverage": row.behavioral_fidelity.observed_rationale_coverage,
                    "atomic_rate": row.question_utility.atomic_rate,
                    "nonredundant_rate": row.question_utility.nonredundant_rate,
                }
            )
    def numeric_mean(values: Sequence[float | None]) -> float | None:
        present = [value for value in values if value is not None]
        return sum(present) / len(present) if present else None

    aggregate = {
        "schema_version": "rehearsal-question-fidelity-v1",
        "episode_count": len(evaluations),
        "question_count": sum(row.question_count for row in evaluations),
        "mean_semantic_similarity": numeric_mean(
            [row.behavioral_fidelity.mean_semantic_similarity for row in evaluations]
        ),
        "mean_rationale_f1": numeric_mean(
            [row.behavioral_fidelity.mean_rationale_f1 for row in evaluations]
        ),
        "mean_observed_rationale_coverage": numeric_mean(
            [row.behavioral_fidelity.observed_rationale_coverage for row in evaluations]
        ),
        "mean_atomic_rate": numeric_mean(
            [row.question_utility.atomic_rate for row in evaluations]
        ),
        "generation_provider_calls": 0,
    }
    (root / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "retrieval-ablation.json").write_text(
        json.dumps(list(ablations), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Rehearsal Question Fidelity",
        "",
        f"Evaluated {aggregate['question_count']} generated questions across "
        f"{aggregate['episode_count']} session-episodes. No generation-provider calls were made.",
        "",
        "| Episode | Questions | Observed | Semantic similarity | Rationale F1 | Rationale coverage | Atomic | Nonredundant |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evaluations:
        behavior = row.behavioral_fidelity
        utility = row.question_utility
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"
        lines.append(
            f"| {row.episode_slug} | {row.question_count} | {row.observed_question_count} | "
            f"{fmt(behavior.mean_semantic_similarity)} | {fmt(behavior.mean_rationale_f1)} | "
            f"{fmt(behavior.observed_rationale_coverage)} | {utility.atomic_rate:.3f} | "
            f"{utility.nonredundant_rate:.3f} |"
        )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
