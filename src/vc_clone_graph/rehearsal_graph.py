"""Interruptible LangGraph workflow for founder pitch rehearsals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, TypedDict, TypeVar

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ValidationError

from .prompts_v4 import pitch_evidence_index
from .providers.base import GenerationProvider, GenerationRequest, GenerationResult
from .question_memory import (
    atomic_question_findings,
    question_evidence_gap_key,
    spoken_question_findings,
)
from .rehearsal_artifacts import RehearsalArtifactStore
from .rehearsal_classifier import LinearClassifierArtifact
from .rehearsal_effects import enforce_evidence_effect_delta
from .rehearsal_features import rehearsal_feature_map
from .rehearsal_prompts import (
    atomic_question_repair_prompt,
    continuation_prompt,
    final_assessment_prompt,
    founder_report_prompt,
    grounded_answer_update_prompt,
    initial_assessment_prompt,
    repair_prompt,
    select_question_prompt,
    question_coverage_repair_prompt,
    update_prompt,
)
from .rehearsal_schemas import (
    AnswerUpdate,
    CoachingOutput,
    ClassificationSnapshot,
    ClassificationSummary,
    ContinuationDecision,
    FinalAssessment,
    FounderAnswerEvidence,
    FounderReport,
    GroundedAnswerUpdate,
    GroundedDecisionComparison,
    GroundedRationaleChange,
    InitialAssessment,
    RehearsalQuestion,
    normalize_grounded_no_change_update,
    validate_rationale_state,
    validate_grounded_update_context,
    validate_taxonomy_labels,
    normalize_final_score_reconciliation,
)
from .rehearsal_decision_policy import calibrate_commitment_amounts
from .rehearsal_grounding import (
    CanonicalRehearsalBaseline,
    baseline_initial_assessment,
    baseline_summary,
)
from .rehearsal_grounded_questions import (
    grounded_question_candidates,
    rank_grounded_questions,
)
from .rehearsal_phase2_bridge import GroundedPhase2Synthesizer
from .rehearsal_question_policy import (
    QuestionCandidate,
    rank_counterfactual_questions,
    should_continue_interview,
)
from .rehearsal_question_coverage import (
    EvidenceStatement,
    assess_question_coverage,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class RehearsalEvidenceBundle:
    wiki_evidence: tuple[dict[str, Any], ...]
    precedent_evidence: tuple[dict[str, Any], ...]
    portfolio_evidence: tuple[dict[str, Any], ...]

    def all_evidence(self) -> tuple[dict[str, Any], ...]:
        return (
            *self.wiki_evidence,
            *self.precedent_evidence,
            *self.portfolio_evidence,
        )


class RehearsalRetriever(Protocol):
    def retrieve(self, query: str) -> RehearsalEvidenceBundle: ...


class RehearsalState(TypedDict, total=False):
    session_id: str
    vc_slug: str
    investor_name: str
    pitch: str
    pitch_sha256: str
    pitch_evidence: list[dict[str, Any]]
    taxonomy: list[dict[str, str]]
    evidence_registry: dict[str, dict[str, Any]]
    initial_assessment: dict[str, Any]
    current_rationales: list[dict[str, Any]]
    current_likelihood: float
    current_confidence: float
    unresolved_questions: list[str]
    current_question: dict[str, Any]
    pending_answer: dict[str, Any]
    pending_update: dict[str, Any]
    answers: list[dict[str, Any]]
    updates: list[dict[str, Any]]
    question_count: int
    founder_requested_finish: bool
    continue_interview: bool
    stopping_reason: str
    status: str
    findings: list[str]
    final_assessment: dict[str, Any]
    founder_report: dict[str, Any]
    usage: dict[str, int | float]
    usage_by_phase: dict[str, dict[str, int | float]]
    classification_status: str
    classification_fallback_reason: str
    classification_snapshots: list[dict[str, Any]]
    current_features: dict[str, float]
    question_priorities: list[dict[str, Any]]
    asked_classifier_labels: list[str]
    covered_evidence_gap_keys: list[str]
    pending_update_normalization: list[str]
    grounded_changes: list[dict[str, Any]]
    grounded_final_status: str
    grounded_final_artifact_path: str
    grounded_final_artifact_sha256: str
    asked_distinctive_question: bool


class ResumableModelError(RuntimeError):
    pass


ModelT = TypeVar("ModelT", bound=BaseModel)


class RehearsalWorkflow:
    """Run an investor-specific interview while keeping interim judgment private."""

    def __init__(
        self,
        *,
        provider: GenerationProvider,
        checkpointer: Any,
        store: RehearsalArtifactStore,
        investor_name: str,
        taxonomy: Sequence[dict[str, str]],
        retriever: RehearsalRetriever,
        max_questions: int = 8,
        max_output_tokens: int = 8192,
        reasoning_effort: str | None = None,
        repair_attempts: int = 1,
        classification_mode: str = "rationale_only",
        classifier_artifact: LinearClassifierArtifact | None = None,
        classifier_fallback_reason: str | None = None,
        minimum_questions: int = 1,
        minimum_probability_impact: float = 0.05,
        classification_maximum_questions: int | None = None,
        grounded_baseline: CanonicalRehearsalBaseline | None = None,
        phase2_synthesizer: GroundedPhase2Synthesizer | None = None,
        classifier_tiebreaker: bool = True,
        progress_callback: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        if not 1 <= max_questions <= 8:
            raise ValueError("max_questions must be between 1 and 8")
        if repair_attempts not in {0, 1}:
            raise ValueError("repair_attempts must be zero or one")
        if classification_mode not in {
            "classification_informed",
            "rationale_only",
            "v41_grounded",
        }:
            raise ValueError("unknown rehearsal classification mode")
        if classification_mode == "v41_grounded" and grounded_baseline is None:
            raise ValueError("v41_grounded rehearsal requires a canonical baseline")
        if classification_mode == "v41_grounded" and phase2_synthesizer is None:
            raise ValueError("v41_grounded rehearsal requires a Phase 2 synthesizer")
        classification_cap = classification_maximum_questions or max_questions
        if not 0 <= minimum_questions <= classification_cap <= max_questions:
            raise ValueError("classification question bounds are invalid")
        if not 0 <= minimum_probability_impact <= 1:
            raise ValueError("minimum_probability_impact must be between zero and one")
        self.provider = provider
        self.store = store
        self.investor_name = investor_name
        self.taxonomy = tuple(taxonomy)
        self.taxonomy_labels = {row["label"] for row in self.taxonomy}
        self.retriever = retriever
        self.max_questions = max_questions
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.repair_attempts = repair_attempts
        self.classification_mode = classification_mode
        self.classifier_artifact = classifier_artifact
        self.classifier_fallback_reason = classifier_fallback_reason
        self.minimum_questions = minimum_questions
        self.minimum_probability_impact = minimum_probability_impact
        self.classification_maximum_questions = classification_cap
        self.grounded_baseline = grounded_baseline
        self.phase2_synthesizer = phase2_synthesizer
        self.classifier_tiebreaker = classifier_tiebreaker
        self.progress_callback = progress_callback
        self._call_counts = self._existing_call_counts()
        self.graph = self._build().compile(checkpointer=checkpointer)

    def _existing_call_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        calls = self.store.session_root / "calls"
        if not calls.is_dir():
            return counts
        for phase in calls.iterdir():
            if phase.is_dir():
                counts[phase.name] = len(tuple(phase.glob("call-*.json")))
        return counts

    def _next_call(self, phase: str) -> int:
        value = self._call_counts.get(phase, 0) + 1
        self._call_counts[phase] = value
        return value

    def _generate(
        self,
        phase: str,
        prompt: str,
        model_type: type[ModelT],
    ) -> tuple[ModelT, list[GenerationResult]]:
        public_stages = {
            "answer_update": (
                "answer_update_running",
                "Reviewing your evidence against the active rationales",
            ),
            "continuation": (
                "continuation_running",
                "Deciding whether another question could change the assessment",
            ),
            "select_question": (
                "question_selection_running",
                "Selecting the next investor-style question",
            ),
            "final_assessment": (
                "decision_synthesis_running",
                "Synthesizing the investment decision",
            ),
            "founder_report": (
                "founder_feedback_running",
                "Preparing founder feedback",
            ),
        }
        public_stage = public_stages.get(phase)
        if public_stage is not None and self.progress_callback is not None:
            stage, message = public_stage
            self.progress_callback(stage, {"message": message})
        results: list[GenerationResult] = []
        current_prompt = prompt
        attempts = 1 + self.repair_attempts
        last_error = "model output was invalid"
        for attempt in range(attempts):
            request_phase = phase if attempt == 0 else f"{phase}_repair"
            request = GenerationRequest(
                phase=request_phase,
                prompt=current_prompt,
                schema=model_type.model_json_schema(),
                max_output_tokens=self.max_output_tokens,
                reasoning_effort=self.reasoning_effort,  # type: ignore[arg-type]
            )
            result = self.provider.generate(request)
            results.append(result)
            sequence = self._next_call(request_phase)
            self.store.write_call(
                request_phase,
                sequence,
                {
                    "phase": request.phase,
                    "prompt": request.prompt,
                    "schema": request.schema,
                    "max_output_tokens": request.max_output_tokens,
                    "reasoning_effort": request.reasoning_effort,
                },
                {
                    "parsed": result.parsed,
                    "content": result.content,
                    "usage": result.usage.model_dump(mode="json"),
                    "elapsed_seconds": result.elapsed_seconds,
                    "raw_metadata": result.raw_metadata,
                },
            )
            try:
                model = model_type.model_validate(result.parsed)
            except ValidationError as exc:
                last_error = str(exc)
                if attempt + 1 < attempts:
                    current_prompt = repair_prompt(
                        original_prompt=prompt,
                        invalid_output=result.content,
                        error=last_error,
                    )
                    continue
                self.store.write_failed_call(
                    phase,
                    self._next_call(f"{phase}-failure"),
                    {"content": result.content, "parsed": result.parsed},
                    last_error,
                )
                raise ResumableModelError(f"{phase} output remained invalid after repair") from exc
            return model, results
        raise ResumableModelError(last_error)  # pragma: no cover

    @staticmethod
    def _zero_usage() -> dict[str, int | float]:
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }

    def _with_usage(
        self,
        state: RehearsalState,
        phase: str,
        results: Sequence[GenerationResult],
    ) -> dict[str, Any]:
        total = dict(state.get("usage", self._zero_usage()))
        by_phase = {
            key: dict(value) for key, value in state.get("usage_by_phase", {}).items()
        }
        phase_usage = dict(by_phase.get(phase, self._zero_usage()))
        for result in results:
            usage = result.usage.model_dump(mode="json")
            for key in self._zero_usage():
                total[key] = total.get(key, 0) + usage.get(key, 0)
                phase_usage[key] = phase_usage.get(key, 0) + usage.get(key, 0)
        by_phase[phase] = phase_usage
        return {"usage": total, "usage_by_phase": by_phase}

    @staticmethod
    def _registry_add(
        registry: dict[str, dict[str, Any]], rows: Sequence[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        updated = dict(registry)
        for row in rows:
            identifier = str(row["evidence_id"])
            if identifier in updated and updated[identifier] != row:
                raise ValueError(f"evidence ID collision: {identifier}")
            updated[identifier] = dict(row)
        return updated

    def _ingest_pitch(self, state: RehearsalState) -> RehearsalState:
        pitch_rows = [
            {
                "evidence_id": row["evidence_id"],
                "source_kind": "pitch",
                "source_path": "pitch.txt",
                "excerpt": row["text"],
                "text": row["text"],
            }
            for row in pitch_evidence_index(state["pitch"])
        ]
        return {
            "pitch_evidence": pitch_rows,
            "evidence_registry": {row["evidence_id"]: row for row in pitch_rows},
            "answers": [],
            "updates": [],
            "question_count": 0,
            "founder_requested_finish": False,
            "findings": [],
            "asked_distinctive_question": False,
            "status": "running",
            "usage": self._zero_usage(),
            "usage_by_phase": {},
        }

    def _initial_assessment(self, state: RehearsalState) -> RehearsalState:
        bundle = self.retriever.retrieve(state["pitch"])
        registry = self._registry_add(
            state["evidence_registry"], bundle.all_evidence()
        )
        prompt = initial_assessment_prompt(
            investor_name=self.investor_name,
            pitch_evidence=state["pitch_evidence"],
            taxonomy=self.taxonomy,
            wiki_evidence=bundle.wiki_evidence,
            precedent_evidence=bundle.precedent_evidence,
            portfolio_evidence=bundle.portfolio_evidence,
            max_questions=self.max_questions,
        )
        assessment, calls = self._generate(
            "initial_assessment", prompt, InitialAssessment
        )
        validate_rationale_state(assessment.rationales, self.taxonomy_labels)
        for question in assessment.candidate_questions:
            validate_taxonomy_labels(question.rationale_labels, self.taxonomy_labels)
        payload = assessment.model_dump(mode="json")
        self.store.write_accepted("initial-assessment.json", payload)
        return {
            "evidence_registry": registry,
            "initial_assessment": payload,
            "current_rationales": payload["rationales"],
            "current_likelihood": assessment.investment_likelihood,
            "current_confidence": assessment.decision_confidence,
            "unresolved_questions": list(assessment.unresolved_questions),
            **self._with_usage(state, "initial_assessment", calls),
        }

    def _grounded_initial_assessment(self, state: RehearsalState) -> RehearsalState:
        assert self.grounded_baseline is not None
        assessment = baseline_initial_assessment(self.grounded_baseline)
        payload = assessment.model_dump(mode="json")
        registry = dict(state["evidence_registry"])
        for evidence_id, row in self.grounded_baseline.phase1_state.get(
            "evidence_registry", {}
        ).items():
            if not isinstance(row, dict):
                continue
            source_kind = (
                "wiki"
                if str(evidence_id).startswith("W-")
                else "precedent"
                if str(evidence_id).startswith("H-")
                else "portfolio"
            )
            excerpt = str(row.get("text") or row.get("excerpt") or "").strip()
            if not excerpt:
                continue
            registry[str(evidence_id)] = {
                **row,
                "evidence_id": str(evidence_id),
                "source_kind": source_kind,
                "source_path": row.get("source_path") or row.get("episode_slug"),
                "excerpt": excerpt,
                "text": excerpt,
            }
        for rationale in assessment.rationales:
            for evidence in rationale.evidence_refs:
                registry.setdefault(
                    evidence.evidence_id,
                    {
                        "evidence_id": evidence.evidence_id,
                        "source_kind": evidence.source_kind,
                        "source_path": evidence.source_path,
                        "excerpt": evidence.excerpt,
                        "text": evidence.excerpt,
                    },
                )
        priorities = self._grounded_priorities(
            assessment.rationales,
            asked_labels=(),
        )
        self.store.write_accepted("initial-assessment.json", payload)
        self.store.write_accepted(
            "grounding/baseline.json",
            baseline_summary(
                self.grounded_baseline,
                workspace=self.grounded_baseline.run_root.parents[4],
            ).model_dump(mode="json"),
        )
        return {
            "evidence_registry": registry,
            "initial_assessment": payload,
            "current_rationales": payload["rationales"],
            "current_likelihood": assessment.investment_likelihood,
            "current_confidence": assessment.decision_confidence,
            "unresolved_questions": list(assessment.unresolved_questions),
            "question_priorities": priorities,
            "grounded_changes": [],
            "covered_evidence_gap_keys": [],
        }

    def _grounded_priorities(
        self,
        rationales: Sequence[Any],
        *,
        asked_labels: Sequence[str],
        covered_evidence_gap_keys: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        assert self.grounded_baseline is not None
        candidates = grounded_question_candidates(
            baseline=self.grounded_baseline,
            rationales=tuple(
                row if isinstance(row, BaseModel) else self._rationale_model(row)
                for row in rationales
            ),
            asked_labels=asked_labels,
            covered_evidence_gap_keys=covered_evidence_gap_keys,
        )
        classifier_priorities: dict[str, float] = {}
        if self.classifier_tiebreaker and self.classifier_artifact is not None:
            rationale_payloads = [
                row.model_dump(mode="json") if isinstance(row, BaseModel) else row
                for row in rationales
            ]
            features = rehearsal_feature_map(
                rationales=rationale_payloads,
                investment_likelihood=self.grounded_baseline.decision.investment_likelihood,
                decision_confidence=self.grounded_baseline.decision.decision_confidence,
                direct_decision=self.grounded_baseline.decision.decision,
                controlling_rationale_ids=self.grounded_baseline.decision.controlling_rationale_ids,
            )
            priorities = rank_counterfactual_questions(
                self.classifier_artifact,
                features,
                tuple(
                    QuestionCandidate(
                        rationale_label=row.taxonomy_label,
                        evidence_gap=row.evidence_gap,
                        question_archetype=row.evidence_gap,
                    )
                    for row in candidates
                ),
                minimum_probability_impact=self.minimum_probability_impact,
            )
            classifier_priorities = {
                row.rationale_label: row.probability_impact for row in priorities
            }
        ranked = rank_grounded_questions(
            candidates, classifier_priorities=classifier_priorities
        )
        return [
            {
                **asdict(row),
                "rationale_label": row.taxonomy_label,
                "question_archetype": row.evidence_gap,
                "probability_impact": row.classifier_tiebreaker_score,
            }
            for row in ranked
        ]

    @staticmethod
    def _rationale_model(payload: dict[str, Any]):
        from .rehearsal_schemas import RationaleState

        return RationaleState.model_validate(payload)

    @staticmethod
    def _compact_rationales(
        rationales: Sequence[dict[str, Any]], *, excerpt_limit: int = 600
    ) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for rationale in rationales:
            row = dict(rationale)
            row["evidence_refs"] = [
                {
                    **evidence,
                    "excerpt": str(evidence.get("excerpt", ""))[:excerpt_limit],
                }
                for evidence in rationale.get("evidence_refs", [])
            ]
            compact.append(row)
        return compact

    def _score_classification(
        self,
        state: RehearsalState,
        *,
        stage: str,
        rationales: Sequence[dict[str, Any]],
        likelihood: float,
        confidence: float,
        direct_decision: str,
        controlling_rationale_ids: Sequence[str] = (),
    ) -> RehearsalState:
        if self.classification_mode != "classification_informed":
            return {}
        artifact = self.classifier_artifact
        if artifact is None:
            reason = self.classifier_fallback_reason or "classifier artifact unavailable"
            payload = {"status": "fallback", "fallback_reason": reason}
            if not (self.store.session_root / "classification/fallback.json").exists():
                self.store.write_accepted("classification/fallback.json", payload)
            return {
                "classification_status": "fallback",
                "classification_fallback_reason": reason,
                "classification_snapshots": [],
                "question_priorities": [],
            }
        features = rehearsal_feature_map(
            rationales=rationales,
            investment_likelihood=likelihood,
            decision_confidence=confidence,
            direct_decision="In" if direct_decision == "In" else "Out",
            controlling_rationale_ids=controlling_rationale_ids,
        )
        score = artifact.score(features)
        snapshot = ClassificationSnapshot(
            stage=stage,
            probability_in=score.probability_in,
            predicted_decision=score.predicted_decision,
        ).model_dump(mode="json")
        snapshots = [*state.get("classification_snapshots", []), snapshot]
        asked = set(state.get("asked_classifier_labels", []))
        unresolved_text = " ".join(state.get("unresolved_questions", []))
        candidates = tuple(
            QuestionCandidate(
                rationale_label=row["label"],
                evidence_gap=(
                    unresolved_text
                    if unresolved_text
                    else f"Evidence for {row['definition']} remains unresolved."
                ),
                question_archetype=(
                    f"Ask one founder-answerable question testing {row['definition']}"
                ),
            )
            for row in self.taxonomy
            if row["label"] not in asked
        )
        priorities = rank_counterfactual_questions(
            artifact,
            features,
            candidates,
            minimum_probability_impact=self.minimum_probability_impact,
        )
        private_payload = {
            **snapshot,
            "artifact_id": artifact.artifact_id,
            "model_version": artifact.model_version,
            "training_context": artifact.training_context,
            "excluded_episode_slug": artifact.excluded_episode_slug,
            "feature_count": len(features),
        }
        self.store.write_accepted(f"classification/{stage}.json", private_payload)
        return {
            "classification_status": "available",
            "classification_snapshots": snapshots,
            "current_features": features,
            "question_priorities": [asdict(row) for row in priorities],
        }

    def _initial_classification(self, state: RehearsalState) -> RehearsalState:
        if self.classification_mode == "v41_grounded":
            assessment = state["initial_assessment"]
            return self._grounded_classification_snapshot(
                state,
                stage="initial",
                rationales=state["current_rationales"],
                likelihood=float(state["current_likelihood"]),
                confidence=float(state["current_confidence"]),
                direct_decision=str(assessment["decision"]),
            )
        assessment = state["initial_assessment"]
        return self._score_classification(
            state,
            stage="initial",
            rationales=state["current_rationales"],
            likelihood=float(state["current_likelihood"]),
            confidence=float(state["current_confidence"]),
            direct_decision=str(assessment["decision"]),
        )

    def _select_question(self, state: RehearsalState) -> RehearsalState:
        priorities = state.get("question_priorities", [])
        selected_priority = priorities[0] if priorities else None
        if (
            selected_priority is not None
            and state.get("question_count", 0) == self.max_questions - 1
            and not state.get("asked_distinctive_question", False)
        ):
            distinctive = next(
                (
                    row
                    for row in priorities
                    if float(row.get("investor_distinctiveness", 0.0)) >= 0.5
                    and float(selected_priority.get("primary_score", 0.0))
                    - float(row.get("primary_score", 0.0))
                    <= 0.05
                ),
                None,
            )
            if distinctive is not None:
                selected_priority = distinctive
        query = (
            f"{selected_priority['rationale_label']} {selected_priority['evidence_gap']}"
            if selected_priority is not None
            else "\n".join(state.get("unresolved_questions", [])) or state["pitch"]
        )
        bundle = self.retriever.retrieve(query)
        prior_question_texts = tuple(
            str(answer["question"]) for answer in state.get("answers", [])
        )
        query_terms = set(query.casefold().replace("_", " ").split())
        relevant_labels = tuple(
            dict.fromkeys(
                str(row["taxonomy_label"])
                for row in state.get("current_rationales", [])
                if query_terms
                & set(
                    (
                        str(row.get("taxonomy_label", ""))
                        + " "
                        + str(row.get("assessment", ""))
                    )
                    .casefold()
                    .replace("_", " ")
                    .split()
                )
            )
        )
        if selected_priority is not None:
            relevant_labels = (str(selected_priority["rationale_label"]),)
        elif not relevant_labels:
            relevant_labels = tuple(
                dict.fromkeys(
                    str(row["taxonomy_label"])
                    for row in state.get("current_rationales", [])
                    if row.get("salience") == "primary"
                )
            )
        question_search = getattr(self.retriever, "retrieve_question_archetypes", None)
        question_archetypes = (
            tuple(
                question_search(
                    query,
                    rationale_labels=relevant_labels,
                    prior_questions=prior_question_texts,
                )
            )
            if callable(question_search)
            else ()
        )
        registry = self._registry_add(
            state["evidence_registry"],
            tuple(
                row
                for row in bundle.all_evidence()
                if row["evidence_id"] not in state["evidence_registry"]
            ),
        )
        prior_questions = [
            {"question": answer["question"], "answer": answer["text_verbatim"]}
            for answer in state.get("answers", [])
        ]
        prompt_evidence_ids = {
            str(row["evidence_id"]) for row in bundle.all_evidence()
        }
        if selected_priority is not None:
            selected_id = str(selected_priority.get("rationale_id", ""))
            for rationale in state.get("current_rationales", []):
                if rationale.get("rationale_id") == selected_id:
                    prompt_evidence_ids.update(
                        str(row["evidence_id"])
                        for row in rationale.get("evidence_refs", [])
                    )
        prompt_registry = {
            identifier: {
                **row,
                "excerpt": str(row.get("excerpt") or row.get("text") or "")[:2500],
                "text": str(row.get("text") or row.get("excerpt") or "")[:2500],
            }
            for identifier, row in registry.items()
            if identifier in prompt_evidence_ids
        }
        prompt = select_question_prompt(
            investor_name=self.investor_name,
            rationale_state=self._compact_rationales(state["current_rationales"]),
            unresolved_questions=state.get("unresolved_questions", []),
            prior_questions=prior_questions,
            evidence_registry=prompt_registry,
            taxonomy=self.taxonomy,
            question_archetypes=question_archetypes,
            covered_evidence_gap_keys=state.get(
                "covered_evidence_gap_keys", []
            ),
            classifier_guidance=(
                {
                    "rationale_label": str(selected_priority["rationale_label"]),
                    "evidence_gap": str(selected_priority["evidence_gap"]),
                    "question_archetype": str(selected_priority["question_archetype"]),
                    "impact_band": (
                        "low"
                        if float(selected_priority["probability_impact"]) < 0.05
                        else "medium"
                        if float(selected_priority["probability_impact"]) < 0.15
                        else "high"
                    ),
                }
                if selected_priority is not None
                else None
            ),
        )
        question, calls = self._generate(
            "select_question", prompt, RehearsalQuestion
        )
        original_payload = question.model_dump(mode="json")
        suggest_labels = getattr(
            self.retriever, "suggest_question_rationale_labels", None
        )
        alignment_check = getattr(
            self.retriever, "question_rationale_alignment_findings", None
        )
        original_suggestions = (
            tuple(suggest_labels(question.text, top_k=3))
            if callable(suggest_labels)
            else ()
        )
        original_alignment = (
            tuple(
                alignment_check(
                    question=question.text,
                    selected_labels=question.rationale_labels,
                )
            )
            if callable(alignment_check)
            else ()
        )
        original_findings = tuple(
            dict.fromkeys(
                (
                    *atomic_question_findings(question.text),
                    *spoken_question_findings(question.text),
                    *original_alignment,
                )
            )
        )
        repair_calls: list[GenerationResult] = []
        if original_findings:
            repair, repair_calls = self._generate(
                "atomic_question_repair",
                atomic_question_repair_prompt(
                    investor_name=self.investor_name,
                    question=original_payload,
                    findings=original_findings,
                    question_archetypes=question_archetypes,
                    taxonomy=self.taxonomy,
                    suggested_rationale_labels=original_suggestions,
                ),
                RehearsalQuestion,
            )
            question = repair
        proposed_labels = tuple(question.rationale_labels)
        resolve_labels = getattr(
            self.retriever, "resolve_question_rationale_labels", None
        )
        resolved_labels = (
            tuple(
                resolve_labels(
                    question.text,
                    selected_labels=question.rationale_labels,
                )
            )
            if callable(resolve_labels)
            else proposed_labels
        )
        if resolved_labels and resolved_labels != proposed_labels:
            dimension_for = getattr(
                self.retriever, "question_rationale_dimension", None
            )
            question = question.model_copy(
                update={
                    "rationale_labels": resolved_labels,
                    "dimension": (
                        dimension_for(resolved_labels)
                        if callable(dimension_for)
                        else question.dimension
                    ),
                }
            )
        coverage_enabled = self.classification_mode == "v41_grounded"
        pitch_statements = tuple(
            EvidenceStatement(
                evidence_id=str(row["evidence_id"]),
                text=str(row.get("text") or row.get("excerpt") or ""),
                rationale_labels=tuple(question.rationale_labels),
            )
            for row in state.get("pitch_evidence", [])
            if coverage_enabled
        )
        prior_answer_statements = tuple(
            EvidenceStatement(
                evidence_id=str(row["answer_id"]),
                text=f"Question: {row.get('question', '')} Answer: {row['text_verbatim']}",
                rationale_labels=tuple(row.get("rationale_labels", ())),
            )
            for row in state.get("answers", [])
            if coverage_enabled
        )
        original_coverage = assess_question_coverage(
            question=question.text,
            rationale_labels=question.rationale_labels,
            evidence=pitch_statements,
            prior_answers=prior_answer_statements,
        )
        coverage_repair_calls: list[GenerationResult] = []
        coverage_repair_attempted = False
        coverage_findings: list[str] = []
        if original_coverage.redundant:
            coverage_repair_attempted = True
            support_ids = set(original_coverage.supporting_evidence_ids)
            support_rows = [
                row for row in state.get("pitch_evidence", [])
                if str(row["evidence_id"]) in support_ids
            ]
            try:
                question, coverage_repair_calls = self._generate(
                    "question_coverage_repair",
                    question_coverage_repair_prompt(
                        investor_name=self.investor_name,
                        question=question.model_dump(mode="json"),
                        supporting_evidence=support_rows,
                        remaining_priorities=[
                            row for row in priorities if row is not selected_priority
                        ][:4],
                        question_archetypes=question_archetypes,
                        taxonomy=self.taxonomy,
                    ),
                    RehearsalQuestion,
                )
                validate_taxonomy_labels(
                    question.rationale_labels, self.taxonomy_labels
                )
            except ResumableModelError:
                coverage_findings.append("question_coverage_repair_failed")
        final_coverage = assess_question_coverage(
            question=question.text,
            rationale_labels=question.rationale_labels,
            evidence=pitch_statements,
            prior_answers=prior_answer_statements,
        )
        final_suggestions = (
            tuple(suggest_labels(question.text, top_k=3))
            if callable(suggest_labels)
            else ()
        )
        final_alignment = (
            tuple(
                alignment_check(
                    question=question.text,
                    selected_labels=question.rationale_labels,
                )
            )
            if callable(alignment_check)
            else ()
        )
        final_findings = tuple(
            dict.fromkeys(
                (
                    *atomic_question_findings(question.text),
                    *spoken_question_findings(question.text),
                    *final_alignment,
                    *coverage_findings,
                )
            )
        )
        expected_id = f"Q-{state.get('question_count', 0) + 1:03d}"
        if question.question_id != expected_id:
            final_findings = (
                *final_findings,
                f"normalized_question_id:{question.question_id}->{expected_id}",
            )
            question = question.model_copy(update={"question_id": expected_id})
        validate_taxonomy_labels(question.rationale_labels, self.taxonomy_labels)
        prior_text = {
            str(answer["question"]).strip().casefold()
            for answer in state.get("answers", [])
        }
        if question.text.strip().casefold() in prior_text:
            raise ResumableModelError("selected question duplicates an accepted turn")
        payload = question.model_dump(mode="json")
        gap_key_for = getattr(self.retriever, "question_evidence_gap_key", None)
        evidence_gap_key = (
            gap_key_for(question.text, rationale_labels=question.rationale_labels)
            if callable(gap_key_for)
            else question_evidence_gap_key(
                question.text, question.rationale_labels
            )
        )
        turn_path = f"turns/turn-{state.get('question_count', 0) + 1:02d}"
        self.store.write_accepted(
            f"{turn_path}/question-archetypes.json",
            {
                "query": query,
                "requested_rationale_labels": list(relevant_labels),
                "prior_questions": list(prior_question_texts),
                "retrieval_policy": "hybrid",
                "archetypes": list(question_archetypes),
            },
        )
        self.store.write_accepted(
            f"{turn_path}/question-quality.json",
            {
                "original_question": original_payload,
                "original_findings": list(original_findings),
                "original_rationale_suggestions": list(original_suggestions),
                "repair_attempted": bool(original_findings),
                "final_question": payload,
                "final_findings": list(final_findings),
                "final_rationale_suggestions": list(final_suggestions),
                "evidence_gap_key": evidence_gap_key,
                "label_resolution": {
                    "proposed": list(proposed_labels),
                    "final": list(question.rationale_labels),
                },
                "accepted_with_findings": bool(final_findings),
            },
        )
        self.store.write_accepted(
            f"{turn_path}/question-coverage.json",
            {
                "original": asdict(original_coverage),
                "repair_attempted": coverage_repair_attempted,
                "final": asdict(final_coverage),
            },
        )
        self.store.write_accepted(
            f"{turn_path}/question.json",
            payload,
        )
        if selected_priority is not None:
            self.store.write_accepted(
                f"{turn_path}/classifier-priority.json",
                selected_priority,
            )
        if self.classification_mode == "v41_grounded":
            self.store.write_accepted(
                f"grounding/candidates/turn-{state.get('question_count', 0) + 1:02d}.json",
                {
                    "selected": selected_priority,
                    "candidates": state.get("question_priorities", []),
                },
            )
        asked_labels = list(state.get("asked_classifier_labels", []))
        asked_labels.extend(question.rationale_labels)
        if selected_priority is not None:
            asked_labels.append(str(selected_priority["rationale_label"]))
        return {
            "evidence_registry": registry,
            "current_question": payload,
            "findings": [
                *state.get("findings", []),
                *(f"question_quality:{value}" for value in final_findings),
            ],
            "asked_classifier_labels": list(dict.fromkeys(asked_labels)),
            "asked_distinctive_question": (
                state.get("asked_distinctive_question", False)
                or bool(
                    selected_priority
                    and float(
                        selected_priority.get("investor_distinctiveness", 0.0)
                    ) >= 0.5
                )
            ),
            "covered_evidence_gap_keys": list(
                dict.fromkeys(
                    [
                        *state.get("covered_evidence_gap_keys", []),
                        evidence_gap_key,
                    ]
                )
            ),
            **self._with_usage(
                state,
                "select_question",
                [*calls, *repair_calls, *coverage_repair_calls],
            ),
        }

    def _await_founder_answer(self, state: RehearsalState) -> RehearsalState:
        response = interrupt(
            RehearsalQuestion.model_validate(state["current_question"]).public_payload()
        )
        if isinstance(response, dict) and response.get("finish") is True:
            return {"founder_requested_finish": True}
        text = response.get("text") if isinstance(response, dict) else response
        if not isinstance(text, str) or not text.strip():
            raise ValueError("founder answer must be non-empty text")
        number = state.get("question_count", 0) + 1
        submission = self.store.accept_answer_submission(
            state["current_question"]["question_id"], text
        )
        answer = FounderAnswerEvidence(
            answer_id=f"A-{number:03d}",
            question_id=state["current_question"]["question_id"],
            text_verbatim=text,
            received_at=_now(),
            submission_sha256=str(submission["submission_sha256"]),
        )
        payload = answer.model_dump(mode="json")
        registry_row = {
            "evidence_id": answer.answer_id,
            "source_kind": "founder_answer",
            "source_path": f"turns/turn-{number:02d}/answer.json",
            "excerpt": answer.text_verbatim,
            "text": answer.text_verbatim,
        }
        return {
            "pending_answer": payload,
            "evidence_registry": self._registry_add(
                state["evidence_registry"], (registry_row,)
            ),
        }

    def _update_evidence(self, state: RehearsalState) -> RehearsalState:
        if state.get("founder_requested_finish"):
            return {}
        pending_submission = self.store.read_json(
            "pending-answer-submission.json"
        )
        if not self.store.verify_answer_submission(
            pending_submission, state["pending_answer"]
        ):
            raise ResumableModelError("answer_integrity_mismatch")
        if self.classification_mode == "v41_grounded":
            assert self.grounded_baseline is not None
            answer = FounderAnswerEvidence.model_validate(state["pending_answer"])
            prompt = grounded_answer_update_prompt(
                investor_name=self.investor_name,
                baseline_rationale_ids=tuple(
                    row.rationale_id for row in self.grounded_baseline.investigation.rationales
                ),
                rationale_state=self._compact_rationales(
                    state["current_rationales"]
                ),
                current_decision=(
                    "In" if float(state["current_likelihood"]) >= 0.5 else "Out"
                ),
                current_likelihood=float(state["current_likelihood"]),
                current_confidence=float(state["current_confidence"]),
                question=state["current_question"],
                answer=answer,
                evidence_registry={
                    identifier: {
                        **row,
                        "excerpt": str(
                            row.get("excerpt") or row.get("text") or ""
                        )[:2500],
                        "text": str(row.get("text") or row.get("excerpt") or "")[
                            :2500
                        ],
                    }
                    for identifier, row in state["evidence_registry"].items()
                    if identifier == answer.answer_id
                    or identifier
                    in {
                        str(evidence["evidence_id"])
                        for rationale in state["current_rationales"]
                        if rationale.get("taxonomy_label")
                        in state["current_question"].get("rationale_labels", [])
                        for evidence in rationale.get("evidence_refs", [])
                    }
                },
                taxonomy=self.taxonomy,
            )
            update, calls = self._generate(
                "answer_update", prompt, GroundedAnswerUpdate
            )
            validate_grounded_update_context(
                update,
                (row["rationale_id"] for row in state["current_rationales"]),
            )
            update, normalization_findings = normalize_grounded_no_change_update(
                update,
                current_likelihood=float(state["current_likelihood"]),
                current_confidence=float(state["current_confidence"]),
            )
            excerpt = update.supporting_answer_excerpt
            excerpt_is_verbatim = not excerpt or excerpt in answer.text_verbatim
            effect_delta = enforce_evidence_effect_delta(
                effect=update.evidence_effect,
                before=float(state["current_likelihood"]),
                proposed_after=update.investment_likelihood_after,
                supporting_excerpt=excerpt if excerpt_is_verbatim else None,
            )
            effect_findings = list(effect_delta.findings)
            if not excerpt_is_verbatim:
                effect_findings.append(
                    "supporting_answer_excerpt_not_verbatim"
                )
            if effect_delta.after != update.investment_likelihood_after:
                update = update.model_copy(
                    update={
                        "materially_changed_assessment": False,
                        "investment_likelihood_before": effect_delta.before,
                        "investment_likelihood_after": effect_delta.after,
                        "decision_confidence_after": float(
                            state["current_confidence"]
                        ),
                    }
                )
            normalization_findings = tuple(
                dict.fromkeys((*normalization_findings, *effect_findings))
            )
        else:
            prompt = update_prompt(
                investor_name=self.investor_name,
                initial_assessment=state["initial_assessment"],
                current_rationales=state["current_rationales"],
                question=state["current_question"],
                answer=state["pending_answer"],
                evidence_registry=state["evidence_registry"],
                taxonomy=self.taxonomy,
            )
            update, calls = self._generate("answer_update", prompt, AnswerUpdate)
            normalization_findings = ()
        if update.answer_id != state["pending_answer"]["answer_id"]:
            raise ResumableModelError("answer update references the wrong answer")
        validate_taxonomy_labels(
            update.affected_rationale_labels, self.taxonomy_labels
        )
        validate_rationale_state(update.rationale_state_after, self.taxonomy_labels)
        registry_ids = set(state["evidence_registry"])
        cited = {
            evidence.evidence_id
            for rationale in update.rationale_state_after
            for evidence in rationale.evidence_refs
        }
        if not cited <= registry_ids:
            raise ResumableModelError("answer update cites unknown evidence")
        return {
            "pending_update": update.model_dump(mode="json"),  # type: ignore[typeddict-unknown-key]
            "pending_update_normalization": list(normalization_findings),
            **self._with_usage(state, "answer_update", calls),
        }

    def _reassess(self, state: RehearsalState) -> RehearsalState:
        if state.get("founder_requested_finish"):
            return {}
        answer = state["pending_answer"]
        update = state["pending_update"]  # type: ignore[typeddict-item]
        number = state.get("question_count", 0) + 1
        accepted_answer = {
            **answer,
            "question": state["current_question"]["text"],
            "dimension": state["current_question"]["dimension"],
            "rationale_labels": state["current_question"].get(
                "rationale_labels", []
            ),
        }
        self.store.append_answer(answer["answer_id"], answer)
        self.store.mark_answer_submission_processed(answer["answer_id"])
        self.store.write_accepted(f"turns/turn-{number:02d}/update.json", update)
        if state.get("pending_update_normalization"):
            self.store.write_accepted(
                f"turns/turn-{number:02d}/update-normalization.json",
                {
                    "findings": state["pending_update_normalization"],
                    "normalized_update": update,
                },
            )
        if self.classification_mode == "v41_grounded":
            self.store.write_accepted(
                f"grounding/changes/{answer['answer_id'].lower()}.json",
                {
                    "answer_id": answer["answer_id"],
                    "changes": update.get("grounded_changes", []),
                    "materially_changed_assessment": update[
                        "materially_changed_assessment"
                    ],
                },
            )
        rationale_state_after = update["rationale_state_after"]
        if self.classification_mode == "v41_grounded":
            changed = {
                row["rationale_id"]: row for row in update["rationale_state_after"]
            }
            rationale_state_after = [
                changed.get(row["rationale_id"], row)
                for row in state["current_rationales"]
            ]
            existing_ids = {
                row["rationale_id"] for row in state["current_rationales"]
            }
            rationale_state_after.extend(
                row
                for row in update["rationale_state_after"]
                if row["rationale_id"] not in existing_ids
            )
        return {
            "answers": [*state.get("answers", []), accepted_answer],
            "updates": [*state.get("updates", []), update],
            "question_count": number,
            "current_rationales": rationale_state_after,
            "current_likelihood": update["investment_likelihood_after"],
            "current_confidence": update["decision_confidence_after"],
            "unresolved_questions": update["unresolved_questions_after"],
            "pending_answer": {},
            "pending_update": {},  # type: ignore[typeddict-unknown-key]
            "pending_update_normalization": [],
            "grounded_changes": [
                *state.get("grounded_changes", []),
                *update.get("grounded_changes", []),
            ],
        }

    def _updated_classification(self, state: RehearsalState) -> RehearsalState:
        if state.get("founder_requested_finish"):
            return {}
        if self.classification_mode == "v41_grounded":
            return {
                "question_priorities": self._grounded_priorities(
                    state["current_rationales"],
                    asked_labels=state.get("asked_classifier_labels", []),
                    covered_evidence_gap_keys=state.get(
                        "covered_evidence_gap_keys", []
                    ),
                )
            }
        return self._score_classification(
            state,
            stage="updated",
            rationales=state["current_rationales"],
            likelihood=float(state["current_likelihood"]),
            confidence=float(state["current_confidence"]),
            direct_decision=(
                "In" if float(state["current_likelihood"]) >= 0.5 else "Out"
            ),
        )

    def _should_continue(self, state: RehearsalState) -> RehearsalState:
        if state.get("founder_requested_finish"):
            return {
                "continue_interview": False,
                "stopping_reason": "founder_requested",
            }
        if state.get("question_count", 0) >= self.max_questions:
            return {
                "continue_interview": False,
                "stopping_reason": "question_cap",
            }
        remaining = self.max_questions - state.get("question_count", 0)
        prompt = continuation_prompt(
            investor_name=self.investor_name,
            current_rationales=state["current_rationales"],
            unresolved_questions=state.get("unresolved_questions", []),
            questions_and_answers=state.get("answers", []),
            remaining_question_budget=remaining,
        )
        decision, calls = self._generate(
            "continuation", prompt, ContinuationDecision
        )
        if (
            self.classification_mode == "classification_informed"
            and self.classifier_artifact is not None
        ):
            highest_impact = max(
                (
                    float(row["probability_impact"])
                    for row in state.get("question_priorities", [])
                ),
                default=0.0,
            )
            policy = should_continue_interview(
                question_count=state.get("question_count", 0),
                minimum_questions=self.minimum_questions,
                maximum_questions=self.classification_maximum_questions,
                founder_requested_finish=False,
                information_sufficient=not decision.continue_interview,
                highest_probability_impact=highest_impact,
                minimum_probability_impact=self.minimum_probability_impact,
            )
            return {
                "continue_interview": policy.continue_interview,
                "stopping_reason": policy.reason,
                **self._with_usage(state, "continuation", calls),
            }
        return {
            "continue_interview": decision.continue_interview,
            "stopping_reason": (
                "continuing" if decision.continue_interview else "information_sufficient"
            ),
            **self._with_usage(state, "continuation", calls),
        }

    @staticmethod
    def _route_continue(state: RehearsalState) -> str:
        return "ask" if state.get("continue_interview") else "finish"

    def _synthesize_decision(self, state: RehearsalState) -> RehearsalState:
        if self.classification_mode == "v41_grounded":
            assert self.grounded_baseline is not None
            assert self.phase2_synthesizer is not None
            synthesis = self.phase2_synthesizer.synthesize(
                baseline=self.grounded_baseline,
                answers=tuple(
                    FounderAnswerEvidence.model_validate(
                        {
                            key: value
                            for key, value in row.items()
                            if key in {
                                "answer_id",
                                "question_id",
                                "text_verbatim",
                                "received_at",
                                "submission_sha256",
                            }
                        }
                    )
                    for row in state.get("answers", [])
                ),
                changes=tuple(
                    GroundedRationaleChange.model_validate(row)
                    for row in state.get("grounded_changes", [])
                ),
                current_rationales=tuple(
                    self._rationale_model(row) for row in state["current_rationales"]
                ),
                conversational_likelihood=float(state["current_likelihood"]),
            )
            if synthesis.final_assessment is None:
                raise ResumableModelError(
                    synthesis.failure_reason or "canonical Phase 2 synthesis failed"
                )
            assessment, commitment_findings = calibrate_commitment_amounts(
                synthesis.final_assessment,
                evidence_registry=state["evidence_registry"],
            )
            assessment, reconciliation_findings = normalize_final_score_reconciliation(
                assessment,
                conversational_likelihood=float(state["current_likelihood"]),
            )
            normalization_findings = tuple(
                dict.fromkeys((*commitment_findings, *reconciliation_findings))
            )
            payload = assessment.model_dump(mode="json")
            self.store.write_accepted("final-assessment.json", payload)
            if normalization_findings:
                self.store.write_accepted(
                    "final-assessment-normalization.json",
                    {"findings": list(normalization_findings), "assessment": payload},
                )
            return {
                "final_assessment": payload,
                "grounded_final_status": synthesis.status,
                "grounded_final_artifact_path": synthesis.artifact_path,
                "grounded_final_artifact_sha256": synthesis.artifact_sha256,
                **self._with_external_usage(
                    state, "grounded_final_phase2", synthesis.usage
                ),
            }
        prompt = final_assessment_prompt(
            investor_name=self.investor_name,
            initial_assessment=state["initial_assessment"],
            current_rationales=state["current_rationales"],
            questions_and_answers=state.get("answers", []),
            unresolved_questions=state.get("unresolved_questions", []),
            evidence_registry=state["evidence_registry"],
            stopping_reason=state["stopping_reason"],
            conversational_likelihood=float(state["current_likelihood"]),
            conversational_confidence=float(state["current_confidence"]),
        )
        assessment, calls = self._generate(
            "final_assessment", prompt, FinalAssessment
        )
        validate_rationale_state(assessment.rationale_state, self.taxonomy_labels)
        registry_ids = set(state["evidence_registry"])
        cited = {
            evidence.evidence_id
            for rationale in assessment.rationale_state
            for evidence in rationale.evidence_refs
        }
        if not cited <= registry_ids:
            raise ResumableModelError("final assessment cites unknown evidence")
        assessment, commitment_findings = calibrate_commitment_amounts(
            assessment,
            evidence_registry=state["evidence_registry"],
        )
        assessment, reconciliation_findings = normalize_final_score_reconciliation(
            assessment,
            conversational_likelihood=float(state["current_likelihood"]),
        )
        normalization_findings = tuple(
            dict.fromkeys((*commitment_findings, *reconciliation_findings))
        )
        payload = assessment.model_dump(mode="json")
        self.store.write_accepted("final-assessment.json", payload)
        if normalization_findings:
            self.store.write_accepted(
                "final-assessment-normalization.json",
                {"findings": list(normalization_findings), "assessment": payload},
            )
        return {
            "final_assessment": payload,
            **self._with_usage(state, "final_assessment", calls),
        }

    def _with_external_usage(
        self,
        state: RehearsalState,
        phase: str,
        usage: dict[str, int | float],
    ) -> RehearsalState:
        total = dict(state.get("usage", self._zero_usage()))
        for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
            total[key] = int(total.get(key, 0)) + int(usage.get(key, 0))
        total["cost_usd"] = float(total.get("cost_usd", 0.0)) + float(
            usage.get("cost_usd", 0.0)
        )
        by_phase = dict(state.get("usage_by_phase", {}))
        by_phase[phase] = dict(usage)
        return {"usage": total, "usage_by_phase": by_phase}

    def _final_classification(self, state: RehearsalState) -> RehearsalState:
        if self.classification_mode == "v41_grounded":
            assessment = state["final_assessment"]
            return self._grounded_classification_snapshot(
                state,
                stage="final",
                rationales=assessment["rationale_state"],
                likelihood=float(assessment["investment_likelihood"]),
                confidence=float(assessment["decision_confidence"]),
                direct_decision=str(assessment["decision"]),
            )
        assessment = state["final_assessment"]
        controlling = [
            *assessment.get("strongest_positive_rationale_ids", []),
            *assessment.get("decisive_concern_rationale_ids", []),
        ]
        return self._score_classification(
            state,
            stage="final",
            rationales=assessment["rationale_state"],
            likelihood=float(assessment["investment_likelihood"]),
            confidence=float(assessment["decision_confidence"]),
            direct_decision=str(assessment["decision"]),
            controlling_rationale_ids=controlling,
        )

    def _grounded_classification_snapshot(
        self,
        state: RehearsalState,
        *,
        stage: str,
        rationales: Sequence[dict[str, Any]],
        likelihood: float,
        confidence: float,
        direct_decision: str,
    ) -> RehearsalState:
        artifact = self.classifier_artifact
        if artifact is None:
            reason = self.classifier_fallback_reason or "classifier artifact unavailable"
            return {
                "classification_status": "fallback",
                "classification_fallback_reason": reason,
                "classification_snapshots": [],
            }
        features = rehearsal_feature_map(
            rationales=rationales,
            investment_likelihood=likelihood,
            decision_confidence=confidence,
            direct_decision="In" if direct_decision == "In" else "Out",
            controlling_rationale_ids=(
                self.grounded_baseline.decision.controlling_rationale_ids
                if self.grounded_baseline is not None
                else ()
            ),
        )
        score = artifact.score(features)
        snapshot = ClassificationSnapshot(
            stage=stage,
            probability_in=score.probability_in,
            predicted_decision=score.predicted_decision,
        ).model_dump(mode="json")
        self.store.write_accepted(
            f"classification/{stage}.json",
            {
                **snapshot,
                "artifact_id": artifact.artifact_id,
                "diagnostic_only": True,
            },
        )
        return {
            "classification_status": "available",
            "classification_snapshots": [
                *state.get("classification_snapshots", []), snapshot
            ],
        }

    def _generate_founder_report(self, state: RehearsalState) -> RehearsalState:
        prompt = founder_report_prompt(
            investor_name=self.investor_name,
            initial_assessment={
                **state["initial_assessment"],
                "rationales": self._compact_rationales(
                    state["initial_assessment"]["rationales"], excerpt_limit=300
                ),
            },
            final_assessment={
                **state["final_assessment"],
                "rationale_state": self._compact_rationales(
                    state["final_assessment"]["rationale_state"], excerpt_limit=300
                ),
            },
            answers=state.get("answers", []),
            updates=state.get("updates", []),
        )
        coaching, calls = self._generate("founder_report", prompt, CoachingOutput)
        usage_update = self._with_usage(state, "founder_report", calls)
        classification: ClassificationSummary | None = None
        if self.classification_mode in {"classification_informed", "v41_grounded"}:
            if state.get("classification_status") == "available":
                assert self.classifier_artifact is not None
                classification = ClassificationSummary(
                    status="available",
                    model_version=self.classifier_artifact.model_version,
                    artifact_id=self.classifier_artifact.artifact_id,
                    snapshots=tuple(
                        ClassificationSnapshot.model_validate(row)
                        for row in state.get("classification_snapshots", [])
                    ),
                )
            else:
                classification = ClassificationSummary(
                    status="fallback",
                    fallback_reason=state.get(
                        "classification_fallback_reason",
                        self.classifier_fallback_reason or "classifier unavailable",
                    ),
                )
        grounded_artifact_path = state.get(
            "grounded_final_artifact_path",
            "grounding/final-phase2/decision.json",
        )
        grounded_artifact = self.store.session_root / grounded_artifact_path
        grounded_artifact_sha256 = state.get("grounded_final_artifact_sha256")
        if grounded_artifact_sha256 is None and grounded_artifact.is_file():
            grounded_artifact_sha256 = sha256(grounded_artifact.read_bytes()).hexdigest()
        report = FounderReport(
            session_id=state["session_id"],
            vc_slug=state["vc_slug"],
            disclosure=(
                f"Simulation of a {self.investor_name} rehearsal generated from source-linked "
                "public investment evidence; it is not endorsed by the investor."
            ),
            initial_assessment=InitialAssessment.model_validate(
                state["initial_assessment"]
            ),
            final_assessment=FinalAssessment.model_validate(
                state["final_assessment"]
            ),
            material_answer_ids=coaching.material_answer_ids,
            pitch_improvement_suggestions=coaching.pitch_improvement_suggestions,
            founder_reflection_prompts=coaching.founder_reflection_prompts,
            usage=usage_update["usage"],
            classification=classification,
            grounded=(
                GroundedDecisionComparison(
                    baseline=baseline_summary(
                        self.grounded_baseline,
                        workspace=self.grounded_baseline.run_root.parents[4],
                    ),
                    final_contract_version=self.grounded_baseline.contract_version,
                    final_decision=state["final_assessment"]["decision"],
                    final_investment_likelihood=state["final_assessment"][
                        "investment_likelihood"
                    ],
                    final_decision_confidence=state["final_assessment"][
                        "decision_confidence"
                    ],
                    decision_changed=(
                        state["final_assessment"]["decision"]
                        != self.grounded_baseline.decision.decision
                    ),
                    rationale_changes=tuple(
                        GroundedRationaleChange.model_validate(row)
                        for row in state.get("grounded_changes", [])
                    ),
                    final_artifact_path=grounded_artifact_path,
                    final_artifact_sha256=grounded_artifact_sha256,
                )
                if self.classification_mode == "v41_grounded"
                and self.grounded_baseline is not None
                else None
            ),
        )
        payload = report.model_dump(mode="json")
        if report.grounded is not None:
            self.store.write_accepted(
                "grounding/comparison.json",
                report.grounded.model_dump(mode="json"),
            )
        self.store.write_accepted("founder-report.json", payload)
        self.store.write_accepted(
            "summary.json",
            {
                "session_id": state["session_id"],
                "vc_slug": state["vc_slug"],
                "status": "complete",
                "question_count": state.get("question_count", 0),
                "stopping_reason": state["stopping_reason"],
                "decision": state["final_assessment"]["decision"],
                "investment_likelihood": state["final_assessment"][
                    "investment_likelihood"
                ],
                "classifier_decision": (
                    state.get("classification_snapshots", [{}])[-1].get(
                        "predicted_decision"
                    )
                    if state.get("classification_snapshots")
                    else None
                ),
            },
        )
        self.store.write_accepted("usage.json", usage_update["usage"])
        return {
            "founder_report": payload,
            "status": "complete",
            **usage_update,
        }

    def _build(self) -> StateGraph:
        builder = StateGraph(RehearsalState)
        builder.add_node("ingest_pitch", self._ingest_pitch)
        builder.add_node(
            "initial_assessment",
            self._grounded_initial_assessment
            if self.classification_mode == "v41_grounded"
            else self._initial_assessment,
        )
        builder.add_node("initial_classification", self._initial_classification)
        builder.add_node("select_question", self._select_question)
        builder.add_node("await_founder_answer", self._await_founder_answer)
        builder.add_node("update_evidence", self._update_evidence)
        builder.add_node("reassess", self._reassess)
        builder.add_node("updated_classification", self._updated_classification)
        builder.add_node("should_continue", self._should_continue)
        builder.add_node("synthesize_decision", self._synthesize_decision)
        builder.add_node("final_classification", self._final_classification)
        builder.add_node("generate_founder_report", self._generate_founder_report)
        builder.add_edge(START, "ingest_pitch")
        builder.add_edge("ingest_pitch", "initial_assessment")
        builder.add_edge("initial_assessment", "initial_classification")
        builder.add_edge("initial_classification", "select_question")
        builder.add_edge("select_question", "await_founder_answer")
        builder.add_edge("await_founder_answer", "update_evidence")
        builder.add_edge("update_evidence", "reassess")
        builder.add_edge("reassess", "updated_classification")
        builder.add_edge("updated_classification", "should_continue")
        builder.add_conditional_edges(
            "should_continue",
            self._route_continue,
            {"ask": "select_question", "finish": "synthesize_decision"},
        )
        builder.add_edge("synthesize_decision", "final_classification")
        builder.add_edge("final_classification", "generate_founder_report")
        builder.add_edge("generate_founder_report", END)
        return builder

    @staticmethod
    def _config(thread_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": thread_id}}

    def _persist_state(self, state: RehearsalState) -> None:
        serializable = {
            key: value for key, value in state.items() if not key.startswith("__")
        }
        # LangGraph returns the state immediately before an interrupt.  The
        # workflow's last ordinary node still says ``running`` at that point,
        # even though a question has been selected and execution is paused for
        # founder input.  Persist the public lifecycle state, not that internal
        # node marker, so a process restart can recover the interaction.
        if (
            serializable.get("status") not in {"complete", "resumable_error"}
            and isinstance(serializable.get("current_question"), dict)
        ):
            serializable["status"] = "awaiting_answer"
        self.store.write_accepted("state.json", serializable)

    def _public(self, state: RehearsalState) -> dict[str, Any]:
        if state.get("status") == "complete":
            return {
                "status": "complete",
                "session_id": state["session_id"],
                "answers": state.get("answers", []),
                "stopping_reason": state["stopping_reason"],
                "final_assessment": state["final_assessment"],
                "founder_report": state["founder_report"],
                "usage": state["usage"],
            }
        question = RehearsalQuestion.model_validate(state["current_question"])
        return {
            "status": "awaiting_answer",
            "session_id": state["session_id"],
            "question": question.public_payload(),
            "answers": state.get("answers", []),
        }

    def start(self, thread_id: str, *, pitch: str, vc_slug: str) -> dict[str, Any]:
        initial: RehearsalState = {
            "session_id": thread_id,
            "vc_slug": vc_slug,
            "investor_name": self.investor_name,
            "pitch": pitch,
            "taxonomy": list(self.taxonomy),
        }
        try:
            result = self.graph.invoke(initial, self._config(thread_id))
        except ResumableModelError as exc:
            return self._resumable_error(thread_id, exc)
        self._persist_state(result)
        return self._public(result)

    def answer(self, thread_id: str, text: str) -> dict[str, Any]:
        try:
            result = self.graph.invoke(
                Command(resume={"text": text}), self._config(thread_id)
            )
        except ResumableModelError as exc:
            return self._resumable_error(thread_id, exc)
        self._persist_state(result)
        return self._public(result)

    def finish(self, thread_id: str) -> dict[str, Any]:
        snapshot = self.graph.get_state(self._config(thread_id))
        if not snapshot.values.get("answers"):
            raise ValueError(
                "finish requires at least one accepted answer"
            )
        try:
            result = self.graph.invoke(
                Command(resume={"finish": True}), self._config(thread_id)
            )
        except ResumableModelError as exc:
            return self._resumable_error(thread_id, exc)
        self._persist_state(result)
        return self._public(result)

    def retry(self, thread_id: str) -> dict[str, Any]:
        """Retry the checkpointed failed node without resubmitting founder input."""
        try:
            result = self.graph.invoke(None, self._config(thread_id))
        except ResumableModelError as exc:
            return self._resumable_error(thread_id, exc)
        self._persist_state(result)
        return self._public(result)

    def _resumable_error(
        self, thread_id: str, error: ResumableModelError
    ) -> dict[str, Any]:
        snapshot = self.graph.get_state(self._config(thread_id))
        state = dict(snapshot.values)
        findings = [*state.get("findings", []), str(error)]
        state.update({"status": "resumable_error", "findings": findings})
        self._persist_state(state)
        return {
            "status": "resumable_error",
            "session_id": state.get("session_id", thread_id),
            "answers": state.get("answers", []),
            "current_question": state.get("current_question", {}),
            "findings": findings,
        }
