"""Adaptive, evidence-linked LangGraph workflow for contract v4."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .artifacts import freeze_model, write_json
from .graph import WorkflowSettings, WorkflowState, _event, _usage, _usage_by_phase
from .precedents import PrecedentCorpus
from .portfolio_memory import FilteredPortfolioIndex
from .prompts_v4 import (
    phase1_investigation_v4_prompt,
    phase1_investigation_v41_prompt,
    phase1_plan_v4_prompt,
    phase1_plan_v41_prompt,
    phase1_rationale_lock_v42_prompt,
    phase1_rationale_mapping_v43_prompt,
    phase2_decision_v4_prompt,
    phase2_decision_v41_prompt,
    phase2_plan_v4_prompt,
    phase2_plan_v41_prompt,
    taxonomy_reflection_v4_prompt,
    pitch_evidence_index,
)
from .providers.base import GenerationProvider, GenerationRequest
from .retrieval import HybridWikiIndex
from .retrieval_v4 import V4RetrievalResult, retrieve_v4
from .rationale_association_annotations import RationaleAssociationAnnotations
from .schemas import StrictModel
from .schemas_v4 import (
    DecisionV4,
    DecisionV41,
    InvestigationV4,
    InvestigationV41,
    InvestigationV42,
    InvestigationV43,
    RationaleMappingV43,
    RationaleLockV42,
    RetrievalPlanV4,
    TaxonomyReflectionV4,
    decision_v4_json_schema,
    decision_v41_json_schema,
    investigation_v4_json_schema,
    investigation_v41_json_schema,
    rationale_lock_v42_json_schema,
    rationale_mapping_v43_json_schema,
    model_facing_taxonomy,
    normalize_investigation_v41_payload,
    normalize_investigation_v41_evidence_payload,
    normalize_rationale_lock_v42_payload,
    normalize_rationale_mapping_v43_payload,
    validate_decision_v41_against_investigation,
)
from .schemas_v5 import InvestigationV5


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def apply_rationale_mapping_v43(
    candidate: InvestigationV41,
    mapping: RationaleMappingV43,
    *,
    taxonomy: dict[str, str],
    mapping_sha256: str,
) -> InvestigationV43:
    """Apply a complete family-bounded mapping without mutating cited evidence."""
    candidate_by_id = {row.rationale_id: row for row in candidate.rationales}
    dispositions = {row.rationale_id: row for row in mapping.dispositions}
    if set(candidate_by_id) != set(dispositions):
        raise ValueError("candidate rationales must be dispositioned exactly once")
    for identifier, disposition in dispositions.items():
        current = candidate_by_id[identifier].taxonomy_label
        target = disposition.target_taxonomy_label
        if current not in taxonomy or target not in taxonomy:
            raise ValueError("mapping uses an unknown taxonomy label")
        if taxonomy[current] != taxonomy[target]:
            raise ValueError("relabel and merge targets must remain in the same taxonomy family")
        if disposition.action == "keep" and target != current:
            raise ValueError("keep action must retain the current taxonomy label")
        if disposition.action == "relabel" and target == current:
            raise ValueError("relabel action must change the taxonomy label")
        if disposition.action == "merge":
            merge_target = disposition.merge_into_rationale_id
            if merge_target not in candidate_by_id:
                raise ValueError("merge target is absent from candidate rationales")
            if dispositions[merge_target].action == "merge":
                raise ValueError("merge chains are not permitted")

    final_by_id = {
        identifier: row.model_copy(
            update={"taxonomy_label": dispositions[identifier].target_taxonomy_label}
        )
        for identifier, row in candidate_by_id.items()
        if dispositions[identifier].action != "merge"
    }
    merged_ids: dict[str, str] = {}
    for identifier, disposition in dispositions.items():
        if disposition.action != "merge":
            continue
        target_id = disposition.merge_into_rationale_id
        assert target_id is not None
        source = candidate_by_id[identifier]
        target = final_by_id[target_id]
        if disposition.target_taxonomy_label != target.taxonomy_label:
            raise ValueError("merged rationale must use the surviving target label")
        final_by_id[target_id] = target.model_copy(
            update={
                "pitch_evidence_ids": _dedupe(
                    [*target.pitch_evidence_ids, *source.pitch_evidence_ids]
                ),
                "wiki_evidence_ids": _dedupe(
                    [*target.wiki_evidence_ids, *source.wiki_evidence_ids]
                ),
                "historical_evidence_ids": _dedupe(
                    [*target.historical_evidence_ids, *source.historical_evidence_ids]
                ),
                "salience": (
                    "primary"
                    if "primary" in {target.salience, source.salience}
                    else "secondary"
                ),
                "confidence": max(target.confidence, source.confidence),
                "justification": f"{target.justification} {source.justification}",
            }
        )
        merged_ids[identifier] = target_id

    payload = candidate.model_dump(mode="json")
    for collection in ("material_statement_coverage", "constraint_assessments"):
        for row in payload[collection]:
            row["mapped_ids"] = _dedupe(
                [merged_ids.get(identifier, identifier) for identifier in row["mapped_ids"]]
            )
    payload.update(
        {
            "schema_version": "investigation-v4.3",
            "rationales": [
                final_by_id[row.rationale_id].model_dump(mode="json")
                for row in candidate.rationales
                if row.rationale_id in final_by_id
            ],
            "candidate_rationale_ids": list(candidate_by_id),
            "candidate_investigation_sha256": mapping.candidate_investigation_sha256,
            "rationale_mapping_sha256": mapping_sha256,
            "relabeled_candidate_count": sum(
                row.action == "relabel" for row in mapping.dispositions
            ),
            "merged_candidate_count": len(merged_ids),
            "mapping_status": "accepted",
        }
    )
    return InvestigationV43.model_validate(payload)


def constrain_rationale_mapping_v43(
    candidate: InvestigationV41,
    mapping: RationaleMappingV43,
    *,
    taxonomy: dict[str, str],
    definitions: dict[str, str],
) -> tuple[RationaleMappingV43, list[str]]:
    """Fail closed per disposition when a model proposes a cross-family label."""
    candidate_by_id = {row.rationale_id: row for row in candidate.rationales}
    findings: list[str] = []
    dispositions = []
    for disposition in mapping.dispositions:
        candidate_row = candidate_by_id.get(disposition.rationale_id)
        if candidate_row is None:
            dispositions.append(disposition)
            continue
        current = candidate_row.taxonomy_label
        target = disposition.target_taxonomy_label
        if (
            current in taxonomy
            and target in taxonomy
            and taxonomy[current] != taxonomy[target]
        ):
            findings.append(
                f"MAPPING_CROSS_FAMILY_RETAINED_ORIGINAL:{disposition.rationale_id}"
            )
            dispositions.append(
                disposition.model_copy(
                    update={
                        "action": "keep",
                        "target_taxonomy_label": current,
                        "merge_into_rationale_id": None,
                        "winning_definition": definitions[current],
                        "rejected_alternatives": _dedupe(
                            [target, *disposition.rejected_alternatives]
                        ),
                        "mapping_justification": (
                            "The proposed target crossed the prespecified taxonomy-family "
                            "boundary, so this canary retains the original label. "
                            + disposition.mapping_justification
                        ),
                    }
                )
            )
        else:
            dispositions.append(disposition)
    return mapping.model_copy(update={"dispositions": dispositions}), findings


class WorkflowStateV4(WorkflowState, total=False):
    phase1_plan: dict[str, Any]
    phase1_feedback: list[str]
    phase1_current_retrieval: dict[str, Any]
    phase1_retrieval_warnings: list[str]
    phase2_current_retrieval: dict[str, Any]
    phase2_retrieval_warnings: list[str]
    evidence_registry: dict[str, dict[str, Any]]
    decision_sha256: str
    taxonomy_reflection: dict[str, Any]
    reflection_findings: list[str]
    decision_bound_investigation: dict[str, Any]
    decision_bound_investigation_sha256: str
    phase1_signature: str
    phase2_signature: str
    phase2_decision_streak: int
    phase1_replay_frozen: bool
    phase1_portfolio_candidates: list[dict[str, Any]]
    portfolio_filtered_manifest: dict[str, Any]
    candidate_investigation: dict[str, Any]
    candidate_investigation_sha256: str
    rationale_lock: dict[str, Any]
    rationale_lock_sha256: str
    rationale_mapping: dict[str, Any]
    rationale_mapping_sha256: str
    phase1_candidate_action: str
    rationale_association_annotations: dict[str, Any]
    rehearsal_context: dict[str, Any]


class Phase2RehearsalContext(StrictModel):
    baseline_decision_sha256: str
    baseline_decision: str
    baseline_investment_likelihood: float
    baseline_decision_confidence: float
    baseline_decision_justification: str
    founder_answers: list[dict[str, str]]
    rationale_changes: list[dict[str, Any]]
    current_rationale_state: list[dict[str, Any]]


class VCDecisionWorkflowV4:
    """A natural two-phase workflow that always retains the last valid candidate."""

    def __init__(
        self,
        settings: WorkflowSettings,
        index: HybridWikiIndex | None,
        provider: GenerationProvider | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        *,
        phase1_provider: GenerationProvider | None = None,
        phase2_provider: GenerationProvider | None = None,
        precedent_corpus: PrecedentCorpus | None = None,
        portfolio_index: FilteredPortfolioIndex | None = None,
    ) -> None:
        if settings.contract_version not in {"v4", "v4.1", "v4.2", "v4.3", "v5"}:
            raise ValueError(
                "VCDecisionWorkflowV4 requires contract_version v4, v4.1, v4.2, v4.3, or v5"
            )
        if checkpointer is None:
            raise ValueError("checkpointer is required")
        self.phase1_provider = phase1_provider or provider
        self.phase2_provider = phase2_provider or provider
        if self.phase1_provider is None:
            raise ValueError("a Phase 1 provider is required")
        if settings.execution_mode == "full" and self.phase2_provider is None:
            raise ValueError("a Phase 2 provider is required for full execution")
        if not 1 <= settings.phase1_max_iterations <= 4:
            raise ValueError("v4 Phase 1 max iterations must be between one and four")
        if not 1 <= settings.phase2_max_iterations <= 4:
            raise ValueError("v4 Phase 2 max iterations must be between one and four")
        if precedent_corpus is not None:
            if precedent_corpus.target_slug != settings.episode_slug:
                raise ValueError("precedent corpus must be filtered to the target episode")
            if any(
                row.episode_slug == settings.episode_slug
                for row in precedent_corpus.list_episodes()
            ):
                raise ValueError("target episode remains accessible in precedent corpus")
        self.settings = settings
        self.is_v42 = settings.contract_version == "v4.2"
        self.is_v43 = settings.contract_version == "v4.3"
        self.is_v5 = settings.contract_version == "v5"
        self.is_v41 = settings.contract_version in {"v4.1", "v4.2", "v4.3"}
        self.index = index
        self.precedent_corpus = precedent_corpus
        self.portfolio_index = portfolio_index
        if settings.portfolio_memory_enabled:
            if portfolio_index is None:
                raise ValueError("portfolio memory is enabled but no filtered index was supplied")
            if portfolio_index.filtered.target_episode_slug != settings.episode_slug:
                raise ValueError("portfolio memory must be filtered to the target episode")
        self.taxonomy = model_facing_taxonomy(settings.taxonomy_records)
        self.taxonomy_labels = {row["label"] for row in self.taxonomy}
        settings.run_root.mkdir(parents=True, exist_ok=True)
        self.graph = self._build().compile(checkpointer=checkpointer)
        self.phase2_graph = self._build_phase2().compile(checkpointer=checkpointer)

    def _record_call(
        self,
        phase: str,
        turn: int,
        step: str,
        request: GenerationRequest,
        result: Any,
    ) -> None:
        write_json(
            self.settings.run_root
            / phase
            / f"turn-{turn:02d}"
            / f"{step}-model-response.json",
            {
                "prompt": request.prompt,
                "schema": request.schema,
                "parsed": result.parsed,
                "content": result.content,
                "usage": result.usage.model_dump(mode="json"),
                "elapsed_seconds": result.elapsed_seconds,
                "raw_metadata": result.raw_metadata,
            },
        )

    def _generate_plan_with_repair(
        self,
        *,
        artifact_phase: str,
        request_phase: str,
        turn: int,
        prompt: str,
        provider: GenerationProvider,
        max_output_tokens: int,
        reasoning_effort: str | None,
    ) -> tuple[RetrievalPlanV4 | None, list[Any], list[str]]:
        results: list[Any] = []
        findings: list[str] = []
        current_prompt = prompt
        prefix = artifact_phase.upper()
        for attempt in (1, 2):
            request = GenerationRequest(
                request_phase,
                current_prompt,
                RetrievalPlanV4.model_json_schema(),
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
            )
            result = provider.generate(request)
            self._record_call(
                artifact_phase,
                turn,
                "plan" if attempt == 1 else "plan-retry",
                request,
                result,
            )
            results.append(result)
            try:
                return RetrievalPlanV4.model_validate(result.parsed), results, findings
            except Exception as exc:
                findings.append(f"{prefix}_PLAN_INVALID_ATTEMPT_{attempt}: {exc}")
                current_prompt = f"""{prompt}

REPAIR INSTRUCTION: The previous response could not be accepted because:
{exc}

Return one concise complete JSON object matching the requested schema. Use short questions
and search queries, emit no commentary or formatting outside JSON, and do not emit invisible
or directional control characters. This is the only repair attempt.
"""
        return None, results, findings

    @staticmethod
    def _account_plan_results(
        state: WorkflowStateV4, results: list[Any], phase: str
    ) -> tuple[dict[str, int | float], dict[str, dict[str, int | float]]]:
        accounting = dict(state)
        for result in results:
            accounting["usage"] = _usage(accounting, result)
            accounting["usage_by_phase"] = _usage_by_phase(
                accounting, result, phase  # type: ignore[arg-type]
            )
        return accounting["usage"], accounting["usage_by_phase"]

    def _plan_phase1(self, state: WorkflowStateV4) -> WorkflowStateV4:
        turn = state.get("phase1_iteration", 0) + 1
        previous = state.get("investigation")
        prompt_builder = phase1_plan_v41_prompt if self.is_v41 else phase1_plan_v4_prompt
        prompt = prompt_builder(
            self.settings.investor_name,
            self.settings.episode_slug,
            self.settings.pitch,
            self.taxonomy,
            previous,
            (previous or {}).get("searchable_questions", []),
            state.get("phase1_feedback", []),
        )
        candidate, results, plan_findings = self._generate_plan_with_repair(
            artifact_phase="phase1",
            request_phase="phase1_plan",
            turn=turn,
            prompt=prompt,
            provider=self.phase1_provider,
            max_output_tokens=self.settings.phase1_planning_max_output_tokens,
            reasoning_effort=self.settings.phase1_planning_reasoning_effort,
        )
        findings = [*state.get("phase1_findings", []), *plan_findings]
        if candidate is None:
            findings.append("PHASE1_PLAN_FALLBACK_USED")
            candidate = RetrievalPlanV4(
                questions=["Which material investor rationale is still unresolved?"],
                wiki_queries=["investment criteria material risk"],
                precedent_queries=["similar pitch investor decision rationale"],
                continuation_focus="Recover a focused material rationale assessment.",
            )
        if self.is_v41:
            policy_query = (
                "hard rules category expertise conflicts founder ambition ownership "
                "venture economics"
            )
            candidate = candidate.model_copy(
                update={
                    "wiki_queries": _dedupe(
                        [policy_query, *candidate.wiki_queries]
                    )[:6],
                    "precedent_queries": _dedupe(
                        [policy_query, *candidate.precedent_queries]
                    )[:6],
                }
            )
        usage, usage_by_phase = self._account_plan_results(state, results, "phase1")
        return {
            "phase1_plan": candidate.model_dump(mode="json"),
            "phase1_findings": _dedupe(findings),
            "usage": usage,
            "usage_by_phase": usage_by_phase,
            "events": _event(state, "phase1_plan", turn=turn),
        }

    def _retrieve(self, state: WorkflowStateV4, phase: str) -> WorkflowStateV4:
        key = f"{phase}_plan"
        plan = RetrievalPlanV4.model_validate(state[key])
        turn = state.get(f"{phase}_iteration", 0) + 1
        result = retrieve_v4(
            wiki_index=self.index,
            precedent_corpus=self.precedent_corpus,
            wiki_queries=plan.wiki_queries,
            precedent_queries=plan.precedent_queries[
                : (
                    self.settings.phase1_max_precedent_searches
                    if phase == "phase1"
                    else self.settings.phase2_max_precedent_searches
                )
            ],
            top_k=self.settings.retrieval_top_k,
            max_wiki_reads=self.settings.max_exact_reads,
            max_precedent_reads=(
                self.settings.phase1_max_precedent_reads
                if phase == "phase1"
                else self.settings.phase2_max_precedent_reads
            ),
            phase=phase,
            turn=turn,
            selection_policy=(
                self.settings.precedent_selection_policy
                if phase == "phase2"
                else "semantic"
            ),
            candidate_pool_k=self.settings.precedent_candidate_pool_k,
            in_slots=self.settings.precedent_in_slots,
            out_slots=self.settings.precedent_out_slots,
        )
        directory = self.settings.run_root / phase / f"turn-{turn:02d}"
        historical_by_id = {
            row["evidence_id"]: row
            for row in state.get(f"{phase}_historical_evidence", [])
        }
        for row in result.historical_evidence:
            historical_by_id.setdefault(row["evidence_id"], row)
        historical_history = list(historical_by_id.values())
        precedent_search_history = [
            *state.get(f"{phase}_precedent_searches", []),
            *result.precedent_searches,
        ]
        precedent_read_history = [
            *state.get(f"{phase}_precedent_reads", []),
            *result.precedent_reads,
        ]
        write_json(directory / "wiki-searches.json", list(result.wiki_searches))
        write_json(directory / "wiki-reads.json", list(result.wiki_evidence))
        write_json(directory / "precedent-searches.json", list(result.precedent_searches))
        write_json(directory / "precedent-reads.json", list(result.precedent_reads))
        write_json(directory / "historical-evidence.json", historical_history)
        write_json(directory / "retrieval.json", asdict(result))
        registry = dict(state.get("evidence_registry", {}))
        for row in (*result.wiki_evidence, *result.historical_evidence):
            registry[row["evidence_id"]] = row
        return {
            f"{phase}_current_retrieval": asdict(result),
            f"{phase}_precedent_searches": precedent_search_history,
            f"{phase}_precedent_reads": precedent_read_history,
            f"{phase}_historical_evidence": historical_history,
            "evidence_registry": registry,
            f"{phase}_retrieval_warnings": _dedupe(
                [*state.get(f"{phase}_retrieval_warnings", []), *result.warnings]
            ),
            "events": _event(
                state,
                f"{phase}_retrieval",
                turn=turn,
                wiki_evidence=len(result.wiki_evidence),
                historical_evidence=len(result.historical_evidence),
            ),
        }

    def _retrieve_phase1(self, state: WorkflowStateV4) -> WorkflowStateV4:
        return self._retrieve(state, "phase1")

    def _retrieve_portfolio(self, state: WorkflowStateV4) -> WorkflowStateV4:
        turn = state.get("phase1_iteration", 0) + 1
        if self.portfolio_index is None:
            candidates: list[dict[str, Any]] = []
            manifest: dict[str, Any] = {}
        else:
            hits = self.portfolio_index.search(
                self.settings.pitch,
                limit=self.settings.portfolio_retrieval_top_k,
                candidate_pool_k=self.settings.portfolio_candidate_pool_k,
            )
            disclosures = {
                row.disclosure_id: row
                for row in self.portfolio_index.filtered.disclosures
            }
            candidates = [
                {
                    "entity": hit.model_dump(mode="json"),
                    "disclosures": [
                        disclosures[identifier].model_dump(mode="json")
                        for identifier in hit.disclosure_ids
                    ],
                }
                for hit in hits
            ]
            manifest = self.portfolio_index.filtered.manifest().model_dump(mode="json")
        directory = self.settings.run_root / "phase1" / f"turn-{turn:02d}"
        write_json(directory / "portfolio-filtered-manifest.json", manifest)
        write_json(directory / "portfolio-candidates.json", candidates)
        return {
            "phase1_portfolio_candidates": candidates,
            "portfolio_filtered_manifest": manifest,
            "events": _event(
                state,
                "phase1_portfolio_retrieval",
                turn=turn,
                candidate_count=len(candidates),
            ),
        }

    def _investigation_findings(
        self, candidate: InvestigationV4 | InvestigationV41, state: WorkflowStateV4
    ) -> list[str]:
        findings: list[str] = []
        if candidate.episode_slug != self.settings.episode_slug:
            findings.append("EPISODE_SLUG_MISMATCH")
        if candidate.information_sufficient and candidate.searchable_questions:
            findings.append("SUFFICIENT_WITH_SEARCHABLE_QUESTIONS")
        if candidate.information_sufficient and candidate.next_search_objectives:
            findings.append("SUFFICIENT_WITH_NEXT_SEARCH_OBJECTIVES")
        if not candidate.information_sufficient and not candidate.searchable_questions:
            findings.append("INSUFFICIENT_WITHOUT_SEARCHABLE_QUESTION")
        registry = state.get("evidence_registry", {})
        pitch_ids = {row["evidence_id"] for row in pitch_evidence_index(self.settings.pitch)}
        for rationale in candidate.rationales:
            if rationale.taxonomy_label not in self.taxonomy_labels:
                findings.append(f"UNKNOWN_TAXONOMY_LABEL:{rationale.taxonomy_label}")
            for evidence_id in [
                *rationale.wiki_evidence_ids,
                *rationale.historical_evidence_ids,
            ]:
                if evidence_id not in registry:
                    findings.append(f"INACCESSIBLE_EVIDENCE:{evidence_id}")
            for evidence_id in rationale.pitch_evidence_ids:
                if evidence_id not in pitch_ids:
                    findings.append(f"INACCESSIBLE_PITCH_EVIDENCE:{evidence_id}")
        for question in candidate.questions:
            for evidence_id in question.evidence_ids:
                if evidence_id not in registry:
                    findings.append(f"INACCESSIBLE_EVIDENCE:{evidence_id}")
        for observation in candidate.unmapped_observations:
            for evidence_id in observation.evidence_ids:
                if evidence_id not in registry:
                    findings.append(f"INACCESSIBLE_EVIDENCE:{evidence_id}")
        rationale_ids = [row.rationale_id for row in candidate.rationales]
        if len(rationale_ids) != len(set(rationale_ids)):
            findings.append("DUPLICATE_RATIONALE_IDS")
        if isinstance(candidate, InvestigationV41):
            expected = pitch_ids
            reviewed = set(candidate.reviewed_pitch_evidence_ids)
            for evidence_id in sorted(expected - reviewed):
                findings.append(f"PITCH_REVIEW_COVERAGE_MISSING:{evidence_id}")
            for evidence_id in sorted(reviewed - expected):
                findings.append(f"PITCH_REVIEW_COVERAGE_EXTRA:{evidence_id}")
            for constraint in candidate.constraint_assessments:
                for evidence_id in constraint.pitch_evidence_ids:
                    if evidence_id not in pitch_ids:
                        findings.append(
                            f"INACCESSIBLE_PITCH_EVIDENCE:{evidence_id}"
                        )
                for evidence_id in [
                    *constraint.wiki_evidence_ids,
                    *constraint.historical_evidence_ids,
                ]:
                    if evidence_id not in registry:
                        findings.append(f"INACCESSIBLE_EVIDENCE:{evidence_id}")
            supplied = {
                row["entity"]["entity_id"]: set(row["entity"]["disclosure_ids"])
                for row in state.get("phase1_portfolio_candidates", [])
            }
            assessed = {
                row.portfolio_entity_id for row in candidate.portfolio_overlap_assessments
            }
            for entity_id in sorted(set(supplied) - assessed):
                findings.append(f"PORTFOLIO_OVERLAP_MISSING:{entity_id}")
            for overlap in candidate.portfolio_overlap_assessments:
                allowed = supplied.get(overlap.portfolio_entity_id)
                if allowed is None:
                    findings.append(
                        f"INACCESSIBLE_PORTFOLIO_ENTITY:{overlap.portfolio_entity_id}"
                    )
                elif not set(overlap.disclosure_ids) <= allowed:
                    findings.append(
                        f"INACCESSIBLE_PORTFOLIO_DISCLOSURE:{overlap.portfolio_entity_id}"
                    )
        return _dedupe(findings)

    def _investigate(self, state: WorkflowStateV4) -> WorkflowStateV4:
        turn = state.get("phase1_iteration", 0) + 1
        retrieval = V4RetrievalResult(**state["phase1_current_retrieval"])
        prompt_builder = (
            phase1_investigation_v41_prompt
            if self.is_v41
            else phase1_investigation_v4_prompt
        )
        prompt = prompt_builder(
            self.settings.investor_name,
            self.settings.episode_slug,
            self.settings.pitch,
            self.taxonomy,
            retrieval.wiki_evidence,
            retrieval.historical_evidence,
            state.get("investigation"),
            retrieval.warnings,
            state.get("phase1_portfolio_candidates", []),
        )
        schema_builder = (
            investigation_v41_json_schema
            if self.is_v41
            else investigation_v4_json_schema
        )
        request = GenerationRequest(
            "phase1",
            prompt,
            schema_builder(
                episode_slug=self.settings.episode_slug,
                pitch_evidence_ids=tuple(
                    row["evidence_id"]
                    for row in pitch_evidence_index(self.settings.pitch)
                ),
                wiki_evidence_ids=tuple(
                    evidence_id
                    for evidence_id in state.get("evidence_registry", {})
                    if evidence_id.startswith("W-")
                ),
                historical_evidence_ids=tuple(
                    evidence_id
                    for evidence_id in state.get("evidence_registry", {})
                    if evidence_id.startswith("H-")
                ),
                **(
                    {
                        "portfolio_entity_ids": tuple(
                            row["entity"]["entity_id"]
                            for row in state.get("phase1_portfolio_candidates", [])
                        ),
                        "portfolio_disclosure_ids": tuple(
                            disclosure_id
                            for row in state.get("phase1_portfolio_candidates", [])
                            for disclosure_id in row["entity"]["disclosure_ids"]
                        ),
                    }
                    if self.is_v41
                    else {}
                ),
            ),
            max_output_tokens=self.settings.phase1_max_output_tokens,
            reasoning_effort=self.settings.phase1_reasoning_effort,
        )
        result = self.phase1_provider.generate(request)
        self._record_call("phase1", turn, "investigation", request, result)
        findings = list(state.get("phase1_findings", []))
        candidate: InvestigationV4 | InvestigationV41 | None = None
        try:
            investigation_model = InvestigationV41 if self.is_v41 else InvestigationV4
            parsed_payload = result.parsed
            if self.is_v41:
                parsed_payload, normalization_findings = (
                    normalize_investigation_v41_payload(parsed_payload)
                )
                findings.extend(normalization_findings)
                parsed_payload, evidence_normalization_findings = (
                    normalize_investigation_v41_evidence_payload(
                        parsed_payload,
                        wiki_evidence_ids={
                            value
                            for value in state.get("evidence_registry", {})
                            if value.startswith("W-")
                        },
                        historical_evidence_ids={
                            value
                            for value in state.get("evidence_registry", {})
                            if value.startswith("H-")
                        },
                    )
                )
                findings.extend(evidence_normalization_findings)
            parsed = investigation_model.model_validate(parsed_payload)
            structural = self._investigation_findings(parsed, state)
            findings.extend(structural)
            blockers = [
                item for item in structural
                if item == "EPISODE_SLUG_MISMATCH"
                or item == "DUPLICATE_RATIONALE_IDS"
                or item.startswith("UNKNOWN_TAXONOMY_LABEL:")
            ]
            if blockers:
                raise ValueError("; ".join(blockers))
            candidate = parsed
        except Exception as exc:
            findings.append(f"PHASE1_CANDIDATE_INVALID: {exc}")

        prior = state.get("investigation")
        selected = candidate.model_dump(mode="json") if candidate is not None else prior
        signature = "|".join(
            sorted(
                [*(r.taxonomy_label for r in candidate.rationales), *candidate.searchable_questions]
            )
        ) if candidate is not None else state.get("phase1_signature", "")
        repeated = bool(candidate is not None and state.get("phase1_signature") == signature)
        cap = turn >= self.settings.phase1_max_iterations
        coverage_blockers = [
            item
            for item in (structural if candidate is not None else [])
            if item.startswith("PITCH_REVIEW_COVERAGE_")
            or item.startswith("PORTFOLIO_OVERLAP_MISSING:")
            or item.startswith("INACCESSIBLE_PORTFOLIO_")
        ]
        if (
            candidate is not None
            and candidate.information_sufficient
            and not coverage_blockers
        ):
            action = "freeze_accepted"
        elif repeated and selected is not None:
            findings.append("PHASE1_NO_NOVELTY_STOP")
            action = "freeze_provisional"
        elif cap and selected is not None:
            action = "freeze_provisional"
        elif selected is not None:
            action = "revisit"
        else:
            action = "failed" if cap else "revisit"
        return {
            "phase1_iteration": turn,
            "investigation": selected,
            "phase1_action": action,
            "phase1_candidate_action": action,
            "phase1_findings": _dedupe(findings),
            "phase1_feedback": coverage_blockers,
            "phase1_signature": signature,
            "usage": _usage(state, result),
            "usage_by_phase": _usage_by_phase(state, result, "phase1"),
            "events": _event(
                state,
                "phase1_investigation",
                turn=turn,
                valid=candidate is not None,
                action=action,
            ),
        }

    def _freeze_phase1(self, state: WorkflowStateV4) -> WorkflowStateV4:
        investigation_model = (
            InvestigationV43
            if self.is_v43
            else InvestigationV42
            if self.is_v42
            else InvestigationV41
            if self.is_v41
            else InvestigationV4
        )
        candidate = investigation_model.model_validate(state["investigation"])
        _, digest = freeze_model(
            self.settings.run_root / "phase1", "investigation", candidate
        )
        status = "accepted" if state["phase1_action"] == "freeze_accepted" else "provisional"
        return {
            "phase1_status": status,
            "phase2_status": (
                "not_run" if self.settings.execution_mode == "phase1_only" else "running"
            ),
            "investigation_sha256": digest,
            "events": _event(state, "phase1_frozen", status=status, sha256=digest),
        }

    def _record_mapping_call(
        self,
        attempt: int,
        request: GenerationRequest,
        result: Any,
    ) -> None:
        write_json(
            self.settings.run_root
            / "phase1"
            / "mapping"
            / ("model-response.json" if attempt == 1 else "model-response-retry.json"),
            {
                "prompt": request.prompt,
                "schema": request.schema,
                "parsed": result.parsed,
                "content": result.content,
                "usage": result.usage.model_dump(mode="json"),
                "elapsed_seconds": result.elapsed_seconds,
                "raw_metadata": result.raw_metadata,
            },
        )

    def _map_phase1_rationales(self, state: WorkflowStateV4) -> WorkflowStateV4:
        candidate = InvestigationV41.model_validate(state["investigation"])
        _, candidate_digest = freeze_model(
            self.settings.run_root / "phase1", "candidate-investigation", candidate
        )
        candidate_payload = candidate.model_dump(mode="json")
        candidate_ids = [row.rationale_id for row in candidate.rationales]
        cited_ids = {
            evidence_id
            for rationale in candidate.rationales
            for evidence_id in [
                *rationale.wiki_evidence_ids,
                *rationale.historical_evidence_ids,
            ]
        }
        registry = state.get("evidence_registry", {})
        cited_evidence = {
            evidence_id: registry[evidence_id]
            for evidence_id in sorted(cited_ids)
            if evidence_id in registry
        }
        taxonomy_families = {
            row["label"]: row["coarse_parent"] for row in self.taxonomy
        }
        taxonomy_definitions = {
            row["label"]: row["definition"] for row in self.taxonomy
        }
        activated_families = {
            taxonomy_families[row.taxonomy_label] for row in candidate.rationales
        }
        allowed_labels = sorted(
            label
            for label, family in taxonomy_families.items()
            if family in activated_families
        )
        base_prompt = phase1_rationale_mapping_v43_prompt(
            self.settings.investor_name,
            self.settings.episode_slug,
            self.taxonomy,
            candidate_payload,
            candidate_digest,
            cited_evidence,
        )
        schema = rationale_mapping_v43_json_schema(
            episode_slug=self.settings.episode_slug,
            candidate_investigation_sha256=candidate_digest,
            candidate_rationale_ids=candidate_ids,
            taxonomy_labels=allowed_labels,
        )
        findings = list(state.get("phase1_findings", []))
        accounting: dict[str, Any] = dict(state)
        accepted_mapping: RationaleMappingV43 | None = None
        for attempt in (1, 2):
            prompt = base_prompt
            if attempt == 2:
                prompt += (
                    "\nREPAIR INSTRUCTION: Return one complete JSON object matching the schema, "
                    "disposition every candidate exactly once, and remain within its taxonomy family."
                )
            request = GenerationRequest(
                "phase1_mapping",
                prompt,
                schema,
                max_output_tokens=self.settings.phase1_max_output_tokens,
                reasoning_effort=self.settings.phase1_reasoning_effort,
            )
            result = self.phase1_provider.generate(request)
            self._record_mapping_call(attempt, request, result)
            accounting["usage"] = _usage(accounting, result)
            accounting["usage_by_phase"] = _usage_by_phase(accounting, result, "phase1")
            try:
                normalized, normalization_findings = (
                    normalize_rationale_mapping_v43_payload(result.parsed)
                )
                findings.extend(normalization_findings)
                parsed = RationaleMappingV43.model_validate(normalized)
                if parsed.episode_slug != self.settings.episode_slug:
                    raise ValueError("episode slug mismatch")
                if parsed.candidate_investigation_sha256 != candidate_digest:
                    raise ValueError("candidate investigation hash mismatch")
                parsed, constraint_findings = constrain_rationale_mapping_v43(
                    candidate,
                    parsed,
                    taxonomy=taxonomy_families,
                    definitions=taxonomy_definitions,
                )
                findings.extend(constraint_findings)
                apply_rationale_mapping_v43(
                    candidate,
                    parsed,
                    taxonomy=taxonomy_families,
                    mapping_sha256="0" * 64,
                )
                accepted_mapping = parsed
                break
            except Exception as exc:
                findings.append(f"PHASE1_RATIONALE_MAPPING_INVALID_ATTEMPT_{attempt}: {exc}")

        fallback = accepted_mapping is None
        if fallback:
            findings.append("PHASE1_RATIONALE_MAPPING_FALLBACK_ORIGINAL")
            accepted_mapping = RationaleMappingV43(
                schema_version="rationale-mapping-v4.3",
                episode_slug=self.settings.episode_slug,
                candidate_investigation_sha256=candidate_digest,
                dispositions=[
                    {
                        "rationale_id": row.rationale_id,
                        "action": "keep",
                        "target_taxonomy_label": row.taxonomy_label,
                        "merge_into_rationale_id": None,
                        "winning_definition": next(
                            item["definition"]
                            for item in self.taxonomy
                            if item["label"] == row.taxonomy_label
                        ),
                        "rejected_alternatives": [],
                        "mapping_justification": (
                            "The mapping response was unusable; the original evidence-linked "
                            "candidate is retained provisionally."
                        ),
                    }
                    for row in candidate.rationales
                ],
                mapping_summary=(
                    "Fallback mapping retained the complete original candidate rationale set."
                ),
            )
        _, mapping_digest = freeze_model(
            self.settings.run_root / "phase1", "rationale-mapping", accepted_mapping
        )
        final = apply_rationale_mapping_v43(
            candidate,
            accepted_mapping,
            taxonomy=taxonomy_families,
            mapping_sha256=mapping_digest,
        )
        if fallback:
            final = final.model_copy(update={"mapping_status": "fallback_original"})
        return {
            "investigation": final.model_dump(mode="json"),
            "candidate_investigation": candidate_payload,
            "candidate_investigation_sha256": candidate_digest,
            "rationale_mapping": accepted_mapping.model_dump(mode="json"),
            "rationale_mapping_sha256": mapping_digest,
            "phase1_action": (
                "freeze_provisional" if fallback else state["phase1_candidate_action"]
            ),
            "phase1_findings": _dedupe(findings),
            "usage": accounting["usage"],
            "usage_by_phase": accounting["usage_by_phase"],
            "events": _event(
                state,
                "phase1_rationale_mapping",
                valid=not fallback,
                fallback=fallback,
                relabeled=final.relabeled_candidate_count,
                merged=final.merged_candidate_count,
            ),
        }

    def _record_lock_call(
        self,
        attempt: int,
        request: GenerationRequest,
        result: Any,
    ) -> None:
        write_json(
            self.settings.run_root
            / "phase1"
            / "lock"
            / ("model-response.json" if attempt == 1 else "model-response-retry.json"),
            {
                "prompt": request.prompt,
                "schema": request.schema,
                "parsed": result.parsed,
                "content": result.content,
                "usage": result.usage.model_dump(mode="json"),
                "elapsed_seconds": result.elapsed_seconds,
                "raw_metadata": result.raw_metadata,
            },
        )

    def _lock_phase1_rationales(self, state: WorkflowStateV4) -> WorkflowStateV4:
        candidate = InvestigationV41.model_validate(state["investigation"])
        _, candidate_digest = freeze_model(
            self.settings.run_root / "phase1", "candidate-investigation", candidate
        )
        candidate_payload = candidate.model_dump(mode="json")
        candidate_ids = [row.rationale_id for row in candidate.rationales]
        cited_ids = {
            evidence_id
            for rationale in candidate.rationales
            for evidence_id in [
                *rationale.wiki_evidence_ids,
                *rationale.historical_evidence_ids,
            ]
        }
        registry = state.get("evidence_registry", {})
        cited_evidence = {
            evidence_id: registry[evidence_id]
            for evidence_id in sorted(cited_ids)
            if evidence_id in registry
        }
        base_prompt = phase1_rationale_lock_v42_prompt(
            self.settings.investor_name,
            self.settings.episode_slug,
            self.settings.pitch,
            self.taxonomy,
            candidate_payload,
            candidate_digest,
            cited_evidence,
        )
        schema = rationale_lock_v42_json_schema(
            episode_slug=self.settings.episode_slug,
            candidate_investigation_sha256=candidate_digest,
            candidate_rationale_ids=candidate_ids,
        )
        findings = list(state.get("phase1_findings", []))
        accounting: dict[str, Any] = dict(state)
        accepted_lock: RationaleLockV42 | None = None
        for attempt in (1, 2):
            prompt = base_prompt
            if attempt == 2:
                prompt += "\nREPAIR INSTRUCTION: Return one complete JSON object matching the schema and disposition every candidate rationale exactly once."
            request = GenerationRequest(
                "phase1_lock",
                prompt,
                schema,
                max_output_tokens=self.settings.phase1_max_output_tokens,
                reasoning_effort=self.settings.phase1_reasoning_effort,
            )
            result = self.phase1_provider.generate(request)
            self._record_lock_call(attempt, request, result)
            accounting["usage"] = _usage(accounting, result)
            accounting["usage_by_phase"] = _usage_by_phase(accounting, result, "phase1")
            try:
                normalized_lock, normalization_findings = (
                    normalize_rationale_lock_v42_payload(result.parsed)
                )
                findings.extend(normalization_findings)
                parsed = RationaleLockV42.model_validate(normalized_lock)
                if parsed.episode_slug != self.settings.episode_slug:
                    raise ValueError("episode slug mismatch")
                if parsed.candidate_investigation_sha256 != candidate_digest:
                    raise ValueError("candidate investigation hash mismatch")
                disposition_ids = {row.rationale_id for row in parsed.dispositions}
                if disposition_ids != set(candidate_ids):
                    raise ValueError("candidate rationales must be dispositioned exactly once")
                accepted_lock = parsed
                break
            except Exception as exc:
                findings.append(f"PHASE1_RATIONALE_LOCK_INVALID_ATTEMPT_{attempt}: {exc}")
        if accepted_lock is None:
            return {
                "phase1_action": "failed",
                "phase1_findings": _dedupe(findings),
                "candidate_investigation": candidate_payload,
                "candidate_investigation_sha256": candidate_digest,
                "usage": accounting["usage"],
                "usage_by_phase": accounting["usage_by_phase"],
                "events": _event(state, "phase1_rationale_lock", valid=False),
            }
        _, lock_digest = freeze_model(
            self.settings.run_root / "phase1", "rationale-lock", accepted_lock
        )
        by_id = {row.rationale_id: row for row in accepted_lock.dispositions}
        locked = []
        for rationale in candidate.rationales:
            disposition = by_id[rationale.rationale_id]
            if disposition.decision != "locked":
                continue
            locked.append(
                rationale.model_copy(
                    update={
                        "direction": disposition.direction,
                        "salience": disposition.salience,
                        "confidence": disposition.confidence,
                    }
                ).model_dump(mode="json")
            )
        final_payload = candidate_payload | {
            "schema_version": "investigation-v4.2",
            "rationales": locked,
            "candidate_rationale_ids": candidate_ids,
            "candidate_investigation_sha256": candidate_digest,
            "rationale_lock_sha256": lock_digest,
            "rejected_candidate_count": len(candidate_ids) - len(locked),
        }
        final = InvestigationV42.model_validate(final_payload)
        return {
            "investigation": final.model_dump(mode="json"),
            "candidate_investigation": candidate_payload,
            "candidate_investigation_sha256": candidate_digest,
            "rationale_lock": accepted_lock.model_dump(mode="json"),
            "rationale_lock_sha256": lock_digest,
            "phase1_action": state["phase1_candidate_action"],
            "phase1_findings": _dedupe(findings),
            "usage": accounting["usage"],
            "usage_by_phase": accounting["usage_by_phase"],
            "events": _event(
                state,
                "phase1_rationale_lock",
                valid=True,
                locked=len(locked),
                rejected=len(candidate_ids) - len(locked),
            ),
        }

    def _plan_phase2(self, state: WorkflowStateV4) -> WorkflowStateV4:
        turn = state.get("phase2_iteration", 0) + 1
        previous = state.get("decision")
        prompt_builder = phase2_plan_v41_prompt if self.is_v41 else phase2_plan_v4_prompt
        prompt = prompt_builder(
            self.settings.investor_name,
            self.settings.episode_slug,
            self.settings.pitch,
            state["investigation"],
            previous,
            (previous or {}).get("searchable_questions", []),
            (previous or {}).get("missing_considerations", []),
            state.get("rationale_association_annotations"),
            state.get("rehearsal_context"),
        )
        candidate, results, plan_findings = self._generate_plan_with_repair(
            artifact_phase="phase2",
            request_phase="phase2_plan",
            turn=turn,
            prompt=prompt,
            provider=self.phase2_provider,
            max_output_tokens=self.settings.phase2_planning_max_output_tokens,
            reasoning_effort=self.settings.phase2_planning_reasoning_effort,
        )
        findings = [*state.get("phase2_findings", []), *plan_findings]
        if candidate is None:
            findings.append("PHASE2_PLAN_FALLBACK_USED")
            candidate = RetrievalPlanV4(
                questions=["What evidence controls the capital-commitment decision?"],
                wiki_queries=["investment threshold exceptions"],
                precedent_queries=["similar investment decision opposing case"],
                continuation_focus="Recover a focused decision assessment.",
            )
        usage, usage_by_phase = self._account_plan_results(state, results, "phase2")
        return {
            "phase2_plan": candidate.model_dump(mode="json"),
            "phase2_findings": _dedupe(findings),
            "usage": usage,
            "usage_by_phase": usage_by_phase,
            "events": _event(state, "phase2_plan", turn=turn),
        }

    def _retrieve_phase2(self, state: WorkflowStateV4) -> WorkflowStateV4:
        return self._retrieve(state, "phase2")

    def _decision_findings(
        self, candidate: DecisionV4 | DecisionV41, state: WorkflowStateV4
    ) -> list[str]:
        findings: list[str] = []
        if candidate.episode_slug != self.settings.episode_slug:
            findings.append("EPISODE_SLUG_MISMATCH")
        if candidate.investigation_sha256 != state["investigation_sha256"]:
            findings.append("INVESTIGATION_SHA256_MISMATCH")
        if candidate.information_sufficient and candidate.searchable_questions:
            findings.append("SUFFICIENT_WITH_SEARCHABLE_QUESTIONS")
        if candidate.information_sufficient and candidate.next_search_objectives:
            findings.append("SUFFICIENT_WITH_NEXT_SEARCH_OBJECTIVES")
        if not candidate.information_sufficient and not candidate.searchable_questions:
            findings.append("INSUFFICIENT_WITHOUT_SEARCHABLE_QUESTION")
        allowed = {
            row["rationale_id"] for row in state["investigation"]["rationales"]
        } | {
            row["observation_id"]
            for row in state["investigation"]["unmapped_observations"]
        }
        for rationale_id in candidate.controlling_rationale_ids:
            if rationale_id not in allowed:
                findings.append(f"UNKNOWN_CONTROLLING_RATIONALE:{rationale_id}")
        for rationale_id in candidate.founder_exception_rationale_ids:
            if rationale_id not in allowed:
                findings.append(f"UNKNOWN_FOUNDER_EXCEPTION_RATIONALE:{rationale_id}")
        registry = state.get("evidence_registry", {})
        pitch_ids = {row["evidence_id"] for row in pitch_evidence_index(self.settings.pitch)}
        for basis in candidate.evidence_basis:
            for evidence_id in basis.evidence_ids:
                if evidence_id.startswith("P-") and evidence_id not in pitch_ids:
                    findings.append(f"INACCESSIBLE_PITCH_EVIDENCE:{evidence_id}")
                elif not evidence_id.startswith("P-") and evidence_id not in registry:
                    findings.append(f"INACCESSIBLE_EVIDENCE:{evidence_id}")
        for evidence_id in candidate.founder_exception_precedent_ids:
            if evidence_id not in registry:
                findings.append(f"INACCESSIBLE_EVIDENCE:{evidence_id}")
            elif (
                isinstance(candidate, DecisionV41)
                and candidate.decision_path == "founder_conviction_exception"
                and registry[evidence_id].get("decision_status") != "In"
            ):
                findings.append(
                    f"FOUNDER_EXCEPTION_PRECEDENT_NOT_IN:{evidence_id}"
                )
        if isinstance(candidate, DecisionV41):
            investigation_payload = state["investigation"]
            if investigation_payload.get("schema_version") == "investigation-v5":
                investigation_payload = self._v5_source_payload(investigation_payload)
            investigation = InvestigationV41.model_validate(investigation_payload)
            try:
                validate_decision_v41_against_investigation(
                    candidate, investigation
                )
            except ValueError as exc:
                message = str(exc)
                if "triggered hard constraint" in message:
                    identifiers = message.rsplit(":", 1)[-1].strip()
                    findings.append(
                        f"IN_CONFLICTS_WITH_TRIGGERED_HARD_CONSTRAINT:{identifiers}"
                    )
                else:
                    findings.append(f"CONSTRAINT_BINDING_MISMATCH:{message}")
        return _dedupe(findings)

    @staticmethod
    def _v5_source_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Recover the immutable canonical source contract used by v4.1 guardrails."""
        source = dict(payload)
        source["schema_version"] = source.pop("source_schema_version")
        for key in (
            "source_investigation_sha256",
            "association_training_episode_slugs",
            "association_thresholds",
            "episode_level_associations",
        ):
            source.pop(key, None)
        source["rationales"] = [
            {
                key: value
                for key, value in row.items()
                if key not in {"associated_rationales", "pitch_evidence"}
            }
            for row in source.get("rationales", [])
        ]
        source["material_statement_coverage"] = [
            {key: value for key, value in row.items() if key != "associated_rationales"}
            for row in source.get("material_statement_coverage", [])
        ]
        source["unmapped_observations"] = [
            {
                key: value
                for key, value in row.items()
                if key not in {"associated_rationales", "association_status"}
            }
            for row in source.get("unmapped_observations", [])
        ]
        return source

    def _decide(self, state: WorkflowStateV4) -> WorkflowStateV4:
        turn = state.get("phase2_iteration", 0) + 1
        retrieval = V4RetrievalResult(**state["phase2_current_retrieval"])
        prompt_builder = (
            phase2_decision_v41_prompt
            if self.is_v41
            else phase2_decision_v4_prompt
        )
        prompt = prompt_builder(
            self.settings.investor_name,
            self.settings.episode_slug,
            self.settings.pitch,
            state["investigation"],
            state["investigation_sha256"],
            retrieval.wiki_evidence,
            retrieval.historical_evidence,
            state.get("decision"),
            retrieval.warnings,
            state.get("rationale_association_annotations"),
            state.get("rehearsal_context"),
        )
        schema_builder = (
            decision_v41_json_schema if self.is_v41 else decision_v4_json_schema
        )
        schema_arguments = {
            "episode_slug": self.settings.episode_slug,
            "investigation_sha256": state["investigation_sha256"],
            "pitch_evidence_ids": tuple(
                row["evidence_id"]
                for row in pitch_evidence_index(self.settings.pitch)
            ),
            "wiki_evidence_ids": tuple(
                evidence_id
                for evidence_id in state.get("evidence_registry", {})
                if evidence_id.startswith("W-")
            ),
            "historical_evidence_ids": tuple(
                evidence_id
                for evidence_id in state.get("evidence_registry", {})
                if evidence_id.startswith("H-")
            ),
            "rationale_ids": tuple(
                [
                    row["rationale_id"]
                    for row in state["investigation"]["rationales"]
                ]
                + [
                    row["observation_id"]
                    for row in state["investigation"]["unmapped_observations"]
                ]
            ),
        }
        if self.is_v41:
            schema_arguments["constraint_ids"] = tuple(
                row["constraint_id"]
                for row in state["investigation"]["constraint_assessments"]
            )
        request = GenerationRequest(
            "phase2",
            prompt,
            schema_builder(**schema_arguments),
            max_output_tokens=self.settings.phase2_max_output_tokens,
            reasoning_effort=self.settings.phase2_reasoning_effort,
        )
        result = self.phase2_provider.generate(request)
        self._record_call("phase2", turn, "decision", request, result)
        findings = list(state.get("phase2_findings", []))
        candidate: DecisionV4 | DecisionV41 | None = None
        try:
            decision_model = DecisionV41 if self.is_v41 else DecisionV4
            parsed = decision_model.model_validate(result.parsed)
            structural = self._decision_findings(parsed, state)
            findings.extend(structural)
            blockers = [
                item for item in structural
                if item in {"EPISODE_SLUG_MISMATCH", "INVESTIGATION_SHA256_MISMATCH"}
                or item.startswith("UNKNOWN_CONTROLLING_RATIONALE:")
                or item.startswith("UNKNOWN_FOUNDER_EXCEPTION_RATIONALE:")
            ]
            if blockers:
                raise ValueError("; ".join(blockers))
            candidate = parsed
        except Exception as exc:
            findings.append(f"PHASE2_CANDIDATE_INVALID: {exc}")

        prior = state.get("decision")
        signature = "|".join(sorted(candidate.searchable_questions)) if candidate else state.get("phase2_signature", "")
        repeated = bool(candidate is not None and state.get("phase2_signature") == signature and signature)
        prior_streak = state.get("phase2_decision_streak", 0)
        same_decision = bool(
            candidate is not None
            and prior is not None
            and candidate.decision == prior.get("decision")
        )
        candidate_streak = prior_streak + 1 if same_decision else 1
        no_novelty_reversal = bool(
            candidate is not None
            and prior is not None
            and prior_streak >= 2
            and repeated
            and candidate.decision != prior.get("decision")
            and set(candidate.controlling_rationale_ids).issubset(
                set(prior.get("controlling_rationale_ids", []))
            )
        )
        if no_novelty_reversal:
            selected = prior
            selected_streak = prior_streak
            findings.append(
                "PHASE2_NO_NOVELTY_REVERSAL_RETAINED_STABLE_DECISION"
            )
        else:
            selected = candidate.model_dump(mode="json") if candidate is not None else prior
            selected_streak = candidate_streak if candidate is not None else prior_streak
        selected_investigation = state.get("investigation")
        selected_investigation_sha256 = state.get("investigation_sha256")
        if candidate is None and prior is not None:
            prior_sha = prior.get("investigation_sha256")
            if prior_sha != state.get("investigation_sha256"):
                selected_investigation = state.get("decision_bound_investigation")
                selected_investigation_sha256 = state.get(
                    "decision_bound_investigation_sha256"
                )
                if selected_investigation is None or selected_investigation_sha256 != prior_sha:
                    selected = None
                    findings.append("PHASE2_FALLBACK_BINDING_UNAVAILABLE")
                else:
                    freeze_model(
                        self.settings.run_root / "phase1",
                        "investigation",
                        (
                            InvestigationV41
                            if self.is_v41
                            else InvestigationV4
                        ).model_validate(selected_investigation),
                    )
                    findings.append("PHASE1_ROLLED_BACK_TO_RETAIN_DECISION_BINDING")
        cap = turn >= self.settings.phase2_max_iterations
        can_reopen = (
            candidate is not None
            and candidate.phase1_reopen_recommended
            and not candidate.information_sufficient
            and not state.get("phase1_replay_frozen", False)
            and state.get("phase1_iteration", 0) < self.settings.phase1_max_iterations
            and not cap
        )
        guardrail_blockers = [
            item
            for item in (structural if candidate is not None else [])
            if item.startswith("IN_CONFLICTS_WITH_TRIGGERED_HARD_CONSTRAINT:")
            or item.startswith("CONSTRAINT_BINDING_MISMATCH:")
            or item.startswith("FOUNDER_EXCEPTION_PRECEDENT_NOT_IN:")
        ]
        if can_reopen:
            action = "reopen_phase1"
        elif (
            candidate is not None
            and candidate.information_sufficient
            and not guardrail_blockers
        ):
            action = "freeze_accepted"
        elif repeated and selected is not None:
            findings.append("PHASE2_NO_NOVELTY_STOP")
            action = "freeze_provisional"
        elif cap and selected is not None:
            action = "freeze_provisional"
        elif selected is not None:
            action = "revisit"
        else:
            action = "failed" if cap else "revisit"
        return {
            "phase2_iteration": turn,
            "decision": selected,
            "investigation": selected_investigation,
            "investigation_sha256": selected_investigation_sha256,
            "decision_bound_investigation": (
                state["investigation"]
                if candidate is not None
                else state.get("decision_bound_investigation")
            ),
            "decision_bound_investigation_sha256": (
                state["investigation_sha256"]
                if candidate is not None
                else state.get("decision_bound_investigation_sha256")
            ),
            "phase2_action": action,
            "phase2_findings": _dedupe(findings),
            "phase1_feedback": (
                candidate.missing_considerations if can_reopen and candidate else []
            ),
            "phase2_signature": signature,
            "phase2_decision_streak": selected_streak,
            "usage": _usage(state, result),
            "usage_by_phase": _usage_by_phase(state, result, "phase2"),
            "events": _event(
                state,
                "phase2_decision",
                turn=turn,
                valid=candidate is not None,
                action=action,
            ),
        }

    def _freeze_phase2(self, state: WorkflowStateV4) -> WorkflowStateV4:
        decision_model = DecisionV41 if self.is_v41 else DecisionV4
        candidate = decision_model.model_validate(state["decision"])
        _, digest = freeze_model(self.settings.run_root / "phase2", "decision", candidate)
        status = "accepted" if state["phase2_action"] == "freeze_accepted" else "provisional"
        return {
            "phase2_status": status,
            "decision_sha256": digest,
            "events": _event(state, "phase2_frozen", status=status, sha256=digest),
        }

    def _reflection_route(self, state: WorkflowStateV4) -> str:
        return "reflect" if state["investigation"].get("unmapped_observations") else "done"

    def _reflect_taxonomy(self, state: WorkflowStateV4) -> WorkflowStateV4:
        prompt = taxonomy_reflection_v4_prompt(
            self.settings.investor_name,
            self.settings.episode_slug,
            self.settings.pitch,
            self.taxonomy,
            state["investigation"],
            state["decision"],
            state["decision_sha256"],
        )
        request = GenerationRequest(
            "taxonomy_reflection",
            prompt,
            TaxonomyReflectionV4.model_json_schema(),
            max_output_tokens=self.settings.phase2_max_output_tokens,
            reasoning_effort=self.settings.phase2_reasoning_effort,
        )
        result = self.phase2_provider.generate(request)
        self._record_call("reflection", 1, "taxonomy", request, result)
        findings = list(state.get("reflection_findings", []))
        try:
            reflection = TaxonomyReflectionV4.model_validate(result.parsed)
            if reflection.episode_slug != self.settings.episode_slug:
                raise ValueError("episode slug mismatch")
            if reflection.decision_sha256 != state["decision_sha256"]:
                raise ValueError("decision hash mismatch")
        except Exception as exc:
            findings.append(f"TAXONOMY_REFLECTION_INVALID: {exc}")
            reflection = TaxonomyReflectionV4(
                schema_version="taxonomy-reflection-v4",
                episode_slug=self.settings.episode_slug,
                decision_sha256=state["decision_sha256"],
                proposals=[],
                review_summary="No valid proposal was retained; human review may revisit the unmapped observation.",
            )
        write_json(
            self.settings.run_root / "reflection" / "taxonomy-proposals.json",
            reflection.model_dump(mode="json"),
        )
        return {
            "taxonomy_reflection": reflection.model_dump(mode="json"),
            "reflection_findings": findings,
            "usage": _usage(state, result),
            "usage_by_phase": _usage_by_phase(state, result, "phase2"),
            "events": _event(
                state, "taxonomy_reflection", proposals=len(reflection.proposals)
            ),
        }

    def _route_phase1(self, state: WorkflowStateV4) -> str:
        action = state["phase1_action"]
        if (
            self.is_v43
            and action in {"freeze_accepted", "freeze_provisional"}
            and "rationale_mapping" not in state
        ):
            return "map"
        if (
            self.is_v42
            and action in {"freeze_accepted", "freeze_provisional"}
            and "rationale_lock" not in state
        ):
            return "lock"
        return action

    def _route_after_phase1_freeze(self, state: WorkflowStateV4) -> str:
        return "done" if self.settings.execution_mode == "phase1_only" else "phase2"

    @staticmethod
    def _route_phase2(state: WorkflowStateV4) -> str:
        return state["phase2_action"]

    def _build(self) -> StateGraph:
        builder = StateGraph(WorkflowStateV4)
        builder.add_node("phase1_plan", self._plan_phase1)
        builder.add_node("phase1_retrieve", self._retrieve_phase1)
        builder.add_node("phase1_portfolio_retrieve", self._retrieve_portfolio)
        builder.add_node("phase1_investigate", self._investigate)
        builder.add_node("phase1_rationale_lock", self._lock_phase1_rationales)
        builder.add_node("phase1_rationale_mapping", self._map_phase1_rationales)
        builder.add_node("phase1_freeze", self._freeze_phase1)
        builder.add_node("phase2_plan", self._plan_phase2)
        builder.add_node("phase2_retrieve", self._retrieve_phase2)
        builder.add_node("phase2_decide", self._decide)
        builder.add_node("phase2_freeze", self._freeze_phase2)
        builder.add_node("taxonomy_reflection", self._reflect_taxonomy)
        builder.add_edge(START, "phase1_plan")
        builder.add_edge("phase1_plan", "phase1_retrieve")
        builder.add_edge("phase1_retrieve", "phase1_portfolio_retrieve")
        builder.add_edge("phase1_portfolio_retrieve", "phase1_investigate")
        builder.add_conditional_edges(
            "phase1_investigate",
            self._route_phase1,
            {
                "revisit": "phase1_plan",
                "lock": "phase1_rationale_lock",
                "map": "phase1_rationale_mapping",
                "freeze_accepted": "phase1_freeze",
                "freeze_provisional": "phase1_freeze",
                "failed": END,
            },
        )
        builder.add_conditional_edges(
            "phase1_rationale_lock",
            self._route_phase1,
            {
                "freeze_accepted": "phase1_freeze",
                "freeze_provisional": "phase1_freeze",
                "failed": END,
            },
        )
        builder.add_conditional_edges(
            "phase1_rationale_mapping",
            self._route_phase1,
            {
                "freeze_accepted": "phase1_freeze",
                "freeze_provisional": "phase1_freeze",
                "failed": END,
            },
        )
        builder.add_conditional_edges(
            "phase1_freeze",
            self._route_after_phase1_freeze,
            {"phase2": "phase2_plan", "done": END},
        )
        builder.add_edge("phase2_plan", "phase2_retrieve")
        builder.add_edge("phase2_retrieve", "phase2_decide")
        builder.add_conditional_edges(
            "phase2_decide",
            self._route_phase2,
            {
                "revisit": "phase2_plan",
                "reopen_phase1": "phase1_plan",
                "freeze_accepted": "phase2_freeze",
                "freeze_provisional": "phase2_freeze",
                "failed": END,
            },
        )
        builder.add_conditional_edges(
            "phase2_freeze",
            self._reflection_route,
            {"reflect": "taxonomy_reflection", "done": END},
        )
        builder.add_edge("taxonomy_reflection", END)
        return builder

    def _build_phase2(self) -> StateGraph:
        builder = StateGraph(WorkflowStateV4)
        builder.add_node("phase2_plan", self._plan_phase2)
        builder.add_node("phase2_retrieve", self._retrieve_phase2)
        builder.add_node("phase2_decide", self._decide)
        builder.add_node("phase2_freeze", self._freeze_phase2)
        builder.add_node("taxonomy_reflection", self._reflect_taxonomy)
        builder.add_edge(START, "phase2_plan")
        builder.add_edge("phase2_plan", "phase2_retrieve")
        builder.add_edge("phase2_retrieve", "phase2_decide")
        builder.add_conditional_edges(
            "phase2_decide",
            self._route_phase2,
            {
                "revisit": "phase2_plan",
                "reopen_phase1": "phase2_plan",
                "freeze_accepted": "phase2_freeze",
                "freeze_provisional": "phase2_freeze",
                "failed": END,
            },
        )
        builder.add_conditional_edges(
            "phase2_freeze",
            self._reflection_route,
            {"reflect": "taxonomy_reflection", "done": END},
        )
        builder.add_edge("taxonomy_reflection", END)
        return builder

    def _initial_state(self) -> WorkflowStateV4:
        zero: dict[str, int | float] = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }
        return {
            "phase1_iteration": 0,
            "phase2_iteration": 0,
            "phase1_status": "running",
            "phase2_status": "pending",
            "phase1_findings": list(self.settings.initial_phase1_quality_findings),
            "phase2_findings": [],
            "phase1_feedback": [],
            "evidence_registry": {},
            "phase1_portfolio_candidates": [],
            "portfolio_filtered_manifest": {},
            "precedent_manifest_sha256": self.settings.precedent_manifest_sha256 or "",
            "accessible_precedent_count": self.settings.accessible_precedent_count,
            "events": [],
            "usage": dict(zero),
            "usage_by_phase": {"phase1": dict(zero), "phase2": dict(zero)},
        }

    def _persist_result(self, result: WorkflowStateV4) -> WorkflowStateV4:
        if result.get("phase1_action") == "failed":
            result["phase1_status"] = "failed"
        if result.get("phase2_action") == "failed":
            result["phase2_status"] = "failed"
        summary = {
            "contract_version": self.settings.contract_version,
            "episode_slug": self.settings.episode_slug,
            "phase1_status": result.get("phase1_status"),
            "phase1_iterations": result.get("phase1_iteration"),
            "phase1_findings": result.get("phase1_findings", []),
            "phase2_status": result.get("phase2_status"),
            "phase2_iterations": result.get("phase2_iteration"),
            "phase2_findings": result.get("phase2_findings", []),
            "usage": result.get("usage", {}),
            "usage_by_phase": result.get("usage_by_phase", {}),
            "decision": result.get("decision", {}),
            "taxonomy_reflection": result.get("taxonomy_reflection"),
        }
        write_json(self.settings.run_root / "summary.json", summary)
        write_json(self.settings.run_root / "state.json", result)
        return result

    def invoke(self, thread_id: str) -> WorkflowStateV4:
        result = self.graph.invoke(
            self._initial_state(), {"configurable": {"thread_id": thread_id}}
        )
        return self._persist_result(result)

    def resume(self, thread_id: str) -> WorkflowStateV4:
        result = self.graph.invoke(
            None, {"configurable": {"thread_id": thread_id}}
        )
        return self._persist_result(result)

    def invoke_phase2(
        self,
        thread_id: str,
        investigation: InvestigationV4,
        investigation_sha256: str,
        *,
        phase1_status: str,
        phase1_exact_reads: list[dict[str, Any]] | None = None,
        phase1_historical_evidence: list[dict[str, Any]] | None = None,
        phase1_precedent_reads: list[dict[str, Any]] | None = None,
        phase1_state: dict[str, Any] | None = None,
        rationale_association_annotations: dict[str, Any] | None = None,
        rehearsal_context: Phase2RehearsalContext | None = None,
    ) -> WorkflowStateV4:
        zero: dict[str, int | float] = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }
        source = phase1_state or {}
        annotations = rationale_association_annotations or source.get(
            "rationale_association_annotations"
        )
        if annotations is not None:
            if annotations.get("episode_slug") != self.settings.episode_slug:
                raise ValueError("association annotation episode mismatch")
            if annotations.get("investigation_sha256") != investigation_sha256:
                raise ValueError("association annotation investigation hash mismatch")
            validated_annotations = RationaleAssociationAnnotations.model_validate(
                annotations
            )
            if self.settings.episode_slug in validated_annotations.training_episode_slugs:
                raise ValueError("held episode appears in association training provenance")
            annotations = validated_annotations.model_dump(mode="json")
        carried_keys = (
            "phase1_iteration",
            "phase1_action",
            "phase1_findings",
            "phase1_feedback",
            "phase1_signature",
            "phase1_plan",
            "phase1_current_retrieval",
            "phase1_retrieval_warnings",
            "phase1_precedent_searches",
            "phase1_precedent_reads",
            "phase1_historical_evidence",
            "exact_reads",
            "evidence_registry",
            "phase1_portfolio_candidates",
            "portfolio_filtered_manifest",
            "precedent_manifest_sha256",
            "accessible_precedent_count",
        )
        carried = {key: source[key] for key in carried_keys if key in source}
        phase1_usage = dict(source.get("usage_by_phase", {}).get("phase1", zero))
        initial: WorkflowStateV4 = {
            **self._initial_state(),
            **carried,
            "phase1_status": phase1_status,
            "phase2_status": "pending",
            "phase1_replay_frozen": True,
            "investigation": investigation.model_dump(mode="json"),
            "investigation_sha256": investigation_sha256,
            "decision": {},
            **(
                {"rehearsal_context": rehearsal_context.model_dump(mode="json")}
                if rehearsal_context is not None
                else {}
            ),
            **(
                {"rationale_association_annotations": annotations}
                if annotations is not None
                else {}
            ),
            "phase2_iteration": 0,
            "phase2_findings": [],
            "phase2_precedent_searches": [],
            "phase2_precedent_reads": [],
            "phase2_historical_evidence": [],
            "phase1_historical_evidence": carried.get(
                "phase1_historical_evidence", phase1_historical_evidence or []
            ),
            "phase1_precedent_reads": carried.get(
                "phase1_precedent_reads", phase1_precedent_reads or []
            ),
            "exact_reads": carried.get("exact_reads", phase1_exact_reads or []),
            "events": [
                {
                    "sequence": 1,
                    "kind": "phase2_replay_started",
                    "investigation_sha256": investigation_sha256,
                }
            ],
            "usage": dict(phase1_usage),
            "usage_by_phase": {
                "phase1": dict(phase1_usage),
                "phase2": dict(zero),
            },
        }
        result = self.phase2_graph.invoke(
            initial, {"configurable": {"thread_id": thread_id}}
        )
        return self._persist_result(result)

    def invoke_phase2_v5(
        self,
        thread_id: str,
        investigation: InvestigationV5,
        investigation_sha256: str,
        *,
        phase1_status: str,
        phase1_state: dict[str, Any] | None = None,
    ) -> WorkflowStateV4:
        """Run Phase 2 over a source-bound v5 enrichment without changing Phase 1."""
        if not self.is_v5:
            raise ValueError("invoke_phase2_v5 requires contract_version v5")
        if investigation.episode_slug != self.settings.episode_slug:
            raise ValueError("v5 investigation episode mismatch")
        if self.settings.episode_slug in investigation.association_training_episode_slugs:
            raise ValueError("held episode appears in association training provenance")
        self.is_v41 = investigation.source_schema_version == "investigation-v4.1"
        annotations = {
            "schema": "rationale-association-annotations-v1",
            "scientific_status": "fold_safe_advisory_hypotheses",
            "vc_slug": self.settings.investor_name,
            "episode_slug": investigation.episode_slug,
            "investigation_sha256": investigation_sha256,
            "thresholds": investigation.association_thresholds.model_dump(mode="json"),
            "training_episode_slugs": investigation.association_training_episode_slugs,
            "claim_annotations": [
                {
                    "rationale_id": row.rationale_id,
                    "source_taxonomy_label": row.taxonomy_label,
                    "associated_rationales": [
                        item.model_dump(mode="json") for item in row.associated_rationales
                    ],
                }
                for row in investigation.rationales
            ],
            "material_statement_annotations": [
                {
                    "pitch_evidence_id": row.pitch_evidence_id,
                    "mapped_ids": row.mapped_ids,
                    "source_taxonomy_labels": sorted({
                        rationale.taxonomy_label
                        for rationale in investigation.rationales
                        if rationale.rationale_id in row.mapped_ids
                    }),
                    "associated_rationales": [
                        item.model_dump(mode="json") for item in row.associated_rationales
                    ],
                }
                for row in investigation.material_statement_coverage
            ],
            "unmapped_observation_annotations": [
                {
                    "observation_id": row.observation_id,
                    "status": "unavailable_unmapped",
                    "associated_rationales": [],
                }
                for row in investigation.unmapped_observations
            ],
            "episode_level_associations": [
                row.model_dump(mode="json")
                for row in investigation.episode_level_associations
            ],
        }
        return self.invoke_phase2(
            thread_id,
            investigation,  # type: ignore[arg-type]
            investigation_sha256,
            phase1_status=phase1_status,
            phase1_state=phase1_state,
            rationale_association_annotations=annotations,
        )
