"""Bounded, checkpointed two-phase VC decision workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from .artifacts import freeze_model, write_json
from .prompts import (
    phase1_investigation_prompt,
    phase1_plan_prompt,
    phase1_retrieval_plan_prompt,
    phase2_plan_prompt,
    phase2_prompt,
)
from .precedents import PrecedentCorpus
from .providers.base import GenerationProvider, GenerationRequest
from .retrieval import HybridWikiIndex
from .schemas import (
    Decision,
    DecisionV2,
    DecisionV3,
    Investigation,
    InvestigationV2,
    InvestigationV3,
    Phase1RetrievalPlan,
    Phase2RetrievalPlan,
    QueryPlan,
    decision_quality_findings,
    decision_json_schema,
    decision_v2_json_schema,
    decision_v3_json_schema,
    investigation_json_schema,
    investigation_v2_json_schema,
    investigation_v3_json_schema,
    normalize_investigation_v2_payload,
    normalize_investigation_v3_payload,
    normalize_retrieval_plan_payload,
    validate_decision,
    validate_decision_v2,
    validate_decision_v3,
    validate_investigation,
    validate_investigation_v2,
    validate_investigation_v3,
)


class WorkflowState(TypedDict, total=False):
    phase1_iteration: int
    phase2_iteration: int
    phase1_status: str
    phase2_status: str
    phase1_action: str
    phase2_action: str
    query_plan: dict[str, Any]
    exact_reads: list[dict[str, Any]]
    investigation: dict[str, Any]
    investigation_sha256: str
    decision: dict[str, Any]
    phase1_findings: list[str]
    phase2_findings: list[str]
    phase1_audit_findings: list[str]
    phase2_audit_findings: list[str]
    phase1_quality_findings: list[str]
    query_history: list[str]
    retrieval_passes: list[dict[str, Any]]
    precedent_manifest_sha256: str
    accessible_precedent_count: int
    phase1_precedent_searches: list[dict[str, Any]]
    phase1_precedent_reads: list[dict[str, Any]]
    phase1_historical_evidence: list[dict[str, Any]]
    phase1_retrieval_findings: list[str]
    phase1_current_retrieval_findings: list[str]
    phase2_plan: dict[str, Any]
    phase2_exact_reads: list[dict[str, Any]]
    phase2_query_history: list[str]
    phase2_retrieval_passes: list[dict[str, Any]]
    phase2_precedent_searches: list[dict[str, Any]]
    phase2_precedent_reads: list[dict[str, Any]]
    phase2_historical_evidence: list[dict[str, Any]]
    phase2_retrieval_findings: list[str]
    phase2_current_retrieval_findings: list[str]
    phase2_quality_findings: list[str]
    phase2_no_retrieval_novelty: bool
    no_retrieval_novelty: bool
    events: list[dict[str, Any]]
    usage: dict[str, int | float]
    usage_by_phase: dict[str, dict[str, int | float]]


@dataclass(frozen=True)
class WorkflowSettings:
    episode_slug: str
    investor_name: str
    pitch: str
    taxonomy_labels: set[str]
    check_tiers: set[str]
    phase1_min_iterations: int
    phase1_max_iterations: int
    phase2_min_iterations: int
    phase2_max_iterations: int
    retrieval_top_k: int
    max_exact_reads: int
    run_root: Path
    taxonomy_records: tuple[dict[str, str], ...] = ()
    contract_version: str = "v1"
    execution_mode: Literal["full", "phase1_only"] = "full"
    initial_phase1_quality_findings: tuple[str, ...] = ()
    phase1_max_output_tokens: int | None = None
    phase1_reasoning_effort: Literal["low", "medium", "high"] | None = None
    phase1_planning_max_output_tokens: int = 768
    phase1_planning_reasoning_effort: Literal["low", "medium", "high"] | None = None
    phase2_max_output_tokens: int | None = None
    phase2_reasoning_effort: Literal["low", "medium", "high"] | None = None
    phase2_planning_max_output_tokens: int = 768
    phase2_planning_reasoning_effort: Literal["low", "medium", "high"] | None = None
    precedent_manifest_sha256: str | None = None
    accessible_precedent_count: int = 0
    phase1_max_precedent_searches: int = 0
    phase1_max_precedent_reads: int = 0
    phase2_max_precedent_searches: int = 0
    phase2_max_precedent_reads: int = 0
    allow_full_transcript: bool = False
    precedent_selection_policy: Literal["semantic", "contrastive"] = "semantic"
    precedent_candidate_pool_k: int = 30
    precedent_in_slots: int = 2
    portfolio_memory_enabled: bool = False
    portfolio_retrieval_top_k: int = 5
    portfolio_candidate_pool_k: int = 15
    precedent_out_slots: int = 2


def _usage(state: WorkflowState, result: Any) -> dict[str, int | float]:
    current = dict(state.get("usage", {}))
    for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
        current[key] = current.get(key, 0) + int(getattr(result.usage, key, 0))
    current["cost_usd"] = float(current.get("cost_usd", 0.0)) + float(
        getattr(result.usage, "cost_usd", 0.0)
    )
    return current


def _usage_by_phase(
    state: WorkflowState, result: Any, phase: Literal["phase1", "phase2"]
) -> dict[str, dict[str, int | float]]:
    zero = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    current = {
        name: {**zero, **values}
        for name, values in state.get("usage_by_phase", {}).items()
    }
    current.setdefault("phase1", dict(zero))
    current.setdefault("phase2", dict(zero))
    for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
        current[phase][key] = int(current[phase][key]) + int(
            getattr(result.usage, key, 0)
        )
    current[phase]["cost_usd"] = float(current[phase]["cost_usd"]) + float(
        getattr(result.usage, "cost_usd", 0.0)
    )
    return current


def _event(state: WorkflowState, kind: str, **payload: Any) -> list[dict[str, Any]]:
    return [*state.get("events", []), {"sequence": len(state.get("events", [])) + 1, "kind": kind, **payload}]


class VCDecisionWorkflow:
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
    ) -> None:
        if checkpointer is None:
            raise ValueError("checkpointer is required")
        phase1_provider = phase1_provider or provider
        phase2_provider = phase2_provider or provider
        if phase1_provider is None or phase2_provider is None:
            raise ValueError("both phase providers are required")
        self.settings = settings
        self.index = index
        self.precedent_corpus = precedent_corpus
        if precedent_corpus is not None:
            if precedent_corpus.target_slug != settings.episode_slug:
                raise ValueError("precedent corpus must be filtered to the target episode")
            accessible = precedent_corpus.list_episodes()
            if any(row.episode_slug == settings.episode_slug for row in accessible):
                raise ValueError("target episode remains accessible in precedent corpus")
            if settings.accessible_precedent_count != len(accessible):
                raise ValueError("accessible precedent count does not match corpus")
            if settings.precedent_manifest_sha256 != precedent_corpus.filtered_manifest().sha256:
                raise ValueError("precedent manifest hash does not match corpus")
        self.provider = provider or phase1_provider
        self.phase1_provider = phase1_provider
        self.phase2_provider = phase2_provider
        self.settings.run_root.mkdir(parents=True, exist_ok=True)
        self.graph = self._build().compile(checkpointer=checkpointer)
        self.phase2_graph = self._build_phase2().compile(checkpointer=checkpointer)

    def _record_call(
        self, phase: str, turn: int, step: str, request: GenerationRequest, result: Any
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

    def _plan_phase1(self, state: WorkflowState) -> WorkflowState:
        turn = state.get("phase1_iteration", 0) + 1
        v3 = self.settings.contract_version == "v3"
        if v3:
            prompt = phase1_retrieval_plan_prompt(
                self.settings.investor_name,
                self.settings.episode_slug,
                self.settings.pitch,
                (
                    [
                        row.model_dump(mode="json")
                        for row in self.precedent_corpus.list_episodes()
                    ]
                    if self.precedent_corpus is not None
                    else []
                ),
                state.get("investigation"),
                state.get("phase1_findings", []),
                navigation_results={
                    "precedent_searches": state.get("phase1_precedent_searches", []),
                    "precedent_reads": state.get("phase1_precedent_reads", []),
                    "retrieval_findings": state.get("phase1_retrieval_findings", []),
                },
                taxonomy_records=self.settings.taxonomy_records,
            )
        else:
            prompt = phase1_plan_prompt(
                self.settings.investor_name,
                self.settings.episode_slug,
                self.settings.pitch,
                state.get("investigation"),
                state.get("phase1_findings", []),
            )
        request = GenerationRequest(
            "phase1_plan",
            prompt,
            (Phase1RetrievalPlan if v3 else QueryPlan).model_json_schema(),
            max_output_tokens=self.settings.phase1_planning_max_output_tokens,
            reasoning_effort=self.settings.phase1_planning_reasoning_effort,
        )
        result = self.phase1_provider.generate(request)
        self._record_call("phase1", turn, "plan", request, result)
        normalized, plan_audit_findings = normalize_retrieval_plan_payload(
            result.parsed
        )
        try:
            plan = (Phase1RetrievalPlan if v3 else QueryPlan).model_validate(
                normalized
            ).model_dump(mode="json")
        except Exception as exc:
            plan = {
                "questions": ["What is the strongest positive signal?", "What could make this fail?"],
                "reconsideration_focus": f"Planner repair: {exc}",
                **(
                    {
                        "wiki_queries": ["founder team execution", "market scale constraints"],
                        "precedent_queries": [],
                        "transcript_reads": [],
                        "decision_reads": [],
                    }
                    if v3
                    else {"search_queries": ["founder team execution", "market scale constraints"]}
                ),
            }
            quality_findings = [
                *state.get("phase1_quality_findings", []),
                "PLANNER_FALLBACK_USED",
            ]
        else:
            quality_findings = state.get("phase1_quality_findings", [])
        return {
            "query_plan": plan,
            "phase1_audit_findings": list(dict.fromkeys([
                *state.get("phase1_audit_findings", []),
                *plan_audit_findings,
            ])),
            "phase1_quality_findings": list(dict.fromkeys(quality_findings)),
            "usage": _usage(state, result),
            "usage_by_phase": _usage_by_phase(state, result, "phase1"),
            "events": _event(
                state,
                "phase1_plan",
                turn=turn,
                queries=plan["wiki_queries"] if v3 else plan["search_queries"],
            ),
        }

    def _retrieve(self, state: WorkflowState) -> WorkflowState:
        if self.settings.contract_version == "v3":
            return self._retrieve_v3(state)
        if self.index is None:
            raise RuntimeError("Phase 1 retrieval requires a wiki index")
        plan = state["query_plan"]
        exact_by_id: dict[str, dict[str, Any]] = {
            row["chunk_id"]: row for row in state.get("exact_reads", [])
        }
        turn = state.get("phase1_iteration", 0) + 1
        mandatory_query = "portfolio holdings existing investments conflict overlap"
        queries = list(plan["search_queries"])
        if turn == 1:
            queries.insert(0, mandatory_query)
        normalized_queries = list(
            dict.fromkeys(" ".join(query.casefold().split()) for query in queries)
        )
        prior_queries = set(state.get("query_history", []))
        new_queries = [query for query in normalized_queries if query not in prior_queries]
        prior_ids = set(exact_by_id)
        searches: list[dict[str, Any]] = []
        for query in normalized_queries:
            hits = self.index.search(query, self.settings.retrieval_top_k)
            searches.append(
                {"query": query, "hits": [asdict(hit) for hit in hits]}
            )
            for hit in hits:
                if len(exact_by_id) >= self.settings.max_exact_reads:
                    break
                if hit.chunk_id not in exact_by_id:
                    exact_by_id[hit.chunk_id] = asdict(self.index.read(hit.chunk_id))
        new_chunk_ids = sorted(set(exact_by_id) - prior_ids)
        no_novelty = turn > 1 and not new_queries and not new_chunk_ids
        retrieval_pass = {
            "turn": turn,
            "normalized_queries": normalized_queries,
            "new_queries": new_queries,
            "new_chunk_ids": new_chunk_ids,
            "no_novelty": no_novelty,
        }
        write_json(
            self.settings.run_root / "phase1" / f"turn-{turn:02d}" / "retrieval.json",
            {"searches": searches, "exact_reads": list(exact_by_id.values())},
        )
        write_json(self.settings.run_root / "phase1" / f"turn-{turn:02d}" / "wiki-searches.json", searches)
        write_json(self.settings.run_root / "phase1" / f"turn-{turn:02d}" / "wiki-reads.json", list(exact_by_id.values()))
        return {
            "exact_reads": list(exact_by_id.values()),
            "query_history": [*state.get("query_history", []), *new_queries],
            "retrieval_passes": [*state.get("retrieval_passes", []), retrieval_pass],
            "no_retrieval_novelty": no_novelty,
            "phase1_current_retrieval_findings": [],
            "events": _event(
                state,
                "wiki_tools",
                turn=turn,
                search_calls=len(searches),
                exact_read_count=len(exact_by_id),
            ),
        }

    def _retrieve_v3(
        self, state: WorkflowState, phase: Literal["phase1", "phase2"] = "phase1"
    ) -> WorkflowState:
        if self.index is None:
            raise RuntimeError(f"{phase} retrieval requires a wiki index")
        prefix = phase
        plan = state["query_plan" if phase == "phase1" else "phase2_plan"]
        turn = state.get(f"{prefix}_iteration", 0) + 1
        turn_root = self.settings.run_root / phase / f"turn-{turn:02d}"
        wiki_reads_key = "exact_reads" if phase == "phase1" else "phase2_exact_reads"
        query_history_key = "query_history" if phase == "phase1" else "phase2_query_history"
        retrieval_passes_key = "retrieval_passes" if phase == "phase1" else "phase2_retrieval_passes"
        max_searches = getattr(self.settings, f"{prefix}_max_precedent_searches")
        max_reads = getattr(self.settings, f"{prefix}_max_precedent_reads")

        exact_by_id = {
            row["chunk_id"]: row for row in state.get(wiki_reads_key, [])
        }
        prior_wiki_ids = set(exact_by_id)
        mandatory_query = "portfolio holdings existing investments conflict overlap"
        wiki_queries = list(plan["wiki_queries"])
        if turn == 1 and phase == "phase1":
            wiki_queries.insert(0, mandatory_query)
        normalized_wiki = list(
            dict.fromkeys(" ".join(query.casefold().split()) for query in wiki_queries)
        )
        prior_queries = set(state.get(query_history_key, []))
        new_queries = [query for query in normalized_wiki if query not in prior_queries]
        wiki_searches: list[dict[str, Any]] = []
        for query in normalized_wiki:
            hits = self.index.search(query, self.settings.retrieval_top_k)
            wiki_searches.append({"query": query, "hits": [asdict(hit) for hit in hits]})
            for hit in hits:
                if len(exact_by_id) >= self.settings.max_exact_reads:
                    break
                if hit.chunk_id not in exact_by_id:
                    exact_by_id[hit.chunk_id] = asdict(self.index.read(hit.chunk_id))

        search_history = list(state.get(f"{prefix}_precedent_searches", []))
        read_history = list(state.get(f"{prefix}_precedent_reads", []))
        evidence_by_id = {
            row["evidence_id"]: row
            for row in state.get(f"{prefix}_historical_evidence", [])
        }
        findings = list(state.get(f"{prefix}_retrieval_findings", []))
        current_findings: list[str] = []

        def record_finding(code: str) -> None:
            findings.append(code)
            current_findings.append(code)

        prior_search_keys = {
            row.get("operation_key")
            for row in search_history
            if row.get("error_code") != "PRECEDENT_SEARCH_ERROR"
        }
        prior_read_keys = {
            row.get("operation_key")
            for row in read_history
            if row.get("error_code") != "PRECEDENT_READ_ERROR"
        }
        charged_searches = sum(bool(row.get("charged")) for row in search_history)
        charged_reads = sum(bool(row.get("charged")) for row in read_history)
        turn_searches: list[dict[str, Any]] = []
        turn_reads: list[dict[str, Any]] = []

        for position, raw_query in enumerate(plan["precedent_queries"], start=1):
            query = " ".join(raw_query.split())
            normalized = query.casefold()
            operation_key = f"search:{normalized}"
            if operation_key in prior_search_keys:
                continue
            query_id = f"{prefix}-t{turn:02d}-search-{position:02d}"
            if self.settings.episode_slug.casefold() in normalized:
                record = {
                    "operation_key": operation_key,
                    "query_id": query_id,
                    "query": query,
                    "status": "rejected",
                    "error_code": "TARGET_PRECEDENT_ACCESS_REJECTED",
                    "target_episode_slug": self.settings.episode_slug,
                    "charged": False,
                }
                record_finding("TARGET_PRECEDENT_ACCESS_REJECTED")
            elif self.precedent_corpus is None:
                record = {
                    "operation_key": operation_key,
                    "query_id": query_id,
                    "query": query,
                    "status": "error",
                    "error_code": "PRECEDENT_CORPUS_DISABLED",
                    "charged": False,
                }
                record_finding("PRECEDENT_CORPUS_DISABLED")
            elif charged_searches >= max_searches:
                record = {
                    "operation_key": operation_key,
                    "query_id": query_id,
                    "query": query,
                    "status": "error",
                    "error_code": "PRECEDENT_SEARCH_BUDGET_EXHAUSTED",
                    "charged": False,
                }
                record_finding("PRECEDENT_SEARCH_BUDGET_EXHAUSTED")
            else:
                charged_searches += 1
                try:
                    assert self.precedent_corpus is not None
                    result = self.precedent_corpus.search(
                        query,
                        top_k=self.settings.retrieval_top_k,
                        query_id=query_id,
                        selection_policy=self.settings.precedent_selection_policy,
                        candidate_pool_k=self.settings.precedent_candidate_pool_k,
                        in_slots=self.settings.precedent_in_slots,
                        out_slots=self.settings.precedent_out_slots,
                    )
                    record = {
                        **result.model_dump(mode="json"),
                        "selection_policy": self.settings.precedent_selection_policy,
                        "operation_key": operation_key,
                        "status": "ok",
                        "charged": True,
                    }
                    for finding in result.quality_findings:
                        record_finding(finding)
                except Exception as exc:
                    record = {
                        "operation_key": operation_key,
                        "query_id": query_id,
                        "query": query,
                        "status": "error",
                        "error_code": "PRECEDENT_SEARCH_ERROR",
                        "error": str(exc),
                        "charged": True,
                    }
                    record_finding("PRECEDENT_SEARCH_ERROR")
            prior_search_keys.add(operation_key)
            search_history.append(record)
            turn_searches.append(record)

        read_requests: list[tuple[str, str, int | None, int | None, int]] = []
        for position, request in enumerate(plan["transcript_reads"], start=1):
            read_requests.append(
                (
                    "transcript",
                    request["episode_slug"],
                    request["turn_start"],
                    request["turn_end"],
                    position,
                )
            )
        for position, slug in enumerate(plan["decision_reads"], start=1):
            read_requests.append(("decision", slug, None, None, position))

        permitted_in_slugs: set[str] | None = None
        if (
            self.settings.precedent_selection_policy == "contrastive"
            and self.precedent_corpus is not None
        ):
            decision_by_slug = {
                row.episode_slug: row.decision_status
                for row in self.precedent_corpus.list_episodes()
            }
            opened_slugs = {
                str(row.get("episode_slug"))
                for row in read_history
                if row.get("status") == "ok"
            }
            existing_in = {
                slug
                for slug in opened_slugs
                if decision_by_slug.get(slug) == "In"
            }
            existing_out = {
                slug
                for slug in opened_slugs
                if decision_by_slug.get(slug) == "Out"
            }
            requested_in = list(
                dict.fromkeys(
                    slug
                    for _, slug, _, _, _ in read_requests
                    if decision_by_slug.get(slug) == "In"
                )
            )
            requested_out = {
                slug
                for _, slug, _, _, _ in read_requests
                if decision_by_slug.get(slug) == "Out"
            }
            if self.settings.precedent_out_slots == 0:
                allowed_in_total = len(existing_in | set(requested_in))
            else:
                available_out_total = len(existing_out | requested_out)
                allowed_in_total = max(
                    min(1, self.settings.precedent_in_slots),
                    (
                        available_out_total
                        * self.settings.precedent_in_slots
                        // self.settings.precedent_out_slots
                    ),
                )
            permitted_in_slugs = set(existing_in)
            for slug in requested_in:
                if len(permitted_in_slugs) >= allowed_in_total:
                    break
                permitted_in_slugs.add(slug)

        for read_type, slug, start, end, position in read_requests:
            operation_key = f"{read_type}:{slug}:{start}:{end}"
            if operation_key in prior_read_keys:
                continue
            query_id = f"{prefix}-t{turn:02d}-{read_type}-{position:02d}"
            base = {
                "operation_key": operation_key,
                "query_id": query_id,
                "read_type": read_type,
                "episode_slug": slug,
            }
            if slug == self.settings.episode_slug:
                record = {
                    **base,
                    "status": "rejected",
                    "error_code": "TARGET_PRECEDENT_ACCESS_REJECTED",
                    "target_episode_slug": slug,
                    "charged": False,
                }
                record_finding("TARGET_PRECEDENT_ACCESS_REJECTED")
            elif self.precedent_corpus is None:
                record = {
                    **base,
                    "turn_start": start,
                    "turn_end": end,
                    "status": "error",
                    "error_code": "PRECEDENT_CORPUS_DISABLED",
                    "charged": False,
                }
                record_finding("PRECEDENT_CORPUS_DISABLED")
            elif (
                permitted_in_slugs is not None
                and slug not in permitted_in_slugs
                and decision_by_slug.get(slug) == "In"
            ):
                record = {
                    **base,
                    "turn_start": start,
                    "turn_end": end,
                    "status": "rejected",
                    "error_code": "PRECEDENT_CONTRASTIVE_READ_RATIO",
                    "charged": False,
                }
            elif (
                read_type == "transcript"
                and start is None
                and not self.settings.allow_full_transcript
            ):
                record = {
                    **base,
                    "turn_start": start,
                    "turn_end": end,
                    "status": "rejected",
                    "error_code": "FULL_TRANSCRIPT_READ_DISABLED",
                    "charged": False,
                }
                record_finding("FULL_TRANSCRIPT_READ_DISABLED")
            elif charged_reads >= max_reads:
                record = {
                    **base,
                    "turn_start": start,
                    "turn_end": end,
                    "status": "error",
                    "error_code": "PRECEDENT_READ_BUDGET_EXHAUSTED",
                    "charged": False,
                }
                record_finding("PRECEDENT_READ_BUDGET_EXHAUSTED")
            else:
                charged_reads += 1
                try:
                    assert self.precedent_corpus is not None
                    if read_type == "transcript":
                        opened = self.precedent_corpus.open_transcript(
                            slug,
                            start=0 if start is None else start,
                            end=end,
                            query_id=query_id,
                        )
                        record = {
                            **base,
                            **opened.model_dump(mode="json"),
                            "status": "ok",
                            "charged": True,
                        }
                        opened_evidence = (() if opened.evidence is None else (opened.evidence,))
                    else:
                        opened = self.precedent_corpus.open_decision(
                            slug, query_id=query_id
                        )
                        record = {
                            **base,
                            **opened.model_dump(mode="json"),
                            "status": "ok",
                            "charged": True,
                        }
                        opened_evidence = opened.evidence
                    for evidence in opened_evidence:
                        evidence_by_id.setdefault(
                            evidence.evidence_id,
                            evidence.model_dump(mode="json"),
                        )
                except PermissionError as exc:
                    record = {
                        **base,
                        "status": "rejected",
                        "error_code": "TARGET_PRECEDENT_ACCESS_REJECTED",
                        "error": str(exc),
                        "charged": True,
                    }
                    record_finding("TARGET_PRECEDENT_ACCESS_REJECTED")
                except KeyError as exc:
                    record = {
                        **base,
                        "status": "error",
                        "error_code": "UNKNOWN_PRECEDENT_EPISODE",
                        "error": str(exc),
                        "charged": True,
                    }
                    record_finding("UNKNOWN_PRECEDENT_EPISODE")
                except IndexError as exc:
                    record = {
                        **base,
                        "status": "error",
                        "error_code": "INVALID_PRECEDENT_TURN_RANGE",
                        "error": str(exc),
                        "charged": True,
                    }
                    record_finding("INVALID_PRECEDENT_TURN_RANGE")
                except Exception as exc:
                    record = {
                        **base,
                        "status": "error",
                        "error_code": "PRECEDENT_READ_ERROR",
                        "error": str(exc),
                        "charged": True,
                    }
                    record_finding("PRECEDENT_READ_ERROR")
            prior_read_keys.add(operation_key)
            read_history.append(record)
            turn_reads.append(record)

        new_wiki_ids = sorted(set(exact_by_id) - prior_wiki_ids)
        successful_precedent_operation = any(
            row.get("status") == "ok" for row in [*turn_searches, *turn_reads]
        )
        no_novelty = (
            turn > 1
            and not new_queries
            and not new_wiki_ids
            and not successful_precedent_operation
        )
        retrieval_pass = {
            "turn": turn,
            "normalized_queries": normalized_wiki,
            "new_queries": new_queries,
            "new_chunk_ids": new_wiki_ids,
            "no_novelty": no_novelty,
        }
        write_json(turn_root / "retrieval.json", {"searches": wiki_searches, "exact_reads": list(exact_by_id.values())})
        write_json(turn_root / "wiki-searches.json", wiki_searches)
        write_json(turn_root / "wiki-reads.json", list(exact_by_id.values()))
        write_json(turn_root / "precedent-searches.json", turn_searches)
        write_json(turn_root / "precedent-reads.json", turn_reads)
        write_json(turn_root / "historical-evidence.json", list(evidence_by_id.values()))
        return {
            wiki_reads_key: list(exact_by_id.values()),
            query_history_key: [*state.get(query_history_key, []), *new_queries],
            retrieval_passes_key: [*state.get(retrieval_passes_key, []), retrieval_pass],
            ("no_retrieval_novelty" if phase == "phase1" else "phase2_no_retrieval_novelty"): no_novelty,
            f"{prefix}_precedent_searches": search_history,
            f"{prefix}_precedent_reads": read_history,
            f"{prefix}_historical_evidence": list(evidence_by_id.values()),
            f"{prefix}_retrieval_findings": list(dict.fromkeys(findings)),
            f"{prefix}_current_retrieval_findings": list(
                dict.fromkeys(current_findings)
            ),
            f"{prefix}_quality_findings": list(
                state.get(f"{prefix}_quality_findings", [])
            ),
            "events": _event(
                state,
                f"{prefix}_retrieval",
                turn=turn,
                wiki_search_calls=len(wiki_searches),
                precedent_search_calls=sum(bool(row.get("charged")) for row in turn_searches),
                precedent_read_calls=sum(bool(row.get("charged")) for row in turn_reads),
            ),
        }

    def _investigate(self, state: WorkflowState) -> WorkflowState:
        turn = state.get("phase1_iteration", 0) + 1
        prompt = phase1_investigation_prompt(
            self.settings.investor_name,
            self.settings.episode_slug,
            self.settings.pitch,
            sorted(self.settings.taxonomy_labels),
            state.get("exact_reads", []),
            state.get("investigation"),
            state.get("phase1_findings", []),
            contract_version=self.settings.contract_version,
            exact_historical_evidence=state.get("phase1_historical_evidence", []),
            accessible_episode_inventory=(
                [
                    row.model_dump(mode="json")
                    for row in self.precedent_corpus.list_episodes()
                ]
                if self.precedent_corpus is not None
                else []
            ),
            navigation_results={
                "precedent_searches": state.get("phase1_precedent_searches", []),
                "precedent_reads": state.get("phase1_precedent_reads", []),
                "retrieval_findings": state.get("phase1_retrieval_findings", []),
            },
            taxonomy_records=self.settings.taxonomy_records,
        )
        exact_wiki_ids = {row["chunk_id"] for row in state.get("exact_reads", [])}
        exact_historical_ids = {
            row["evidence_id"]
            for row in state.get("phase1_historical_evidence", [])
        }
        v3 = self.settings.contract_version == "v3"
        v2 = self.settings.contract_version == "v2"
        schema = (
            investigation_v3_json_schema(
                episode_slug=self.settings.episode_slug,
                taxonomy_labels=self.settings.taxonomy_labels,
                exact_wiki_ids=exact_wiki_ids,
                exact_historical_ids=exact_historical_ids,
            )
            if v3
            else investigation_v2_json_schema(
                episode_slug=self.settings.episode_slug,
                taxonomy_labels=self.settings.taxonomy_labels,
                exact_wiki_ids=exact_wiki_ids,
            )
            if v2
            else investigation_json_schema(
                episode_slug=self.settings.episode_slug,
                taxonomy_labels=self.settings.taxonomy_labels,
                exact_wiki_ids=exact_wiki_ids,
            )
        )
        request = GenerationRequest(
            "phase1",
            prompt,
            schema,
            max_output_tokens=self.settings.phase1_max_output_tokens,
            reasoning_effort=self.settings.phase1_reasoning_effort,
        )
        result = self.phase1_provider.generate(request)
        self._record_call("phase1", turn, "investigation", request, result)
        findings: list[str] = []
        audit_findings = list(state.get("phase1_audit_findings", []))
        persistent_quality_findings = list(
            state.get("phase1_quality_findings", [])
        )
        quality_findings = list(persistent_quality_findings)
        quality_findings.extend(state.get("phase1_current_retrieval_findings", []))
        if state.get("no_retrieval_novelty"):
            quality_findings.append("NO_RETRIEVAL_NOVELTY")
        quality_findings = list(dict.fromkeys(quality_findings))
        candidate: Investigation | InvestigationV2 | InvestigationV3 | None = None
        try:
            if v3:
                normalized, normalization_findings = (
                    normalize_investigation_v3_payload(result.parsed)
                )
                persistent_quality_findings.extend(normalization_findings)
                persistent_quality_findings = list(
                    dict.fromkeys(persistent_quality_findings)
                )
                quality_findings.extend(normalization_findings)
                quality_findings = list(dict.fromkeys(quality_findings))
                candidate = InvestigationV3.model_validate(normalized)
                validate_investigation_v3(
                    candidate,
                    episode_slug=self.settings.episode_slug,
                    pitch=self.settings.pitch,
                    taxonomy_labels=self.settings.taxonomy_labels,
                    exact_wiki_ids=exact_wiki_ids,
                    exact_historical_ids=exact_historical_ids,
                )
            elif v2:
                normalized, normalization_findings = (
                    normalize_investigation_v2_payload(result.parsed)
                )
                persistent_quality_findings.extend(normalization_findings)
                persistent_quality_findings = list(
                    dict.fromkeys(persistent_quality_findings)
                )
                quality_findings.extend(normalization_findings)
                quality_findings = list(dict.fromkeys(quality_findings))
                candidate = InvestigationV2.model_validate(normalized)
                validate_investigation_v2(
                    candidate,
                    episode_slug=self.settings.episode_slug,
                    pitch=self.settings.pitch,
                    taxonomy_labels=self.settings.taxonomy_labels,
                    exact_wiki_ids=exact_wiki_ids,
                )
            else:
                candidate = Investigation.model_validate(result.parsed)
                validate_investigation(
                    candidate,
                    episode_slug=self.settings.episode_slug,
                    pitch=self.settings.pitch,
                    taxonomy_labels=self.settings.taxonomy_labels,
                    exact_wiki_ids=exact_wiki_ids,
                )
        except Exception as exc:
            findings.append(str(exc))
        candidate_valid = candidate is not None and not findings
        if not candidate_valid:
            if turn >= self.settings.phase1_max_iterations and state.get("investigation"):
                action = "freeze_provisional"
                findings = list(dict.fromkeys([
                    *state.get("phase1_findings", []),
                    *findings,
                    "INVALID_RECONSIDERATION_PRESERVED_PRIOR_CANDIDATE",
                ]))
            else:
                action = "failed" if turn >= self.settings.phase1_max_iterations else "revisit"
        elif turn < self.settings.phase1_min_iterations:
            action = "revisit"
        elif v3 and quality_findings:
            action = (
                "freeze_provisional"
                if turn >= self.settings.phase1_max_iterations
                else "revisit"
            )
        elif state.get("no_retrieval_novelty"):
            action = "freeze_provisional"
        elif candidate.saturated:
            action = "freeze_provisional" if quality_findings else "freeze_accepted"
        elif turn >= self.settings.phase1_max_iterations:
            action = "freeze_provisional"
            findings.append("maximum iterations reached without saturation")
        else:
            action = "revisit"
            findings.append("investigation is not saturated")
        update: WorkflowState = {
            "phase1_iteration": turn,
            "phase1_action": action,
            "phase1_findings": [*audit_findings, *quality_findings, *findings],
            "phase1_quality_findings": persistent_quality_findings,
            "usage": _usage(state, result),
            "usage_by_phase": _usage_by_phase(state, result, "phase1"),
            "events": _event(state, "phase1_validation", turn=turn, action=action, findings=findings),
        }
        if candidate_valid:
            update["investigation"] = candidate.model_dump(mode="json")
        return update

    def _freeze_phase1(self, state: WorkflowState) -> WorkflowState:
        model = (
            InvestigationV3
            if self.settings.contract_version == "v3"
            else InvestigationV2
            if self.settings.contract_version == "v2"
            else Investigation
        )
        candidate = model.model_validate(state["investigation"])
        _, digest = freeze_model(self.settings.run_root / "phase1", "investigation", candidate)
        status = "accepted" if state["phase1_action"] == "freeze_accepted" else "provisional"
        return {
            "investigation_sha256": digest,
            "phase1_status": status,
            "events": _event(state, "phase1_frozen", status=status, sha256=digest),
        }

    def _plan_phase2(self, state: WorkflowState) -> WorkflowState:
        turn = state.get("phase2_iteration", 0) + 1
        inventory = (
            [row.model_dump(mode="json") for row in self.precedent_corpus.list_episodes()]
            if self.precedent_corpus is not None
            else []
        )
        request = GenerationRequest(
            "phase2_plan",
            phase2_plan_prompt(
                self.settings.investor_name,
                self.settings.episode_slug,
                state["investigation"],
                state["investigation_sha256"],
                inventory,
                state.get("decision"),
                state.get("phase2_findings", []),
                {
                    "wiki_searches": state.get("phase2_retrieval_passes", []),
                    "precedent_searches": state.get("phase2_precedent_searches", []),
                    "precedent_reads": state.get("phase2_precedent_reads", []),
                    "retrieval_findings": state.get("phase2_retrieval_findings", []),
                },
            ),
            Phase2RetrievalPlan.model_json_schema(),
            max_output_tokens=self.settings.phase2_planning_max_output_tokens,
            reasoning_effort=self.settings.phase2_planning_reasoning_effort,
        )
        result = self.phase2_provider.generate(request)
        self._record_call("phase2", turn, "plan", request, result)
        findings: list[str] = []
        normalized, plan_audit_findings = normalize_retrieval_plan_payload(
            result.parsed
        )
        try:
            plan = Phase2RetrievalPlan.model_validate(normalized).model_dump(mode="json")
        except Exception as exc:
            findings.append(f"PHASE2_PLAN_SCHEMA_ERROR: {exc}")
            plan = {
                "decision_questions": [
                    "What supports an any-check decision?",
                    "What is the strongest opposing case?",
                ],
                "wiki_queries": [],
                "precedent_queries": [],
                "transcript_reads": [],
                "decision_reads": [],
                "opposing_case_to_test": "Repair the invalid retrieval plan.",
            }
        return {
            "phase2_plan": plan,
            "phase2_audit_findings": list(dict.fromkeys([
                *state.get("phase2_audit_findings", []),
                *plan_audit_findings,
            ])),
            "phase2_current_retrieval_findings": findings,
            "usage": _usage(state, result),
            "usage_by_phase": _usage_by_phase(state, result, "phase2"),
            "events": _event(state, "phase2_plan", turn=turn),
        }

    def _retrieve_phase2(self, state: WorkflowState) -> WorkflowState:
        update = self._retrieve_v3(state, "phase2")
        planner_findings = state.get("phase2_current_retrieval_findings", [])
        update["phase2_current_retrieval_findings"] = list(dict.fromkeys([
            *planner_findings,
            *update.get("phase2_current_retrieval_findings", []),
        ]))
        return update

    def _decide(self, state: WorkflowState) -> WorkflowState:
        turn = state.get("phase2_iteration", 0) + 1
        prompt = phase2_prompt(
            self.settings.investor_name,
            self.settings.episode_slug,
            state["investigation"],
            state["investigation_sha256"],
            sorted(self.settings.check_tiers),
            state.get("decision"),
            state.get("phase2_findings", []),
            contract_version=self.settings.contract_version,
            exact_wiki_evidence=[
                *state.get("exact_reads", []), *state.get("phase2_exact_reads", [])
            ],
            exact_historical_evidence=[
                *state.get("phase1_historical_evidence", []),
                *state.get("phase2_historical_evidence", []),
            ],
            accessible_episode_inventory=(
                [row.model_dump(mode="json") for row in self.precedent_corpus.list_episodes()]
                if self.precedent_corpus is not None else []
            ),
            navigation_results={
                "precedent_searches": state.get("phase2_precedent_searches", []),
                "precedent_reads": state.get("phase2_precedent_reads", []),
                "retrieval_findings": state.get("phase2_retrieval_findings", []),
            },
        )
        rationale_ids = {
            row["rationale_id"] for row in state["investigation"]["rationales"]
        }
        v3 = self.settings.contract_version == "v3"
        v2 = self.settings.contract_version == "v2"
        all_historical = [
            *state.get("phase1_historical_evidence", []),
            *state.get("phase2_historical_evidence", []),
        ]
        all_precedent_reads = [
            *state.get("phase1_precedent_reads", []),
            *state.get("phase2_precedent_reads", []),
        ]
        schema_builder = decision_v2_json_schema if v2 else decision_json_schema
        schema = (
            decision_v3_json_schema(
                episode_slug=self.settings.episode_slug,
                investigation_sha256=state["investigation_sha256"],
                rationale_ids=rationale_ids,
                check_tiers=self.settings.check_tiers,
                exact_historical_ids={row["evidence_id"] for row in all_historical},
                available_precedent_slugs={
                    row.episode_slug for row in self.precedent_corpus.list_episodes()
                } if self.precedent_corpus is not None else set(),
            )
            if v3
            else schema_builder(
                episode_slug=self.settings.episode_slug,
                investigation_sha256=state["investigation_sha256"],
                rationale_ids=rationale_ids,
                check_tiers=self.settings.check_tiers,
            )
        )
        request = GenerationRequest(
            "phase2",
            prompt,
            schema,
            max_output_tokens=self.settings.phase2_max_output_tokens,
            reasoning_effort=self.settings.phase2_reasoning_effort,
        )
        result = self.phase2_provider.generate(request)
        self._record_call("phase2", turn, "decision", request, result)
        structural_findings: list[str] = []
        audit_findings = list(state.get("phase2_audit_findings", []))
        quality_findings: list[str] = []
        candidate: Decision | DecisionV2 | DecisionV3 | None = None
        try:
            if v3:
                candidate = DecisionV3.model_validate(result.parsed)
                validate_decision_v3(
                    candidate,
                    InvestigationV3.model_validate(state["investigation"]),
                    state["investigation_sha256"],
                    self.settings.check_tiers,
                    historical_evidence=all_historical,
                    precedent_reads=all_precedent_reads,
                    target_episode_slug=self.settings.episode_slug,
                )
                quality_findings = decision_quality_findings(candidate)
            elif v2:
                candidate = DecisionV2.model_validate(result.parsed)
                investigation = (
                    InvestigationV3.model_validate(state["investigation"])
                    if self.settings.contract_version == "v3"
                    else InvestigationV2.model_validate(state["investigation"])
                )
                validate_decision(
                    candidate,
                    self.settings.episode_slug,
                    state["investigation_sha256"],
                    rationale_ids,
                    self.settings.check_tiers,
                )
                quality_findings = decision_quality_findings(candidate)
                try:
                    validate_decision_v2(
                        candidate,
                        investigation,
                        state["investigation_sha256"],
                        self.settings.check_tiers,
                    )
                except ValueError as exc:
                    quality_findings.append(str(exc))
            else:
                candidate = Decision.model_validate(result.parsed)
                validate_decision(
                    candidate,
                    self.settings.episode_slug,
                    state["investigation_sha256"],
                    rationale_ids,
                    self.settings.check_tiers,
                )
                quality_findings = decision_quality_findings(candidate)
        except Exception as exc:
            finding = str(exc)
            missing_consistency = (
                v3
                and candidate is not None
                and finding.endswith(
                    "deliberation must finish with a consistency step"
                )
            )
            if missing_consistency:
                try:
                    validate_decision_v3(
                        candidate,
                        InvestigationV3.model_validate(state["investigation"]),
                        state["investigation_sha256"],
                        self.settings.check_tiers,
                        historical_evidence=all_historical,
                        precedent_reads=all_precedent_reads,
                        target_episode_slug=self.settings.episode_slug,
                        allow_missing_final_consistency=True,
                    )
                    quality_findings = decision_quality_findings(candidate)
                    audit_findings.extend([
                        finding,
                        "MINOR_AUDIT_OMISSION_ACCEPTED_WITH_WARNING",
                    ])
                    audit_findings = list(dict.fromkeys(audit_findings))
                except Exception as relaxed_exc:
                    structural_findings.append(str(relaxed_exc))
            else:
                structural_findings.append(finding)
        current_blockers = list(state.get("phase2_current_retrieval_findings", []))
        quality_findings = list(dict.fromkeys([*current_blockers, *quality_findings]))
        findings = structural_findings or [*quality_findings, *audit_findings]
        structurally_valid = candidate is not None and not structural_findings
        if not structurally_valid:
            if turn >= self.settings.phase2_max_iterations and state.get("decision"):
                action = "freeze_provisional"
                findings = list(dict.fromkeys([
                    *state.get("phase2_findings", []),
                    *findings,
                    "INVALID_RECONSIDERATION_PRESERVED_PRIOR_CANDIDATE",
                ]))
            elif turn >= self.settings.phase2_max_iterations and candidate is not None:
                action = "freeze_provisional"
                findings = list(dict.fromkeys([
                    *findings,
                    "SEMANTICALLY_INVALID_FINAL_DRAFT_PRESERVED",
                ]))
            else:
                action = "failed" if turn >= self.settings.phase2_max_iterations else "revisit"
        elif quality_findings or (v3 and not candidate.stable):
            action = (
                "freeze_provisional"
                if turn >= self.settings.phase2_max_iterations
                else "revisit"
            )
        elif turn < self.settings.phase2_min_iterations:
            action = "revisit"
        elif audit_findings:
            action = "freeze_accepted_with_warnings"
        else:
            action = "freeze_accepted"
        update: WorkflowState = {
            "phase2_iteration": turn,
            "phase2_action": action,
            "phase2_findings": findings,
            "usage": _usage(state, result),
            "usage_by_phase": _usage_by_phase(state, result, "phase2"),
            "events": _event(state, "phase2_validation", turn=turn, action=action, findings=findings),
        }
        if candidate is not None and (
            structurally_valid
            or (
                action == "freeze_provisional"
                and "SEMANTICALLY_INVALID_FINAL_DRAFT_PRESERVED" in findings
            )
        ):
            update["decision"] = candidate.model_dump(mode="json")
        return update

    def _freeze_phase2(self, state: WorkflowState) -> WorkflowState:
        model = DecisionV3 if self.settings.contract_version == "v3" else DecisionV2 if self.settings.contract_version == "v2" else Decision
        candidate = model.model_validate(state["decision"])
        _, digest = freeze_model(self.settings.run_root / "phase2", "decision", candidate)
        status = {
            "freeze_provisional": "provisional",
            "freeze_accepted_with_warnings": "accepted_with_warnings",
        }.get(state["phase2_action"], "accepted")
        final = {
            "phase2_status": status,
            "events": _event(state, "phase2_frozen", status=status, sha256=digest),
        }
        write_json(
            self.settings.run_root / "summary.json",
            {
                "episode_slug": self.settings.episode_slug,
                "phase1_status": state["phase1_status"],
                "phase1_iterations": state.get("phase1_iteration"),
                "phase1_findings": state.get("phase1_findings", []),
                "phase2_status": final["phase2_status"],
                "phase2_iterations": state.get("phase2_iteration"),
                "phase2_findings": state.get("phase2_findings", []),
                "usage": state.get("usage", {}),
                "usage_by_phase": state.get("usage_by_phase", {}),
                "decision": candidate.model_dump(mode="json"),
            },
        )
        return final

    @staticmethod
    def _route_phase1(state: WorkflowState) -> str:
        return state["phase1_action"]

    @staticmethod
    def _route_phase2(state: WorkflowState) -> str:
        return state["phase2_action"]

    def _build(self) -> StateGraph:
        builder = StateGraph(WorkflowState)
        builder.add_node("phase1_plan", self._plan_phase1)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("investigate", self._investigate)
        builder.add_node("freeze_phase1", self._freeze_phase1)
        builder.add_node("decide", self._decide)
        builder.add_node("freeze_phase2", self._freeze_phase2)
        builder.add_edge(START, "phase1_plan")
        builder.add_edge("phase1_plan", "retrieve")
        builder.add_edge("retrieve", "investigate")
        builder.add_conditional_edges(
            "investigate",
            self._route_phase1,
            {
                "revisit": "phase1_plan",
                "freeze_accepted": "freeze_phase1",
                "freeze_provisional": "freeze_phase1",
                "failed": END,
            },
        )
        if self.settings.contract_version == "v3":
            builder.add_node("phase2_plan", self._plan_phase2)
            builder.add_node("phase2_retrieve", self._retrieve_phase2)
            builder.add_edge("freeze_phase1", "phase2_plan")
            builder.add_edge("phase2_plan", "phase2_retrieve")
            builder.add_edge("phase2_retrieve", "decide")
        else:
            builder.add_edge("freeze_phase1", "decide")
        builder.add_conditional_edges(
            "decide",
            self._route_phase2,
            {
                "revisit": "phase2_plan" if self.settings.contract_version == "v3" else "decide",
                "freeze_accepted": "freeze_phase2",
                "freeze_accepted_with_warnings": "freeze_phase2",
                "freeze_provisional": "freeze_phase2",
                "failed": END,
            },
        )
        builder.add_edge("freeze_phase2", END)
        return builder

    def _build_phase2(self) -> StateGraph:
        builder = StateGraph(WorkflowState)
        if self.settings.contract_version == "v3":
            builder.add_node("phase2_plan", self._plan_phase2)
            builder.add_node("phase2_retrieve", self._retrieve_phase2)
        builder.add_node("decide", self._decide)
        builder.add_node("freeze_phase2", self._freeze_phase2)
        if self.settings.contract_version == "v3":
            builder.add_edge(START, "phase2_plan")
            builder.add_edge("phase2_plan", "phase2_retrieve")
            builder.add_edge("phase2_retrieve", "decide")
        else:
            builder.add_edge(START, "decide")
        builder.add_conditional_edges(
            "decide",
            self._route_phase2,
            {
                "revisit": "phase2_plan" if self.settings.contract_version == "v3" else "decide",
                "freeze_accepted": "freeze_phase2",
                "freeze_accepted_with_warnings": "freeze_phase2",
                "freeze_provisional": "freeze_phase2",
                "failed": END,
            },
        )
        builder.add_edge("freeze_phase2", END)
        return builder

    def _persist_result(self, result: WorkflowState) -> WorkflowState:
        if result.get("phase1_action") == "failed":
            result["phase1_status"] = "failed"
        if result.get("phase2_action") == "failed":
            result["phase2_status"] = "failed"
        write_json(
            self.settings.run_root / "summary.json",
            {
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
            },
        )
        write_json(self.settings.run_root / "state.json", result)
        return result

    def invoke(self, thread_id: str) -> WorkflowState:
        initial: WorkflowState = {
            "phase1_iteration": 0,
            "phase2_iteration": 0,
            "phase1_status": "running",
            "phase2_status": "pending",
            "exact_reads": [],
            "query_history": [],
            "retrieval_passes": [],
            "precedent_manifest_sha256": self.settings.precedent_manifest_sha256 or "",
            "accessible_precedent_count": self.settings.accessible_precedent_count,
            "phase1_precedent_searches": [],
            "phase1_precedent_reads": [],
            "phase1_historical_evidence": [],
            "phase1_retrieval_findings": [],
            "phase1_current_retrieval_findings": [],
            "phase1_audit_findings": [],
            "phase1_quality_findings": list(
                self.settings.initial_phase1_quality_findings
            ),
            "phase1_findings": list(
                self.settings.initial_phase1_quality_findings
            ),
            "phase2_findings": [],
            "phase2_audit_findings": [],
            "phase2_plan": {},
            "phase2_exact_reads": [],
            "phase2_query_history": [],
            "phase2_retrieval_passes": [],
            "phase2_precedent_searches": [],
            "phase2_precedent_reads": [],
            "phase2_historical_evidence": [],
            "phase2_retrieval_findings": [],
            "phase2_current_retrieval_findings": [],
            "phase2_quality_findings": [],
            "events": [],
            "usage": {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            },
            "usage_by_phase": {
                "phase1": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
                "phase2": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            },
        }
        result = self.graph.invoke(initial, {"configurable": {"thread_id": thread_id}})
        return self._persist_result(result)

    def resume(self, thread_id: str) -> WorkflowState:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        current = snapshot.values
        fresh = list(self.settings.initial_phase1_quality_findings)
        if current and fresh:
            quality_findings = list(dict.fromkeys([
                *current.get("phase1_quality_findings", []), *fresh
            ]))
            phase1_findings = list(dict.fromkeys([
                *current.get("phase1_findings", []), *fresh
            ]))
            phase1_status = current.get("phase1_status")
            if phase1_status == "accepted":
                phase1_status = "provisional"
            self.graph.update_state(
                config,
                {
                    "phase1_quality_findings": quality_findings,
                    "phase1_findings": phase1_findings,
                    "phase1_status": phase1_status,
                },
            )
        result = self.graph.invoke(None, config)
        return self._persist_result(result)

    def invoke_phase2(
        self,
        thread_id: str,
        investigation: Investigation | InvestigationV2 | InvestigationV3,
        investigation_sha256: str,
        *,
        phase1_status: str,
        phase1_exact_reads: list[dict[str, Any]] | None = None,
        phase1_historical_evidence: list[dict[str, Any]] | None = None,
        phase1_precedent_reads: list[dict[str, Any]] | None = None,
        phase1_state: dict[str, Any] | None = None,
    ) -> WorkflowState:
        zero_usage: dict[str, int | float] = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }
        carried_keys = (
            "phase1_iteration",
            "phase1_action",
            "query_plan",
            "query_history",
            "retrieval_passes",
            "exact_reads",
            "phase1_precedent_searches",
            "phase1_precedent_reads",
            "phase1_historical_evidence",
            "phase1_findings",
            "phase1_audit_findings",
            "phase1_quality_findings",
            "phase1_retrieval_findings",
            "phase1_current_retrieval_findings",
            "no_retrieval_novelty",
            "precedent_manifest_sha256",
            "accessible_precedent_count",
        )
        carried = {
            key: phase1_state[key]
            for key in carried_keys
            if phase1_state is not None and key in phase1_state
        }
        phase1_usage = dict(
            (phase1_state or {}).get("usage_by_phase", {}).get(
                "phase1", zero_usage
            )
        )
        initial: WorkflowState = {
            **carried,
            "phase1_iteration": carried.get("phase1_iteration", 0),
            "phase2_iteration": 0,
            "phase1_status": phase1_status,
            "phase2_status": "pending",
            "investigation": investigation.model_dump(mode="json"),
            "investigation_sha256": investigation_sha256,
            "decision": {},
            "phase2_findings": [],
            "phase2_audit_findings": [],
            "phase2_plan": {},
            "phase2_exact_reads": [],
            "phase2_query_history": [],
            "phase2_retrieval_passes": [],
            "phase2_precedent_searches": [],
            "phase2_precedent_reads": [],
            "phase2_historical_evidence": [],
            "phase2_retrieval_findings": [],
            "phase2_current_retrieval_findings": [],
            "phase1_historical_evidence": carried.get(
                "phase1_historical_evidence", phase1_historical_evidence or []
            ),
            "phase1_precedent_reads": carried.get(
                "phase1_precedent_reads", phase1_precedent_reads or []
            ),
            "exact_reads": carried.get("exact_reads", phase1_exact_reads or []),
            "precedent_manifest_sha256": carried.get(
                "precedent_manifest_sha256",
                self.settings.precedent_manifest_sha256 or "",
            ),
            "accessible_precedent_count": carried.get(
                "accessible_precedent_count",
                self.settings.accessible_precedent_count,
            ),
            "events": [{
                "sequence": 1,
                "kind": "phase2_replay_started",
                "investigation_sha256": investigation_sha256,
            }],
            "usage": dict(phase1_usage),
            "usage_by_phase": {
                "phase1": dict(phase1_usage),
                "phase2": dict(zero_usage),
            },
        }
        result = self.phase2_graph.invoke(
            initial, {"configurable": {"thread_id": thread_id}}
        )
        return self._persist_result(result)
