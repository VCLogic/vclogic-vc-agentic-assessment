from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from langgraph.checkpoint.memory import InMemorySaver
import pytest

from vc_clone_graph.graph import WorkflowSettings
from vc_clone_graph.artifacts import verify_phase1_artifacts, verify_run_artifacts
from vc_clone_graph.providers.base import GenerationRequest, GenerationResult, Usage
from vc_clone_graph.retrieval import HybridWikiIndex
from vc_clone_graph.retrieval_v4 import V4RetrievalResult
from vc_clone_graph.portfolio_memory import (
    DisclosureEvidence,
    PortfolioDisclosure,
    PortfolioMemoryCorpus,
    PortfolioMemoryIndex,
    make_disclosure_id,
)
from vc_clone_graph.schemas_v4 import InvestigationV4
from vc_clone_graph.schemas_v5 import InvestigationV5
from vc_clone_graph.workflow_v4 import VCDecisionWorkflowV4


class TinyEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class ScriptedProvider:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        started = perf_counter()
        self.requests.append(request)
        output = self.outputs.pop(0)
        parsed = output(request) if callable(output) else output
        return GenerationResult(
            parsed=parsed,
            content=json.dumps(parsed),
            usage=Usage(input_tokens=10, output_tokens=5),
            elapsed_seconds=perf_counter() - started,
            raw_metadata={"provider": "scripted"},
        )


def make_index(tmp_path: Path) -> HybridWikiIndex:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "memory.md").write_text(
        "# Founder learning\nThe investor values rapid customer learning and resourceful execution.\n",
        encoding="utf-8",
    )
    return HybridWikiIndex.build(wiki, TinyEmbedder())


def make_portfolio_index():
    slug = "10-calendar"
    disclosure = PortfolioDisclosure(
        disclosure_id=make_disclosure_id("charles", slug, 1, "LetsMeet"),
        vc_slug="charles", company_name="LetsMeet", aliases=("LetsMeet",),
        descriptor="group scheduling software", relationship="investment",
        observed_overlap="possible", observed_consequence="permission_or_check_required",
        source_episode_slug=slug, source_episode_number=10,
        evidence=(DisclosureEvidence(
            source_sha256="a" * 64, turn_index=1, speaker="Charles",
            text="I invested in LetsMeet, a group scheduling company.",
        ),),
        confidence=0.95, validation_status="automated_candidate",
    )
    corpus = PortfolioMemoryCorpus("charles", (disclosure,))
    return PortfolioMemoryIndex.build(corpus, TinyEmbedder()).for_target("18-rowvigor")


def settings(
    tmp_path: Path,
    *,
    p1_max: int = 4,
    p2_max: int = 4,
    contract_version: str = "v4",
    execution_mode: str = "full",
) -> WorkflowSettings:
    return WorkflowSettings(
        episode_slug="18-rowvigor",
        investor_name="Charles Hudson",
        pitch="The founders report five paid pilots and are raising $1m.",
        taxonomy_labels={"founder_execution"},
        taxonomy_records=(
            {
                "label": "founder_execution",
                "definition": "Evidence that the team can turn insight into execution.",
                "coarse_parent": "founding_team",
            },
        ),
        check_tiers={"exploratory_lt_100k", "standard"},
        phase1_min_iterations=1,
        phase1_max_iterations=p1_max,
        phase2_min_iterations=1,
        phase2_max_iterations=p2_max,
        retrieval_top_k=2,
        max_exact_reads=2,
        run_root=tmp_path / "run",
        contract_version=contract_version,
        execution_mode=execution_mode,
    )


def plan(*, query: str = "customer learning") -> dict:
    return {
        "questions": ["Can the founders execute and learn quickly?"],
        "wiki_queries": [query],
        "precedent_queries": [],
        "continuation_focus": "Test founder execution.",
    }


def investigation(
    request: GenerationRequest, *, sufficient: bool = True, unmapped: bool = False
) -> dict:
    evidence_id = re.search(r'"evidence_id": "(W-[A-Za-z0-9_-]+)"', request.prompt).group(1)
    return {
        "schema_version": "investigation-v4",
        "episode_slug": "18-rowvigor",
        "questions": [
            {
                "question": "Can the founders execute and learn quickly?",
                "status": "answered",
                "answer": "Paid pilots are affirmative execution evidence.",
                "evidence_ids": [evidence_id],
            }
        ],
        "rationales": [
            {
                "rationale_id": "R1",
                "taxonomy_label": "founder_execution",
                "direction": "positive",
                "salience": "primary",
                "confidence": 0.8,
                "pitch_evidence": ["five paid pilots"],
                "pitch_evidence_ids": ["P-001"],
                "wiki_evidence_ids": [evidence_id],
                "historical_evidence_ids": [],
                "justification": "The pitch signal matches the investor's execution principle.",
            }
        ],
        "conflicts": [],
        "unmapped_observations": (
            [
                {
                    "observation_id": "U1",
                    "description": "An unusual distribution advantage is material.",
                    "pitch_evidence": ["five paid pilots"],
                    "evidence_ids": [evidence_id],
                    "decision_relevance": "It raises confidence in efficient customer access.",
                }
            ]
            if unmapped
            else []
        ),
        "information_sufficient": sufficient,
        "sufficiency_assessment": (
            "Material execution evidence is sufficient."
            if sufficient
            else "Market evidence could materially alter the rationale record."
        ),
        "searchable_questions": [] if sufficient else ["Can this market scale?"],
        "diligence_questions": [],
        "next_search_objectives": [] if sufficient else ["Find the investor's market-scale bar."],
        "summary": "Founder execution is the controlling positive rationale.",
    }


def decision(request: GenerationRequest, *, sufficient: bool = True, reopen: bool = False) -> dict:
    digest = re.search(r"investigation_sha256 exactly\s+([0-9a-f]{64})", request.prompt).group(1)
    return {
        "schema_version": "decision-v4",
        "episode_slug": "18-rowvigor",
        "investigation_sha256": digest,
        "decision": "In",
        "decision_path": "conventional_fit",
        "investment_likelihood": 0.62,
        "decision_confidence": 0.7,
        "decision_justification": "Paid pilots clear the investor's early execution bar for a small check.",
        "recommended_check_tier": "exploratory_lt_100k",
        "review_priority_score": 0.84,
        "review_priority_reason": "Strong early execution merits direct review.",
        "controlling_rationale_ids": ["R1"],
        "evidence_basis": [
            {
                "source_type": "pitch",
                "source_reference": "five paid pilots",
                "evidence_ids": ["P-001"],
                "interpretation": "Customers already pay.",
                "effect_on_decision": "supports",
            }
        ],
        "strongest_opposing_case": {
            "argument": "The market remains uncertain.",
            "response": "The small check contains that risk.",
        },
        "searchable_questions": [] if sufficient else ["How large can the market become?"],
        "diligence_questions": [],
        "reversal_conditions": ["The pilots are not genuinely paid."],
        "information_sufficient": sufficient,
        "sufficiency_assessment": "A usable current decision exists.",
        "next_search_objectives": [] if sufficient else ["Find comparable market exceptions."],
        "phase1_reopen_recommended": reopen,
        "missing_considerations": ["market_size_assessment"] if reopen else [],
        "founder_exception_considered": True,
        "founder_exception_rationale_ids": [],
        "founder_exception_precedent_ids": [],
        "founder_exception_assessment": "The conventional case is sufficient; no exception is required.",
    }


def investigation_v41(
    request: GenerationRequest,
    *,
    sufficient: bool = True,
    reviewed_pitch_evidence_ids: list[str] | None = None,
    triggered_hard_constraint: bool = False,
) -> dict:
    payload = investigation(request, sufficient=sufficient)
    for row in payload["rationales"]:
        row.pop("pitch_evidence", None)
    evidence_id = payload["rationales"][0]["wiki_evidence_ids"][0]
    payload.update(
        schema_version="investigation-v4.1",
        reviewed_pitch_evidence_ids=(
            reviewed_pitch_evidence_ids
            if reviewed_pitch_evidence_ids is not None
            else ["P-001"]
        ),
        material_statement_coverage=[
            {
                "pitch_evidence_id": "P-001",
                "decision_dimension": "founder_execution",
                "direction": "positive",
                "constraint_signal": (
                    "triggered" if triggered_hard_constraint else "none"
                ),
                "mapping_type": "taxonomy_rationale",
                "mapped_ids": ["R1"],
                "assessment": "The paid pilots are material execution evidence.",
            }
        ],
        constraint_assessments=(
            [
                {
                    "constraint_id": "C1",
                    "constraint_kind": "category_or_expertise_fit",
                    "policy_statement": "The investor only commits inside areas they can evaluate.",
                    "status": "triggered",
                    "severity": "hard",
                    "pitch_evidence_ids": ["P-001"],
                    "wiki_evidence_ids": [evidence_id],
                    "historical_evidence_ids": [],
                    "mapped_ids": ["R1"],
                    "assessment": "The supplied record establishes a category mismatch.",
                }
            ]
            if triggered_hard_constraint
            else []
        ),
    )
    return payload


def decision_v41(
    request: GenerationRequest,
    *,
    sufficient: bool = True,
) -> dict:
    payload = decision(request, sufficient=sufficient)
    payload.update(
        schema_version="decision-v4.1",
        blocking_constraint_ids=[],
        constraint_assessment_summary="No supplied constraint blocks the small check.",
        founder_exception_precedent_match="No founder exception is required.",
    )
    return payload


def workflow(
    tmp_path: Path,
    outputs,
    *,
    p1_max: int = 4,
    p2_max: int = 4,
    contract_version: str = "v4",
    portfolio_index=None,
    execution_mode: str = "full",
):
    provider = ScriptedProvider(outputs)
    workflow_settings = settings(
            tmp_path,
            p1_max=p1_max,
            p2_max=p2_max,
            contract_version=contract_version,
            execution_mode=execution_mode,
        )
    if portfolio_index is not None:
        workflow_settings = replace(
            workflow_settings,
            portfolio_memory_enabled=True,
            portfolio_retrieval_top_k=5,
            portfolio_candidate_pool_k=15,
        )
    graph = VCDecisionWorkflowV4(
        workflow_settings,
        make_index(tmp_path),
        checkpointer=InMemorySaver(),
        phase1_provider=provider,
        phase2_provider=provider,
        portfolio_index=portfolio_index,
    )
    return graph, provider


def rationale_lock_v42(request: GenerationRequest) -> dict:
    digest = request.schema["properties"]["candidate_investigation_sha256"]["const"]
    return {
        "schema_version": "rationale-lock-v4.2",
        "episode_slug": "18-rowvigor",
        "candidate_investigation_sha256": digest,
        "dispositions": [
            {
                "rationale_id": "R1",
                "decision": "locked",
                "rejection_reason": None,
                "direction": "positive",
                "salience": "primary",
                "confidence": 0.79,
                "materiality_justification": "Paid pilots directly and materially activate execution.",
            }
        ],
        "lock_summary": "The single candidate is material and investor-specific.",
    }


def test_v42_phase1_only_locks_candidates_and_never_calls_phase2(tmp_path: Path):
    graph, provider = workflow(
        tmp_path,
        [plan(), investigation_v41, rationale_lock_v42],
        contract_version="v4.2",
        execution_mode="phase1_only",
    )

    result = graph.invoke("v42-phase1-only")

    assert [request.phase for request in provider.requests] == [
        "phase1_plan",
        "phase1",
        "phase1_lock",
    ]
    assert result["phase1_status"] == "accepted"
    assert result["phase2_status"] == "not_run"
    assert result["investigation"]["schema_version"] == "investigation-v4.2"
    assert [row["rationale_id"] for row in result["investigation"]["rationales"]] == ["R1"]
    assert (tmp_path / "run/phase1/candidate-investigation.json").is_file()
    assert (tmp_path / "run/phase1/rationale-lock.json").is_file()
    assert (tmp_path / "run/phase1/investigation.json").is_file()
    assert not (tmp_path / "run/phase2").exists()
    verify_phase1_artifacts(tmp_path / "run")


def test_v42_can_reject_every_candidate_without_losing_phase1_context(tmp_path: Path):
    def reject_all(request: GenerationRequest) -> dict:
        payload = rationale_lock_v42(request)
        payload["dispositions"][0].update(
            decision="rejected",
            rejection_reason="diligence_only",
            direction=None,
            salience=None,
            confidence=None,
            materiality_justification="The execution item is only ordinary diligence here.",
        )
        return payload

    graph, _ = workflow(
        tmp_path,
        [plan(), investigation_v41, reject_all],
        contract_version="v4.2",
        execution_mode="phase1_only",
    )

    result = graph.invoke("v42-zero-lock")

    assert result["investigation"]["rationales"] == []
    assert result["investigation"]["questions"]
    assert result["investigation"]["material_statement_coverage"]
    assert result["investigation"]["rejected_candidate_count"] == 1
    verify_phase1_artifacts(tmp_path / "run")


def test_v42_repairs_one_invalid_lock_response(tmp_path: Path):
    graph, provider = workflow(
        tmp_path,
        [plan(), investigation_v41, None, rationale_lock_v42],
        contract_version="v4.2",
        execution_mode="phase1_only",
    )

    result = graph.invoke("v42-lock-repair")

    assert result["phase1_status"] == "accepted"
    assert len([row for row in provider.requests if row.phase == "phase1_lock"]) == 2
    assert any(
        row.startswith("PHASE1_RATIONALE_LOCK_INVALID_ATTEMPT_1:")
        for row in result["phase1_findings"]
    )


def test_v42_normalizes_structured_output_branch_fill_without_retry(tmp_path: Path):
    def branch_filled(request: GenerationRequest) -> dict:
        payload = rationale_lock_v42(request)
        payload["dispositions"][0]["rejection_reason"] = "generic_checklist"
        return payload

    graph, provider = workflow(
        tmp_path,
        [plan(), investigation_v41, branch_filled],
        contract_version="v4.2",
        execution_mode="phase1_only",
    )

    result = graph.invoke("v42-lock-normalization")

    assert result["phase1_status"] == "accepted"
    assert len([row for row in provider.requests if row.phase == "phase1_lock"]) == 1
    assert "NORMALIZED_LOCKED_REJECTION_REASON:R1" in result["phase1_findings"]


def test_v41_retrieves_temporal_portfolio_and_records_overlap(tmp_path: Path):
    portfolio = make_portfolio_index()

    def with_overlap(request: GenerationRequest) -> dict:
        payload = investigation_v41(request)
        entity = portfolio.entities[0]
        payload["portfolio_overlap_assessments"] = [{
            "portfolio_entity_id": entity.entity_id,
            "disclosure_ids": list(entity.disclosure_ids),
            "pitch_evidence_ids": ["P-001"],
            "overlap_status": "possible_conflict",
            "decision_consequence": "permission_required",
            "confidence": 0.72,
            "assessment": "Both products coordinate groups, but direct competition is not established.",
        }]
        return payload

    graph, provider = workflow(
        tmp_path,
        [plan(), with_overlap, plan(), decision_v41],
        contract_version="v4.1",
        portfolio_index=portfolio,
    )
    result = graph.invoke("portfolio")
    assert result["phase1_status"] == "accepted"
    assert result["investigation"]["portfolio_overlap_assessments"][0][
        "decision_consequence"
    ] == "permission_required"
    prompt = provider.requests[1].prompt
    assert "I invested in LetsMeet" in prompt
    assert (tmp_path / "run/phase1/turn-01/portfolio-candidates.json").is_file()


def test_v4_freezes_each_phase_after_one_sufficient_iteration(tmp_path: Path):
    graph, provider = workflow(tmp_path, [plan(), investigation, plan(), decision])

    result = graph.invoke("one")

    assert result["phase1_status"] == "accepted"
    assert result["phase2_status"] == "accepted"
    assert result["phase1_iteration"] == 1
    assert result["phase2_iteration"] == 1
    assert result["decision"]["decision"] == "In"
    assert len(provider.requests) == 4
    assert (tmp_path / "run/phase1/investigation.json").exists()
    assert (tmp_path / "run/phase2/decision.json").exists()
    verify_run_artifacts(tmp_path / "run")


def test_v41_freezes_complete_coverage_and_preserves_contract_version(tmp_path: Path):
    graph, _ = workflow(
        tmp_path,
        [plan(), investigation_v41, plan(), decision_v41],
        contract_version="v4.1",
    )

    result = graph.invoke("v41-complete")

    assert result["phase1_status"] == "accepted"
    assert result["phase2_status"] == "accepted"
    assert result["investigation"]["reviewed_pitch_evidence_ids"] == ["P-001"]
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["contract_version"] == "v4.1"
    verify_run_artifacts(tmp_path / "run")


def test_v41_normalizes_repairable_response_and_preserves_raw_artifact(
    tmp_path: Path,
):
    def repairable(request: GenerationRequest) -> dict:
        payload = investigation_v41(request)
        payload["questions"][0]["evidence_ids"].append("P-001")
        payload["rationales"][0]["rationale_id"] = "R-001"
        payload["material_statement_coverage"][0]["mapped_ids"] = [
            "founder_execution"
        ]
        return payload

    graph, _ = workflow(
        tmp_path,
        [plan(), repairable, plan(), decision_v41],
        contract_version="v4.1",
    )

    result = graph.invoke("v41-normalize")

    assert result["phase1_status"] == "accepted"
    assert result["investigation"]["rationales"][0]["rationale_id"] == "R1"
    assert any(
        finding == "NORMALIZED_GENERATED_ID:R-001->R1"
        for finding in result["phase1_findings"]
    )
    raw = json.loads(
        (
            tmp_path
            / "run/phase1/turn-01/investigation-model-response.json"
        ).read_text()
    )
    assert raw["parsed"]["rationales"][0]["rationale_id"] == "R-001"


def test_v41_drops_inaccessible_citation_when_valid_investor_evidence_remains(
    tmp_path: Path,
):
    def citation_typo(request: GenerationRequest) -> dict:
        payload = investigation_v41(request)
        payload["rationales"][0]["wiki_evidence_ids"].append("W-not-retrieved")
        return payload

    graph, _ = workflow(
        tmp_path,
        [plan(), citation_typo, plan(), decision_v41],
        contract_version="v4.1",
    )

    result = graph.invoke("v41-citation-filter")

    assert "W-not-retrieved" not in result["investigation"]["rationales"][0][
        "wiki_evidence_ids"
    ]
    assert any(
        row.startswith("NORMALIZED_DROPPED_INACCESSIBLE_RATIONALE_EVIDENCE:R1:")
        for row in result["phase1_findings"]
    )


def test_v41_revisits_incomplete_pitch_review_before_accepting(tmp_path: Path):
    graph, _ = workflow(
        tmp_path,
        [
            plan(),
            lambda request: investigation_v41(
                request, reviewed_pitch_evidence_ids=["P-999"]
            ),
            plan(),
            investigation_v41,
            plan(),
            decision_v41,
        ],
        p1_max=2,
        contract_version="v4.1",
    )

    result = graph.invoke("v41-coverage-revisit")

    assert result["phase1_iteration"] == 2
    assert result["phase1_status"] == "accepted"
    assert any(
        finding.startswith("PITCH_REVIEW_COVERAGE_MISSING:")
        for finding in result["phase1_findings"]
    )


def test_v41_retains_but_does_not_accept_in_over_triggered_hard_constraint(
    tmp_path: Path,
):
    graph, _ = workflow(
        tmp_path,
        [
            plan(),
            lambda request: investigation_v41(
                request, triggered_hard_constraint=True
            ),
            plan(),
            decision_v41,
        ],
        p2_max=1,
        contract_version="v4.1",
    )

    result = graph.invoke("v41-hard-constraint")

    assert result["decision"]["decision"] == "In"
    assert result["phase2_status"] == "provisional"
    assert any(
        finding.startswith("IN_CONFLICTS_WITH_TRIGGERED_HARD_CONSTRAINT:")
        for finding in result["phase2_findings"]
    )


def test_v41_prepends_general_policy_retrieval_query(tmp_path: Path):
    graph, _ = workflow(
        tmp_path,
        [plan(query="customer learning"), investigation_v41, plan(), decision_v41],
        contract_version="v4.1",
    )

    result = graph.invoke("v41-policy-query")

    assert "hard rules category expertise" in result["phase1_plan"]["wiki_queries"][0]
    assert "hard rules category expertise" in result["phase1_plan"]["precedent_queries"][0]


def test_v4_generation_schemas_bind_run_identifiers(tmp_path: Path):
    graph, provider = workflow(tmp_path, [plan(), investigation, plan(), decision])

    result = graph.invoke("schema-identifiers")

    investigation_request = provider.requests[1]
    decision_request = provider.requests[3]
    assert investigation_request.schema["properties"]["episode_slug"]["const"] == (
        "18-rowvigor"
    )
    assert decision_request.schema["properties"]["episode_slug"]["const"] == (
        "18-rowvigor"
    )
    assert decision_request.schema["properties"]["investigation_sha256"][
        "const"
    ] == result["investigation_sha256"]


def test_v4_phase2_replay_starts_at_decision_planning(tmp_path: Path):
    (tmp_path / "source").mkdir()
    source_graph, _ = workflow(
        tmp_path / "source", [plan(), investigation, plan(), decision]
    )
    source_result = source_graph.invoke("source")
    frozen = InvestigationV4.model_validate(source_result["investigation"])

    replay_provider = ScriptedProvider([plan(), decision])
    (tmp_path / "replay").mkdir()
    replay_settings = replace(
        settings(tmp_path / "replay"),
        run_root=tmp_path / "replay" / "run",
    )
    replay = VCDecisionWorkflowV4(
        replay_settings,
        make_index(tmp_path / "replay"),
        checkpointer=InMemorySaver(),
        phase1_provider=replay_provider,
        phase2_provider=replay_provider,
    )

    result = replay.invoke_phase2(
        "phase2-only",
        frozen,
        source_result["investigation_sha256"],
        phase1_status=source_result["phase1_status"],
        phase1_state=source_result,
    )

    assert result["phase2_status"] == "accepted"
    assert [request.phase for request in replay_provider.requests] == [
        "phase2_plan",
        "phase2",
    ]
    assert result["usage_by_phase"]["phase1"] == source_result["usage_by_phase"][
        "phase1"
    ]


def test_v4_phase2_replay_carries_source_bound_rationale_associations(tmp_path: Path):
    (tmp_path / "source").mkdir()
    source_graph, _ = workflow(
        tmp_path / "source", [plan(), investigation, plan(), decision]
    )
    source_result = source_graph.invoke("source-with-associations")
    frozen = InvestigationV4.model_validate(source_result["investigation"])
    associations = {
        "schema": "rationale-association-annotations-v1",
        "scientific_status": "fold_safe_advisory_hypotheses",
        "vc_slug": "charles-hudson",
        "episode_slug": "18-rowvigor",
        "investigation_sha256": source_result["investigation_sha256"],
        "thresholds": {
            "min_posterior_probability": 0.6,
            "min_support": 3,
            "min_lift": 1.0,
            "max_per_source": 5,
        },
        "training_episode_slugs": ["20-harper-wilde"],
        "claim_annotations": [],
        "material_statement_annotations": [],
        "unmapped_observation_annotations": [],
        "episode_level_associations": [],
    }
    replay_provider = ScriptedProvider([plan(), decision])
    (tmp_path / "replay").mkdir()
    replay = VCDecisionWorkflowV4(
        replace(settings(tmp_path / "replay"), run_root=tmp_path / "replay/run"),
        make_index(tmp_path / "replay"),
        checkpointer=InMemorySaver(),
        phase1_provider=replay_provider,
        phase2_provider=replay_provider,
    )

    result = replay.invoke_phase2(
        "phase2-with-associations",
        frozen,
        source_result["investigation_sha256"],
        phase1_status=source_result["phase1_status"],
        phase1_state=source_result,
        rationale_association_annotations=associations,
    )

    assert result["rationale_association_annotations"] == associations
    assert all(
        "fold_safe_advisory_hypotheses" in request.prompt
        for request in replay_provider.requests
    )
    persisted = (tmp_path / "replay/run/state.json").read_text(encoding="utf-8")
    assert "rationale-association-annotations-v1" in persisted


def test_v5_phase2_replay_uses_nested_associations_without_reopening_phase1(
    tmp_path: Path,
):
    source = investigation(
        GenerationRequest("phase1", '"evidence_id": "W-test"', {})
    )
    source["rationales"][0]["associated_rationales"] = [
        {
            "taxonomy_label": "market_size_assessment",
            "posterior_probability": 0.75,
            "support": 4,
            "lift": 2.0,
            "antecedent_labels": ["founder_execution"],
            "status": "hypothesis_only",
        }
    ]
    v5 = InvestigationV5.model_validate({
        **source,
        "schema_version": "investigation-v5",
        "source_schema_version": "investigation-v4",
        "source_investigation_sha256": "a" * 64,
        "association_training_episode_slugs": ["20-harper-wilde"],
        "association_thresholds": {
            "min_posterior_probability": 0.6,
            "min_support": 3,
            "min_lift": 1.0,
            "max_per_source": 5,
        },
        "episode_level_associations": source["rationales"][0]["associated_rationales"],
    })
    provider = ScriptedProvider([plan(), decision])
    graph = VCDecisionWorkflowV4(
        replace(
            settings(tmp_path, contract_version="v5"),
            run_root=tmp_path / "run",
        ),
        make_index(tmp_path),
        checkpointer=InMemorySaver(),
        phase1_provider=provider,
        phase2_provider=provider,
    )

    result = graph.invoke_phase2_v5(
        "phase2-v5", v5, "b" * 64, phase1_status="accepted"
    )

    assert result["phase2_status"] == "accepted"
    assert result["investigation"]["schema_version"] == "investigation-v5"
    assert all("market_size_assessment" in request.prompt for request in provider.requests)
    assert result["decision"]["controlling_rationale_ids"] == ["R1"]


def test_v4_retrieval_persists_accumulated_historical_state_and_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = VCDecisionWorkflowV4(
        settings(tmp_path),
        make_index(tmp_path),
        checkpointer=InMemorySaver(),
        phase1_provider=ScriptedProvider([]),
        phase2_provider=ScriptedProvider([]),
    )
    prior = {"evidence_id": "H-prior", "episode_slug": "10-prior"}
    current = {"evidence_id": "H-current", "episode_slug": "11-current"}
    search = {"query_id": "phase2-t01-precedent-01", "query": "peer"}
    read = {"episode_slug": "11-current", "status": "ok"}
    monkeypatch.setattr(
        "vc_clone_graph.workflow_v4.retrieve_v4",
        lambda **_: V4RetrievalResult(
            wiki_searches=(), precedent_searches=(search,), wiki_evidence=(),
            historical_evidence=(current,), precedent_reads=(read,),
            opened_episode_slugs=("11-current",), warnings=(),
        ),
    )

    update = graph._retrieve({
        "phase2_plan": plan(), "phase2_iteration": 0,
        "phase2_historical_evidence": [prior],
        "phase2_precedent_searches": [], "phase2_precedent_reads": [],
        "evidence_registry": {"H-prior": prior}, "events": [],
    }, "phase2")

    assert update["phase2_historical_evidence"] == [prior, current]
    assert update["phase2_precedent_searches"] == [search]
    assert update["phase2_precedent_reads"] == [read]
    artifact = json.loads((
        tmp_path / "run/phase2/turn-01/historical-evidence.json"
    ).read_text(encoding="utf-8"))
    assert artifact == [prior, current]


def test_v4_phase2_replay_rejects_mismatched_rationale_associations(tmp_path: Path):
    provider = ScriptedProvider([])
    graph = VCDecisionWorkflowV4(
        settings(tmp_path),
        make_index(tmp_path),
        checkpointer=InMemorySaver(),
        phase1_provider=provider,
        phase2_provider=provider,
    )
    frozen = InvestigationV4.model_validate(
        investigation(
            GenerationRequest("phase1", '"evidence_id": "W-test"', {}),
        )
    )

    with pytest.raises(ValueError, match="association annotation investigation hash mismatch"):
        graph.invoke_phase2(
            "bad-associations",
            frozen,
            "a" * 64,
            phase1_status="accepted",
            rationale_association_annotations={
                "episode_slug": "18-rowvigor",
                "investigation_sha256": "b" * 64,
            },
        )


def test_v4_phase2_replay_rejects_association_training_leakage(tmp_path: Path):
    provider = ScriptedProvider([])
    graph = VCDecisionWorkflowV4(
        settings(tmp_path),
        make_index(tmp_path),
        checkpointer=InMemorySaver(),
        phase1_provider=provider,
        phase2_provider=provider,
    )
    frozen = InvestigationV4.model_validate(
        investigation(GenerationRequest("phase1", '"evidence_id": "W-test"', {}))
    )

    with pytest.raises(ValueError, match="held episode appears in association training"):
        graph.invoke_phase2(
            "leaking-associations",
            frozen,
            "a" * 64,
            phase1_status="accepted",
            rationale_association_annotations={
                "schema": "rationale-association-annotations-v1",
                "scientific_status": "fold_safe_advisory_hypotheses",
                "vc_slug": "charles-hudson",
                "episode_slug": "18-rowvigor",
                "investigation_sha256": "a" * 64,
                "thresholds": {
                    "min_posterior_probability": 0.6,
                    "min_support": 3,
                    "min_lift": 1.0,
                    "max_per_source": 5,
                },
                "training_episode_slugs": ["18-rowvigor"],
                "claim_annotations": [],
                "material_statement_annotations": [],
                "unmapped_observation_annotations": [],
                "episode_level_associations": [],
            },
        )


def test_v4_retries_invalid_phase1_plan_once_before_fallback(tmp_path: Path):
    graph, provider = workflow(
        tmp_path,
        [None, plan(), investigation, plan(), decision],
    )

    result = graph.invoke("planner-repair")

    phase1_plans = [
        request for request in provider.requests if request.phase == "phase1_plan"
    ]
    assert len(phase1_plans) == 2
    assert "Return one concise complete JSON object" in phase1_plans[1].prompt
    assert (
        tmp_path
        / "run/phase1/turn-01/plan-retry-model-response.json"
    ).exists()
    assert not any("FALLBACK" in finding for finding in result["phase1_findings"])
    assert any(
        "PHASE1_PLAN_INVALID_ATTEMPT_1" in finding
        for finding in result["phase1_findings"]
    )
    assert result["usage"]["input_tokens"] == 50
    assert result["usage_by_phase"]["phase1"]["input_tokens"] == 30


def test_v4_uses_fallback_after_both_phase1_plan_attempts_fail(tmp_path: Path):
    graph, provider = workflow(
        tmp_path,
        [None, {}, investigation, plan(), decision],
    )

    result = graph.invoke("planner-double-failure")

    assert len(
        [request for request in provider.requests if request.phase == "phase1_plan"]
    ) == 2
    assert any(
        "PHASE1_PLAN_INVALID_ATTEMPT_1" in finding
        for finding in result["phase1_findings"]
    )
    assert any(
        "PHASE1_PLAN_INVALID_ATTEMPT_2" in finding
        for finding in result["phase1_findings"]
    )
    assert "PHASE1_PLAN_FALLBACK_USED" in result["phase1_findings"]
    assert result["decision"]["decision"] == "In"


def test_v4_continues_on_material_gap_and_uses_only_current_turn_evidence(tmp_path: Path):
    graph, provider = workflow(
        tmp_path,
        [
            plan(query="customer learning"),
            lambda request: investigation(request, sufficient=False),
            plan(query="market scale"),
            investigation,
            plan(),
            decision,
        ],
    )

    result = graph.invoke("adaptive")

    assert result["phase1_iteration"] == 2
    first_prompt = provider.requests[1].prompt
    second_prompt = provider.requests[3].prompt
    first_evidence = set(re.findall(r'"evidence_id": "(W-[A-Za-z0-9_-]+)"', first_prompt))
    second_evidence = set(re.findall(r'"evidence_id": "(W-[A-Za-z0-9_-]+)"', second_prompt))
    assert first_evidence
    assert second_evidence
    assert "accessible_episode_inventory" not in second_prompt


def test_v4_retains_prior_valid_investigation_after_malformed_final_turn(tmp_path: Path):
    graph, _ = workflow(
        tmp_path,
        [
            plan(),
            lambda request: investigation(request, sufficient=False),
            plan(query="market scale"),
            {},
            plan(),
            decision,
        ],
        p1_max=2,
    )

    result = graph.invoke("fallback")

    assert result["phase1_status"] == "provisional"
    assert result["phase1_iteration"] == 2
    assert result["investigation"]["summary"] == "Founder execution is the controlling positive rationale."
    assert any("INVALID" in finding for finding in result["phase1_findings"])
    assert result["decision"]["decision"] == "In"


def test_v4_retains_decision_when_reconsideration_is_malformed(tmp_path: Path):
    graph, _ = workflow(
        tmp_path,
        [
            plan(),
            investigation,
            plan(),
            lambda request: decision(request, sufficient=False),
            plan(query="market exception"),
            {},
        ],
        p2_max=2,
    )

    result = graph.invoke("decision-fallback")

    assert result["phase2_status"] == "provisional"
    assert result["phase2_iteration"] == 2
    assert result["decision"]["decision"] == "In"
    assert result["decision"]["review_priority_score"] == 0.84


def test_v4_retains_stable_decision_when_no_novelty_turn_flips_without_new_rationale(
    tmp_path: Path,
):
    def candidate(
        request: GenerationRequest,
        *,
        value: str,
        question: str,
        likelihood: float,
    ) -> dict:
        payload = decision(request, sufficient=False)
        payload.update(
            decision=value,
            decision_path=("conventional_fit" if value == "In" else "out"),
            investment_likelihood=likelihood,
            decision_confidence=0.6,
            searchable_questions=[question],
            next_search_objectives=[question],
        )
        if value == "Out":
            payload["recommended_check_tier"] = "none"
        return payload

    graph, _ = workflow(
        tmp_path,
        [
            plan(),
            investigation,
            plan(),
            lambda request: candidate(
                request, value="In", question="Check regulation.", likelihood=0.58
            ),
            plan(query="regulation"),
            lambda request: candidate(
                request, value="In", question="Check economics.", likelihood=0.57
            ),
            plan(query="economics"),
            lambda request: candidate(
                request, value="In", question="Check capital needs.", likelihood=0.57
            ),
            plan(query="capital needs"),
            lambda request: candidate(
                request, value="Out", question="Check capital needs.", likelihood=0.36
            ),
        ],
        p2_max=4,
    )

    result = graph.invoke("stable-no-novelty-reversal")

    assert result["phase2_status"] == "provisional"
    assert result["phase2_iteration"] == 4
    assert result["decision"]["decision"] == "In"
    assert result["decision"]["investment_likelihood"] == 0.57
    assert "PHASE2_NO_NOVELTY_STOP" in result["phase2_findings"]
    assert (
        "PHASE2_NO_NOVELTY_REVERSAL_RETAINED_STABLE_DECISION"
        in result["phase2_findings"]
    )


def test_v4_phase2_can_reopen_phase1_without_deleting_current_decision(tmp_path: Path):
    graph, provider = workflow(
        tmp_path,
        [
            plan(),
            investigation,
            plan(),
            lambda request: decision(request, sufficient=False, reopen=True),
            plan(query="market size bar"),
            investigation,
            plan(),
            decision,
        ],
        p1_max=2,
        p2_max=2,
    )

    result = graph.invoke("reopen")

    assert result["phase1_iteration"] == 2
    assert result["phase2_iteration"] == 2
    assert result["decision"]["decision"] == "In"
    assert result["phase2_status"] == "accepted"
    decision_events = [e for e in result["events"] if e["kind"] == "phase2_decision"]
    assert [e["action"] for e in decision_events] == ["reopen_phase1", "freeze_accepted"]
    assert len(provider.requests) == 8


def test_v4_rolls_back_reopened_phase1_if_final_decision_response_is_malformed(
    tmp_path: Path,
):
    def revised_investigation(request: GenerationRequest) -> dict:
        value = investigation(request)
        value["summary"] = "A materially revised Phase 1 candidate."
        return value

    graph, _ = workflow(
        tmp_path,
        [
            plan(),
            investigation,
            plan(),
            lambda request: decision(request, sufficient=False, reopen=True),
            plan(query="market size bar"),
            revised_investigation,
            plan(),
            {},
        ],
        p1_max=2,
        p2_max=2,
    )

    result = graph.invoke("reopen-fallback")

    assert result["decision"]["decision"] == "In"
    assert result["investigation"]["summary"] == "Founder execution is the controlling positive rationale."
    assert "PHASE1_ROLLED_BACK_TO_RETAIN_DECISION_BINDING" in result["phase2_findings"]
    verify_run_artifacts(tmp_path / "run")


def test_v4_reflection_runs_only_after_decision_and_writes_separate_proposal(tmp_path: Path):
    def reflection(request: GenerationRequest) -> dict:
        assert (tmp_path / "run/phase2/decision.json").exists()
        digest = re.search(r"decision_sha256 exactly ([0-9a-f]{64})", request.prompt).group(1)
        evidence_id = re.search(r'"evidence_ids": \[\s*"(W-[A-Za-z0-9_-]+)"', request.prompt).group(1)
        return {
            "schema_version": "taxonomy-reflection-v4",
            "episode_slug": "18-rowvigor",
            "decision_sha256": digest,
            "proposals": [
                {
                    "provisional_id": "P1",
                    "proposed_label": "distribution_access_advantage",
                    "proposed_definition": "Investor-specific access that materially lowers distribution friction.",
                    "confidence": 0.92,
                    "material_to_frozen_decision": True,
                    "reusable_in_future_decisions": True,
                    "pitch_evidence": ["five paid pilots"],
                    "investor_evidence_ids": [evidence_id],
                    "nearest_existing_labels": ["founder_execution"],
                    "why_existing_taxonomy_is_insufficient": "Execution does not capture privileged distribution access.",
                    "necessity_justification": "The distinction materially affected the frozen decision.",
                }
            ],
            "review_summary": "One high-confidence proposal requires human review.",
        }

    configured = settings(tmp_path)
    original_taxonomy = configured.taxonomy_records
    provider = ScriptedProvider(
        [
            plan(),
            lambda request: investigation(request, unmapped=True),
            plan(),
            decision,
            reflection,
        ]
    )
    graph = VCDecisionWorkflowV4(
        configured,
        make_index(tmp_path),
        checkpointer=InMemorySaver(),
        phase1_provider=provider,
        phase2_provider=provider,
    )

    result = graph.invoke("reflection")

    assert result["taxonomy_reflection"]["proposals"][0]["confidence"] == 0.92
    assert configured.taxonomy_records == original_taxonomy
    assert (tmp_path / "run/reflection/taxonomy-proposals.json").exists()
    assert provider.requests[-1].phase == "taxonomy_reflection"


def rationale_mapping_v43(request: GenerationRequest) -> dict:
    digest = request.schema["properties"]["candidate_investigation_sha256"]["const"]
    return {
        "schema_version": "rationale-mapping-v4.3",
        "episode_slug": "18-rowvigor",
        "candidate_investigation_sha256": digest,
        "dispositions": [{
            "rationale_id": "R1",
            "action": "keep",
            "target_taxonomy_label": "founder_execution",
            "merge_into_rationale_id": None,
            "winning_definition": "Evidence that the team can execute.",
            "rejected_alternatives": [],
            "mapping_justification": "Paid pilots directly support execution.",
        }],
        "mapping_summary": "The candidate already uses the best family label.",
    }


def test_v43_phase1_only_maps_candidates_and_never_calls_phase2(tmp_path: Path):
    graph, provider = workflow(
        tmp_path,
        [plan(), investigation_v41, rationale_mapping_v43],
        contract_version="v4.3",
        execution_mode="phase1_only",
    )

    result = graph.invoke("v43-phase1-only")

    assert [request.phase for request in provider.requests] == [
        "phase1_plan", "phase1", "phase1_mapping"
    ]
    assert result["phase1_status"] == "accepted"
    assert result["phase2_status"] == "not_run"
    assert result["investigation"]["schema_version"] == "investigation-v4.3"
    assert result["investigation"]["mapping_status"] == "accepted"
    assert (tmp_path / "run/phase1/candidate-investigation.json").is_file()
    assert (tmp_path / "run/phase1/rationale-mapping.json").is_file()
    assert not (tmp_path / "run/phase2").exists()
    verify_phase1_artifacts(tmp_path / "run")


def test_v43_invalid_mapping_falls_back_to_usable_candidate(tmp_path: Path):
    graph, provider = workflow(
        tmp_path,
        [plan(), investigation_v41, None, None],
        contract_version="v4.3",
        execution_mode="phase1_only",
    )

    result = graph.invoke("v43-mapping-fallback")

    assert len([row for row in provider.requests if row.phase == "phase1_mapping"]) == 2
    assert result["phase1_status"] == "provisional"
    assert result["investigation"]["mapping_status"] == "fallback_original"
    assert result["investigation"]["rationales"][0]["taxonomy_label"] == "founder_execution"
    assert any("MAPPING_FALLBACK" in row for row in result["phase1_findings"])
    verify_phase1_artifacts(tmp_path / "run")
