from dataclasses import replace
from pathlib import Path
from hashlib import sha256
import json

from langgraph.checkpoint.memory import InMemorySaver
import pytest

from vc_clone_graph.graph import VCDecisionWorkflow, WorkflowSettings
from vc_clone_graph.artifacts import canonical_bytes
from vc_clone_graph.providers.fake import FakeProvider
from vc_clone_graph.precedents import (
    DecisionEvidence,
    PrecedentCorpus,
    PrecedentDecision,
    PrecedentEpisode,
    TranscriptTurn,
)
from vc_clone_graph.retrieval import HybridWikiIndex
from vc_clone_graph.schemas import (
    Investigation,
    InvestigationV2,
    InvestigationV3,
    investigation_v3_json_schema,
    validate_investigation_v3,
)


class TinyEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


def make_index(tmp_path: Path) -> HybridWikiIndex:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "persona.md").write_text(
        "# Founder\nElizabeth values fast customer learning by resourceful founders.\n\n"
        "# Market\nA credible path to a large market matters.\n",
        encoding="utf-8",
    )
    (wiki / "portfolio.md").write_text(
        "# Portfolio\nExisting portfolio holdings and conflict review matter.\n",
        encoding="utf-8",
    )
    return HybridWikiIndex.build(wiki, TinyEmbedder())


def phase1(chunk_id: str, *, saturated: bool = True, label: str = "founder_execution") -> dict:
    return {
        "episode_slug": "135-thoras-ai-the-twin-effect",
        "questions": ["Are founders learning from customers?"],
        "rationales": [
            {
                "rationale_id": "R1",
                "label": label,
                "direction": "positive",
                "salience": "primary",
                "confidence": 0.8,
                "pitch_evidence": ["five paid pilots"],
                "wiki_evidence_ids": [chunk_id],
                "interpretation": "Paid pilots support founder learning.",
            }
        ],
        "conflicts": [],
        "unanswered_questions": [],
        "saturated": saturated,
        "summary": "Strong customer-learning signal.",
    }


def phase1_v2(chunk_id: str, *, unresolved: bool = False) -> dict:
    candidate = phase1(chunk_id)
    candidate["schema_version"] = "investigation-v2"
    candidate["rationales"][0].update(
        {
            "direction": "neutral" if unresolved else "positive",
            "evidence_status": "unresolved" if unresolved else "affirmative_positive",
            "constraint_kind": "portfolio_overlap" if unresolved else "none",
            "constraint_severity": "routine" if unresolved else "none",
            "severity_basis": (
                "Only category adjacency is visible."
                if unresolved
                else "This is a positive signal, not a constraint."
            ),
        }
    )
    unknown = {"status": "unknown", "value": None, "pitch_evidence": []}
    candidate["deal_context"] = {
        field: dict(unknown)
        for field in (
            "total_round_size",
            "company_stage",
            "entry_valuation",
            "possible_investor_check",
            "lead_required",
            "ownership_feasibility",
        )
    }
    return candidate


def phase1_v3(
    chunk_id: str,
    historical_ids: list[str],
    *,
    saturated: bool,
    taxonomy_labels: set[str] | None = None,
) -> dict:
    labels = taxonomy_labels or {"founder_execution"}
    candidate = phase1_v2(chunk_id)
    candidate["schema_version"] = "investigation-v3"
    candidate["saturated"] = saturated
    candidate["rationales"][0].update(
        {
            "historical_evidence_ids": historical_ids,
            "precedent_interpretation": (
                "The opened precedent supports the execution pattern."
                if historical_ids
                else "No exact historical precedent has been opened yet."
            ),
        }
    )
    candidate.update(
        {
            "activated_candidates": ["founder_execution"],
            "queried_candidates": sorted(labels),
            "rejected_candidates": sorted(labels - {"founder_execution"}),
            "unmapped_candidates": [],
            "taxonomy_dispositions": [
                {
                    "label": label,
                    "disposition": (
                        "activated" if label == "founder_execution" else "rejected"
                    ),
                    "basis": (
                        "Five paid pilots provide pitch-grounded execution evidence."
                        if label == "founder_execution"
                        else "The pitch and retrieved evidence do not activate this rationale."
                    ),
                }
                for label in sorted(labels)
            ],
        }
    )
    return candidate


def make_precedent_corpus(embedder=None) -> PrecedentCorpus:
    def episode(slug: str, number: int, verdict: str) -> PrecedentEpisode:
        turns = (
            TranscriptTurn(turn_index=0, speaker="Founder", text=f"{slug} founder signal"),
            TranscriptTurn(turn_index=1, speaker="Investor", text="Strong execution pattern"),
        )
        return PrecedentEpisode(
            episode_slug=slug,
            episode_number=number,
            source_path=f"sources/{slug}.json",
            source_sha256=sha256(slug.encode()).hexdigest(),
            investor_aliases=("Elizabeth",),
            investor_present=True,
            turns=turns,
            decision=PrecedentDecision(
                status=verdict,
                context="initial_panel",
                check_tier=None,
                conditions=(),
                evidence=(
                    DecisionEvidence(
                        turn_start=1,
                        turn_end=1,
                        text="Strong execution pattern",
                    ),
                ),
                audit_source="fixture",
                audit_notes="fixture",
            ),
        )

    return PrecedentCorpus.from_episodes(
        (
            episode("18-rowvigor", 18, "In"),
            episode("135-thoras-ai-the-twin-effect", 135, "Out"),
            episode("150-after-target", 150, "In"),
        ),
        embedder or TinyEmbedder(),
        target_slug="135-thoras-ai-the-twin-effect",
    )


def decision(investigation: dict) -> dict:
    return {
        "episode_slug": "135-thoras-ai-the-twin-effect",
        "investigation_sha256": sha256(
            canonical_bytes(Investigation.model_validate(investigation))
        ).hexdigest(),
        "decision": "In",
        "investment_likelihood": 0.7,
        "decision_confidence": 0.65,
        "ranking_score": 0.8,
        "check_tier": "small_exploratory",
        "decision_endpoint": "any_check",
        "recommended_check_tier": "small_exploratory",
        "controlling_rationale_ids": ["R1"],
        "any_check": {
            "decision": "In", "likelihood": 0.7, "confidence": 0.65,
            "supporting_rationale_ids": ["R1"], "opposing_rationale_ids": [],
            "fatal_constraint_present": False,
            "optionality_explanation": "Paid pilots justify a bounded check.",
            "strongest_counterargument": {
                "argument": "Production evidence is limited.",
                "rationale_ids": ["R1"],
                "response": "The limitation constrains conviction, not all participation.",
            },
            "reversal_conditions": ["Paid pilots cannot be verified."],
        },
        "standard_check": {
            "decision": "Out", "likelihood": 0.3, "confidence": 0.75,
            "supporting_rationale_ids": ["R1"], "opposing_rationale_ids": ["R1"],
            "failure_rationale_ids": ["R1"],
            "market_gate": {
                "status": "unresolved", "controlling_rationale_ids": ["R1"],
                "explanation": "Scale evidence remains incomplete.",
            },
            "strongest_counterargument": {
                "argument": "Paid pilots may justify a standard check.",
                "rationale_ids": ["R1"],
                "response": "They do not yet establish repeatability or scale.",
            },
            "upgrade_conditions": ["Show repeatable conversion and scale."],
        },
        "risk_ledger": [],
        "rationale_assessments": [{
            "rationale_id": "R1", "effective_direction": "positive",
            "decision_weight": "decisive", "assessment": "Paid pilots support proceeding.",
            "counterevidence": ["Production evidence is limited."],
        }],
        "deliberation_steps": [
            {
                "step_id": f"D{i}", "endpoint": endpoint, "stage": stage,
                "question": "Does this clear the bar?",
                "rationale_ids": ["R1"], "evidence_assessment": "Positive but early.",
                "likelihood_before": before, "likelihood_after": after,
                "effect": effect, "check_tier_implication": tier,
                "decision_update": "Move toward a small check.",
            }
            for i, endpoint, stage, before, after, effect, tier in [
                (1, "any_check", "assessment", 0.45, 0.55, "raises", "small_exploratory"),
                (2, "any_check", "opposing_case", 0.55, 0.62, "raises", "small_exploratory"),
                (3, "any_check", "consistency", 0.62, 0.7, "raises", "small_exploratory"),
                (4, "standard_check", "assessment", 0.5, 0.4, "lowers", "standard_initial"),
                (5, "standard_check", "opposing_case", 0.4, 0.35, "lowers", "standard_initial"),
                (6, "standard_check", "consistency", 0.35, 0.3, "lowers", "no_check_tier"),
            ]
        ],
        "strongest_counterargument": "Production evidence is limited.",
        "unresolved_questions": [],
        "reversal_conditions": [],
        "feedback": "Proceed to diligence.",
    }


def decision_v2(investigation: dict) -> dict:
    v1_investigation = {
        key: value
        for key, value in investigation.items()
        if key not in {"schema_version", "deal_context"}
    }
    v1_investigation["rationales"] = [
        {
            key: value
            for key, value in rationale.items()
            if key
            not in {
                "evidence_status",
                "constraint_kind",
                "constraint_severity",
                "severity_basis",
            }
        }
        for rationale in investigation["rationales"]
    ]
    candidate = decision(v1_investigation)
    candidate["schema_version"] = "decision-v2"
    candidate["investigation_sha256"] = sha256(
        canonical_bytes(InvestigationV2.model_validate(investigation))
    ).hexdigest()
    return candidate


def phase2_plan() -> dict:
    return {
        "decision_questions": ["What supports In?", "What supports Out?"],
        "wiki_queries": [],
        "precedent_queries": [],
        "transcript_reads": [],
        "decision_reads": [],
        "opposing_case_to_test": "Test the strongest opposing precedent case.",
    }


def as_decision_v3(candidate: dict, *, stable: bool = True) -> dict:
    candidate = dict(candidate)
    candidate.update({
        "schema_version": "decision-v3",
        "stable": stable,
        "decisive_precedents": [],
        "exception_analogies": [],
    })
    return candidate


def settings(tmp_path: Path) -> WorkflowSettings:
    return WorkflowSettings(
        episode_slug="135-thoras-ai-the-twin-effect",
        investor_name="Elizabeth Yin",
        pitch="The founders have five paid pilots.",
        taxonomy_labels={"founder_execution"},
        check_tiers={
            "no_check_tier",
            "small_exploratory",
            "standard_initial",
            "larger_conviction",
        },
        phase1_min_iterations=1,
        phase1_max_iterations=2,
        phase2_min_iterations=1,
        phase2_max_iterations=2,
        retrieval_top_k=2,
        max_exact_reads=4,
        run_root=tmp_path / "run",
    )


def test_phase1_iteration_configuration_is_not_silently_clamped(tmp_path: Path) -> None:
    configured = replace(settings(tmp_path), phase1_max_iterations=5)
    workflow = VCDecisionWorkflow(
        configured,
        make_index(tmp_path),
        FakeProvider(outputs=[]),
        InMemorySaver(),
    )

    assert workflow.settings.phase1_max_iterations == 5


def test_v3_requires_a_disposition_for_every_taxonomy_label(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    candidate = InvestigationV3.model_validate(
        phase1_v3(chunk_id, [], saturated=True)
    )

    with pytest.raises(ValueError, match="all taxonomy labels"):
        validate_investigation_v3(
            candidate,
            episode_slug="135-thoras-ai-the-twin-effect",
            pitch="The founders have five paid pilots.",
            taxonomy_labels={"founder_execution", "market_size_assessment"},
            exact_wiki_ids={chunk_id},
            exact_historical_ids=set(),
        )


def test_v3_schema_exposes_explicit_taxonomy_dispositions() -> None:
    schema = investigation_v3_json_schema(
        episode_slug="135-thoras-ai-the-twin-effect",
        taxonomy_labels={"founder_execution", "market_size_assessment"},
        exact_wiki_ids={"W-0123456789abcdef0123"},
        exact_historical_ids={"H-0123456789abcdef0123"},
    )

    assert "taxonomy_dispositions" in schema["properties"]


def test_phase2_v3_plans_reads_and_freezes_decisive_exact_precedent(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    corpus = make_precedent_corpus()
    chunk_id = index.search("founder", 1)[0].chunk_id
    investigation_payload = phase1_v3(chunk_id, [], saturated=True)
    investigation = InvestigationV3.model_validate(investigation_payload)
    digest = sha256(canonical_bytes(investigation)).hexdigest()
    opened = corpus.open_decision("18-rowvigor", query_id="fixture")
    historical_id = opened.evidence[0].evidence_id
    candidate = decision(phase1(chunk_id))
    candidate.update({
        "schema_version": "decision-v3",
        "investigation_sha256": digest,
        "stable": True,
        "decisive_precedents": [{
            "episode_slug": "18-rowvigor",
            "historical_evidence_ids": [historical_id],
            "observed_decision": "In",
            "comparison": "supports",
            "explanation": "The exact decision evidence supports a bounded check.",
        }],
        "exception_analogies": [],
    })
    plan = {
        "decision_questions": ["What supports In?", "What supports Out?"],
        "wiki_queries": [],
        "precedent_queries": ["execution pattern"],
        "transcript_reads": [],
        "decision_reads": ["18-rowvigor"],
        "opposing_case_to_test": "The market remains uncertain.",
    }
    provider = FakeProvider(outputs=[plan, candidate])
    configured = replace(
        settings(tmp_path), contract_version="v3",
        precedent_manifest_sha256=corpus.filtered_manifest().sha256,
        accessible_precedent_count=2,
        phase2_max_precedent_searches=1,
        phase2_max_precedent_reads=1,
    )
    result = VCDecisionWorkflow(
        configured, index, provider, InMemorySaver(), precedent_corpus=corpus
    ).invoke_phase2("phase2-v3", investigation, digest, phase1_status="accepted")
    assert result["phase2_status"] == "accepted"
    assert result["phase2_precedent_reads"][0]["episode_slug"] == "18-rowvigor"
    assert result["decision"]["decisive_precedents"][0]["historical_evidence_ids"] == [historical_id]
    assert [request.phase for request in provider.requests] == ["phase2_plan", "phase2"]


def test_v3_without_precedent_corpus_preserves_wiki_only_retrieval(
    tmp_path: Path,
) -> None:
    workflow = VCDecisionWorkflow(
        replace(settings(tmp_path), contract_version="v3"),
        make_index(tmp_path),
        FakeProvider(outputs=[]),
        InMemorySaver(),
    )

    result = workflow._retrieve(
        {
            "phase1_iteration": 0,
            "query_plan": {
                "wiki_queries": ["founder"],
                "precedent_queries": [],
                "transcript_reads": [],
                "decision_reads": [],
            },
        }
    )

    assert result["exact_reads"]
    assert result["phase1_precedent_searches"] == []
    assert result["phase1_precedent_reads"] == []


def test_phase1_searches_then_opens_exact_precedents_across_iterations(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    corpus = make_precedent_corpus()
    decision_read = corpus.open_decision("18-rowvigor", query_id="expected")
    historical_id = decision_read.evidence[0].evidence_id
    first = phase1_v3(chunk_id, [], saturated=False)
    final = phase1_v3(chunk_id, [historical_id], saturated=True)
    interim_v1 = phase1(chunk_id)
    phase2 = decision(interim_v1)
    phase2["schema_version"] = "decision-v2"
    phase2["investigation_sha256"] = sha256(
        canonical_bytes(InvestigationV3.model_validate(final))
    ).hexdigest()
    phase2 = as_decision_v3(phase2)
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder pattern?", "Exception pattern?"],
                "wiki_queries": ["founder"],
                "precedent_queries": ["execution pattern"],
                "transcript_reads": [],
                "decision_reads": [],
                "reconsideration_focus": "Find a relevant episode",
            },
            first,
            {
                "questions": ["Founder pattern?", "Decision analogy?"],
                "wiki_queries": ["founder"],
                "precedent_queries": [],
                "transcript_reads": [
                    {
                        "episode_slug": "150-after-target",
                        "turn_start": 0,
                        "turn_end": 0,
                    }
                ],
                "decision_reads": ["18-rowvigor"],
                "reconsideration_focus": "Open exact evidence",
            },
            final,
            phase2_plan(), phase2,
        ]
    )
    configured = replace(
        settings(tmp_path),
        contract_version="v3",
        phase1_max_iterations=2,
        phase1_max_precedent_searches=2,
        phase1_max_precedent_reads=2,
        precedent_manifest_sha256=corpus.filtered_manifest().sha256,
        accessible_precedent_count=2,
    )

    result = VCDecisionWorkflow(
        configured,
        index,
        provider,
        InMemorySaver(),
        precedent_corpus=corpus,
    ).invoke("precedent-two-turn")

    assert result["phase1_iteration"] == 2
    assert result["phase1_precedent_searches"][0]["query"] == "execution pattern"
    assert {row["episode_slug"] for row in result["phase1_precedent_reads"]} == {
        "18-rowvigor",
        "150-after-target",
    }
    assert historical_id in {
        row["evidence_id"] for row in result["phase1_historical_evidence"]
    }
    assert historical_id in result["investigation"]["rationales"][0][
        "historical_evidence_ids"
    ]
    assert "150-after-target" in provider.requests[0].prompt
    assert "135-thoras-ai-the-twin-effect founder signal" not in json.dumps(result)
    assert (tmp_path / "run/phase1/turn-01/precedent-searches.json").is_file()
    assert (tmp_path / "run/phase1/turn-02/precedent-reads.json").is_file()
    assert (tmp_path / "run/phase1/turn-02/historical-evidence.json").is_file()


def v3_workflow(
    tmp_path: Path,
    corpus: PrecedentCorpus,
    *,
    searches: int = 2,
    reads: int = 2,
    allow_full: bool = False,
) -> VCDecisionWorkflow:
    configured = replace(
        settings(tmp_path),
        contract_version="v3",
        precedent_manifest_sha256=corpus.filtered_manifest().sha256,
        accessible_precedent_count=len(corpus.list_episodes()),
        phase1_max_precedent_searches=searches,
        phase1_max_precedent_reads=reads,
        allow_full_transcript=allow_full,
    )
    return VCDecisionWorkflow(
        configured,
        make_index(tmp_path),
        FakeProvider(outputs=[]),
        InMemorySaver(),
        precedent_corpus=corpus,
    )


def test_phase1_planner_normalizes_full_transcript_range_without_fallback(
    tmp_path: Path,
) -> None:
    corpus = make_precedent_corpus()
    plan = {
        "questions": ["Founder pattern?", "Market pattern?"],
        "wiki_queries": ["founder execution"],
        "precedent_queries": ["consumer precedent"],
        "transcript_reads": [
            {
                "episode_slug": "18-rowvigor",
                "turn_start": 0,
                "turn_end": None,
            }
        ],
        "decision_reads": [],
        "reconsideration_focus": "Inspect the full precedent.",
    }
    provider = FakeProvider(outputs=[plan])
    configured = replace(
        settings(tmp_path),
        contract_version="v3",
        precedent_manifest_sha256=corpus.filtered_manifest().sha256,
        accessible_precedent_count=len(corpus.list_episodes()),
        phase1_max_precedent_reads=1,
        allow_full_transcript=True,
    )
    workflow = VCDecisionWorkflow(
        configured,
        make_index(tmp_path),
        provider,
        InMemorySaver(),
        precedent_corpus=corpus,
    )

    planned = workflow._plan_phase1({"phase1_iteration": 0})
    retrieved = workflow._retrieve({**planned, "phase1_iteration": 0})

    assert "PLANNER_FALLBACK_USED" not in planned["phase1_quality_findings"]
    assert planned["phase1_audit_findings"] == [
        "NORMALIZED_FULL_TRANSCRIPT_RANGE:18-rowvigor"
    ]
    assert planned["query_plan"]["transcript_reads"][0]["turn_start"] is None
    opened = retrieved["phase1_precedent_reads"][0]
    assert (opened["turn_start"], opened["turn_end"]) == (0, 1)


def test_phase2_planner_normalizes_full_transcript_range_without_schema_error(
    tmp_path: Path,
) -> None:
    corpus = make_precedent_corpus()
    plan = {
        "decision_questions": ["What supports In?", "What supports Out?"],
        "wiki_queries": ["consumer economics"],
        "precedent_queries": ["consumer precedent"],
        "transcript_reads": [
            {
                "episode_slug": "18-rowvigor",
                "turn_start": 0,
                "turn_end": None,
            }
        ],
        "decision_reads": [],
        "opposing_case_to_test": "The company may not scale.",
    }
    provider = FakeProvider(outputs=[plan])
    configured = replace(
        settings(tmp_path),
        contract_version="v3",
        precedent_manifest_sha256=corpus.filtered_manifest().sha256,
        accessible_precedent_count=len(corpus.list_episodes()),
        phase2_max_precedent_reads=1,
        allow_full_transcript=True,
    )
    workflow = VCDecisionWorkflow(
        configured,
        make_index(tmp_path),
        provider,
        InMemorySaver(),
        precedent_corpus=corpus,
    )
    investigation = phase1_v3(
        workflow.index.search("founder", 1)[0].chunk_id,
        [],
        saturated=True,
    )

    planned = workflow._plan_phase2(
        {
            "phase2_iteration": 0,
            "investigation": investigation,
            "investigation_sha256": "a" * 64,
        }
    )

    assert planned["phase2_current_retrieval_findings"] == []
    assert planned["phase2_audit_findings"] == [
        "NORMALIZED_FULL_TRANSCRIPT_RANGE:18-rowvigor"
    ]
    assert planned["phase2_plan"]["wiki_queries"] == ["consumer economics"]
    assert planned["phase2_plan"]["transcript_reads"][0]["turn_start"] is None


def test_phase2_normalized_plan_freezes_accepted_with_audit_warning(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    corpus = make_precedent_corpus()
    chunk_id = index.search("founder", 1)[0].chunk_id
    investigation = InvestigationV3.model_validate(
        phase1_v3(chunk_id, [], saturated=True)
    )
    digest = sha256(canonical_bytes(investigation)).hexdigest()
    candidate = as_decision_v3(decision(phase1(chunk_id)))
    candidate["investigation_sha256"] = digest
    plan = {
        "decision_questions": ["What supports In?", "What supports Out?"],
        "wiki_queries": [],
        "precedent_queries": [],
        "transcript_reads": [
            {
                "episode_slug": "18-rowvigor",
                "turn_start": 0,
                "turn_end": None,
            }
        ],
        "decision_reads": [],
        "opposing_case_to_test": "The market remains uncertain.",
    }
    provider = FakeProvider(outputs=[plan, candidate])
    configured = replace(
        settings(tmp_path),
        contract_version="v3",
        phase2_max_iterations=1,
        precedent_manifest_sha256=corpus.filtered_manifest().sha256,
        accessible_precedent_count=len(corpus.list_episodes()),
        phase2_max_precedent_reads=1,
        allow_full_transcript=True,
    )

    result = VCDecisionWorkflow(
        configured,
        index,
        provider,
        InMemorySaver(),
        precedent_corpus=corpus,
    ).invoke_phase2(
        "phase2-normalized-warning",
        investigation,
        digest,
        phase1_status="accepted",
    )

    assert result["phase2_status"] == "accepted_with_warnings"
    assert result["phase2_findings"] == [
        "NORMALIZED_FULL_TRANSCRIPT_RANGE:18-rowvigor"
    ]


def test_phase1_rejects_target_search_and_direct_reads_without_leaking_content(
    tmp_path: Path,
) -> None:
    corpus = make_precedent_corpus()
    workflow = v3_workflow(tmp_path, corpus)
    target = "135-thoras-ai-the-twin-effect"
    result = workflow._retrieve(
        {
            "phase1_iteration": 0,
            "query_plan": {
                "wiki_queries": ["founder"],
                "precedent_queries": [target],
                "transcript_reads": [
                    {"episode_slug": target, "turn_start": 0, "turn_end": 0}
                ],
                "decision_reads": [target],
            },
        }
    )

    assert result["phase1_retrieval_findings"] == [
        "TARGET_PRECEDENT_ACCESS_REJECTED"
    ]
    assert all(row["status"] == "rejected" for row in result["phase1_precedent_reads"])
    serialized = json.dumps(result)
    assert f"{target} founder signal" not in serialized
    assert not result["phase1_historical_evidence"]


def test_v3_records_contrastive_precedent_selection_policy(tmp_path: Path) -> None:
    corpus = make_precedent_corpus()
    workflow = v3_workflow(tmp_path, corpus)
    workflow.settings = replace(
        workflow.settings,
        precedent_selection_policy="contrastive",
        precedent_candidate_pool_k=10,
        precedent_in_slots=1,
        precedent_out_slots=1,
    )

    result = workflow._retrieve(
        {
            "phase1_iteration": 0,
            "query_plan": {
                "wiki_queries": [],
                "precedent_queries": ["execution pattern"],
                "transcript_reads": [],
                "decision_reads": [],
            },
        }
    )

    search = result["phase1_precedent_searches"][0]
    assert search["selection_policy"] == "contrastive"
    assert {hit["episode_slug"] for hit in search["hits"]} == {
        "18-rowvigor",
        "150-after-target",
    }


def test_contrastive_read_policy_limits_in_precedents_to_configured_ratio(
    tmp_path: Path,
) -> None:
    template = make_precedent_corpus().open_transcript("18-rowvigor")

    def precedent(slug: str, decision_status: str) -> PrecedentEpisode:
        turns = template.turns
        return PrecedentEpisode(
            episode_slug=slug,
            episode_number=int(slug.split("-", 1)[0]),
            source_path=f"sources/{slug}.json",
            source_sha256=sha256(slug.encode()).hexdigest(),
            investor_aliases=("Elizabeth",),
            investor_present=True,
            turns=turns,
            decision=PrecedentDecision(
                status=decision_status,
                context="initial_panel",
                check_tier=None,
                conditions=(),
                evidence=(
                    DecisionEvidence(
                        turn_start=1,
                        turn_end=1,
                        text="Strong execution pattern",
                    ),
                ),
                audit_source="fixture",
                audit_notes="fixture",
            ),
        )

    corpus = PrecedentCorpus.from_episodes(
        (
            precedent("18-first-in", "In"),
            precedent("20-second-in", "In"),
            precedent("22-first-out", "Out"),
            precedent("24-second-out", "Out"),
            precedent("26-third-out", "Out"),
        ),
        target_slug="135-thoras-ai-the-twin-effect",
    )
    configured = replace(
        settings(tmp_path),
        contract_version="v3",
        precedent_manifest_sha256=corpus.filtered_manifest().sha256,
        accessible_precedent_count=5,
        phase1_max_precedent_reads=12,
        precedent_selection_policy="contrastive",
        precedent_candidate_pool_k=10,
        precedent_in_slots=1,
        precedent_out_slots=3,
    )
    workflow = VCDecisionWorkflow(
        configured,
        make_index(tmp_path),
        FakeProvider(outputs=[]),
        InMemorySaver(),
        precedent_corpus=corpus,
    )

    result = workflow._retrieve(
        {
            "phase1_iteration": 0,
            "query_plan": {
                "wiki_queries": [],
                "precedent_queries": [],
                "transcript_reads": [],
                "decision_reads": [
                    "18-first-in",
                    "20-second-in",
                    "22-first-out",
                    "24-second-out",
                    "26-third-out",
                ],
            },
        }
    )

    successful = [
        row for row in result["phase1_precedent_reads"] if row["status"] == "ok"
    ]
    rejected = [
        row
        for row in result["phase1_precedent_reads"]
        if row.get("error_code") == "PRECEDENT_CONTRASTIVE_READ_RATIO"
    ]
    assert {row["episode_slug"] for row in successful} == {
        "18-first-in",
        "22-first-out",
        "24-second-out",
        "26-third-out",
    }
    assert [row["episode_slug"] for row in rejected] == ["20-second-in"]
    assert rejected[0]["charged"] is False


def test_phase1_precedent_read_budget_deduplicates_and_controls_full_reads(
    tmp_path: Path,
) -> None:
    corpus = make_precedent_corpus()
    workflow = v3_workflow(tmp_path, corpus, reads=1)
    result = workflow._retrieve(
        {
            "phase1_iteration": 0,
            "query_plan": {
                "wiki_queries": ["founder"],
                "precedent_queries": [],
                "transcript_reads": [
                    {
                        "episode_slug": "18-rowvigor",
                        "turn_start": None,
                        "turn_end": None,
                    }
                ],
                "decision_reads": ["18-rowvigor", "18-rowvigor", "150-after-target"],
            },
        }
    )

    assert [row["error_code"] for row in result["phase1_precedent_reads"] if row["status"] != "ok"] == [
        "FULL_TRANSCRIPT_READ_DISABLED",
        "PRECEDENT_READ_BUDGET_EXHAUSTED",
    ]
    assert sum(row["charged"] for row in result["phase1_precedent_reads"]) == 1
    assert len([row for row in result["phase1_precedent_reads"] if row["read_type"] == "decision" and row["episode_slug"] == "18-rowvigor"]) == 1


def test_phase1_precedent_search_budget_is_cumulative_across_iterations(
    tmp_path: Path,
) -> None:
    corpus = make_precedent_corpus()
    workflow = v3_workflow(tmp_path, corpus, searches=1)
    first = workflow._retrieve(
        {
            "phase1_iteration": 0,
            "query_plan": {
                "wiki_queries": ["founder"],
                "precedent_queries": ["execution pattern"],
                "transcript_reads": [],
                "decision_reads": [],
            },
        }
    )
    second = workflow._retrieve(
        {
            **first,
            "phase1_iteration": 1,
            "query_plan": {
                "wiki_queries": ["market"],
                "precedent_queries": ["different historical query"],
                "transcript_reads": [],
                "decision_reads": [],
            },
        }
    )

    assert len(second["phase1_precedent_searches"]) == 2
    assert sum(row["charged"] for row in second["phase1_precedent_searches"]) == 1
    assert second["phase1_precedent_searches"][-1]["error_code"] == (
        "PRECEDENT_SEARCH_BUDGET_EXHAUSTED"
    )
    assert second["phase1_current_retrieval_findings"] == [
        "PRECEDENT_SEARCH_BUDGET_EXHAUSTED"
    ]


def test_phase1_full_transcript_read_allowed_returns_exact_full_evidence(
    tmp_path: Path,
) -> None:
    corpus = make_precedent_corpus()
    workflow = v3_workflow(tmp_path, corpus, reads=1, allow_full=True)
    result = workflow._retrieve(
        {
            "phase1_iteration": 0,
            "query_plan": {
                "wiki_queries": ["founder"],
                "precedent_queries": [],
                "transcript_reads": [
                    {
                        "episode_slug": "18-rowvigor",
                        "turn_start": None,
                        "turn_end": None,
                    }
                ],
                "decision_reads": [],
            },
        }
    )

    opened = result["phase1_precedent_reads"][0]
    evidence = result["phase1_historical_evidence"][0]
    assert opened["status"] == "ok"
    assert (opened["turn_start"], opened["turn_end"]) == (0, 1)
    assert evidence["text"] == (
        "18-rowvigor founder signal\nStrong execution pattern"
    )
    assert (evidence["turn_start"], evidence["turn_end"]) == (0, 1)


def test_phase1_directly_opens_episode_outside_search_top_k(tmp_path: Path) -> None:
    corpus = make_precedent_corpus()
    top_hit = corpus.search("execution pattern", 1)[0].episode_slug
    outside = (
        {row.episode_slug for row in corpus.list_episodes()} - {top_hit}
    ).pop()
    configured = replace(
        settings(tmp_path),
        contract_version="v3",
        retrieval_top_k=1,
        precedent_manifest_sha256=corpus.filtered_manifest().sha256,
        accessible_precedent_count=2,
        phase1_max_precedent_searches=1,
        phase1_max_precedent_reads=1,
    )
    workflow = VCDecisionWorkflow(
        configured,
        make_index(tmp_path),
        FakeProvider(outputs=[]),
        InMemorySaver(),
        precedent_corpus=corpus,
    )
    result = workflow._retrieve(
        {
            "phase1_iteration": 0,
            "query_plan": {
                "wiki_queries": ["founder"],
                "precedent_queries": ["execution pattern"],
                "transcript_reads": [
                    {"episode_slug": outside, "turn_start": 0, "turn_end": 0}
                ],
                "decision_reads": [],
            },
        }
    )

    assert outside not in {
        hit["episode_slug"]
        for hit in result["phase1_precedent_searches"][0]["hits"]
    }
    assert result["phase1_precedent_reads"][0]["episode_slug"] == outside
    assert result["phase1_precedent_reads"][0]["status"] == "ok"


def test_phase1_precedent_embedding_failure_uses_lexical_results_and_propagates_finding(
    tmp_path: Path,
) -> None:
    class FailingEmbedder:
        def embed(self, texts):
            raise RuntimeError("offline embedding failure")

    corpus = make_precedent_corpus(FailingEmbedder())
    workflow = v3_workflow(tmp_path, corpus, searches=1)
    result = workflow._retrieve(
        {
            "phase1_iteration": 0,
            "query_plan": {
                "wiki_queries": ["founder"],
                "precedent_queries": ["execution pattern"],
                "transcript_reads": [],
                "decision_reads": [],
            },
        }
    )

    assert result["phase1_precedent_searches"][0]["hits"]
    assert "PRECEDENT_EMBEDDING_FALLBACK" in result["phase1_retrieval_findings"]
    assert "PRECEDENT_EMBEDDING_FALLBACK" in result[
        "phase1_current_retrieval_findings"
    ]


def test_resume_does_not_repeat_completed_v3_plan_or_precedent_search(
    tmp_path: Path, monkeypatch
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    corpus = make_precedent_corpus()
    investigation = phase1_v3(chunk_id, [], saturated=True)
    interim_phase2 = decision(phase1(chunk_id))
    interim_phase2["schema_version"] = "decision-v2"
    interim_phase2["investigation_sha256"] = sha256(
        canonical_bytes(InvestigationV3.model_validate(investigation))
    ).hexdigest()
    plan = {
        "questions": ["Founder pattern?", "Market pattern?"],
        "wiki_queries": ["founder"],
        "precedent_queries": ["execution pattern"],
        "transcript_reads": [],
        "decision_reads": [],
        "reconsideration_focus": "Initial",
    }

    class InterruptingProvider(FakeProvider):
        def __init__(self):
            super().__init__([plan, investigation, phase2_plan(), as_decision_v3(interim_phase2)])
            self.phase_calls = []
            self.interrupted = False

        def generate(self, request):
            self.phase_calls.append(request.phase)
            if request.phase == "phase1" and not self.interrupted:
                self.interrupted = True
                raise RuntimeError("simulated interruption")
            return super().generate(request)

    search_calls = 0
    original_search = PrecedentCorpus.search

    def counted_search(self, *args, **kwargs):
        nonlocal search_calls
        search_calls += 1
        return original_search(self, *args, **kwargs)

    monkeypatch.setattr(PrecedentCorpus, "search", counted_search)
    provider = InterruptingProvider()
    workflow = VCDecisionWorkflow(
        replace(
            settings(tmp_path),
            contract_version="v3",
            precedent_manifest_sha256=corpus.filtered_manifest().sha256,
            accessible_precedent_count=2,
            phase1_max_precedent_searches=1,
            phase1_max_precedent_reads=1,
        ),
        index,
        provider,
        InMemorySaver(),
        precedent_corpus=corpus,
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        workflow.invoke("resume-v3")
    result = workflow.resume("resume-v3")

    assert result["phase1_status"] == "accepted"
    assert provider.phase_calls.count("phase1_plan") == 1
    assert search_calls == 1


def test_transient_precedent_search_error_reconsiders_then_accepts(
    tmp_path: Path, monkeypatch
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    corpus = make_precedent_corpus()
    investigation = phase1_v3(chunk_id, [], saturated=True)
    phase2 = decision(phase1(chunk_id))
    phase2["schema_version"] = "decision-v2"
    phase2["investigation_sha256"] = sha256(
        canonical_bytes(InvestigationV3.model_validate(investigation))
    ).hexdigest()
    plan = {
        "questions": ["Founder pattern?", "Exception pattern?"],
        "wiki_queries": ["founder"],
        "precedent_queries": ["execution pattern"],
        "transcript_reads": [],
        "decision_reads": [],
        "reconsideration_focus": "Retry transient retrieval",
    }
    provider = FakeProvider(outputs=[plan, investigation, plan, investigation, phase2_plan(), as_decision_v3(phase2)])
    original_search = PrecedentCorpus.search
    attempts = 0

    def transient_search(self, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient retrieval failure")
        return original_search(self, *args, **kwargs)

    monkeypatch.setattr(PrecedentCorpus, "search", transient_search)
    workflow = VCDecisionWorkflow(
        replace(
            settings(tmp_path),
            contract_version="v3",
            phase1_max_iterations=2,
            precedent_manifest_sha256=corpus.filtered_manifest().sha256,
            accessible_precedent_count=2,
            phase1_max_precedent_searches=2,
            phase1_max_precedent_reads=0,
        ),
        index,
        provider,
        InMemorySaver(),
        precedent_corpus=corpus,
    )

    result = workflow.invoke("transient-search")

    assert attempts == 2
    assert result["phase1_status"] == "accepted"
    assert result["phase1_current_retrieval_findings"] == []
    assert "PRECEDENT_SEARCH_ERROR" in result["phase1_retrieval_findings"]


def test_persistent_precedent_search_error_freezes_provisional_at_cap(
    tmp_path: Path, monkeypatch
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    corpus = make_precedent_corpus()
    investigation = phase1_v3(chunk_id, [], saturated=True)
    phase2 = decision(phase1(chunk_id))
    phase2["schema_version"] = "decision-v2"
    phase2["investigation_sha256"] = sha256(
        canonical_bytes(InvestigationV3.model_validate(investigation))
    ).hexdigest()
    plan = {
        "questions": ["Founder pattern?", "Exception pattern?"],
        "wiki_queries": ["founder"],
        "precedent_queries": ["execution pattern"],
        "transcript_reads": [],
        "decision_reads": [],
        "reconsideration_focus": "Retry persistent retrieval",
    }
    provider = FakeProvider(outputs=[plan, investigation, plan, investigation, phase2_plan(), as_decision_v3(phase2)])

    def failed_search(self, *args, **kwargs):
        raise RuntimeError("persistent retrieval failure")

    monkeypatch.setattr(PrecedentCorpus, "search", failed_search)
    workflow = VCDecisionWorkflow(
        replace(
            settings(tmp_path),
            contract_version="v3",
            phase1_max_iterations=2,
            precedent_manifest_sha256=corpus.filtered_manifest().sha256,
            accessible_precedent_count=2,
            phase1_max_precedent_searches=2,
            phase1_max_precedent_reads=0,
        ),
        index,
        provider,
        InMemorySaver(),
        precedent_corpus=corpus,
    )

    result = workflow.invoke("persistent-search")

    assert result["phase1_status"] == "provisional"
    assert result["phase1_current_retrieval_findings"] == [
        "PRECEDENT_SEARCH_ERROR"
    ]


def test_graph_runs_two_frozen_phases_and_records_iterative_reads(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    investigation = phase1(chunk_id)
    final_decision = decision(investigation)
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder learning?", "Market scale?"],
                "search_queries": ["founder customer learning", "large market"],
                "reconsideration_focus": "Initial investigation",
            },
            investigation,
            final_decision,
        ]
    )
    workflow = VCDecisionWorkflow(settings(tmp_path), index, provider, InMemorySaver())

    result = workflow.invoke("thread-1")

    assert result["phase1_status"] == "accepted"
    assert result["phase2_status"] == "accepted"
    assert result["decision"]["decision"] == "In"
    assert len(result["exact_reads"]) >= 2
    assert (tmp_path / "run/phase1/investigation.json").is_file()
    assert (tmp_path / "run/phase2/decision.json").is_file()
    assert (tmp_path / "run/phase1/turn-01/plan-model-response.json").is_file()
    assert (
        tmp_path / "run/phase1/turn-01/investigation-model-response.json"
    ).is_file()
    assert (tmp_path / "run/phase2/turn-01/decision-model-response.json").is_file()
    assert provider.requests[0].max_output_tokens == 768
    retrieval = json.loads(
        (tmp_path / "run/phase1/turn-01/retrieval.json").read_text()
    )
    assert "portfolio holdings existing investments conflict overlap" in {
        row["query"] for row in retrieval["searches"]
    }


def test_graph_routes_phase_calls_to_distinct_providers_with_request_controls(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    investigation = phase1(chunk_id)
    phase1_provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder learning?"],
                "search_queries": ["founder"],
                "reconsideration_focus": "Initial",
            },
            investigation,
        ]
    )
    phase2_provider = FakeProvider(outputs=[decision(investigation)])
    configured = replace(
        settings(tmp_path),
        phase1_max_output_tokens=512,
        phase1_reasoning_effort="low",
        phase1_planning_max_output_tokens=1536,
        phase1_planning_reasoning_effort="medium",
        phase2_max_output_tokens=1024,
        phase2_reasoning_effort="high",
    )
    workflow = VCDecisionWorkflow(
        configured,
        index,
        checkpointer=InMemorySaver(),
        phase1_provider=phase1_provider,
        phase2_provider=phase2_provider,
    )

    result = workflow.invoke("distinct-providers")

    assert result["phase2_status"] == "accepted"
    assert [request.phase for request in phase1_provider.requests] == [
        "phase1_plan",
        "phase1",
    ]
    assert [request.phase for request in phase2_provider.requests] == ["phase2"]
    assert [request.max_output_tokens for request in phase1_provider.requests] == [
        1536,
        512,
    ]
    assert [request.reasoning_effort for request in phase1_provider.requests] == [
        "medium",
        "low",
    ]
    assert phase2_provider.requests[0].max_output_tokens == 1024
    assert phase2_provider.requests[0].reasoning_effort == "high"


def test_graph_constrains_run_specific_identifiers_in_generation_schemas(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    investigation = phase1(chunk_id)
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder learning?", "Market scale?"],
                "search_queries": ["founder", "market"],
                "reconsideration_focus": "Initial",
            },
            investigation,
            decision(investigation),
        ]
    )

    VCDecisionWorkflow(settings(tmp_path), index, provider, InMemorySaver()).invoke("t")

    phase1_schema = provider.requests[1].schema
    rationale_properties = phase1_schema["$defs"]["Rationale"]["properties"]
    assert phase1_schema["properties"]["episode_slug"]["const"] == settings(
        tmp_path
    ).episode_slug
    assert rationale_properties["label"]["enum"] == ["founder_execution"]
    assert chunk_id in rationale_properties["wiki_evidence_ids"]["items"]["enum"]

    phase2_schema = provider.requests[2].schema
    assert phase2_schema["properties"]["episode_slug"]["const"] == settings(
        tmp_path
    ).episode_slug
    assert phase2_schema["properties"]["investigation_sha256"]["const"]
    assert phase2_schema["properties"]["check_tier"]["enum"] == [
        "larger_conviction",
        "no_check_tier",
        "small_exploratory",
        "standard_initial",
    ]
    assert phase2_schema["properties"]["controlling_rationale_ids"]["items"][
        "enum"
    ] == ["R1"]
    assert phase2_schema["$defs"]["RationaleAssessment"]["properties"]["rationale_id"]["enum"] == ["R1"]
    assert phase2_schema["$defs"]["DeliberationStep"]["properties"]["rationale_ids"]["items"]["enum"] == ["R1"]
    assert phase2_schema["$defs"]["AnyCheckDecision"]["properties"]["supporting_rationale_ids"]["items"]["enum"] == ["R1"]
    assert phase2_schema["$defs"]["RiskLedgerEntry"]["properties"]["rationale_id"]["enum"] == ["R1"]


def test_v2_graph_parses_validates_and_freezes_v2_contracts(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    investigation = phase1_v2(chunk_id)
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder learning?", "Market scale?"],
                "search_queries": ["founder", "market"],
                "reconsideration_focus": "Initial",
            },
            investigation,
            decision_v2(investigation),
        ]
    )
    v2_settings = replace(settings(tmp_path), contract_version="v2")

    result = VCDecisionWorkflow(
        v2_settings, index, provider, InMemorySaver()
    ).invoke("v2")

    assert result["phase1_status"] == "accepted"
    assert result["phase2_status"] == "accepted"
    assert result["investigation"]["schema_version"] == "investigation-v2"
    assert result["decision"]["schema_version"] == "decision-v2"
    assert "RationaleV2" in provider.requests[1].schema["$defs"]
    assert provider.requests[2].schema["properties"]["schema_version"]["const"] == (
        "decision-v2"
    )


def test_v2_graph_preserves_and_notes_repairable_untyped_constraint(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    raw = phase1_v2(chunk_id)
    raw["rationales"][0].update(
        {
            "direction": "negative",
            "evidence_status": "affirmative_adverse",
            "constraint_kind": "none",
            "constraint_severity": "material",
        }
    )
    normalized = json.loads(json.dumps(raw))
    normalized["rationales"][0]["constraint_kind"] = "other"
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder learning?", "Market scale?"],
                "search_queries": ["founder", "market"],
                "reconsideration_focus": "Initial",
            },
            raw,
            decision_v2(normalized),
        ]
    )
    v2_settings = replace(settings(tmp_path), contract_version="v2")

    result = VCDecisionWorkflow(
        v2_settings, index, provider, InMemorySaver()
    ).invoke("v2-normalized")

    assert result["phase1_status"] == "provisional"
    assert "NORMALIZED_UNTYPED_CONSTRAINT:R1:none->other" in result["phase1_findings"]
    assert result["investigation"]["rationales"][0]["constraint_kind"] == "other"
    assert result["phase2_status"] == "accepted"


def test_v3_graph_applies_repairable_investigation_normalization(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    corpus = make_precedent_corpus()
    raw = phase1_v3(chunk_id, [], saturated=True)
    raw["rationales"][0]["severity_basis"] = ""
    normalized = json.loads(json.dumps(raw))
    normalized["rationales"][0]["severity_basis"] = "No constraint identified."
    phase2 = decision(phase1(chunk_id))
    phase2["schema_version"] = "decision-v2"
    phase2["investigation_sha256"] = sha256(
        canonical_bytes(InvestigationV3.model_validate(normalized))
    ).hexdigest()
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder pattern?"],
                "wiki_queries": ["founder"],
                "precedent_queries": [],
                "transcript_reads": [],
                "decision_reads": [],
                "reconsideration_focus": "Initial",
            },
            raw,
            phase2_plan(),
            as_decision_v3(phase2),
        ]
    )
    configured = replace(
        settings(tmp_path),
        contract_version="v3",
        phase1_max_iterations=1,
        precedent_manifest_sha256=corpus.filtered_manifest().sha256,
        accessible_precedent_count=2,
    )

    result = VCDecisionWorkflow(
        configured,
        index,
        provider,
        InMemorySaver(),
        precedent_corpus=corpus,
    ).invoke("v3-normalized")

    assert result["phase1_status"] == "provisional"
    assert result["investigation"]["rationales"][0]["severity_basis"] == (
        "No constraint identified."
    )
    assert "NORMALIZED_EMPTY_NO_CONSTRAINT_BASIS:R1" in result["phase1_findings"]


def test_v2_graph_flags_and_preserves_phase2_fatal_escalation_from_routine_phase1(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    investigation = phase1_v2(chunk_id, unresolved=True)
    escalated = decision_v2(investigation)
    escalated["any_check"]["fatal_constraint_present"] = True
    escalated["risk_ledger"] = [
        {
            "rationale_id": "R1",
            "risk_type": "fatal_constraint",
            "controlling_for_any_check": False,
            "counterevidence": [],
            "explanation": "Unsupported escalation.",
        }
    ]
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder learning?", "Market scale?"],
                "search_queries": ["founder", "market"],
                "reconsideration_focus": "Initial",
            },
            investigation,
            escalated,
        ]
    )
    v2_settings = replace(
        settings(tmp_path),
        contract_version="v2",
        phase2_max_iterations=1,
    )

    result = VCDecisionWorkflow(
        v2_settings, index, provider, InMemorySaver()
    ).invoke("v2-escalation")

    assert result["phase2_status"] == "provisional"
    assert result["decision"]["schema_version"] == "decision-v2"
    assert any(
        "fatal risk requires affirmative adverse fatal provenance" in finding
        for finding in result["phase2_findings"]
    )


def test_phase2_prompt_never_contains_raw_pitch(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder learning?", "Market scale?"],
                "search_queries": ["founder", "market"],
                "reconsideration_focus": "Initial",
            },
            (investigation := phase1(chunk_id)),
            decision(investigation),
        ]
    )
    VCDecisionWorkflow(settings(tmp_path), index, provider, InMemorySaver()).invoke("t")
    phase2_request = provider.requests[-1]
    assert phase2_request.phase == "phase2"
    assert "The founders have five paid pilots." not in phase2_request.prompt


def test_invalid_first_candidate_can_freeze_valid_provisional_at_cap(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder learning?", "Market scale?"],
                "search_queries": ["founder", "market"],
                "reconsideration_focus": "Initial",
            },
            phase1(chunk_id, label="not_in_taxonomy"),
            {
                "questions": ["Founder learning?", "Market scale?"],
                "search_queries": ["founder", "market"],
                "reconsideration_focus": "Repair taxonomy",
            },
            (final_investigation := phase1(chunk_id, saturated=False)),
            decision(final_investigation),
        ]
    )
    result = VCDecisionWorkflow(settings(tmp_path), index, provider, InMemorySaver()).invoke("t")
    assert result["phase1_status"] == "provisional"
    assert result["phase1_iteration"] == 2
    assert result["phase2_status"] == "accepted"


def test_valid_unsaturated_candidate_is_available_to_next_planning_pass(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    first = phase1(chunk_id, saturated=False)
    final = phase1(chunk_id, saturated=True)
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder learning?"],
                "search_queries": ["founder"],
                "reconsideration_focus": "Initial",
            },
            first,
            {
                "questions": ["What remains unresolved?"],
                "search_queries": ["market"],
                "reconsideration_focus": "Reconsider prior candidate",
            },
            final,
            decision(final),
        ]
    )

    result = VCDecisionWorkflow(settings(tmp_path), index, provider, InMemorySaver()).invoke(
        "t"
    )

    assert result["phase1_iteration"] == 2
    assert "Strong customer-learning signal." in provider.requests[2].prompt


def test_graph_aggregates_provider_reported_cost(tmp_path: Path) -> None:
    class CostedFakeProvider(FakeProvider):
        def generate(self, request):
            result = super().generate(request)
            return result.model_copy(
                update={
                    "usage": result.usage.model_copy(update={"cost_usd": 0.01})
                }
            )

    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    investigation = phase1(chunk_id)
    provider = CostedFakeProvider(
        outputs=[
            {
                "questions": ["Founder learning?"],
                "search_queries": ["founder"],
                "reconsideration_focus": "Initial",
            },
            investigation,
            decision(investigation),
        ]
    )

    result = VCDecisionWorkflow(settings(tmp_path), index, provider, InMemorySaver()).invoke(
        "cost-thread"
    )

    assert result["usage"]["cost_usd"] == pytest.approx(0.03)


def test_phase2_only_replay_uses_frozen_investigation_without_phase1_calls(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    frozen = phase1(chunk_id)
    provider = FakeProvider(outputs=[decision(frozen)])
    workflow = VCDecisionWorkflow(settings(tmp_path), None, provider, InMemorySaver())

    result = workflow.invoke_phase2(
        "phase2-only",
        Investigation.model_validate(frozen),
        sha256(canonical_bytes(Investigation.model_validate(frozen))).hexdigest(),
        phase1_status="accepted",
    )

    assert result["phase2_status"] == "accepted"
    assert [request.phase for request in provider.requests] == ["phase2"]
    assert (tmp_path / "run/phase2/decision.json").is_file()


def test_phase2_replay_carries_complete_phase1_state_and_usage(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    frozen = phase1(chunk_id)
    parsed = Investigation.model_validate(frozen)
    digest = sha256(canonical_bytes(parsed)).hexdigest()
    provider = FakeProvider(outputs=[decision(frozen)])
    phase1_usage = {
        "input_tokens": 101,
        "cached_input_tokens": 11,
        "output_tokens": 17,
        "cost_usd": 0.25,
    }
    phase1_state = {
        "phase1_iteration": 2,
        "phase1_action": "freeze_accepted",
        "query_plan": {"search_queries": ["founder"]},
        "query_history": ["founder"],
        "retrieval_passes": [{"turn": 1}],
        "exact_reads": [{"chunk_id": chunk_id}],
        "phase1_precedent_searches": [{"query": "founder precedent"}],
        "phase1_precedent_reads": [{"episode_slug": "18-rowvigor"}],
        "phase1_historical_evidence": [{"evidence_id": "H-fixture"}],
        "phase1_findings": ["fixture finding"],
        "phase1_quality_findings": ["fixture quality"],
        "phase1_retrieval_findings": ["fixture retrieval"],
        "phase1_current_retrieval_findings": [],
        "no_retrieval_novelty": False,
        "accessible_precedent_count": 2,
        "precedent_manifest_sha256": "a" * 64,
        "usage_by_phase": {"phase1": phase1_usage},
    }

    result = VCDecisionWorkflow(
        settings(tmp_path), None, provider, InMemorySaver()
    ).invoke_phase2(
        "phase2-carry",
        parsed,
        digest,
        phase1_status="accepted",
        phase1_state=phase1_state,
    )

    for key, expected in phase1_state.items():
        if key != "usage_by_phase":
            assert result[key] == expected
    assert result["usage_by_phase"]["phase1"] == phase1_usage
    assert result["usage"]["input_tokens"] > phase1_usage["input_tokens"]
    assert result["usage"]["cost_usd"] == pytest.approx(phase1_usage["cost_usd"])


def test_phase2_replay_clears_prior_decision_when_reusing_thread(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    frozen = phase1(chunk_id)
    parsed = Investigation.model_validate(frozen)
    digest = sha256(canonical_bytes(parsed)).hexdigest()
    replay_settings = replace(settings(tmp_path), phase2_max_iterations=1)
    provider = FakeProvider(outputs=[decision(frozen), {"invalid": "response"}])
    workflow = VCDecisionWorkflow(replay_settings, None, provider, InMemorySaver())

    first = workflow.invoke_phase2("same-thread", parsed, digest, phase1_status="accepted")
    second = workflow.invoke_phase2("same-thread", parsed, digest, phase1_status="accepted")

    assert first["decision"]["decision"] == "In"
    assert second["phase2_status"] == "failed"
    assert second.get("decision") == {}
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["phase2_status"] == "failed"
    assert summary["decision"] == {}


def test_invalid_planner_uses_recorded_fallback_and_freezes_valid_work(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    frozen = phase1(chunk_id)
    provider = FakeProvider(outputs=[{"invalid": True}, frozen, decision(frozen)])

    result = VCDecisionWorkflow(
        settings(tmp_path), index, provider, InMemorySaver()
    ).invoke("planner-fallback")

    assert "PLANNER_FALLBACK_USED" in result["phase1_findings"]
    assert result["phase1_status"] == "provisional"
    assert result["phase2_status"] == "accepted"


def test_repeated_queries_and_reads_stop_phase1_without_a_third_plan(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    first = phase1(chunk_id, saturated=False)
    second = phase1(chunk_id, saturated=False)
    repeated_plan = {
        "questions": ["Founder learning?", "Market scale?"],
        "search_queries": ["founder", "market"],
        "reconsideration_focus": "Repeat",
    }
    provider = FakeProvider(
        outputs=[repeated_plan, first, repeated_plan, second, decision(second)]
    )
    configured = replace(settings(tmp_path), phase1_max_iterations=5, retrieval_top_k=1)

    result = VCDecisionWorkflow(
        configured, index, provider, InMemorySaver()
    ).invoke("no-novelty")

    assert result["phase1_iteration"] == 2
    assert result["phase1_status"] == "provisional"
    assert "NO_RETRIEVAL_NOVELTY" in result["phase1_findings"]
    assert len([r for r in provider.requests if r.phase == "phase1_plan"]) == 2


@pytest.mark.parametrize("cap", [4, 5])
def test_novel_queries_honor_configured_phase1_cap(
    tmp_path: Path, cap: int,
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    first = phase1(chunk_id, saturated=False)
    plans = [
        {
            "questions": ["Founder?", "Market?"],
            "search_queries": queries,
            "reconsideration_focus": queries[0],
        }
        for queries in (
            ["founder", "customer learning"],
            ["market", "market scale"],
            ["pre-seed", "investment constraints"],
            ["ownership", "check sizing"],
            ["defensibility", "competitive advantage"],
        )
    ]
    outputs = [item for plan in plans[:cap] for item in (plan, first)]
    provider = FakeProvider(outputs=[*outputs, decision(first)])
    configured = replace(
        settings(tmp_path), phase1_max_iterations=cap, retrieval_top_k=1
    )

    result = VCDecisionWorkflow(
        configured, index, provider, InMemorySaver()
    ).invoke(f"{cap}-pass-cap")

    assert result["phase1_iteration"] == cap
    assert result["phase1_status"] == "provisional"
    assert len([r for r in provider.requests if r.phase == "phase1_plan"]) == cap


def test_phase2_reconsiders_material_in_then_preserves_it_as_provisional(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    frozen = phase1(chunk_id)
    material_in = decision(frozen)
    material_in["any_check"]["fatal_constraint_present"] = True
    material_in["any_check"]["opposing_rationale_ids"] = ["R1"]
    material_in["risk_ledger"] = [
        {
            "rationale_id": "R1",
            "risk_type": "fatal_constraint",
            "controlling_for_any_check": True,
            "counterevidence": [],
            "explanation": "A material portfolio conflict remains unresolved.",
        }
    ]
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder?", "Conflict?"],
                "search_queries": ["founder", "portfolio conflict"],
                "reconsideration_focus": "Initial",
            },
            frozen,
            material_in,
            material_in,
        ]
    )

    result = VCDecisionWorkflow(
        settings(tmp_path), index, provider, InMemorySaver()
    ).invoke("material-in")

    assert result["phase2_iteration"] == 2
    assert result["phase2_status"] == "provisional"
    assert result["decision"]["any_check"]["decision"] == "In"
    assert result["phase2_findings"] == [
        "ANY_CHECK_IN_WITH_MATERIAL_CONSTRAINT"
    ]


def test_mandatory_conflict_query_gets_exact_read_before_planner_budget(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    portfolio_id = index.search(
        "portfolio holdings existing investments conflict overlap", 1
    )[0].chunk_id
    frozen = phase1(portfolio_id)
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder?", "Market?"],
                "search_queries": [
                    "founder team execution",
                    "market scale",
                    "customer learning",
                    "unit economics",
                    "competition",
                    "defensibility",
                    "round terms",
                    "timing",
                ],
                "reconsideration_focus": "Initial",
            },
            frozen,
            decision(frozen),
        ]
    )
    configured = replace(settings(tmp_path), max_exact_reads=1)

    result = VCDecisionWorkflow(
        configured, index, provider, InMemorySaver()
    ).invoke("mandatory-first")

    assert result["exact_reads"][0]["chunk_id"] == portfolio_id


def test_phase1_preserves_prior_valid_candidate_after_invalid_repair(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    valid = phase1(chunk_id, saturated=False)
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder?", "Market?"],
                "search_queries": ["founder", "market"],
                "reconsideration_focus": "Initial",
            },
            valid,
            {
                "questions": ["Conflict?", "Scale?"],
                "search_queries": ["portfolio conflict", "venture scale"],
                "reconsideration_focus": "Repair",
            },
            {"invalid": True},
            decision(valid),
        ]
    )

    result = VCDecisionWorkflow(
        settings(tmp_path), index, provider, InMemorySaver()
    ).invoke("preserve-phase1")

    assert result["phase1_status"] == "provisional"
    assert result["investigation"]["summary"] == valid["summary"]
    assert "INVALID_RECONSIDERATION_PRESERVED_PRIOR_CANDIDATE" in result[
        "phase1_findings"
    ]


def test_phase2_preserves_prior_valid_decision_after_invalid_repair(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    frozen = phase1(chunk_id)
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder?", "Market?"],
                "search_queries": ["founder", "market"],
                "reconsideration_focus": "Initial",
            },
            frozen,
            decision(frozen),
            {"invalid": True},
        ]
    )
    configured = replace(
        settings(tmp_path), phase2_min_iterations=2, phase2_max_iterations=2
    )

    result = VCDecisionWorkflow(
        configured, index, provider, InMemorySaver()
    ).invoke("preserve-phase2")

    assert result["phase2_status"] == "provisional"
    assert result["decision"]["decision"] == "In"
    assert "INVALID_RECONSIDERATION_PRESERVED_PRIOR_CANDIDATE" in result[
        "phase2_findings"
    ]


def test_phase2_accepts_usable_final_draft_with_missing_consistency_as_warning(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    corpus = make_precedent_corpus()
    investigation = phase1_v3(chunk_id, [], saturated=True)
    draft = decision(phase1(chunk_id))
    draft["schema_version"] = "decision-v2"
    draft["investigation_sha256"] = sha256(
        canonical_bytes(InvestigationV3.model_validate(investigation))
    ).hexdigest()
    draft = as_decision_v3(draft)
    draft["deliberation_steps"][-1]["stage"] = "decision"
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder pattern?"],
                "wiki_queries": ["founder"],
                "precedent_queries": [],
                "transcript_reads": [],
                "decision_reads": [],
                "reconsideration_focus": "Initial",
            },
            investigation,
            phase2_plan(),
            draft,
        ]
    )
    configured = replace(
        settings(tmp_path),
        contract_version="v3",
        phase1_max_iterations=1,
        phase2_max_iterations=1,
        precedent_manifest_sha256=corpus.filtered_manifest().sha256,
        accessible_precedent_count=2,
    )

    result = VCDecisionWorkflow(
        configured,
        index,
        provider,
        InMemorySaver(),
        precedent_corpus=corpus,
    ).invoke("preserve-final-draft")

    assert result["phase2_status"] == "accepted_with_warnings"
    assert result["decision"]["decision"] == "In"
    assert "standard_check deliberation must finish with a consistency step" in result[
        "phase2_findings"
    ]
    assert "MINOR_AUDIT_OMISSION_ACCEPTED_WITH_WARNING" in result["phase2_findings"]
    assert "SEMANTICALLY_INVALID_FINAL_DRAFT_PRESERVED" not in result["phase2_findings"]


def test_phase2_preserves_prior_quality_findings_after_invalid_repair(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    frozen = phase1(chunk_id)
    material_in = decision(frozen)
    material_in["any_check"]["fatal_constraint_present"] = True
    material_in["risk_ledger"] = [{
        "rationale_id": "R1",
        "risk_type": "fatal_constraint",
        "controlling_for_any_check": True,
        "counterevidence": [],
        "explanation": "A material constraint remains unresolved.",
    }]
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder?", "Market?"],
                "search_queries": ["founder", "market"],
                "reconsideration_focus": "Initial",
            },
            frozen,
            material_in,
            {"invalid": True},
        ]
    )
    configured = replace(
        settings(tmp_path), phase2_min_iterations=1, phase2_max_iterations=2
    )

    result = VCDecisionWorkflow(
        configured, index, provider, InMemorySaver()
    ).invoke("preserve-phase2-quality")

    assert result["phase2_status"] == "provisional"
    assert result["decision"]["any_check"]["fatal_constraint_present"] is True
    assert "ANY_CHECK_IN_WITH_MATERIAL_CONSTRAINT" in result["phase2_findings"]
    assert "INVALID_RECONSIDERATION_PRESERVED_PRIOR_CANDIDATE" in result[
        "phase2_findings"
    ]


def test_resume_merges_fresh_phase1_quality_findings(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    chunk_id = index.search("founder", 1)[0].chunk_id
    frozen = phase1(chunk_id)
    provider = FakeProvider(
        outputs=[
            {
                "questions": ["Founder?", "Market?"],
                "search_queries": ["founder", "market"],
                "reconsideration_focus": "Initial",
            },
            frozen,
            decision(frozen),
        ]
    )
    saver = InMemorySaver()
    VCDecisionWorkflow(settings(tmp_path), index, provider, saver).invoke(
        "resume-quality"
    )
    resumed_settings = replace(
        settings(tmp_path),
        initial_phase1_quality_findings=("SANITIZED_EMBEDDING_FALLBACK",),
    )

    result = VCDecisionWorkflow(
        resumed_settings, index, provider, saver
    ).resume("resume-quality")

    assert result["phase1_status"] == "provisional"
    assert "SANITIZED_EMBEDDING_FALLBACK" in result["phase1_quality_findings"]
    assert "SANITIZED_EMBEDDING_FALLBACK" in result["phase1_findings"]
