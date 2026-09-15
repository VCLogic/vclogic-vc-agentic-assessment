from collections import deque
from pathlib import Path
from time import perf_counter

from langgraph.checkpoint.memory import InMemorySaver
import pytest

from vc_clone_graph.providers.base import GenerationResult, Usage
from vc_clone_graph.rehearsal_artifacts import (
    PendingAnswerConflict,
    RehearsalArtifactStore,
)
from vc_clone_graph.rehearsal_classifier import LinearClassifierArtifact
from vc_clone_graph.rehearsal_graph import RehearsalEvidenceBundle, RehearsalWorkflow
from vc_clone_graph.rehearsal_grounding import load_canonical_baseline


TAXONOMY = (
    {
        "label": "traction_customer_validation",
        "definition": "Evidence of customer adoption and validation.",
        "coarse_parent": "traction",
    },
    {
        "label": "investment_thesis_constraint",
        "definition": "Compatibility with the investor's stated investment scope.",
        "coarse_parent": "investor_fit",
    },
)


def rationale(*, evidence_id: str = "P-001") -> dict:
    return {
        "rationale_id": "R-001",
        "taxonomy_label": "traction_customer_validation",
        "direction": "positive",
        "salience": "primary",
        "confidence": 0.7,
        "assessment": "Early customers support validation.",
        "evidence_refs": [
            {
                "evidence_id": evidence_id,
                "source_kind": "founder_answer" if evidence_id.startswith("A-") else "pitch",
                "source_path": None,
                "excerpt": "Evidence excerpt.",
            }
        ],
    }


def initial() -> dict:
    return {
        "decision": "Out",
        "investment_likelihood": 0.4,
        "decision_confidence": 0.6,
        "rationales": [rationale()],
        "unresolved_questions": ["Retention is unknown."],
        "candidate_questions": [],
    }


def question(number: int) -> dict:
    return {
        "question_id": f"Q-{number:03d}",
        "text": f"What is retention measure {number}?",
        "dimension": "traction",
        "rationale_labels": ["traction_customer_validation"],
        "expected_decision_value": 0.8,
        "why_now": "Retention remains decision-relevant.",
    }


def update(number: int) -> dict:
    return {
        "answer_id": f"A-{number:03d}",
        "affected_rationale_labels": ["traction_customer_validation"],
        "direction": "positive",
        "confidence": 0.8,
        "resolved_question": number == 8,
        "materially_changed_assessment": True,
        "investment_likelihood_before": 0.4,
        "investment_likelihood_after": 0.65,
        "decision_confidence_after": 0.7,
        "rationale_state_after": [rationale(evidence_id=f"A-{number:03d}")],
        "unresolved_questions_after": ([] if number == 8 else ["More retention detail is useful."]),
        "update_summary": "The founder claim raises confidence.",
    }


def continuation(keep_asking: bool) -> dict:
    return {
        "continue_interview": keep_asking,
        "reason": "Another material detail remains." if keep_asking else "Information is sufficient.",
        "next_question_focus": "retention depth" if keep_asking else None,
    }


def final(*, evidence_id: str = "A-001") -> dict:
    return {
        "decision": "In",
        "investment_likelihood": 0.68,
        "decision_confidence": 0.7,
        "strongest_positive_rationale_ids": ["R-001"],
        "decisive_concern_rationale_ids": [],
        "unresolved_uncertainties": [],
        "reversal_conditions": ["Retention is not reproducible."],
        "rationale_state": [rationale(evidence_id=evidence_id)],
        "decision_justification": "The founder claim supports a simulated pitch-stage In.",
    }


def coaching() -> dict:
    return {
        "material_answer_ids": ["A-001"],
        "pitch_improvement_suggestions": ["Put retention in the opening pitch."],
        "founder_reflection_prompts": ["Can the retention claim be evidenced?"],
    }


class ScriptedProvider:
    model = "scripted"

    def __init__(self, rows: list[dict | None]) -> None:
        self.rows = deque(rows)
        self.requests = []

    def generate(self, request):
        started = perf_counter()
        self.requests.append(request)
        parsed = self.rows.popleft()
        return GenerationResult(
            parsed=parsed,
            content="{}" if parsed is not None else "not-json",
            usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.001),
            elapsed_seconds=perf_counter() - started,
            raw_metadata={"provider": "scripted", "model": self.model},
        )


class RecordingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.question_requests: list[dict] = []

    def retrieve(self, query: str) -> RehearsalEvidenceBundle:
        self.queries.append(query)
        return RehearsalEvidenceBundle(
            wiki_evidence=(
                {
                    "evidence_id": f"W-{len(self.queries):03d}",
                    "source_kind": "wiki",
                    "source_path": "principles.md",
                    "excerpt": "Retention and execution matter.",
                },
            ),
            precedent_evidence=(),
            portfolio_evidence=(),
        )

    def retrieve_question_archetypes(
        self,
        query: str,
        *,
        rationale_labels=(),
        prior_questions=(),
    ):
        self.question_requests.append(
            {
                "query": query,
                "rationale_labels": tuple(rationale_labels),
                "prior_questions": tuple(prior_questions),
            }
        )
        return (
            {
                "archetype_id": "HQ-1234567890abcdef",
                "episode_slug": "20-example",
                "turn_index": 4,
                "text": "What does retention look like?",
                "context": "A subscription business.",
                "similarity": 0.8,
            },
        )


def workflow(
    tmp_path: Path,
    rows: list[dict | None],
    *,
    max_questions: int = 8,
    progress_callback=None,
    retriever=None,
):
    store = RehearsalArtifactStore.create(
        tmp_path, "charles-hudson-precursor-ventures", "session-1", "Five paid pilots."
    )
    provider = ScriptedProvider(rows)
    retriever = retriever or RecordingRetriever()
    options = {"progress_callback": progress_callback} if progress_callback else {}
    result = RehearsalWorkflow(
        provider=provider,
        checkpointer=InMemorySaver(),
        store=store,
        investor_name="Charles Hudson",
        taxonomy=TAXONOMY,
        retriever=retriever,
        max_questions=max_questions,
        **options,
    )
    return result, provider, retriever


class CorrectingRetriever(RecordingRetriever):
    def resolve_question_rationale_labels(self, question, *, selected_labels):
        if "outbound converted" in question.casefold():
            return ("traction_customer_validation",)
        return tuple(selected_labels)

    def question_rationale_dimension(self, labels):
        return "Traction Growth"

    def question_evidence_gap_key(self, question, *, rationale_labels=()):
        if "outbound" in question.casefold():
            return "acquisition_repeatability"
        return "rationale:" + "+".join(rationale_labels)


def test_answer_emits_actual_public_workflow_stages(tmp_path: Path) -> None:
    events: list[tuple[str, dict]] = []
    graph, _, _ = workflow(
        tmp_path,
        [initial(), question(1), update(1), continuation(True), question(2)],
        progress_callback=lambda stage, payload: events.append((stage, payload)),
    )
    graph.start(
        "session-1",
        pitch="Five paid pilots.",
        vc_slug="charles-hudson-precursor-ventures",
    )
    events.clear()

    graph.answer("session-1", "Seventy percent renewed after six months.")

    assert [stage for stage, _ in events] == [
        "answer_update_running",
        "continuation_running",
        "question_selection_running",
    ]
    assert all("prompt" not in payload for _, payload in events)
    submission = graph.store.read_json("pending-answer-submission.json")
    accepted = graph.store.read_json("turns/turn-01/answer.json")
    assert submission["status"] == "processed"
    assert submission["answer_id"] == "A-001"
    assert accepted["text_verbatim"] == "Seventy percent renewed after six months."
    assert accepted["submission_sha256"] == submission["submission_sha256"]


def test_different_pending_answer_stops_before_provider_update(
    tmp_path: Path,
) -> None:
    graph, provider, _ = workflow(
        tmp_path,
        [initial(), question(1), update(1)],
    )
    graph.start(
        "session-1",
        pitch="Five paid pilots.",
        vc_slug="charles-hudson-precursor-ventures",
    )
    graph.store.accept_answer_submission("Q-001", "First founder answer")

    with pytest.raises(PendingAnswerConflict):
        graph.answer("session-1", "Stale duplicated answer")

    assert "answer_update" not in [request.phase for request in provider.requests]


def classifier() -> LinearClassifierArtifact:
    return LinearClassifierArtifact.model_validate(
        {
            "schema": "rehearsal-linear-classifier-v1",
            "artifact_id": "charles-held-test-v1",
            "model_version": "test-v1",
            "vc_slug": "charles-hudson-precursor-ventures",
            "training_context": "leave_one_episode_out",
            "excluded_episode_slug": "999-test",
            "feature_names": [
                "rationale__traction_customer_validation__signed_confidence",
                "investment_likelihood",
            ],
            "scales": [1.0, 1.0],
            "coefficients": [2.0, 0.5],
            "intercept": -1.0,
            "decision_threshold": 0.5,
            "training_episode_slugs": ["18-rowvigor", "20-harper-wilde"],
        }
    )


def informed_workflow(tmp_path: Path, rows: list[dict | None]):
    store = RehearsalArtifactStore.create(
        tmp_path, "charles-hudson-precursor-ventures", "session-1", "Five paid pilots."
    )
    provider = ScriptedProvider(rows)
    graph = RehearsalWorkflow(
        provider=provider,
        checkpointer=InMemorySaver(),
        store=store,
        investor_name="Charles Hudson",
        taxonomy=TAXONOMY,
        retriever=RecordingRetriever(),
        max_questions=4,
        classification_mode="classification_informed",
        classifier_artifact=classifier(),
        minimum_questions=1,
        minimum_probability_impact=0.05,
    )
    return graph, provider


def grounded_workflow(
    tmp_path: Path,
    extra_rows=(),
    *,
    include_collateral=False,
    pitch: str = "Canonical historical pitch.",
    first_question: dict | None = None,
):
    workspace = Path(__file__).resolve().parents[1]
    if not (workspace / "outputs/canonical-v4-v41-portfolio-2026-08-15/investors/charles-hudson/39-this-pitch-is-damn-near-perfect" / "summary.json").is_file():
        pytest.skip("requires archived canonical assessment outputs")
    baseline = load_canonical_baseline(
        workspace=workspace,
        registry_path=workspace
        / "evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json",
        canonical_vc_slug="charles-hudson",
        episode_slug="39-this-pitch-is-damn-near-perfect",
    )
    labels = tuple(dict.fromkeys(row.taxonomy_label for row in baseline.investigation.rationales))
    taxonomy = tuple(
        {"label": label, "definition": label.replace("_", " "), "coarse_parent": "test"}
        for label in labels
    )
    if include_collateral:
        taxonomy = (*taxonomy, {
            "label": "investment_thesis_constraint",
            "definition": "Compatibility with the investor's stated investment scope.",
            "coarse_parent": "investor_fit",
        })
    first_label = baseline.investigation.rationales[0].taxonomy_label
    provider = ScriptedProvider(
        [first_question or {
            "question_id": "Q-001",
            "text": "What is repeat purchase after six months?",
            "dimension": "traction",
            "rationale_labels": [first_label],
            "expected_decision_value": 0.8,
            "why_now": "The canonical decision records this as unresolved.",
        }, *extra_rows]
    )
    store = RehearsalArtifactStore.create(
        tmp_path,
        "charles-hudson-precursor-ventures",
        "grounded-session",
        pitch,
    )
    graph = RehearsalWorkflow(
        provider=provider,
        checkpointer=InMemorySaver(),
        store=store,
        investor_name="Charles Hudson",
        taxonomy=taxonomy,
        retriever=RecordingRetriever(),
        max_questions=4,
        classification_mode="v41_grounded",
        grounded_baseline=baseline,
        phase2_synthesizer=object(),
    )
    return graph, provider


def test_grounded_start_uses_canonical_assessment_without_regeneration(
    tmp_path: Path,
) -> None:
    graph, provider = grounded_workflow(tmp_path)

    result = graph.start(
        "grounded-session",
        pitch="Canonical historical pitch.",
        vc_slug="charles-hudson-precursor-ventures",
    )

    assert result["status"] == "awaiting_answer"
    state = graph.store.read_json("state.json")
    assert state["status"] == "awaiting_answer"
    assert state["initial_assessment"]["investment_likelihood"] == 0.63
    assert "initial_assessment" not in [request.phase for request in provider.requests]
    assert graph.store.read_json("grounding/baseline.json")["contract_version"] == "v4"


def test_question_already_answered_by_pitch_is_repaired_before_interrupt(
    tmp_path: Path,
) -> None:
    redundant = {
        "question_id": "Q-001",
        "text": "Have customers agreed to pay for an expansion module?",
        "dimension": "Product adoption",
        "rationale_labels": ["product_adoption"],
        "expected_decision_value": 0.8,
        "why_now": "Expansion adoption could change the assessment.",
    }
    replacement = {
        **redundant,
        "text": "What implementation work can your team complete without a founder?",
        "dimension": "Founder execution",
        "rationale_labels": ["founder_execution"],
    }
    pitch = "Expansion-module willingness to pay has not been validated."
    graph, provider = grounded_workflow(
        tmp_path,
        extra_rows=(replacement,),
        pitch=pitch,
        first_question=redundant,
    )
    store = graph.store

    result = graph.start(
        "grounded-session",
        pitch=pitch,
        vc_slug="charles-hudson-precursor-ventures",
    )

    assert result["question"]["question"] == replacement["text"]
    coverage = store.read_json("turns/turn-01/question-coverage.json")
    assert coverage["original"]["redundant"] is True
    assert coverage["original"]["supporting_evidence_ids"] == ["P-001"]
    assert coverage["repair_attempted"] is True
    assert coverage["final"]["redundant"] is False


def test_grounded_empty_change_update_preserves_checkpointed_assessment(
    tmp_path: Path,
) -> None:
    no_change_update = {
        "answer_id": "A-001",
        "affected_rationale_labels": [],
        "direction": "neutral",
        "confidence": 0.9,
        "resolved_question": False,
        "materially_changed_assessment": False,
        "investment_likelihood_before": 0.63,
        "investment_likelihood_after": 0.18,
        "decision_confidence_after": 0.2,
        "rationale_state_after": [],
        "unresolved_questions_after": ["Repeat purchase remains unknown."],
        "update_summary": "The founder confirmed that evidence is unavailable.",
        "grounded_changes": [],
        "evidence_effect": "unresolved",
        "supporting_answer_excerpt": None,
    }
    next_question = question(2)
    graph, _ = grounded_workflow(
        tmp_path,
        extra_rows=(no_change_update, continuation(True), next_question),
    )
    next_question["rationale_labels"] = [
        graph.grounded_baseline.investigation.rationales[1].taxonomy_label
    ]
    graph.start(
        "grounded-session",
        pitch="Canonical historical pitch.",
        vc_slug="charles-hudson-precursor-ventures",
    )

    graph.answer("grounded-session", "We have not measured repeat purchase yet.")

    state = graph.store.read_json("state.json")
    assert state["current_likelihood"] == 0.63
    assert state["current_confidence"] == 0.72
    update = graph.store.read_json("turns/turn-01/update.json")
    assert update["investment_likelihood_after"] == 0.63
    normalization = graph.store.read_json(
        "turns/turn-01/update-normalization.json"
    )
    assert normalization["findings"] == [
        "no_grounded_changes_preserved_assessment"
    ]


def test_grounded_clarification_cannot_lower_likelihood_even_with_a_change(
    tmp_path: Path,
) -> None:
    graph, provider = grounded_workflow(tmp_path)
    graph.start(
        "grounded-session",
        pitch="Canonical historical pitch.",
        vc_slug="charles-hudson-precursor-ventures",
    )
    state = graph.store.read_json("state.json")
    current = state["current_rationales"][0]
    changed = {
        **current,
        "assessment": "The founder clarified an uncertainty already in the pitch.",
        "evidence_refs": [
            {
                "evidence_id": "A-001",
                "source_kind": "founder_answer",
                "source_path": "turns/turn-01/answer.json",
                "excerpt": "We still have not measured repeat purchase.",
            }
        ],
    }
    provider.rows.extend(
        [
            {
                "answer_id": "A-001",
                "affected_rationale_labels": [current["taxonomy_label"]],
                "direction": "neutral",
                "confidence": 0.8,
                "resolved_question": False,
                "materially_changed_assessment": True,
                "investment_likelihood_before": 0.63,
                "investment_likelihood_after": 0.16,
                "decision_confidence_after": 0.8,
                "rationale_state_after": [changed],
                "unresolved_questions_after": ["Repeat purchase remains unknown."],
                "update_summary": "The founder clarified that proof is unavailable.",
                "evidence_effect": "clarification",
                "supporting_answer_excerpt": None,
                "grounded_changes": [
                    {
                        "rationale_id": current["rationale_id"],
                        "taxonomy_label": current["taxonomy_label"],
                        "change": "redirected",
                        "before_direction": current["direction"],
                        "after_direction": current["direction"],
                        "answer_ids": ["A-001"],
                        "justification": "The answer clarified the existing uncertainty.",
                    }
                ],
            },
            continuation(True),
            {
                "question_id": "Q-002",
                "text": "What would establish repeat purchase?",
                "dimension": "traction",
                "rationale_labels": [current["taxonomy_label"]],
                "expected_decision_value": 0.5,
                "why_now": "The evidence remains unresolved.",
            },
        ]
    )

    graph.answer(
        "grounded-session", "We still have not measured repeat purchase."
    )

    updated = graph.store.read_json("state.json")
    assert updated["current_likelihood"] == 0.63
    normalization = graph.store.read_json(
        "turns/turn-01/update-normalization.json"
    )
    assert "non_material_effect_preserved_assessment" in normalization["findings"]


def test_grounded_answer_can_activate_a_collateral_taxonomy_rationale(
    tmp_path: Path,
) -> None:
    graph, provider = grounded_workflow(tmp_path, include_collateral=True)
    graph.start(
        "grounded-session",
        pitch="Canonical historical pitch.",
        vc_slug="charles-hudson-precursor-ventures",
    )
    state = graph.store.read_json("state.json")
    first_label = state["current_rationales"][0]["taxonomy_label"]
    assert "investment_thesis_constraint" not in {
        row["taxonomy_label"] for row in state["current_rationales"]
    }
    added = {
        "rationale_id": "AR-A-001-investment_thesis_constraint",
        "taxonomy_label": "investment_thesis_constraint",
        "direction": "positive",
        "salience": "secondary",
        "confidence": 0.74,
        "assessment": "The founder supplied new evidence of investor-scope compatibility.",
        "evidence_refs": [
            {
                "evidence_id": "A-001",
                "source_kind": "founder_answer",
                "source_path": "turns/turn-01/answer.json",
                "excerpt": "Our regulated workflow is sold as B2B software.",
            }
        ],
    }
    provider.rows.extend(
        [
            {
                "answer_id": "A-001",
                "affected_rationale_labels": ["investment_thesis_constraint"],
                "direction": "positive",
                "confidence": 0.74,
                "resolved_question": False,
                "materially_changed_assessment": True,
                "investment_likelihood_before": 0.63,
                "investment_likelihood_after": 0.68,
                "decision_confidence_after": 0.75,
                "rationale_state_after": [added],
                "unresolved_questions_after": ["Retention is still unknown."],
                "update_summary": "The target remains unresolved, but the answer adds investor-fit evidence.",
                "evidence_effect": "new_positive",
                "supporting_answer_excerpt": "Our regulated workflow is sold as B2B software.",
                "grounded_changes": [
                    {
                        "rationale_id": "AR-A-001-investment_thesis_constraint",
                        "taxonomy_label": "investment_thesis_constraint",
                        "change": "added",
                        "before_direction": None,
                        "after_direction": "positive",
                        "answer_ids": ["A-001"],
                        "justification": "The answer activates a separate investor-fit rationale.",
                    }
                ],
            },
            continuation(True),
            {
                "question_id": "Q-002",
                "text": "What retention have the customers shown?",
                "dimension": "traction",
                "rationale_labels": [first_label],
                "expected_decision_value": 0.8,
                "why_now": "The original target remains unresolved.",
            },
        ]
    )

    graph.answer(
        "grounded-session", "Our regulated workflow is sold as B2B software."
    )

    updated = graph.store.read_json("state.json")
    assert updated["current_likelihood"] == 0.68
    assert updated["current_confidence"] == 0.75
    assert "investment_thesis_constraint" in {
        row["taxonomy_label"] for row in updated["current_rationales"]
    }


def test_informed_graph_scores_privately_and_guides_question(tmp_path: Path) -> None:
    graph, provider = informed_workflow(tmp_path, [initial(), question(1)])

    result = graph.start(
        "session-1",
        pitch="Five paid pilots.",
        vc_slug="charles-hudson-precursor-ventures",
    )

    assert result["status"] == "awaiting_answer"
    snapshot = graph.store.read_json("classification/initial.json")
    assert snapshot["stage"] == "initial"
    assert 0 <= snapshot["probability_in"] <= 1
    select_prompt = next(
        request.prompt for request in provider.requests if request.phase == "select_question"
    )
    assert "classifier_question_guidance" in select_prompt
    assert "probability_in" not in select_prompt
    priority = graph.store.read_json("turns/turn-01/classifier-priority.json")
    assert priority["rationale_label"] == "traction_customer_validation"


def test_informed_report_preserves_classifier_trajectory(tmp_path: Path) -> None:
    graph, _ = informed_workflow(
        tmp_path,
        [initial(), question(1), update(1), continuation(False), final(), coaching()],
    )
    graph.start(
        "session-1",
        pitch="Five paid pilots.",
        vc_slug="charles-hudson-precursor-ventures",
    )

    result = graph.answer("session-1", "Seventy percent at month three.")

    summary = result["founder_report"]["classification"]
    assert summary["status"] == "available"
    assert summary["artifact_id"] == "charles-held-test-v1"
    assert [row["stage"] for row in summary["snapshots"]] == [
        "initial",
        "updated",
        "final",
    ]
    assert result["final_assessment"]["decision"] == "In"


def test_informed_mode_without_artifact_falls_back_without_failure(tmp_path: Path) -> None:
    store = RehearsalArtifactStore.create(
        tmp_path, "charles-hudson-precursor-ventures", "session-1", "Five paid pilots."
    )
    graph = RehearsalWorkflow(
        provider=ScriptedProvider([initial(), question(1)]),
        checkpointer=InMemorySaver(),
        store=store,
        investor_name="Charles Hudson",
        taxonomy=TAXONOMY,
        retriever=RecordingRetriever(),
        classification_mode="classification_informed",
        classifier_fallback_reason="held-episode artifact unavailable",
    )

    result = graph.start(
        "session-1",
        pitch="Five paid pilots.",
        vc_slug="charles-hudson-precursor-ventures",
    )

    assert result["status"] == "awaiting_answer"
    fallback = graph.store.read_json("classification/fallback.json")
    assert fallback["status"] == "fallback"


def test_founder_finish_does_not_duplicate_the_updated_classifier_snapshot(
    tmp_path: Path,
) -> None:
    graph, _ = informed_workflow(
        tmp_path,
        [
            initial(),
            question(1),
            update(1),
            continuation(True),
            question(2),
            final(evidence_id="A-001"),
            coaching(),
        ],
    )
    graph.start(
        "session-1",
        pitch="Five paid pilots.",
        vc_slug="charles-hudson-precursor-ventures",
    )

    graph.answer("session-1", "Seventy percent retained.")
    result = graph.finish("session-1")

    assert [
        row["stage"]
        for row in result["founder_report"]["classification"]["snapshots"]
    ] == ["initial", "updated", "final"]


def test_graph_interrupts_then_resumes_with_founder_answer(tmp_path: Path) -> None:
    graph, _, retriever = workflow(
        tmp_path, [initial(), question(1), update(1), continuation(False), final(), coaching()]
    )

    first = graph.start(
        "session-1",
        pitch="Five paid pilots.",
        vc_slug="charles-hudson-precursor-ventures",
    )
    resumed = graph.answer("session-1", "Seventy percent at month three.")

    assert first["status"] == "awaiting_answer"
    assert first["question"] == {
        "question_id": "Q-001",
        "question": "What is retention measure 1?",
        "dimension": "traction",
    }
    assert resumed["status"] == "complete"
    assert resumed["answers"][0]["text_verbatim"] == "Seventy percent at month three."
    assert resumed["final_assessment"]["score_reconciliation"] == {
        "before": 0.65,
        "after": 0.68,
        "direction": "increased",
        "explanation": (
            "The final synthesis increased the likelihood by 3 percentage points "
            "after balancing all answer-linked rationale changes."
        ),
    }
    assert len(retriever.queries) >= 2
    assert retriever.question_requests[0]["rationale_labels"] == (
        "traction_customer_validation",
    )
    assert retriever.question_requests[0]["prior_questions"] == ()


def test_final_decision_removes_precedent_only_exact_check_amount(
    tmp_path: Path,
) -> None:
    exact = final()
    exact["decision_justification"] = (
        "I would commit approximately $50,000 because a precedent did so."
    )
    graph, _, _ = workflow(
        tmp_path,
        [initial(), question(1), update(1), continuation(False), exact, coaching()],
    )
    graph.start(
        "session-1",
        pitch="Five paid pilots.",
        vc_slug="charles-hudson-precursor-ventures",
    )

    result = graph.answer("session-1", "Seventy percent retained.")

    justification = result["final_assessment"]["decision_justification"]
    assert "$50,000" not in justification
    assert "a small first check" in justification
    normalization = graph.store.read_json("final-assessment-normalization.json")
    assert "unsupported_exact_check_amount_removed" in normalization["findings"]


def test_compound_question_gets_one_bounded_repair_and_keeps_running(
    tmp_path: Path,
) -> None:
    compound = question(1)
    compound["text"] = "What is revenue, and what is retention?"
    repaired = question(1)
    repaired["text"] = "What is retention?"
    graph, provider, _ = workflow(tmp_path, [initial(), compound, repaired])

    result = graph.start(
        "session-1",
        pitch="Five paid pilots.",
        vc_slug="charles-hudson-precursor-ventures",
    )

    assert result["status"] == "awaiting_answer"
    assert result["question"]["question"] == "What is retention?"
    assert [request.phase for request in provider.requests] == [
        "initial_assessment",
        "select_question",
        "atomic_question_repair",
    ]
    quality = graph.store.read_json("turns/turn-01/question-quality.json")
    assert quality["original_findings"] == ["multiple_requests"]
    assert quality["final_findings"] == []
    archetypes = graph.store.read_json("turns/turn-01/question-archetypes.json")
    assert archetypes["archetypes"][0]["archetype_id"] == "HQ-1234567890abcdef"


def test_investor_memory_question_is_repaired_into_founder_evidence_request(
    tmp_path: Path,
) -> None:
    invalid = question(1)
    invalid.update({
        "text": (
            "What distinguishes investable home-health operations software from "
            "the healthcare opportunities you exclude?"
        ),
        "dimension": "investment thesis fit",
        "rationale_labels": ["investment_thesis_constraint"],
    })
    repaired = dict(invalid)
    repaired["text"] = (
        "What regulated clinical activities does your company perform beyond "
        "providing workflow software?"
    )
    graph, provider, _ = workflow(tmp_path, [initial(), invalid, repaired])

    result = graph.start(
        "session-1",
        pitch="Five paid pilots.",
        vc_slug="charles-hudson-precursor-ventures",
    )

    assert result["question"]["question"] == repaired["text"]
    assert [request.phase for request in provider.requests] == [
        "initial_assessment",
        "select_question",
        "atomic_question_repair",
    ]
    quality = graph.store.read_json("turns/turn-01/question-quality.json")
    assert quality["original_findings"] == ["investor_memory_request"]
    assert quality["final_findings"] == []
    assert quality["final_question"]["rationale_labels"] == [
        "investment_thesis_constraint"
    ]


def test_diligence_memo_question_is_repaired_into_spoken_investor_voice(
    tmp_path: Path,
) -> None:
    invalid = question(1)
    invalid["text"] = (
        "What independently verified ROI, if any, have your customer cohorts achieved "
        "relative to their pre-implementation replacement-coordination baseline?"
    )
    repaired = dict(invalid)
    repaired["text"] = "What measurable result are customers actually seeing today?"
    graph, provider, _ = workflow(tmp_path, [initial(), invalid, repaired])

    result = graph.start(
        "session-1",
        pitch="Five paid pilots.",
        vc_slug="charles-hudson-precursor-ventures",
    )

    assert result["question"]["question"] == repaired["text"]
    assert [request.phase for request in provider.requests] == [
        "initial_assessment",
        "select_question",
        "atomic_question_repair",
    ]
    quality = graph.store.read_json("turns/turn-01/question-quality.json")
    assert quality["original_findings"] == [
        "memo_language:independently_verified",
        "memo_language:customer_cohorts",
        "memo_language:pre_implementation_baseline",
    ]
    assert quality["final_findings"] == []


def test_generated_question_is_relabelled_from_its_actual_subject(
    tmp_path: Path,
) -> None:
    mislabeled = question(1)
    mislabeled.update(
        {
            "text": "How many agencies has outbound converted without adding headcount?",
            "dimension": "Fund economics constraint",
            "rationale_labels": ["investment_thesis_constraint"],
        }
    )
    graph, _, _ = workflow(
        tmp_path,
        [initial(), mislabeled],
        retriever=CorrectingRetriever(),
    )

    result = graph.start(
        "session-1",
        pitch="Five paid pilots.",
        vc_slug="charles-hudson-precursor-ventures",
    )

    assert result["question"]["dimension"] == "Traction Growth"
    quality = graph.store.read_json("turns/turn-01/question-quality.json")
    assert quality["final_question"]["rationale_labels"] == [
        "traction_customer_validation"
    ]
    assert quality["evidence_gap_key"] == "acquisition_repeatability"
    assert quality["label_resolution"] == {
        "proposed": ["investment_thesis_constraint"],
        "final": ["traction_customer_validation"],
    }


def test_reused_question_id_is_normalized_without_stopping(tmp_path: Path) -> None:
    reused = question(1)
    reused["text"] = "What is retention after six months?"
    graph, _, _ = workflow(
        tmp_path,
        [initial(), question(1), update(1), continuation(True), reused],
    )
    graph.start(
        "session-1",
        pitch="Five paid pilots.",
        vc_slug="charles-hudson-precursor-ventures",
    )

    result = graph.answer("session-1", "Seventy percent at month three.")

    assert result["status"] == "awaiting_answer"
    assert result["question"]["question_id"] == "Q-002"


def test_founder_cannot_finish_before_answering_one_question(tmp_path: Path) -> None:
    graph, _, _ = workflow(
        tmp_path, [initial(), question(1)]
    )
    graph.start("session-1", pitch="Five paid pilots.", vc_slug="charles-hudson-precursor-ventures")

    with pytest.raises(ValueError, match="at least one accepted answer"):
        graph.finish("session-1")


def test_founder_finish_after_one_answer_forces_final_decision(
    tmp_path: Path,
) -> None:
    graph, _, _ = workflow(
        tmp_path,
        [
            initial(),
            question(1),
            update(1),
            continuation(True),
            question(2),
            final(evidence_id="A-001"),
            coaching(),
        ],
    )
    graph.start("session-1", pitch="Five paid pilots.", vc_slug="charles-hudson-precursor-ventures")
    graph.answer("session-1", "Seventy percent retained.")

    result = graph.finish("session-1")

    assert result["status"] == "complete"
    assert len(result["answers"]) == 1
    assert result["final_assessment"]["decision"] in {"In", "Out"}
    assert result["stopping_reason"] == "founder_requested"


def test_graph_stops_after_eight_answers(tmp_path: Path) -> None:
    rows: list[dict | None] = [initial()]
    for number in range(1, 9):
        rows.extend([question(number), update(number)])
        if number < 8:
            rows.append(continuation(True))
    rows.extend([final(), coaching()])
    graph, _, _ = workflow(tmp_path, rows)
    graph.start("session-1", pitch="Five paid pilots.", vc_slug="charles-hudson-precursor-ventures")

    result = None
    for number in range(1, 9):
        result = graph.answer("session-1", f"Answer {number}")

    assert result is not None
    assert result["status"] == "complete"
    assert len(result["answers"]) == 8
    assert result["stopping_reason"] == "question_cap"


def test_invalid_output_allows_one_repair_and_preserves_turns(tmp_path: Path) -> None:
    graph, provider, _ = workflow(tmp_path, [initial(), question(1), None, None])
    graph.start("session-1", pitch="Five paid pilots.", vc_slug="charles-hudson-precursor-ventures")

    result = graph.answer("session-1", "Founder answer")

    assert result["status"] == "resumable_error"
    assert result["answers"] == []
    assert result["current_question"]["question_id"] == "Q-001"
    assert [request.phase for request in provider.requests][-2:] == [
        "answer_update",
        "answer_update_repair",
    ]


def test_retry_continues_failed_step_without_repeating_founder_answer(
    tmp_path: Path,
) -> None:
    graph, _, _ = workflow(
        tmp_path,
        [
            initial(),
            question(1),
            None,
            None,
            update(1),
            continuation(False),
            final(),
            coaching(),
        ],
    )
    graph.start(
        "session-1",
        pitch="Five paid pilots.",
        vc_slug="charles-hudson-precursor-ventures",
    )
    failed = graph.answer("session-1", "Founder answer")

    retried = graph.retry("session-1")

    assert failed["status"] == "resumable_error"
    assert retried["status"] == "complete"
    assert [row["text_verbatim"] for row in retried["answers"]] == ["Founder answer"]
