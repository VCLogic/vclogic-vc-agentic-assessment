import json

import pytest
from pydantic import ValidationError

from vc_clone_graph.rehearsal_replay import (
    AnswerCompatibility,
    AnswerCompatibilityError,
    answer_compatibility_prompt,
    answer_is_usable,
    candidate_is_plausible,
    compatibility_usage,
    judge_answer_compatibility,
    rank_historical_candidates,
    rank_panel_founder_statements,
    PanelFounderStatement,
    select_compatible_answer,
)
from vc_clone_graph.rehearsal_canary import ObservedQuestion
from vc_clone_graph.providers.base import GenerationResult, Usage


def compatibility(coverage: str, **overrides):
    payload = {
        "coverage": coverage,
        "answered_clauses": ["app-only availability"],
        "unanswered_clauses": [],
        "contradiction": False,
        "confidence": 0.9,
        "explanation": "The answer directly addresses this clause.",
    }
    payload.update(overrides)
    return AnswerCompatibility.model_validate(payload)


def test_answer_compatibility_enforces_coverage_states() -> None:
    assert compatibility("full").coverage == "full"
    assert (
        compatibility(
            "material_partial",
            unanswered_clauses=["observed conversion"],
        ).coverage
        == "material_partial"
    )
    assert (
        compatibility(
            "none",
            answered_clauses=[],
            unanswered_clauses=["all requested evidence"],
        ).coverage
        == "none"
    )

    with pytest.raises(ValidationError):
        compatibility("full", unanswered_clauses=["retention"])
    with pytest.raises(ValidationError):
        compatibility("material_partial", unanswered_clauses=[])
    with pytest.raises(ValidationError):
        compatibility("none")


def test_answer_is_usable_requires_material_noncontradictory_coverage() -> None:
    assert answer_is_usable(compatibility("full"))
    assert answer_is_usable(
        compatibility("material_partial", unanswered_clauses=["retention"])
    )
    assert not answer_is_usable(
        compatibility("full", contradiction=True)
    )
    assert not answer_is_usable(
        compatibility(
            "none",
            answered_clauses=[],
            unanswered_clauses=["everything"],
        )
    )


def test_candidate_prefilter_accepts_similarity_or_rationale_overlap() -> None:
    assert candidate_is_plausible(semantic_similarity=0.45, rationale_f1=0.0)
    assert candidate_is_plausible(semantic_similarity=0.2, rationale_f1=0.5)
    assert not candidate_is_plausible(semantic_similarity=0.44, rationale_f1=0.0)


def test_compatibility_prompt_is_decision_blind_and_preserves_verbatim_data() -> None:
    prompt = answer_compatibility_prompt(
        generated_question="What is bundle conversion?",
        generated_labels=["business_model_assessment"],
        observed_question="Can I buy only the app?",
        observed_labels=["business_model_assessment"],
        founder_answer="We have an app-only model.",
    )

    assert "actual decision" in prompt.casefold()
    assert "must not" in prompt.casefold()
    payload = json.dumps(
        {
            "generated_question": "What is bundle conversion?",
            "generated_rationale_labels": ["business_model_assessment"],
            "observed_question": "Can I buy only the app?",
            "observed_rationale_labels": ["business_model_assessment"],
            "founder_answer_verbatim": "We have an app-only model.",
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    assert payload in prompt


class QueueProvider:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        payload = self.payloads.pop(0)
        return GenerationResult(
            parsed=payload,
            content=json.dumps(payload),
            usage=Usage(
                input_tokens=100,
                cached_input_tokens=10,
                output_tokens=20,
                cost_usd=0.001,
            ),
            elapsed_seconds=0.5,
            raw_metadata={"provider": "fake"},
        )


def test_judge_builds_strict_request_and_returns_valid_judgment() -> None:
    provider = QueueProvider(
        [
            {
                "coverage": "material_partial",
                "answered_clauses": ["app-only availability"],
                "unanswered_clauses": ["observed conversion"],
                "contradiction": False,
                "confidence": 0.92,
                "explanation": "One material clause is directly answered.",
            }
        ]
    )

    result = judge_answer_compatibility(
        provider,
        prompt="compatibility prompt",
        max_output_tokens=2048,
        reasoning_effort="high",
    )

    assert result.judgment.coverage == "material_partial"
    assert len(result.attempts) == 1
    request = provider.requests[0]
    assert request.phase == "answer_compatibility"
    assert request.max_output_tokens == 2048
    assert request.reasoning_effort == "high"
    assert request.schema["additionalProperties"] is False


def test_judge_repairs_one_invalid_structured_response() -> None:
    provider = QueueProvider(
        [
            {"coverage": "full"},
            {
                "coverage": "none",
                "answered_clauses": [],
                "unanswered_clauses": ["all requested clauses"],
                "contradiction": False,
                "confidence": 0.95,
                "explanation": "The answer is merely topical.",
            },
        ]
    )

    result = judge_answer_compatibility(
        provider,
        prompt="compatibility prompt",
        repair_attempts=1,
    )

    assert result.judgment.coverage == "none"
    assert [request.phase for request in provider.requests] == [
        "answer_compatibility",
        "answer_compatibility_repair",
    ]
    assert "validation" in provider.requests[1].prompt.casefold()
    assert compatibility_usage(result.attempts) == {
        "input_tokens": 200,
        "cached_input_tokens": 20,
        "output_tokens": 40,
        "cost_usd": 0.002,
        "elapsed_seconds": 1.0,
        "call_count": 2,
    }


def test_judge_raises_after_bounded_invalid_output() -> None:
    provider = QueueProvider([{"coverage": "full"}, {"coverage": "full"}])

    with pytest.raises(AnswerCompatibilityError):
        judge_answer_compatibility(
            provider,
            prompt="compatibility prompt",
            repair_attempts=1,
        )

    assert len(provider.requests) == 2


class KeywordEmbedder:
    def embed(self, texts):
        return self.embed_queries(texts)

    def embed_queries(self, texts):
        return [
            [
                0.01,
                float("app" in text.casefold()),
                float("retention" in text.casefold()),
            ]
            for text in texts
        ]


def observed(turn, text, *labels):
    return ObservedQuestion(
        turn_index=turn,
        text=text,
        rationale_labels=labels,
    )


def test_rank_historical_candidates_is_target_blind_and_deterministic() -> None:
    candidates = rank_historical_candidates(
        generated_question="Can customers buy only the app?",
        generated_labels={"business_model_assessment"},
        observed_questions=[
            observed(4, "What is retention?", "traction_repeatability_concern"),
            observed(8, "Can I buy the app without hardware?", "business_model_assessment"),
        ],
        founder_answers=["Ninety percent.", "We have an app-only model."],
        used_indices=set(),
        embedder=KeywordEmbedder(),
    )

    assert [candidate.observed_index for candidate in candidates] == [1]
    assert candidates[0].founder_answer == "We have an app-only model."
    assert candidates[0].rationale_f1 == 1.0


def test_rank_panel_statements_uses_predecision_founder_text_without_labels() -> None:
    candidates = rank_panel_founder_statements(
        generated_question="Can customers buy only the app?",
        statements=[
            PanelFounderStatement(turn_index=5, speaker="Founder", text="Retention is high."),
            PanelFounderStatement(
                turn_index=9,
                speaker="Founder",
                text="Customers can buy the app without hardware.",
            ),
        ],
        used_indices=set(),
        embedder=KeywordEmbedder(),
    )

    assert [candidate.observed_index for candidate in candidates] == [1]
    assert candidates[0].founder_answer == "Customers can buy the app without hardware."
    assert candidates[0].source_kind == "panel_founder_statement"


def test_select_compatible_answer_preserves_material_partial_answer_verbatim() -> None:
    candidates = rank_historical_candidates(
        generated_question="What is app-only conversion and retention?",
        generated_labels={"business_model_assessment"},
        observed_questions=[
            observed(8, "Can I buy the app without hardware?", "business_model_assessment")
        ],
        founder_answers=["  We have an app-only model.  "],
        used_indices=set(),
        embedder=KeywordEmbedder(),
    )
    provider = QueueProvider(
        [
            {
                "coverage": "material_partial",
                "answered_clauses": ["app-only availability"],
                "unanswered_clauses": ["conversion", "retention"],
                "contradiction": False,
                "confidence": 0.9,
                "explanation": "Availability is answered; performance is not.",
            }
        ]
    )

    selection = select_compatible_answer(
        provider,
        generated_question="What is app-only conversion and retention?",
        generated_labels=["business_model_assessment"],
        candidates=candidates,
    )

    assert selection.accepted
    assert selection.observed_index == 0
    assert selection.founder_answer == "  We have an app-only model.  "
    assert selection.unanswered_clauses == ("conversion", "retention")
    assert len(selection.judge_attempts) == 1


def test_select_compatible_answer_rejects_topical_candidate() -> None:
    candidates = rank_historical_candidates(
        generated_question="What is app-only conversion?",
        generated_labels={"business_model_assessment"},
        observed_questions=[
            observed(8, "Can I buy the app without hardware?", "business_model_assessment")
        ],
        founder_answers=["We launched a studio."],
        used_indices=set(),
        embedder=KeywordEmbedder(),
    )
    provider = QueueProvider(
        [
            {
                "coverage": "none",
                "answered_clauses": [],
                "unanswered_clauses": ["app-only conversion"],
                "contradiction": False,
                "confidence": 0.95,
                "explanation": "The answer does not address the question.",
            }
        ]
    )

    selection = select_compatible_answer(
        provider,
        generated_question="What is app-only conversion?",
        generated_labels=["business_model_assessment"],
        candidates=candidates,
    )

    assert not selection.accepted
    assert selection.founder_answer is None
    assert selection.observed_index is None
