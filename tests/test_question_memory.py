from hashlib import sha256

from vc_clone_graph.precedents import (
    DecisionEvidence,
    PrecedentDecision,
    PrecedentEpisode,
    TranscriptTurn,
)
from vc_clone_graph.question_memory import (
    QuestionMemory,
    atomic_question_findings,
    extract_question_archetypes,
    question_evidence_gap_key,
    question_subject_label,
    rationale_dimension,
    spoken_question_findings,
)
from vc_clone_graph.rehearsal_config import RehearsalQuestionMemorySettings


TAXONOMY = (
    {
        "label": "traction_repeatability_concern",
        "definition": "Whether retention and repeat usage are durable.",
        "coarse_parent": "traction",
    },
    {
        "label": "market_size_assessment",
        "definition": "Whether the market is large enough.",
        "coarse_parent": "market",
    },
)


class KeywordEmbedder:
    def embed(self, texts):
        return self.embed_documents(texts)

    def embed_documents(self, texts):
        return [
            [
                0.01,
                float("retention" in text.casefold()),
                float("market" in text.casefold()),
            ]
            for text in texts
        ]

    def embed_queries(self, texts):
        return self.embed_documents(texts)


def episode(slug="20-company"):
    episode_number = int(slug.split("-", 1)[0])
    turns = (
        TranscriptTurn(turn_index=0, speaker="Founder", text="We sell subscriptions."),
        TranscriptTurn(turn_index=1, speaker="Charles", text="What does retention look like?"),
        TranscriptTurn(turn_index=2, speaker="Founder", text="Seventy percent."),
        TranscriptTurn(turn_index=3, speaker="Charles", text="How large is this market?"),
        TranscriptTurn(turn_index=4, speaker="Founder", text="Ten billion dollars."),
        TranscriptTurn(turn_index=5, speaker="Charles", text="I am in. Any final questions?"),
    )
    return PrecedentEpisode(
        episode_slug=slug,
        episode_number=episode_number,
        source_path=f"sources/{slug}.json",
        source_sha256=sha256(slug.encode()).hexdigest(),
        investor_aliases=("Charles", "Charles Hudson"),
        investor_present=True,
        turns=turns,
        decision=PrecedentDecision(
            status="In",
            context="initial_panel",
            check_tier=None,
            conditions=(),
            evidence=(DecisionEvidence(turn_start=5, turn_end=5, text=turns[5].text),),
            audit_source="fixture",
            audit_notes="fixture",
        ),
    )


def test_extract_question_archetypes_excludes_decision_turns() -> None:
    rows = extract_question_archetypes(
        [episode()], investor_aliases={"Charles", "Charles Hudson"}
    )

    assert [row.text for row in rows] == [
        "What does retention look like?",
        "How large is this market?",
    ]
    assert rows[0].archetype_id.startswith("HQ-")
    assert rows[0].episode_slug == "20-company"
    assert "I am in" not in rows[0].context


def test_question_archetype_ids_are_stable_and_episode_bound() -> None:
    first = extract_question_archetypes([episode("20-company")], investor_aliases={"Charles"})
    repeated = extract_question_archetypes([episode("20-company")], investor_aliases={"Charles"})
    other = extract_question_archetypes([episode("21-company")], investor_aliases={"Charles"})

    assert first[0].archetype_id == repeated[0].archetype_id
    assert first[0].archetype_id != other[0].archetype_id


def test_question_memory_retrieves_semantically_relevant_archetypes() -> None:
    memory = QuestionMemory(
        extract_question_archetypes([episode()], investor_aliases={"Charles"}),
        KeywordEmbedder(),
        taxonomy=TAXONOMY,
    )

    results = memory.search("Customer retention is unresolved", top_k=1)

    assert len(results) == 1
    assert "retention" in results[0].text.casefold()
    assert 0 <= results[0].similarity <= 1
    assert results[0].semantic_similarity == results[0].similarity


def test_hybrid_search_prefers_atomic_rationale_aligned_question() -> None:
    rows = list(
        extract_question_archetypes([episode()], investor_aliases={"Charles"})
    )
    memory = QuestionMemory(
        rows,
        KeywordEmbedder(),
        taxonomy=TAXONOMY,
        settings=RehearsalQuestionMemorySettings(minimum_hybrid_score=0),
    )

    hits = memory.search(
        "Retention is unknown",
        rationale_labels=("traction_repeatability_concern",),
    )

    assert "retention" in hits[0].text.casefold()
    assert hits[0].score_components.rationale > 0
    assert hits[0].score_components.atomicity > 0


def test_hybrid_search_falls_back_to_semantics_without_labels() -> None:
    memory = QuestionMemory(
        extract_question_archetypes([episode()], investor_aliases={"Charles"}),
        KeywordEmbedder(),
        taxonomy=TAXONOMY,
        settings=RehearsalQuestionMemorySettings(minimum_hybrid_score=0),
    )

    hits = memory.search("market", rationale_labels=(), policy="semantic")

    assert "market" in hits[0].text.casefold()
    assert hits[0].semantic_similarity >= hits[1].semantic_similarity


def test_hybrid_search_penalizes_duplicate_prior_question() -> None:
    memory = QuestionMemory(
        extract_question_archetypes([episode()], investor_aliases={"Charles"}),
        KeywordEmbedder(),
        taxonomy=TAXONOMY,
        settings=RehearsalQuestionMemorySettings(minimum_hybrid_score=0),
    )

    hits = memory.search(
        "retention",
        prior_questions=("What does retention look like?",),
    )
    duplicate = next(row for row in hits if "retention" in row.text.casefold())

    assert duplicate.score_components.duplicate < 0


def test_empty_question_memory_returns_no_results() -> None:
    assert QuestionMemory((), KeywordEmbedder()).search("anything", top_k=5) == ()


def test_atomic_question_findings_detects_bundled_requests_without_failing() -> None:
    assert atomic_question_findings("What is monthly retention?") == ()
    findings = atomic_question_findings(
        "What is retention, and how much does each customer cost, and which channel works?"
    )
    assert "multiple_requests" in findings
    assert "too_long" not in findings


def test_atomic_question_findings_detects_investor_memory_request() -> None:
    findings = atomic_question_findings(
        "What distinguishes investable home-health operations software from "
        "the healthcare opportunities you exclude?"
    )

    assert "investor_memory_request" in findings
    assert atomic_question_findings(
        "What regulated clinical activities does your company perform?"
    ) == ()


def test_spoken_question_findings_flag_diligence_memo_language() -> None:
    findings = spoken_question_findings(
        "What independently verified ROI, if any, have your customer cohorts achieved "
        "relative to their pre-implementation replacement-coordination baseline?"
    )

    assert "memo_language:independently_verified" in findings
    assert "memo_language:customer_cohorts" in findings
    assert "memo_language:pre_implementation_baseline" in findings


def test_spoken_question_findings_accept_direct_founder_question() -> None:
    assert spoken_question_findings(
        "What measurable result are customers actually seeing today?"
    ) == ()


class AlignmentEmbedder:
    def embed(self, texts):
        return self.embed_documents(texts)

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_queries(self, texts):
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text):
        lowered = text.casefold()
        return [
            float(any(token in lowered for token in ("customer", "result", "retention"))),
            float(any(token in lowered for token in ("valuation", "stage", "price"))),
        ]


def test_question_memory_suggests_semantically_aligned_rationale_labels() -> None:
    taxonomy = (
        {
            "label": "traction_repeatability_concern",
            "definition": "Whether customer results and retention are repeatable.",
            "coarse_parent": "traction",
        },
        {
            "label": "stage_valuation_mismatch",
            "definition": "Whether valuation and price fit the investment stage.",
            "coarse_parent": "deal_terms",
        },
    )
    memory = QuestionMemory((), AlignmentEmbedder(), taxonomy=taxonomy)
    question = "What measurable result are customers actually seeing today?"

    suggestions = memory.suggest_rationale_labels(question, top_k=2)

    assert suggestions[0]["label"] == "traction_repeatability_concern"
    assert memory.rationale_alignment_findings(
        question=question,
        selected_labels=("stage_valuation_mismatch",),
    ) == (
        "rationale_alignment:stage_valuation_mismatch->traction_repeatability_concern",
    )
    assert memory.rationale_alignment_findings(
        question=question,
        selected_labels=("traction_repeatability_concern",),
    ) == ()


def test_shiftpilot_questions_map_to_their_actual_subjects() -> None:
    allowed = {
        "competitive_advantage_assessment",
        "fund_economics_constraint",
        "go_to_market_strategy_assessment",
        "market_problem_validation",
        "stage_valuation_mismatch",
        "traction_repeatability_concern",
        "unit_economics_assessment",
    }

    assert question_subject_label(
        "What independently verified ROI have customers achieved against their baseline?",
        allowed,
    ) == "unit_economics_assessment"
    assert question_subject_label(
        "How many agencies has outbound converted without adding headcount?",
        allowed,
    ) == "traction_repeatability_concern"
    assert question_subject_label(
        "Which workflow becomes harder to displace as more decisions are processed?",
        allowed,
    ) == "competitive_advantage_assessment"


def test_outbound_questions_share_one_evidence_gap() -> None:
    first = question_evidence_gap_key(
        "How many agencies has outbound converted without adding headcount?"
    )
    second = question_evidence_gap_key(
        "When you repeated the same outbound approach, what conversion pattern did you see?"
    )

    assert first == second == "acquisition_repeatability"
    assert question_evidence_gap_key(
        "What independently verified ROI have customers achieved against their baseline?"
    ) == "customer_value_proof"


def test_question_memory_replaces_structurally_incompatible_labels() -> None:
    taxonomy = (
        {
            "label": "traction_repeatability_concern",
            "definition": "Whether customer acquisition and traction repeat.",
            "coarse_parent": "traction_growth",
        },
        {
            "label": "fund_economics_constraint",
            "definition": "Whether fund check size and ownership constraints fit.",
            "coarse_parent": "investor_fit_constraints",
        },
    )
    memory = QuestionMemory((), AlignmentEmbedder(), taxonomy=taxonomy)

    labels = memory.resolve_rationale_labels(
        "How many agencies has outbound converted without adding headcount?",
        selected_labels=("fund_economics_constraint",),
    )

    assert labels == ("traction_repeatability_concern",)
    assert rationale_dimension(labels, taxonomy) == "Traction Growth"
