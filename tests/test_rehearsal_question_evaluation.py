from vc_clone_graph.rehearsal_canary import ObservedQuestion
from vc_clone_graph.rehearsal_question_evaluation import (
    aggregate_episode,
    score_generated_question,
    write_evaluation_outputs,
)


class KeywordEmbedder:
    def embed_queries(self, texts):
        return [
            [
                0.01,
                float("retention" in text.casefold() or "churn" in text.casefold()),
                float("market" in text.casefold()),
            ]
            for text in texts
        ]

    def embed_documents(self, texts):
        return self.embed_queries(texts)

    def embed(self, texts):
        return self.embed_queries(texts)


def test_score_generated_question_selects_nearest_observed() -> None:
    result = score_generated_question(
        question_id="Q-001",
        generated_text="What is retention?",
        generated_labels=("traction_repeatability_concern",),
        observed=(
            ObservedQuestion(
                turn_index=3,
                text="What is churn?",
                rationale_labels=("traction_repeatability_concern",),
            ),
            ObservedQuestion(
                turn_index=7,
                text="How large is the market?",
                rationale_labels=("market_size_assessment",),
            ),
        ),
        embedder=KeywordEmbedder(),
        prior_generated=(),
        taxonomy_parents={
            "traction_repeatability_concern": "traction",
            "market_size_assessment": "market",
        },
    )

    assert result.nearest_observed_turn == 3
    assert result.rationale_f1 == 1.0
    assert result.coarse_dimension_agreement is True
    assert result.atomic


def test_score_generated_question_handles_no_observed_questions() -> None:
    result = score_generated_question(
        question_id="Q-001",
        generated_text="What is retention?",
        generated_labels=(),
        observed=(),
        embedder=KeywordEmbedder(),
        prior_generated=(),
        taxonomy_parents={},
    )

    assert result.nearest_observed_turn is None
    assert result.semantic_similarity is None
    assert result.rationale_f1 is None


def test_episode_metrics_separate_fidelity_and_utility() -> None:
    observed = (
        ObservedQuestion(
            turn_index=3,
            text="What is churn?",
            rationale_labels=("traction_repeatability_concern",),
        ),
    )
    first = score_generated_question(
        question_id="Q-001",
        generated_text="What is retention?",
        generated_labels=("traction_repeatability_concern",),
        observed=observed,
        embedder=KeywordEmbedder(),
        prior_generated=(),
        taxonomy_parents={"traction_repeatability_concern": "traction"},
    )
    second = score_generated_question(
        question_id="Q-002",
        generated_text="What is retention?",
        generated_labels=("traction_repeatability_concern",),
        observed=observed,
        embedder=KeywordEmbedder(),
        prior_generated=("What is retention?",),
        taxonomy_parents={"traction_repeatability_concern": "traction"},
    )

    result = aggregate_episode(
        episode_slug="20-company",
        records=(first, second),
        observed_questions=observed,
    )

    assert result.behavioral_fidelity.mean_semantic_similarity == 1.0
    assert result.behavioral_fidelity.observed_rationale_coverage == 1.0
    assert result.question_utility.atomic_rate == 1.0
    assert result.question_utility.nonredundant_rate == 0.5


def test_writer_creates_all_local_evaluation_outputs(tmp_path) -> None:
    observed = (
        ObservedQuestion(
            turn_index=3,
            text="What is churn?",
            rationale_labels=("traction_repeatability_concern",),
        ),
    )
    record = score_generated_question(
        question_id="Q-001",
        generated_text="What is retention?",
        generated_labels=("traction_repeatability_concern",),
        observed=observed,
        embedder=KeywordEmbedder(),
        prior_generated=(),
        taxonomy_parents={"traction_repeatability_concern": "traction"},
    )
    evaluation = aggregate_episode(
        episode_slug="20-company",
        records=(record,),
        observed_questions=observed,
    )

    write_evaluation_outputs(
        evaluations=(evaluation,), ablations=(), output_root=tmp_path
    )

    assert {path.name for path in tmp_path.iterdir()} == {
        "question-records.jsonl",
        "episode-summary.csv",
        "aggregate.json",
        "report.md",
        "retrieval-ablation.json",
    }
    assert '"generation_provider_calls": 0' in (tmp_path / "aggregate.json").read_text()
