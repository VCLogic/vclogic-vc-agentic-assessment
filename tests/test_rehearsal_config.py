from pathlib import Path

import pytest

from vc_clone_graph.rehearsal_config import (
    RehearsalClassificationSettings,
    RehearsalQuestionMemorySettings,
    load_rehearsal_config,
)


def _write(tmp_path: Path) -> Path:
    path = tmp_path / "rehearsal.toml"
    path.write_text(
        '''[rehearsal]
input_root = "inputs"
output_root = "outputs/rehearsals"
checkpoint_path = "outputs/rehearsals/checkpoints.sqlite"
taxonomy_path = "taxonomy/codebook_v_final.json"
max_questions = 8

[provider]
kind = "openrouter"
model = "openai/gpt-5.6-luna"
api_key_env = "OPENROUTER_API_KEY"
max_output_tokens = 8192

[embedding]
kind = "sentence_transformers"
model = "nomic-ai/nomic-embed-text-v1.5"
revision = "e9b6763023c676ca8431644204f50c2b100d9aab"

[retrieval]
top_k = 5
max_exact_reads = 12

[precedents]
enabled = true
corpus_path = "data/investors/<vc_slug>/precedents"
selection_policy = "semantic"
candidate_pool_k = 30
in_slots = 2
out_slots = 3

[portfolio_memory]
enabled = true
corpus_path = "data/investors/<vc_slug>/portfolio-memory"
retrieval_top_k = 5
candidate_pool_k = 15
''',
        encoding="utf-8",
    )
    return path


def test_loads_reusable_rehearsal_config(tmp_path: Path) -> None:
    config = load_rehearsal_config(_write(tmp_path))

    assert config.rehearsal.max_questions == 8
    assert config.provider.model == "openai/gpt-5.6-luna"
    assert config.precedents.corpus_path.endswith("/<vc_slug>/precedents")
    assert config.question_memory.enabled
    assert config.question_memory.top_k == 5
    assert config.question_memory.candidate_pool_k == 30
    assert config.question_memory.semantic_weight > 0
    assert config.classification.mode == "classification_informed"
    assert config.classification.minimum_questions == 1
    assert config.classification.minimum_probability_impact == 0.05
    assert config.classification.fallback_to_rationale_only is True


def test_classification_settings_accept_rationale_only_mode() -> None:
    settings = RehearsalClassificationSettings(mode="rationale_only")

    assert settings.mode == "rationale_only"


def test_classification_settings_accept_v41_grounded_mode() -> None:
    settings = RehearsalClassificationSettings(
        mode="v41_grounded",
        canonical_registry_path=(
            "evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json"
        ),
        live_contract_version="v4.1",
        classifier_tiebreaker=True,
    )

    assert settings.mode == "v41_grounded"
    assert settings.live_contract_version == "v4.1"
    assert settings.classifier_tiebreaker is True


def test_grounded_settings_accept_canonical_config_path() -> None:
    settings = RehearsalClassificationSettings(
        mode="v41_grounded",
        canonical_config_path="configs/openrouter-luna-charles-v4.toml",
    )

    assert settings.canonical_config_path.endswith("charles-v4.toml")


@pytest.mark.parametrize("value", ("/tmp/run.toml", "../run.toml"))
def test_canonical_config_path_must_be_safe(value: str) -> None:
    with pytest.raises(ValueError, match="safe relative"):
        RehearsalClassificationSettings(
            mode="v41_grounded",
            canonical_config_path=value,
        )


def test_grounded_registry_must_be_workspace_relative() -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        RehearsalClassificationSettings(
            mode="v41_grounded",
            canonical_registry_path="../labels.json",
        )


def test_grounded_live_contract_is_v4_or_v41() -> None:
    with pytest.raises(ValueError, match="live_contract_version"):
        RehearsalClassificationSettings(
            mode="v41_grounded",
            live_contract_version="v4.4",
        )


def test_classification_question_bounds_must_be_possible() -> None:
    with pytest.raises(ValueError, match="minimum_questions"):
        RehearsalClassificationSettings(minimum_questions=3, maximum_questions=2)


def test_classification_rejects_negative_impact_threshold() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        RehearsalClassificationSettings(minimum_probability_impact=-0.01)


def test_question_memory_requires_positive_scoring_mass() -> None:
    with pytest.raises(ValueError, match="positive scoring weight"):
        RehearsalQuestionMemorySettings(
            semantic_weight=0,
            rationale_weight=0,
            lexical_weight=0,
            atomicity_bonus=0,
        )


def test_question_memory_top_k_cannot_exceed_pool() -> None:
    with pytest.raises(ValueError, match="top_k must not exceed"):
        RehearsalQuestionMemorySettings(top_k=6, candidate_pool_k=5)


def test_rejects_more_than_eight_questions(tmp_path: Path) -> None:
    path = _write(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "max_questions = 8", "max_questions = 9"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="less than or equal to 8"):
        load_rehearsal_config(path)


def test_rejects_remote_precedent_embeddings(tmp_path: Path) -> None:
    path = _write(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'kind = "sentence_transformers"\n'
            'model = "nomic-ai/nomic-embed-text-v1.5"\n'
            'revision = "e9b6763023c676ca8431644204f50c2b100d9aab"',
            'kind = "openrouter"\nmodel = "remote-embedding"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="local"):
        load_rehearsal_config(path)


def test_rejects_unsafe_paths(tmp_path: Path) -> None:
    path = _write(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'output_root = "outputs/rehearsals"', 'output_root = "../elsewhere"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="safe relative"):
        load_rehearsal_config(path)
