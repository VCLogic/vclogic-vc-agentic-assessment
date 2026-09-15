from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import vc_clone_graph.posthoc_semantic_models as semantic_models
from vc_clone_graph.posthoc_semantic_models import (
    PosthocPrediction,
    SemanticCalibrationRecord,
    evaluate_fixed_budget_shortlists,
    evaluate_posthoc_predictions,
    extract_reasoning_views,
    extract_structured_semantic_features,
    nested_posthoc_predictions,
    prototype_probability,
    write_fixed_budget_shortlist_analysis,
    write_posthoc_analysis,
)


def test_tfidf_vectorizer_is_built_once_per_inner_fold(monkeypatch) -> None:
    real_tfidf = semantic_models._tfidf
    call_count = 0

    def counted_tfidf():
        nonlocal call_count
        call_count += 1
        return real_tfidf()

    monkeypatch.setattr(semantic_models, "_tfidf", counted_tfidf)
    targets = np.asarray([1 if index % 3 == 0 else 0 for index in range(18)])
    texts = [
        ("strong founder execution and traction " if target else "fatal market constraint and weak economics ")
        + f"episode {index}"
        for index, target in enumerate(targets)
    ]

    semantic_models._tfidf_outer(texts, targets, "held founder execution", 0.20)

    # Three stratified inner folds plus the final fit on the complete outer train set.
    assert call_count == 4


def _investigation() -> dict:
    return {
        "schema_version": "investigation-v4",
        "episode_slug": "18-rowvigor",
        "questions": [
            {
                "question": "Can this team execute?",
                "status": "partial",
                "answer": "The launch is real, while retention is unresolved.",
                "evidence_ids": ["W-founder"],
            }
        ],
        "rationales": [
            {
                "rationale_id": "R1",
                "taxonomy_label": "founder_execution",
                "direction": "positive",
                "salience": "primary",
                "confidence": 0.82,
                "pitch_evidence": ["We shipped the product."],
                "pitch_evidence_ids": ["P-001"],
                "wiki_evidence_ids": ["W-founder"],
                "historical_evidence_ids": ["H-prior"],
                "justification": "The founder has demonstrated execution.",
            },
            {
                "rationale_id": "R2",
                "taxonomy_label": "market_size_assessment",
                "direction": "negative",
                "salience": "secondary",
                "confidence": 0.61,
                "pitch_evidence": ["The initial segment is narrow."],
                "pitch_evidence_ids": ["P-002"],
                "wiki_evidence_ids": ["W-market"],
                "historical_evidence_ids": [],
                "justification": "Venture scale remains uncertain.",
            },
        ],
        "conflicts": ["Execution is positive but scale is unresolved."],
        "unmapped_observations": [],
        "information_sufficient": True,
        "sufficiency_assessment": "Enough information exists for a decision.",
        "searchable_questions": [],
        "diligence_questions": ["What is retention?"],
        "next_search_objectives": [],
        "summary": "Execution is positive and scale is uncertain.",
    }


def _decision() -> dict:
    return {
        "schema_version": "decision-v4",
        "episode_slug": "18-rowvigor",
        "investigation_sha256": "a" * 64,
        "decision": "In",
        "decision_path": "founder_conviction_exception",
        "investment_likelihood": 0.62,
        "decision_confidence": 0.74,
        "decision_justification": "I would make a small check because execution outweighs uncertainty.",
        "recommended_check_tier": "exploratory_lt_100k",
        "review_priority_score": 0.81,
        "review_priority_reason": "The founder warrants review.",
        "controlling_rationale_ids": ["R1", "R2"],
        "evidence_basis": [
            {
                "source_type": "precedent",
                "source_reference": "Prior small-check precedent.",
                "evidence_ids": ["H-prior"],
                "interpretation": "Charles previously accepted unresolved scale with execution.",
                "effect_on_decision": "supports",
            }
        ],
        "strongest_opposing_case": {
            "argument": "The market may be too narrow.",
            "response": "A small check limits exposure while the founder tests scale.",
        },
        "searchable_questions": [],
        "diligence_questions": ["Can acquisition scale?"],
        "reversal_conditions": ["Retention is weak."],
        "information_sufficient": True,
        "sufficiency_assessment": "Enough evidence exists for a bounded decision.",
        "next_search_objectives": [],
        "phase1_reopen_recommended": False,
        "missing_considerations": [],
        "founder_exception_considered": True,
        "founder_exception_rationale_ids": ["R1"],
        "founder_exception_precedent_ids": ["H-prior"],
        "founder_exception_assessment": "Execution justifies a deliberately small check.",
    }


def test_reasoning_views_are_deterministic_and_exclude_outcome_fields() -> None:
    first = extract_reasoning_views(_investigation(), _decision())
    second = extract_reasoning_views(_investigation(), _decision())

    assert first == second
    assert "founder_execution" in first["phase1_rationales"]
    assert "positive primary confidence=0.820" in first["phase1_rationales"]
    assert "strongest opposing case" in first["full_reasoning"].lower()
    assert "Charles previously accepted unresolved scale" in first["full_reasoning"]
    assert "actual_decision" not in first["full_reasoning"]
    assert "pitch_window_decision" not in first["full_reasoning"]


def test_structured_features_keep_rationale_semantics_without_target() -> None:
    features = extract_structured_semantic_features(
        _investigation(), _decision(), {"phase1_status": "accepted", "phase2_status": "accepted"}
    )

    assert features["label__founder_execution__signed_confidence"] == pytest.approx(0.82)
    assert features["label__market_size_assessment__signed_confidence"] == pytest.approx(-0.61)
    assert features["evidence_source_effect__precedent__supports__count"] == 1.0
    assert not any("target" in key or "actual" in key for key in features)


def test_prototype_probability_uses_training_centroids() -> None:
    train_vectors = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])
    targets = np.asarray([1, 1, 0, 0])

    positive = prototype_probability(train_vectors, targets, np.asarray([0.9, 0.1]))
    negative = prototype_probability(train_vectors, targets, np.asarray([0.1, 0.9]))

    assert 0.0 <= negative < 0.5 < positive <= 1.0


def _records(*, flip_first: bool = False) -> list[SemanticCalibrationRecord]:
    records = []
    for index in range(18):
        target = int(index % 3 == 0)
        if flip_first and index == 0:
            target = 1 - target
        signal = 1.0 if index % 3 == 0 else -1.0
        records.append(
            SemanticCalibrationRecord(
                episode_slug=f"{index + 1}-example",
                target=target,
                target_decision="In" if target else "Out",
                original_decision="In" if index % 4 == 0 else "Out",
                original_likelihood=0.5 + 0.1 * signal,
                review_priority_score=0.6 + 0.05 * signal,
                phase1_text=("strong founder execution " if signal > 0 else "fatal market constraint ")
                + f"case {index % 2}",
                full_text=("strong founder execution and traction " if signal > 0 else "fatal market and economics constraint ")
                + f"case {index % 2}",
                structured_features={
                    "positive_primary_confidence_sum": max(signal, 0.0),
                    "negative_primary_confidence_sum": max(-signal, 0.0),
                    "label__founder_execution__signed_confidence": signal,
                },
                artifact_root=f"runs/{index + 1}-example",
                phase1_sha256=str(index).zfill(64),
                phase2_sha256=str(index + 1).zfill(64),
            )
        )
    return records


def _embeddings(records: list[SemanticCalibrationRecord]) -> dict[str, np.ndarray]:
    vectors = np.asarray(
        [[1.0, 0.0] if record.target else [0.0, 1.0] for record in records],
        dtype=float,
    )
    return {"phase1_rationales": vectors, "full_reasoning": vectors.copy()}


def test_nested_methods_exclude_held_episode_and_ignore_held_label() -> None:
    records = _records()
    embeddings = _embeddings(records)

    original = nested_posthoc_predictions(
        records,
        embeddings,
        methods=("tfidf_full", "embedding_prototype", "embedding_logistic", "structured_elastic", "late_fusion"),
        false_positive_rate_cap=0.20,
    )
    flipped = nested_posthoc_predictions(
        _records(flip_first=True),
        embeddings,
        methods=("embedding_prototype",),
        false_positive_rate_cap=0.20,
    )

    assert set(original) == {
        "tfidf_full",
        "embedding_prototype",
        "embedding_logistic",
        "structured_elastic",
        "late_fusion",
    }
    for predictions in original.values():
        assert len(predictions) == len(records)
        assert all(
            prediction.episode_slug not in prediction.training_slugs
            and len(prediction.training_slugs) == len(records) - 1
            for prediction in predictions
        )
    assert flipped["embedding_prototype"][0].score == pytest.approx(
        original["embedding_prototype"][0].score
    )
    assert flipped["embedding_prototype"][0].threshold == pytest.approx(
        original["embedding_prototype"][0].threshold
    )


def test_evaluate_and_write_posthoc_analysis(tmp_path: Path) -> None:
    records = _records()
    predictions = nested_posthoc_predictions(
        records,
        _embeddings(records),
        methods=("embedding_prototype", "structured_elastic", "late_fusion"),
        false_positive_rate_cap=0.20,
    )

    metrics = evaluate_posthoc_predictions(records, predictions)
    result = write_posthoc_analysis(
        records,
        predictions,
        tmp_path / "analysis",
        embedding_metadata={"model": "test-model", "revision": "abc123"},
        source_manifest={"status_path": "status.json", "labels_path": "labels.json"},
    )

    assert set(metrics) == set(predictions)
    assert {path.name for path in (tmp_path / "analysis").iterdir()} == {
        "embedding-metadata.json",
        "manifest.json",
        "metrics.json",
        "predictions.csv",
        "rankings.csv",
        "report.md",
    }
    manifest = json.loads(
        (tmp_path / "analysis" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["api_cost_usd"] == 0.0
    assert manifest["population"] == {"episodes": 18, "ins": 6, "outs": 12}
    assert result["success_criterion"] == {"minimum_tp": 7, "maximum_fp": 12}


def _fixed_predictions(
    records: list[SemanticCalibrationRecord], *, target_flip: bool = False
) -> dict[str, list[PosthocPrediction]]:
    methods = ("tfidf_full", "embedding_logistic", "structured_boost")
    result: dict[str, list[PosthocPrediction]] = {method: [] for method in methods}
    for index, record in enumerate(records):
        for offset, method in enumerate(methods):
            result[method].append(
                PosthocPrediction(
                    method=method,
                    episode_slug=record.episode_slug,
                    target=(1 - record.target) if target_flip else record.target,
                    score=float((index * (offset + 2)) % len(records)),
                    predicted=int((index + offset) % 4 == 0),
                    threshold=0.5,
                    selected_config={},
                    training_slugs=tuple(
                        other.episode_slug
                        for other in records
                        if other.episode_slug != record.episode_slug
                    ),
                )
            )
    return result


def test_fixed_budget_shortlists_are_label_blind_and_preserve_review_budget() -> None:
    records = _records()
    predictions = _fixed_predictions(records)

    first = evaluate_fixed_budget_shortlists(records, predictions, review_budget=7)
    flipped = evaluate_fixed_budget_shortlists(
        _records(flip_first=True), _fixed_predictions(records, target_flip=True), review_budget=7
    )

    assert set(first) == {
        "semantic_structured_rank_fusion",
        "raw_semantic_rank_fusion",
        "budget_matched_semantic_rescue",
    }
    assert sum(row["selected"] for row in first["semantic_structured_rank_fusion"]["rows"]) == 7
    assert sum(row["selected"] for row in first["raw_semantic_rank_fusion"]["rows"]) == 7
    assert sum(row["selected"] for row in first["budget_matched_semantic_rescue"]["rows"]) == sum(
        record.original_decision == "In" for record in records
    )
    for method in first:
        selected = {
            row["episode_slug"] for row in first[method]["rows"] if row["selected"]
        }
        flipped_selected = {
            row["episode_slug"] for row in flipped[method]["rows"] if row["selected"]
        }
        assert selected == flipped_selected


def test_write_fixed_budget_shortlist_analysis(tmp_path: Path) -> None:
    records = _records()
    result = write_fixed_budget_shortlist_analysis(
        records,
        _fixed_predictions(records),
        tmp_path / "shortlists",
        review_budget=7,
        source_manifest={"predictions_path": "predictions.csv"},
    )

    assert {path.name for path in (tmp_path / "shortlists").iterdir()} == {
        "manifest.json",
        "metrics.json",
        "shortlists.csv",
        "report.md",
    }
    assert result["evaluation"] == "label_blind_fixed_budget_batch_prioritization"
    assert result["api_cost_usd"] == 0.0
