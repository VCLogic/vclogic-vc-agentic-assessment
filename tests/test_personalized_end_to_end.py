from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="requires the optional setfit/personalized extra")

import json

import numpy as np

from vc_clone_graph.personalized_cases import PersonalizedCase
from vc_clone_graph.personalized_evaluation import (
    PredictionRow,
    evaluate_personalized_predictions,
    write_personalized_evaluation,
)
from vc_clone_graph.personalized_folds import make_episode_folds
from vc_clone_graph.personalized_runner import (
    PreparedEmbeddings,
    build_fold_views,
    run_checkpointed_stage,
)
from vc_clone_graph.personalized_setfit import run_setfit_fold
from vc_clone_graph.personalized_tabpfn import run_tabpfn_fold


class FakeTabPFN:
    def fit(self, matrix, targets):
        self.direction = matrix[targets == 1].mean(0) - matrix[targets == 0].mean(0)
        return self

    def predict_proba(self, matrix):
        score = 1 / (1 + np.exp(-np.clip(matrix @ self.direction, -10, 10)))
        return np.column_stack((1 - score, score))


class TinyEncoder:
    def adapt(self, population, pairs, *, epochs, seed, condition):
        del population, pairs, epochs, seed, condition

    def encode(self, texts):
        return np.asarray([[float("Alpha" in text), float("Beta" in text), len(text) / 100] for text in texts])


def population() -> list[PersonalizedCase]:
    result = []
    for episode in range(6):
        for vc, target in (("alpha", 1), ("beta", 0)):
            result.append(PersonalizedCase(
                vc_slug=vc, vc_input_slug=f"{vc}-fund", vc_name=vc.title(),
                episode_slug=f"{episode}-shared", group=f"{episode}-shared", target=target,
                pitch_text=f"pitch {episode}", phase1_text="market rationale",
                phase2_text=f"likelihood {0.7 if target else 0.3}",
                phase1_features={"rationale__market__signed_confidence": 0.7},
                phase2_features={"investment_likelihood": 0.7 if target else 0.3},
                wiki_text=f"{vc} investment profile",
                actual_rationale_targets={"market": target},
                actual_rationale_available=True, actual_rationale_source_tier="test",
                actual_rationale_source_format="test",
                source_hashes={"pitch": f"{episode:064x}"},
            ))
    return result


def prepared(cases):
    pitch = np.asarray([[float(index), 1, 0] for index in range(len(cases))])
    phase1 = np.asarray([[0, 1, float(row.target)] for row in cases])
    phase2 = np.asarray([[0, float(row.target), 1] for row in cases])
    wiki_by_vc = {"alpha": np.asarray([1.0, 0, 0]), "beta": np.asarray([0, 1.0, 0])}
    wiki = np.asarray([wiki_by_vc[row.vc_slug] for row in cases])
    return PreparedEmbeddings(
        pitch_model=np.column_stack((pitch, pitch)),
        phase1_model=np.column_stack((phase1, phase1)),
        phase2_model=np.column_stack((phase2, phase2)),
        wiki_model=np.column_stack((wiki, wiki)),
        pitch_similarity=pitch, phase1_similarity=phase1, phase2_similarity=phase2,
        wiki_similarity_by_vc=wiki_by_vc,
        portfolio_similarity_by_vc={
            "alpha": np.asarray([[1.0, 0, 0]]), "beta": np.asarray([[0, 1.0, 0]])
        },
        embedding_metadata={"model": "fake", "revision": "locked"},
    )


def test_personalized_pipeline_runs_and_resumes_without_held_provenance(tmp_path) -> None:
    cases = population()
    fold = make_episode_folds(cases, inner_splits=2, seed=71)[0]
    views = build_fold_views(cases, prepared(cases), fold, ("market",), seed=71)
    tab = run_tabpfn_fold(
        cases, views, fold, condition="all", n_estimators=(2,),
        pca_components=(1,), device="cpu", seed=71,
        factory=lambda **kwargs: FakeTabPFN(),
    )
    setfit = run_setfit_fold(
        cases, views, fold, condition="all", epochs=(1,), lambda_rank=(0.25,),
        lambda_aux=(0.1,), max_pairs=32, taxonomy_labels=("market",), seed=71,
        encoder_factory=TinyEncoder, device="cpu",
    )
    assert all(fold.held_episode not in row.training_episodes for row in (*tab, *setfit))
    assert all(
        fold.held_episode not in episodes
        for row in (*tab, *setfit) for episodes in row.inner_validation_episodes
    )

    calls = 0
    def execute(_condition, _fold):
        nonlocal calls
        calls += 1
        return tab

    first = run_checkpointed_stage(
        family="tabpfn", conditions=("all",), folds=(fold,), output=tmp_path,
        source_manifest={"registry_sha256": "a" * 64}, run_fold=execute,
        max_retries=0, resume=True,
    )
    second = run_checkpointed_stage(
        family="tabpfn", conditions=("all",), folds=(fold,), output=tmp_path,
        source_manifest={"registry_sha256": "a" * 64}, run_fold=execute,
        max_retries=0, resume=True,
    )
    assert first[0]["status"] == second[0]["status"] == "complete"
    assert first[0]["predictions"][0]["episode_slug"] == second[0]["predictions"][0]["episode_slug"]
    assert calls == 1

    rows = []
    for family, predictions in (("tabpfn", tab), ("setfit", setfit)):
        rows.extend(PredictionRow(
            method=family, family=family, condition="all", vc_slug=row.vc_slug,
            vc_name=row.vc_name, episode_slug=row.episode_slug, target=row.target,
            classification_score=row.classification_score, ranking_score=row.ranking_score,
            predicted=row.predicted, status="complete", runtime_seconds=row.runtime_seconds,
        ) for row in predictions)
    rows.extend(PredictionRow(
        method="raw_phase2", family="raw_phase2", condition="raw", vc_slug=row.vc_slug,
        vc_name=row.vc_name, episode_slug=row.episode_slug, target=row.target,
        classification_score=0.7 if row.target else 0.3,
        ranking_score=0.7 if row.target else 0.3, predicted=row.target, status="complete",
    ) for row in tab)
    result = evaluate_personalized_predictions(
        rows, expected_case_count=len(tab), review_budgets=(1, 2),
        bootstrap_samples=10, randomization_samples=10, seed=71,
    )
    write_personalized_evaluation(
        tmp_path / "report", result,
        source_manifest=json.loads((tmp_path / "run-manifest.json").read_text()),
        automated_rationale_references=True,
    )
    assert (tmp_path / "report/report.md").is_file()
    assert set(result["classification_macro"]) == {"raw_phase2", "setfit", "tabpfn"}
