from __future__ import annotations

from dataclasses import replace

import numpy as np

from vc_clone_graph.personalized_matrices import (
    ABLATIONS,
    RawFeatureViews,
    _correlation_keep,
    fit_ablation_preprocessor,
)


def rows() -> list[RawFeatureViews]:
    result = []
    for index in range(6):
        result.append(RawFeatureViews(
            pitch_embedding=np.asarray([index, index % 2, 1.0], dtype=float),
            phase1_embedding=np.asarray([index * 0.5, 1.0 - index % 2], dtype=float),
            phase2_embedding=np.asarray([index * 0.25, index % 3], dtype=float),
            wiki_embedding=np.asarray([1.0, float(index % 2)], dtype=float),
            pitch_structured={"pitch__length": float(index + 1), "constant": 1.0},
            phase1_structured={"p1__market": float(index % 2)},
            phase2_structured={"p2__likelihood": index / 5},
            investor_structured={
                "vc_identity__alpha": float(index < 3),
                "vc_identity__beta": float(index >= 3),
                "similarity__wiki": index / 10,
            },
            teacher_structured={"teacher__market": index / 6},
            interaction_structured={"interaction__agreement": float(index % 2)},
        ))
    return result


def test_all_prespecified_ablation_views_are_available() -> None:
    assert ABLATIONS == (
        "pitch", "phase1", "phase2", "pitch_phase1", "pitch_phase2",
        "phase1_phase2", "all", "all_no_teacher", "all_no_vc_identity",
        "all_no_wiki",
    )


def test_preprocessing_is_fit_only_on_training_rows() -> None:
    original = rows()
    fit = fit_ablation_preprocessor(
        original, train_indices=(0, 1, 2, 3, 4), condition="all",
        pca_components=2, maximum_features=64,
    )
    changed = list(original)
    changed[5] = replace(
        changed[5],
        pitch_embedding=np.asarray([1e9, -1e9, 1e9]),
        phase2_structured={"held_only_feature": 1e12},
    )
    repeated = fit_ablation_preprocessor(
        changed, train_indices=(0, 1, 2, 3, 4), condition="all",
        pca_components=2, maximum_features=64,
    )

    assert fit.feature_names == repeated.feature_names
    assert np.allclose(fit.transform(original, (0, 1, 2, 3, 4)), repeated.transform(changed, (0, 1, 2, 3, 4)))
    assert "held_only_feature" not in " ".join(fit.feature_names)
    assert not any(name.endswith("constant") for name in fit.feature_names)


def test_ablation_removes_only_requested_feature_family() -> None:
    source = rows()
    no_teacher = fit_ablation_preprocessor(
        source, train_indices=(0, 1, 2, 3, 4), condition="all_no_teacher",
        pca_components=1, maximum_features=64,
    )
    no_identity = fit_ablation_preprocessor(
        source, train_indices=(0, 1, 2, 3, 4), condition="all_no_vc_identity",
        pca_components=1, maximum_features=64,
    )
    no_wiki = fit_ablation_preprocessor(
        source, train_indices=(0, 1, 2, 3, 4), condition="all_no_wiki",
        pca_components=1, maximum_features=64,
    )

    assert not any(name.startswith("teacher__") for name in no_teacher.feature_names)
    assert not any("vc_identity__" in name for name in no_identity.feature_names)
    assert not any(name.startswith("wiki_embedding") for name in no_wiki.feature_names)
    assert all(model.transform(source, (5,)).shape[0] == 1 for model in (no_teacher, no_identity, no_wiki))


def test_correlation_pruning_computes_one_vectorized_matrix(monkeypatch) -> None:
    random = np.random.default_rng(17)
    matrix = random.normal(size=(60, 30))
    matrix[:, 20:25] = matrix[:, :5]
    observed_calls = 0
    original = np.corrcoef

    def counted(*args, **kwargs):
        nonlocal observed_calls
        observed_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "corrcoef", counted)
    kept = _correlation_keep(matrix)

    assert observed_calls == 1
    assert set(range(20, 25)).isdisjoint(kept)
