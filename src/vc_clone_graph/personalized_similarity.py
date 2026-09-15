"""Fold-local investor, precedent, and portfolio similarity features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .personalized_cases import PersonalizedCase


@dataclass(frozen=True)
class SimilarityFeatures:
    features_by_index: Mapping[int, Mapping[str, float]]
    provenance_by_index: Mapping[int, Mapping[str, object]]


def _vector(value: np.ndarray, dimensions: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (dimensions,) or not np.isfinite(result).all():
        raise ValueError(f"invalid {name} embedding")
    return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator else 0.0


def _summary(
    query: np.ndarray, candidates: Sequence[np.ndarray], prefix: str
) -> dict[str, float]:
    if not candidates:
        return {
            f"similarity__{prefix}_max": 0.0,
            f"similarity__{prefix}_mean": 0.0,
            f"similarity__{prefix}_count": 0.0,
            f"similarity__{prefix}_unavailable": 1.0,
        }
    scores = [_cosine(query, candidate) for candidate in candidates]
    return {
        f"similarity__{prefix}_max": max(scores),
        f"similarity__{prefix}_mean": float(np.mean(scores)),
        f"similarity__{prefix}_count": float(len(scores)),
        f"similarity__{prefix}_unavailable": 0.0,
    }


def similarity_features(
    cases: Sequence[PersonalizedCase],
    pitch_embeddings: np.ndarray,
    *,
    wiki_embeddings: Mapping[str, np.ndarray],
    portfolio_embeddings: Mapping[str, np.ndarray],
    reference_indices: Sequence[int],
    query_indices: Sequence[int],
) -> SimilarityFeatures:
    """Compute VC-specific history features with query-episode exclusion."""
    matrix = np.asarray(pitch_embeddings, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(cases) or not np.isfinite(matrix).all():
        raise ValueError("pitch embedding population is invalid")
    dimensions = int(matrix.shape[1])
    reference = tuple(int(index) for index in reference_indices)
    queries = tuple(int(index) for index in query_indices)
    if any(index < 0 or index >= len(cases) for index in reference + queries):
        raise ValueError("similarity index is out of range")

    features: dict[int, Mapping[str, float]] = {}
    provenance: dict[int, Mapping[str, object]] = {}
    for query_index in queries:
        query_case = cases[query_index]
        query = matrix[query_index]
        wiki = _vector(
            wiki_embeddings[query_case.vc_slug], dimensions, "wiki"
        )
        eligible = [
            index
            for index in reference
            if cases[index].vc_slug == query_case.vc_slug
            and cases[index].episode_slug != query_case.episode_slug
        ]
        positives = [index for index in eligible if cases[index].target == 1]
        negatives = [index for index in eligible if cases[index].target == 0]
        row = {"similarity__wiki": _cosine(query, wiki)}
        row.update(_summary(query, [matrix[index] for index in positives], "prior_in"))
        row.update(_summary(query, [matrix[index] for index in negatives], "prior_out"))
        row["similarity__in_minus_out_max"] = (
            row["similarity__prior_in_max"] - row["similarity__prior_out_max"]
            if positives and negatives else 0.0
        )

        portfolio_raw = portfolio_embeddings.get(query_case.vc_slug)
        portfolio: list[np.ndarray] = []
        if portfolio_raw is not None:
            portfolio_matrix = np.asarray(portfolio_raw, dtype=float)
            if portfolio_matrix.ndim != 2 or portfolio_matrix.shape[1] != dimensions:
                raise ValueError("portfolio embedding matrix has the wrong dimension")
            if not np.isfinite(portfolio_matrix).all():
                raise ValueError("portfolio embedding matrix is non-finite")
            portfolio = [row_vector for row_vector in portfolio_matrix]
        row.update(_summary(query, portfolio, "portfolio"))
        features[query_index] = row
        provenance[query_index] = {
            "query_episode": query_case.episode_slug,
            "vc_slug": query_case.vc_slug,
            "prior_in_episodes": tuple(sorted({cases[index].episode_slug for index in positives})),
            "prior_out_episodes": tuple(sorted({cases[index].episode_slug for index in negatives})),
            "portfolio_embedding_count": len(portfolio),
        }
    return SimilarityFeatures(features, provenance)
