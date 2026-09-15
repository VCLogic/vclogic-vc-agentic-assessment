"""Fold-local feature matrix assembly for personalized model ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ABLATIONS = (
    "pitch",
    "phase1",
    "phase2",
    "pitch_phase1",
    "pitch_phase2",
    "phase1_phase2",
    "all",
    "all_no_teacher",
    "all_no_vc_identity",
    "all_no_wiki",
)


@dataclass(frozen=True)
class RawFeatureViews:
    pitch_embedding: np.ndarray
    phase1_embedding: np.ndarray
    phase2_embedding: np.ndarray
    wiki_embedding: np.ndarray
    pitch_structured: Mapping[str, float]
    phase1_structured: Mapping[str, float]
    phase2_structured: Mapping[str, float]
    investor_structured: Mapping[str, float]
    teacher_structured: Mapping[str, float]
    interaction_structured: Mapping[str, float]


def _families(condition: str) -> tuple[set[str], bool, bool, bool]:
    if condition not in ABLATIONS:
        raise ValueError(f"unknown personalized feature condition: {condition}")
    base = condition.removeprefix("all_no_")
    if condition.startswith("all"):
        selected = {"pitch", "phase1", "phase2"}
    else:
        selected = set(base.split("_"))
    include_teacher = condition == "all"
    if condition == "all_no_teacher":
        include_teacher = False
    include_identity = condition != "all_no_vc_identity"
    include_wiki = condition != "all_no_wiki"
    return selected, include_teacher, include_identity, include_wiki


def _selected(
    row: RawFeatureViews, condition: str
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    families, include_teacher, include_identity, include_wiki = _families(condition)
    blocks: dict[str, np.ndarray] = {}
    structured: dict[str, float] = {}
    for family in sorted(families):
        blocks[f"{family}_embedding"] = np.asarray(
            getattr(row, f"{family}_embedding"), dtype=float
        )
        structured.update(getattr(row, f"{family}_structured"))
    if include_wiki:
        blocks["wiki_embedding"] = np.asarray(row.wiki_embedding, dtype=float)
    for name, value in row.investor_structured.items():
        if not include_identity and name.startswith("vc_identity__"):
            continue
        if not include_wiki and "wiki" in name:
            continue
        structured[name] = float(value)
    if include_teacher:
        structured.update(row.teacher_structured)
    if len(families) > 1 or condition.startswith("all"):
        structured.update(row.interaction_structured)
    return blocks, structured


@dataclass
class FittedAblationPreprocessor:
    condition: str
    block_names: tuple[str, ...]
    block_dimensions: Mapping[str, int]
    pca_by_block: Mapping[str, PCA | None]
    vectorizer: DictVectorizer
    imputer: SimpleImputer
    variance_indices: np.ndarray
    correlation_indices: np.ndarray
    cap_indices: np.ndarray
    scaler: StandardScaler
    feature_names: tuple[str, ...]

    def _raw_matrix(
        self, rows: Sequence[RawFeatureViews], indices: Sequence[int]
    ) -> np.ndarray:
        selected = [_selected(rows[index], self.condition) for index in indices]
        parts = []
        for name in self.block_names:
            matrix = np.asarray([blocks[name] for blocks, _ in selected], dtype=float)
            if matrix.ndim != 2 or matrix.shape[1] != self.block_dimensions[name]:
                raise ValueError(f"embedding block changed shape: {name}")
            pca = self.pca_by_block[name]
            parts.append(pca.transform(matrix) if pca is not None else matrix)
        dictionaries = [structured for _, structured in selected]
        structured_matrix = self.vectorizer.transform(dictionaries)
        structured_matrix = self.imputer.transform(structured_matrix)
        parts.append(np.asarray(structured_matrix, dtype=float))
        return np.hstack(parts)

    def transform(
        self, rows: Sequence[RawFeatureViews], indices: Sequence[int]
    ) -> np.ndarray:
        matrix = self._raw_matrix(rows, indices)
        matrix = matrix[:, self.variance_indices]
        matrix = matrix[:, self.correlation_indices]
        matrix = matrix[:, self.cap_indices]
        return self.scaler.transform(matrix)


def _correlation_keep(matrix: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    if matrix.shape[1] < 2:
        return np.arange(matrix.shape[1], dtype=int)
    correlations = np.corrcoef(matrix, rowvar=False)
    keep = []
    for index in range(matrix.shape[1]):
        observed = correlations[index, keep]
        redundant = bool(
            np.any(np.isfinite(observed) & (np.abs(observed) >= threshold))
        )
        if not redundant:
            keep.append(index)
    return np.asarray(keep, dtype=int)


def fit_ablation_preprocessor(
    rows: Sequence[RawFeatureViews],
    *,
    train_indices: Sequence[int],
    condition: str,
    pca_components: int,
    maximum_features: int,
) -> FittedAblationPreprocessor:
    """Fit every distribution-dependent transform on the supplied rows only."""
    if not train_indices or pca_components < 1 or maximum_features < 1:
        raise ValueError("invalid preprocessing bounds")
    selected = [_selected(rows[index], condition) for index in train_indices]
    block_names = tuple(sorted(selected[0][0]))
    if any(tuple(sorted(blocks)) != block_names for blocks, _ in selected):
        raise ValueError("embedding blocks are not aligned")
    block_dimensions: dict[str, int] = {}
    pca_by_block: dict[str, PCA | None] = {}
    parts = []
    names = []
    for name in block_names:
        matrix = np.asarray([blocks[name] for blocks, _ in selected], dtype=float)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError(f"invalid embedding block: {name}")
        block_dimensions[name] = int(matrix.shape[1])
        components = min(pca_components, matrix.shape[1], len(matrix))
        if components < matrix.shape[1] and np.ptp(matrix, axis=0).max() > 1e-12:
            pca = PCA(n_components=components, svd_solver="full").fit(matrix)
            matrix = pca.transform(matrix)
            pca_by_block[name] = pca
            names.extend(f"{name}__pc{index + 1}" for index in range(components))
        else:
            pca_by_block[name] = None
            names.extend(f"{name}__dim{index + 1}" for index in range(matrix.shape[1]))
        parts.append(matrix)

    vectorizer = DictVectorizer(sparse=False)
    structured = vectorizer.fit_transform([values for _, values in selected])
    imputer = SimpleImputer(strategy="median", add_indicator=True).fit(structured)
    structured = imputer.transform(structured)
    structured_names = list(vectorizer.get_feature_names_out())
    indicator = getattr(imputer, "indicator_", None)
    if indicator is not None:
        structured_names.extend(
            f"missing__{vectorizer.get_feature_names_out()[index]}"
            for index in indicator.features_
        )
    names.extend(str(value) for value in structured_names)
    parts.append(np.asarray(structured, dtype=float))
    raw = np.hstack(parts)
    if not np.isfinite(raw).all():
        raise ValueError("preprocessed feature matrix is non-finite")
    variance_indices = np.flatnonzero(np.ptp(raw, axis=0) > 1e-12)
    if not len(variance_indices):
        raise ValueError("all candidate features are constant")
    varying = raw[:, variance_indices]
    varying_names = np.asarray(names, dtype=object)[variance_indices]
    correlation_indices = _correlation_keep(varying)
    decorrelated = varying[:, correlation_indices]
    decorrelated_names = varying_names[correlation_indices]
    if decorrelated.shape[1] > maximum_features:
        ranked = np.argsort(-np.var(decorrelated, axis=0), kind="stable")[:maximum_features]
        cap_indices = np.sort(ranked)
    else:
        cap_indices = np.arange(decorrelated.shape[1], dtype=int)
    capped = decorrelated[:, cap_indices]
    feature_names = tuple(str(value) for value in decorrelated_names[cap_indices])
    scaler = StandardScaler().fit(capped)
    return FittedAblationPreprocessor(
        condition=condition,
        block_names=block_names,
        block_dimensions=block_dimensions,
        pca_by_block=pca_by_block,
        vectorizer=vectorizer,
        imputer=imputer,
        variance_indices=variance_indices,
        correlation_indices=correlation_indices,
        cap_indices=cap_indices,
        scaler=scaler,
        feature_names=feature_names,
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator else 0.0


def cross_view_interactions(
    pitch: np.ndarray,
    phase1: np.ndarray,
    phase2: np.ndarray,
    wiki: np.ndarray,
    *,
    phase1_signed_mass: float,
    phase2_likelihood: float,
    phase2_decision_in: float,
) -> dict[str, float]:
    """Return prespecified semantic and decision-agreement interactions."""
    return {
        "interaction__pitch_phase1_cosine": cosine(pitch, phase1),
        "interaction__phase1_phase2_cosine": cosine(phase1, phase2),
        "interaction__pitch_wiki_cosine": cosine(pitch, wiki),
        "interaction__signed_mass_x_likelihood": phase1_signed_mass * phase2_likelihood,
        "interaction__phase1_phase2_direction_agreement": float(
            (phase1_signed_mass >= 0) == bool(phase2_decision_in)
        ),
    }
