"""Personalized SetFit-style contrastive and multi-task decision model."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random
import time
from typing import Callable, Protocol, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .personalized_cases import PersonalizedCase
from .personalized_folds import EpisodeFold
from .personalized_matrices import RawFeatureViews, fit_ablation_preprocessor
from .personalized_tabpfn import (
    PersonalizedPrediction,
    TabPFNCandidate,
    select_tabpfn_heads,
)


@dataclass(frozen=True)
class ContrastivePair:
    left_index: int
    right_index: int
    similar: int


@dataclass(frozen=True)
class RankingPair:
    in_index: int
    out_index: int
    vc_slug: str


@dataclass(frozen=True)
class SetFitPairs:
    contrastive: tuple[ContrastivePair, ...]
    ranking: tuple[RankingPair, ...]


def serialize_case(case: PersonalizedCase, condition: str = "all") -> str:
    """Serialize deployable investor-conditioned views without reference labels."""
    if condition.startswith("all"):
        families = {"pitch", "phase1", "phase2"}
    else:
        families = set(condition.split("_"))
    profile = "" if condition == "all_no_wiki" else f" [PROFILE] {case.wiki_text.strip()}"
    lines = [f"[INVESTOR] {case.vc_name}{profile}"]
    if "pitch" in families:
        lines.append(f"[PITCH] {case.pitch_text.strip()}")
    if "phase1" in families:
        lines.append(f"[PHASE1] {case.phase1_text.strip()}")
    if "phase2" in families:
        lines.append(f"[PHASE2] {case.phase2_text.strip()}")
    return "\n".join(lines)


def _bounded(
    rows: list[object], maximum: int, randomizer: random.Random
) -> list[object]:
    randomizer.shuffle(rows)
    return rows[:maximum]


def build_setfit_pairs(
    cases: Sequence[PersonalizedCase],
    train_indices: Sequence[int],
    *,
    max_pairs: int,
    seed: int,
) -> SetFitPairs:
    """Create deterministic training-only contrastive and within-VC rank pairs."""
    if max_pairs < 2:
        raise ValueError("SetFit max_pairs must be at least two")
    indices = tuple(int(index) for index in train_indices)
    if len(indices) != len(set(indices)) or any(
        index < 0 or index >= len(cases) for index in indices
    ):
        raise ValueError("SetFit training indices are invalid")
    positive_pairs: list[ContrastivePair] = []
    negative_pairs: list[ContrastivePair] = []
    for offset, left in enumerate(indices):
        for right in indices[offset + 1:]:
            pair = ContrastivePair(
                left_index=left,
                right_index=right,
                similar=int(cases[left].target == cases[right].target),
            )
            (positive_pairs if pair.similar else negative_pairs).append(pair)
    randomizer = random.Random(seed)
    per_class = max_pairs // 2
    selected_positive = _bounded(positive_pairs, per_class, randomizer)
    selected_negative = _bounded(negative_pairs, per_class, randomizer)
    remaining = max_pairs - len(selected_positive) - len(selected_negative)
    leftovers = positive_pairs[per_class:] + negative_pairs[per_class:]
    selected = selected_positive + selected_negative + _bounded(
        leftovers, remaining, randomizer
    )
    selected.sort(key=lambda row: (row.left_index, row.right_index, row.similar))

    ranking: list[RankingPair] = []
    for vc in sorted({cases[index].vc_slug for index in indices}):
        ins = [index for index in indices if cases[index].vc_slug == vc and cases[index].target]
        outs = [index for index in indices if cases[index].vc_slug == vc and not cases[index].target]
        ranking.extend(
            RankingPair(in_index=in_index, out_index=out_index, vc_slug=vc)
            for in_index in ins for out_index in outs
        )
    ranking = _bounded(ranking, max_pairs, random.Random(seed + 1))
    ranking.sort(key=lambda row: (row.vc_slug, row.in_index, row.out_index))
    return SetFitPairs(tuple(selected), tuple(ranking))


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    classification: torch.Tensor
    ranking: torch.Tensor
    auxiliary: torch.Tensor


def multitask_loss(
    *,
    decision_logits: torch.Tensor,
    decisions: torch.Tensor,
    ranking_pairs: Sequence[tuple[int, int]],
    rationale_logits: torch.Tensor,
    rationale_targets: torch.Tensor,
    rationale_mask: torch.Tensor,
    lambda_rank: float,
    lambda_aux: float,
    positive_weight: float,
) -> LossBreakdown:
    """Combine class-balanced decision, pairwise rank, and rationale losses."""
    classification = F.binary_cross_entropy_with_logits(
        decision_logits,
        decisions,
        pos_weight=torch.as_tensor(
            positive_weight, dtype=decision_logits.dtype, device=decision_logits.device
        ),
    )
    if ranking_pairs:
        in_positions = torch.as_tensor(
            [pair[0] for pair in ranking_pairs], device=decision_logits.device
        )
        out_positions = torch.as_tensor(
            [pair[1] for pair in ranking_pairs], device=decision_logits.device
        )
        ranking = F.softplus(
            -(decision_logits[in_positions] - decision_logits[out_positions])
        ).mean()
    else:
        ranking = decision_logits.sum() * 0.0
    if rationale_logits.shape[1] and bool(rationale_mask.any()):
        auxiliary = F.binary_cross_entropy_with_logits(
            rationale_logits[rationale_mask], rationale_targets[rationale_mask]
        )
    else:
        auxiliary = decision_logits.sum() * 0.0
    total = classification + lambda_rank * ranking + lambda_aux * auxiliary
    return LossBreakdown(total, classification, ranking, auxiliary)


class SetFitEncoder(Protocol):
    def adapt(
        self,
        population: Sequence[PersonalizedCase],
        pairs: SetFitPairs,
        *,
        epochs: int,
        seed: int,
        condition: str,
    ) -> None: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerSetFitEncoder:
    """SetFit-style contrastive adapter around a pinned Sentence Transformer."""

    def __init__(self, model: str, revision: str, device: str = "auto") -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            model,
            revision=revision,
            device=None if device == "auto" else device,
            trust_remote_code=False,
        )

    def adapt(
        self,
        population: Sequence[PersonalizedCase],
        pairs: SetFitPairs,
        *,
        epochs: int,
        seed: int,
        condition: str,
    ) -> None:
        from sentence_transformers import InputExample, losses
        from torch.utils.data import DataLoader

        torch.manual_seed(seed)
        examples = [
            InputExample(
                texts=[
                    serialize_case(population[pair.left_index], condition),
                    serialize_case(population[pair.right_index], condition),
                ],
                label=float(pair.similar),
            )
            for pair in pairs.contrastive
        ]
        if not examples:
            return
        loader = DataLoader(examples, batch_size=min(16, len(examples)), shuffle=True)
        self.model.fit(
            train_objectives=[(loader, losses.CosineSimilarityLoss(self.model))],
            epochs=epochs,
            warmup_steps=0,
            show_progress_bar=False,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ), dtype=float)


class _MultiTaskHead(nn.Module):
    def __init__(
        self, text_dimensions: int, structured_dimensions: int,
        vc_count: int, rationale_count: int,
    ) -> None:
        super().__init__()
        self.identity = nn.Embedding(vc_count, 8)
        total = text_dimensions + structured_dimensions + 8
        self.shared = nn.Sequential(
            nn.Linear(total, 32), nn.ReLU(), nn.Dropout(0.15)
        )
        self.decision = nn.Linear(32, 1)
        self.rationales = nn.Linear(32, rationale_count)

    def forward(
        self, text: torch.Tensor, structured: torch.Tensor, vc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(torch.cat((text, structured, self.identity(vc)), dim=1))
        return self.decision(hidden).squeeze(1), self.rationales(hidden)


def _config(epochs: int, lambda_rank: float, lambda_aux: float) -> str:
    return (
        f"epochs={epochs};lambda_rank={lambda_rank:g};lambda_aux={lambda_aux:g}"
    )


def _parse_config(config: str) -> tuple[int, float, float]:
    values = dict(item.split("=", 1) for item in config.split(";"))
    return int(values["epochs"]), float(values["lambda_rank"]), float(values["lambda_aux"])


def _fit_score(
    population: Sequence[PersonalizedCase],
    views: Sequence[RawFeatureViews],
    fit_indices: Sequence[int],
    predict_indices: Sequence[int],
    *,
    condition: str,
    config: str,
    max_pairs: int,
    taxonomy_labels: Sequence[str],
    seed: int,
    encoder_factory: Callable[[], SetFitEncoder],
    device: str,
    adapted_text_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    epochs, lambda_rank, lambda_aux = _parse_config(config)
    pairs = build_setfit_pairs(
        population, fit_indices, max_pairs=max_pairs, seed=seed
    )
    all_indices = tuple(fit_indices) + tuple(predict_indices)
    if adapted_text_matrix is None:
        encoder = encoder_factory()
        encoder.adapt(
            population, pairs, epochs=epochs, seed=seed, condition=condition
        )
        texts = [serialize_case(population[index], condition) for index in all_indices]
        text_matrix = np.asarray(encoder.encode(texts), dtype=float)
    else:
        text_matrix = np.asarray(adapted_text_matrix, dtype=float)
    if text_matrix.ndim != 2 or len(text_matrix) != len(all_indices):
        raise ValueError("SetFit encoder returned an invalid matrix")
    fit_count = len(fit_indices)
    preprocessor = fit_ablation_preprocessor(
        views,
        train_indices=fit_indices,
        condition=condition,
        pca_components=8,
        maximum_features=512,
    )
    structured_fit = preprocessor.transform(views, fit_indices)
    structured_predict = preprocessor.transform(views, predict_indices)
    vcs = tuple(sorted({population[index].vc_slug for index in fit_indices}))
    vc_to_index = {vc: index for index, vc in enumerate(vcs)}
    if any(population[index].vc_slug not in vc_to_index for index in predict_indices):
        raise ValueError("held investor was absent from SetFit training")
    local = {index: position for position, index in enumerate(fit_indices)}
    ranking_pairs = tuple(
        (local[pair.in_index], local[pair.out_index]) for pair in pairs.ranking
    )
    y = np.asarray([population[index].target for index in fit_indices], dtype=float)
    rationale_targets = np.asarray([
        [float(population[index].actual_rationale_targets.get(label, 0)) for label in taxonomy_labels]
        for index in fit_indices
    ], dtype=float)
    rationale_mask = np.asarray([
        population[index].actual_rationale_available for index in fit_indices
    ], dtype=bool)
    selected_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else device
        if device != "auto" else "cpu"
    )
    torch.manual_seed(seed)
    model = _MultiTaskHead(
        text_matrix.shape[1], structured_fit.shape[1], len(vcs), len(taxonomy_labels)
    ).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    text_fit = torch.as_tensor(text_matrix[:fit_count], dtype=torch.float32, device=selected_device)
    structured_tensor = torch.as_tensor(structured_fit, dtype=torch.float32, device=selected_device)
    vc_tensor = torch.as_tensor(
        [vc_to_index[population[index].vc_slug] for index in fit_indices],
        dtype=torch.long, device=selected_device,
    )
    decision_tensor = torch.as_tensor(y, dtype=torch.float32, device=selected_device)
    rationale_tensor = torch.as_tensor(rationale_targets, dtype=torch.float32, device=selected_device)
    mask_tensor = torch.as_tensor(rationale_mask, dtype=torch.bool, device=selected_device)
    positive_weight = float((len(y) - y.sum()) / max(1.0, y.sum()))
    model.train()
    for _ in range(max(15, epochs * 15)):
        optimizer.zero_grad()
        decision_logits, rationale_logits = model(text_fit, structured_tensor, vc_tensor)
        loss = multitask_loss(
            decision_logits=decision_logits,
            decisions=decision_tensor,
            ranking_pairs=ranking_pairs,
            rationale_logits=rationale_logits,
            rationale_targets=rationale_tensor,
            rationale_mask=mask_tensor,
            lambda_rank=lambda_rank,
            lambda_aux=lambda_aux,
            positive_weight=positive_weight,
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model(
            torch.as_tensor(text_matrix[fit_count:], dtype=torch.float32, device=selected_device),
            torch.as_tensor(structured_predict, dtype=torch.float32, device=selected_device),
            torch.as_tensor(
                [vc_to_index[population[index].vc_slug] for index in predict_indices],
                dtype=torch.long, device=selected_device,
            ),
        )
        scores = torch.sigmoid(logits).cpu().numpy()
    return np.asarray(scores, dtype=float), preprocessor.feature_names


def _adapted_text_matrix(
    population: Sequence[PersonalizedCase],
    fit_indices: Sequence[int],
    predict_indices: Sequence[int],
    *,
    condition: str,
    epochs: int,
    max_pairs: int,
    seed: int,
    encoder_factory: Callable[[], SetFitEncoder],
) -> np.ndarray:
    pairs = build_setfit_pairs(population, fit_indices, max_pairs=max_pairs, seed=seed)
    encoder = encoder_factory()
    encoder.adapt(
        population, pairs, epochs=epochs, seed=seed, condition=condition
    )
    indices = tuple(fit_indices) + tuple(predict_indices)
    return np.asarray(encoder.encode([
        serialize_case(population[index], condition) for index in indices
    ]), dtype=float)


def run_setfit_fold(
    population: Sequence[PersonalizedCase],
    views: Sequence[RawFeatureViews],
    fold: EpisodeFold,
    *,
    condition: str,
    epochs: Sequence[int],
    lambda_rank: Sequence[float],
    lambda_aux: Sequence[float],
    max_pairs: int,
    taxonomy_labels: Sequence[str],
    seed: int,
    encoder_factory: Callable[[], SetFitEncoder],
    device: str = "cpu",
) -> list[PersonalizedPrediction]:
    """Nested SetFit-style multi-task selection for one held episode."""
    started = time.monotonic()
    position = {index: offset for offset, index in enumerate(fold.train_indices)}
    targets = np.asarray([population[index].target for index in fold.train_indices], dtype=int)
    vcs = np.asarray([population[index].vc_slug for index in fold.train_indices], dtype=object)
    candidates = []
    adapted_inner: dict[tuple[int, int], np.ndarray] = {}
    for epoch_count in epochs:
        for split_index, (inner_train, inner_valid) in enumerate(fold.inner_splits):
            adapted_inner[(epoch_count, split_index)] = _adapted_text_matrix(
                population, inner_train, inner_valid, condition=condition,
                epochs=epoch_count, max_pairs=max_pairs, seed=seed + split_index,
                encoder_factory=encoder_factory,
            )
        for rank_weight in lambda_rank:
            for aux_weight in lambda_aux:
                config = _config(epoch_count, rank_weight, aux_weight)
                scores = np.zeros(len(fold.train_indices), dtype=float)
                for split_index, (inner_train, inner_valid) in enumerate(fold.inner_splits):
                    predicted, _ = _fit_score(
                        population, views, inner_train, inner_valid,
                        condition=condition, config=config, max_pairs=max_pairs,
                        taxonomy_labels=taxonomy_labels,
                        seed=seed + split_index, encoder_factory=encoder_factory,
                        device=device,
                        adapted_text_matrix=adapted_inner[(epoch_count, split_index)],
                    )
                    for index, score in zip(inner_valid, predicted, strict=True):
                        scores[position[index]] = score
                candidates.append(TabPFNCandidate(config, scores))
    selected = select_tabpfn_heads(targets, vcs, candidates)
    candidate_by_config = {candidate.config: candidate.scores for candidate in candidates}
    classification_inner_oof = tuple(
        (
            population[index].vc_slug,
            population[index].episode_slug,
            float(candidate_by_config[selected.classification_config][position[index]]),
        )
        for index in fold.train_indices
    )
    ranking_inner_oof = tuple(
        (
            population[index].vc_slug,
            population[index].episode_slug,
            float(candidate_by_config[selected.ranking_config][position[index]]),
        )
        for index in fold.train_indices
    )
    classification_scores, names = _fit_score(
        population, views, fold.train_indices, fold.test_indices,
        condition=condition, config=selected.classification_config,
        max_pairs=max_pairs, taxonomy_labels=taxonomy_labels,
        seed=seed + 50000, encoder_factory=encoder_factory, device=device,
    )
    ranking_scores, _ = _fit_score(
        population, views, fold.train_indices, fold.test_indices,
        condition=condition, config=selected.ranking_config,
        max_pairs=max_pairs, taxonomy_labels=taxonomy_labels,
        seed=seed + 60000, encoder_factory=encoder_factory, device=device,
    )
    digest = sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()
    training_episodes = tuple(sorted({population[index].episode_slug for index in fold.train_indices}))
    validation = tuple(tuple(sorted({population[index].episode_slug for index in valid})) for _, valid in fold.inner_splits)
    runtime = time.monotonic() - started
    return [PersonalizedPrediction(
        family="setfit", condition=condition,
        vc_slug=population[index].vc_slug, vc_name=population[index].vc_name,
        episode_slug=population[index].episode_slug, target=population[index].target,
        classification_score=float(classification_score), ranking_score=float(ranking_score),
        predicted=int(classification_score >= selected.classification_threshold),
        classification_threshold=selected.classification_threshold,
        classification_config=selected.classification_config,
        ranking_config=selected.ranking_config,
        training_episodes=training_episodes,
        inner_validation_episodes=validation,
        feature_schema_sha256=digest,
        runtime_seconds=runtime,
        device=device,
        status="complete",
        classification_inner_oof=classification_inner_oof,
        ranking_inner_oof=ranking_inner_oof,
    ) for index, classification_score, ranking_score in zip(
        fold.test_indices, classification_scores, ranking_scores, strict=True
    )]
