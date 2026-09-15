"""Label-blind multimodal base features for personalized decision models."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

import numpy as np

from .personalized_cases import PersonalizedCase
from .personalized_embedding_cache import EmbeddingCache


@dataclass(frozen=True)
class TextChunk:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class TextViewEmbeddings:
    pooled_mean: np.ndarray
    pooled_max: np.ndarray
    chunks: tuple[str, ...]
    cache_keys: tuple[str, ...]


@dataclass(frozen=True)
class StructuredFeatureViews:
    pitch: Mapping[str, float]
    phase1: Mapping[str, float]
    phase2: Mapping[str, float]


def chunk_text(
    text: str, *, max_chars: int = 6000, overlap_chars: int = 500
) -> tuple[TextChunk, ...]:
    """Split text deterministically without truncation and with bounded overlap."""
    if not text:
        raise ValueError("cannot chunk empty text")
    if max_chars < 32 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("invalid text chunk bounds")
    chunks: list[TextChunk] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + max_chars)
        end = hard_end
        if hard_end < len(text):
            lower = start + max_chars // 2
            candidates = (
                text.rfind("\n\n", lower, hard_end),
                text.rfind(". ", lower, hard_end),
                text.rfind(" ", lower, hard_end),
            )
            boundary = max(candidates)
            if boundary >= lower:
                end = boundary + (2 if text[boundary:boundary + 2] in {"\n\n", ". "} else 1)
        chunks.append(TextChunk(start=start, end=end, text=text[start:end]))
        if end == len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return tuple(chunks)


def embed_text_view(
    text: str,
    *,
    source_sha256: str,
    view: str,
    provider: object,
    cache: EmbeddingCache,
    max_chars: int = 6000,
    overlap_chars: int = 500,
) -> TextViewEmbeddings:
    chunks = chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
    metadata = {
        "schema": "personalized-text-embedding-cache-v1",
        "source_sha256": source_sha256,
        "view": view,
        "max_chars": max_chars,
        "overlap_chars": overlap_chars,
        "provider": dict(getattr(provider, "metadata")),
        "chunk_bounds": [[chunk.start, chunk.end] for chunk in chunks],
    }
    key = cache.key(metadata)
    matrix = cache.load(key)
    if matrix is None:
        matrix = np.asarray(
            provider.embed_documents([chunk.text for chunk in chunks]), dtype=float
        )
        if matrix.ndim != 2 or len(matrix) != len(chunks):
            raise ValueError("embedding provider returned an invalid chunk matrix")
        cache.store(key, matrix, metadata)
    return TextViewEmbeddings(
        pooled_mean=np.mean(matrix, axis=0),
        pooled_max=np.max(matrix, axis=0),
        chunks=tuple(chunk.text for chunk in chunks),
        cache_keys=(key,),
    )


_TOKEN = re.compile(r"\b[\w'-]+\b")
_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:k|m|b|million|billion)?\b", re.I)
_PERCENT = re.compile(r"\b\d+(?:\.\d+)?\s*%")
_REVENUE = re.compile(r"\$\s*\d|\b(?:arr|mrr|revenue|sales)\b", re.I)
_TRACTION = re.compile(r"\b(?:customers?|users?|revenue|sales|arr|mrr|growth|growing|retention|pilots?)\b", re.I)
_FOUNDER = re.compile(r"\b(?:founder (?:built|founded)|serial founder|previous company|prior startup)\b", re.I)
_MARKET = re.compile(r"\b(?:market|tam|sam|som|industry|category)\b", re.I)
_COMPETITOR = re.compile(r"\b(?:competitors?|competition|incumbents?|alternative)\b", re.I)
_UNKNOWN = re.compile(r"\b(?:unknown|unanswered|not provided|not disclosed|unclear)\b", re.I)


def pitch_indicator_features(text: str) -> dict[str, float]:
    """Return deterministic descriptive indicators without outcome information."""
    return {
        "pitch__character_count": float(len(text)),
        "pitch__token_count": float(len(_TOKEN.findall(text))),
        "pitch__paragraph_count": float(len([row for row in re.split(r"\n\s*\n", text) if row.strip()])),
        "pitch__numeric_mentions": float(len(_NUMBER.findall(text))),
        "pitch__percentage_mentions": float(len(_PERCENT.findall(text))),
        "pitch__currency_or_revenue_mentions": float(len(_REVENUE.findall(text))),
        "pitch__traction_mentions": float(len(_TRACTION.findall(text))),
        "pitch__founder_history_mentions": float(len(_FOUNDER.findall(text))),
        "pitch__market_mentions": float(len(_MARKET.findall(text))),
        "pitch__competitor_mentions": float(len(_COMPETITOR.findall(text))),
        "pitch__unknown_mentions": float(len(_UNKNOWN.findall(text))),
    }


def _safe_copy(features: Mapping[str, float], prefix: str) -> dict[str, float]:
    result = {}
    for name, value in features.items():
        lowered = name.lower()
        if "target" in lowered or "actual" in lowered or name.startswith("vc__"):
            continue
        result[f"{prefix}{name}"] = float(value)
    return result


def build_structured_feature_views(
    case: PersonalizedCase, *, taxonomy_labels: Sequence[str]
) -> StructuredFeatureViews:
    """Materialize fixed-schema pitch, Phase 1, and Phase 2 feature families."""
    phase1 = _safe_copy(case.phase1_features, "p1_raw__")
    for label in taxonomy_labels:
        directions = {}
        saliences = {}
        for direction in ("positive", "negative", "neutral"):
            value = sum(
                float(case.phase1_features.get(
                    f"label_direction__{label}__{direction}__{salience}__count", 0.0
                ))
                for salience in ("primary", "secondary")
            )
            directions[direction] = value
            phase1[f"p1__{label}__direction_{direction}"] = float(value > 0)
        for salience in ("primary", "secondary"):
            value = sum(
                float(case.phase1_features.get(
                    f"label_direction__{label}__{direction}__{salience}__count", 0.0
                ))
                for direction in ("positive", "negative", "neutral")
            )
            saliences[salience] = value
            phase1[f"p1__{label}__salience_{salience}"] = float(value > 0)
        signed = float(case.phase1_features.get(
            f"label__{label}__signed_confidence", 0.0
        ))
        confidence = float(case.phase1_features.get(
            f"label__{label}__confidence_sum", abs(signed)
        ))
        phase1[f"p1__{label}__activated"] = float(
            confidence > 0 or any(directions.values()) or any(saliences.values())
        )
        phase1[f"p1__{label}__confidence"] = confidence
        phase1[f"p1__{label}__signed_confidence"] = signed
        for source in ("pitch", "wiki", "historical", "portfolio"):
            phase1[f"p1__{label}__{source}_evidence_count"] = float(
                case.phase1_features.get(
                    f"label__{label}__{source}_evidence_count", 0.0
                )
            )

    phase2 = _safe_copy(case.phase2_features, "p2__")
    required = (
        "decision_in", "investment_likelihood", "decision_confidence",
        "review_priority_score", "phase2_provisional",
    )
    for name in required:
        phase2[f"p2_missing__{name}"] = float(name not in case.phase2_features)
        phase2.setdefault(f"p2__{name}", 0.0)
    return StructuredFeatureViews(
        pitch=pitch_indicator_features(case.pitch_text),
        phase1=phase1,
        phase2=phase2,
    )
