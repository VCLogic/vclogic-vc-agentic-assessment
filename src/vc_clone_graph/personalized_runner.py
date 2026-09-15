"""Resumable fold-level orchestration for personalized model families."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .personalized_cases import PersonalizedCase
from .personalized_config import PersonalizedConfig
from .personalized_embedding_cache import EmbeddingCache
from .personalized_features import (
    build_structured_feature_views,
    embed_text_view,
)
from .personalized_folds import EpisodeFold
from .personalized_matrices import RawFeatureViews, cross_view_interactions
from .personalized_rationale_teacher import (
    cross_fit_rationale_teacher,
    teacher_comparison_features,
)
from .personalized_similarity import similarity_features
from .providers.sentence_transformers import SentenceTransformerEmbeddingProvider


@dataclass(frozen=True)
class FoldRunResult:
    status: str
    predictions: Sequence[object]
    reason: str | None = None


FoldRunner = Callable[[str, EpisodeFold], Sequence[object] | FoldRunResult]


def implementation_digest(paths: Sequence[Path], *, root: Path) -> str:
    """Hash implementation path identities and bytes in stable order."""
    base = Path(root).resolve()
    digest = sha256()
    resolved = sorted((Path(path).resolve() for path in paths), key=lambda path: path.as_posix())
    for path in resolved:
        try:
            relative = path.relative_to(base).as_posix()
        except ValueError as exc:
            raise ValueError(f"implementation path is outside digest root: {path}") from exc
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedEmbeddings:
    pitch_model: np.ndarray
    phase1_model: np.ndarray
    phase2_model: np.ndarray
    wiki_model: np.ndarray
    pitch_similarity: np.ndarray
    phase1_similarity: np.ndarray
    phase2_similarity: np.ndarray
    wiki_similarity_by_vc: Mapping[str, np.ndarray]
    portfolio_similarity_by_vc: Mapping[str, np.ndarray]
    embedding_metadata: Mapping[str, object]


def model_family_for_stage(stage: str) -> str:
    if stage in {"tabpfn", "setfit", "ensemble", "graph"}:
        return stage
    raise ValueError(f"{stage} is not a directly fitted model stage")


def _embed_case_view(
    text: str,
    digest: str,
    view: str,
    provider: SentenceTransformerEmbeddingProvider,
    cache: EmbeddingCache,
) -> tuple[np.ndarray, np.ndarray]:
    embedded = embed_text_view(
        text,
        source_sha256=digest,
        view=view,
        provider=provider,
        cache=cache,
    )
    return (
        np.concatenate((embedded.pooled_mean, embedded.pooled_max)),
        embedded.pooled_mean,
    )


def _portfolio_embeddings(
    cases: Sequence[PersonalizedCase],
    input_root: Path,
    config: PersonalizedConfig,
    dimensions: int,
) -> dict[str, np.ndarray]:
    result = {}
    for case in cases:
        if case.vc_slug in result:
            continue
        root = (
            Path(input_root) / "data/investors" / case.vc_input_slug / "portfolio-memory"
        )
        metadata = json.loads((root / "embedding-index.json").read_text(encoding="utf-8"))
        identity = metadata.get("embedding", {})
        if (
            identity.get("model") != config.embedding.model
            or identity.get("revision") != config.embedding.revision
            or bool(identity.get("normalize")) != config.embedding.normalize
        ):
            raise ValueError(f"portfolio embedding identity mismatch: {case.vc_slug}")
        entities = json.loads((root / "company-index.json").read_text(encoding="utf-8"))
        vectors = [row.get("embedding") for row in entities if row.get("embedding") is not None]
        matrix = np.asarray(vectors, dtype=float) if vectors else np.empty((0, dimensions))
        if matrix.ndim != 2 or matrix.shape[1] != dimensions or not np.isfinite(matrix).all():
            raise ValueError(f"invalid portfolio embeddings: {case.vc_slug}")
        result[case.vc_slug] = matrix
    return result


def prepare_embeddings(
    cases: Sequence[PersonalizedCase],
    config: PersonalizedConfig,
    *,
    input_root: Path,
    cache_root: Path,
) -> PreparedEmbeddings:
    """Embed frozen deployable text views once under source-bound cache keys."""
    provider = SentenceTransformerEmbeddingProvider(
        config.embedding.model,
        revision=config.embedding.revision,
        device=config.embedding.device,
        batch_size=config.embedding.batch_size,
        normalize=config.embedding.normalize,
        document_prefix=config.embedding.document_prefix,
        query_prefix=config.embedding.query_prefix,
    )
    cache = EmbeddingCache(cache_root)
    pitch_model, pitch_mean = [], []
    phase1_model, phase1_mean = [], []
    phase2_model, phase2_mean = [], []
    wiki_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    wiki_model, wiki_mean = [], []
    for case in cases:
        model, mean = _embed_case_view(
            case.pitch_text, case.source_hashes["pitch"], "pitch", provider, cache
        )
        pitch_model.append(model)
        pitch_mean.append(mean)
        model, mean = _embed_case_view(
            case.phase1_text, case.source_hashes["phase1"], "phase1", provider, cache
        )
        phase1_model.append(model)
        phase1_mean.append(mean)
        model, mean = _embed_case_view(
            case.phase2_text, case.source_hashes["phase2"], "phase2", provider, cache
        )
        phase2_model.append(model)
        phase2_mean.append(mean)
        if case.vc_slug not in wiki_cache:
            wiki_cache[case.vc_slug] = _embed_case_view(
                case.wiki_text, case.source_hashes["wiki"], "wiki", provider, cache
            )
        wiki_model.append(wiki_cache[case.vc_slug][0])
        wiki_mean.append(wiki_cache[case.vc_slug][1])
    similarity_dimensions = len(pitch_mean[0])
    return PreparedEmbeddings(
        pitch_model=np.asarray(pitch_model),
        phase1_model=np.asarray(phase1_model),
        phase2_model=np.asarray(phase2_model),
        wiki_model=np.asarray(wiki_model),
        pitch_similarity=np.asarray(pitch_mean),
        phase1_similarity=np.asarray(phase1_mean),
        phase2_similarity=np.asarray(phase2_mean),
        wiki_similarity_by_vc={vc: value[1] for vc, value in wiki_cache.items()},
        portfolio_similarity_by_vc=_portfolio_embeddings(
            cases, input_root, config, similarity_dimensions
        ),
        embedding_metadata=provider.metadata,
    )


def _fixed_projection(vector: np.ndarray, *, seed: int, dimensions: int = 24) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    random = np.random.default_rng(seed)
    projection = random.normal(
        0.0, 1.0 / np.sqrt(max(1, len(values))), size=(len(values), dimensions)
    )
    return values @ projection


def build_fold_views(
    cases: Sequence[PersonalizedCase],
    embeddings: PreparedEmbeddings,
    fold: EpisodeFold,
    taxonomy_labels: Sequence[str],
    *,
    seed: int,
) -> list[RawFeatureViews]:
    """Build cross-fitted teacher/history views for one outer episode fold."""
    structured = [
        build_structured_feature_views(case, taxonomy_labels=taxonomy_labels)
        for case in cases
    ]
    similarity_rows: dict[int, Mapping[str, float]] = {}
    for inner_train, inner_valid in fold.inner_splits:
        inner = similarity_features(
            cases,
            embeddings.pitch_similarity,
            wiki_embeddings=embeddings.wiki_similarity_by_vc,
            portfolio_embeddings=embeddings.portfolio_similarity_by_vc,
            reference_indices=inner_train,
            query_indices=inner_valid,
        )
        similarity_rows.update(inner.features_by_index)
    held_similarity = similarity_features(
        cases,
        embeddings.pitch_similarity,
        wiki_embeddings=embeddings.wiki_similarity_by_vc,
        portfolio_embeddings=embeddings.portfolio_similarity_by_vc,
        reference_indices=fold.train_indices,
        query_indices=fold.test_indices,
    )
    similarity_rows.update(held_similarity.features_by_index)
    if set(similarity_rows) != set(range(len(cases))):
        raise ValueError("cross-fitted similarity rows do not cover the outer fold")
    vc_slugs = tuple(sorted({case.vc_slug for case in cases}))
    teacher_rows = []
    for index, case in enumerate(cases):
        row = {
            **structured[index].pitch,
            **structured[index].phase1,
            **structured[index].phase2,
            **{f"vc_identity__{vc}": float(case.vc_slug == vc) for vc in vc_slugs},
        }
        for view_index, vector in enumerate((
            embeddings.pitch_similarity[index],
            embeddings.phase1_similarity[index],
            embeddings.phase2_similarity[index],
            embeddings.wiki_similarity_by_vc[case.vc_slug],
        )):
            for component, value in enumerate(
                _fixed_projection(vector, seed=seed + view_index), start=1
            ):
                row[f"semantic_view_{view_index}__projection_{component}"] = float(value)
        teacher_rows.append(row)
    train_teacher, held_teacher = cross_fit_rationale_teacher(
        cases, fold, teacher_rows, taxonomy_labels, c_value=1.0, seed=seed
    )
    teacher_by_index = {
        index: output for index, output in zip(fold.train_indices, train_teacher, strict=True)
    }
    teacher_by_index.update({
        index: output for index, output in zip(fold.test_indices, held_teacher, strict=True)
    })
    result = []
    for index, case in enumerate(cases):
        phase1_signed_mass = sum(
            float(value)
            for name, value in case.phase1_features.items()
            if name.endswith("__signed_confidence")
        )
        likelihood = float(case.phase2_features.get("investment_likelihood", 0.0))
        decision_in = float(case.phase2_features.get("decision_in", 0.0))
        interactions = cross_view_interactions(
            embeddings.pitch_similarity[index],
            embeddings.phase1_similarity[index],
            embeddings.phase2_similarity[index],
            embeddings.wiki_similarity_by_vc[case.vc_slug],
            phase1_signed_mass=phase1_signed_mass,
            phase2_likelihood=likelihood,
            phase2_decision_in=decision_in,
        )
        identity = {f"vc_identity__{vc}": float(case.vc_slug == vc) for vc in vc_slugs}
        result.append(RawFeatureViews(
            pitch_embedding=embeddings.pitch_model[index],
            phase1_embedding=embeddings.phase1_model[index],
            phase2_embedding=embeddings.phase2_model[index],
            wiki_embedding=embeddings.wiki_model[index],
            pitch_structured=structured[index].pitch,
            phase1_structured=structured[index].phase1,
            phase2_structured=structured[index].phase2,
            investor_structured={
                **identity,
                **similarity_rows[index],
            },
            teacher_structured=teacher_comparison_features(
                teacher_by_index[index], case.phase1_features, taxonomy_labels
            ),
            interaction_structured=interactions,
        ))
    return result


def _write_atomic(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _json_row(value: object) -> Mapping[str, object]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"fold prediction is not serializable: {type(value).__name__}")


def run_checkpointed_stage(
    *,
    family: str,
    conditions: Sequence[str],
    folds: Sequence[EpisodeFold],
    output: Path,
    source_manifest: Mapping[str, object],
    run_fold: FoldRunner,
    max_retries: int,
    resume: bool,
    max_folds: int | None = None,
    dry_run: bool = False,
) -> list[dict[str, object]]:
    """Run atomic family/condition/fold units with one bounded retry."""
    if max_retries not in {0, 1}:
        raise ValueError("max_retries must be zero or one")
    selected_folds = tuple(folds[:max_folds]) if max_folds is not None else tuple(folds)
    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be positive")
    planned = [
        {
            "schema": "personalized-fold-checkpoint-v1",
            "family": family,
            "condition": condition,
            "held_episode": fold.held_episode,
            "status": "planned",
            "attempts": 0,
        }
        for condition in conditions
        for fold in selected_folds
    ]
    if dry_run:
        return planned

    root = Path(output)
    manifest_path = root / "run-manifest.json"
    manifest_payload = {
        "schema": "personalized-run-manifest-v1",
        "sources": dict(source_manifest),
    }
    if manifest_path.exists():
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if observed != manifest_payload:
            raise ValueError("resume source manifest mismatch")
    else:
        _write_atomic(manifest_path, manifest_payload)

    results = []
    for condition in conditions:
        for fold in selected_folds:
            checkpoint = root / family / condition / f"{fold.held_episode}.json"
            if resume and checkpoint.exists():
                results.append(json.loads(checkpoint.read_text(encoding="utf-8")))
                continue
            error = None
            predictions = []
            fold_status = "complete"
            reason = None
            attempts = 0
            for attempt in range(max_retries + 1):
                attempts = attempt + 1
                try:
                    result = run_fold(condition, fold)
                    if isinstance(result, FoldRunResult):
                        if result.status not in {"complete", "skipped"}:
                            raise ValueError(f"invalid fold result status: {result.status}")
                        fold_status = result.status
                        reason = result.reason
                        predictions = [_json_row(row) for row in result.predictions]
                    else:
                        fold_status = "complete"
                        reason = None
                        predictions = [_json_row(row) for row in result]
                    error = None
                    break
                except Exception as exc:  # fold boundary intentionally captures failures
                    error = f"{type(exc).__name__}: {exc}"
            payload = {
                "schema": "personalized-fold-checkpoint-v1",
                "family": family,
                "condition": condition,
                "held_episode": fold.held_episode,
                "status": "failed" if error is not None else fold_status,
                "attempts": attempts,
                "predictions": predictions,
                "error": error,
                "reason": reason,
            }
            _write_atomic(checkpoint, payload)
            results.append(payload)
    return results
