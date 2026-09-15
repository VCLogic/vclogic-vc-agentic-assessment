"""Deterministic, leakage-safe source packet assembly."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tomllib
from typing import Iterable

from config import BaselineConfig
from prompt import BaselineContext, PrecedentContext, WikiDocument
from vc_clone_graph.firewall import sha256_file, verify_package
from vc_clone_graph.precedents import PrecedentCorpus
from vc_clone_graph.providers.sentence_transformers import SentenceTransformerEmbeddingProvider


def select_unique_precedents(hits: Iterable, target_slug: str, count: int):
    selected, seen = [], {target_slug}
    for hit in hits:
        if hit.episode_slug in seen:
            continue
        seen.add(hit.episode_slug)
        selected.append(hit)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"semantic retrieval returned only {len(selected)} unique precedents")
    return tuple(selected)


def _embedder(config: BaselineConfig):
    row = config.embedding
    return SentenceTransformerEmbeddingProvider(
        row.model, revision=row.revision, device=row.device,
        batch_size=row.batch_size, normalize=row.normalize,
        document_prefix=row.document_prefix, query_prefix=row.query_prefix,
    )


def _taxonomy(path: Path) -> tuple[dict, ...]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("taxonomy must be a nonempty list")
    labels = [row.get("label") for row in rows if isinstance(row, dict)]
    if len(labels) != len(rows) or len(set(labels)) != len(labels):
        raise ValueError("taxonomy labels must be present and unique")
    return tuple(rows)


def assemble_context(config: BaselineConfig, episode_slug: str):
    root = Path(config.input_root)
    package = verify_package(root, config.vc_slug, episode_slug, taxonomy_path=config.taxonomy_path)
    if package.precedents is None:
        raise ValueError("verified package has no precedent corpus")
    embedder = _embedder(config)
    index_path = root / "indexes" / f"{config.vc_slug}.precedents.json"
    corpus = PrecedentCorpus.load(
        index_path, package.precedents, embedder,
        require_complete_embeddings=config.embedding.require_complete_index,
    ).for_target(episode_slug)
    pitch = package.pitch.read_text(encoding="utf-8")
    search = corpus.search(
        pitch, top_k=config.retrieval.candidate_pool_k,
        query_id="baseline-target-pitch", selection_policy="semantic",
    )
    hits = select_unique_precedents(search.hits, episode_slug, config.retrieval.top_k)

    precedent_rows = []
    selected_manifest = []
    for hit in hits:
        transcript = corpus.open_transcript(hit.episode_slug)
        decision = corpus.open_decision(hit.episode_slug)
        transcript_text = "\n".join(f"{turn.speaker}: {turn.text}" for turn in transcript.turns)
        evidence_text = "\n".join(span.text for span in decision.evidence)
        precedent_rows.append(PrecedentContext(
            episode_slug=hit.episode_slug, transcript=transcript_text,
            observed_decision=decision.status, decision_evidence=evidence_text,
            similarity=hit.score,
        ))
        selected_manifest.append({
            "episode_slug": hit.episode_slug, "similarity": hit.score,
            "source_sha256": transcript.source_sha256,
            "decision_status": decision.status,
            "transcript_sha256": sha256(transcript_text.encode()).hexdigest(),
            "decision_evidence_sha256": sha256(evidence_text.encode()).hexdigest(),
        })

    wiki_paths = tuple(sorted(path for path in package.wiki.rglob("*") if path.is_file()))
    wiki = tuple(WikiDocument(path.relative_to(package.wiki).as_posix(), path.read_text(encoding="utf-8")) for path in wiki_paths)
    taxonomy = _taxonomy(package.taxonomy)
    with package.registry.open("rb") as stream:
        investor_name = tomllib.load(stream)["display_name"]
    context = BaselineContext(
        episode_slug=episode_slug, investor_name=investor_name,
        current_pitch=pitch, wiki=wiki, taxonomy=taxonomy,
        precedents=tuple(precedent_rows),
    )
    metadata = corpus.embedding_index or {}
    manifest = {
        "schema": "one-shot-baseline-context-v1",
        "target_episode_slug": episode_slug,
        "pitch_sha256": sha256_file(package.pitch),
        "wiki_sha256s": {doc.path: sha256(doc.content.encode()).hexdigest() for doc in wiki},
        "taxonomy_sha256": sha256_file(package.taxonomy),
        "filtered_precedent_manifest": corpus.filtered_manifest().model_dump(mode="json"),
        "selected_precedents": selected_manifest,
        "embedding": {**metadata, "complete": metadata.get("coverage") == 1.0},
    }
    return context, manifest
