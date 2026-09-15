"""Command-line interface for indexing, running, resuming, and verifying."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import shutil
import sys
import tomllib
from typing import Literal

from langgraph.checkpoint.sqlite import SqliteSaver

from .artifacts import (
    read_verified_frozen,
    verify_frozen,
    verify_phase1_artifacts,
    verify_run_artifacts,
    verify_v3_phase1_replay_source,
    v44_execution_settings_sha256,
    write_json,
)
from .config import EmbeddingSettings, RunConfig, load_config
from .firewall import verify_package
from .graph import VCDecisionWorkflow, WorkflowSettings
from .workflow_v4 import VCDecisionWorkflowV4
from .workflow_v44 import VCDecisionWorkflowV44, _ArtifactStore
from .providers.base import (
    CompositeProvider,
    GenerationRequest,
    GenerationResult,
    Usage,
    embed_documents,
    embed_queries,
    embedding_identity,
)
from .providers.fake import DemoFakeProvider
from .providers.ollama import OllamaProvider
from .providers.openai import OpenAIProvider
from .providers.openrouter import OpenRouterProvider
from .providers.sentence_transformers import SentenceTransformerEmbeddingProvider
from .precedents import PrecedentCorpus
from .portfolio_memory import (
    FilteredPortfolioIndex,
    PortfolioMemoryCorpus,
    PortfolioMemoryIndex,
)
from .retrieval import HybridWikiIndex
from .schemas import (
    Investigation,
    InvestigationV2,
    InvestigationV3,
    validate_investigation_v3,
)
from .schemas_v4 import InvestigationV4, InvestigationV41
from .schemas_v5 import InvestigationV5


class _DemoFakeProviderV44:
    """Deterministic public CLI fixture for the isolated v4.4 Phase 1 contract."""

    model = "deterministic-fixture"

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def phase1_fingerprint(self) -> dict:
        return {
            "schema_version": "phase1-provider-fingerprint-v1",
            "provider": "fake",
            "model": self.model,
        }

    @staticmethod
    def _enum(schema: dict, definition: str, field: str) -> list[str]:
        value = schema["$defs"][definition]["properties"][field]
        candidate = value.get("items", value).get("enum", [])
        return [item for item in candidate if isinstance(item, str)]

    def generate(self, request: GenerationRequest) -> GenerationResult:
        from time import perf_counter

        started = perf_counter()
        self.requests.append(request)
        episode = request.schema["properties"]["episode_slug"]["const"]
        if request.phase in {
            "phase1_claim_extraction",
            "phase1_claim_extraction_repair",
        }:
            pitch_ids = list(
                request.schema["$defs"]["ClaimCoverageV44"]["properties"]
                ["pitch_evidence_id"]["enum"]
            )
            claims = [
                {
                    "claim_id": f"C{position}",
                    "claim_type": "material",
                    "statement": f"Pitch evidence {evidence_id} contains a material claim.",
                    "topic": "founder execution",
                    "decision_relevance": "The claim is material to investor fit.",
                    "pitch_evidence_ids": [evidence_id],
                }
                for position, evidence_id in enumerate(pitch_ids, start=1)
            ]
            parsed = {
                "schema_version": "claim-map-v4.4",
                "episode_slug": episode,
                "material_claims": claims,
                "adverse_claims": [],
                "unanswered_questions": [],
                "claim_coverage": [
                    {
                        "pitch_evidence_id": evidence_id,
                        "claim_id": f"C{position}",
                        "question_id": None,
                        "non_material_justification": None,
                    }
                    for position, evidence_id in enumerate(pitch_ids, start=1)
                ],
            }
        elif request.phase in {
            "phase1_adjudication",
            "phase1_adjudication_repair",
        }:
            labels = self._enum(
                request.schema, "RationaleDispositionV44", "taxonomy_label"
            )
            claim_ids = self._enum(
                request.schema, "RationaleDispositionV44", "claim_ids"
            )
            pitch_ids = self._enum(
                request.schema, "RationaleDispositionV44", "pitch_evidence_ids"
            )
            wiki_ids = self._enum(
                request.schema, "RationaleDispositionV44", "wiki_evidence_ids"
            )
            historical_ids = self._enum(
                request.schema,
                "RationaleDispositionV44",
                "historical_evidence_ids",
            )
            investor_field = (
                {"wiki_evidence_ids": [wiki_ids[0]], "historical_evidence_ids": []}
                if wiki_ids
                else {
                    "wiki_evidence_ids": [],
                    "historical_evidence_ids": [historical_ids[0]],
                }
            )
            parsed = {
                "schema_version": "rationale-adjudication-v4.4",
                "episode_slug": episode,
                "dispositions": [
                    {
                        "taxonomy_label": labels[0],
                        "disposition": "core",
                        "claim_ids": [claim_ids[0]],
                        "question_ids": [],
                        "pitch_evidence_ids": [pitch_ids[0]],
                        **investor_field,
                        "portfolio_disclosure_ids": [],
                        "direction": "positive",
                        "salience": "primary",
                        "confidence": 0.75,
                        "justification": "Accessible investor evidence supports this signal.",
                    }
                ],
                "unmapped_observations": [],
                "constraint_assessments": [],
                "portfolio_overlap_assessments": [],
                "adjudication_status": "valid",
                "validator_findings": [],
                "requested_retrieval_ids": [],
            }
        else:
            raise RuntimeError(f"unsupported v4.4 fake-provider phase: {request.phase}")
        content = json.dumps(parsed, sort_keys=True)
        return GenerationResult(
            parsed=parsed,
            content=content,
            usage=Usage(
                input_tokens=max(1, len(request.prompt) // 4),
                output_tokens=max(1, len(content) // 4),
            ),
            elapsed_seconds=perf_counter() - started,
            raw_metadata={"provider": "fake", "model": self.model},
        )


class _ConfiguredPhase1ProviderV44:
    """Delegate generation while binding the complete CLI execution configuration."""

    __module__ = "vc_clone_graph.cli"

    def __init__(self, delegate, config: RunConfig) -> None:
        self.delegate = delegate
        self.model = config.phase1.model or config.provider.model
        self.provider_kind = config.provider.kind
        self.execution_settings_sha256 = v44_execution_settings_sha256(config)

    def phase1_fingerprint(self) -> dict:
        return {
            "schema_version": "phase1-provider-fingerprint-v1",
            "provider": self.provider_kind,
            "model": self.model,
            "execution_settings_sha256": self.execution_settings_sha256,
        }

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return self.delegate.generate(request)


class _CliDecisionWorkflowV44(VCDecisionWorkflowV44):
    """Add immutable CLI audits using the workflow's own lock and artifact store."""

    @staticmethod
    def _audit_bytes(value: dict) -> bytes:
        return (
            json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

    def bootstrap_cli_audits(
        self,
        artifacts: dict[str, dict],
        *,
        require_existing: bool,
    ) -> None:
        with _ArtifactStore.open(self.settings.run_root) as store:
            self._store = store
            with self._run_lock():
                missing: list[tuple[str, bytes]] = []
                for relative, value in artifacts.items():
                    expected = self._audit_bytes(value)
                    if store.exists(relative):
                        if store.read_bytes(relative) != expected:
                            raise ValueError(
                                f"v4.4 CLI audit artifact mismatch: {relative}"
                            )
                    else:
                        missing.append((relative, expected))
                if require_existing and missing:
                    raise ValueError(
                        "v4.4 CLI audit artifact is missing: "
                        + ", ".join(relative for relative, _ in missing)
                    )
                for relative, raw in missing:
                    store.atomic_write(relative, raw)


def _verify_replay_historical_registry(state: dict, target_slug: str) -> None:
    reads = [
        row for row in state.get("phase1_precedent_reads", [])
        if isinstance(row, dict) and row.get("status") == "ok"
    ]
    opened: dict[str, dict] = {}
    for read in reads:
        if read.get("episode_slug") == target_slug:
            raise ValueError("Phase 1 registry contains a target precedent read")
        nested = read.get("evidence")
        values = nested if isinstance(nested, list) else [nested] if isinstance(nested, dict) else []
        for evidence in values:
            opened[evidence["evidence_id"]] = evidence
    for evidence in state.get("phase1_historical_evidence", []):
        identity = "H-" + sha256(
            f"{evidence['source_sha256']}\0{evidence['turn_start']}\0{evidence['turn_end']}\0{evidence['text']}".encode()
        ).hexdigest()[:20]
        if evidence.get("evidence_id") != identity or opened.get(identity) != evidence:
            raise ValueError("Phase 1 historical registry integrity mismatch")
        if evidence.get("episode_slug") == target_slug:
            raise ValueError("Phase 1 registry contains target evidence")


def _generation_provider(
    config: RunConfig, phase: Literal["phase1", "phase2"] = "phase1"
):
    provider = config.provider
    phase_settings = config.phase1 if phase == "phase1" else config.phase2
    model = phase_settings.model or provider.model
    max_output_tokens = (
        phase_settings.max_output_tokens or provider.max_output_tokens
    )
    if provider.kind == "fake":
        if config.run.contract_version == "v4.4" and phase == "phase1":
            result = _DemoFakeProviderV44()
        else:
            result = DemoFakeProvider()
    elif provider.kind == "ollama":
        if not provider.base_url:
            raise ValueError("Ollama generation requires base_url")
        result = OllamaProvider(
            model,
            provider.embedding_model,
            provider.base_url,
            max_output_tokens=max_output_tokens,
            enable_thinking=provider.enable_thinking,
            context_window=provider.context_window,
            structured_output_mode=provider.structured_output_mode,
        )
    elif provider.kind == "openai":
        result = OpenAIProvider(
            model,
            provider.embedding_model or "text-embedding-3-small",
            max_output_tokens=max_output_tokens,
        )
    else:
        result = OpenRouterProvider(
            model,
            base_url=provider.base_url or "https://openrouter.ai/api/v1",
            api_key_env=provider.api_key_env or "OPENROUTER_API_KEY",
            max_output_tokens=max_output_tokens,
            require_parameters=provider.require_parameters,
            data_collection=provider.data_collection,
            request_timeout_seconds=provider.request_timeout_seconds,
        )
    if config.run.contract_version == "v4.4" and phase == "phase1":
        return _ConfiguredPhase1ProviderV44(result, config)
    return result


def _embedding_provider(config: RunConfig, settings: EmbeddingSettings):
    if settings.kind == "sentence_transformers":
        assert settings.revision is not None
        return SentenceTransformerEmbeddingProvider(
            settings.model,
            revision=settings.revision,
            device=settings.device,
            batch_size=settings.batch_size,
            normalize=settings.normalize,
            document_prefix=settings.document_prefix,
            query_prefix=settings.query_prefix,
        )
    if settings.kind == "ollama":
        if not settings.base_url:
            raise ValueError("Ollama embedding requires base_url")
        return OllamaProvider(
            "embedding-only",
            settings.model,
            settings.base_url,
            embedding_batch_size=settings.batch_size,
        )
    if settings.kind == "openai":
        return OpenAIProvider(config.provider.model, settings.model)
    raise ValueError("OpenRouter embedding is not configured for this workflow")


def _provider(config: RunConfig):
    generator = _generation_provider(config, "phase1")
    if config.embedding is None:
        return generator
    return CompositeProvider(generator, _embedding_provider(config, config.embedding))


def _indexing_provider(config: RunConfig):
    if config.embedding is not None:
        return _embedding_provider(config, config.embedding)
    if config.precedents.enabled:
        return None
    provider = _generation_provider(config, "phase1")
    if isinstance(provider, OpenRouterProvider):
        raise ValueError("OpenRouter generation cannot be used for embeddings")
    return provider


def _index_path(config: RunConfig) -> Path:
    return config.resolve_path(config.run.input_root) / "indexes" / f"{config.run.vc_slug}.json"


def _precedent_index_path(config: RunConfig) -> Path:
    return (
        config.resolve_path(config.run.input_root)
        / "indexes"
        / f"{config.run.vc_slug}.precedents.json"
    )


def _filtered_precedents_for_target(
    canonical: PrecedentCorpus,
    *,
    target_slug: str,
    target_company_aliases: tuple[str, ...],
) -> PrecedentCorpus:
    if canonical.target_slug is not None:
        raise ValueError("canonical precedent index must be unfiltered")
    filtered = canonical.for_target(
        target_slug,
        excluded_aliases=target_company_aliases,
    )
    if any(row.episode_slug == target_slug for row in filtered.list_episodes()):
        raise ValueError("target episode remains accessible after precedent filtering")
    return filtered


def _filtered_portfolio_for_target(
    config: RunConfig, package, embedder
) -> FilteredPortfolioIndex | None:
    if not config.portfolio_memory.enabled:
        return None
    if package.portfolio_memory is None:
        raise ValueError(
            "portfolio memory is enabled but the verified package has no corpus"
        )
    events = package.portfolio_memory / "disclosure-events.jsonl"
    corpus = PortfolioMemoryCorpus.load_events(config.run.vc_slug, events)
    canonical = PortfolioMemoryIndex.load(
        package.portfolio_memory / "embedding-index.json",
        corpus,
        embedder,
        events,
    )
    if config.portfolio_memory.require_complete_embeddings and any(
        vector is None for vector in canonical.event_embeddings.values()
    ):
        raise ValueError("portfolio memory embedding coverage is incomplete")
    return canonical.for_target(config.run.episode_slug)


def _load_identity(package) -> tuple[str, set[str]]:
    with package.registry.open("rb") as stream:
        raw = tomllib.load(stream)
    return raw["display_name"], set(raw["check_tiers"])


def _taxonomy_records(package) -> tuple[dict[str, str], ...]:
    rows = json.loads(package.taxonomy.read_text(encoding="utf-8"))
    if type(rows) is not list:
        raise ValueError("taxonomy must be a list")
    records: list[dict[str, str]] = []
    for row in rows:
        if type(row) is not dict:
            raise ValueError("taxonomy rows must be JSON objects")
        try:
            record = {
                "label": row["label"],
                "definition": row["definition"],
                "coarse_parent": row["coarse_parent"],
            }
        except KeyError as exc:
            raise ValueError("taxonomy row lacks coverage metadata") from exc
        if not all(type(value) is str and value for value in record.values()):
            raise ValueError("taxonomy coverage metadata must be nonempty strings")
        records.append(record)
    labels = [row["label"] for row in records]
    if not labels:
        raise ValueError("taxonomy contains no labels")
    if len(labels) != len(set(labels)):
        raise ValueError("taxonomy labels must be unique")
    return tuple(records)


def _taxonomy_labels(package) -> set[str]:
    return {row["label"] for row in _taxonomy_records(package)}


def _require_v44_embedding_config(config: RunConfig) -> None:
    if config.run.contract_version != "v4.4":
        return
    settings = config.embedding
    if (
        settings is None
        or settings.kind != "sentence_transformers"
        or not settings.revision
    ):
        raise ValueError(
            "v4.4 requires a pinned local sentence_transformers embedding index"
        )


def _preflight_v44_taxonomy_embeddings(config: RunConfig, package, embedder) -> dict:
    """Exercise both sides of the exact local embedding contract used at runtime."""
    if config.run.contract_version != "v4.4":
        return {}
    assert config.embedding is not None
    records = _taxonomy_records(package)
    documents = [
        f"{row['label']} {row['coarse_parent']} {row['definition']}"
        for row in records
    ]
    queries = [f"{row['label']} {row['definition']}" for row in records]

    def validated(value, *, expected: int, dimension: int | None = None):
        if not isinstance(value, (list, tuple)) or len(value) != expected:
            raise ValueError("taxonomy embedding preflight returned the wrong vector count")
        result: list[list[float]] = []
        for vector in value:
            if not isinstance(vector, (list, tuple)) or not vector:
                raise ValueError("taxonomy embedding preflight returned an empty vector")
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not isfinite(float(item))
                for item in vector
            ):
                raise ValueError("taxonomy embedding preflight returned a non-finite vector")
            row = [float(item) for item in vector]
            if dimension is not None and len(row) != dimension:
                raise ValueError("taxonomy document/query embedding dimensions differ")
            result.append(row)
        first_dimension = len(result[0])
        if any(len(row) != first_dimension for row in result):
            raise ValueError("taxonomy embedding dimensions are inconsistent")
        return result

    document_vectors = validated(
        embed_documents(embedder, documents), expected=len(documents)
    )
    query_vectors = validated(
        embed_queries(embedder, queries),
        expected=len(queries),
        dimension=len(document_vectors[0]),
    )
    identity = embedding_identity(embedder)
    if (
        identity.get("model") != config.embedding.model
        or identity.get("revision") != config.embedding.revision
    ):
        raise ValueError("taxonomy embedding preflight identity does not match config")
    return {
        "model": identity["model"],
        "revision": identity["revision"],
        "document_count": len(document_vectors),
        "query_count": len(query_vectors),
        "dimension": len(document_vectors[0]),
    }


def _workflow(
    config: RunConfig,
    phase1_provider,
    phase2_provider,
    index,
    package,
    checkpointer,
    *,
    sanitization_findings: tuple[str, ...] = (),
    precedent_corpus: PrecedentCorpus | None = None,
    precedent_manifest_sha256: str | None = None,
    portfolio_index: FilteredPortfolioIndex | None = None,
) -> VCDecisionWorkflow | VCDecisionWorkflowV4 | VCDecisionWorkflowV44:
    investor_name, check_tiers = _load_identity(package)
    run_root = config.resolve_path(config.run.output_root) / config.run.episode_slug
    taxonomy_records = _taxonomy_records(package)
    settings = WorkflowSettings(
        episode_slug=config.run.episode_slug,
        investor_name=investor_name,
        pitch=package.pitch.read_text(encoding="utf-8"),
        taxonomy_labels={row["label"] for row in taxonomy_records},
        taxonomy_records=taxonomy_records,
        check_tiers=check_tiers,
        phase1_min_iterations=config.phase1.min_iterations,
        phase1_max_iterations=config.phase1.max_iterations,
        phase2_min_iterations=config.phase2.min_iterations,
        phase2_max_iterations=config.phase2.max_iterations,
        retrieval_top_k=config.retrieval.top_k,
        max_exact_reads=config.retrieval.max_exact_reads,
        run_root=run_root,
        contract_version=config.run.contract_version,
        execution_mode=config.run.mode,
        initial_phase1_quality_findings=sanitization_findings,
        phase1_max_output_tokens=(
            config.phase1.max_output_tokens or config.provider.max_output_tokens
            if config.run.contract_version == "v4.4"
            else config.phase1.max_output_tokens
        ),
        phase1_reasoning_effort=config.phase1.reasoning_effort,
        phase1_planning_max_output_tokens=config.phase1.planning_max_output_tokens,
        phase1_planning_reasoning_effort=config.phase1.planning_reasoning_effort,
        phase2_max_output_tokens=config.phase2.max_output_tokens,
        phase2_reasoning_effort=config.phase2.reasoning_effort,
        phase2_planning_max_output_tokens=config.phase2.planning_max_output_tokens,
        phase2_planning_reasoning_effort=config.phase2.planning_reasoning_effort,
        precedent_manifest_sha256=precedent_manifest_sha256,
        accessible_precedent_count=(
            len(precedent_corpus.list_episodes())
            if precedent_corpus is not None
            else 0
        ),
        phase1_max_precedent_searches=config.phase1.max_precedent_searches,
        phase1_max_precedent_reads=config.phase1.max_precedent_reads,
        phase2_max_precedent_searches=config.phase2.max_precedent_searches,
        phase2_max_precedent_reads=config.phase2.max_precedent_reads,
        allow_full_transcript=config.precedents.allow_full_transcript,
        precedent_selection_policy=config.precedents.selection_policy,
        precedent_candidate_pool_k=config.precedents.candidate_pool_k,
        precedent_in_slots=config.precedents.in_slots,
        precedent_out_slots=config.precedents.out_slots,
        portfolio_memory_enabled=config.portfolio_memory.enabled,
        portfolio_retrieval_top_k=config.portfolio_memory.retrieval_top_k,
        portfolio_candidate_pool_k=config.portfolio_memory.candidate_pool_k,
    )
    if config.run.contract_version == "v4.4":
        if config.phase1_v44 is None:
            raise ValueError("v4.4 requires phase1_v44 settings")
        return _CliDecisionWorkflowV44(
            settings,
            index,
            checkpointer=checkpointer,
            phase1_provider=phase1_provider,
            v44_settings=config.phase1_v44,
            precedent_corpus=precedent_corpus,
            portfolio_index=portfolio_index,
        )
    workflow_class = (
        VCDecisionWorkflowV4
        if config.run.contract_version in {"v4", "v4.1", "v4.2", "v4.3", "v5"}
        else VCDecisionWorkflow
    )
    return workflow_class(
        settings,
        index,
        checkpointer=checkpointer,
        phase1_provider=phase1_provider,
        phase2_provider=phase2_provider,
        precedent_corpus=precedent_corpus,
        **(
            {"portfolio_index": portfolio_index}
            if workflow_class is VCDecisionWorkflowV4
            else {}
        ),
    )


def _decision_summary(result: dict) -> dict:
    decision = result.get("decision", {})
    any_check = decision.get("any_check", {})
    standard_check = decision.get("standard_check", {})
    summary = {
        "phase1_status": result.get("phase1_status"),
        "phase1_iterations": result.get("phase1_iteration"),
        "phase1_findings": result.get("phase1_findings", []),
        "phase2_status": result.get("phase2_status"),
        "phase2_iterations": result.get("phase2_iteration"),
        "phase2_findings": result.get("phase2_findings", []),
        "decision": decision.get("decision"),
        "investment_likelihood": decision.get("investment_likelihood"),
        "any_check_decision": any_check.get("decision"),
        "any_check_likelihood": any_check.get("likelihood"),
        "standard_check_decision": standard_check.get("decision"),
        "standard_check_likelihood": standard_check.get("likelihood"),
        "recommended_check_tier": decision.get("recommended_check_tier"),
        "ranking_score": decision.get("ranking_score"),
    }
    if "review_priority_score" in decision:
        summary["review_priority_score"] = decision["review_priority_score"]
        summary["decision_confidence"] = decision.get("decision_confidence")
    return summary


def command_index(config: RunConfig) -> None:
    _require_v44_embedding_config(config)
    package = verify_package(
        config.resolve_path(config.run.input_root),
        config.run.vc_slug,
        config.run.episode_slug,
        taxonomy_path=config.run.taxonomy_path,
    )
    if config.precedents.enabled and package.precedents is None:
        raise ValueError("precedent retrieval is enabled but the verified package has no corpus")
    provider = _indexing_provider(config)
    if config.portfolio_memory.enabled:
        filtered = _filtered_portfolio_for_target(config, package, provider)
        assert filtered is not None
        print(
            f"validated {len(filtered.filtered.disclosures)} eligible portfolio disclosures"
        )
    require_complete = config.run.contract_version == "v4.4" or bool(
        config.embedding is not None and config.embedding.require_complete_index
    )
    index = HybridWikiIndex.build(
        package.wiki,
        provider,
        require_complete_embeddings=require_complete,
    )
    precedents = None
    if config.precedents.enabled:
        assert package.precedents is not None
        precedents = PrecedentCorpus.build(
            package.precedents,
            provider,
            require_complete_embeddings=require_complete,
        )
        if precedents.target_slug is not None:
            raise ValueError("canonical precedent index must be unfiltered")
    index.save(_index_path(config))
    print(f"indexed {len(index.chunks)} wiki sections -> {_index_path(config)}")
    if precedents is not None:
        precedents.save(_precedent_index_path(config))
        print(
            f"indexed {len(precedents.list_episodes())} precedent episodes -> "
            f"{_precedent_index_path(config)}"
        )


def command_run(config: RunConfig, *, resume: bool = False) -> None:
    _require_v44_embedding_config(config)
    package = verify_package(
        config.resolve_path(config.run.input_root),
        config.run.vc_slug,
        config.run.episode_slug,
        taxonomy_path=config.run.taxonomy_path,
    )
    phase1_provider = _generation_provider(config, "phase1")
    phase2_provider = (
        None
        if config.run.mode == "phase1_only"
        else _generation_provider(config, "phase2")
    )
    embedder = (
        _embedding_provider(config, config.embedding)
        if config.embedding is not None
        else None
        if config.precedents.enabled
        else phase1_provider
    )
    if isinstance(embedder, OpenRouterProvider):
        raise ValueError("OpenRouter generation cannot be used for embeddings")
    require_complete = config.run.contract_version == "v4.4" or bool(
        config.embedding is not None and config.embedding.require_complete_index
    )
    canonical_index = HybridWikiIndex.load(
        _index_path(config),
        package.wiki,
        embedder,
        require_complete_embeddings=require_complete,
    )
    index, sanitization = canonical_index.for_pitch(package.target_company_aliases)
    run_root = config.resolve_path(config.run.output_root) / config.run.episode_slug
    input_provenance = {
        "pitch_path": str(package.pitch),
        "pitch_sha256": sha256(package.pitch.read_bytes()).hexdigest(),
        "package_manifest_path": str(package.manifest),
        "package_manifest_sha256": sha256(package.manifest.read_bytes()).hexdigest(),
    }
    cli_audits: dict[str, dict] = {}
    if config.run.contract_version == "v4.4":
        cli_audits = {
            "run-config.json": config.model_dump(mode="json"),
            "input-provenance.json": input_provenance,
            "wiki-sanitization.json": asdict(sanitization),
        }
    else:
        write_json(run_root / "run-config.json", config.model_dump(mode="json"))
        write_json(run_root / "input-provenance.json", input_provenance)
        write_json(run_root / "wiki-sanitization.json", asdict(sanitization))
    precedent_corpus: PrecedentCorpus | None = None
    precedent_manifest_sha256: str | None = None
    if config.precedents.enabled:
        if package.precedents is None:
            raise ValueError(
                "precedent retrieval is enabled but the verified package has no corpus"
            )
        canonical_precedents = PrecedentCorpus.load(
            _precedent_index_path(config), package.precedents, embedder,
            require_complete_embeddings=require_complete,
        )
        precedent_corpus = _filtered_precedents_for_target(
            canonical_precedents,
            target_slug=config.run.episode_slug,
            target_company_aliases=package.target_company_aliases,
        )
        filtered_manifest = precedent_corpus.filtered_manifest()
        filtered_payload = filtered_manifest.model_dump(mode="json")
        if config.run.contract_version == "v4.4":
            cli_audits["precedent-manifest.filtered.json"] = filtered_payload
        else:
            write_json(
                run_root / "precedent-manifest.filtered.json",
                filtered_payload,
            )
        precedent_manifest_sha256 = filtered_manifest.sha256
    portfolio_index = _filtered_portfolio_for_target(config, package, embedder)
    if portfolio_index is not None:
        portfolio_payload = portfolio_index.filtered.manifest().model_dump(mode="json")
        if config.run.contract_version == "v4.4":
            cli_audits["portfolio-manifest.filtered.json"] = portfolio_payload
        else:
            write_json(
                run_root / "portfolio-manifest.filtered.json",
                portfolio_payload,
            )
    checkpoint = config.resolve_path(config.run.checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    thread_id = f"{config.run.vc_slug}:{config.run.episode_slug}"
    with SqliteSaver.from_conn_string(str(checkpoint)) as saver:
        workflow = _workflow(
            config,
            phase1_provider,
            phase2_provider,
            index,
            package,
            saver,
            sanitization_findings=sanitization.quality_findings,
            precedent_corpus=precedent_corpus,
            precedent_manifest_sha256=precedent_manifest_sha256,
            portfolio_index=portfolio_index,
        )
        if isinstance(workflow, _CliDecisionWorkflowV44):
            workflow.bootstrap_cli_audits(
                cli_audits,
                require_existing=resume,
            )
        if resume:
            result = workflow.resume(thread_id)
        else:
            result = workflow.invoke(thread_id)
        if isinstance(workflow, _CliDecisionWorkflowV44):
            workflow.bootstrap_cli_audits(
                cli_audits,
                require_existing=True,
            )
    print(json.dumps(_decision_summary(result), sort_keys=True))


def command_decide(config: RunConfig, phase1_from: Path) -> None:
    if config.run.contract_version == "v4.4":
        raise ValueError("v4.4 is Phase-1-only and cannot run decide")
    package = verify_package(
        config.resolve_path(config.run.input_root),
        config.run.vc_slug,
        config.run.episode_slug,
        taxonomy_path=config.run.taxonomy_path,
    )
    source = phase1_from.resolve()
    raw_investigation, digest = read_verified_frozen(
        source / "investigation.json", source / "investigation.sha256"
    )
    investigation_model = (
        InvestigationV5
        if config.run.contract_version == "v5"
        else
        InvestigationV41
        if config.run.contract_version == "v4.1"
        else InvestigationV4
        if config.run.contract_version == "v4"
        else InvestigationV3
        if config.run.contract_version == "v3"
        else InvestigationV2
        if config.run.contract_version == "v2"
        else Investigation
    )
    investigation = investigation_model.model_validate_json(raw_investigation)
    if investigation.episode_slug != config.run.episode_slug:
        raise ValueError("frozen Phase 1 episode does not match config")
    run_root = (config.resolve_path(config.run.output_root) / config.run.episode_slug).resolve()
    if run_root.is_relative_to(source):
        raise ValueError("Phase 2 replay output cannot be inside the source Phase 1")
    if run_root.exists() and any(run_root.iterdir()):
        raise ValueError(f"Phase 2 replay output already exists: {run_root}")
    checkpoint = config.resolve_path(config.run.checkpoint_path).resolve()
    copied_phase1 = (run_root / "phase1").resolve()
    if checkpoint.is_relative_to(source) or checkpoint.is_relative_to(copied_phase1):
        raise ValueError("Phase 2 replay checkpoint cannot be inside frozen Phase 1")
    if checkpoint.exists():
        raise ValueError(f"Phase 2 replay requires a fresh checkpoint: {checkpoint}")
    run_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, run_root / "phase1")
    copied_digest = verify_frozen(
        run_root / "phase1/investigation.json",
        run_root / "phase1/investigation.sha256",
    )
    if copied_digest != digest:
        raise ValueError("copied Phase 1 does not match the verified source")
    source_state = source.parent / "state.json"
    phase1_status = "accepted"
    source_state_payload: dict = {}
    if source_state.is_file():
        source_state_payload = json.loads(source_state.read_text(encoding="utf-8"))
        phase1_status = source_state_payload.get(
            "phase1_status", "accepted"
        )
    replay_with_state = config.run.contract_version in {"v3", "v4", "v4.1", "v5"}
    if replay_with_state:
        if not source_state.is_file():
            raise ValueError(
                f"{config.run.contract_version} Phase 2 replay source lacks state.json"
            )
        if phase1_status not in {"accepted", "provisional"}:
            raise ValueError(
                f"{config.run.contract_version} Phase 2 replay source Phase 1 is not frozen"
            )
    if config.run.contract_version == "v3":
        required_phase1_keys = {
            "phase1_iteration",
            "query_plan",
            "query_history",
            "retrieval_passes",
            "exact_reads",
            "phase1_precedent_searches",
            "phase1_precedent_reads",
            "phase1_historical_evidence",
            "phase1_findings",
            "phase1_quality_findings",
            "phase1_retrieval_findings",
            "phase1_current_retrieval_findings",
            "usage_by_phase",
            "precedent_manifest_sha256",
            "accessible_precedent_count",
        }
        missing = sorted(required_phase1_keys - source_state_payload.keys())
        if missing:
            raise ValueError(
                "v3 Phase 2 replay source state is incomplete: " + ", ".join(missing)
            )
        if source_state_payload.get("investigation") != investigation.model_dump(
            mode="json"
        ):
            raise ValueError("v3 Phase 2 replay source state investigation changed")
        verify_v3_phase1_replay_source(source.parent, source_state_payload)
    elif config.run.contract_version in {"v4", "v4.1", "v5"}:
        required_phase1_keys = {
            "phase1_iteration",
            "phase1_action",
            "phase1_findings",
            "phase1_plan",
            "phase1_current_retrieval",
            "phase1_retrieval_warnings",
            "evidence_registry",
            "usage_by_phase",
            "precedent_manifest_sha256",
            "accessible_precedent_count",
            "investigation",
            "investigation_sha256",
        }
        missing = sorted(required_phase1_keys - source_state_payload.keys())
        if missing:
            raise ValueError(
                f"{config.run.contract_version} Phase 2 replay source state is incomplete: "
                + ", ".join(missing)
            )
        if source_state_payload["investigation"] != investigation.model_dump(
            mode="json"
        ):
            raise ValueError(
                f"{config.run.contract_version} Phase 2 replay source state investigation changed"
            )
        if source_state_payload["investigation_sha256"] != digest:
            raise ValueError(
                f"{config.run.contract_version} Phase 2 replay state investigation hash changed"
            )
        if config.portfolio_memory.enabled:
            portfolio_required = {
                "phase1_portfolio_candidates", "portfolio_filtered_manifest"
            }
            portfolio_missing = sorted(
                portfolio_required - source_state_payload.keys()
            )
            if portfolio_missing:
                raise ValueError(
                    "v4.1 Phase 2 replay source portfolio state is incomplete: "
                    + ", ".join(portfolio_missing)
                )
    _verify_replay_historical_registry(source_state_payload, config.run.episode_slug)
    if config.run.contract_version == "v3":
        if source_state_payload.get("investigation_sha256") != digest:
            raise ValueError("Phase 2 replay state investigation hash changed")
        validate_investigation_v3(
            investigation,
            episode_slug=config.run.episode_slug,
            pitch=package.pitch.read_text(encoding="utf-8"),
            taxonomy_labels=_taxonomy_labels(package),
            exact_wiki_ids={
                row["chunk_id"] for row in source_state_payload.get("exact_reads", [])
            },
            exact_historical_ids={
                row["evidence_id"]
                for row in source_state_payload.get("phase1_historical_evidence", [])
            },
        )
    source_provenance_path = source.parent / "input-provenance.json"
    if replay_with_state and not source_provenance_path.is_file():
        raise ValueError(
            f"{config.run.contract_version} Phase 2 replay source lacks input provenance"
        )
    if source_provenance_path.is_file():
        source_provenance = json.loads(source_provenance_path.read_text(encoding="utf-8"))
        if source_provenance.get("pitch_sha256") != sha256(package.pitch.read_bytes()).hexdigest():
            raise ValueError("Phase 2 replay pitch provenance changed")
        if source_provenance.get("package_manifest_sha256") != sha256(package.manifest.read_bytes()).hexdigest():
            raise ValueError("Phase 2 replay package provenance changed")
    write_json(
        run_root / "phase1-provenance.json",
        {
            "source_phase1": str(source),
            "investigation_sha256": digest,
            "phase1_status": phase1_status,
            "replayed_phases": ["phase2"],
        },
    )
    phase2_provider = _generation_provider(config, "phase2")
    embedder = (
        _embedding_provider(config, config.embedding)
        if config.embedding is not None
        else None
        if config.precedents.enabled
        else _generation_provider(config, "phase1")
    )
    if isinstance(embedder, OpenRouterProvider):
        raise ValueError("OpenRouter generation cannot be used for embeddings")
    require_complete = bool(
        config.embedding is not None and config.embedding.require_complete_index
    )
    canonical_index = HybridWikiIndex.load(
        _index_path(config), package.wiki, embedder,
        require_complete_embeddings=require_complete,
    )
    index, sanitization = canonical_index.for_pitch(package.target_company_aliases)
    write_json(run_root / "wiki-sanitization.json", asdict(sanitization))
    precedent_corpus: PrecedentCorpus | None = None
    precedent_manifest_sha256: str | None = None
    if config.precedents.enabled:
        if package.precedents is None:
            raise ValueError("precedent retrieval is enabled but package has no corpus")
        canonical = PrecedentCorpus.load(
            _precedent_index_path(config), package.precedents, embedder,
            require_complete_embeddings=require_complete,
        )
        precedent_corpus = _filtered_precedents_for_target(
            canonical,
            target_slug=config.run.episode_slug,
            target_company_aliases=package.target_company_aliases,
        )
        filtered_manifest = precedent_corpus.filtered_manifest()
        precedent_manifest_sha256 = filtered_manifest.sha256
        source_manifest_path = source.parent / "precedent-manifest.filtered.json"
        if not source_manifest_path.is_file():
            raise ValueError("Phase 2 replay source lacks filtered precedent manifest")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_payload = {
            key: value for key, value in source_manifest.items() if key != "sha256"
        }
        source_manifest_digest = sha256(
            json.dumps(source_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if source_manifest.get("sha256") != source_manifest_digest:
            raise ValueError("Phase 2 replay source precedent manifest hash changed")
        if source_manifest.get("target_episode_slug") != config.run.episode_slug:
            raise ValueError("Phase 2 replay source precedent manifest target changed")
        source_accessible = source_manifest.get("accessible_episodes")
        if not isinstance(source_accessible, list) or any(
            isinstance(row, dict)
            and row.get("episode_slug") == config.run.episode_slug
            for row in source_accessible
        ):
            raise ValueError("Phase 2 replay source precedent manifest is not filtered")
        if source_state_payload.get("accessible_precedent_count") != len(
            source_accessible
        ):
            raise ValueError(
                "Phase 2 replay source accessible precedent count changed"
            )
        if source_manifest.get("sha256") != precedent_manifest_sha256:
            raise ValueError("Phase 2 replay precedent manifest changed")
        if source_state_payload.get("precedent_manifest_sha256") != precedent_manifest_sha256:
            raise ValueError("Phase 2 replay state precedent manifest hash changed")
        write_json(run_root / "precedent-manifest.filtered.json", filtered_manifest.model_dump(mode="json"))
    portfolio_index = _filtered_portfolio_for_target(config, package, embedder)
    if portfolio_index is not None:
        filtered_portfolio_manifest = (
            portfolio_index.filtered.manifest().model_dump(mode="json")
        )
        if source_state_payload.get("portfolio_filtered_manifest") != filtered_portfolio_manifest:
            raise ValueError("Phase 2 replay portfolio manifest changed")
        write_json(
            run_root / "portfolio-manifest.filtered.json",
            filtered_portfolio_manifest,
        )
    write_json(run_root / "run-config.json", config.model_dump(mode="json"))
    write_json(
        run_root / "input-provenance.json",
        {
            "pitch_path": str(package.pitch),
            "pitch_sha256": sha256(package.pitch.read_bytes()).hexdigest(),
            "package_manifest_path": str(package.manifest),
            "package_manifest_sha256": sha256(package.manifest.read_bytes()).hexdigest(),
        },
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    thread_id = f"{config.run.vc_slug}:{config.run.episode_slug}:phase2-replay"
    with SqliteSaver.from_conn_string(str(checkpoint)) as saver:
        workflow = _workflow(
            config,
            phase2_provider,
            phase2_provider,
            index,
            package,
            saver,
            sanitization_findings=sanitization.quality_findings,
            precedent_corpus=precedent_corpus,
            precedent_manifest_sha256=precedent_manifest_sha256,
            portfolio_index=portfolio_index,
        )
        if config.run.contract_version == "v5":
            result = workflow.invoke_phase2_v5(
                thread_id,
                investigation,
                digest,
                phase1_status=phase1_status,
                phase1_state=source_state_payload,
            )
        else:
            result = workflow.invoke_phase2(
                thread_id,
                investigation,
                digest,
                phase1_status=phase1_status,
                phase1_exact_reads=source_state_payload.get("exact_reads", []),
                phase1_historical_evidence=source_state_payload.get("phase1_historical_evidence", []),
                phase1_precedent_reads=source_state_payload.get("phase1_precedent_reads", []),
                phase1_state=(
                    source_state_payload
                    if replay_with_state
                    else None
                ),
            )
    print(json.dumps(_decision_summary(result), sort_keys=True))


def command_verify(config: RunConfig) -> None:
    package = verify_package(
        config.resolve_path(config.run.input_root),
        config.run.vc_slug,
        config.run.episode_slug,
        taxonomy_path=config.run.taxonomy_path,
    )
    run_root = config.resolve_path(config.run.output_root) / config.run.episode_slug
    if config.run.mode == "phase1_only":
        verify_phase1_artifacts(
            run_root,
            provenance_mode=(
                "cli" if config.run.contract_version == "v4.4" else "direct"
            ),
            expected_config=(config if config.run.contract_version == "v4.4" else None),
            expected_package=(
                package if config.run.contract_version == "v4.4" else None
            ),
        )
    else:
        verify_run_artifacts(run_root)
    print(f"verified {run_root}")


class _PreflightEmbedder:
    """Keep preflight deterministic and offline while exercising index validation."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("preflight does not compute embeddings")


def command_preflight(config: RunConfig, config_path: Path) -> None:
    _require_v44_embedding_config(config)
    package = verify_package(
        config.resolve_path(config.run.input_root),
        config.run.vc_slug,
        config.run.episode_slug,
        taxonomy_path=config.run.taxonomy_path,
    )
    audit = json.loads(package.audit.read_text(encoding="utf-8"))
    if config.portfolio_memory.enabled and package.portfolio_memory is None:
        raise ValueError(
            "portfolio memory is enabled but the verified package has no corpus"
        )
    corpus_count = 0
    projected_access_count = 0
    if package.precedent_manifest is not None:
        corpus_manifest = json.loads(
            package.precedent_manifest.read_text(encoding="utf-8")
        )
        corpus_count = len(corpus_manifest["records"])
        projected_access_count = corpus_count - sum(
            row["episode_slug"] == config.run.episode_slug
            for row in corpus_manifest["records"]
        )

    wiki_path = _index_path(config)
    required_indexes = [wiki_path]
    if config.precedents.enabled:
        required_indexes.append(_precedent_index_path(config))
    missing = [path.as_posix() for path in required_indexes if not path.is_file()]

    phase1_model = config.phase1.model or config.provider.model
    phase2_model = (
        None
        if config.run.contract_version == "v4.4"
        else config.phase2.model or config.provider.model
    )
    report = {
        "status": "not_ready" if missing else "ready",
        "target": config.run.episode_slug,
        "pitch": {
            "status": audit["status"],
            "sha256": sha256(package.pitch.read_bytes()).hexdigest(),
        },
        "corpus_count": corpus_count,
        "accessible_precedent_count": projected_access_count,
        "filtered_manifest_sha256": None,
        "target_accessible": False,
        "provider": config.provider.kind,
        "phase1_model": phase1_model,
        "phase2_model": phase2_model,
        "budgets": {
            "phase1": {
                "min_iterations": config.phase1.min_iterations,
                "max_iterations": config.phase1.max_iterations,
                "max_output_tokens": (
                    config.phase1.max_output_tokens
                    or config.provider.max_output_tokens
                ),
                "planning_max_output_tokens": (
                    config.phase1.planning_max_output_tokens
                ),
                "max_precedent_searches": config.phase1.max_precedent_searches,
                "max_precedent_reads": config.phase1.max_precedent_reads,
            },
            "phase2": {
                "min_iterations": config.phase2.min_iterations,
                "max_iterations": config.phase2.max_iterations,
                "max_output_tokens": (
                    config.phase2.max_output_tokens
                    or config.provider.max_output_tokens
                ),
                "planning_max_output_tokens": (
                    config.phase2.planning_max_output_tokens
                ),
                "max_precedent_searches": config.phase2.max_precedent_searches,
                "max_precedent_reads": config.phase2.max_precedent_reads,
            },
            "retrieval": {
                "top_k": config.retrieval.top_k,
                "max_exact_reads": config.retrieval.max_exact_reads,
                "precedent_selection_policy": config.precedents.selection_policy,
                "precedent_candidate_pool_k": config.precedents.candidate_pool_k,
                "precedent_in_slots": config.precedents.in_slots,
                "precedent_out_slots": config.precedents.out_slots,
            },
        },
    }
    if config.run.contract_version == "v4.4":
        assert config.phase1_v44 is not None
        report["budgets"]["phase1_v44"] = config.phase1_v44.model_dump(
            mode="json"
        )
    if missing:
        report["missing_indexes"] = missing
        report["index_command"] = (
            f"uv run vc-clone-graph index --config {config_path}"
        )
        print(json.dumps(report, sort_keys=True))
        return
    require_complete = config.run.contract_version == "v4.4" or bool(
        config.embedding is not None and config.embedding.require_complete_index
    )
    offline = (
        _embedding_provider(config, config.embedding)
        if (require_complete or config.portfolio_memory.enabled)
        and config.embedding is not None
        else _PreflightEmbedder()
    )
    if wiki_path.is_file():
        canonical_wiki = HybridWikiIndex.load(
            wiki_path,
            package.wiki,
            offline,
            require_complete_embeddings=require_complete,
        )
        pitch_wiki, _ = canonical_wiki.for_pitch(package.target_company_aliases)
        if require_complete and not any(
            "dense" in hit.retrieval_modes
            for hit in pitch_wiki.search("founder market investment", 3)
        ):
            raise ValueError("wiki semantic preflight produced no dense retrieval")
        report["wiki_embedding_index"] = canonical_wiki.embedding_index
        if config.run.contract_version == "v4.4":
            identity = embedding_identity(offline)
            index_identity = canonical_wiki.embedding_index
            if (
                not isinstance(index_identity, dict)
                or index_identity.get("model") != identity.get("model")
                or index_identity.get("revision") != identity.get("revision")
            ):
                raise ValueError(
                    "v4.4 wiki embedding index identity does not match provider"
                )
            report["taxonomy_embedding_preflight"] = (
                _preflight_v44_taxonomy_embeddings(config, package, offline)
            )
    precedent_path = _precedent_index_path(config)
    if config.precedents.enabled and precedent_path.is_file():
        if package.precedents is None:
            raise ValueError(
                "precedent retrieval is enabled but the verified package has no corpus"
            )
        canonical = PrecedentCorpus.load(
            precedent_path, package.precedents, offline,
            require_complete_embeddings=require_complete,
        )
        filtered = _filtered_precedents_for_target(
            canonical,
            target_slug=config.run.episode_slug,
            target_company_aliases=package.target_company_aliases,
        )
        episodes = filtered.list_episodes()
        target_accessible = any(
            row.episode_slug == config.run.episode_slug for row in episodes
        )
        manifest = filtered.filtered_manifest()
        report["accessible_precedent_count"] = len(episodes)
        report["filtered_manifest_sha256"] = manifest.sha256
        report["target_accessible"] = target_accessible
        if require_complete:
            search = filtered.search(
                "founder market investment",
                top_k=3,
                query_id="preflight-semantic",
            )
            if not any("dense" in hit.retrieval_modes for hit in search.hits):
                raise ValueError(
                    "precedent semantic preflight produced no dense retrieval"
                )
        report["precedent_embedding_index"] = canonical.embedding_index
    elif not config.precedents.enabled:
        report["accessible_precedent_count"] = 0
    if config.portfolio_memory.enabled:
        filtered_portfolio = _filtered_portfolio_for_target(
            config, package, offline
        )
        assert filtered_portfolio is not None
        portfolio_manifest = filtered_portfolio.filtered.manifest()
        report["portfolio_memory"] = {
            "eligible_disclosure_count": len(
                filtered_portfolio.filtered.disclosures
            ),
            "excluded_current_or_later_count": (
                portfolio_manifest.excluded_current_or_later_count
            ),
            "excluded_unordered_count": portfolio_manifest.excluded_unordered_count,
            "filtered_manifest_sha256": portfolio_manifest.sha256,
            "target_accessible": False,
        }
    print(json.dumps(report, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vc-clone-graph")
    parser.add_argument(
        "command",
        choices=("index", "preflight", "run", "resume", "decide", "verify"),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase1-from", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "index":
            command_index(config)
        elif args.command == "preflight":
            command_preflight(config, args.config)
        elif args.command == "run":
            command_run(config)
        elif args.command == "resume":
            command_run(config, resume=True)
        elif args.command == "decide":
            if args.phase1_from is None:
                raise ValueError("decide requires --phase1-from")
            command_decide(config, args.phase1_from)
        else:
            command_verify(config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
