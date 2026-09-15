"""Runtime composition for configuration-driven founder rehearsal sessions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import tomllib
from typing import Any, Sequence

from langgraph.checkpoint.memory import InMemorySaver

from .portfolio_memory import (
    FilteredPortfolioIndex,
    PortfolioMemoryCorpus,
    PortfolioMemoryIndex,
)
from .precedents import PrecedentCorpus
from .providers.base import EmbeddingProvider, GenerationProvider
from .providers.ollama import OllamaProvider
from .providers.openai import OpenAIProvider
from .providers.openrouter import OpenRouterProvider
from .providers.sentence_transformers import SentenceTransformerEmbeddingProvider
from .question_memory import QuestionMemory
from .rehearsal_config import RehearsalConfig
from .rehearsal_classifier import ClassifierRegistry, ClassifierResolution
from .rehearsal_graph import RehearsalEvidenceBundle
from .rehearsal_grounding import CanonicalRehearsalBaseline, load_canonical_baseline
from .rehearsal_artifacts import RehearsalArtifactStore
from .rehearsal_phase2_bridge import CanonicalPhase2Bridge
from .graph import WorkflowSettings
from .workflow_v4 import VCDecisionWorkflowV4
from .retrieval import HybridWikiIndex


@dataclass(frozen=True)
class InvestorProfile:
    vc_slug: str
    display_name: str
    firm: str
    role: str
    wiki_path: Path
    registry_path: Path


@dataclass
class RehearsalRuntime:
    investor: InvestorProfile
    provider: GenerationProvider
    embedder: EmbeddingProvider
    wiki: HybridWikiIndex
    precedents: PrecedentCorpus | None
    portfolio: FilteredPortfolioIndex | None
    question_memory: QuestionMemory
    taxonomy: tuple[dict[str, str], ...]
    retriever: "RuntimeEvidenceRetriever"
    sanitization_findings: tuple[str, ...]
    classifier_resolution: ClassifierResolution
    grounded_baseline: CanonicalRehearsalBaseline | None


def build_grounded_phase2_synthesizer(
    *,
    runtime: RehearsalRuntime,
    config: RehearsalConfig,
    store: RehearsalArtifactStore,
    pitch: str,
) -> CanonicalPhase2Bridge:
    """Compose one leakage-safe canonical Phase 2 replay for a rehearsal."""
    baseline = runtime.grounded_baseline
    if baseline is None:
        raise ValueError("canonical baseline is required for grounded Phase 2")
    raw = json.loads((baseline.run_root / "run-config.json").read_text(encoding="utf-8"))
    phase2 = raw.get("phase2", {})
    precedents = raw.get("precedents", {})
    settings = WorkflowSettings(
        episode_slug=baseline.episode_slug,
        investor_name=runtime.investor.display_name,
        pitch=pitch,
        taxonomy_labels={row["label"] for row in runtime.taxonomy},
        taxonomy_records=runtime.taxonomy,
        check_tiers={"none", "exploratory_lt_100k", "standard"},
        phase1_min_iterations=1,
        phase1_max_iterations=1,
        phase2_min_iterations=int(phase2.get("min_iterations", 1)),
        phase2_max_iterations=int(phase2.get("max_iterations", 4)),
        retrieval_top_k=config.retrieval.top_k,
        max_exact_reads=config.retrieval.max_exact_reads,
        run_root=store.session_root / "grounding/canonical-phase2-workflow",
        contract_version=baseline.contract_version,
        phase2_max_output_tokens=int(
            phase2.get("max_output_tokens", config.rehearsal.max_output_tokens)
        ),
        phase2_reasoning_effort=phase2.get(
            "reasoning_effort", config.rehearsal.reasoning_effort
        ),
        phase2_planning_max_output_tokens=int(
            phase2.get("planning_max_output_tokens", 8192)
        ),
        phase2_planning_reasoning_effort=phase2.get("planning_reasoning_effort"),
        phase2_max_precedent_searches=int(phase2.get("max_precedent_searches", 4)),
        phase2_max_precedent_reads=int(phase2.get("max_precedent_reads", 8)),
        allow_full_transcript=bool(precedents.get("allow_full_transcript", True)),
        precedent_selection_policy=str(
            precedents.get("selection_policy", config.precedents.selection_policy)
        ),  # type: ignore[arg-type]
        precedent_candidate_pool_k=int(
            precedents.get("candidate_pool_k", config.precedents.candidate_pool_k)
        ),
        precedent_in_slots=int(
            precedents.get("in_slots", config.precedents.in_slots)
        ),
        precedent_out_slots=int(
            precedents.get("out_slots", config.precedents.out_slots)
        ),
        portfolio_memory_enabled=runtime.portfolio is not None,
        portfolio_retrieval_top_k=config.portfolio_memory.retrieval_top_k,
        portfolio_candidate_pool_k=config.portfolio_memory.candidate_pool_k,
    )
    workflow = VCDecisionWorkflowV4(
        settings,
        runtime.wiki,
        checkpointer=InMemorySaver(),
        phase1_provider=runtime.provider,
        phase2_provider=runtime.provider,
        precedent_corpus=runtime.precedents,
        portfolio_index=runtime.portfolio,
    )
    return CanonicalPhase2Bridge(workflow=workflow, store=store)


def resolve_canonical_pitch(
    baseline: CanonicalRehearsalBaseline,
    *,
    vc_slug: str,
    workspace: Path,
) -> str:
    """Resolve the immutable pitch copy bound to the canonical baseline digest."""
    provenance = json.loads(
        (baseline.run_root / "input-provenance.json").read_text(encoding="utf-8")
    )
    expected = str(provenance["pitch_sha256"])
    candidates = [
        Path(str(provenance["pitch_path"])),
        workspace.parent
        / "agentic-vc-clone-framework/data/cuts"
        / f"{baseline.episode_slug}__{vc_slug}/bq.txt",
    ]
    for candidate in candidates:
        if candidate.is_file() and sha256(candidate.read_bytes()).hexdigest() == expected:
            return candidate.read_text(encoding="utf-8")
    raise ValueError(
        "canonical pitch copy is unavailable or does not match the baseline digest"
    )


def _classifier_vc_slug(profile: InvestorProfile) -> str:
    """Map the public profile identity to the canonical evaluation identity."""
    tokens = re.findall(r"[a-z0-9]+", profile.display_name.casefold())
    return "-".join(tokens[:2])


def _classifier_resolution(
    config: RehearsalConfig,
    profile: InvestorProfile,
    *,
    excluded_episode_slug: str | None,
) -> ClassifierResolution:
    if config.classification.mode == "rationale_only":
        return ClassifierResolution(
            status="fallback",
            artifact=None,
            reason="classification mode is rationale_only",
        )
    path = config.resolve_path(config.classification.registry_path)
    try:
        registry = ClassifierRegistry.load(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ClassifierResolution(
            status="fallback",
            artifact=None,
            reason=f"classifier registry unavailable: {exc}",
        )
    return registry.resolve_or_fallback(
        _classifier_vc_slug(profile),
        excluded_episode_slug=excluded_episode_slug,
    )


def _registry_profile(input_root: Path, path: Path) -> InvestorProfile:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    required = ("vc_slug", "display_name", "wiki_path")
    missing = [field for field in required if not raw.get(field)]
    if missing:
        raise ValueError(f"investor registry lacks fields: {', '.join(missing)}")
    wiki = (input_root / str(raw["wiki_path"])).resolve()
    if not wiki.is_relative_to(input_root.resolve()) or not wiki.is_dir():
        raise ValueError("investor wiki path is invalid")
    return InvestorProfile(
        vc_slug=str(raw["vc_slug"]),
        display_name=str(raw["display_name"]),
        firm=str(raw.get("firm", "")),
        role=str(raw.get("role", "Investor")),
        wiki_path=wiki,
        registry_path=path.resolve(),
    )


def list_investors(input_root: Path) -> tuple[InvestorProfile, ...]:
    """Discover every configured VC without hard-coded investor branches."""
    root = Path(input_root).resolve()
    registry_root = root / "investors"
    if not registry_root.is_dir():
        raise ValueError(f"investor registry directory is missing: {registry_root}")
    profiles = tuple(
        _registry_profile(root, path)
        for path in sorted(registry_root.glob("*.toml"))
    )
    if not profiles:
        raise ValueError("no investor profiles are configured")
    slugs = [profile.vc_slug for profile in profiles]
    if len(slugs) != len(set(slugs)):
        raise ValueError("investor registry contains duplicate VC slugs")
    return profiles


def _taxonomy(path: Path) -> tuple[dict[str, str], ...]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("rationale taxonomy must be a non-empty JSON array")
    result: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("rationale taxonomy rows must be objects")
        record = {
            "label": row.get("label"),
            "definition": row.get("definition"),
            "coarse_parent": row.get("coarse_parent"),
        }
        if not all(isinstance(value, str) and value for value in record.values()):
            raise ValueError("rationale taxonomy rows lack label, definition, or parent")
        result.append(record)  # type: ignore[arg-type]
    labels = [row["label"] for row in result]
    if len(labels) != len(set(labels)):
        raise ValueError("rationale taxonomy labels must be unique")
    return tuple(result)


def _generation_provider(config: RehearsalConfig) -> GenerationProvider:
    settings = config.provider
    if settings.kind == "openrouter":
        return OpenRouterProvider(
            settings.model,
            base_url=settings.base_url or "https://openrouter.ai/api/v1",
            api_key_env=settings.api_key_env or "OPENROUTER_API_KEY",
            max_output_tokens=settings.max_output_tokens,
            require_parameters=settings.require_parameters,
            data_collection=settings.data_collection,
            request_timeout_seconds=settings.request_timeout_seconds,
        )
    if settings.kind == "openai":
        return OpenAIProvider(
            settings.model,
            settings.embedding_model or "text-embedding-3-small",
            max_output_tokens=settings.max_output_tokens,
        )
    if settings.kind == "ollama":
        if not settings.base_url:
            raise ValueError("Ollama generation requires provider.base_url")
        return OllamaProvider(
            settings.model,
            settings.embedding_model,
            settings.base_url,
            max_output_tokens=settings.max_output_tokens,
            enable_thinking=settings.enable_thinking,
            context_window=settings.context_window,
            structured_output_mode=settings.structured_output_mode,
        )
    raise ValueError("fake rehearsal generation requires an injected provider")


def _embedding_provider(config: RehearsalConfig) -> EmbeddingProvider:
    settings = config.embedding
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
            raise ValueError("Ollama embeddings require embedding.base_url")
        return OllamaProvider(
            "embedding-only",
            settings.model,
            settings.base_url,
            embedding_batch_size=settings.batch_size,
        )
    raise ValueError("rehearsal embeddings must be local")


def _load_wiki(
    input_root: Path,
    profile: InvestorProfile,
    embedder: EmbeddingProvider,
    aliases: Sequence[str],
) -> tuple[HybridWikiIndex, tuple[str, ...]]:
    path = input_root / "indexes" / f"{profile.vc_slug}.json"
    if path.is_file():
        canonical = HybridWikiIndex.load(
            path,
            profile.wiki_path,
            embedder,
            require_complete_embeddings=True,
        )
    else:
        canonical = HybridWikiIndex.build(
            profile.wiki_path, embedder, require_complete_embeddings=True
        )
        canonical.save(path)
    normalized = tuple(alias.strip() for alias in aliases if alias.strip())
    if not normalized:
        return canonical, ()
    filtered, report = canonical.for_pitch(normalized)
    return filtered, report.quality_findings


def _load_precedents(
    config: RehearsalConfig,
    input_root: Path,
    vc_slug: str,
    embedder: EmbeddingProvider,
    *,
    excluded_episode_slug: str | None,
    excluded_aliases: Sequence[str],
) -> PrecedentCorpus | None:
    if not config.precedents.enabled:
        return None
    corpus_root = input_root / config.precedents.corpus_path.replace(
        "<vc_slug>", vc_slug
    )
    index_path = input_root / "indexes" / f"{vc_slug}.precedents.json"
    if not corpus_root.is_dir():
        raise ValueError(f"precedent corpus is missing: {corpus_root}")
    if index_path.is_file():
        canonical = PrecedentCorpus.load(
            index_path,
            corpus_root,
            embedder,
            require_complete_embeddings=True,
        )
    else:
        canonical = PrecedentCorpus.build(
            corpus_root, embedder, require_complete_embeddings=True
        )
        canonical.save(index_path)
    if canonical.target_slug is not None:
        raise ValueError("canonical precedent index is unexpectedly filtered")
    if excluded_episode_slug is None:
        return canonical
    filtered = canonical.for_target(
        excluded_episode_slug, excluded_aliases=excluded_aliases
    )
    if any(
        row.episode_slug == excluded_episode_slug for row in filtered.list_episodes()
    ):
        raise ValueError("excluded episode remains accessible in precedents")
    return filtered


def _load_portfolio(
    config: RehearsalConfig,
    input_root: Path,
    vc_slug: str,
    embedder: EmbeddingProvider,
    *,
    excluded_episode_slug: str | None,
) -> FilteredPortfolioIndex | None:
    if not config.portfolio_memory.enabled:
        return None
    root = input_root / config.portfolio_memory.corpus_path.replace(
        "<vc_slug>", vc_slug
    )
    events = root / "disclosure-events.jsonl"
    index_path = root / "embedding-index.json"
    if not events.is_file() or not index_path.is_file():
        raise ValueError(f"portfolio memory is incomplete: {root}")
    corpus = PortfolioMemoryCorpus.load_events(vc_slug, events)
    canonical = PortfolioMemoryIndex.load(
        index_path, corpus, embedder, events
    )
    target = excluded_episode_slug or "999999-founder-rehearsal"
    return canonical.for_target(target)


class RuntimeEvidenceRetriever:
    """Search wiki, precedents, and portfolio memory on every interview pass."""

    def __init__(
        self,
        *,
        wiki: HybridWikiIndex,
        precedents: PrecedentCorpus | None,
        portfolio: FilteredPortfolioIndex | None,
        top_k: int,
        max_exact_reads: int,
        precedent_policy: str,
        precedent_candidate_pool_k: int,
        precedent_in_slots: int,
        precedent_out_slots: int,
        portfolio_candidate_pool_k: int,
        question_memory: QuestionMemory,
    ) -> None:
        self.wiki = wiki
        self.precedents = precedents
        self.portfolio = portfolio
        self.top_k = top_k
        self.max_exact_reads = max_exact_reads
        self.precedent_policy = precedent_policy
        self.precedent_candidate_pool_k = precedent_candidate_pool_k
        self.precedent_in_slots = precedent_in_slots
        self.precedent_out_slots = precedent_out_slots
        self.portfolio_candidate_pool_k = portfolio_candidate_pool_k
        self.question_memory = question_memory
        self.query_count = 0

    def retrieve_question_archetypes(
        self,
        query: str,
        *,
        rationale_labels: Sequence[str] = (),
        prior_questions: Sequence[str] = (),
    ) -> tuple[dict[str, Any], ...]:
        """Retrieve compact question exemplars without exposing episode inventory."""
        return tuple(
            hit.model_dump(mode="json")
            for hit in self.question_memory.search(
                query,
                rationale_labels=rationale_labels,
                prior_questions=prior_questions,
            )
        )

    def suggest_question_rationale_labels(
        self, question: str, *, top_k: int = 3
    ) -> tuple[dict[str, str | float], ...]:
        return self.question_memory.suggest_rationale_labels(question, top_k=top_k)

    def question_rationale_alignment_findings(
        self, *, question: str, selected_labels: Sequence[str]
    ) -> tuple[str, ...]:
        return self.question_memory.rationale_alignment_findings(
            question=question, selected_labels=selected_labels
        )

    def resolve_question_rationale_labels(
        self, question: str, *, selected_labels: Sequence[str]
    ) -> tuple[str, ...]:
        return self.question_memory.resolve_rationale_labels(
            question, selected_labels=selected_labels
        )

    def question_rationale_dimension(self, labels: Sequence[str]) -> str:
        from .question_memory import rationale_dimension

        return rationale_dimension(labels, self.question_memory.taxonomy)

    @staticmethod
    def question_evidence_gap_key(
        question: str, *, rationale_labels: Sequence[str] = ()
    ) -> str:
        from .question_memory import question_evidence_gap_key

        return question_evidence_gap_key(question, rationale_labels)

    def retrieve(self, query: str) -> RehearsalEvidenceBundle:
        self.query_count += 1
        wiki_rows: list[dict[str, Any]] = []
        for hit in self.wiki.search(query, self.top_k)[: self.max_exact_reads]:
            exact = self.wiki.read(hit.chunk_id)
            wiki_rows.append(
                {
                    "evidence_id": exact.chunk_id,
                    "source_kind": "wiki",
                    "source_path": exact.source_path,
                    "excerpt": exact.text,
                }
            )
        precedent_rows: list[dict[str, Any]] = []
        if self.precedents is not None:
            result = self.precedents.search(
                query,
                top_k=self.top_k,
                query_id=f"rehearsal-{self.query_count:03d}",
                selection_policy=self.precedent_policy,  # type: ignore[arg-type]
                candidate_pool_k=self.precedent_candidate_pool_k,
                in_slots=self.precedent_in_slots,
                out_slots=self.precedent_out_slots,
                observed_only=True,
                substantive_only=True,
            )
            for hit in result.hits:
                precedent_rows.append(
                    {
                        "evidence_id": hit.chunk_id,
                        "source_kind": "precedent",
                        "source_path": hit.episode_slug,
                        "excerpt": hit.excerpt,
                    }
                )
        portfolio_rows: list[dict[str, Any]] = []
        if self.portfolio is not None:
            disclosures = {
                row.disclosure_id: row for row in self.portfolio.filtered.disclosures
            }
            for candidate in self.portfolio.search(
                query,
                limit=self.top_k,
                candidate_pool_k=self.portfolio_candidate_pool_k,
            ):
                evidence_text = " ".join(
                    evidence.text
                    for disclosure_id in candidate.disclosure_ids
                    for evidence in disclosures[disclosure_id].evidence
                )
                portfolio_rows.append(
                    {
                        "evidence_id": candidate.entity_id,
                        "source_kind": "portfolio",
                        "source_path": "portfolio-memory/disclosure-events.jsonl",
                        "excerpt": " ".join(
                            filter(
                                None,
                                [candidate.company_name or "", candidate.descriptor, evidence_text],
                            )
                        ),
                        "disclosure_ids": list(candidate.disclosure_ids),
                    }
                )
        return RehearsalEvidenceBundle(
            wiki_evidence=tuple(wiki_rows),
            precedent_evidence=tuple(precedent_rows),
            portfolio_evidence=tuple(portfolio_rows),
        )


def resolve_investor(
    config: RehearsalConfig,
    vc_slug: str,
    *,
    target_company_aliases: Sequence[str] = (),
    excluded_episode_slug: str | None = None,
    provider: GenerationProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    grounded_baseline: CanonicalRehearsalBaseline | None = None,
) -> RehearsalRuntime:
    """Load one investor's complete evidence environment for rehearsal."""
    input_root = config.resolve_path(config.rehearsal.input_root)
    profiles = {row.vc_slug: row for row in list_investors(input_root)}
    if vc_slug not in profiles:
        raise ValueError(f"unknown investor profile: {vc_slug}")
    profile = profiles[vc_slug]
    if (
        grounded_baseline is None
        and config.classification.mode == "v41_grounded"
        and excluded_episode_slug
    ):
        registry = config.resolve_path(config.classification.canonical_registry_path)
        try:
            grounded_baseline = load_canonical_baseline(
                workspace=config.workspace,
                registry_path=registry,
                canonical_vc_slug=_classifier_vc_slug(profile),
                episode_slug=excluded_episode_slug,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"canonical baseline unavailable: {exc}") from exc
    generation = provider or _generation_provider(config)
    embeddings = embedder or _embedding_provider(config)
    wiki, sanitization = _load_wiki(
        input_root, profile, embeddings, target_company_aliases
    )
    precedents = _load_precedents(
        config,
        input_root,
        vc_slug,
        embeddings,
        excluded_episode_slug=excluded_episode_slug,
        excluded_aliases=target_company_aliases,
    )
    portfolio = _load_portfolio(
        config,
        input_root,
        vc_slug,
        embeddings,
        excluded_episode_slug=excluded_episode_slug,
    )
    taxonomy = _taxonomy(input_root / config.rehearsal.taxonomy_path)
    investor_aliases = (profile.display_name, profile.display_name.split()[0])
    question_memory = QuestionMemory(
        (
            precedents.question_archetypes(investor_aliases)
            if precedents and config.question_memory.enabled
            else ()
        ),
        embeddings,
        taxonomy=taxonomy,
        settings=config.question_memory,
    )
    retriever = RuntimeEvidenceRetriever(
        wiki=wiki,
        precedents=precedents,
        portfolio=portfolio,
        top_k=config.retrieval.top_k,
        max_exact_reads=config.retrieval.max_exact_reads,
        precedent_policy=config.precedents.selection_policy,
        precedent_candidate_pool_k=config.precedents.candidate_pool_k,
        precedent_in_slots=config.precedents.in_slots,
        precedent_out_slots=config.precedents.out_slots,
        portfolio_candidate_pool_k=config.portfolio_memory.candidate_pool_k,
        question_memory=question_memory,
    )
    classifier_resolution = _classifier_resolution(
        config,
        profile,
        excluded_episode_slug=excluded_episode_slug,
    )
    if (
        config.classification.mode == "classification_informed"
        and classifier_resolution.status == "fallback"
        and not config.classification.fallback_to_rationale_only
    ):
        raise ValueError(
            "classifier required but unavailable: "
            f"{classifier_resolution.reason or 'unknown reason'}"
        )
    return RehearsalRuntime(
        investor=profile,
        provider=generation,
        embedder=embeddings,
        wiki=wiki,
        precedents=precedents,
        portfolio=portfolio,
        question_memory=question_memory,
        taxonomy=taxonomy,
        retriever=retriever,
        sanitization_findings=sanitization,
        classifier_resolution=classifier_resolution,
        grounded_baseline=grounded_baseline,
    )
