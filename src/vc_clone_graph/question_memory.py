"""Target-excluded investor question archetypes for founder rehearsal."""

from __future__ import annotations

from hashlib import sha256
import re
from typing import Literal, Sequence, Set

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .precedents import PrecedentEpisode
from .providers.base import EmbeddingProvider, embed_documents, embed_queries
from .rehearsal_config import RehearsalQuestionMemorySettings


class RationaleAffinity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    similarity: float = Field(ge=0, le=1)


class HybridScoreComponents(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic: float
    rationale: float
    lexical: float
    atomicity: float
    length: float
    multiple_requests: float
    duplicate: float


class QuestionArchetype(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    archetype_id: str = Field(pattern=r"^HQ-[0-9a-f]{16}$")
    episode_slug: str
    turn_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    context: str
    word_count: int = Field(default=1, ge=1)
    atomicity_findings: tuple[str, ...] = ()
    rationale_affinities: tuple[RationaleAffinity, ...] = ()


class QuestionArchetypeHit(QuestionArchetype):
    similarity: float = Field(ge=0, le=1)
    semantic_similarity: float = Field(ge=0, le=1)
    hybrid_score: float
    score_components: HybridScoreComponents


def _identity(episode_slug: str, turn_index: int, text: str) -> str:
    material = f"{episode_slug}\0{turn_index}\0{text}".encode("utf-8")
    return "HQ-" + sha256(material).hexdigest()[:16]


def extract_question_archetypes(
    episodes: Sequence[PrecedentEpisode],
    *,
    investor_aliases: Set[str],
) -> tuple[QuestionArchetype, ...]:
    """Extract investor questions while physically excluding decision turns."""
    names = {value.casefold().strip() for value in investor_aliases if value.strip()}
    rows: list[QuestionArchetype] = []
    for episode in episodes:
        decision_turns = {
            turn
            for evidence in episode.decision.evidence
            for turn in range(evidence.turn_start, evidence.turn_end + 1)
        }
        for position, turn in enumerate(episode.turns):
            if (
                turn.turn_index in decision_turns
                or turn.speaker.casefold().strip() not in names
                or "?" not in turn.text
            ):
                continue
            context_turns = [
                candidate.text.strip()
                for candidate in episode.turns[max(0, position - 2) : position]
                if candidate.turn_index not in decision_turns and candidate.text.strip()
            ]
            text = turn.text.strip()
            rows.append(
                QuestionArchetype(
                    archetype_id=_identity(
                        episode.episode_slug, turn.turn_index, text
                    ),
                    episode_slug=episode.episode_slug,
                    turn_index=turn.turn_index,
                    text=text,
                    context="\n".join(context_turns),
                    word_count=len(text.split()),
                    atomicity_findings=atomic_question_findings(text),
                )
            )
    rows.sort(key=lambda row: (row.episode_slug, row.turn_index, row.archetype_id))
    return tuple(rows)


_TOKEN = re.compile(r"[a-z0-9]+")

_SUBJECT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "traction_repeatability_concern",
        re.compile(
            r"\b(?:outbound|acquisition|sales|channel)\b.{0,80}"
            r"\b(?:convert|conversion|repeat|pattern|scale|headcount)\b|"
            r"\b(?:convert|conversion|repeat|pattern|scale)\b.{0,80}"
            r"\b(?:outbound|acquisition|sales|channel)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "unit_economics_assessment",
        re.compile(
            r"\b(?:roi|return on investment|payback|labor savings?|cost savings?)\b|"
            r"\b(?:measurable|verified) (?:result|outcome|value)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "competitive_advantage_assessment",
        re.compile(
            r"\b(?:harder to displace|switching costs?|defensib(?:le|ility)|"
            r"competitive advantage|moat)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "go_to_market_strategy_assessment",
        re.compile(
            r"\b(?:outbound|customer acquisition|sales motion|sales channel|"
            r"distribution channel|go[- ]to[- ]market)\b",
            re.IGNORECASE,
        ),
    ),
)

_STRUCTURAL_LABEL_CUES: dict[str, re.Pattern[str]] = {
    "stage_valuation_mismatch": re.compile(
        r"\b(?:valuation|post[- ]money|pre[- ]money|valuation cap|round price|"
        r"priced round|deal terms?)\b",
        re.IGNORECASE,
    ),
    "fund_economics_constraint": re.compile(
        r"\b(?:fund size|check size|ownership|allocation|reserve|return the fund|"
        r"fund economics)\b",
        re.IGNORECASE,
    ),
    "portfolio_conflict": re.compile(
        r"\b(?:portfolio|conflict|overlap|already invested|existing investment)\b",
        re.IGNORECASE,
    ),
}


def question_subject_label(text: str, allowed_labels: Set[str]) -> str | None:
    """Return a high-precision taxonomy label for recognizable question subjects."""
    for label, pattern in _SUBJECT_RULES:
        if label in allowed_labels and pattern.search(text):
            return label
    return None


def _label_compatible_with_question(label: str, text: str) -> bool:
    required = _STRUCTURAL_LABEL_CUES.get(label)
    return required is None or bool(required.search(text))


def question_evidence_gap_key(
    text: str, rationale_labels: Sequence[str] = ()
) -> str:
    """Normalize differently worded questions that test the same evidence gap."""
    lowered = text.casefold()
    if re.search(r"\b(?:outbound|acquisition|sales|channel)\b", lowered) and re.search(
        r"\b(?:convert\w*|repeat\w*|pattern|scale|headcount)\b", lowered
    ):
        return "acquisition_repeatability"
    if re.search(
        r"\b(?:roi|return on investment|baseline|measurable result|"
        r"customer outcome|labor savings?|cost savings?)\b",
        lowered,
    ):
        return "customer_value_proof"
    if re.search(
        r"\b(?:displac\w*|switching costs?|defensib\w*|competitive advantage|moat)\b",
        lowered,
    ):
        return "competitive_durability"
    if rationale_labels:
        return "rationale:" + "+".join(sorted(dict.fromkeys(rationale_labels)))
    normalized = _normalized_text(text)
    return "question:" + sha256(normalized.encode("utf-8")).hexdigest()[:16]


def rationale_dimension(
    labels: Sequence[str], taxonomy: Sequence[dict[str, str]]
) -> str:
    """Return one neutral public dimension from corrected taxonomy labels."""
    parent_by_label = {
        row["label"]: row.get("coarse_parent", "investment assessment")
        for row in taxonomy
    }
    parent = parent_by_label.get(labels[0], "investment assessment") if labels else "investment assessment"
    return parent.replace("_", " ").title()


def _normalized_text(value: str) -> str:
    return " ".join(_TOKEN.findall(value.casefold()))


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(_TOKEN.findall(left.casefold()))
    right_tokens = set(_TOKEN.findall(right.casefold()))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    similarity = float(left @ right / denominator) if denominator else 0.0
    return min(1.0, max(0.0, similarity))


class QuestionMemory:
    """Small local semantic index over exact historical investor questions."""

    def __init__(
        self,
        archetypes: Sequence[QuestionArchetype],
        embedder: EmbeddingProvider,
        *,
        taxonomy: Sequence[dict[str, str]] = (),
        settings: RehearsalQuestionMemorySettings | None = None,
    ) -> None:
        self.embedder = embedder
        self.settings = settings or RehearsalQuestionMemorySettings()
        original = tuple(archetypes)
        documents = [row.text for row in original]
        self.vectors = (
            np.asarray(embed_documents(embedder, documents), dtype=float)
            if documents
            else np.empty((0, 0), dtype=float)
        )
        taxonomy_rows = tuple(taxonomy)
        taxonomy_vectors = (
            np.asarray(
                embed_documents(
                    embedder,
                    [f"{row['label']}: {row['definition']}" for row in taxonomy_rows],
                ),
                dtype=float,
            )
            if taxonomy_rows
            else np.empty((0, 0), dtype=float)
        )
        self.taxonomy = taxonomy_rows
        self.taxonomy_vectors = taxonomy_vectors
        enriched: list[QuestionArchetype] = []
        self._affinity_by_id: dict[str, dict[str, float]] = {}
        for index, row in enumerate(original):
            affinities: list[RationaleAffinity] = []
            full_affinities: dict[str, float] = {}
            for taxonomy_index, taxonomy_row in enumerate(taxonomy_rows):
                similarity = _cosine(self.vectors[index], taxonomy_vectors[taxonomy_index])
                full_affinities[taxonomy_row["label"]] = similarity
                affinities.append(
                    RationaleAffinity(
                        label=taxonomy_row["label"], similarity=similarity
                    )
                )
            affinities.sort(key=lambda item: (item.similarity, item.label), reverse=True)
            self._affinity_by_id[row.archetype_id] = full_affinities
            enriched.append(
                QuestionArchetype(
                    **{
                        **row.model_dump(),
                        "word_count": len(row.text.split()),
                        "atomicity_findings": atomic_question_findings(row.text),
                        "rationale_affinities": tuple(affinities[:3]),
                    }
                )
            )
        self.archetypes = tuple(enriched)

    def suggest_rationale_labels(
        self, question: str, *, top_k: int = 3
    ) -> tuple[dict[str, str | float], ...]:
        """Rank taxonomy labels by semantic alignment with a founder question."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not self.taxonomy:
            return ()
        vector = np.asarray(embed_queries(self.embedder, [question])[0], dtype=float)
        ranked = sorted(
            (
                {
                    "label": row["label"],
                    "similarity": _cosine(vector, taxonomy_vector),
                }
                for row, taxonomy_vector in zip(
                    self.taxonomy, self.taxonomy_vectors, strict=True
                )
            ),
            key=lambda row: (float(row["similarity"]), str(row["label"])),
            reverse=True,
        )
        return tuple(ranked[:top_k])

    def rationale_alignment_findings(
        self,
        *,
        question: str,
        selected_labels: Sequence[str],
        minimum_margin: float = 0.12,
    ) -> tuple[str, ...]:
        """Flag a clearly mismatched label without treating similarity as ground truth."""
        if not self.taxonomy or not selected_labels:
            return ()
        ranked = self.suggest_rationale_labels(question, top_k=len(self.taxonomy))
        similarities = {
            str(row["label"]): float(row["similarity"]) for row in ranked
        }
        selected = max(
            selected_labels,
            key=lambda label: (similarities.get(label, -1.0), label),
        )
        best = ranked[0]
        best_label = str(best["label"])
        if (
            best_label not in selected_labels
            and float(best["similarity"])
            >= similarities.get(selected, -1.0) + minimum_margin
        ):
            return (f"rationale_alignment:{selected}->{best_label}",)
        return ()

    def resolve_rationale_labels(
        self, question: str, *, selected_labels: Sequence[str]
    ) -> tuple[str, ...]:
        """Correct labels whose required subject matter is absent from the question."""
        allowed = {row["label"] for row in self.taxonomy}
        subject = question_subject_label(question, allowed)
        if subject is not None:
            return (subject,)
        compatible = tuple(
            dict.fromkeys(
                label
                for label in selected_labels
                if label in allowed and _label_compatible_with_question(label, question)
            )
        )
        if compatible:
            return compatible
        return tuple(
            str(row["label"])
            for row in self.suggest_rationale_labels(question, top_k=len(allowed))
            if _label_compatible_with_question(str(row["label"]), question)
        )[:1]

    def search(
        self,
        query: str,
        *,
        rationale_labels: Sequence[str] = (),
        prior_questions: Sequence[str] = (),
        top_k: int | None = None,
        policy: Literal["hybrid", "semantic"] = "hybrid",
    ) -> tuple[QuestionArchetypeHit, ...]:
        limit = self.settings.top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k must be positive")
        if not self.archetypes:
            return ()
        query_vector = np.asarray(embed_queries(self.embedder, [query])[0], dtype=float)
        scored: list[tuple[float, QuestionArchetype]] = []
        for vector, row in zip(self.vectors, self.archetypes):
            scored.append((_cosine(vector, query_vector), row))
        scored.sort(key=lambda pair: (pair[0], pair[1].archetype_id), reverse=True)
        pool = scored[: self.settings.candidate_pool_k]
        requested_labels = tuple(dict.fromkeys(rationale_labels))
        prior_normalized = {_normalized_text(value) for value in prior_questions}
        hits: list[QuestionArchetypeHit] = []
        for semantic, row in pool:
            rationale = max(
                (
                    self._affinity_by_id.get(row.archetype_id, {}).get(label, 0.0)
                    for label in requested_labels
                ),
                default=0.0,
            )
            lexical = _token_jaccard(query, row.text)
            findings = set(row.atomicity_findings)
            components = HybridScoreComponents(
                semantic=self.settings.semantic_weight * semantic,
                rationale=self.settings.rationale_weight * rationale,
                lexical=self.settings.lexical_weight * lexical,
                atomicity=(self.settings.atomicity_bonus if not findings else 0.0),
                length=(
                    -self.settings.long_question_penalty
                    if row.word_count > 35
                    else 0.0
                ),
                multiple_requests=(
                    -self.settings.multi_request_penalty
                    if findings
                    & {"interrogative_count", "semicolon_bundle", "multiple_requests"}
                    else 0.0
                ),
                duplicate=(
                    -self.settings.duplicate_penalty
                    if _normalized_text(row.text) in prior_normalized
                    else 0.0
                ),
            )
            hybrid = sum(components.model_dump().values())
            score = semantic if policy == "semantic" else hybrid
            if policy == "hybrid" and score < self.settings.minimum_hybrid_score:
                continue
            hits.append(
                QuestionArchetypeHit(
                    **row.model_dump(),
                    similarity=semantic,
                    semantic_similarity=semantic,
                    hybrid_score=hybrid,
                    score_components=components,
                )
            )
        hits.sort(
            key=lambda row: (
                row.semantic_similarity if policy == "semantic" else row.hybrid_score,
                row.semantic_similarity,
                row.archetype_id,
            ),
            reverse=True,
        )
        return tuple(hits[:limit])


_SECOND_REQUEST = re.compile(
    r"\b(?:and|or)\s+(?:what|how|which|when|where|who|why|do|does|did|is|are|can|could|would|will|have|has)\b",
    re.IGNORECASE,
)

_INVESTOR_MEMORY_REQUEST = re.compile(
    r"(?:"
    r"\b(?:investable|investment|portfolio|fund|thesis|opportunit(?:y|ies))\b"
    r".{0,120}\b(?:you|your)\s+(?:invest|exclude|back|fund|prefer|avoid)\b"
    r"|"
    r"\b(?:you|your)\s+(?:invest|exclude|back|fund|prefer|avoid)\b"
    r".{0,120}\b(?:investable|investment|portfolio|fund|thesis|opportunit(?:y|ies))\b"
    r"|"
    r"\b(?:the\s+investor|investor(?:'s)?)\s+"
    r"(?:thesis|portfolio|precedent|fund|constraint|policy|exclusion)\b"
    r")",
    re.IGNORECASE,
)

_MEMO_LANGUAGE = {
    "independently_verified": re.compile(r"\bindependently verified\b", re.IGNORECASE),
    "customer_cohorts": re.compile(r"\bcustomer cohorts?\b", re.IGNORECASE),
    "pre_implementation_baseline": re.compile(
        r"\brelative to\b.{0,80}\bpre[- ]implementation\b.{0,80}\bbaseline\b",
        re.IGNORECASE,
    ),
}


def atomic_question_findings(text: str) -> tuple[str, ...]:
    """Return non-fatal findings when a founder question bundles requests."""
    findings: list[str] = []
    if text.count("?") != 1:
        findings.append("interrogative_count")
    if len(text.split()) > 35:
        findings.append("too_long")
    if ";" in text:
        findings.append("semicolon_bundle")
    if _SECOND_REQUEST.search(text):
        findings.append("multiple_requests")
    if _INVESTOR_MEMORY_REQUEST.search(text):
        findings.append("investor_memory_request")
    return tuple(findings)


def spoken_question_findings(text: str) -> tuple[str, ...]:
    """Return non-fatal findings for memo-like rather than spoken founder questions."""
    findings = [
        f"memo_language:{label}"
        for label, pattern in _MEMO_LANGUAGE.items()
        if pattern.search(text)
    ]
    if len(text.split()) > 20:
        findings.append("spoken_style_length")
    return tuple(findings)
