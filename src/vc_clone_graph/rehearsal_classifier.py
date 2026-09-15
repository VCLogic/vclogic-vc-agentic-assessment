"""Safe JSON classifier artifacts for classification-informed rehearsal."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClassifierScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    probability_in: float = Field(ge=0, le=1)
    predicted_decision: Literal["In", "Out"]
    decision_threshold: float = Field(ge=0, le=1)


class LinearClassifierArtifact(BaseModel):
    """Auditable linear classifier serialized without executable code."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal["rehearsal-linear-classifier-v1"] = Field(alias="schema")
    artifact_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    vc_slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    training_context: Literal["production", "leave_one_episode_out"]
    excluded_episode_slug: str | None = None
    feature_names: tuple[str, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float = Field(allow_inf_nan=False)
    decision_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    training_episode_slugs: tuple[str, ...]
    source_registry: str | None = None
    label_version: str | None = None
    feature_schema: str | None = None
    seed: int | None = None
    class_counts: dict[Literal["Out", "In"], int] | None = None
    selected_config: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "LinearClassifierArtifact":
        lengths = {
            len(self.feature_names),
            len(self.scales),
            len(self.coefficients),
        }
        if len(lengths) != 1:
            raise ValueError("feature_names, scales, and coefficients need the same length")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        if any(not name or "target" in name or "actual" in name for name in self.feature_names):
            raise ValueError("feature_names contain an unsafe or outcome-derived feature")
        if any(not math.isfinite(value) or value <= 0 for value in self.scales):
            raise ValueError("scales must be finite and greater than zero")
        if any(not math.isfinite(value) for value in self.coefficients):
            raise ValueError("coefficients must be finite")
        if self.training_context == "production" and self.excluded_episode_slug is not None:
            raise ValueError("production artifact cannot exclude an episode")
        if self.training_context == "leave_one_episode_out":
            if not self.excluded_episode_slug:
                raise ValueError("leave-one-episode-out artifact requires excluded_episode_slug")
            if self.excluded_episode_slug in self.training_episode_slugs:
                raise ValueError("excluded episode appears in training episodes")
        return self

    def score(self, features: Mapping[str, float]) -> ClassifierScore:
        logit = self.intercept
        for name, scale, coefficient in zip(
            self.feature_names, self.scales, self.coefficients, strict=True
        ):
            value = float(features.get(name, 0.0))
            if not math.isfinite(value):
                raise ValueError(f"runtime feature is not finite: {name}")
            logit += (value / scale) * coefficient
        if logit >= 0:
            probability = 1.0 / (1.0 + math.exp(-logit))
        else:
            exponential = math.exp(logit)
            probability = exponential / (1.0 + exponential)
        return ClassifierScore(
            artifact_id=self.artifact_id,
            probability_in=probability,
            predicted_decision=(
                "In" if probability >= self.decision_threshold else "Out"
            ),
            decision_threshold=self.decision_threshold,
        )


class _RegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(min_length=1)
    vc_slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    training_context: Literal["production", "leave_one_episode_out"]
    excluded_episode_slug: str | None = None
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _RegistryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal["rehearsal-classifier-registry-v1"] = Field(
        alias="schema"
    )
    artifacts: tuple[_RegistryEntry, ...]


@dataclass(frozen=True)
class ClassifierResolution:
    status: Literal["available", "fallback"]
    artifact: LinearClassifierArtifact | None
    reason: str | None = None


class ClassifierRegistry:
    """Validated artifact registry with leakage-safe context selection."""

    def __init__(self, artifacts: tuple[LinearClassifierArtifact, ...]) -> None:
        self._artifacts = artifacts

    @classmethod
    def load(cls, path: Path) -> "ClassifierRegistry":
        registry_path = Path(path).resolve()
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
        document = _RegistryDocument.model_validate(raw)
        loaded: list[LinearClassifierArtifact] = []
        seen: set[tuple[str, str, str | None]] = set()
        for entry in document.artifacts:
            pure = PurePosixPath(entry.path)
            if pure.is_absolute() or ".." in pure.parts or "\\" in entry.path:
                raise ValueError("artifact path must be safe relative")
            artifact_path = (registry_path.parent / pure).resolve()
            if not artifact_path.is_relative_to(registry_path.parent):
                raise ValueError("artifact path must be safe relative")
            content = artifact_path.read_bytes()
            if sha256(content).hexdigest() != entry.sha256:
                raise ValueError(f"artifact digest mismatch: {entry.artifact_id}")
            artifact = LinearClassifierArtifact.model_validate(json.loads(content))
            if artifact.artifact_id != entry.artifact_id:
                raise ValueError("artifact identity does not match registry")
            if artifact.vc_slug != entry.vc_slug:
                raise ValueError("artifact VC identity does not match registry")
            if artifact.training_context != entry.training_context:
                raise ValueError("artifact training context does not match registry")
            if artifact.excluded_episode_slug != entry.excluded_episode_slug:
                raise ValueError("artifact excluded episode does not match registry")
            key = (
                artifact.vc_slug,
                artifact.training_context,
                artifact.excluded_episode_slug,
            )
            if key in seen:
                raise ValueError("registry contains duplicate classifier context")
            seen.add(key)
            loaded.append(artifact)
        return cls(tuple(loaded))

    def resolve(
        self, vc_slug: str, *, excluded_episode_slug: str | None = None
    ) -> LinearClassifierArtifact:
        if excluded_episode_slug is None:
            context = "production"
            matches = [
                artifact
                for artifact in self._artifacts
                if artifact.vc_slug == vc_slug
                and artifact.training_context == context
            ]
            description = "production"
        else:
            matches = [
                artifact
                for artifact in self._artifacts
                if artifact.vc_slug == vc_slug
                and artifact.training_context == "leave_one_episode_out"
                and artifact.excluded_episode_slug == excluded_episode_slug
            ]
            description = f"held-episode {excluded_episode_slug}"
        if not matches:
            raise LookupError(f"no {description} classifier for {vc_slug}")
        if len(matches) != 1:  # pragma: no cover - rejected during load
            raise ValueError(f"ambiguous {description} classifier for {vc_slug}")
        return matches[0]

    def resolve_or_fallback(
        self, vc_slug: str, *, excluded_episode_slug: str | None = None
    ) -> ClassifierResolution:
        try:
            artifact = self.resolve(
                vc_slug, excluded_episode_slug=excluded_episode_slug
            )
        except (LookupError, ValueError) as exc:
            return ClassifierResolution(status="fallback", artifact=None, reason=str(exc))
        return ClassifierResolution(status="available", artifact=artifact)
