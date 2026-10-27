"""Components configuration, kept separate from RunConfig.

RunConfig is serialized into run-config.json and compared exactly during v4.4
verification, so adding a section there would invalidate saved runs. The
components read their own TOML file instead.
"""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class CoverageSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = True
    model: str = Field(default="openai/gpt-5.6-luna", min_length=1)
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = Field(default="OPENROUTER_API_KEY", pattern=r"^[A-Z][A-Z0-9_]*$")
    data_collection: Literal["allow", "deny"] = "deny"
    reasoning_effort: Literal["low", "medium", "high"] | None = "medium"
    max_output_tokens: int = Field(default=4096, ge=256, le=16384)
    max_additions: int = Field(default=8, ge=1, le=20)
    max_policy_chars_per_excerpt: int = Field(default=1200, ge=200, le=5000)


class ConsolidateSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = True
    calibration_path: str = "evaluation/pitchshow/results/components-calibration.json"
    rule: Literal["veto", "tally", "phase2"] = "tally"
    use_coverage: bool = True
    coverage_weight: float = Field(default=1.0, ge=0.0, le=2.0)
    # Per-investor share of pitches that are In (replaces the calibration quota, 30%):
    # higher makes the investor more positive, lower more negative. Decision only;
    # ranking is unchanged. Example: investor_quota = {"elizabeth-yin-hustle-fund" = 0.2}
    investor_quota: dict[str, Annotated[float, Field(gt=0.0, le=1.0)]] = Field(default_factory=dict)


class _ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = Field(default="openai/gpt-5.6-luna", min_length=1)
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = Field(default="OPENROUTER_API_KEY", pattern=r"^[A-Z][A-Z0-9_]*$")
    data_collection: Literal["allow", "deny"] = "deny"
    reasoning_effort: Literal["low", "medium", "high"] | None = "medium"
    max_output_tokens: int = Field(default=4096, ge=256, le=16384)


class FactsheetSettings(_ModelSettings):
    max_facts: int = Field(default=25, ge=1, le=60)


class LensSettings(_ModelSettings):
    # On by default: the contrastive lens met all four pre-specified split-panel criteria
    # (investor-fit and decision-linked recall up; see docs). Reported rationales are the
    # lens output; the decision stays with consolidate (weighted score).
    enabled: bool = True
    # memory: Investment Memory only; precedents: plus the investor's own past
    # remarks (target turns only); facts: plus the founders' in-room fact sheet;
    # profile: precedents plus the investor's distinctive rationales from the
    # annotations of their other episodes; contrastive: profile plus the other
    # panelists' distinctive rationales; fit: precedents plus an explicit
    # investor-fit check.
    # unique-wiki / unique-record: alignment agent on top of the decision agent,
    # using the investor's uniqueness page (plus their on-show record).
    variant: Literal["memory", "precedents", "facts", "profile", "contrastive", "fit",
                     "unique-wiki", "unique-record"] = "contrastive"
    include_memory: bool = True
    annotation_root: str = "rationale_annotation/annotations"
    max_output_tokens: int = Field(default=8192, ge=256, le=16384)
    max_additions: int = Field(default=6, ge=0, le=20)
    # Selective lens: keep at most this many reasons in total (kept + added).
    max_kept: int | None = Field(default=None, ge=1, le=40)
    max_memory_chars: int = Field(default=60000, ge=2000, le=200000)
    precedent_episodes: int = Field(default=5, ge=1, le=10)
    max_precedent_chars: int = Field(default=4000, ge=500, le=20000)
    factsheet_root: str = "outputs/components-lens/factsheets"
    uniqueness_root: str = "outputs/uniqueness/pages"


class ComponentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    coverage: CoverageSettings = CoverageSettings()
    consolidate: ConsolidateSettings = ConsolidateSettings()
    factsheet: FactsheetSettings = FactsheetSettings()
    lens: LensSettings = LensSettings()


def load_components_config(path: Path) -> ComponentsConfig:
    try:
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot load components configuration: {path}") from exc
    return ComponentsConfig.model_validate(raw)
