"""Strict output contract for the context-rich, one-call baseline."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vc_clone_graph.schemas import (
    AnyCheckDecision,
    DeliberationStep,
    RationaleAssessment,
    RiskLedgerEntry,
    StandardCheckDecision,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaselineRationale(StrictModel):
    rationale_id: str = Field(pattern=r"^R[1-9][0-9]*$")
    label: str = Field(min_length=1)
    direction: Literal["positive", "negative", "neutral"]
    salience: Literal["primary", "secondary"]
    confidence: float = Field(ge=0, le=1)
    pitch_evidence: list[str] = Field(min_length=1)
    wiki_evidence: list[str] = Field(min_length=1)
    precedent_evidence: list[str]
    interpretation: str = Field(min_length=1)


class BaselineDecision(StrictModel):
    decision: Literal["In", "Out"]
    investment_likelihood: float = Field(ge=0, le=1)
    decision_confidence: float = Field(ge=0, le=1)
    ranking_score: float = Field(ge=0, le=1)
    check_tier: str = Field(min_length=1)
    decision_endpoint: Literal["any_check"]
    recommended_check_tier: str = Field(min_length=1)
    controlling_rationale_ids: list[str] = Field(min_length=1)
    any_check: AnyCheckDecision
    standard_check: StandardCheckDecision
    risk_ledger: list[RiskLedgerEntry]
    rationale_assessments: list[RationaleAssessment] = Field(min_length=1)
    deliberation_steps: list[DeliberationStep] = Field(min_length=3, max_length=8)
    strongest_counterargument: str = Field(min_length=1)
    unresolved_questions: list[str]
    reversal_conditions: list[str]
    feedback: str = Field(min_length=1)
    stable: bool
    decisive_precedents: list[str]
    exception_analogies: list[str]


def _decision_references(decision: BaselineDecision) -> set[str]:
    refs = set(decision.controlling_rationale_ids)
    refs.update(row.rationale_id for row in decision.rationale_assessments)
    refs.update(row.rationale_id for row in decision.risk_ledger)
    for step in decision.deliberation_steps:
        refs.update(step.rationale_ids)
    for endpoint in (decision.any_check, decision.standard_check):
        refs.update(endpoint.supporting_rationale_ids)
        refs.update(endpoint.opposing_rationale_ids)
        refs.update(endpoint.strongest_counterargument.rationale_ids)
    refs.update(decision.standard_check.failure_rationale_ids)
    refs.update(decision.standard_check.market_gate.controlling_rationale_ids)
    return refs


class OneShotBaselineResponse(StrictModel):
    schema_version: Literal["one-shot-baseline-v1"]
    episode_slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    rationales: list[BaselineRationale] = Field(min_length=1)
    decision: BaselineDecision

    @model_validator(mode="after")
    def bind_decision_to_rationales(self) -> "OneShotBaselineResponse":
        ids = [row.rationale_id for row in self.rationales]
        if len(ids) != len(set(ids)):
            raise ValueError("baseline rationale IDs must be unique")
        unknown = _decision_references(self.decision) - set(ids)
        if unknown:
            raise ValueError(f"decision references unknown baseline rationale: {sorted(unknown)[0]}")
        return self

    def validate_taxonomy(self, taxonomy_labels: set[str]) -> None:
        unknown = {row.label for row in self.rationales} - taxonomy_labels
        if unknown:
            raise ValueError(f"unknown taxonomy label: {sorted(unknown)[0]}")


def baseline_json_schema(episode_slug: str, taxonomy_labels: set[str]) -> dict:
    schema = deepcopy(OneShotBaselineResponse.model_json_schema())
    schema["properties"]["episode_slug"]["const"] = episode_slug
    schema["$defs"]["BaselineRationale"]["properties"]["label"]["enum"] = sorted(taxonomy_labels)
    return schema
