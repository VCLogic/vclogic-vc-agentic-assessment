"""Scripted provider for deterministic graph tests."""

from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any, Sequence

from .base import GenerationRequest, GenerationResult, Usage


class FakeProvider:
    def __init__(
        self,
        outputs: Sequence[dict[str, Any]],
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        self._outputs = list(outputs)
        self._embeddings = embeddings or {}
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        started = perf_counter()
        self.requests.append(request)
        if not self._outputs:
            raise RuntimeError("fake provider output script exhausted")
        parsed = self._outputs.pop(0)
        content = json.dumps(parsed, sort_keys=True)
        return GenerationResult(
            parsed=parsed,
            content=content,
            usage=Usage(
                input_tokens=max(1, len(request.prompt) // 4),
                output_tokens=max(1, len(content) // 4),
            ),
            elapsed_seconds=perf_counter() - started,
            raw_metadata={"provider": "fake"},
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embeddings.get(text, [float(len(text)), 1.0]) for text in texts]


class DemoFakeProvider:
    """Dynamic deterministic provider used by CLI smoke tests."""

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        started = perf_counter()
        self.requests.append(request)
        if request.phase == "phase1_plan":
            parsed = {
                "questions": [
                    "How quickly is the team learning from customers?",
                    "Can this become a venture-scale market entry?",
                ],
                "search_queries": ["founder customer learning", "market venture scale"],
                "reconsideration_focus": "Deterministic capability fixture",
            }
        elif request.phase == "phase1":
            chunk_ids = sorted(set(re.findall(r"W-[0-9a-f]{20}", request.prompt)))
            if not chunk_ids:
                raise RuntimeError("fixture received no exact wiki evidence")
            parsed = {
                "episode_slug": "135-thoras-ai-the-twin-effect",
                "questions": ["How quickly is the team learning from customers?"],
                "rationales": [
                    {
                        "rationale_id": "R1",
                        "label": "founder_execution",
                        "direction": "positive",
                        "salience": "primary",
                        "confidence": 0.75,
                        "pitch_evidence": ["paid pilots"],
                        "wiki_evidence_ids": [chunk_ids[0]],
                        "interpretation": "Paid pilots provide an early execution signal.",
                    }
                ],
                "conflicts": ["Production repeatability remains unknown."],
                "unanswered_questions": ["Will pilots convert?"],
                "saturated": True,
                "summary": "Early founder execution is positive with material uncertainty.",
            }
            if "schema_version" in request.schema.get("properties", {}):
                parsed["schema_version"] = "investigation-v2"
                parsed["rationales"][0].update(
                    {
                        "evidence_status": "affirmative_positive",
                        "constraint_kind": "none",
                        "constraint_severity": "none",
                        "severity_basis": "This is a positive signal, not a constraint.",
                    }
                )
                unknown = {"status": "unknown", "value": None, "pitch_evidence": []}
                parsed["deal_context"] = {
                    field: dict(unknown)
                    for field in (
                        "total_round_size",
                        "company_stage",
                        "entry_valuation",
                        "possible_investor_check",
                        "lead_required",
                        "ownership_feasibility",
                    )
                }
        elif request.phase == "phase2":
            match = re.search(r"Investigation SHA-256: ([0-9a-f]{64})", request.prompt)
            if match is None:
                raise RuntimeError("fixture received no investigation hash")
            parsed = {
                "episode_slug": "135-thoras-ai-the-twin-effect",
                "investigation_sha256": match.group(1),
                "decision": "In",
                "investment_likelihood": 0.65,
                "decision_confidence": 0.6,
                "ranking_score": 0.75,
                "check_tier": "small_exploratory",
                "decision_endpoint": "any_check",
                "recommended_check_tier": "small_exploratory",
                "controlling_rationale_ids": ["R1"],
                "any_check": {
                    "decision": "In",
                    "likelihood": 0.65,
                    "confidence": 0.6,
                    "supporting_rationale_ids": ["R1"],
                    "opposing_rationale_ids": [],
                    "fatal_constraint_present": False,
                    "optionality_explanation": "Paid pilots justify a bounded check.",
                    "strongest_counterargument": {
                        "argument": "Production repeatability is unknown.",
                        "rationale_ids": ["R1"],
                        "response": "That limits conviction rather than all participation.",
                    },
                    "reversal_conditions": ["Pilots fail to convert."],
                },
                "standard_check": {
                    "decision": "Out",
                    "likelihood": 0.3,
                    "confidence": 0.75,
                    "supporting_rationale_ids": ["R1"],
                    "opposing_rationale_ids": ["R1"],
                    "failure_rationale_ids": ["R1"],
                    "market_gate": {
                        "status": "unresolved",
                        "controlling_rationale_ids": ["R1"],
                        "explanation": "Scale evidence remains incomplete.",
                    },
                    "strongest_counterargument": {
                        "argument": "Paid pilots may justify a standard check.",
                        "rationale_ids": ["R1"],
                        "response": "They do not establish repeatability or scale.",
                    },
                    "upgrade_conditions": ["Show repeatable conversion and scale."],
                },
                "risk_ledger": [],
                "rationale_assessments": [{
                    "rationale_id": "R1",
                    "effective_direction": "positive",
                    "decision_weight": "decisive",
                    "assessment": "Paid pilots support proceeding.",
                    "counterevidence": ["Production repeatability is unknown."],
                }],
                "deliberation_steps": [
                    {
                        "step_id": f"D{index}",
                        "endpoint": endpoint,
                        "stage": stage,
                        "question": "Does the execution evidence clear the bar?",
                        "rationale_ids": ["R1"],
                        "evidence_assessment": "The signal is positive but early.",
                        "likelihood_before": before,
                        "likelihood_after": after,
                        "effect": effect,
                        "check_tier_implication": tier,
                        "decision_update": "Move toward a small exploratory check.",
                    }
                    for index, endpoint, stage, before, after, effect, tier in [
                        (1, "any_check", "assessment", 0.45, 0.55, "raises", "small_exploratory"),
                        (2, "any_check", "opposing_case", 0.55, 0.60, "raises", "small_exploratory"),
                        (3, "any_check", "consistency", 0.60, 0.65, "raises", "small_exploratory"),
                        (4, "standard_check", "assessment", 0.5, 0.4, "lowers", "standard_initial"),
                        (5, "standard_check", "opposing_case", 0.4, 0.35, "lowers", "standard_initial"),
                        (6, "standard_check", "consistency", 0.35, 0.3, "lowers", "no_check_tier"),
                    ]
                ],
                "strongest_counterargument": "Production repeatability is unknown.",
                "unresolved_questions": ["Will pilots convert?"],
                "reversal_conditions": ["Pilots fail to convert."],
                "feedback": "Proceed to focused diligence.",
            }
            if "schema_version" in request.schema.get("properties", {}):
                parsed["schema_version"] = "decision-v2"
        else:
            raise RuntimeError(f"unsupported fixture phase: {request.phase}")
        content = json.dumps(parsed, sort_keys=True)
        return GenerationResult(
            parsed=parsed,
            content=content,
            usage=Usage(
                input_tokens=max(1, len(request.prompt) // 4),
                output_tokens=max(1, len(content) // 4),
            ),
            elapsed_seconds=perf_counter() - started,
            raw_metadata={"provider": "fake", "fixture": "dynamic"},
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [
                float(len(text)),
                float(text.lower().count("founder")),
                float(text.lower().count("market")),
            ]
            for text in texts
        ]
