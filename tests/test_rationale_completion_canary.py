from __future__ import annotations

import json

import pytest

from vc_clone_graph.providers.base import GenerationResult, Usage
from vc_clone_graph.rationale_completion_canary import (
    conservative_pricing_preflight,
    run_verification_call,
)


class FakeProvider:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return GenerationResult(
            parsed=self.payload,
            content=json.dumps(self.payload),
            usage=Usage(input_tokens=120, output_tokens=80, cost_usd=0.004),
            elapsed_seconds=0.2,
            raw_metadata={"provider": "fake"},
        )


def _payload() -> dict[str, object]:
    return {
        "schema_version": "rationale-completion-verification-v1",
        "episode_slug": "held",
        "dispositions": [
            {
                "taxonomy_label": "market_size_assessment",
                "disposition": "question_only",
                "direction": None,
                "salience": None,
                "confidence": None,
                "justification": "The pitch does not establish a venture-scale market.",
                "pitch_evidence_ids": ["P-001"],
                "investor_evidence_ids": ["W-001"],
            }
        ],
        "validator_findings": [],
    }


def test_verification_call_is_decision_blind_validated_and_auditable(tmp_path):
    provider = FakeProvider(_payload())

    result = run_verification_call(
        provider=provider,
        investor_name="Charles Hudson",
        episode_slug="held",
        pitch_evidence=[{"evidence_id": "P-001", "text": "A narrow initial segment."}],
        canonical_rationales=[{"taxonomy_label": "founder_execution"}],
        candidate_definitions={
            "market_size_assessment": "Whether the market can support venture returns."
        },
        investor_evidence=[
            {"evidence_id": "W-001", "text": "Charles considers venture-scale outcomes."}
        ],
        output_dir=tmp_path,
        max_cost_usd=0.50,
        spent_cost_usd=0.0,
        input_price_per_million=0.22,
        output_price_per_million=1.32,
        max_output_tokens=4096,
    )

    assert result.verification.dispositions[0].disposition == "question_only"
    assert result.cost_usd == pytest.approx(0.004)
    assert len(provider.requests) == 1
    prompt = provider.requests[0].prompt
    assert "actual_decision" not in prompt
    assert "completion_probability" not in prompt
    assert "reference_rationales" not in prompt
    assert json.loads((tmp_path / "verification.json").read_text())["episode_slug"] == "held"
    assert json.loads((tmp_path / "usage.json").read_text())["cost_usd"] == 0.004
    assert (tmp_path / "prompt.txt").read_text() == prompt
    assert (tmp_path / "raw-response.json").is_file()


def test_cost_gate_blocks_before_provider_call(tmp_path):
    provider = FakeProvider(_payload())

    with pytest.raises(ValueError, match="cost cap"):
        run_verification_call(
            provider=provider,
            investor_name="Charles Hudson",
            episode_slug="held",
            pitch_evidence=[{"evidence_id": "P-001", "text": "Pitch."}],
            canonical_rationales=[],
            candidate_definitions={"market_size_assessment": "Definition."},
            investor_evidence=[{"evidence_id": "W-001", "text": "Memory."}],
            output_dir=tmp_path,
            max_cost_usd=0.50,
            spent_cost_usd=0.47,
            input_price_per_million=100_000.0,
            output_price_per_million=100_000.0,
        )

    assert provider.requests == []


def test_pricing_bound_includes_full_schema_and_framing():
    schema = {"type": "object", "description": "x" * 2_000}
    result = conservative_pricing_preflight(
        prompt="short prompt",
        schema=schema,
        max_output_tokens=100,
        input_price_per_million=1.0,
        output_price_per_million=1.0,
        spent_cost_usd=0.0,
        max_cost_usd=1.0,
    )

    assert result["schema_byte_count"] >= 2_000
    assert result["input_token_upper_bound"] >= (
        len("short prompt".encode()) + result["schema_byte_count"] + 4_096
    )


def test_nonpromotion_activation_fields_are_cleared_and_noted(tmp_path):
    payload = _payload()
    disposition = payload["dispositions"][0]
    disposition.update(direction="neutral", salience="secondary", confidence=0.91)
    provider = FakeProvider(payload)

    result = run_verification_call(
        provider=provider,
        investor_name="Charles Hudson",
        episode_slug="held",
        pitch_evidence=[{"evidence_id": "P-001", "text": "Pitch."}],
        canonical_rationales=[],
        candidate_definitions={"market_size_assessment": "Definition."},
        investor_evidence=[{"evidence_id": "W-001", "text": "Memory."}],
        output_dir=tmp_path,
        max_cost_usd=0.50,
        spent_cost_usd=0.0,
        input_price_per_million=0.22,
        output_price_per_million=1.32,
    )

    row = result.verification.dispositions[0]
    assert (row.direction, row.salience, row.confidence) == (None, None, None)
    assert result.verification.validator_findings == (
        "normalized inapplicable activation attributes for market_size_assessment",
    )
    assert json.loads((tmp_path / "raw-response.json").read_text())["parsed"][
        "dispositions"
    ][0]["confidence"] == 0.91


def test_unknown_citations_are_dropped_but_supported_promotion_survives(tmp_path):
    payload = _payload()
    disposition = payload["dispositions"][0]
    disposition.update(
        disposition="activated_candidate",
        direction="negative",
        salience="primary",
        confidence=0.8,
        investor_evidence_ids=["W-001", "H-hallucinated"],
    )
    provider = FakeProvider(payload)

    result = run_verification_call(
        provider=provider,
        investor_name="Charles Hudson",
        episode_slug="held",
        pitch_evidence=[{"evidence_id": "P-001", "text": "Pitch."}],
        canonical_rationales=[],
        candidate_definitions={"market_size_assessment": "Definition."},
        investor_evidence=[{"evidence_id": "W-001", "text": "Memory."}],
        output_dir=tmp_path,
        max_cost_usd=0.50,
        spent_cost_usd=0.0,
        input_price_per_million=0.22,
        output_price_per_million=1.32,
    )

    assert result.verification.dispositions[0].investor_evidence_ids == ("W-001",)
    assert result.verification.validator_findings == (
        "dropped unknown investor evidence IDs for market_size_assessment: H-hallucinated",
    )


def test_promotion_without_remaining_two_sided_evidence_fails_closed(tmp_path):
    payload = _payload()
    disposition = payload["dispositions"][0]
    disposition.update(
        disposition="activated_core",
        direction="positive",
        salience="primary",
        confidence=0.9,
        investor_evidence_ids=["H-hallucinated"],
    )
    provider = FakeProvider(payload)

    with pytest.raises(ValueError, match="promotion requires pitch and investor evidence"):
        run_verification_call(
            provider=provider,
            investor_name="Charles Hudson",
            episode_slug="held",
            pitch_evidence=[{"evidence_id": "P-001", "text": "Pitch."}],
            canonical_rationales=[],
            candidate_definitions={"market_size_assessment": "Definition."},
            investor_evidence=[{"evidence_id": "W-001", "text": "Memory."}],
            output_dir=tmp_path,
            max_cost_usd=0.50,
            spent_cost_usd=0.0,
            input_price_per_million=0.22,
            output_price_per_million=1.32,
        )

    assert (tmp_path / "raw-response.json").is_file()
    assert not (tmp_path / "verification.json").exists()
