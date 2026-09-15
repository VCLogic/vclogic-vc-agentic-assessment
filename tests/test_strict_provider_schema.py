from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from vc_clone_graph.providers.base import GenerationRequest
from vc_clone_graph.providers.openrouter import OpenRouterProvider
from vc_clone_graph.graph import VCDecisionWorkflow
from vc_clone_graph.schemas import (
    DecisionV3,
    DealFact,
    Phase1RetrievalPlan,
    Phase2RetrievalPlan,
    TranscriptReadRequest,
    decision_v2_json_schema,
    decision_json_schema,
    decision_v3_json_schema,
    investigation_json_schema,
    investigation_v2_json_schema,
    investigation_v3_json_schema,
)


WIKI_ID = "W-12345678901234567890"
HISTORICAL_ID = "H-12345678901234567890"


def _assert_openai_strict_schema(value: object, path: tuple[object, ...] = ()) -> None:
    if isinstance(value, dict):
        if "anyOf" in value:
            raise AssertionError(f"unsupported anyOf at {path}")
        if "uniqueItems" in value:
            raise AssertionError(f"unsupported uniqueItems at {path}")
        for keyword in ("minLength", "maxLength"):
            if keyword in value:
                raise AssertionError(f"unsupported {keyword} at {path}")
        assert "default" not in value, f"unsupported default at {path}"
        if value.get("type") == "object" or "properties" in value:
            properties = value.get("properties", {})
            assert value.get("additionalProperties") is False, path
            assert value.get("required") == list(properties), path
        for key, nested in value.items():
            _assert_openai_strict_schema(nested, (*path, key))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_openai_strict_schema(nested, (*path, index))


def _model_facing_v3_schemas() -> dict[str, dict]:
    return {
        "phase1_plan": Phase1RetrievalPlan.model_json_schema(),
        "phase2_plan": Phase2RetrievalPlan.model_json_schema(),
        "investigation": investigation_v3_json_schema(
            episode_slug="135-thoras-ai-the-twin-effect",
            taxonomy_labels={"founder_execution"},
            exact_wiki_ids={WIKI_ID},
            exact_historical_ids={HISTORICAL_ID},
        ),
        "decision": decision_v3_json_schema(
            episode_slug="135-thoras-ai-the-twin-effect",
            investigation_sha256="a" * 64,
            rationale_ids={"R1"},
            check_tiers={"exploratory_lt_100k"},
            exact_historical_ids={HISTORICAL_ID},
            available_precedent_slugs={"20-harper-wilde"},
        ),
    }


def test_raw_phase1_plan_reproduces_openrouter_anyof_rejection() -> None:
    with pytest.raises(
        AssertionError,
        match=r"unsupported anyOf at \('\$defs', 'TranscriptReadRequest'",
    ):
        _assert_openai_strict_schema(Phase1RetrievalPlan.model_json_schema())


@pytest.mark.parametrize("phase", list(_model_facing_v3_schemas()))
def test_generation_request_normalizes_every_model_facing_v3_schema(phase: str) -> None:
    raw = _model_facing_v3_schemas()[phase]
    original = deepcopy(raw)

    request = GenerationRequest(phase, "prompt", raw)

    _assert_openai_strict_schema(request.schema)
    assert raw == original


def test_nullable_scalars_keep_constraints_and_runtime_semantics() -> None:
    request = GenerationRequest(
        "phase1_plan", "prompt", Phase1RetrievalPlan.model_json_schema()
    )
    turn_start = request.schema["$defs"]["TranscriptReadRequest"]["properties"][
        "turn_start"
    ]
    assert turn_start["type"] == ["integer", "null"]
    assert turn_start["minimum"] == 0
    assert "anyOf" not in turn_start
    assert TranscriptReadRequest(
        episode_slug="20-harper-wilde", turn_start=0, turn_end=1
    ).turn_start == 0
    assert TranscriptReadRequest(
        episode_slug="20-harper-wilde", turn_start=None, turn_end=None
    ).turn_start is None

    investigation = GenerationRequest(
        "phase1",
        "prompt",
        _model_facing_v3_schemas()["investigation"],
    ).schema
    value = investigation["$defs"]["DealFact"]["properties"]["value"]
    assert value["type"] == ["string", "null"]
    assert DealFact(status="observed", value="Seed", pitch_evidence=["Seed"]).value == "Seed"
    assert DealFact(status="unknown", value=None, pitch_evidence=[]).value is None


def test_provider_schema_drops_unique_items_without_mutating_runtime_schema() -> None:
    raw = _model_facing_v3_schemas()["investigation"]
    investigation = GenerationRequest(
        "phase1",
        "prompt",
        raw,
    ).schema

    assert "uniqueItems" not in investigation["properties"]["activated_candidates"]
    assert raw["properties"]["activated_candidates"]["uniqueItems"] is True


def test_non_nullable_union_fails_locally_before_provider_call() -> None:
    with pytest.raises(ValueError, match="unsupported provider schema union.*anyOf"):
        GenerationRequest(
            "test",
            "prompt",
            {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        )


def test_openrouter_receives_and_archive_records_the_request_schema(
    tmp_path,
) -> None:
    captured: dict = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
                usage=SimpleNamespace(),
            )

    request = GenerationRequest(
        "phase1_plan", "prompt", Phase1RetrievalPlan.model_json_schema()
    )
    provider = OpenRouterProvider(
        "model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),  # type: ignore[arg-type]
    )

    result = provider.generate(request)

    workflow = object.__new__(VCDecisionWorkflow)
    workflow.settings = SimpleNamespace(run_root=tmp_path)
    workflow._record_call("phase1", 1, "plan", request, result)

    sent = captured["response_format"]["json_schema"]["schema"]
    archived = json.loads(
        (tmp_path / "phase1/turn-01/plan-model-response.json").read_text()
    )["schema"]
    assert sent == request.schema
    assert archived == sent
    _assert_openai_strict_schema(sent)


def test_v1_and_v2_schemas_remain_compatible_with_strict_requests() -> None:
    schemas = [
        investigation_json_schema(
            episode_slug="135-thoras-ai-the-twin-effect",
            taxonomy_labels={"founder_execution"},
            exact_wiki_ids={WIKI_ID},
        ),
        investigation_v2_json_schema(
            episode_slug="135-thoras-ai-the-twin-effect",
            taxonomy_labels={"founder_execution"},
            exact_wiki_ids={WIKI_ID},
        ),
        decision_v2_json_schema(
            episode_slug="135-thoras-ai-the-twin-effect",
            investigation_sha256="a" * 64,
            rationale_ids={"R1"},
            check_tiers={"exploratory_lt_100k"},
        ),
        decision_json_schema(
            episode_slug="135-thoras-ai-the-twin-effect",
            investigation_sha256="a" * 64,
            rationale_ids={"R1"},
            check_tiers={"exploratory_lt_100k"},
        ),
    ]

    for schema in schemas:
        _assert_openai_strict_schema(GenerationRequest("legacy", "prompt", schema).schema)
    assert DecisionV3.model_json_schema()["properties"]["schema_version"]["const"] == (
        "decision-v3"
    )
