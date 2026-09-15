from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from vc_clone_graph.providers.base import GenerationRequest
from tests.test_phase1_v44_contract import (
    adjudication_payload,
    claim_map_model,
    constraint_assessment_payload,
    disposition,
    neighborhood_model,
    retrieval_model,
)


SCRIPT = Path(__file__).parents[1] / "scripts/replay_phase1_v44_adjudications.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("replay_phase1_v44", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_attempt(root: Path, *, duplicate: bool = False) -> None:
    phase1 = root / "phase1"
    call_root = phase1 / "adjudication/turn-01"
    call_root.mkdir(parents=True)
    claim = claim_map_model()
    retrieval = retrieval_model()
    neighborhood = neighborhood_model()
    (phase1 / "claim-map.json").write_text(claim.model_dump_json())
    (phase1 / "claim-retrieval.json").write_text(retrieval.model_dump_json())
    (phase1 / "taxonomy-neighborhood.json").write_text(
        neighborhood.model_dump_json()
    )
    adjudication = adjudication_payload()
    repeated = disposition(
        "founder_execution", "candidate", claim_ids=["C2"]
    )
    repeated.update(
        {
            "pitch_evidence_ids": ["P-002"],
            "wiki_evidence_ids": [],
            "historical_evidence_ids": ["H-renewal"],
            "direction": "negative",
            "salience": "secondary",
            "confidence": 0.65,
        }
    )
    adjudication["dispositions"].append(repeated)
    if duplicate:
        adjudication["dispositions"].append(
            json.loads(json.dumps(adjudication["dispositions"][0]))
        )
    constraint = constraint_assessment_payload()
    constraint["pitch_evidence_ids"] = ["P-001"]
    constraint["mapped_ids"] = ["R99"]
    adjudication["constraint_assessments"] = [constraint]
    call = {
        "phase": "phase1_adjudication",
        "parsed": adjudication,
        "raw": json.dumps(adjudication, sort_keys=True),
        "usage": {
            "input_tokens": 100,
            "cached_input_tokens": 0,
            "output_tokens": 50,
            "cost_usd": 0.125,
        },
        "cost_usd": 0.125,
    }
    (call_root / "call-01.json").write_text(json.dumps(call))
    (root / "state.json").write_text(
        json.dumps({"usage": call["usage"], "phase1_status": "failed"})
    )


def test_replay_attempt_preserves_sources_and_finalizes_instances(
    tmp_path: Path,
) -> None:
    module = _load_module()
    root = tmp_path / "attempt"
    _write_attempt(root, duplicate=True)
    before = module.source_hash_inventory(root)

    result = module.replay_attempt(root)

    assert module.source_hash_inventory(root) == before
    assert result["source_files_unchanged"] is True
    assert result["raw_disposition_count"] == 5
    assert result["normalized_disposition_count"] == 4
    assert result["activated_instance_count"] == 3
    assert result["exact_duplicate_dispositions_removed"] == 1
    assert not any(
        value.startswith("DUPLICATE_DISPOSITION")
        for value in result["findings"]
    )
    assert result["constraint_mapping_revisions"][0][
        "original_mapped_ids"
    ] == ["R99"]
    assert result["constraint_mapping_revisions"][0][
        "pitch_evidence_ids"
    ] == ["P-001"]
    assert result["constraint_mapping_revisions"][0][
        "recomputed_mapped_ids"
    ] == ["R1", "R3"]
    assert result["api_calls"] == 0
    assert result["api_cost_usd"] == 0.0
    assert result["recorded_source_cost_usd"] == pytest.approx(0.125)


def test_replay_rejects_less_than_three_or_duplicate_attempt_roots(
    tmp_path: Path,
) -> None:
    module = _load_module()
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_attempt(first)
    _write_attempt(second)

    with pytest.raises(ValueError, match="exactly three distinct"):
        module.run_replays([first, second], tmp_path / "report")
    with pytest.raises(ValueError, match="exactly three distinct"):
        module.run_replays([first, second, first], tmp_path / "report")


def test_run_replays_writes_report_and_csv_without_mutating_attempts(
    tmp_path: Path,
) -> None:
    module = _load_module()
    roots = [tmp_path / f"attempt-{index}" for index in range(3)]
    for index, root in enumerate(roots):
        _write_attempt(root, duplicate=index == 2)
    before = {str(root): module.source_hash_inventory(root) for root in roots}
    output = tmp_path / "report"

    report = module.run_replays(roots, output)

    assert report["attempt_count"] == 3
    assert report["all_source_files_unchanged"] is True
    assert report["new_api_calls"] == 0
    assert report["new_api_cost_usd"] == 0.0
    assert (output / "replay-report.json").is_file()
    assert (output / "replay-cases.csv").is_file()
    assert (output / "source-hashes-before.json").is_file()
    assert (output / "source-hashes-after.json").is_file()
    assert {str(root): module.source_hash_inventory(root) for root in roots} == before


def test_source_inventory_rejects_symlinks(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "attempt"
    root.mkdir()
    external = tmp_path / "external.json"
    external.write_text("{}")
    (root / "linked.json").symlink_to(external)

    with pytest.raises(ValueError, match="symlink"):
        module.source_hash_inventory(root)


def test_recorded_replay_provider_is_zero_cost_and_phase_bound() -> None:
    module = _load_module()
    provider = module.RecordedReplayProvider(
        [
            {
                "phase": "phase1_claim_extraction",
                "parsed": {"answer": "recorded"},
            }
        ],
        model="openai/gpt-5.6-luna",
    )
    request = GenerationRequest(
        phase="phase1_claim_extraction",
        prompt="local replay",
        schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    result = provider.generate(request)

    assert result.parsed == {"answer": "recorded"}
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0
    assert result.usage.cost_usd == 0.0
    assert result.raw_metadata == {
        "provider": "openrouter",
        "requested_model": "openai/gpt-5.6-luna",
        "returned_model": "openai/gpt-5.6-luna",
        "model": "openai/gpt-5.6-luna",
        "replay": "recorded-local",
    }
    assert provider.api_calls == 0
    assert provider.local_replay_calls == 1

    exhausted = module.RecordedReplayProvider([], model="model")
    with pytest.raises(RuntimeError, match="exhausted"):
        exhausted.generate(request)


def test_recorded_replay_provider_can_repeat_only_terminal_adjudication() -> None:
    module = _load_module()
    provider = module.RecordedReplayProvider(
        [
            {
                "phase": "phase1_claim_extraction",
                "parsed": {"answer": "claim"},
            },
            {
                "phase": "phase1_adjudication",
                "parsed": {"answer": "adjudication"},
            },
        ],
        model="openai/gpt-5.6-luna",
        terminal_adjudication_repeats=1,
    )
    claim_request = GenerationRequest(
        phase="phase1_claim_extraction",
        prompt="claim",
        schema={"type": "object"},
    )
    adjudication_request = GenerationRequest(
        phase="phase1_adjudication",
        prompt="adjudication",
        schema={"type": "object"},
    )

    assert provider.generate(claim_request).parsed == {"answer": "claim"}
    assert provider.generate(adjudication_request).parsed == {
        "answer": "adjudication"
    }
    assert provider.generate(adjudication_request).parsed == {
        "answer": "adjudication"
    }
    with pytest.raises(RuntimeError, match="exhausted"):
        provider.generate(adjudication_request)
    assert provider.local_replay_calls == 3
    assert provider.api_calls == 0
