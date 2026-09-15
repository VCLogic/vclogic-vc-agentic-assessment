from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import os
import subprocess
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import vc_clone_graph.workflow_v44 as workflow_v44_module
from vc_clone_graph.artifacts import verify_phase1_artifacts
from vc_clone_graph.config import Phase1V44Settings
from vc_clone_graph.graph import WorkflowSettings
from vc_clone_graph.providers.base import GenerationResult, Usage
from vc_clone_graph.providers.fake import FakeProvider
from vc_clone_graph.retrieval import HybridWikiIndex
from vc_clone_graph.workflow_v44 import VCDecisionWorkflowV44
from tests.test_phase1_v44_contract import retrieval_model, retrieval_payload


class LocalEmbedder:
    metadata = {
        "backend": "sentence_transformers",
        "model": "workflow-v44-embedding",
        "revision": "rev-1",
        "normalize": True,
    }

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [
                1.0,
                float("execution" in text.casefold()),
                float("risk" in text.casefold()),
            ]
            for text in texts
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)


class RemoteEmbedder(LocalEmbedder):
    metadata = {
        "backend": "not-local",
        "model": "remote-embedding",
        "revision": "latest",
        "normalize": True,
    }


class UnpinnedOllamaEmbedder(LocalEmbedder):
    metadata = {
        "backend": "ollama",
        "model": "nomic-embed-text",
        "revision": None,
        "normalize": False,
    }


class IdentifiedFakeProvider(FakeProvider):
    def __init__(self, outputs: list[dict]) -> None:
        super().__init__(outputs)
        self.model = "safe-model"
        self.base_url = "https://user:secret@example.test/v1?api_key=secret"
        self.api_key = "must-not-be-persisted"
        self.arbitrary_config = {"nested_secret": "must-not-be-persisted"}

    def phase1_fingerprint(self) -> dict:
        return {
            "schema_version": "phase1-provider-fingerprint-v1",
            "provider": "identified-fake",
            "model": self.model,
            "base_url": self.base_url,
        }

    def generate(self, request: object) -> GenerationResult:
        result = super().generate(request)  # type: ignore[arg-type]
        return result.model_copy(
            update={
                "raw_metadata": {
                    "provider": "identified-fake",
                    "model": self.model,
                }
            }
        )


class OpaqueCustomProvider(FakeProvider):
    def __init__(self, outputs: list[dict], behavior_mode: str) -> None:
        super().__init__(outputs)
        self.behavior_mode = behavior_mode


class ConfigurableCustomProvider(OpaqueCustomProvider):
    def phase1_fingerprint(self) -> dict:
        return {
            "schema_version": "phase1-provider-fingerprint-v1",
            "provider": "configurable-custom",
            "behavior_mode": self.behavior_mode,
        }


class ConfigurableCustomProviderTwin(OpaqueCustomProvider):
    def phase1_fingerprint(self) -> dict:
        return {
            "schema_version": "phase1-provider-fingerprint-v1",
            "provider": "configurable-custom",
            "behavior_mode": self.behavior_mode,
        }


class SpoofClassProvider(OpaqueCustomProvider):
    def phase1_fingerprint(self) -> dict:
        return {
            "schema_version": "phase1-provider-fingerprint-v1",
            "provider": "spoof-custom",
            "behavior_mode": self.behavior_mode,
            "class": "spoofed.Provider",
        }


class SecretFingerprintProvider(OpaqueCustomProvider):
    def phase1_fingerprint(self) -> dict:
        return {
            "schema_version": "phase1-provider-fingerprint-v1",
            "provider": "unsafe-custom",
            "transport": {"authorization": "must-never-persist"},
        }


class CostedWalProvider(ConfigurableCustomProvider):
    def generate(self, request: object) -> GenerationResult:
        result = super().generate(request)  # type: ignore[arg-type]
        return result.model_copy(
            update={
                "usage": Usage(
                    input_tokens=result.usage.input_tokens,
                    cached_input_tokens=result.usage.cached_input_tokens,
                    output_tokens=result.usage.output_tokens,
                    cost_usd=1.25,
                )
            }
        )


def make_index(
    tmp_path: Path, embedder: LocalEmbedder | None = None
) -> HybridWikiIndex:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "principles.md").write_text(
        "# Execution and risk\n"
        "The investor values founder execution and explicit customer-risk diligence.\n",
        encoding="utf-8",
    )
    return HybridWikiIndex.build(
        wiki,
        embedder or LocalEmbedder(),
        require_complete_embeddings=True,
    )


def settings(tmp_path: Path, *, pitch_lines: int = 1) -> WorkflowSettings:
    pitch = "Founder reports five paid pilots."
    if pitch_lines == 2:
        pitch += "\nFounder reports customer concentration risk."
    return WorkflowSettings(
        episode_slug="18-rowvigor",
        investor_name="Charles Hudson",
        pitch=pitch,
        taxonomy_labels={"founder_execution", "customer_risk"},
        taxonomy_records=(
            {
                "label": "founder_execution",
                "definition": "Evidence the founders reliably turn insight into execution.",
                "coarse_parent": "founding_team",
            },
            {
                "label": "customer_risk",
                "definition": "Evidence that customer concentration creates material risk.",
                "coarse_parent": "traction",
            },
        ),
        check_tiers={"exploratory_lt_100k"},
        phase1_min_iterations=1,
        phase1_max_iterations=2,
        phase2_min_iterations=1,
        phase2_max_iterations=1,
        retrieval_top_k=2,
        max_exact_reads=2,
        run_root=tmp_path / "run",
        contract_version="v4.4",
        execution_mode="phase1_only",
        phase1_max_output_tokens=2048,
        phase1_reasoning_effort="medium",
    )


def v44_settings(*, max_revisits: int = 1) -> Phase1V44Settings:
    return Phase1V44Settings(
        taxonomy_top_k=2,
        claim_retrieval_top_k=1,
        max_wiki_reads_per_claim=1,
        max_precedent_reads_per_claim=0,
        max_revisits=max_revisits,
    )


def claim_map(*, claims: int = 1) -> dict:
    material = [
        {
            "claim_id": "C1",
            "claim_type": "material",
            "statement": "The company has five paid pilots.",
            "topic": "founder execution",
            "decision_relevance": "Paid pilots test execution.",
            "pitch_evidence_ids": ["P-001"],
        }
    ]
    adverse = []
    coverage = [
        {
            "pitch_evidence_id": "P-001",
            "claim_id": "C1",
            "question_id": None,
            "non_material_justification": None,
        }
    ]
    if claims == 2:
        adverse.append(
            {
                "claim_id": "C2",
                "claim_type": "adverse",
                "statement": "Customer concentration is high.",
                "topic": "customer risk",
                "decision_relevance": "Concentration may weaken durability.",
                "pitch_evidence_ids": ["P-002"],
            }
        )
        coverage.append(
            {
                "pitch_evidence_id": "P-002",
                "claim_id": "C2",
                "question_id": None,
                "non_material_justification": None,
            }
        )
    return {
        "schema_version": "claim-map-v4.4",
        "episode_slug": "18-rowvigor",
        "material_claims": material,
        "adverse_claims": adverse,
        "unanswered_questions": [],
        "claim_coverage": coverage,
    }


def adjudication(
    evidence_id: str,
    *,
    claims: tuple[str, ...] = ("C1",),
    provisional: bool = False,
    request_id: str | None = None,
) -> dict:
    pitch_ids = [f"P-{int(claim_id[1:]):03d}" for claim_id in claims]
    return {
        "schema_version": "rationale-adjudication-v4.4",
        "episode_slug": "18-rowvigor",
        "dispositions": [
            {
                "taxonomy_label": "founder_execution",
                "disposition": "core",
                "claim_ids": list(claims),
                "question_ids": [],
                "pitch_evidence_ids": pitch_ids,
                "wiki_evidence_ids": [evidence_id],
                "historical_evidence_ids": [],
                "portfolio_disclosure_ids": [],
                "direction": "positive",
                "salience": "primary",
                "confidence": 0.8,
                "justification": "The retrieved principle supports the founder signal.",
            }
        ],
        "unmapped_observations": [],
        "constraint_assessments": [],
        "portfolio_overlap_assessments": [],
        "adjudication_status": "provisional" if provisional else "valid",
        "validator_findings": [],
        "requested_retrieval_ids": [request_id] if request_id else [],
    }


def test_workflow_normalizes_nonactivating_fields_without_paid_repair(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    output = adjudication(evidence_id)
    output["dispositions"].append(
        {
            "taxonomy_label": "customer_risk",
            "disposition": "rejected",
            "claim_ids": [],
            "question_ids": [],
            "pitch_evidence_ids": [],
            "wiki_evidence_ids": [],
            "historical_evidence_ids": [],
            "portfolio_disclosure_ids": [],
            "direction": "neutral",
            "salience": "secondary",
            "confidence": 0.99,
            "justification": "The available claim does not activate this rationale.",
        }
    )
    provider = FakeProvider([claim_map(), output])
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    result = workflow.invoke("normalize-nonactivating")

    assert result["phase1_status"] == "accepted"
    assert [request.phase for request in provider.requests] == [
        "phase1_claim_extraction",
        "phase1_adjudication",
    ]
    rejected = result["adjudication"]["dispositions"][1]
    assert rejected["direction"] is None
    assert rejected["salience"] is None
    assert rejected["confidence"] is None
    adjudication_event = next(
        row for row in result["events"] if row["kind"] == "phase1_adjudication"
    )
    assert adjudication_event["normalized_nonactivating_dispositions"] == 1
    call = json.loads(
        (tmp_path / "run/phase1/adjudication/turn-01/call-01.json").read_text(
            encoding="utf-8"
        )
    )
    assert call["parsed"]["dispositions"][1]["confidence"] == 0.99


def test_workflow_finalizes_distinct_instances_and_constraint_mappings(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    output = adjudication(evidence_id, claims=("C1",))
    distinct = {
        **output["dispositions"][0],
        "disposition": "candidate",
        "claim_ids": ["C2"],
        "pitch_evidence_ids": ["P-002"],
        "direction": "negative",
        "salience": "secondary",
        "confidence": 0.65,
        "justification": "Customer concentration creates a distinct execution risk.",
    }
    output["dispositions"] = [
        output["dispositions"][0],
        dict(output["dispositions"][0]),
        distinct,
    ]
    output["constraint_assessments"] = [
        {
            "constraint_id": "C1",
            "constraint_kind": "stage_or_check_fit",
            "policy_statement": "Concentration must remain within the fund's risk bar.",
            "status": "possible",
            "severity": "material",
            "pitch_evidence_ids": ["P-002"],
            "wiki_evidence_ids": [evidence_id],
            "historical_evidence_ids": [],
            "mapped_ids": ["R1"],
            "assessment": "The second claim raises the constraint.",
        }
    ]
    provider = FakeProvider([claim_map(claims=2), output])
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path, pitch_lines=2),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    result = workflow.invoke("instance-finalization")

    assert result["phase1_status"] == "accepted"
    assert len(result["adjudication"]["dispositions"]) == 2
    assert result["adjudication"]["constraint_assessments"][0][
        "mapped_ids"
    ] == ["R2"]
    event = next(
        row for row in result["events"] if row["kind"] == "phase1_adjudication"
    )
    assert event["normalized_nonactivating_dispositions"] == 0
    assert event["exact_duplicate_dispositions_removed"] == 1
    assert event["removed_disposition_positions"] == [1]
    assert event["cross_target_evidence_reuse"] == []
    assert event["constraint_mapping_revisions"] == [
        {
            "constraint_id": "C1",
            "original_mapped_ids": ["R1"],
            "recomputed_mapped_ids": ["R2"],
        }
    ]
    call = json.loads(
        (tmp_path / "run/phase1/adjudication/turn-01/call-01.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(call["parsed"]["dispositions"]) == 3
    assert call["parsed"]["constraint_assessments"][0]["mapped_ids"] == [
        "R1"
    ]
    assert len(provider.requests) == 2


def test_workflow_fails_closed_on_unresolved_constraint_mapping(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    output = adjudication(evidence_id, claims=("C1",))
    output["dispositions"].append(
        {
            "taxonomy_label": "customer_risk",
            "disposition": "rejected",
            "claim_ids": ["C2"],
            "question_ids": [],
            "pitch_evidence_ids": [],
            "wiki_evidence_ids": [],
            "historical_evidence_ids": [],
            "portfolio_disclosure_ids": [],
            "direction": None,
            "salience": None,
            "confidence": None,
            "justification": "The second claim does not activate this label.",
        }
    )
    output["constraint_assessments"] = [
        {
            "constraint_id": "C1",
            "constraint_kind": "stage_or_check_fit",
            "policy_statement": "Concentration must remain within the fund's risk bar.",
            "status": "possible",
            "severity": "material",
            "pitch_evidence_ids": ["P-002"],
            "wiki_evidence_ids": [evidence_id],
            "historical_evidence_ids": [],
            "mapped_ids": ["R1"],
            "assessment": "No activated instance supports this constraint.",
        }
    ]
    provider = FakeProvider([claim_map(claims=2), output])
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path, pitch_lines=2),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    result = workflow.invoke("unresolved-constraint")

    assert result["phase1_status"] == "failed"
    assert "UNRESOLVED_CONSTRAINT_MAPPING:C1" in result["phase1_findings"]


def test_workflow_revisits_then_preserves_question_linked_constraint(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    extracted = claim_map(claims=2)
    extracted["unanswered_questions"] = [
        {
            "question_id": "Q1",
            "question": "Is customer concentration already declining?",
            "why_material": "The pitch does not provide a current concentration trend.",
            "anchor_pitch_evidence_ids": ["P-002"],
        }
    ]
    output = adjudication(evidence_id)
    output["dispositions"].append(
        {
            "taxonomy_label": "customer_risk",
            "disposition": "question_only",
            "claim_ids": ["C2"],
            "question_ids": ["Q1"],
            "pitch_evidence_ids": ["P-002"],
            "wiki_evidence_ids": [evidence_id],
            "historical_evidence_ids": [],
            "portfolio_disclosure_ids": [],
            "direction": None,
            "salience": None,
            "confidence": None,
            "justification": "The pitch raises but does not resolve concentration risk.",
        }
    )
    output["constraint_assessments"] = [
        {
            "constraint_id": "C1",
            "constraint_kind": "other",
            "policy_statement": "Material concentration requires explicit diligence.",
            "status": "possible",
            "severity": "material",
            "pitch_evidence_ids": ["P-002"],
            "wiki_evidence_ids": [evidence_id],
            "historical_evidence_ids": [],
            "mapped_ids": ["R1"],
            "assessment": "The pitch does not establish whether concentration is improving.",
        }
    ]
    output["adjudication_status"] = "valid"
    output["requested_retrieval_ids"] = ["Q1"]
    provider = IdentifiedFakeProvider([extracted, output, deepcopy(output)])
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path, pitch_lines=2),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(max_revisits=1),
    )

    result = workflow.invoke("question-linked-constraint")

    assert result["phase1_status"] == "provisional"
    assert result["phase1_findings"] == []
    assert result["phase1_iteration"] == 2
    assert result["adjudication"]["constraint_assessments"][0]["mapped_ids"] == [
        "Q1"
    ]
    assert [request.phase for request in provider.requests] == [
        "phase1_claim_extraction",
        "phase1_adjudication",
        "phase1_adjudication",
    ]
    investigation = json.loads(
        (tmp_path / "run/phase1/investigation.json").read_text(encoding="utf-8")
    )
    assert investigation["investigation_status"] == "provisional"
    assert investigation["constraint_assessments"][0]["mapped_ids"] == ["Q1"]
    verify_phase1_artifacts(tmp_path / "run", provenance_mode="direct")


def make_workflow(
    tmp_path: Path,
    outputs: list[dict],
    *,
    pitch_lines: int = 1,
    max_revisits: int = 1,
    checkpointer: InMemorySaver | None = None,
) -> tuple[VCDecisionWorkflowV44, FakeProvider]:
    index = make_index(tmp_path)
    provider = FakeProvider(outputs)
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path, pitch_lines=pitch_lines),
        index,
        phase1_provider=provider,
        checkpointer=checkpointer or InMemorySaver(),
        v44_settings=v44_settings(max_revisits=max_revisits),
    )
    return workflow, provider


def assert_call_artifact(
    path: Path,
    request: object,
    parsed: dict,
) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["phase"] == request.phase
    assert record["prompt"] == request.prompt
    assert record["schema"] == request.schema
    assert record["raw"] == json.dumps(parsed, sort_keys=True)
    assert record["parsed"] == parsed
    assert record["provider_metadata"] == {"provider": "fake"}
    assert record["usage"] == {
        "input_tokens": max(1, len(request.prompt) // 4),
        "cached_input_tokens": 0,
        "output_tokens": max(1, len(record["raw"]) // 4),
        "cost_usd": 0.0,
    }
    assert record["cost_usd"] == record["usage"]["cost_usd"]
    assert record["elapsed_seconds"] >= 0.0
    assert record["max_output_tokens"] == request.max_output_tokens
    assert record["reasoning_effort"] == request.reasoning_effort


@pytest.mark.parametrize("symlink_case", ["run-root", "phase1", "lock", "owner"])
def test_run_root_preflight_rejects_static_symlinks_before_any_mutation(
    tmp_path: Path,
    symlink_case: str,
) -> None:
    index = make_index(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("do-not-touch", encoding="utf-8")
    run_root = tmp_path / "run"
    if symlink_case == "run-root":
        run_root.symlink_to(external, target_is_directory=True)
    else:
        run_root.mkdir()
        target = run_root / {
            "phase1": "phase1",
            "lock": ".phase1-v44.lock",
            "owner": "run-owner-v44.json",
        }[symlink_case]
        target.symlink_to(
            external if symlink_case == "phase1" else sentinel,
            target_is_directory=symlink_case == "phase1",
        )
    provider = FakeProvider([claim_map()])
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    with pytest.raises(ValueError, match="symlink"):
        workflow.invoke("symlink-probe")

    assert provider.requests == []
    assert sentinel.read_text(encoding="utf-8") == "do-not-touch"
    owner_path = run_root / "run-owner-v44.json"
    if symlink_case == "owner":
        assert owner_path.is_symlink()
    else:
        assert not owner_path.exists()


def test_held_run_root_fd_defeats_swap_before_atomic_temp_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    provider = FakeProvider([claim_map(), adjudication(evidence_id)])
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )
    run_root = tmp_path / "run"
    held_root = tmp_path / "held-run-root"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("do-not-touch", encoding="utf-8")
    swapped = False

    def swap_checked_path() -> None:
        nonlocal swapped
        if swapped:
            return
        swapped = True
        run_root.rename(held_root)
        run_root.symlink_to(external, target_is_directory=True)

    monkeypatch.setattr(
        workflow_v44_module,
        "_before_atomic_temp_create",
        swap_checked_path,
    )

    result = workflow.invoke("dirfd-race")

    assert result["phase1_status"] == "accepted"
    assert len(provider.requests) == 2
    assert sentinel.read_text(encoding="utf-8") == "do-not-touch"
    assert sorted(path.name for path in external.iterdir()) == ["sentinel.txt"]
    assert (held_root / "run-owner-v44.json").exists()
    assert (held_root / "phase1/claim-extraction/call-01.json").exists()


def test_run_root_inode_lock_ignores_replaced_legacy_lock_filename(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    first = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=FakeProvider([]),
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )
    second = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=FakeProvider([]),
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )
    run_root = tmp_path / "run"
    external = tmp_path / "external-lock-target"
    external.write_text("do-not-lock", encoding="utf-8")

    with workflow_v44_module._ArtifactStore.open(run_root) as first_store:
        with workflow_v44_module._ArtifactStore.open(run_root) as second_store:
            first._store = first_store
            second._store = second_store
            assert os.fstat(first_store.root_fd).st_ino == os.fstat(
                second_store.root_fd
            ).st_ino
            with first._run_lock():
                legacy = run_root / ".phase1-v44.lock"
                legacy.write_text("replace-me", encoding="utf-8")
                legacy.unlink()
                legacy.symlink_to(external)
                probe = os.dup(second_store.root_fd)
                try:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    os.close(probe)
                assert not (run_root / "run-owner-v44.json").exists()
                legacy.unlink()
            with second._run_lock():
                assert not (run_root / "run-owner-v44.json").exists()


def test_failed_atomic_write_fsyncs_parent_after_temp_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    events: list[tuple[str, int]] = []
    real_fsync = os.fsync
    real_unlink = os.unlink

    def record_fsync(descriptor: int) -> None:
        events.append(("fsync", os.fstat(descriptor).st_ino))
        real_fsync(descriptor)

    def record_unlink(
        path: str,
        *,
        dir_fd: int | None = None,
    ) -> None:
        assert dir_fd is not None
        events.append(("unlink", os.fstat(dir_fd).st_ino))
        real_unlink(path, dir_fd=dir_fd)

    def fail_replace(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected replace failure")

    with workflow_v44_module._ArtifactStore.open(run_root) as store:
        root_inode = os.fstat(store.root_fd).st_ino
        monkeypatch.setattr(workflow_v44_module.os, "fsync", record_fsync)
        monkeypatch.setattr(workflow_v44_module.os, "unlink", record_unlink)
        monkeypatch.setattr(workflow_v44_module.os, "replace", fail_replace)

        with pytest.raises(OSError, match="injected replace failure"):
            store.atomic_write("state.json", b"payload")

        cleanup = events.index(("unlink", root_inode))
        assert ("fsync", root_inode) in events[cleanup + 1 :]
        assert store.list_files() == []


def test_happy_path_routes_through_phase1_only_and_accounts_calls(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    provider = FakeProvider([claim_map(), adjudication(evidence_id)])
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    result = workflow.invoke("happy")

    assert [event["kind"] for event in result["events"]] == [
        "phase1_claim_extraction",
        "phase1_claim_retrieval",
        "phase1_taxonomy_retrieval",
        "phase1_adjudication",
        "phase1_freeze",
    ]
    assert result["phase1_status"] == "accepted"
    assert result["phase2_status"] == "not_run"
    assert result["phase1_iteration"] == 1
    assert len(provider.requests) == 2
    assert [request.phase for request in provider.requests] == [
        "phase1_claim_extraction",
        "phase1_adjudication",
    ]
    assert all(request.max_output_tokens == 2048 for request in provider.requests)
    assert all(request.reasoning_effort == "medium" for request in provider.requests)
    assert result["usage"]["input_tokens"] > 0
    assert result["usage_by_phase"]["phase2"]["input_tokens"] == 0
    assert [row["relative_path"] for row in result["call_records"]] == [
        "phase1/claim-extraction/call-01.json",
        "phase1/adjudication/turn-01/call-01.json",
    ]
    assert all(
        row["sha256"]
        == sha256(
            (tmp_path / "run" / row["relative_path"]).read_bytes()
        ).hexdigest()
        for row in result["call_records"]
    )
    assert result["run_fingerprint"]
    assert result["run_thread_id"] == "happy"
    owner = json.loads(
        (tmp_path / "run/run-owner-v44.json").read_text(encoding="utf-8")
    )
    assert owner["thread_id"] == "happy"
    assert owner["run_fingerprint"] == result["run_fingerprint"]
    assert owner["fingerprint_payload"]["provider"]["class"].endswith(
        ".FakeProvider"
    )
    wal_root = tmp_path / "run/.phase1-v44-wal"
    assert not wal_root.exists() or list(wal_root.iterdir()) == []
    assert (tmp_path / "run/phase1/claim-map.json").exists()
    assert (tmp_path / "run/phase1/investigation.json").exists()
    registry_artifact = json.loads(
        (tmp_path / "run/phase1/evidence-registry.json").read_text(encoding="utf-8")
    )
    assert registry_artifact == list(result["evidence_registry"].values())
    for artifact, state_field in (
        ("claim-map", "claim_map_sha256"),
        ("claim-retrieval", "claim_retrieval_sha256"),
        ("taxonomy-neighborhood", "taxonomy_neighborhood_sha256"),
        ("adjudication", "adjudication_sha256"),
        ("investigation", "investigation_sha256"),
    ):
        raw = (tmp_path / f"run/phase1/{artifact}.json").read_bytes()
        digest = sha256(raw).hexdigest()
        assert (tmp_path / f"run/phase1/{artifact}.sha256").read_text().strip() == digest
        assert result[state_field] == digest
    assert result["investigation"]["claim_map_sha256"] == result["claim_map_sha256"]
    assert result["investigation"]["claim_retrieval_sha256"] == result[
        "claim_retrieval_sha256"
    ]
    assert result["investigation"]["taxonomy_neighborhood_sha256"] == result[
        "taxonomy_neighborhood_sha256"
    ]
    assert result["investigation"]["adjudication_sha256"] == result[
        "adjudication_sha256"
    ]
    assert json.loads(
        (tmp_path / "run/state.json").read_text(encoding="utf-8")
    ) == result
    assert_call_artifact(
        tmp_path / "run/phase1/adjudication/turn-01/call-01.json",
        provider.requests[1],
        adjudication(evidence_id),
    )


def test_provider_fingerprint_whitelists_safe_identity_without_secrets(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    provider = IdentifiedFakeProvider([claim_map(), adjudication(evidence_id)])
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    workflow.invoke("safe-provider-identity")

    owner_text = (tmp_path / "run/run-owner-v44.json").read_text(encoding="utf-8")
    owner = json.loads(owner_text)
    identity = owner["fingerprint_payload"]["provider"]
    assert identity["model"] == "safe-model"
    assert identity["base_url"] == "https://example.test/v1"
    assert "must-not-be-persisted" not in owner_text
    assert "api_key" not in owner_text
    assert "arbitrary_config" not in owner_text


def test_custom_provider_requires_explicit_versioned_safe_fingerprint(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    common = {
        "checkpointer": InMemorySaver(),
        "v44_settings": v44_settings(),
    }

    with pytest.raises(ValueError, match="explicit phase1_fingerprint"):
        VCDecisionWorkflowV44(
            settings(tmp_path),
            index,
            phase1_provider=OpaqueCustomProvider([], "hidden-behavior"),
            **common,
        )
    with pytest.raises(ValueError, match="secret-like provider fingerprint key"):
        VCDecisionWorkflowV44(
            settings(tmp_path),
            index,
            phase1_provider=SecretFingerprintProvider([], "unsafe"),
            **common,
        )


def test_same_custom_provider_class_behavior_changes_run_fingerprint(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    first = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=ConfigurableCustomProvider([], "strict"),
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )
    second = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=ConfigurableCustomProvider([], "permissive"),
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    assert first.run_fingerprint != second.run_fingerprint
    assert first.run_fingerprint_payload["provider"]["behavior_mode"] == "strict"
    assert second.run_fingerprint_payload["provider"]["behavior_mode"] == "permissive"


def test_custom_provider_class_is_reserved_and_binds_actual_type(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    common = {
        "checkpointer": InMemorySaver(),
        "v44_settings": v44_settings(),
    }
    with pytest.raises(ValueError, match="reserved.*class"):
        VCDecisionWorkflowV44(
            settings(tmp_path),
            index,
            phase1_provider=SpoofClassProvider([], "same"),
            **common,
        )
    first = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=ConfigurableCustomProvider([], "same"),
        **common,
    )
    second = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=ConfigurableCustomProviderTwin([], "same"),
        **common,
    )

    assert first.run_fingerprint != second.run_fingerprint
    first_class = first.run_fingerprint_payload["provider"]["class"]
    second_class = second.run_fingerprint_payload["provider"]["class"]
    assert first_class != second_class


def test_one_targeted_revisit_merges_full_manifests(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    provider = FakeProvider(
        [
            claim_map(claims=2),
            adjudication(
                evidence_id,
                claims=("C1",),
                provisional=True,
                request_id="C2",
            ),
            adjudication(evidence_id, claims=("C1", "C2")),
        ]
    )
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path, pitch_lines=2),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    result = workflow.invoke("revisit")

    assert result["phase1_iteration"] == 2
    assert result["targeted_retrieval_ids"] == ["C2"]
    assert result["phase1_status"] == "accepted"
    assert len(provider.requests) == 3
    assert [row["target_id"] for row in result["claim_retrieval"]["claim_bundles"]] == [
        "C1",
        "C2",
    ]
    assert [
        row["target_id"]
        for row in result["taxonomy_neighborhood"]["claim_neighborhoods"]
    ] == ["C1", "C2"]
    assert [event["kind"] for event in result["events"]].count(
        "phase1_adjudication"
    ) == 2
    assert_call_artifact(
        tmp_path / "run/phase1/adjudication/turn-01/call-01.json",
        provider.requests[1],
        adjudication(
            evidence_id,
            claims=("C1",),
            provisional=True,
            request_id="C2",
        ),
    )
    assert_call_artifact(
        tmp_path / "run/phase1/adjudication/turn-02/call-01.json",
        provider.requests[2],
        adjudication(evidence_id, claims=("C1", "C2")),
    )


@pytest.mark.parametrize("exact_arrives_first", [True, False])
def test_retrieval_revisit_preserves_exact_registry_text_over_search_excerpt(
    exact_arrives_first: bool,
) -> None:
    exact_payload = retrieval_payload()
    excerpt_payload = deepcopy(exact_payload)
    full_text = exact_payload["evidence_registry"][0]["text"]
    excerpt = full_text[:20]
    excerpt_payload["evidence_registry"][0]["text"] = excerpt
    excerpt_payload["claim_bundles"][0]["wiki_evidence"][0]["text"] = excerpt
    excerpt_payload["claim_bundles"][0]["wiki_evidence"][0]["eligible"] = False
    previous, additional = (
        (retrieval_model(exact_payload), retrieval_model(excerpt_payload))
        if exact_arrives_first
        else (retrieval_model(excerpt_payload), retrieval_model(exact_payload))
    )

    merged = workflow_v44_module._merge_retrieval_manifests(previous, additional)

    registry = {row.evidence_id: row for row in merged.evidence_registry}
    assert registry["W-execution"].text == full_text
    occurrence = merged.claim_bundles[0].wiki_evidence[0]
    assert occurrence.text == full_text
    assert occurrence.eligible is True


def test_retrieval_revisit_rejects_conflicting_source_text() -> None:
    previous_payload = retrieval_payload()
    conflicting_payload = deepcopy(previous_payload)
    conflicting_payload["evidence_registry"][0]["text"] = "Unrelated source text."
    conflicting_payload["claim_bundles"][0]["wiki_evidence"][0]["text"] = (
        "Unrelated source text."
    )

    with pytest.raises(ValueError, match="evidence identity changed on revisit"):
        workflow_v44_module._merge_retrieval_manifests(
            retrieval_model(previous_payload),
            retrieval_model(conflicting_payload),
        )


def test_deterministic_unhandled_claim_finding_triggers_targeted_revisit(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    incomplete = adjudication(
        evidence_id,
        claims=("C1",),
        provisional=True,
    )
    provider = FakeProvider(
        [
            claim_map(claims=2),
            incomplete,
            adjudication(evidence_id, claims=("C1", "C2")),
        ]
    )
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path, pitch_lines=2),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    result = workflow.invoke("finding-only-revisit")

    assert result["phase1_iteration"] == 2
    assert result["targeted_retrieval_ids"] == ["C2"]
    assert result["phase1_status"] == "accepted"
    assert len(provider.requests) == 3


def test_first_structural_binding_blocker_fails_without_targeted_revisit(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    structurally_invalid = adjudication(evidence_id, claims=("C1",))
    structurally_invalid["dispositions"][0]["pitch_evidence_ids"] = ["P-002"]
    structurally_invalid["dispositions"].append(
        {
            "taxonomy_label": "customer_risk",
            "disposition": "core",
            "claim_ids": ["C2"],
            "question_ids": [],
            "pitch_evidence_ids": ["P-002"],
            "wiki_evidence_ids": [evidence_id],
            "historical_evidence_ids": [],
            "portfolio_disclosure_ids": [],
            "direction": "negative",
            "salience": "primary",
            "confidence": 0.7,
            "justification": "The retrieved principle supports the risk signal.",
        }
    )
    provider = FakeProvider(
        [
            claim_map(claims=2),
            structurally_invalid,
            adjudication(evidence_id, claims=("C1", "C2")),
        ]
    )
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path, pitch_lines=2),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    result = workflow.invoke("first-structural-blocker")

    assert result["phase1_status"] == "failed"
    assert result["phase2_status"] == "not_run"
    assert result["phase1_iteration"] == 1
    assert result["phase1_action"] == "failed"
    assert "PITCH_EVIDENCE_NOT_BOUND_TO_TARGET:P-002" in result[
        "phase1_findings"
    ]
    assert len(provider.requests) == 2
    assert [event["kind"] for event in result["events"]].count(
        "phase1_claim_retrieval"
    ) == 1


def test_exhausted_revisit_budget_freezes_provisional(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    provider = FakeProvider(
        [
            claim_map(claims=2),
            adjudication(
                evidence_id,
                claims=("C1",),
                provisional=True,
                request_id="C2",
            ),
        ]
    )
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path, pitch_lines=2),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(max_revisits=0),
    )

    result = workflow.invoke("provisional")

    assert result["phase1_status"] == "provisional"
    assert result["investigation"]["investigation_status"] == "provisional"
    assert result["phase2_status"] == "not_run"
    assert result["phase1_iteration"] == 1
    assert len(provider.requests) == 2


def test_claim_extraction_allows_one_mechanical_repair(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    provider = FakeProvider(
        [{"invalid": "shape"}, claim_map(), adjudication(evidence_id)]
    )
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    result = workflow.invoke("repair")

    assert result["phase1_status"] == "accepted"
    assert len(provider.requests) == 3
    assert [request.phase for request in provider.requests] == [
        "phase1_claim_extraction",
        "phase1_claim_extraction_repair",
        "phase1_adjudication",
    ]
    records = sorted((tmp_path / "run/phase1/claim-extraction").glob("call-*.json"))
    assert len(records) == 2
    recorded = json.loads(records[0].read_text(encoding="utf-8"))
    assert set(recorded) >= {
        "prompt",
        "schema",
        "raw",
        "parsed",
        "provider_metadata",
        "usage",
        "cost_usd",
        "elapsed_seconds",
    }
    assert recorded["raw"] == json.dumps({"invalid": "shape"}, sort_keys=True)
    assert recorded["parsed"] == {"invalid": "shape"}
    assert recorded["provider_metadata"] == {"provider": "fake"}
    assert recorded["cost_usd"] == recorded["usage"]["cost_usd"] == 0.0
    all_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "run/phase1").glob("**/call-*.json")
    ]
    assert result["usage"]["input_tokens"] == sum(
        row["usage"]["input_tokens"] for row in all_records
    )
    assert result["usage"]["output_tokens"] == sum(
        row["usage"]["output_tokens"] for row in all_records
    )
    assert list((tmp_path / "run/.phase1-v44-wal").iterdir()) == []


def test_every_provider_call_kind_checkpoints_intent_before_exact_response(
    tmp_path: Path,
) -> None:
    checkpointer = InMemorySaver()
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    provider = FakeProvider(
        [
            {"invalid": "claim"},
            claim_map(),
            {"invalid": "adjudication"},
            adjudication(evidence_id),
        ]
    )
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=checkpointer,
        v44_settings=v44_settings(),
    )

    result = workflow.invoke("all-intents")

    assert result["phase1_status"] == "accepted"
    checkpoints = [
        item.checkpoint["channel_values"]
        for item in checkpointer.list(
            {"configurable": {"thread_id": "all-intents"}}
        )
    ]
    for kind in (
        "claim_primary",
        "claim_repair",
        "adjudication_primary",
        "adjudication_repair",
    ):
        states = [
            state
            for state in checkpoints
            if (state.get("pending_call") or {}).get("kind") == kind
        ]
        intent = next(
            state for state in states if state.get("pending_response") is None
        )
        responded = next(
            state for state in states if state.get("pending_response") is not None
        )
        pending = intent["pending_call"]
        response = responded["pending_response"]
        assert response["intent_sha256"] == pending["intent_sha256"]
        assert response["request_sha256"] == pending["request_sha256"]
        assert any(
            record["wal_id"] == response["wal_id"]
            and record["relative_path"] == pending["relative_call_path"]
            for record in responded["call_records"]
        )


def test_second_invalid_claim_extraction_fails_before_adjudication(tmp_path: Path) -> None:
    workflow, provider = make_workflow(
        tmp_path,
        [{"invalid": "first"}, {"invalid": "second"}],
    )

    result = workflow.invoke("invalid-claim")

    assert result["phase1_status"] == "failed"
    assert result["phase2_status"] == "not_run"
    assert len(provider.requests) == 2
    assert not any(request.phase == "phase1_adjudication" for request in provider.requests)


def test_claim_runtime_binding_violation_uses_one_repair(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    wrong_episode = claim_map()
    wrong_episode["episode_slug"] = "17-other"
    provider = FakeProvider(
        [wrong_episode, claim_map(), adjudication(evidence_id)]
    )
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    result = workflow.invoke("claim-binding-repair")

    assert result["phase1_status"] == "accepted"
    assert [row.phase for row in provider.requests] == [
        "phase1_claim_extraction",
        "phase1_claim_extraction_repair",
        "phase1_adjudication",
    ]


def test_claim_runtime_binding_violation_after_repair_fails_cleanly(
    tmp_path: Path,
) -> None:
    wrong_episode = claim_map()
    wrong_episode["episode_slug"] = "17-other"
    workflow, provider = make_workflow(
        tmp_path,
        [wrong_episode, wrong_episode],
    )

    result = workflow.invoke("claim-binding-failed")

    assert result["phase1_status"] == "failed"
    assert result["phase2_status"] == "not_run"
    assert len(provider.requests) == 2
    assert not any(row.phase == "phase1_adjudication" for row in provider.requests)


def test_completed_checkpoint_resume_does_not_repeat_paid_calls(tmp_path: Path) -> None:
    checkpointer = InMemorySaver()
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    provider = FakeProvider([claim_map(), adjudication(evidence_id)])
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=checkpointer,
        v44_settings=v44_settings(),
    )
    first = workflow.invoke("resume")

    resumed = workflow.resume("resume")

    assert len(provider.requests) == 2
    assert resumed == first
    assert resumed["phase2_status"] == "not_run"


def test_resume_rejects_changed_pitch_or_phase1_config_before_artifact_repair(
    tmp_path: Path,
) -> None:
    for case in ("pitch", "config"):
        case_root = tmp_path / case
        case_root.mkdir()
        checkpointer = InMemorySaver()
        index = make_index(case_root)
        evidence_id = index.chunks[0].chunk_id
        original_provider = FakeProvider([claim_map(), adjudication(evidence_id)])
        original = VCDecisionWorkflowV44(
            settings(case_root),
            index,
            phase1_provider=original_provider,
            checkpointer=checkpointer,
            v44_settings=v44_settings(),
        )
        original.invoke("same-thread")
        claim_path = case_root / "run/phase1/claim-map.json"
        claim_path.write_bytes(b"corrupt-before-fingerprint-check")
        changed_provider = FakeProvider([])
        changed = VCDecisionWorkflowV44(
            (
                replace(
                    settings(case_root),
                    pitch="Founder reports six paid pilots.",
                )
                if case == "pitch"
                else settings(case_root)
            ),
            index,
            phase1_provider=changed_provider,
            checkpointer=checkpointer,
            v44_settings=(
                v44_settings()
                if case == "pitch"
                else v44_settings(max_revisits=0)
            ),
        )

        with pytest.raises(ValueError, match="run fingerprint mismatch"):
            changed.resume("same-thread")

        assert changed_provider.requests == []
        assert claim_path.read_bytes() == b"corrupt-before-fingerprint-check"


def test_resume_rebuilds_corrupted_frozen_artifact_without_paid_calls(
    tmp_path: Path,
) -> None:
    checkpointer = InMemorySaver()
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    provider = FakeProvider([claim_map(), adjudication(evidence_id)])
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=checkpointer,
        v44_settings=v44_settings(),
    )
    expected = workflow.invoke("corrupt-resume")
    claim_path = tmp_path / "run/phase1/claim-map.json"
    expected_bytes = claim_path.read_bytes()
    claim_path.write_text('{"stale":true}', encoding="utf-8")

    resumed = workflow.resume("corrupt-resume")

    assert len(provider.requests) == 2
    assert resumed == expected
    assert claim_path.read_bytes() == expected_bytes
    assert sha256(expected_bytes).hexdigest() == (
        tmp_path / "run/phase1/claim-map.sha256"
    ).read_text(encoding="ascii").strip()
    assert json.loads(
        (tmp_path / "run/state.json").read_text(encoding="utf-8")
    ) == resumed


def test_resume_atomically_rebuilds_invalid_utf_artifacts_and_exact_call_records(
    tmp_path: Path,
) -> None:
    checkpointer = InMemorySaver()
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    provider = FakeProvider([claim_map(), adjudication(evidence_id)])
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=checkpointer,
        v44_settings=v44_settings(),
    )
    expected = workflow.invoke("utf-recovery")
    call_path = tmp_path / "run/phase1/claim-extraction/call-01.json"
    expected_call = call_path.read_bytes()
    claim_json_path = tmp_path / "run/phase1/claim-map.json"
    expected_claim_json = claim_json_path.read_bytes()
    call_path.write_bytes(b"\xff\xfe")
    claim_json_path.write_bytes(b"\xff")
    (tmp_path / "run/phase1/evidence-registry.json").write_bytes(b"\xff")
    (tmp_path / "run/phase1/claim-map.sha256").write_bytes(b"\xff")
    (tmp_path / "run/run-owner-v44.json").write_bytes(b"\xff")

    resumed = workflow.resume("utf-recovery")

    assert resumed == expected
    assert len(provider.requests) == 2
    assert call_path.read_bytes() == expected_call
    assert claim_json_path.read_bytes() == expected_claim_json
    json.loads(
        (tmp_path / "run/phase1/evidence-registry.json").read_text(
            encoding="utf-8"
        )
    )
    assert (tmp_path / "run/phase1/claim-map.sha256").read_text(
        encoding="ascii"
    ).strip() == expected["claim_map_sha256"]
    owner = json.loads(
        (tmp_path / "run/run-owner-v44.json").read_text(encoding="utf-8")
    )
    assert owner["run_fingerprint"] == expected["run_fingerprint"]

    call_path.unlink()
    assert workflow.resume("utf-recovery") == expected
    assert call_path.read_bytes() == expected_call
    assert len(provider.requests) == 2


def test_resume_rejects_call_set_or_usage_mismatch_without_provider_calls(
    tmp_path: Path,
) -> None:
    for case in ("call-set", "wal-set", "usage"):
        case_root = tmp_path / case
        case_root.mkdir()
        checkpointer = InMemorySaver()
        index = make_index(case_root)
        evidence_id = index.chunks[0].chunk_id
        provider = FakeProvider([claim_map(), adjudication(evidence_id)])
        workflow = VCDecisionWorkflowV44(
            settings(case_root),
            index,
            phase1_provider=provider,
            checkpointer=checkpointer,
            v44_settings=v44_settings(),
        )
        workflow.invoke("integrity")
        if case == "call-set":
            (case_root / "run/phase1/claim-extraction/call-99.json").write_text(
                "{}", encoding="utf-8"
            )
            message = "call artifact set mismatch"
        elif case == "wal-set":
            (case_root / "run/.phase1-v44-wal/unexpected.json").write_text(
                "{}", encoding="utf-8"
            )
            message = "provider-call WAL set is invalid"
        else:
            workflow.graph.update_state(
                {"configurable": {"thread_id": "integrity"}},
                {
                    "usage": {
                        "input_tokens": 999999,
                        "cached_input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                    }
                },
            )
            message = "call usage mismatch"

        with pytest.raises(ValueError, match=message):
            workflow.resume("integrity")

        assert len(provider.requests) == 2


def test_post_response_record_failure_leaves_uncertain_wal_and_never_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpointer = InMemorySaver()
    index = make_index(tmp_path)
    provider = CostedWalProvider([claim_map(), claim_map()], "costed")
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=checkpointer,
        v44_settings=v44_settings(),
    )

    def fail_after_response(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated call-record crash")

    monkeypatch.setattr(workflow, "_record_call", fail_after_response)

    with pytest.raises(RuntimeError, match="simulated call-record crash"):
        workflow.invoke("wal-crash")

    assert len(provider.requests) == 1
    wal_files = list((tmp_path / "run/.phase1-v44-wal").glob("*.json"))
    assert len(wal_files) == 1
    wal = json.loads(wal_files[0].read_text(encoding="utf-8"))
    assert wal["status"] == "responded_uncheckpointed"
    assert wal["uncertain_usage"]["cost_usd"] == 1.25
    assert wal["request_sha256"]
    assert wal["relative_call_path"] == "phase1/claim-extraction/call-01.json"
    wal_files[0].unlink()

    with pytest.raises(ValueError, match="pending provider intent.*uncertain"):
        workflow.resume("wal-crash")

    assert len(provider.requests) == 1
    assert not wal_files[0].exists()


def test_resume_uses_exact_checkpointed_response_without_wal_or_repeat_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpointer = InMemorySaver()
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    provider = FakeProvider([claim_map(), adjudication(evidence_id)])
    original = VCDecisionWorkflowV44._claim_finalize
    crashed = False

    def crash_once(
        self: VCDecisionWorkflowV44,
        state: workflow_v44_module.WorkflowStateV44,
    ) -> workflow_v44_module.WorkflowStateV44:
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("crash after response checkpoint")
        return original(self, state)

    monkeypatch.setattr(VCDecisionWorkflowV44, "_claim_finalize", crash_once)
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=checkpointer,
        v44_settings=v44_settings(),
    )

    with pytest.raises(RuntimeError, match="after response checkpoint"):
        workflow.invoke("exact-response")

    checkpoint = workflow._checkpoint_state("exact-response")
    assert checkpoint["pending_response"]["result"]["parsed"] == claim_map()
    wal_files = list((tmp_path / "run/.phase1-v44-wal").glob("*.json"))
    assert len(wal_files) == 1
    wal_files[0].unlink()

    result = workflow.resume("exact-response")

    assert result["phase1_status"] == "accepted"
    assert len(provider.requests) == 2


def test_resume_validates_checkpoint_thread_before_rebuilding_owner(
    tmp_path: Path,
) -> None:
    checkpointer = InMemorySaver()
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    provider = FakeProvider([claim_map(), adjudication(evidence_id)])
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=checkpointer,
        v44_settings=v44_settings(),
    )
    workflow.invoke("owner-order")
    owner_path = tmp_path / "run/run-owner-v44.json"
    owner_path.unlink()
    workflow.graph.update_state(
        {"configurable": {"thread_id": "owner-order"}},
        {"run_thread_id": "other-thread"},
    )

    with pytest.raises(ValueError, match="checkpoint thread ownership mismatch"):
        workflow.resume("owner-order")

    assert not owner_path.exists()
    assert len(provider.requests) == 2


def test_adjudication_allows_one_mechanical_repair(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    provider = FakeProvider(
        [claim_map(), {"invalid": "adjudication"}, adjudication(evidence_id)]
    )
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    result = workflow.invoke("adjudication-repair")

    assert result["phase1_status"] == "accepted"
    assert [request.phase for request in provider.requests] == [
        "phase1_claim_extraction",
        "phase1_adjudication",
        "phase1_adjudication_repair",
    ]
    assert len(
        list((tmp_path / "run/phase1/adjudication/turn-01").glob("call-*.json"))
    ) == 2
    assert list((tmp_path / "run/.phase1-v44-wal").iterdir()) == []


def test_invalid_adjudication_after_repair_fails_without_valid_prior(
    tmp_path: Path,
) -> None:
    workflow, provider = make_workflow(
        tmp_path,
        [claim_map(), {"invalid": "first"}, {"invalid": "second"}],
    )

    result = workflow.invoke("invalid-adjudication")

    assert result["phase1_status"] == "failed"
    assert result["phase2_status"] == "not_run"
    assert len(provider.requests) == 3
    assert "investigation" not in result


def test_adjudication_runtime_binding_violation_uses_one_repair(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    wrong_episode = adjudication(evidence_id)
    wrong_episode["episode_slug"] = "17-other"
    provider = FakeProvider(
        [claim_map(), wrong_episode, adjudication(evidence_id)]
    )
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    result = workflow.invoke("adjudication-binding-repair")

    assert result["phase1_status"] == "accepted"
    assert [row.phase for row in provider.requests] == [
        "phase1_claim_extraction",
        "phase1_adjudication",
        "phase1_adjudication_repair",
    ]


def test_adjudication_runtime_binding_violation_after_repair_fails_cleanly(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    wrong_episode = adjudication(evidence_id)
    wrong_episode["episode_slug"] = "17-other"
    provider = FakeProvider([claim_map(), wrong_episode, wrong_episode])
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    result = workflow.invoke("adjudication-binding-failed")

    assert result["phase1_status"] == "failed"
    assert result["phase2_status"] == "not_run"
    assert len(provider.requests) == 3
    assert "investigation" not in result


def test_invalid_revisit_repair_preserves_last_valid_provisional(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    provider = FakeProvider(
        [
            claim_map(claims=2),
            adjudication(
                evidence_id,
                claims=("C1",),
                provisional=True,
                request_id="C2",
            ),
            {"invalid": "revisit"},
            {"still": "invalid"},
        ]
    )
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path, pitch_lines=2),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    result = workflow.invoke("preserve-prior")

    assert result["phase1_iteration"] == 2
    assert result["phase1_status"] == "provisional"
    assert result["investigation"]["investigation_status"] == "provisional"
    assert "ADJUDICATION_REPAIR_FAILED_USING_PRIOR" in result["phase1_findings"]
    assert len(provider.requests) == 4


def test_structurally_invalid_revisit_preserves_prior_adjudication(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    invalid_revisit = adjudication(evidence_id, claims=("C1",))
    invalid_revisit["dispositions"][0]["pitch_evidence_ids"] = ["P-002"]
    prior = adjudication(
        evidence_id,
        claims=("C1", "C2"),
        request_id="C2",
    )
    provider = FakeProvider([claim_map(claims=2), prior, invalid_revisit])
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path, pitch_lines=2),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    result = workflow.invoke("structural-revisit")

    assert result["phase1_status"] == "provisional"
    assert result["adjudication"] == prior
    assert "STRUCTURAL_ADJUDICATION_REJECTED_USING_PRIOR" in result[
        "phase1_findings"
    ]
    assert result["investigation"]["investigation_status"] == "provisional"
    assert "STRUCTURAL_ADJUDICATION_REJECTED_USING_PRIOR" in result[
        "investigation"
    ]["validator_findings"]


def test_reused_thread_or_second_thread_cannot_reinvoke_owned_run_root(
    tmp_path: Path,
) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    first_provider = FakeProvider(
        [{"invalid": "claim"}, claim_map(), adjudication(evidence_id)]
    )
    first = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=first_provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )
    first.invoke("first-run")
    assert (tmp_path / "run/phase1/claim-extraction/call-02.json").exists()

    repair_path = tmp_path / "run/phase1/claim-extraction/call-02.json"
    expected_repair = repair_path.read_bytes()

    with pytest.raises(ValueError, match="checkpoint thread was already used"):
        first.invoke("first-run")
    with pytest.raises(ValueError, match="run root is owned by another thread"):
        first.invoke("second-run")

    assert len(first_provider.requests) == 3
    assert repair_path.read_bytes() == expected_repair


def test_constructor_requires_v44_phase1_only_dependencies(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    provider = FakeProvider([])
    valid = settings(tmp_path)
    common = {
        "phase1_provider": provider,
        "checkpointer": InMemorySaver(),
        "v44_settings": v44_settings(),
    }

    with pytest.raises(ValueError, match="contract_version v4.4"):
        VCDecisionWorkflowV44(replace(valid, contract_version="v4.3"), index, **common)
    with pytest.raises(ValueError, match="execution_mode phase1_only"):
        VCDecisionWorkflowV44(replace(valid, execution_mode="full"), index, **common)
    with pytest.raises(ValueError, match="checkpointer is required"):
        VCDecisionWorkflowV44(valid, index, **{**common, "checkpointer": None})
    with pytest.raises(ValueError, match="Phase 1 provider"):
        VCDecisionWorkflowV44(valid, index, **{**common, "phase1_provider": None})
    with pytest.raises(ValueError, match="HybridWikiIndex"):
        VCDecisionWorkflowV44(valid, None, **common)
    with pytest.raises(ValueError, match="v44_settings is required"):
        VCDecisionWorkflowV44(valid, index, **{**common, "v44_settings": None})


def test_constructor_rejects_every_nonlocal_retrieval_embedder(
    tmp_path: Path,
) -> None:
    remote_index = make_index(tmp_path, RemoteEmbedder())
    provider = FakeProvider([])
    common = {
        "phase1_provider": provider,
        "checkpointer": InMemorySaver(),
        "v44_settings": v44_settings(),
    }

    with pytest.raises(ValueError, match="local embedding backend"):
        VCDecisionWorkflowV44(settings(tmp_path), remote_index, **common)

    local_index = HybridWikiIndex(
        remote_index.wiki_root,
        remote_index.chunks,
        LocalEmbedder(),
        require_complete_embeddings=True,
    )
    remote_precedent = SimpleNamespace(
        target_slug="18-rowvigor",
        embedder=RemoteEmbedder(),
        list_episodes=lambda: (),
    )
    remote_portfolio = SimpleNamespace(
        embedder=RemoteEmbedder(),
        filtered=SimpleNamespace(
            target_episode_slug="18-rowvigor",
            target_episode_number=18,
            disclosures=(),
        ),
    )
    with pytest.raises(ValueError, match="precedent corpus.*local embedding backend"):
        VCDecisionWorkflowV44(
            settings(tmp_path),
            local_index,
            precedent_corpus=remote_precedent,
            **common,
        )
    with pytest.raises(ValueError, match="portfolio index.*local embedding backend"):
        VCDecisionWorkflowV44(
            settings(tmp_path),
            local_index,
            portfolio_index=remote_portfolio,
            **common,
        )


def test_constructor_rejects_unpinned_ollama_embeddings(tmp_path: Path) -> None:
    index = make_index(tmp_path, UnpinnedOllamaEmbedder())

    with pytest.raises(ValueError, match="unpinned Ollama embeddings"):
        VCDecisionWorkflowV44(
            settings(tmp_path),
            index,
            phase1_provider=FakeProvider([]),
            checkpointer=InMemorySaver(),
            v44_settings=v44_settings(),
        )


def test_internal_validation_type_error_propagates_without_repair_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = make_index(tmp_path)
    provider = FakeProvider([claim_map(), claim_map()])
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )

    def programming_error(*_args: object, **_kwargs: object) -> None:
        raise TypeError("internal validation bug")

    monkeypatch.setattr(
        workflow_v44_module,
        "validate_claim_map_runtime_v44",
        programming_error,
    )

    with pytest.raises(TypeError, match="internal validation bug"):
        workflow.invoke("type-error")

    assert len(provider.requests) == 1


def test_constructor_rejects_unfiltered_target_indexes(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    provider = FakeProvider([])
    common = {
        "phase1_provider": provider,
        "checkpointer": InMemorySaver(),
        "v44_settings": v44_settings(),
    }
    wrong_precedent = SimpleNamespace(
        target_slug="17-other",
        list_episodes=lambda: (),
    )
    wrong_portfolio = SimpleNamespace(
        filtered=SimpleNamespace(
            target_episode_slug="17-other",
            target_episode_number=17,
            disclosures=(),
        )
    )
    future_portfolio = SimpleNamespace(
        filtered=SimpleNamespace(
            target_episode_slug="18-rowvigor",
            target_episode_number=18,
            disclosures=(
                SimpleNamespace(
                    source_episode_slug="19-future",
                    source_episode_number=19,
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match="precedent corpus must be filtered"):
        VCDecisionWorkflowV44(
            settings(tmp_path), index, precedent_corpus=wrong_precedent, **common
        )
    with pytest.raises(ValueError, match="portfolio index must be filtered"):
        VCDecisionWorkflowV44(
            settings(tmp_path), index, portfolio_index=wrong_portfolio, **common
        )
    with pytest.raises(ValueError, match="portfolio temporal filter"):
        VCDecisionWorkflowV44(
            settings(tmp_path), index, portfolio_index=future_portfolio, **common
        )


@pytest.mark.parametrize("method", ["invoke", "resume"])
def test_phase2_is_absolutely_absent(tmp_path: Path, method: str) -> None:
    index = make_index(tmp_path)
    evidence_id = index.chunks[0].chunk_id
    provider = FakeProvider([claim_map(), adjudication(evidence_id)])
    workflow = VCDecisionWorkflowV44(
        settings(tmp_path),
        index,
        phase1_provider=provider,
        checkpointer=InMemorySaver(),
        v44_settings=v44_settings(),
    )
    result = workflow.invoke("no-phase2")
    if method == "resume":
        result = workflow.resume("no-phase2")

    assert result["phase2_status"] == "not_run"
    assert all("phase2" not in request.phase for request in provider.requests)
    assert not (tmp_path / "run/phase2").exists()


def test_importing_workflow_v44_does_not_import_full_graph_or_phase2() -> None:
    script = """
import json
import sys
import vc_clone_graph.workflow_v44
forbidden = sorted(
    name for name in sys.modules
    if name == 'vc_clone_graph.graph'
    or name == 'vc_clone_graph.prompts_v4'
    or name.startswith('vc_clone_graph.phase2')
)
print(json.dumps(forbidden))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
