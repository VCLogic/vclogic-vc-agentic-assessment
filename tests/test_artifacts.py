import json
from pathlib import Path
import shutil
from hashlib import sha256

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from vc_clone_graph.artifacts import (
    _canonical_v44_cli_provider_class,
    _v44_secret_safe,
    canonical_bytes,
    freeze_model,
    read_verified_frozen,
    verify_frozen,
    verify_phase1_artifacts,
    verify_run_artifacts,
)
from vc_clone_graph.providers.base import GenerationResult
from vc_clone_graph.providers.fake import FakeProvider
from vc_clone_graph.retrieval import HybridWikiIndex
from vc_clone_graph.config import Phase1V44Settings
from vc_clone_graph.portfolio_memory import (
    DisclosureEvidence,
    PortfolioDisclosure,
    PortfolioMemoryCorpus,
    PortfolioMemoryIndex,
    make_disclosure_id,
)
from vc_clone_graph.precedents import (
    DecisionEvidence,
    PrecedentCorpus,
    PrecedentDecision,
    PrecedentEpisode,
    TranscriptTurn,
)
from vc_clone_graph.phase1_v44 import (
    ClaimMapV44,
    ClaimRetrievalManifestV44,
    RationaleAdjudicationV44,
    TaxonomyNeighborhoodManifestV44,
    build_investigation_v44,
)
from tests.test_workflow_v44 import adjudication as adjudication_v44
from tests.test_workflow_v44 import claim_map as claim_map_v44
from tests.test_workflow_v44 import make_index as make_v44_index
from tests.test_workflow_v44 import make_workflow as make_v44_workflow
from tests.test_workflow_v44 import LocalEmbedder as V44LocalEmbedder
from tests.test_workflow_v44 import settings as v44_workflow_settings
from vc_clone_graph.workflow_v44 import VCDecisionWorkflowV44
from vc_clone_graph.schemas import (
    AnyCheckDecision,
    Counterargument,
    DeliberationStep,
    Decision,
    DecisionV2,
    DealContext,
    DealFact,
    Investigation,
    InvestigationV2,
    MarketGate,
    RationaleAssessment,
    Rationale,
    RationaleV2,
    RiskLedgerEntry,
    StandardCheckDecision,
    decision_v2_json_schema,
    decision_quality_findings,
    investigation_v2_json_schema,
    normalize_investigation_v2_payload,
    validate_decision,
    validate_decision_v2,
    validate_investigation,
    validate_investigation_v2,
)


CHECK_TIERS = {
    "no_check_tier",
    "small_exploratory",
    "standard_initial",
    "larger_conviction",
}


def test_v44_secret_scan_distinguishes_episode_slug_from_secret_assignment() -> None:
    assert _v44_secret_safe(
        "167-levee-cleaning-up-hotels-dirty-secret:turns-0-7"
    )
    assert not _v44_secret_safe("secret: abcdefgh")


class _MetadataFakeProvider(FakeProvider):
    def phase1_fingerprint(self) -> dict:
        return {
            "schema_version": "phase1-provider-fingerprint-v1",
            "provider": "fake",
            "model": "deterministic-v44-fixture",
        }

    def generate(self, request) -> GenerationResult:
        result = super().generate(request)
        return result.model_copy(
            update={
                "raw_metadata": {
                    "provider": "fake",
                    "model": "deterministic-v44-fixture",
                }
            }
        )


def test_v44_cli_provider_class_canonicalizes_exact_module_runner_alias() -> None:
    assert _canonical_v44_cli_provider_class(
        "__main__._ConfiguredPhase1ProviderV44"
    ) == "vc_clone_graph.cli._ConfiguredPhase1ProviderV44"
    assert _canonical_v44_cli_provider_class(
        "attacker._ConfiguredPhase1ProviderV44"
    ) == "attacker._ConfiguredPhase1ProviderV44"


def _adjudication_v44_with_candidate(evidence_id: str) -> dict:
    payload = adjudication_v44(evidence_id)
    candidate = dict(payload["dispositions"][0])
    candidate["taxonomy_label"] = "customer_risk"
    candidate["disposition"] = "candidate"
    candidate["salience"] = "secondary"
    candidate["justification"] = "The same evidence also raises a secondary risk signal."
    payload["dispositions"].append(candidate)
    return payload


@pytest.fixture
def v44_run(tmp_path: Path) -> Path:
    seed = tmp_path / "seed"
    seed.mkdir()
    evidence_id = make_v44_index(seed).chunks[0].chunk_id
    case = tmp_path / "case"
    case.mkdir()
    workflow, _ = make_v44_workflow(
        case, [claim_map_v44(), _adjudication_v44_with_candidate(evidence_id)]
    )
    workflow.phase1_provider = _MetadataFakeProvider(
        [claim_map_v44(), _adjudication_v44_with_candidate(evidence_id)]
    )
    # Provider identity participates in the run fingerprint, so rebuild after swapping it.
    workflow = type(workflow)(
        workflow.settings,
        workflow.index,
        phase1_provider=workflow.phase1_provider,
        checkpointer=workflow.graph.checkpointer,
        v44_settings=workflow.v44_settings,
    )
    workflow.invoke("artifact-v44")
    return case / "run"


def _rewrite_v44_json(path: Path, payload: object, *, hashed: bool = False) -> None:
    raw = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
        if hashed
        else canonical_bytes(payload)  # type: ignore[arg-type]
    )
    path.write_bytes(raw)
    if hashed:
        path.with_suffix(".sha256").write_text(
            sha256(raw).hexdigest() + "\n", encoding="ascii"
        )


def _rewrite_v44_call(v44_run: Path, record_index: int, payload: dict) -> None:
    state_path = v44_run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    record = state["call_records"][record_index]
    raw = canonical_bytes(payload)
    (v44_run / record["relative_path"]).write_bytes(raw)
    record["payload"] = payload
    record["sha256"] = sha256(raw).hexdigest()
    request = {
        key: payload[key]
        for key in (
            "phase",
            "prompt",
            "schema",
            "max_output_tokens",
            "reasoning_effort",
        )
    }
    record["request_sha256"] = sha256(canonical_bytes(request)).hexdigest()
    wal_identity = {
        "run_fingerprint": state["run_fingerprint"],
        "thread_id": state["run_thread_id"],
        "relative_call_path": record["relative_path"],
        "request_sha256": record["request_sha256"],
    }
    record["wal_id"] = sha256(canonical_bytes(wal_identity)).hexdigest()
    _rewrite_v44_json(state_path, state)


def _rebind_v44_owner(v44_run: Path, owner: dict) -> None:
    state_path = v44_run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    fingerprint = sha256(canonical_bytes(owner["fingerprint_payload"])).hexdigest()
    owner["run_fingerprint"] = fingerprint
    state["run_fingerprint"] = fingerprint
    state["run_thread_id"] = owner["thread_id"]
    for record in state["call_records"]:
        wal_identity = {
            "run_fingerprint": fingerprint,
            "thread_id": owner["thread_id"],
            "relative_call_path": record["relative_path"],
            "request_sha256": record["request_sha256"],
        }
        record["wal_id"] = sha256(canonical_bytes(wal_identity)).hexdigest()
    _rewrite_v44_json(v44_run / "run-owner-v44.json", owner)
    _rewrite_v44_json(state_path, state)


def _bind_v44_openrouter_alias_metadata(v44_run: Path) -> None:
    requested = "openrouter/auto"
    returned = "resolved/provider-model-v2"
    owner = json.loads((v44_run / "run-owner-v44.json").read_text(encoding="utf-8"))
    owner["fingerprint_payload"]["provider"] = {
        **owner["fingerprint_payload"]["provider"],
        "provider": "openrouter",
        "model": requested,
    }
    _rebind_v44_owner(v44_run, owner)
    state = json.loads((v44_run / "state.json").read_text(encoding="utf-8"))
    for record_index in range(len(state["call_records"])):
        current = json.loads((v44_run / "state.json").read_text(encoding="utf-8"))
        payload = dict(current["call_records"][record_index]["payload"])
        payload["provider_metadata"] = {
            "provider": "openrouter",
            "requested_model": requested,
            "returned_model": returned,
            "model": returned,
        }
        _rewrite_v44_call(v44_run, record_index, payload)


def test_verifies_complete_phase1_v44_artifact_bundle(v44_run: Path) -> None:
    verify_phase1_artifacts(v44_run, provenance_mode="direct")


def test_v44_verifier_rejects_non_core_compatibility_view(v44_run: Path) -> None:
    path = v44_run / "phase1/investigation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rationales"] = payload["candidate_rationales"]
    _rewrite_v44_json(path, payload, hashed=True)

    with pytest.raises(ValueError, match="core compatibility view"):
        verify_phase1_artifacts(v44_run)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("claim-bytes", "claim-map|hash"),
        ("investor-evidence", "investigation|inaccessible investor evidence"),
        ("target-precedent", "retrieval|target"),
        ("provider-metadata", "provider/model metadata"),
        ("extra-call", "call artifact set"),
        ("missing-call", "call artifact set"),
        ("tampered-call", "call record|hash|bytes"),
        ("usage", "usage"),
        ("fingerprint", "fingerprint|owner"),
        ("owner", "owner"),
        ("pending-call", "pending"),
        ("pending-response", "pending"),
        ("wal", "WAL"),
        ("phase2-call", "Phase 2"),
    ],
)
def test_v44_verifier_rejects_tampered_audit_bundle(
    v44_run: Path, mutation: str, message: str
) -> None:
    state_path = v44_run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if mutation == "claim-bytes":
        (v44_run / "phase1/claim-map.json").write_bytes(
            (v44_run / "phase1/claim-map.json").read_bytes() + b" "
        )
    elif mutation == "investor-evidence":
        investigation_path = v44_run / "phase1/investigation.json"
        investigation = json.loads(investigation_path.read_text(encoding="utf-8"))
        core_id = investigation["core_rationales"][0]["rationale_id"]
        for view in ("core_rationales", "candidate_rationales", "rationales"):
            for rationale in investigation[view]:
                if rationale["rationale_id"] == core_id:
                    rationale["wiki_evidence_ids"] = ["W-inaccessible"]
        _rewrite_v44_json(investigation_path, investigation, hashed=True)
    elif mutation == "target-precedent":
        retrieval_path = v44_run / "phase1/claim-retrieval.json"
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
        retrieval["evidence_registry"][0]["episode_slug"] = state["episode_slug"]
        _rewrite_v44_json(retrieval_path, retrieval, hashed=True)
    elif mutation == "provider-metadata":
        call_path = v44_run / state["call_records"][0]["relative_path"]
        call = json.loads(call_path.read_text(encoding="utf-8"))
        call["provider_metadata"].pop("model")
        _rewrite_v44_json(call_path, call)
        state["call_records"][0]["payload"] = call
        state["call_records"][0]["sha256"] = sha256(canonical_bytes(call)).hexdigest()
        _rewrite_v44_json(state_path, state)
    elif mutation == "extra-call":
        extra = v44_run / "phase1/adjudication/turn-01/call-99.json"
        extra.write_text("{}\n", encoding="utf-8")
    elif mutation == "missing-call":
        (v44_run / state["call_records"][0]["relative_path"]).unlink()
    elif mutation == "tampered-call":
        call_path = v44_run / state["call_records"][0]["relative_path"]
        call_path.write_bytes(call_path.read_bytes() + b" ")
    elif mutation == "usage":
        state["usage"]["input_tokens"] += 1
        _rewrite_v44_json(state_path, state)
    elif mutation == "fingerprint":
        state["run_fingerprint"] = "0" * 64
        _rewrite_v44_json(state_path, state)
    elif mutation == "owner":
        owner_path = v44_run / "run-owner-v44.json"
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        owner["thread_id"] = "other-thread"
        _rewrite_v44_json(owner_path, owner)
    elif mutation == "pending-call":
        state["pending_call"] = {"kind": "phase1_claim_extraction"}
        _rewrite_v44_json(state_path, state)
    elif mutation == "pending-response":
        state["pending_response"] = {"kind": "phase1_claim_extraction"}
        _rewrite_v44_json(state_path, state)
    elif mutation == "wal":
        wal = v44_run / ".phase1-v44-wal" / ("a" * 64 + ".json")
        wal.write_text("{}\n", encoding="utf-8")
    elif mutation == "phase2-call":
        path = v44_run / "phase2/turn-01/decision-model-response.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        verify_phase1_artifacts(v44_run)


def test_v44_verifier_is_read_only_and_rejects_symlink(v44_run: Path) -> None:
    state = v44_run / "state.json"
    original = state.read_bytes()
    external = v44_run.parent / "external-state.json"
    external.write_bytes(original)
    state.unlink()
    state.symlink_to(external)

    with pytest.raises(ValueError, match="symlink"):
        verify_phase1_artifacts(v44_run)
    assert external.read_bytes() == original


def test_v44_verifier_rejects_appended_current_target_decision_prompt(
    v44_run: Path,
) -> None:
    state = json.loads((v44_run / "state.json").read_text(encoding="utf-8"))
    payload = dict(state["call_records"][0]["payload"])
    payload["prompt"] += (
        "\nACTUAL CURRENT TARGET DECISION: INVESTED. "
        "REFERENCE RATIONALE: founder fit."
    )
    _rewrite_v44_call(v44_run, 0, payload)

    with pytest.raises(ValueError, match="prompt|forbidden|leakage"):
        verify_phase1_artifacts(v44_run)


def test_v44_verifier_rejects_unexpected_phase1_evaluation_artifact(
    v44_run: Path,
) -> None:
    (v44_run / "phase1/evaluation.json").write_text(
        json.dumps(
            {
                "target_decision": "invested",
                "reference_rationale": "founder fit",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected|inventory|forbidden"):
        verify_phase1_artifacts(v44_run)


def test_v44_verifier_rejects_self_consistent_schema_replacement(
    v44_run: Path,
) -> None:
    state = json.loads((v44_run / "state.json").read_text(encoding="utf-8"))
    payload = dict(state["call_records"][0]["payload"])
    payload["schema"] = {}
    _rewrite_v44_call(v44_run, 0, payload)

    with pytest.raises(ValueError, match="schema|request"):
        verify_phase1_artifacts(v44_run)


def test_v44_verifier_rejects_provider_identity_mismatch(v44_run: Path) -> None:
    state = json.loads((v44_run / "state.json").read_text(encoding="utf-8"))
    payload = dict(state["call_records"][0]["payload"])
    payload["provider_metadata"] = {
        **payload["provider_metadata"],
        "provider": "tampered-provider",
    }
    _rewrite_v44_call(v44_run, 0, payload)

    with pytest.raises(ValueError, match="provider.*fingerprint|identity"):
        verify_phase1_artifacts(v44_run)


def test_v44_verifier_accepts_openrouter_resolved_model_alias(v44_run: Path) -> None:
    _bind_v44_openrouter_alias_metadata(v44_run)

    verify_phase1_artifacts(v44_run, provenance_mode="direct")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requested_model", "tampered/requested-model"),
        ("model", "different/resolved-model"),
    ],
)
def test_v44_verifier_rejects_openrouter_alias_metadata_tampering(
    v44_run: Path, field: str, value: str
) -> None:
    _bind_v44_openrouter_alias_metadata(v44_run)
    state = json.loads((v44_run / "state.json").read_text(encoding="utf-8"))
    payload = dict(state["call_records"][0]["payload"])
    payload["provider_metadata"] = {
        **payload["provider_metadata"],
        field: value,
    }
    _rewrite_v44_call(v44_run, 0, payload)

    with pytest.raises(ValueError, match="model|provider|fingerprint"):
        verify_phase1_artifacts(v44_run, provenance_mode="direct")


def test_v44_verifier_requires_openrouter_requested_model_binding(
    v44_run: Path,
) -> None:
    _bind_v44_openrouter_alias_metadata(v44_run)
    state = json.loads((v44_run / "state.json").read_text(encoding="utf-8"))
    payload = dict(state["call_records"][0]["payload"])
    metadata = dict(payload["provider_metadata"])
    requested = metadata.pop("requested_model")
    metadata["returned_model"] = requested
    metadata["model"] = requested
    payload["provider_metadata"] = metadata
    _rewrite_v44_call(v44_run, 0, payload)

    with pytest.raises(ValueError, match="model|provider|fingerprint"):
        verify_phase1_artifacts(v44_run, provenance_mode="direct")


def test_v44_verifier_rejects_secret_in_owner_top_level(v44_run: Path) -> None:
    owner_path = v44_run / "run-owner-v44.json"
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    owner["api_key"] = "must-not-be-persisted"
    _rewrite_v44_json(owner_path, owner)

    with pytest.raises(ValueError, match="owner|secret"):
        verify_phase1_artifacts(v44_run)


def test_v44_verifier_rejects_unexplained_accepted_findings(v44_run: Path) -> None:
    state_path = v44_run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["phase1_status"] == "accepted"
    state["phase1_findings"] = ["TAMPERED_FINDING"]
    _rewrite_v44_json(state_path, state)

    with pytest.raises(ValueError, match="findings|accepted"):
        verify_phase1_artifacts(v44_run)


def test_v44_verifier_rejects_taxonomy_hash_rebound_away_from_owner(
    v44_run: Path,
) -> None:
    taxonomy_path = v44_run / "phase1/taxonomy-neighborhood.json"
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    taxonomy["taxonomy_sha256"] = "0" * 64
    _rewrite_v44_json(taxonomy_path, taxonomy, hashed=True)
    taxonomy_digest = sha256(taxonomy_path.read_bytes()).hexdigest()

    investigation_path = v44_run / "phase1/investigation.json"
    investigation = json.loads(investigation_path.read_text(encoding="utf-8"))
    investigation["taxonomy_neighborhood_manifest"] = taxonomy
    investigation["taxonomy_neighborhood_sha256"] = taxonomy_digest
    _rewrite_v44_json(investigation_path, investigation, hashed=True)
    investigation_digest = sha256(investigation_path.read_bytes()).hexdigest()

    state_path = v44_run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["taxonomy_neighborhood"] = taxonomy
    state["taxonomy_neighborhood_manifest"] = taxonomy
    state["taxonomy_neighborhood_sha256"] = taxonomy_digest
    state["investigation"] = investigation
    state["investigation_sha256"] = investigation_digest
    _rewrite_v44_json(state_path, state)

    with pytest.raises(ValueError, match="taxonomy.*fingerprint|taxonomy.*hash"):
        verify_phase1_artifacts(v44_run)


@pytest.fixture
def v44_revisit_run(tmp_path: Path) -> Path:
    seed = tmp_path / "revisit-seed"
    seed.mkdir()
    evidence_id = make_v44_index(seed).chunks[0].chunk_id
    first = adjudication_v44(
        evidence_id, provisional=True, request_id="C1"
    )
    second = adjudication_v44(evidence_id)
    second["dispositions"][0]["confidence"] = 0.95
    case = tmp_path / "revisit-case"
    case.mkdir()
    workflow, _ = make_v44_workflow(case, [claim_map_v44(), first, second])
    workflow = type(workflow)(
        workflow.settings,
        workflow.index,
        phase1_provider=_MetadataFakeProvider([claim_map_v44(), first, second]),
        checkpointer=workflow.graph.checkpointer,
        v44_settings=workflow.v44_settings,
    )
    workflow.invoke("artifact-v44-revisit")
    return case / "run"


def test_v44_verifier_rejects_stale_turn_one_adjudication(
    v44_revisit_run: Path,
) -> None:
    root = v44_revisit_run
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    first_payload = state["call_records"][1]["payload"]["parsed"]
    claim_map = ClaimMapV44.model_validate_json(
        (root / "phase1/claim-map.json").read_bytes()
    )
    retrieval = ClaimRetrievalManifestV44.model_validate_json(
        (root / "phase1/claim-retrieval.json").read_bytes()
    )
    neighborhood = TaxonomyNeighborhoodManifestV44.model_validate_json(
        (root / "phase1/taxonomy-neighborhood.json").read_bytes()
    )
    stale = RationaleAdjudicationV44.model_validate_json(json.dumps(first_payload))
    adjudication_path = root / "phase1/adjudication.json"
    _rewrite_v44_json(
        adjudication_path, stale.model_dump(mode="json"), hashed=True
    )
    adjudication_digest = sha256(adjudication_path.read_bytes()).hexdigest()
    pitch_ids = [
        row["evidence_id"]
        for row in json.loads(
            (root / "run-owner-v44.json").read_text(encoding="utf-8")
        )["fingerprint_payload"]["pitch_evidence"]
    ]
    investigation = build_investigation_v44(
        episode_slug=stale.episode_slug,
        claim_map=claim_map,
        adjudication=stale,
        retrieval_manifest=retrieval,
        neighborhood=neighborhood,
        claim_map_sha256=state["claim_map_sha256"],
        adjudication_sha256=adjudication_digest,
        pitch_evidence_ids=pitch_ids,
    )
    investigation_path = root / "phase1/investigation.json"
    _rewrite_v44_json(
        investigation_path, investigation.model_dump(mode="json"), hashed=True
    )
    investigation_digest = sha256(investigation_path.read_bytes()).hexdigest()
    state["adjudication"] = stale.model_dump(mode="json")
    state["last_valid_adjudication"] = stale.model_dump(mode="json")
    state["adjudication_sha256"] = adjudication_digest
    state["investigation"] = investigation.model_dump(mode="json")
    state["investigation_sha256"] = investigation_digest
    state["phase1_status"] = "provisional"
    state["phase1_action"] = "freeze_provisional"
    state["phase1_findings"] = list(investigation.validator_findings)
    _rewrite_v44_json(state_path, state)

    with pytest.raises(ValueError, match="terminal|last.*valid|stale|turn"):
        verify_phase1_artifacts(root)


@pytest.mark.parametrize(
    "leakage",
    [
        "ACTUAL CURRENT TARGET DECISION: INVESTED; REFERENCE RATIONALE: founder fit",
        "actual current target-decision: invested",
        "reference-rationale: founder fit",
        "evaluation-result: exact match",
    ],
)
def test_v44_verifier_rejects_semantic_leakage_in_optional_json(
    v44_run: Path, leakage: str
) -> None:
    _rewrite_v44_json(
        v44_run / "input-provenance.json",
        {
            "note": leakage
        },
    )

    with pytest.raises(ValueError, match="leakage|semantic|forbidden"):
        verify_phase1_artifacts(v44_run)


@pytest.mark.parametrize(
    "secret_value",
    [
        "Bearer sk-live-not-a-safe-thread-id",
        "sk-live-0123456789abcdefghijklmnop",
        "sk-or-v1-0123456789abcdefghijklmnop",
        "sk-proj-0123456789abcdefghijklmnop",
        "sk-A1b2C3d4E5f6G7h8I9j0K1l2M3n4",
        "sk-abcdefghijklmnopqrstuvwxyz1234",
        "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234",
        "sk-AbCdEfGhIjKlMnOpQrStUvWxYz",
        "https://safe.example/v1?token=sk-live-url-secret",
        "-----BEGIN PRIVATE KEY-----\nnot-safe\n-----END PRIVATE KEY-----",
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_0123456789abcdefghijklmnopqrstuv",
        "xoxb-123456789012-123456789012-abcdefghijklmnopqrstuvwx",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturevalue",
    ],
)
def test_v44_verifier_rejects_secret_shaped_owner_values(
    v44_run: Path, secret_value: str
) -> None:
    owner = json.loads((v44_run / "run-owner-v44.json").read_text(encoding="utf-8"))
    owner["thread_id"] = secret_value
    _rebind_v44_owner(v44_run, owner)

    with pytest.raises(ValueError, match="owner|secret|credential"):
        verify_phase1_artifacts(v44_run)


def test_v44_verifier_accepts_benign_sk_learning_thread(v44_run: Path) -> None:
    owner = json.loads((v44_run / "run-owner-v44.json").read_text(encoding="utf-8"))
    owner["thread_id"] = "sk-learning-run-001"
    _rebind_v44_owner(v44_run, owner)

    verify_phase1_artifacts(v44_run)


@pytest.mark.parametrize(
    "leakage",
    [
        "gold label: invested",
        "ground-truth label: invested",
    ],
)
def test_v44_verifier_rejects_target_labels_outside_pitch_prose(
    v44_run: Path, leakage: str
) -> None:
    _rewrite_v44_json(
        v44_run / "wiki-sanitization.json",
        {
            "alias_hashes": [],
            "changed_chunk_ids": [],
            "changed_chunk_count": 0,
            "reused_embedding_count": 0,
            "recomputed_embedding_count": 0,
            "lexical_only_chunk_ids": [],
            "quality_findings": [leakage],
        },
    )

    with pytest.raises(ValueError, match="leakage|semantic|forbidden"):
        verify_phase1_artifacts(v44_run)


def test_v44_verifier_rejects_reordered_call_records(v44_run: Path) -> None:
    state_path = v44_run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["call_records"].reverse()
    _rewrite_v44_json(state_path, state)

    with pytest.raises(ValueError, match="chronological|call.*order|topology"):
        verify_phase1_artifacts(v44_run)


def test_v44_verifier_rejects_self_consistent_terminal_topology_tamper(
    v44_run: Path,
) -> None:
    state_path = v44_run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["phase1_action"] = "failed"
    state["targeted_retrieval_ids"] = ["Q999"]
    state["events"][-1]["status"] = "failed"
    _rewrite_v44_json(state_path, state)

    with pytest.raises(ValueError, match="event|action|targeted|terminal|topology"):
        verify_phase1_artifacts(v44_run)


def test_v44_verifier_rejects_owner_wiki_contradiction(v44_run: Path) -> None:
    owner = json.loads((v44_run / "run-owner-v44.json").read_text(encoding="utf-8"))
    owner["fingerprint_payload"]["wiki_index"]["chunks"][0]["text"] = (
        "Fabricated wiki content with the same chunk identity."
    )
    _rebind_v44_owner(v44_run, owner)

    with pytest.raises(ValueError, match="wiki.*(evidence|source|chunk|fingerprint)"):
        verify_phase1_artifacts(v44_run)


def test_v44_verifier_rejects_stale_unretrieved_owner_wiki_chunk_id(
    v44_run: Path,
) -> None:
    owner = json.loads((v44_run / "run-owner-v44.json").read_text(encoding="utf-8"))
    wiki = owner["fingerprint_payload"]["wiki_index"]
    stale = dict(wiki["chunks"][0])
    stale["source_path"] = "unused.md"
    stale["source_sha256"] = "c" * 64
    stale["heading"] = "Unused"
    stale["text"] = "Unused owner wiki content."
    stale["chunk_id"] = "W-" + "d" * 20
    wiki["chunks"].append(stale)
    wiki["embedding_index"]["total_count"] = 2
    wiki["embedding_index"]["embedded_count"] = 2
    _rebind_v44_owner(v44_run, owner)

    with pytest.raises(ValueError, match="wiki.*(chunk|identity|fingerprint)"):
        verify_phase1_artifacts(v44_run)


def test_v44_verifier_accepts_valid_targeted_revisit(v44_revisit_run: Path) -> None:
    verify_phase1_artifacts(v44_revisit_run)


def test_v44_verifier_accepts_globally_available_cross_target_label(
    tmp_path: Path,
) -> None:
    class SeparatedEmbedder:
        metadata = {
            "backend": "sentence_transformers",
            "model": "artifact-separated-embedding",
            "revision": "rev-1",
            "normalize": True,
        }

        @staticmethod
        def embed(texts: list[str]) -> list[list[float]]:
            return [
                [
                    float("execution" in text.casefold() or "pilot" in text.casefold()),
                    float("risk" in text.casefold() or "concentration" in text.casefold()),
                ]
                for text in texts
            ]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embed(texts)

        def embed_queries(self, texts: list[str]) -> list[list[float]]:
            return self.embed(texts)

    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "execution.md").write_text(
        "# Execution\nThe investor values founder execution and paid pilots.\n",
        encoding="utf-8",
    )
    (wiki / "risk.md").write_text(
        "# Risk\nThe investor treats customer concentration as material risk.\n",
        encoding="utf-8",
    )
    index = HybridWikiIndex.build(
        wiki,
        SeparatedEmbedder(),
        require_complete_embeddings=True,
    )
    evidence_id = next(
        row.chunk_id for row in index.chunks if row.source_path == "execution.md"
    )
    output = adjudication_v44(evidence_id, claims=("C1", "C2"))
    workflow = VCDecisionWorkflowV44(
        v44_workflow_settings(tmp_path, pitch_lines=2),
        index,
        phase1_provider=_MetadataFakeProvider([claim_map_v44(claims=2), output]),
        checkpointer=InMemorySaver(),
        v44_settings=Phase1V44Settings(
            taxonomy_top_k=1,
            claim_retrieval_top_k=1,
            max_wiki_reads_per_claim=1,
            max_precedent_reads_per_claim=0,
            max_revisits=0,
        ),
    )

    result = workflow.invoke("artifact-v44-cross-target-label")
    neighborhoods = {
        row["target_id"]: {
            candidate["taxonomy_label"] for candidate in row["candidates"]
        }
        for row in result["taxonomy_neighborhood"]["claim_neighborhoods"]
    }
    assert "founder_execution" in result["taxonomy_neighborhood"]["ordered_labels"]
    assert "founder_execution" not in neighborhoods["C2"]

    verify_phase1_artifacts(tmp_path / "run")


def _run_v44_outputs(tmp_path: Path, outputs: list[dict], thread_id: str) -> Path:
    seed = tmp_path / f"{thread_id}-seed"
    seed.mkdir()
    make_v44_index(seed)
    case = tmp_path / f"{thread_id}-case"
    case.mkdir()
    workflow, _ = make_v44_workflow(case, outputs)
    workflow = type(workflow)(
        workflow.settings,
        workflow.index,
        phase1_provider=_MetadataFakeProvider(outputs),
        checkpointer=workflow.graph.checkpointer,
        v44_settings=workflow.v44_settings,
    )
    workflow.invoke(thread_id)
    return case / "run"


def test_v44_verifier_accepts_valid_claim_repair(tmp_path: Path) -> None:
    seed = tmp_path / "repair-evidence"
    seed.mkdir()
    evidence_id = make_v44_index(seed).chunks[0].chunk_id
    root = _run_v44_outputs(
        tmp_path,
        [{"bad": "claim"}, claim_map_v44(), adjudication_v44(evidence_id)],
        "artifact-v44-repair",
    )

    verify_phase1_artifacts(root)


def test_v44_verifier_accepts_valid_adjudication_repair(tmp_path: Path) -> None:
    seed = tmp_path / "adjudication-repair-evidence"
    seed.mkdir()
    evidence_id = make_v44_index(seed).chunks[0].chunk_id
    root = _run_v44_outputs(
        tmp_path,
        [claim_map_v44(), {"bad": "adjudication"}, adjudication_v44(evidence_id)],
        "artifact-v44-adjudication-repair",
    )

    verify_phase1_artifacts(root)


def test_v44_verifier_accepts_valid_prior_preservation(tmp_path: Path) -> None:
    seed = tmp_path / "prior-evidence"
    seed.mkdir()
    evidence_id = make_v44_index(seed).chunks[0].chunk_id
    first = adjudication_v44(evidence_id, provisional=True, request_id="C1")
    root = _run_v44_outputs(
        tmp_path,
        [claim_map_v44(), first, {"bad": "primary"}, {"bad": "repair"}],
        "artifact-v44-prior-preserved",
    )

    verify_phase1_artifacts(root)


@pytest.fixture
def v44_upstream_run(tmp_path: Path) -> Path:
    index = make_v44_index(tmp_path)
    embedder = V44LocalEmbedder()
    historical_text = "Founder describes durable enterprise retention."
    decision_text = "The retention evidence is sufficient for an investment."
    precedent = PrecedentCorpus.from_episodes(
        (
            PrecedentEpisode(
                episode_slug="17-earlier",
                episode_number=17,
                source_path="17-earlier.json",
                source_sha256="a" * 64,
                investor_aliases=("Charles",),
                investor_present=True,
                turns=(
                    TranscriptTurn(
                        turn_index=0, speaker="Founder", text=historical_text
                    ),
                    TranscriptTurn(
                        turn_index=1, speaker="Charles", text=decision_text
                    ),
                ),
                decision=PrecedentDecision(
                    status="In",
                    context="initial_panel",
                    check_tier=None,
                    conditions=(),
                    evidence=(
                        DecisionEvidence(
                            turn_start=1, turn_end=1, text=decision_text
                        ),
                    ),
                    audit_source="fixture",
                    audit_notes="Observed decision.",
                ),
            ),
        ),
        embedder,
        target_slug="18-rowvigor",
        require_complete_embeddings=True,
    )
    disclosure_id = make_disclosure_id(
        "charles-hudson", "17-earlier", 2, "OverlapCo"
    )
    portfolio_corpus = PortfolioMemoryCorpus(
        "charles-hudson",
        (
            PortfolioDisclosure(
                disclosure_id=disclosure_id,
                vc_slug="charles-hudson",
                company_name="OverlapCo",
                aliases=("OverlapCo",),
                descriptor="Enterprise retention platform",
                relationship="investment",
                observed_overlap="possible",
                observed_consequence="permission_or_check_required",
                source_episode_slug="17-earlier",
                source_episode_number=17,
                evidence=(
                    DisclosureEvidence(
                        source_sha256="b" * 64,
                        turn_index=2,
                        speaker="Charles",
                        text="I invested in OverlapCo for enterprise retention.",
                    ),
                ),
                confidence=1.0,
                validation_status="human_validated",
            ),
        ),
    )
    portfolio = PortfolioMemoryIndex.build(portfolio_corpus, embedder).for_target(
        "18-rowvigor"
    )
    evidence_id = index.chunks[0].chunk_id
    outputs = [claim_map_v44(), adjudication_v44(evidence_id)]
    workflow = VCDecisionWorkflowV44(
        v44_workflow_settings(tmp_path),
        index,
        phase1_provider=_MetadataFakeProvider(outputs),
        checkpointer=InMemorySaver(),
        v44_settings=Phase1V44Settings(
            taxonomy_top_k=2,
            claim_retrieval_top_k=1,
            max_wiki_reads_per_claim=1,
            max_precedent_reads_per_claim=1,
            max_revisits=1,
        ),
        precedent_corpus=precedent,
        portfolio_index=portfolio,
    )
    workflow.invoke("artifact-v44-upstream")
    root = tmp_path / "run"
    _rewrite_v44_json(
        root / "precedent-manifest.filtered.json",
        precedent.filtered_manifest().model_dump(mode="json"),
    )
    _rewrite_v44_json(
        root / "portfolio-manifest.filtered.json",
        portfolio.filtered.manifest().model_dump(mode="json"),
    )
    return root


def test_v44_verifier_accepts_bound_precedent_and_portfolio(
    v44_upstream_run: Path,
) -> None:
    verify_phase1_artifacts(v44_upstream_run)


def test_v44_verifier_rejects_precedent_source_mismatch(
    v44_upstream_run: Path,
) -> None:
    owner_path = v44_upstream_run / "run-owner-v44.json"
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    manifest = owner["fingerprint_payload"]["precedent"]["filtered_manifest"]
    manifest["accessible_episodes"][0]["source_sha256"] = "0" * 64
    digest_payload = {key: value for key, value in manifest.items() if key != "sha256"}
    manifest["sha256"] = sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _rewrite_v44_json(v44_upstream_run / "precedent-manifest.filtered.json", manifest)
    _rebind_v44_owner(v44_upstream_run, owner)

    with pytest.raises(ValueError, match="historical|precedent.*(source|fingerprint)"):
        verify_phase1_artifacts(v44_upstream_run)


def test_v44_verifier_rejects_portfolio_disclosure_mismatch(
    v44_upstream_run: Path,
) -> None:
    owner_path = v44_upstream_run / "run-owner-v44.json"
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    owner["fingerprint_payload"]["portfolio"]["disclosures"][0]["evidence"][0][
        "text"
    ] = "Fabricated portfolio disclosure text."
    _rebind_v44_owner(v44_upstream_run, owner)

    with pytest.raises(ValueError, match="portfolio.*(evidence|disclosure)"):
        verify_phase1_artifacts(v44_upstream_run)


def investigation() -> Investigation:
    return Investigation(
        episode_slug="135-thoras-ai-the-twin-effect",
        questions=["Does the team learn quickly from customers?"],
        rationales=[
            Rationale(
                rationale_id="R1",
                label="founder_execution",
                direction="positive",
                salience="primary",
                confidence=0.8,
                pitch_evidence=["five paid pilots"],
                wiki_evidence_ids=["W-12345678901234567890"],
                interpretation="Paid pilots indicate learning speed.",
            )
        ],
        conflicts=[],
        unanswered_questions=[],
        saturated=True,
        summary="One strong founder signal.",
    )


def unknown_deal_fact() -> DealFact:
    return DealFact(status="unknown", value=None, pitch_evidence=[])


def investigation_v2() -> InvestigationV2:
    return InvestigationV2(
        schema_version="investigation-v2",
        episode_slug="135-thoras-ai-the-twin-effect",
        questions=["Does the team learn quickly from customers?"],
        rationales=[
            RationaleV2(
                rationale_id="R1",
                label="founder_execution",
                direction="positive",
                salience="primary",
                confidence=0.8,
                pitch_evidence=["five paid pilots"],
                wiki_evidence_ids=["W-12345678901234567890"],
                interpretation="Paid pilots indicate learning speed.",
                evidence_status="affirmative_positive",
                constraint_kind="none",
                constraint_severity="none",
                severity_basis="This is a positive execution signal, not a constraint.",
            )
        ],
        deal_context=DealContext(
            total_round_size=unknown_deal_fact(),
            company_stage=unknown_deal_fact(),
            entry_valuation=unknown_deal_fact(),
            possible_investor_check=unknown_deal_fact(),
            lead_required=unknown_deal_fact(),
            ownership_feasibility=unknown_deal_fact(),
        ),
        conflicts=[],
        unanswered_questions=[],
        saturated=True,
        summary="One strong founder signal.",
    )


def test_v2_investigation_requires_provenance_and_separated_deal_context() -> None:
    candidate = investigation_v2()

    assert candidate.schema_version == "investigation-v2"
    assert candidate.rationales[0].evidence_status == "affirmative_positive"
    assert set(DealContext.model_fields) == {
        "total_round_size",
        "company_stage",
        "entry_valuation",
        "possible_investor_check",
        "lead_required",
        "ownership_feasibility",
    }


def test_v2_schema_binds_episode_taxonomy_and_wiki_evidence() -> None:
    schema = investigation_v2_json_schema(
        episode_slug="135-thoras-ai-the-twin-effect",
        taxonomy_labels={"founder_execution", "capital_efficiency_assessment"},
        exact_wiki_ids={"W-12345678901234567890"},
    )

    assert schema["properties"]["episode_slug"]["const"] == (
        "135-thoras-ai-the-twin-effect"
    )
    rationale = schema["$defs"]["RationaleV2"]["properties"]
    assert rationale["label"]["enum"] == [
        "capital_efficiency_assessment",
        "founder_execution",
    ]
    assert rationale["wiki_evidence_ids"]["items"]["enum"] == [
        "W-12345678901234567890"
    ]


def test_v2_observed_deal_fact_requires_pitch_supported_evidence() -> None:
    candidate = investigation_v2()
    candidate.deal_context.total_round_size = DealFact(
        status="observed",
        value="$1.5 million",
        pitch_evidence=["raising $1.5 million"],
    )
    validate_investigation_v2(
        candidate,
        episode_slug=candidate.episode_slug,
        pitch="We have five paid pilots and are raising $1.5 million for the seed round.",
        taxonomy_labels={"founder_execution"},
        exact_wiki_ids={"W-12345678901234567890"},
    )
    candidate.deal_context.total_round_size.pitch_evidence = ["raising $3 million"]

    with pytest.raises(ValueError, match="deal fact evidence"):
        validate_investigation_v2(
            candidate,
            episode_slug=candidate.episode_slug,
            pitch="We have five paid pilots and are raising $1.5 million for the seed round.",
            taxonomy_labels={"founder_execution"},
            exact_wiki_ids={"W-12345678901234567890"},
        )


def test_v2_unknown_deal_fact_can_cite_evidence_explaining_uncertainty() -> None:
    candidate = investigation_v2()
    candidate.deal_context.possible_investor_check = DealFact(
        status="unknown",
        value=None,
        pitch_evidence=["soft commitments cover most of the round"],
    )
    validate_investigation_v2(
        candidate,
        episode_slug=candidate.episode_slug,
        pitch=(
            "We have five paid pilots, and soft commitments cover most of the round, "
            "but no investor check is specified."
        ),
        taxonomy_labels={"founder_execution"},
        exact_wiki_ids={"W-12345678901234567890"},
    )

    candidate.deal_context.possible_investor_check.pitch_evidence = [
        "Elizabeth offered a $50,000 check"
    ]
    with pytest.raises(ValueError, match="deal fact evidence"):
        validate_investigation_v2(
            candidate,
            episode_slug=candidate.episode_slug,
            pitch=(
                "We have five paid pilots, and soft commitments cover most of the "
                "round, but no investor check is specified."
            ),
            taxonomy_labels={"founder_execution"},
            exact_wiki_ids={"W-12345678901234567890"},
        )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (
            {
                "evidence_status": "unresolved",
                "constraint_kind": "portfolio_overlap",
                "constraint_severity": "fatal",
            },
            "unresolved evidence cannot be fatal",
        ),
        (
            {
                "evidence_status": "affirmative_adverse",
                "constraint_kind": "total_round_size",
                "constraint_severity": "fatal",
            },
            "total round size cannot be fatal",
        ),
    ],
)
def test_v2_rejects_invalid_evidence_severity_combinations(
    update: dict, message: str
) -> None:
    rationale = investigation_v2().rationales[0].model_copy(update=update)

    with pytest.raises(ValueError, match=message):
        RationaleV2.model_validate(rationale.model_dump())


def test_v2_allows_affirmative_adverse_routine_constraint() -> None:
    rationale = investigation_v2().rationales[0].model_copy(
        update={
            "direction": "negative",
            "evidence_status": "affirmative_adverse",
            "constraint_kind": "portfolio_overlap",
            "constraint_severity": "routine",
        }
    )

    assert RationaleV2.model_validate(rationale.model_dump()).constraint_severity == (
        "routine"
    )


def test_v2_normalizes_untyped_nonzero_constraint_without_losing_severity() -> None:
    payload = investigation_v2().model_dump(mode="json")
    payload["rationales"][0].update(
        {
            "direction": "negative",
            "evidence_status": "affirmative_adverse",
            "constraint_kind": "none",
            "constraint_severity": "material",
        }
    )

    normalized, findings = normalize_investigation_v2_payload(payload)

    assert normalized["rationales"][0]["constraint_kind"] == "other"
    assert normalized["rationales"][0]["constraint_severity"] == "material"
    assert findings == ["NORMALIZED_UNTYPED_CONSTRAINT:R1:none->other"]
    assert InvestigationV2.model_validate(normalized).rationales[0].constraint_severity == (
        "material"
    )


def test_v2_notes_and_repairs_empty_basis_when_no_constraint_exists() -> None:
    payload = investigation_v2().model_dump(mode="json")
    payload["rationales"][0]["severity_basis"] = ""

    normalized, findings = normalize_investigation_v2_payload(payload)

    assert normalized["rationales"][0]["severity_basis"] == "No constraint identified."
    assert findings == ["NORMALIZED_EMPTY_NO_CONSTRAINT_BASIS:R1"]
    assert InvestigationV2.model_validate(normalized).rationales[0].constraint_kind == (
        "none"
    )


def test_v2_notes_and_repairs_empty_basis_when_constraint_exists() -> None:
    payload = investigation_v2().model_dump(mode="json")
    payload["rationales"][0].update(
        {
            "constraint_kind": "total_round_size",
            "constraint_severity": "size_limiting",
            "severity_basis": "",
        }
    )

    normalized, findings = normalize_investigation_v2_payload(payload)

    assert normalized["rationales"][0]["severity_basis"] == (
        "Model supplied no severity basis; review required."
    )
    assert findings == ["NORMALIZED_EMPTY_CONSTRAINT_BASIS:R1"]
    assert InvestigationV2.model_validate(normalized).rationales[0].constraint_severity == (
        "size_limiting"
    )


def test_v2_allows_named_dimension_with_no_constraint_found() -> None:
    rationale = investigation_v2().rationales[0].model_copy(
        update={
            "constraint_kind": "vision_alignment",
            "constraint_severity": "none",
            "severity_basis": "Vision alignment was examined and no constraint was found.",
        }
    )

    assert RationaleV2.model_validate(rationale.model_dump()).constraint_severity == (
        "none"
    )


def test_phase1_schema_contains_no_investment_decision_field() -> None:
    properties = Investigation.model_json_schema()["properties"]
    assert {"decision", "investment_likelihood", "ranking_score"}.isdisjoint(properties)


def test_investigation_requires_real_pitch_and_wiki_evidence() -> None:
    candidate = investigation()
    validate_investigation(
        candidate,
        episode_slug="135-thoras-ai-the-twin-effect",
        pitch="The founders have five paid pilots.",
        taxonomy_labels={"founder_execution"},
        exact_wiki_ids={"W-12345678901234567890"},
    )
    with pytest.raises(ValueError, match="pitch evidence"):
        validate_investigation(
            candidate,
            episode_slug="135-thoras-ai-the-twin-effect",
            pitch="No customers yet.",
            taxonomy_labels={"founder_execution"},
            exact_wiki_ids={"W-12345678901234567890"},
        )


def test_investigation_accepts_close_pitch_paraphrases_but_not_inventions() -> None:
    candidate = investigation()
    candidate.rationales[0].pitch_evidence = [
        "we currently have five design partnerships"
    ]
    validate_investigation(
        candidate,
        episode_slug="135-thoras-ai-the-twin-effect",
        pitch="And so currently we have five design partnerships.",
        taxonomy_labels={"founder_execution"},
        exact_wiki_ids={"W-12345678901234567890"},
    )
    candidate.rationales[0].pitch_evidence = ["three profitable annual contracts"]
    with pytest.raises(ValueError, match="pitch evidence"):
        validate_investigation(
            candidate,
            episode_slug="135-thoras-ai-the-twin-effect",
            pitch="And so currently we have five design partnerships.",
            taxonomy_labels={"founder_execution"},
            exact_wiki_ids={"W-12345678901234567890"},
        )


def test_canonical_freeze_round_trip_and_hash_verification(tmp_path: Path) -> None:
    frozen_path, digest = freeze_model(tmp_path, "investigation", investigation())
    assert verify_frozen(frozen_path, tmp_path / "investigation.sha256") == digest
    assert json.loads(frozen_path.read_text())["episode_slug"].startswith("135-")
    frozen_path.write_text(frozen_path.read_text() + " ")
    with pytest.raises(ValueError, match="hash"):
        verify_frozen(frozen_path, tmp_path / "investigation.sha256")


def test_verified_frozen_read_returns_the_exact_hashed_bytes(tmp_path: Path) -> None:
    frozen_path, digest = freeze_model(tmp_path, "investigation", investigation())
    raw, actual = read_verified_frozen(
        frozen_path, tmp_path / "investigation.sha256"
    )
    assert actual == digest
    assert raw == frozen_path.read_bytes()


def dual_decision() -> Decision:
    any_counterargument = Counterargument(
        argument="Scale is unproven.",
        rationale_ids=["R2"],
        response="That limits size rather than eliminating the check.",
    )
    standard_counterargument = Counterargument(
        argument="Paid demand may justify conviction.",
        rationale_ids=["R1"],
        response="It does not establish standard-check scale.",
    )
    return Decision(
        episode_slug="135-thoras-ai-the-twin-effect",
        investigation_sha256="a" * 64,
        decision="In",
        investment_likelihood=0.61,
        decision_confidence=0.7,
        ranking_score=0.75,
        check_tier="small_exploratory",
        decision_endpoint="any_check",
        recommended_check_tier="small_exploratory",
        controlling_rationale_ids=["R1", "R2"],
        any_check=AnyCheckDecision(
            decision="In",
            likelihood=0.61,
            confidence=0.7,
            supporting_rationale_ids=["R1"],
            opposing_rationale_ids=["R2"],
            fatal_constraint_present=False,
            optionality_explanation="Paid demand supports a bounded check.",
            strongest_counterargument=any_counterargument,
            reversal_conditions=["Paid demand cannot be verified."],
        ),
        standard_check=StandardCheckDecision(
            decision="Out",
            likelihood=0.22,
            confidence=0.86,
            supporting_rationale_ids=["R1"],
            opposing_rationale_ids=["R2"],
            failure_rationale_ids=["R2"],
            market_gate=MarketGate(
                status="negative",
                controlling_rationale_ids=["R2"],
                explanation="The venture-scale path is missing.",
            ),
            strongest_counterargument=standard_counterargument,
            upgrade_conditions=["Show a bottoms-up scale path."],
        ),
        risk_ledger=[
            RiskLedgerEntry(
                rationale_id="R2",
                risk_type="size_limiting",
                controlling_for_any_check=False,
                counterevidence=["Paid demand"],
                explanation="Missing scale evidence limits check size.",
            )
        ],
        rationale_assessments=[
            RationaleAssessment(
                rationale_id="R1",
                effective_direction="positive",
                decision_weight="decisive",
                assessment="Paid pilots support proceeding.",
                counterevidence=["Traction remains early."],
            ),
            RationaleAssessment(
                rationale_id="R2",
                effective_direction="negative",
                decision_weight="decisive",
                assessment="Scale is not demonstrated.",
                counterevidence=["Paid demand reduces downside."],
            ),
        ],
        deliberation_steps=[
            DeliberationStep(
                step_id=f"D{index}",
                endpoint=endpoint,
                stage=stage,
                question="Does the execution signal clear the bar?",
                rationale_ids=rationale_ids,
                evidence_assessment="Promising, with limited production evidence.",
                likelihood_before=before,
                likelihood_after=after,
                effect=effect,
                check_tier_implication=tier,
                decision_update="Update the relevant decision endpoint.",
            )
            for index, endpoint, stage, rationale_ids, before, after, effect, tier in [
                (1, "any_check", "assessment", ["R1"], 0.5, 0.65, "raises", "small_exploratory"),
                (2, "any_check", "opposing_case", ["R2"], 0.65, 0.58, "lowers", "small_exploratory"),
                (3, "any_check", "consistency", ["R1", "R2"], 0.58, 0.61, "raises", "small_exploratory"),
                (4, "standard_check", "assessment", ["R2"], 0.5, 0.3, "lowers", "standard_initial"),
                (5, "standard_check", "opposing_case", ["R1"], 0.3, 0.35, "raises", "standard_initial"),
                (6, "standard_check", "consistency", ["R1", "R2"], 0.35, 0.22, "lowers", "no_check_tier"),
            ]
        ],
        strongest_counterargument="Traction remains early.",
        unresolved_questions=[],
        reversal_conditions=[],
        feedback="Proceed with diligence.",
    )


def dual_decision_v2() -> DecisionV2:
    return DecisionV2.model_validate(
        {"schema_version": "decision-v2", **dual_decision().model_dump()}
    )


def test_run_artifact_verifier_rejects_phase_usage_tamper(tmp_path: Path) -> None:
    investigation = investigation_v2()
    _, investigation_digest = freeze_model(tmp_path / "phase1", "investigation", investigation)
    decision = dual_decision_v2().model_copy(
        update={"investigation_sha256": investigation_digest}
    )
    freeze_model(tmp_path / "phase2", "decision", decision)
    response = tmp_path / "phase1/turn-01/investigation-model-response.json"
    response.parent.mkdir(parents=True)
    response.write_text(json.dumps({"raw_metadata": {"provider": "fake"}}), encoding="utf-8")
    state = {
        "phase1_status": "accepted",
        "phase2_status": "accepted",
        "investigation_sha256": (tmp_path / "phase1/investigation.sha256").read_text().strip(),
        "investigation": investigation.model_dump(mode="json"),
        "phase1_historical_evidence": [],
        "phase2_historical_evidence": [],
        "phase1_precedent_reads": [],
        "phase2_precedent_reads": [],
        "usage": {"input_tokens": 2, "cached_input_tokens": 0, "output_tokens": 1, "cost_usd": 0.1},
        "usage_by_phase": {
            "phase1": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1, "cost_usd": 0.1},
            "phase2": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        },
    }
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    verify_run_artifacts(tmp_path)
    state["usage_by_phase"]["phase2"]["input_tokens"] = 9
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="phase usage"):
        verify_run_artifacts(tmp_path)


def _write_strict_v3_run(root: Path) -> None:
    frozen_investigation = investigation_v2()
    _, investigation_digest = freeze_model(
        root / "phase1", "investigation", frozen_investigation
    )
    freeze_model(
        root / "phase2",
        "decision",
        dual_decision_v2().model_copy(
            update={"investigation_sha256": investigation_digest}
        ),
    )
    target = frozen_investigation.episode_slug
    manifest = {
        "schema_version": "precedent-filtered-manifest-v1",
        "target_episode_slug": target,
        "accessible_episodes": [],
    }
    manifest["sha256"] = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (root / "precedent-manifest.filtered.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (root / "run-config.json").write_text(
        json.dumps(
            {
                "run": {"contract_version": "v3"},
                "provider": {"kind": "ollama"},
                "precedents": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )
    (root / "input-provenance.json").write_text(
        json.dumps(
            {
                "pitch_path": "inputs/pitch.txt",
                "pitch_sha256": "a" * 64,
                "package_manifest_path": "inputs/source-manifest.json",
                "package_manifest_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    usages = {
        ("phase1", "plan"): (2, 1, 3, 0.01),
        ("phase1", "investigation"): (5, 0, 7, 0.02),
        ("phase2", "plan"): (11, 4, 13, 0.03),
        ("phase2", "decision"): (17, 0, 19, 0.04),
    }
    for phase, terminal in (("phase1", "investigation"), ("phase2", "decision")):
        turn = root / phase / "turn-01"
        turn.mkdir(parents=True, exist_ok=True)
        for name in (
            "wiki-searches.json",
            "wiki-reads.json",
            "precedent-searches.json",
            "precedent-reads.json",
            "historical-evidence.json",
        ):
            (turn / name).write_text("[]", encoding="utf-8")
        for step in ("plan", terminal):
            input_tokens, cached, output_tokens, cost = usages[(phase, step)]
            (turn / f"{step}-model-response.json").write_text(
                json.dumps(
                    {
                        "usage": {
                            "input_tokens": input_tokens,
                            "cached_input_tokens": cached,
                            "output_tokens": output_tokens,
                            "cost_usd": cost,
                        },
                        "raw_metadata": {
                            "provider": "ollama",
                            "model": "local-model",
                        },
                    }
                ),
                encoding="utf-8",
            )
    phase1_usage = {
        "input_tokens": 7,
        "cached_input_tokens": 1,
        "output_tokens": 10,
        "cost_usd": 0.03,
    }
    phase2_usage = {
        "input_tokens": 28,
        "cached_input_tokens": 4,
        "output_tokens": 32,
        "cost_usd": 0.07,
    }
    state = {
        "phase1_status": "accepted",
        "phase2_status": "accepted",
        "phase1_iteration": 1,
        "phase2_iteration": 1,
        "investigation_sha256": investigation_digest,
        "investigation": frozen_investigation.model_dump(mode="json"),
        "precedent_manifest_sha256": manifest["sha256"],
        "query_history": [],
        "exact_reads": [],
        "phase2_query_history": [],
        "phase2_exact_reads": [],
        "phase1_precedent_searches": [],
        "phase2_precedent_searches": [],
        "phase1_precedent_reads": [],
        "phase2_precedent_reads": [],
        "phase1_historical_evidence": [],
        "phase2_historical_evidence": [],
        "usage_by_phase": {"phase1": phase1_usage, "phase2": phase2_usage},
        "usage": {
            "input_tokens": 35,
            "cached_input_tokens": 5,
            "output_tokens": 42,
            "cost_usd": 0.1,
        },
    }
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")


def test_v3_precedent_run_verifies_complete_audit(tmp_path: Path) -> None:
    _write_strict_v3_run(tmp_path)

    verify_run_artifacts(tmp_path)


def test_v3_verifier_accepts_phase2_with_audit_warnings(tmp_path: Path) -> None:
    _write_strict_v3_run(tmp_path)
    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["phase2_status"] = "accepted_with_warnings"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    verify_run_artifacts(tmp_path)


def test_v3_verifier_allows_same_exact_evidence_reopened_by_new_query(
    tmp_path: Path,
) -> None:
    _write_strict_v3_run(tmp_path)
    source_sha256 = "c" * 64
    text = "Charles: I invest when the founder has a distinctive customer insight."
    evidence_id = "H-" + sha256(
        f"{source_sha256}\0{0}\0{0}\0{text}".encode("utf-8")
    ).hexdigest()[:20]

    def evidence(query_id: str) -> dict:
        return {
            "citation_mode": "exact",
            "episode_slug": "18-rowvigor",
            "evidence_id": evidence_id,
            "query_id": query_id,
            "source_sha256": source_sha256,
            "text": text,
            "turn_end": 0,
            "turn_start": 0,
        }

    def read(query_id: str) -> dict:
        return {
            "charged": True,
            "episode_slug": "18-rowvigor",
            "evidence": evidence(query_id),
            "query_id": query_id,
            "read_type": "transcript",
            "status": "ok",
            "turn_end": 0,
            "turn_start": 0,
        }

    phase1_read = read("phase1-query")
    phase2_read = read("phase2-query")
    phase1_evidence = evidence("phase1-query")
    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["phase1_precedent_reads"] = [phase1_read]
    state["phase2_precedent_reads"] = [phase2_read]
    state["phase1_historical_evidence"] = [phase1_evidence]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "phase1/turn-01/precedent-reads.json").write_text(
        json.dumps([phase1_read]), encoding="utf-8"
    )
    (tmp_path / "phase2/turn-01/precedent-reads.json").write_text(
        json.dumps([phase2_read]), encoding="utf-8"
    )
    (tmp_path / "phase1/turn-01/historical-evidence.json").write_text(
        json.dumps([phase1_evidence]), encoding="utf-8"
    )

    verify_run_artifacts(tmp_path)


def test_verifier_resolves_v4_nested_transcript_and_decision_evidence(
    tmp_path: Path,
) -> None:
    _write_strict_v3_run(tmp_path)
    source_sha256 = "d" * 64
    text = "Charles: The founder has a distinctive customer insight."
    evidence_id = "H-" + sha256(
        f"{source_sha256}\0{3}\0{3}\0{text}".encode("utf-8")
    ).hexdigest()[:20]
    evidence = {
        "citation_mode": "exact",
        "episode_slug": "18-rowvigor",
        "evidence_id": evidence_id,
        "query_id": "phase2-t01-read-01",
        "source_sha256": source_sha256,
        "text": text,
        "turn_end": 3,
        "turn_start": 3,
    }
    read = {
        "episode_slug": "18-rowvigor",
        "query_id": "phase2-t01-precedent-01",
        "transcript": {"episode_slug": "18-rowvigor", "evidence": evidence},
        "decision": {"episode_slug": "18-rowvigor", "evidence": []},
    }
    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["phase2_precedent_reads"] = [read]
    state["phase2_historical_evidence"] = [evidence]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "phase2/turn-01/precedent-reads.json").write_text(
        json.dumps([read]), encoding="utf-8"
    )
    (tmp_path / "phase2/turn-01/historical-evidence.json").write_text(
        json.dumps([evidence]), encoding="utf-8"
    )

    verify_run_artifacts(tmp_path)


def test_v3_precedent_run_requires_filtered_manifest(tmp_path: Path) -> None:
    _write_strict_v3_run(tmp_path)
    (tmp_path / "precedent-manifest.filtered.json").unlink()

    with pytest.raises(ValueError, match="filtered precedent manifest"):
        verify_run_artifacts(tmp_path)


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("run-config.json", "run config"),
        ("input-provenance.json", "package provenance"),
    ],
)
def test_v3_precedent_run_requires_root_audit_artifacts(
    tmp_path: Path, relative_path: str, message: str
) -> None:
    _write_strict_v3_run(tmp_path)
    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["investigation"]["schema_version"] = "investigation-v3"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / relative_path).unlink()

    with pytest.raises(ValueError, match=message):
        verify_run_artifacts(tmp_path)


@pytest.mark.parametrize(
    "artifact_name",
    [
        "wiki-searches.json",
        "wiki-reads.json",
        "precedent-searches.json",
        "precedent-reads.json",
        "historical-evidence.json",
    ],
)
def test_v3_precedent_run_requires_each_per_turn_audit_type(
    tmp_path: Path, artifact_name: str
) -> None:
    _write_strict_v3_run(tmp_path)
    (tmp_path / "phase1/turn-01" / artifact_name).unlink()

    with pytest.raises(ValueError, match="required audit artifact"):
        verify_run_artifacts(tmp_path)


@pytest.mark.parametrize(
    "relative_path",
    [
        "phase1/turn-01/plan-model-response.json",
        "phase1/turn-01/investigation-model-response.json",
        "phase2/turn-01/plan-model-response.json",
        "phase2/turn-01/decision-model-response.json",
    ],
)
def test_v3_precedent_run_requires_exact_model_response_set(
    tmp_path: Path, relative_path: str
) -> None:
    _write_strict_v3_run(tmp_path)
    (tmp_path / relative_path).unlink()

    with pytest.raises(ValueError, match="model-response artifact set"):
        verify_run_artifacts(tmp_path)


def test_v3_precedent_run_rejects_extra_model_response(tmp_path: Path) -> None:
    _write_strict_v3_run(tmp_path)
    shutil.copyfile(
        tmp_path / "phase1/turn-01/investigation-model-response.json",
        tmp_path / "phase1/turn-01/ambiguous-model-response.json",
    )

    with pytest.raises(ValueError, match="model-response artifact set"):
        verify_run_artifacts(tmp_path)


@pytest.mark.parametrize("metadata_key", ["provider", "model"])
def test_v3_precedent_run_requires_provider_and_model_metadata(
    tmp_path: Path, metadata_key: str
) -> None:
    _write_strict_v3_run(tmp_path)
    path = tmp_path / "phase1/turn-01/plan-model-response.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["raw_metadata"][metadata_key]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="provider/model metadata"):
        verify_run_artifacts(tmp_path)


@pytest.mark.parametrize("field", ["input_tokens", "cost_usd"])
def test_v3_precedent_run_rejects_modified_response_usage(
    tmp_path: Path, field: str
) -> None:
    _write_strict_v3_run(tmp_path)
    path = tmp_path / "phase2/turn-01/decision-model-response.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["usage"][field] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact usage"):
        verify_run_artifacts(tmp_path)


@pytest.mark.parametrize(
    "state_path",
    [
        ("usage_by_phase", "phase1", "output_tokens"),
        ("usage", "cost_usd"),
    ],
)
def test_v3_precedent_run_rejects_modified_state_usage(
    tmp_path: Path, state_path: tuple[str, ...]
) -> None:
    _write_strict_v3_run(tmp_path)
    path = tmp_path / "state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    target = state
    for key in state_path[:-1]:
        target = target[key]
    target[state_path[-1]] += 1
    path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact usage"):
        verify_run_artifacts(tmp_path)


@pytest.mark.parametrize(
    ("artifact_name", "row"),
    [
        ("wiki-searches.json", {"query": "tampered", "hits": []}),
        ("wiki-reads.json", {"chunk_id": "W-tampered"}),
        (
            "precedent-searches.json",
            {"query": "tampered", "hits": []},
        ),
        (
            "precedent-reads.json",
            {"episode_slug": "18-rowvigor", "status": "error"},
        ),
        (
            "historical-evidence.json",
            {"evidence_id": "H-tampered"},
        ),
    ],
)
def test_v3_precedent_run_reconstructs_every_phase_history_unconditionally(
    tmp_path: Path, artifact_name: str, row: dict
) -> None:
    _write_strict_v3_run(tmp_path)
    path = tmp_path / "phase1/turn-01" / artifact_name
    path.write_text(json.dumps([row]), encoding="utf-8")

    with pytest.raises(ValueError, match="artifacts do not match recorded state"):
        verify_run_artifacts(tmp_path)


def investigation_v2_with_constraint(
    *,
    evidence_status: str,
    constraint_kind: str,
    constraint_severity: str,
) -> InvestigationV2:
    candidate = investigation_v2()
    constraint = RationaleV2(
        rationale_id="R2",
        label="market_size_assessment",
        direction="negative" if evidence_status == "affirmative_adverse" else "neutral",
        salience="primary",
        confidence=0.7,
        pitch_evidence=["market scale remains unproven"],
        wiki_evidence_ids=["W-12345678901234567890"],
        interpretation="The concern may limit the investment.",
        evidence_status=evidence_status,
        constraint_kind=constraint_kind,
        constraint_severity=constraint_severity,
        severity_basis="The frozen investigation records this exact severity.",
    )
    return candidate.model_copy(
        update={"rationales": [candidate.rationales[0], constraint]}
    )


def decision_v2_with_risk(risk_type: str, *, fatal: bool = False) -> DecisionV2:
    candidate = dual_decision_v2()
    ledger = candidate.risk_ledger[0].model_copy(update={"risk_type": risk_type})
    any_check = candidate.any_check.model_copy(
        update={"fatal_constraint_present": fatal}
    )
    return candidate.model_copy(update={"risk_ledger": [ledger], "any_check": any_check})


def test_v2_decision_schema_binds_the_frozen_investigation() -> None:
    schema = decision_v2_json_schema(
        episode_slug="135-thoras-ai-the-twin-effect",
        investigation_sha256="a" * 64,
        rationale_ids={"R1", "R2"},
        check_tiers=CHECK_TIERS,
    )

    assert schema["properties"]["schema_version"]["const"] == "decision-v2"
    assert schema["properties"]["investigation_sha256"]["const"] == "a" * 64


def test_v2_decision_cannot_promote_routine_constraint_to_fatal() -> None:
    investigation = investigation_v2_with_constraint(
        evidence_status="unresolved",
        constraint_kind="portfolio_overlap",
        constraint_severity="routine",
    )
    candidate = decision_v2_with_risk("fatal_constraint", fatal=True)

    with pytest.raises(
        ValueError, match="fatal risk requires affirmative adverse fatal provenance"
    ):
        validate_decision_v2(candidate, investigation, "a" * 64, CHECK_TIERS)


def test_v2_decision_cannot_promote_unresolved_evidence_to_adverse() -> None:
    investigation = investigation_v2_with_constraint(
        evidence_status="unresolved",
        constraint_kind="portfolio_overlap",
        constraint_severity="material",
    )
    candidate = decision_v2_with_risk("affirmative_adverse")

    with pytest.raises(
        ValueError,
        match="affirmative adverse risk requires affirmative adverse evidence",
    ):
        validate_decision_v2(candidate, investigation, "a" * 64, CHECK_TIERS)


def test_v2_decision_cannot_treat_total_round_size_as_fatal() -> None:
    investigation = investigation_v2_with_constraint(
        evidence_status="affirmative_adverse",
        constraint_kind="total_round_size",
        constraint_severity="material",
    )
    candidate = decision_v2_with_risk("fatal_constraint", fatal=True)

    with pytest.raises(ValueError, match="total round size cannot become fatal"):
        validate_decision_v2(candidate, investigation, "a" * 64, CHECK_TIERS)


def test_v2_decision_allows_confirmed_fatal_portfolio_conflict() -> None:
    investigation = investigation_v2_with_constraint(
        evidence_status="affirmative_adverse",
        constraint_kind="portfolio_overlap",
        constraint_severity="fatal",
    )
    candidate = decision_v2_with_risk("fatal_constraint", fatal=True)

    validate_decision_v2(candidate, investigation, "a" * 64, CHECK_TIERS)


def test_decision_must_bind_investigation_and_known_rationales() -> None:
    candidate = dual_decision()
    validate_decision(
        candidate,
        "135-thoras-ai-the-twin-effect",
        "a" * 64,
        {"R1", "R2"},
        CHECK_TIERS,
    )
    with pytest.raises(ValueError, match="unknown rationale"):
        validate_decision(
            candidate,
            "135-thoras-ai-the-twin-effect",
            "a" * 64,
            {"R1"},
            CHECK_TIERS,
        )


def test_decision_requires_exact_rationale_coverage_and_likelihood_continuity() -> None:
    candidate = dual_decision()
    validate_decision(candidate, candidate.episode_slug, "a" * 64, {"R1", "R2"}, CHECK_TIERS)

    missing = candidate.model_copy(update={"rationale_assessments": []})
    with pytest.raises(ValueError, match="exactly once"):
        validate_decision(missing, candidate.episode_slug, "a" * 64, {"R1", "R2"}, CHECK_TIERS)

    broken = candidate.model_copy(
        update={
            "deliberation_steps": [
                *candidate.deliberation_steps[:2],
                candidate.deliberation_steps[2].model_copy(
                    update={"likelihood_before": 0.2, "effect": "raises"}
                ),
                *candidate.deliberation_steps[3:],
            ]
        }
    )
    with pytest.raises(ValueError, match="continuity"):
        validate_decision(broken, candidate.episode_slug, "a" * 64, {"R1", "R2"}, CHECK_TIERS)


def test_each_endpoint_track_must_finish_with_consistency() -> None:
    candidate = dual_decision()
    steps = list(candidate.deliberation_steps)
    steps[0] = steps[0].model_copy(update={"stage": "consistency"})
    steps[2] = steps[2].model_copy(update={"stage": "decision"})
    candidate = candidate.model_copy(update={"deliberation_steps": steps})

    with pytest.raises(ValueError, match="finish with a consistency step"):
        validate_decision(candidate, candidate.episode_slug, "a" * 64, {"R1", "R2"}, CHECK_TIERS)


@pytest.mark.parametrize("endpoint", ["any_check", "standard_check"])
def test_each_endpoint_track_requires_an_opposing_case(endpoint: str) -> None:
    candidate = dual_decision()
    steps = [
        step.model_copy(update={"stage": "assessment"})
        if step.endpoint == endpoint and step.stage == "opposing_case"
        else step
        for step in candidate.deliberation_steps
    ]
    candidate = candidate.model_copy(update={"deliberation_steps": steps})

    with pytest.raises(ValueError, match=f"{endpoint} deliberation requires an opposing-case step"):
        validate_decision(candidate, candidate.episode_slug, "a" * 64, {"R1", "R2"}, CHECK_TIERS)


@pytest.mark.parametrize("market_status", ["negative", "unresolved"])
def test_standard_check_in_requires_a_positive_market_gate(market_status: str) -> None:
    candidate = dual_decision()
    standard = candidate.standard_check.model_copy(
        update={
            "decision": "In",
            "likelihood": 0.55,
            "market_gate": candidate.standard_check.market_gate.model_copy(
                update={"status": market_status}
            ),
        }
    )
    steps = list(candidate.deliberation_steps)
    steps[-1] = steps[-1].model_copy(
        update={"likelihood_after": 0.55, "effect": "raises"}
    )
    candidate = candidate.model_copy(
        update={
            "standard_check": standard,
            "recommended_check_tier": "standard_initial",
            "check_tier": "standard_initial",
            "deliberation_steps": steps,
        }
    )

    with pytest.raises(ValueError, match="positive market gate"):
        validate_decision(candidate, candidate.episode_slug, "a" * 64, {"R1", "R2"}, CHECK_TIERS)


def test_endpoint_controlling_rationales_must_appear_in_their_own_tracks() -> None:
    candidate = dual_decision()
    any_steps = [
        step.model_copy(update={"rationale_ids": ["R1"]})
        if step.endpoint == "any_check"
        else step
        for step in candidate.deliberation_steps
    ]
    misplaced_any = candidate.model_copy(update={"deliberation_steps": any_steps})
    controlling_ledger = candidate.risk_ledger[0].model_copy(
        update={"controlling_for_any_check": True}
    )
    misplaced_any = misplaced_any.model_copy(update={"risk_ledger": [controlling_ledger]})
    with pytest.raises(ValueError, match="any-check controlling risks"):
        validate_decision(
            misplaced_any, candidate.episode_slug, "a" * 64, {"R1", "R2"}, CHECK_TIERS
        )

    standard_steps = [
        step.model_copy(update={"rationale_ids": ["R1"]})
        if step.endpoint == "standard_check"
        else step
        for step in candidate.deliberation_steps
    ]
    misplaced_standard = candidate.model_copy(update={"deliberation_steps": standard_steps})
    with pytest.raises(ValueError, match="market-gate controlling rationales"):
        validate_decision(
            misplaced_standard, candidate.episode_slug, "a" * 64, {"R1", "R2"}, CHECK_TIERS
        )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"standard_check": {"decision": "In"}}, "standard-check In"),
        ({"standard_check": {"likelihood": 0.8}}, "cannot exceed"),
        ({"recommended_check_tier": "standard_initial"}, "recommended check tier"),
        ({"investment_likelihood": 0.6}, "compatibility"),
    ],
)
def test_dual_check_rejects_cross_endpoint_inconsistency(update: dict, message: str) -> None:
    candidate = dual_decision()
    if "standard_check" in update:
        candidate = candidate.model_copy(
            update={
                "standard_check": candidate.standard_check.model_copy(
                    update=update["standard_check"]
                )
            }
        )
    else:
        candidate = candidate.model_copy(update=update)
    with pytest.raises(ValueError, match=message):
        validate_decision(candidate, candidate.episode_slug, "a" * 64, {"R1", "R2"}, CHECK_TIERS)


def test_dual_check_notes_fatal_constraint_with_any_check_in_as_quality_finding() -> None:
    candidate = dual_decision().model_copy(
        update={
            "any_check": dual_decision().any_check.model_copy(
                update={"fatal_constraint_present": True}
            ),
            "risk_ledger": [
                dual_decision().risk_ledger[0].model_copy(
                    update={"risk_type": "fatal_constraint", "controlling_for_any_check": True}
                )
            ],
        }
    )
    validate_decision(candidate, candidate.episode_slug, "a" * 64, {"R1", "R2"}, CHECK_TIERS)
    assert decision_quality_findings(candidate) == [
        "ANY_CHECK_IN_WITH_MATERIAL_CONSTRAINT"
    ]


def test_risk_ledger_coverage_mismatch_is_a_quality_finding() -> None:
    candidate = dual_decision().model_copy(update={"risk_ledger": []})
    validate_decision(
        candidate, candidate.episode_slug, "a" * 64, {"R1", "R2"}, CHECK_TIERS
    )
    assert decision_quality_findings(candidate) == [
        "RISK_LEDGER_COVERAGE_MISMATCH"
    ]


def test_size_limiting_condition_can_remain_a_clean_any_check_in() -> None:
    candidate = dual_decision()
    validate_decision(candidate, candidate.episode_slug, "a" * 64, {"R1", "R2"}, CHECK_TIERS)
    assert decision_quality_findings(candidate) == []


def test_candidates_must_bind_to_the_exact_episode_slug() -> None:
    candidate = investigation().model_copy(update={"episode_slug": "thoras-ai"})
    with pytest.raises(ValueError, match="episode slug"):
        validate_investigation(
            candidate,
            episode_slug="135-thoras-ai-the-twin-effect",
            pitch="The founders have five paid pilots.",
            taxonomy_labels={"founder_execution"},
            exact_wiki_ids={"W-12345678901234567890"},
        )
