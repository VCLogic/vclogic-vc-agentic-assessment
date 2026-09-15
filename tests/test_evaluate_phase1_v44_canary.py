from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from vc_clone_graph.phase1_evaluation import (
    Phase1Case,
    PredictedRationale,
    ReferenceRationale,
    TaxonomyLabel,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_evaluator_module():
    path = ROOT / "scripts/evaluate_phase1_v44_canary.py"
    spec = importlib.util.spec_from_file_location("evaluate_phase1_v44_canary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _predicted(label: str, *, confidence: float = 0.8) -> PredictedRationale:
    return PredictedRationale(
        label=label,
        direction="positive",
        salience="primary",
        confidence=confidence,
        justification=f"Evidence supports {label}.",
    )


def _reference(label: str) -> ReferenceRationale:
    return ReferenceRationale(
        label=label,
        direction="positive",
        salience="primary",
        confidence=0.9,
        activation="evaluated",
        utterance_type="decision_reason",
        decision_link="explicit",
        evidence=(f"Reference evidence for {label}.",),
    )


def _canonical_case(position: int, labels: tuple[str, ...]) -> Phase1Case:
    vc_slug = f"vc-{position}"
    return Phase1Case(
        vc_slug=vc_slug,
        vc_name=f"VC {position}",
        episode_slug=f"{position}-case",
        actual_decision="In" if position % 2 else "Out",
        source_tier="previously_validated",
        source_format="transcript-observed-rationales-v1",
        predicted=tuple(_predicted(label) for label in labels),
        reference=(),
        artifact_path=None,
        reference_path=None,
    )


def _rationale(label: str, rationale_id: str, disposition: str) -> dict[str, object]:
    return {
        "rationale_id": rationale_id,
        "taxonomy_label": label,
        "disposition": disposition,
        "direction": "positive",
        "salience": "primary",
        "confidence": 0.8,
        "justification": f"Evidence supports {label}.",
        "claim_ids": ["C1"],
        "question_ids": [],
        "pitch_evidence_ids": ["P-001"],
        "wiki_evidence_ids": ["W-evidence"],
        "historical_evidence_ids": [],
        "portfolio_disclosure_ids": [],
    }


def _write_run(
    runs: Path,
    case: Phase1Case,
    *,
    broad: tuple[str, ...],
    core: tuple[str, ...],
    neighborhood: tuple[str, ...],
    status: str = "accepted",
    iteration: int = 1,
    usage: tuple[int, int, int, float] = (100, 20, 10, 0.25),
    first_adjudication: tuple[str, ...] = (),
    repaired_first_adjudication: tuple[str, ...] | None = None,
) -> Path:
    root = runs / case.vc_slug / case.episode_slug
    phase1 = root / "phase1"
    phase1.mkdir(parents=True, exist_ok=True)
    broad_rows = [
        _rationale(label, f"R{index}", "core" if label in core else "candidate")
        for index, label in enumerate(broad, start=1)
    ]
    core_rows = [row for row in broad_rows if row["taxonomy_label"] in core]
    investigation = {
        "schema_version": "investigation-v4.4",
        "episode_slug": case.episode_slug,
        "material_claims": [{"claim_id": "C1"}],
        "adverse_claims": [],
        "claim_coverage": [{"pitch_evidence_id": "P-001"}],
        "candidate_rationales": broad_rows,
        "core_rationales": core_rows,
        "rationales": core_rows,
        "unanswered_questions": [{"question_id": "Q1"}],
        "question_only_dispositions": [{"taxonomy_label": "B"}],
        "taxonomy_neighborhood_manifest": {
            "ordered_labels": list(neighborhood),
            "claim_neighborhoods": [
                {
                    "target_id": "C1",
                    "candidates": [
                        {"taxonomy_label": label} for label in neighborhood
                    ],
                }
            ],
        },
        "investigation_status": status,
        "validator_findings": [] if status == "accepted" else ["UNRESOLVED:C1"],
    }
    (phase1 / "investigation.json").write_text(
        json.dumps(investigation), encoding="utf-8"
    )
    call_records = [
        {
            "relative_path": "phase1/claim-extraction/call-01.json",
            "payload": {"phase": "phase1_claim_extraction", "parsed": {}},
        }
    ]
    adjudication_events = []
    for turn in range(1, iteration + 1):
        primary_labels = first_adjudication if turn == 1 else broad
        primary_payload: dict[str, object] = {
            "dispositions": [
                {"taxonomy_label": label, "disposition": "candidate"}
                for label in primary_labels
            ]
        }
        if turn == 1 and repaired_first_adjudication is not None:
            primary_payload = {"invalid": list(primary_labels)}
        parsed = {
            "dispositions": [
                {
                    "taxonomy_label": label,
                    "disposition": "candidate",
                }
                for label in primary_labels
            ]
        }
        call_records.append(
            {
                "relative_path": f"phase1/adjudication/turn-{turn:02d}/call-01.json",
                "payload": {
                    "phase": "phase1_adjudication",
                    "parsed": primary_payload,
                },
            }
        )
        if turn == 1 and repaired_first_adjudication is not None:
            parsed = {
                "dispositions": [
                    {"taxonomy_label": label, "disposition": "candidate"}
                    for label in repaired_first_adjudication
                ]
            }
            call_records.append(
                {
                    "relative_path": "phase1/adjudication/turn-01/call-02.json",
                    "payload": {
                        "phase": "phase1_adjudication_repair",
                        "parsed": parsed,
                    },
                }
            )
        adjudication_events.append(
            {
                "sequence": turn,
                "kind": "phase1_adjudication",
                "status": "provisional" if turn < iteration else "valid",
                "normalized_nonactivating_dispositions": 0,
                "exact_duplicate_dispositions_removed": 0,
                "removed_disposition_positions": [],
                "cross_target_evidence_reuse": [],
                "constraint_mapping_revisions": [],
            }
        )
    input_tokens, cached_tokens, output_tokens, cost = usage
    state = {
        "contract_version": "v4.4",
        "episode_slug": case.episode_slug,
        "phase1_iteration": iteration,
        "phase1_status": status,
        "phase2_status": "not_run",
        "call_records": call_records,
        "events": adjudication_events,
        "usage": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        },
    }
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return root


@dataclass
class _Fixture:
    cases: list[Phase1Case]
    taxonomy: dict[str, TaxonomyLabel]
    selected: dict[tuple[str, str], dict[str, str]]
    runs: Path
    runner_status: dict[str, object]


def _fixture(tmp_path: Path) -> _Fixture:
    taxonomy = {
        "A": TaxonomyLabel("A", "Definition A", "team"),
        "B": TaxonomyLabel("B", "Definition B", "market"),
        "C": TaxonomyLabel("C", "Definition C", "market"),
    }
    raw_cases = [
        (1, ("A",), ("A", "B"), ("A", "C"), ("A",), ("A", "B")),
        (2, ("B", "C"), ("B",), ("C",), (), ("C",)),
        (3, ("C",), ("B",), ("B", "C"), (), ("B", "C")),
    ]
    cases: list[Phase1Case] = []
    selected: dict[tuple[str, str], dict[str, str]] = {}
    runs = tmp_path / "runs"
    for position, canonical, reference, broad, core, neighborhood in raw_cases:
        base = _canonical_case(position, canonical)
        case = Phase1Case(
            **{
                **base.__dict__,
                "reference": tuple(_reference(label) for label in reference),
            }
        )
        cases.append(case)
        selected[(case.vc_slug, case.episode_slug)] = {
            "vc_slug": case.vc_slug,
            "episode_slug": case.episode_slug,
            "actual_decision": case.actual_decision,
            "role": ("coverage", "precision", "control")[position - 1],
        }
        _write_run(
            runs,
            case,
            broad=broad,
            core=core,
            neighborhood=neighborhood,
            status="provisional" if position == 2 else "accepted",
            iteration=2 if position == 1 else 1,
            first_adjudication=("A",) if position == 1 else broad,
        )
    status_rows = []
    cumulative = 0.0
    for position, case in enumerate(cases, start=1):
        state = json.loads(
            (runs / case.vc_slug / case.episode_slug / "state.json").read_text()
        )
        cumulative += float(state["usage"]["cost_usd"])
        status_rows.append(
            {
                "position": position,
                "vc_slug": case.vc_slug,
                "episode_slug": case.episode_slug,
                "actual_decision": case.actual_decision,
                "role": selected[(case.vc_slug, case.episode_slug)]["role"],
                "status": "completed",
                "phase1_status": state["phase1_status"],
                "calls": len(state["call_records"]),
                "input_tokens": state["usage"]["input_tokens"],
                "output_tokens": state["usage"]["output_tokens"],
                "cost_usd": state["usage"]["cost_usd"],
                "cumulative_cost_usd": cumulative,
                "state_path": str(
                    runs / case.vc_slug / case.episode_slug / "state.json"
                ),
                "error": "",
            }
        )
    runner_status = {
        "schema": "phase1-v44-canary-status-v1",
        "cumulative_cost_usd": cumulative,
        "cases": status_rows,
    }
    return _Fixture(cases, taxonomy, selected, runs, runner_status)


def _verified(_root: Path) -> None:
    return None


def _make_real_v44_run(tmp_path: Path) -> Path:
    from tests.test_artifacts import (
        _MetadataFakeProvider,
        _adjudication_v44_with_candidate,
    )
    from tests.test_workflow_v44 import (
        claim_map,
        make_index,
        make_workflow,
    )

    seed = tmp_path / "real-seed"
    seed.mkdir()
    evidence_id = make_index(seed).chunks[0].chunk_id
    case = tmp_path / "real-case"
    case.mkdir()
    responses = [claim_map(), _adjudication_v44_with_candidate(evidence_id)]
    workflow, _ = make_workflow(case, responses)
    workflow.phase1_provider = _MetadataFakeProvider(responses)
    workflow = type(workflow)(
        workflow.settings,
        workflow.index,
        phase1_provider=workflow.phase1_provider,
        checkpointer=workflow.graph.checkpointer,
        v44_settings=workflow.v44_settings,
    )
    workflow.invoke("artifact-v44-snapshot-integration")
    return case / "run"


def _materialize_exact_call_ledger(root: Path, state: dict[str, object]) -> None:
    records = state["call_records"]
    usage = state["usage"]
    count = len(records)
    running_cost = 0.0
    for index, record in enumerate(records, start=1):
        payload = record["payload"]
        call_usage: dict[str, int | float] = {}
        for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
            quotient, remainder = divmod(usage[key], count)
            call_usage[key] = quotient + (1 if index <= remainder else 0)
        if index < count:
            call_cost = usage["cost_usd"] / count
            running_cost += call_cost
        else:
            call_cost = usage["cost_usd"] - running_cost
        call_usage["cost_usd"] = call_cost
        payload.update(
            {
                "prompt": "public pitch only",
                "schema": {},
                "raw": "{}",
                "provider_metadata": {},
                "usage": call_usage,
                "cost_usd": call_cost,
                "elapsed_seconds": 0.1,
                "max_output_tokens": 100,
                "reasoning_effort": "high",
            }
        )
        raw = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        call_path = root / record["relative_path"]
        call_path.parent.mkdir(parents=True, exist_ok=True)
        call_path.write_bytes(raw)
        record.update(
            {
                "sha256": sha256(raw).hexdigest(),
                "wal_id": f"{index:064x}",
                "request_sha256": f"{index + 100:064x}",
            }
        )
    state["pending_call"] = None
    state["pending_response"] = None
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")


def _make_exact_failed_state(
    fixture: _Fixture,
    *,
    note: str | None = None,
) -> tuple[Phase1Case, Path, dict[str, object]]:
    case = fixture.cases[-1]
    root = fixture.runs / case.vc_slug / case.episode_slug
    (root / "phase1/investigation.json").unlink(missing_ok=True)
    calls = (
        ("phase1/claim-extraction/call-01.json", "phase1_claim_extraction"),
        ("phase1/adjudication/turn-01/call-01.json", "phase1_adjudication"),
    )
    records = []
    for index, (relative, phase) in enumerate(calls, start=1):
        payload = {
            "phase": phase,
            "prompt": "public pitch only",
            "schema": {},
            "raw": "{}",
            "parsed": {},
            "provider_metadata": {},
            "usage": {
                "input_tokens": 10,
                "cached_input_tokens": 2,
                "output_tokens": 5,
                "cost_usd": 0.05,
            },
            "cost_usd": 0.05,
            "elapsed_seconds": 0.1,
            "max_output_tokens": 100,
            "reasoning_effort": "high",
        }
        raw = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        call_path = root / relative
        call_path.parent.mkdir(parents=True, exist_ok=True)
        call_path.write_bytes(raw)
        records.append(
            {
                "relative_path": relative,
                "sha256": sha256(raw).hexdigest(),
                "payload": payload,
                "wal_id": f"{index:064x}",
                "request_sha256": f"{index + 10:064x}",
            }
        )
    state_path = root / "state.json"
    state = json.loads(state_path.read_text())
    state.update(
        {
            "phase1_status": "failed",
            "phase2_status": "not_run",
            "phase1_iteration": 1,
            "call_records": records,
            "pending_call": None,
            "pending_response": None,
            "usage": {
                "input_tokens": 20,
                "cached_input_tokens": 4,
                "output_tokens": 10,
                "cost_usd": 0.1,
            },
        }
    )
    if note is not None:
        state["note"] = note
    state_path.write_text(json.dumps(state), encoding="utf-8")
    row = fixture.runner_status["cases"][-1]
    row.update(
        {
            "status": "failed_with_state",
            "phase1_status": "failed",
            "calls": 2,
            "input_tokens": 20,
            "output_tokens": 10,
            "cost_usd": 0.1,
            "cumulative_cost_usd": 0.6,
            "state_path": str(state_path),
            "error": "structured adjudication failed",
        }
    )
    fixture.runner_status["cumulative_cost_usd"] = 0.6
    return case, root, state


def test_v44_canary_evaluator_exposes_frozen_inputs_and_output() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/evaluate_phase1_v44_canary.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    for option in (
        "--registry",
        "--references",
        "--taxonomy",
        "--manifest",
        "--runs",
        "--status",
        "--output",
    ):
        assert option in result.stdout


def test_canary_manifest_is_the_exact_frozen_v43_development_cohort(
    tmp_path: Path,
) -> None:
    module = _load_evaluator_module()
    source = ROOT / "reports/evaluation/phase1-v43-diagnostic-2026-08-17/canary-manifest.json"
    frozen = module.load_canary_manifest(source)
    assert len(frozen) == 18
    assert len({key[0] for key in frozen}) == 6
    assert list(row["role"] for row in frozen.values())[:3] == [
        "coverage",
        "precision",
        "control",
    ]

    original = json.loads(source.read_text())
    mutations = {
        "schema": lambda payload: payload.__setitem__("schema", "wrong"),
        "scientific_status": lambda payload: payload.__setitem__(
            "scientific_status", "untouched_holdout"
        ),
        "selected_intervention": lambda payload: payload.__setitem__(
            "selected_intervention", "another_method"
        ),
        "actual_decision": lambda payload: payload["cases"][0].__setitem__(
            "actual_decision", "Maybe"
        ),
        "role": lambda payload: payload["cases"][0].__setitem__("role", "other"),
        "composition": lambda payload: payload["cases"].pop(),
    }
    for name, mutate in mutations.items():
        payload = json.loads(json.dumps(original))
        mutate(payload)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="canary manifest"):
            module.load_canary_manifest(path)


def test_case_input_bindings_cover_exact_prediction_sidecar_and_reference(
    tmp_path: Path,
) -> None:
    module = _load_evaluator_module()
    canonical = tmp_path / "canonical" / "phase1/investigation.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text('{"schema_version":"investigation-v4"}\n')
    sidecar = canonical.with_suffix(".sha256")
    sidecar.write_text(sha256(canonical.read_bytes()).hexdigest() + "\n")
    reference = tmp_path / "references/vc-1/1-case.json"
    reference.parent.mkdir(parents=True)
    reference.write_text(
        json.dumps({"vc_slug": "vc-1", "episode_slug": "1-case"}) + "\n"
    )
    base = _canonical_case(1, ("A",))
    case = Phase1Case(
        **{
            **base.__dict__,
            "artifact_path": canonical,
            "reference_path": reference,
        }
    )

    first = module.case_input_bindings([case], project_root=tmp_path)
    assert set(first[0]) == {
        "vc_slug",
        "episode_slug",
        "canonical_investigation",
        "canonical_sidecar",
        "automated_reference",
    }
    assert first[0]["canonical_investigation"]["sha256"] == sha256(
        canonical.read_bytes()
    ).hexdigest()

    canonical.write_text('{"schema_version":"investigation-v4.1"}\n')
    with pytest.raises(ValueError, match="canonical investigation hash mismatch"):
        module.case_input_bindings([case], project_root=tmp_path)

    canonical.write_text('{"schema_version":"investigation-v4"}\n')
    reference.write_text(
        json.dumps(
            {"vc_slug": "vc-1", "episode_slug": "1-case", "changed": True}
        )
        + "\n"
    )
    second = module.case_input_bindings([case], project_root=tmp_path)
    assert (
        first[0]["automated_reference"]["sha256"]
        != second[0]["automated_reference"]["sha256"]
    )


def test_evaluator_separates_broad_core_and_retrieval_errors(tmp_path: Path) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)

    report = module.evaluate_verified_cases(
        canonical_cases=fixture.cases,
        taxonomy=fixture.taxonomy,
        selected=fixture.selected,
        runner_status=fixture.runner_status,
        runs_root=fixture.runs,
        verifier=_verified,
    )

    metrics = report["metrics"]
    assert metrics["taxonomy_opportunity"] == {
        "reference_total": 4,
        "reference_in_neighborhood": 3,
        "reference_absent_from_neighborhood": 1,
        "neighborhood_recall": pytest.approx(0.75),
    }
    assert metrics["adjudication"]["available_but_not_candidate"] == 1
    assert metrics["broad"]["micro_recall"] > metrics["core"]["micro_recall"]
    assert metrics["core"]["micro_precision"] > metrics["broad"]["micro_precision"]
    assert {row["error_stage"] for row in report["taxonomy_rows"]} == {
        "recovered_core",
        "retrieval_omission",
        "adjudication_omission",
        "core_filtering",
    }
    assert metrics["scientific_status"] == "development_canary_not_untouched_holdout"


def test_evaluator_reports_paired_changes_process_and_exact_usage(tmp_path: Path) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    first_case = fixture.cases[0]
    first_state_path = (
        fixture.runs / first_case.vc_slug / first_case.episode_slug / "state.json"
    )
    first_state = json.loads(first_state_path.read_text())
    final_event = first_state["events"][-1]
    final_event["exact_duplicate_dispositions_removed"] = 1
    final_event["removed_disposition_positions"] = [2]
    final_event["cross_target_evidence_reuse"] = [
        {
            "disposition_position": 0,
            "taxonomy_label": "A",
            "target_ids": ["C1"],
            "evidence_kind": "historical",
            "evidence_id": "H-1",
        }
    ]
    final_event["constraint_mapping_revisions"] = [
        {
            "constraint_id": "C1",
            "original_mapped_ids": ["R1"],
            "recomputed_mapped_ids": ["R2"],
        }
    ]
    first_state_path.write_text(json.dumps(first_state))

    report = module.evaluate_verified_cases(
        canonical_cases=fixture.cases,
        taxonomy=fixture.taxonomy,
        selected=fixture.selected,
        runner_status=fixture.runner_status,
        runs_root=fixture.runs,
        verifier=_verified,
    )

    metrics = report["metrics"]
    assert metrics["usage"] == {
        "calls": 7,
        "input_tokens": 300,
        "cached_input_tokens": 60,
        "output_tokens": 30,
        "total_cost_usd": pytest.approx(0.75),
        "average_cost_usd": pytest.approx(0.25),
    }
    assert metrics["process"] == {
        "accepted": 2,
        "provisional": 1,
        "failed": 0,
        "not_started": 0,
        "revisits_triggered": 1,
        "question_only_dispositions": 3,
        "material_claims": 3,
        "adverse_claims": 0,
        "claim_coverage_rows": 3,
        "revisit_candidate_labels_added": 1,
        "revisit_candidate_labels_removed": 0,
        "activated_instances": 5,
        "unique_activated_labels": 5,
        "exact_duplicate_dispositions_removed": 1,
        "cross_target_evidence_reuse": 1,
        "constraint_mapping_revisions": 1,
        "unresolved_constraint_mappings": 0,
    }
    assert metrics["paired_label_changes"]["canonical_to_broad"]["discordant"] > 0
    assert metrics["paired_label_changes"]["broad_to_core"]["worsened"] >= 1
    assert {row["role"] for row in report["paired_rows"]} == {
        "coverage",
        "precision",
        "control",
    }
    assert set(metrics["by_role"]["broad"]) == {"coverage", "precision", "control"}
    first = report["process_rows"][0]
    assert first["activated_instance_count"] == 2
    assert first["unique_activated_label_count"] == 2
    assert first["exact_duplicate_dispositions_removed"] == 1
    assert first["cross_target_evidence_reuse_count"] == 1
    assert first["constraint_mapping_revision_count"] == 1
    assert first["unresolved_constraint_mapping_count"] == 0


def test_revisit_diagnostics_use_valid_repair_not_rejected_primary(
    tmp_path: Path,
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    case = fixture.cases[0]
    _write_run(
        fixture.runs,
        case,
        broad=("A", "C"),
        core=("A",),
        neighborhood=("A", "B", "C"),
        iteration=2,
        first_adjudication=("A", "B"),
        repaired_first_adjudication=("A",),
    )
    state = json.loads(
        (fixture.runs / case.vc_slug / case.episode_slug / "state.json").read_text()
    )
    fixture.runner_status["cases"][0]["calls"] = len(state["call_records"])

    report = module.evaluate_verified_cases(
        canonical_cases=fixture.cases,
        taxonomy=fixture.taxonomy,
        selected=fixture.selected,
        runner_status=fixture.runner_status,
        runs_root=fixture.runs,
        verifier=_verified,
    )

    first = report["process_rows"][0]
    assert first["revisit_candidate_labels_added"] == 1
    assert first["revisit_candidate_labels_removed"] == 0

    state["events"] = []
    (fixture.runs / case.vc_slug / case.episode_slug / "state.json").write_text(
        json.dumps(state)
    )
    with pytest.raises(ValueError, match="adjudication provenance is ambiguous"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=_verified,
        )


def test_authoritative_failed_case_is_reported_but_partial_failure_fails_closed(
    tmp_path: Path,
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    failed_case = fixture.cases[-1]
    failed_root = fixture.runs / failed_case.vc_slug / failed_case.episode_slug
    shutil.rmtree(failed_root)
    failed_row = fixture.runner_status["cases"][-1]
    failed_row.update(
        {
            "status": "failed",
            "phase1_status": "",
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "cumulative_cost_usd": 0.5,
            "state_path": "",
            "error": "workflow failed before state creation",
        }
    )
    fixture.runner_status["cumulative_cost_usd"] = 0.5

    report = module.evaluate_verified_cases(
        canonical_cases=fixture.cases,
        taxonomy=fixture.taxonomy,
        selected=fixture.selected,
        runner_status=fixture.runner_status,
        runs_root=fixture.runs,
        verifier=_verified,
    )

    assert report["metrics"]["case_count"] == 2
    assert report["metrics"]["process"]["failed"] == 1
    assert report["metrics"]["criteria"]["all_18_cases_usable"] is False
    assert len(report["process_rows"]) == 3
    assert report["process_rows"][-1]["case_category"] == "failed"
    assert report["scored_keys"] == [
        [fixture.cases[0].vc_slug, fixture.cases[0].episode_slug],
        [fixture.cases[1].vc_slug, fixture.cases[1].episode_slug],
    ]

    failed_row.update(
        {
            "calls": 1,
            "input_tokens": 5,
            "output_tokens": 2,
            "cost_usd": 0.1,
            "cumulative_cost_usd": 0.6,
        }
    )
    fixture.runner_status["cumulative_cost_usd"] = 0.6
    with pytest.raises(ValueError, match="failure without state has nonzero accounting"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=_verified,
        )
    failed_row.update(
        {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "cumulative_cost_usd": 0.5,
        }
    )
    fixture.runner_status["cumulative_cost_usd"] = 0.5

    (failed_root / "phase1").mkdir(parents=True)
    (failed_root / "phase1/investigation.json").write_text("{}")
    with pytest.raises(ValueError, match="suspicious partial v4.4 artifacts"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=_verified,
        )


def test_not_started_without_state_must_have_zero_accounting(tmp_path: Path) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    case = fixture.cases[-1]
    shutil.rmtree(fixture.runs / case.vc_slug / case.episode_slug)
    row = fixture.runner_status["cases"][-1]
    row.update(
        {
            "status": "not_started_hard_ceiling",
            "phase1_status": "",
            "calls": 1,
            "input_tokens": 5,
            "output_tokens": 2,
            "cost_usd": 0.1,
            "cumulative_cost_usd": 0.6,
            "state_path": "",
        }
    )
    fixture.runner_status["cumulative_cost_usd"] = 0.6

    with pytest.raises(ValueError, match="non-usable case without state has nonzero accounting"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=_verified,
        )


@pytest.mark.parametrize("status", ["not_started_hard_ceiling", "failed"])
def test_zero_accounting_nonusable_case_rejects_empty_run_root_symlink(
    tmp_path: Path, status: str
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    case = fixture.cases[-1]
    run_root = fixture.runs / case.vc_slug / case.episode_slug
    shutil.rmtree(run_root)
    outside = tmp_path / "empty-outside"
    outside.mkdir()
    run_root.symlink_to(outside, target_is_directory=True)
    row = fixture.runner_status["cases"][-1]
    row.update(
        {
            "status": status,
            "phase1_status": "",
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "cumulative_cost_usd": 0.5,
            "state_path": "",
            "error": "stopped before state creation",
        }
    )
    fixture.runner_status["cumulative_cost_usd"] = 0.5

    with pytest.raises(ValueError, match="symlink|unsafe|run root"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=_verified,
        )


@pytest.mark.parametrize("status", ["not_started_hard_ceiling", "failed"])
def test_zero_accounting_nonusable_allows_absent_leaf_under_real_parent(
    tmp_path: Path, status: str
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    case = fixture.cases[-1]
    run_root = fixture.runs / case.vc_slug / case.episode_slug
    shutil.rmtree(run_root)
    assert run_root.parent.is_dir() and not run_root.parent.is_symlink()
    row = fixture.runner_status["cases"][-1]
    row.update(
        {
            "status": status,
            "phase1_status": "",
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "cumulative_cost_usd": 0.5,
            "state_path": "",
            "error": "stopped before state creation",
        }
    )
    fixture.runner_status["cumulative_cost_usd"] = 0.5

    report = module.evaluate_verified_cases(
        canonical_cases=fixture.cases,
        taxonomy=fixture.taxonomy,
        selected=fixture.selected,
        runner_status=fixture.runner_status,
        runs_root=fixture.runs,
        verifier=_verified,
    )

    bucket = "not_started" if status == "not_started_hard_ceiling" else "failed"
    assert report["metrics"]["process"][bucket] == 1
    assert run_root.exists() is False


@pytest.mark.parametrize(
    "leak",
    [
        {"actual_decision": "In"},
        {"note": "The reference rationales say founder execution."},
    ],
)
def test_failed_state_is_scanned_for_leaked_keys_and_values(
    tmp_path: Path, leak: dict[str, str]
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    case = fixture.cases[-1]
    root = fixture.runs / case.vc_slug / case.episode_slug
    (root / "phase1/investigation.json").unlink()
    state_path = root / "state.json"
    state = json.loads(state_path.read_text())
    state.update({"phase1_status": "failed", **leak})
    state_path.write_text(json.dumps(state))
    row = fixture.runner_status["cases"][-1]
    row.update(
        {
            "status": "failed_with_state",
            "phase1_status": "failed",
            "state_path": str(state_path),
        }
    )

    with pytest.raises(ValueError, match="leaked target/evaluation label"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=_verified,
        )


@pytest.mark.parametrize(
    "attack",
    ["missing_call", "hash_mismatch", "usage_mismatch", "duplicate", "path_traversal"],
)
def test_failed_state_requires_exact_persisted_call_ledger(
    tmp_path: Path, attack: str
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    _, root, state = _make_exact_failed_state(fixture)
    records = state["call_records"]
    if attack == "missing_call":
        (root / records[-1]["relative_path"]).unlink()
    elif attack == "hash_mismatch":
        records[-1]["sha256"] = "f" * 64
    elif attack == "usage_mismatch":
        payload = records[-1]["payload"]
        payload["usage"]["input_tokens"] = 11
        raw = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        records[-1]["sha256"] = sha256(raw).hexdigest()
        (root / records[-1]["relative_path"]).write_bytes(raw)
    elif attack == "duplicate":
        records[-1]["relative_path"] = records[0]["relative_path"]
    else:
        records[-1]["relative_path"] = "phase1/adjudication/../call-01.json"
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="provider (?:call|usage|cost|ledger|artifact)"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=_verified,
        )


def test_exact_failed_call_ledger_is_reportable_and_benign_gold_label_text_allowed(
    tmp_path: Path,
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    _make_exact_failed_state(
        fixture,
        note="Our ordinary gold label quality system is working well.",
    )

    report = module.evaluate_verified_cases(
        canonical_cases=fixture.cases,
        taxonomy=fixture.taxonomy,
        selected=fixture.selected,
        runner_status=fixture.runner_status,
        runs_root=fixture.runs,
        verifier=_verified,
    )

    assert report["metrics"]["process"]["failed"] == 1
    assert report["metrics"]["usage"]["calls"] == 7
    assert report["metrics"]["usage"]["total_cost_usd"] == pytest.approx(0.6)
    failed = report["metrics"]["verified_runs"][-1]
    state_path = (
        fixture.runs
        / fixture.cases[-1].vc_slug
        / fixture.cases[-1].episode_slug
        / "state.json"
    )
    assert failed["state_sha256"] == sha256(state_path.read_bytes()).hexdigest()


def test_nonusable_hash_binding_uses_held_descriptor_snapshot_after_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    case, root, _ = _make_exact_failed_state(fixture)
    state_path = root / "state.json"
    original = state_path.read_bytes()
    outside = tmp_path / "replacement-state.json"
    outside.write_text('{"attacker":"replacement"}', encoding="utf-8")
    swapped = False

    def swap_after_snapshot() -> None:
        nonlocal swapped
        swapped = True
        state_path.unlink()
        state_path.symlink_to(outside)

    monkeypatch.setattr(
        module, "_after_nonusable_snapshot", swap_after_snapshot, raising=False
    )

    report = module.evaluate_verified_cases(
        canonical_cases=fixture.cases,
        taxonomy=fixture.taxonomy,
        selected=fixture.selected,
        runner_status=fixture.runner_status,
        runs_root=fixture.runs,
        verifier=_verified,
    )

    verified = next(
        row
        for row in report["metrics"]["verified_runs"]
        if row["episode_slug"] == case.episode_slug
    )
    assert swapped is True
    assert verified["state_sha256"] == sha256(original).hexdigest()
    assert verified["state_sha256"] != sha256(outside.read_bytes()).hexdigest()


@pytest.mark.parametrize("attack", ["state", "final_call", "ancestor", "wal"])
def test_failed_state_rejects_symlinked_provider_artifacts_without_following(
    tmp_path: Path, attack: str
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    _, root, state = _make_exact_failed_state(fixture)
    outside = tmp_path / "outside"
    outside.mkdir()
    records = state["call_records"]
    if attack == "state":
        state_path = root / "state.json"
        external = outside / "state.json"
        external.write_bytes(state_path.read_bytes())
        state_path.unlink()
        state_path.symlink_to(external)
    elif attack == "final_call":
        call_path = root / records[-1]["relative_path"]
        external = outside / "call.json"
        external.write_bytes(call_path.read_bytes())
        call_path.unlink()
        call_path.symlink_to(external)
    elif attack == "ancestor":
        adjudication = root / "phase1/adjudication"
        external = outside / "adjudication"
        shutil.copytree(adjudication, external)
        shutil.rmtree(adjudication)
        adjudication.symlink_to(external, target_is_directory=True)
    else:
        record = records[-1]
        wal = {
            "schema_version": "phase1-provider-call-wal-v1",
            "wal_id": record["wal_id"],
            "status": "responded_uncheckpointed",
            "uncertain_usage": record["payload"]["usage"],
        }
        external = outside / f'{record["wal_id"]}.json'
        external.write_text(json.dumps(wal), encoding="utf-8")
        wal_root = root / ".phase1-v44-wal"
        wal_root.mkdir()
        (wal_root / external.name).symlink_to(external)

    with pytest.raises(ValueError, match="symlink|unsafe|artifact"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=_verified,
        )


def test_failed_state_rejects_adjudication_ancestor_swapped_during_safe_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    _, root, _ = _make_exact_failed_state(fixture)
    adjudication = root / "phase1/adjudication"
    external = tmp_path / "outside-adjudication"
    shutil.copytree(adjudication, external)
    from vc_clone_graph.workflow_v44 import _ArtifactStore

    original = _ArtifactStore.read_bytes
    swapped = False

    def swap_then_read(store, relative: str) -> bytes:
        nonlocal swapped
        if not swapped and relative.startswith("phase1/adjudication/"):
            swapped = True
            shutil.rmtree(adjudication)
            adjudication.symlink_to(external, target_is_directory=True)
        return original(store, relative)

    monkeypatch.setattr(_ArtifactStore, "read_bytes", swap_then_read)

    with pytest.raises(ValueError, match="symlink|unsafe|artifact"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=_verified,
        )
    assert swapped is True


@pytest.mark.parametrize("attack", ["state", "run_root"])
def test_usable_case_rejects_symlink_escape(
    tmp_path: Path, attack: str
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    case = fixture.cases[0]
    run_root = fixture.runs / case.vc_slug / case.episode_slug
    outside = tmp_path / "usable-outside"
    if attack == "state":
        outside.write_bytes((run_root / "state.json").read_bytes())
        (run_root / "state.json").unlink()
        (run_root / "state.json").symlink_to(outside)
    else:
        shutil.copytree(run_root, outside)
        shutil.rmtree(run_root)
        run_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|unsafe|run root"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=_verified,
        )


@pytest.mark.parametrize("mutation", ["file", "ancestor"])
def test_usable_verifier_original_mutation_cannot_change_score_or_bound_hash(
    tmp_path: Path, mutation: str
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    case = fixture.cases[0]
    run_root = fixture.runs / case.vc_slug / case.episode_slug
    investigation_path = run_root / "phase1/investigation.json"
    original = investigation_path.read_bytes()
    replacement = tmp_path / "replacement-investigation.json"
    replacement.write_text(
        json.dumps(
            {
                "schema_version": "investigation-v4.4",
                "episode_slug": case.episode_slug,
                "actual_decision": "Out",
                "candidate_rationales": [],
                "core_rationales": [],
                "rationales": [],
            }
        ),
        encoding="utf-8",
    )
    replacement_phase1 = tmp_path / "replacement-phase1"
    shutil.copytree(run_root / "phase1", replacement_phase1)

    def verify_and_mutate(path: Path) -> None:
        payload = json.loads((path / "state.json").read_text(encoding="utf-8"))
        if payload["episode_slug"] == case.episode_slug:
            if mutation == "file":
                investigation_path.unlink()
                investigation_path.symlink_to(replacement)
            else:
                shutil.rmtree(run_root / "phase1")
                (run_root / "phase1").symlink_to(
                    replacement_phase1, target_is_directory=True
                )

    report = module.evaluate_verified_cases(
        canonical_cases=fixture.cases,
        taxonomy=fixture.taxonomy,
        selected=fixture.selected,
        runner_status=fixture.runner_status,
        runs_root=fixture.runs,
        verifier=verify_and_mutate,
    )

    bound = report["metrics"]["verified_runs"][0]
    assert bound["investigation_sha256"] == sha256(original).hexdigest()
    assert report["paired_rows"][0]["broad_count"] == 2
    assert report["paired_rows"][0]["core_count"] == 1


@pytest.mark.parametrize(
    ("pending_call", "pending_response"),
    [
        ({"garbage": True}, None),
        (None, {"garbage": True}),
        ({"garbage": True}, {"garbage": True}),
    ],
)
def test_failed_state_rejects_every_unsettled_pending_provider_state(
    tmp_path: Path,
    pending_call: object,
    pending_response: object,
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    _, root, state = _make_exact_failed_state(fixture)
    state["pending_call"] = pending_call
    state["pending_response"] = pending_response
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="pending provider state|outcome/cost"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=_verified,
        )


@pytest.mark.parametrize(
    "embedded",
    [
        'cached text: "actual_decision": "In"',
        'cached text: "target_decision": "Out"',
        'cached text: "reference_rationales": ["founder_execution"]',
    ],
)
def test_failed_state_rejects_embedded_forbidden_json_keys(
    tmp_path: Path, embedded: str
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    _make_exact_failed_state(fixture, note=embedded)

    with pytest.raises(ValueError, match="leaked target/evaluation label"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=_verified,
        )


@pytest.mark.parametrize(
    "runner_failure",
    ["failed_rate_ceiling_breach", "failed_cost_ceiling_exceeded"],
)
def test_post_completion_runner_failure_is_verified_reported_and_not_scored(
    tmp_path: Path, runner_failure: str
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    row = fixture.runner_status["cases"][0]
    row.update(
        {
            "status": runner_failure,
            "error": "runner ceiling was breached after terminal completion",
        }
    )
    run_root = fixture.runs / row["vc_slug"] / row["episode_slug"]
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    _materialize_exact_call_ledger(run_root, state)
    expected_inventory = {
        path.relative_to(run_root).as_posix(): path.read_bytes()
        for path in run_root.rglob("*")
        if path.is_file()
    }
    verified: list[Path] = []
    inspected: list[dict[str, bytes]] = []

    def capture_verification(path: Path) -> None:
        verified.append(path)
        payload = json.loads((path / "state.json").read_text(encoding="utf-8"))
        if payload["episode_slug"] == fixture.cases[0].episode_slug:
            inspected.append(
                {
                    item.relative_to(path).as_posix(): item.read_bytes()
                    for item in path.rglob("*")
                    if item.is_file()
                }
            )

    report = module.evaluate_verified_cases(
        canonical_cases=fixture.cases,
        taxonomy=fixture.taxonomy,
        selected=fixture.selected,
        runner_status=fixture.runner_status,
        runs_root=fixture.runs,
        verifier=capture_verification,
    )

    assert verified[0] != (
        fixture.runs / fixture.cases[0].vc_slug / fixture.cases[0].episode_slug
    )
    original_roots = {
        fixture.runs / case.vc_slug / case.episode_slug for case in fixture.cases
    }
    assert len(verified) == 3
    assert all(path not in original_roots for path in verified)
    assert inspected[0] == expected_inventory
    bound = report["metrics"]["verified_runs"][0]
    assert bound["state_sha256"] == sha256(inspected[0]["state.json"]).hexdigest()
    assert bound["investigation_sha256"] == sha256(
        inspected[0]["phase1/investigation.json"]
    ).hexdigest()
    assert report["metrics"]["case_count"] == 2
    assert report["metrics"]["process"]["failed"] == 1
    assert report["process_rows"][0]["case_category"] == "failed"


def test_post_completion_verifier_and_binding_use_snapshot_before_original_swap(
    tmp_path: Path,
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    case = fixture.cases[0]
    row = fixture.runner_status["cases"][0]
    row.update(
        {
            "status": "failed_rate_ceiling_breach",
            "error": "runner ceiling was breached after terminal completion",
        }
    )
    run_root = fixture.runs / case.vc_slug / case.episode_slug
    state_path = run_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _materialize_exact_call_ledger(run_root, state)
    replacement = tmp_path / "replacement-state.json"
    replacement.write_text('{"attacker":"replacement"}', encoding="utf-8")
    inspected: bytes | None = None

    def verify_and_swap(path: Path) -> None:
        nonlocal inspected
        payload = json.loads((path / "state.json").read_text(encoding="utf-8"))
        if payload["episode_slug"] != case.episode_slug:
            return
        inspected = (path / "state.json").read_bytes()
        state_path.unlink()
        state_path.symlink_to(replacement)

    report = module.evaluate_verified_cases(
        canonical_cases=fixture.cases,
        taxonomy=fixture.taxonomy,
        selected=fixture.selected,
        runner_status=fixture.runner_status,
        runs_root=fixture.runs,
        verifier=verify_and_swap,
    )

    assert inspected is not None
    bound = report["metrics"]["verified_runs"][0]
    assert bound["state_sha256"] == sha256(inspected).hexdigest()
    assert bound["state_sha256"] != sha256(replacement.read_bytes()).hexdigest()


def test_post_completion_snapshot_verifier_rejects_extra_field_before_capture(
    tmp_path: Path,
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    case = fixture.cases[0]
    row = fixture.runner_status["cases"][0]
    row.update(
        {
            "status": "failed_rate_ceiling_breach",
            "error": "runner ceiling was breached after terminal completion",
        }
    )
    run_root = fixture.runs / case.vc_slug / case.episode_slug
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    _materialize_exact_call_ledger(run_root, state)
    investigation_path = run_root / "phase1/investigation.json"
    investigation = json.loads(investigation_path.read_text(encoding="utf-8"))
    investigation["unexpected_field"] = "must be rejected"
    investigation_path.write_text(json.dumps(investigation), encoding="utf-8")

    def reject_extra_field(path: Path) -> None:
        payload = json.loads(
            (path / "phase1/investigation.json").read_text(encoding="utf-8")
        )
        if payload.get("episode_slug") == case.episode_slug:
            assert "unexpected_field" in payload
            raise ValueError("strict verifier rejected unexpected field")

    with pytest.raises(ValueError, match="v4.4 verification failed"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=reject_extra_field,
        )


def test_real_v44_snapshot_bridge_preserves_empty_wal_for_production_verifier(
    tmp_path: Path,
) -> None:
    module = _load_evaluator_module()
    from vc_clone_graph.artifacts import verify_phase1_artifacts

    source = _make_real_v44_run(tmp_path)
    runs_root = tmp_path / "real-runs"
    run_root = runs_root / "vc" / "episode"
    run_root.parent.mkdir(parents=True)
    shutil.copytree(source, run_root)

    snapshot = module._snapshot_nonusable_run(runs_root, run_root)

    assert ".phase1-v44-wal" in snapshot.directories
    module._verify_nonusable_snapshot(
        snapshot,
        lambda path: verify_phase1_artifacts(path, provenance_mode="direct"),
    )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_snapshot_bridge_detects_missing_or_extra_materialized_directory(
    tmp_path: Path, mutation: str
) -> None:
    module = _load_evaluator_module()
    source = _make_real_v44_run(tmp_path)
    runs_root = tmp_path / "real-runs"
    run_root = runs_root / "vc" / "episode"
    run_root.parent.mkdir(parents=True)
    shutil.copytree(source, run_root)
    snapshot = module._snapshot_nonusable_run(runs_root, run_root)

    def mutate_directory(path: Path) -> None:
        if mutation == "missing":
            (path / ".phase1-v44-wal").rmdir()
        else:
            (path / "unexpected-empty-directory").mkdir()

    with pytest.raises(ValueError, match="directory|inventory"):
        module._verify_nonusable_snapshot(snapshot, mutate_directory)


def test_production_verifier_rejects_invalid_state_field_captured_in_snapshot(
    tmp_path: Path,
) -> None:
    module = _load_evaluator_module()
    from vc_clone_graph.artifacts import canonical_bytes, verify_phase1_artifacts

    source = _make_real_v44_run(tmp_path)
    state_path = source / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["unexpected_field"] = "must be rejected"
    state_path.write_bytes(canonical_bytes(state))
    runs_root = tmp_path / "real-runs"
    run_root = runs_root / "vc" / "episode"
    run_root.parent.mkdir(parents=True)
    shutil.copytree(source, run_root)
    snapshot = module._snapshot_nonusable_run(runs_root, run_root)

    with pytest.raises(ValueError, match="exact terminal shape"):
        module._verify_nonusable_snapshot(
            snapshot,
            lambda path: verify_phase1_artifacts(path, provenance_mode="direct"),
        )


def test_promotion_criteria_are_the_eight_prespecified_gates() -> None:
    module = _load_evaluator_module()
    criteria = module.promotion_criteria(
        usable_cases=18,
        canonical={
            "micro_precision": 0.30,
            "micro_recall": 0.60,
            "micro_f1": 0.40,
            "family_micro_f1": 0.70,
            "average_predicted_size": 5.0,
        },
        broad={
            "micro_precision": 0.35,
            "micro_recall": 0.60,
            "micro_f1": 0.44,
            "family_micro_f1": 0.72,
            "average_predicted_size": 10.0,
        },
        core={
            "micro_precision": 0.31,
            "micro_recall": 0.55,
            "micro_f1": 0.43,
            "family_micro_f1": 0.70,
            "average_predicted_size": 4.0,
        },
        canonical_primary_explicit_recall=0.80,
        broad_primary_explicit_recall=0.77,
        canonical_vc_f1={f"vc-{i}": 0.4 for i in range(6)},
        core_vc_f1={f"vc-{i}": 0.4 if i < 4 else 0.39 for i in range(6)},
    )

    assert list(criteria) == [
        "all_18_cases_usable",
        "broad_exact_recall_at_least_canonical",
        "broad_primary_explicit_recall_loss_at_most_0_03",
        "core_exact_f1_gain_at_least_0_03",
        "core_family_f1_not_reduced",
        "core_precision_exceeds_canonical",
        "at_least_four_of_six_vcs_preserve_core_f1",
        "dual_view_rationale_count_bounds",
    ]
    assert all(criteria.values())


def test_evaluator_fails_closed_on_unverified_missing_ambiguous_or_leaked_run(
    tmp_path: Path,
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    first = fixture.cases[0]
    first_root = fixture.runs / first.vc_slug / first.episode_slug

    with pytest.raises(ValueError, match="verification failed"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=lambda _path: (_ for _ in ()).throw(ValueError("tampered")),
        )

    (first_root / "phase1/investigation.json").unlink()
    with pytest.raises(ValueError, match="missing v4.4 investigation"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=_verified,
        )

    fixture = _fixture(tmp_path / "ambiguous")
    with pytest.raises(ValueError, match="canonical cases are ambiguous"):
        module.evaluate_verified_cases(
            canonical_cases=[*fixture.cases, fixture.cases[0]],
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=_verified,
        )

    fixture = _fixture(tmp_path / "mixed")
    mixed = fixture.runs / fixture.cases[0].vc_slug / fixture.cases[0].episode_slug
    payload = json.loads((mixed / "phase1/investigation.json").read_text())
    payload["schema_version"] = "investigation-v4.3"
    (mixed / "phase1/investigation.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="mixed or unsupported v4.4 contract"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=_verified,
        )

    fixture = _fixture(tmp_path / "leaked")
    leaked = fixture.runs / fixture.cases[0].vc_slug / fixture.cases[0].episode_slug
    payload = json.loads((leaked / "phase1/investigation.json").read_text())
    payload["actual_decision"] = "In"
    (leaked / "phase1/investigation.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="leaked target/evaluation label"):
        module.evaluate_verified_cases(
            canonical_cases=fixture.cases,
            taxonomy=fixture.taxonomy,
            selected=fixture.selected,
            runner_status=fixture.runner_status,
            runs_root=fixture.runs,
            verifier=_verified,
        )


def test_output_set_is_deterministic_and_manifest_hash_binds_every_report(
    tmp_path: Path,
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    report = module.evaluate_verified_cases(
        canonical_cases=fixture.cases,
        taxonomy=fixture.taxonomy,
        selected=fixture.selected,
        runner_status=fixture.runner_status,
        runs_root=fixture.runs,
        verifier=_verified,
    )
    output = tmp_path / "report"
    (output / "configs").mkdir(parents=True)
    (output / "configs/runtime.toml").write_text("# runner provenance\n")
    (output / "status.json").write_text('{"runner": "complete"}\n')
    inputs = {
        "registry": {"path": "registry.json", "sha256": "a" * 64},
        "references_manifest": {"path": "references/manifest.json", "sha256": "b" * 64},
        "taxonomy": {"path": "taxonomy.json", "sha256": "c" * 64},
        "canary_manifest": {"path": "canary.json", "sha256": "d" * 64},
        "runner_status": {"path": "status.json", "sha256": "e" * 64},
        "scored_case_artifacts": [
            {
                "vc_slug": "vc-1",
                "episode_slug": "1-case",
                "canonical_investigation": {"path": "c.json", "sha256": "f" * 64},
                "canonical_sidecar": {"path": "c.sha256", "sha256": "1" * 64},
                "automated_reference": {"path": "r.json", "sha256": "2" * 64},
            }
        ],
    }

    module.write_outputs(output, report, inputs=inputs)
    first = {
        path.name: path.read_bytes()
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "status.json"
    }
    module.write_outputs(output, report, inputs=inputs)
    second = {
        path.name: path.read_bytes()
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "status.json"
    }

    assert first == second
    assert set(first) == {
        "metrics.json",
        "canonical-result.json",
        "broad-result.json",
        "core-result.json",
        "paired-cases.csv",
        "taxonomy-opportunity.csv",
        "process-diagnostics.csv",
        "evaluation.md",
        "manifest.json",
    }
    assert (output / "status.json").read_text() == '{"runner": "complete"}\n'
    assert (output / "configs/runtime.toml").read_text() == "# runner provenance\n"
    manifest = json.loads(first["manifest.json"])
    assert manifest["scientific_status"] == "development_canary_not_untouched_holdout"
    assert set(manifest["outputs"]) == set(first) - {"manifest.json"}
    assert manifest["inputs"] == inputs
    for filename, digest in manifest["outputs"].items():
        assert module.sha256_file(output / filename) == digest


@pytest.mark.parametrize(
    "mutated_name",
    [
        "runner-status.json",
        "canonical.json",
        "canonical.sha256",
        "reference.json",
        "taxonomy.json",
        "canary-manifest.json",
    ],
)
def test_report_publication_aborts_if_any_captured_input_mutates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated_name: str,
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    report = module.evaluate_verified_cases(
        canonical_cases=fixture.cases,
        taxonomy=fixture.taxonomy,
        selected=fixture.selected,
        runner_status=fixture.runner_status,
        runs_root=fixture.runs,
        verifier=_verified,
    )
    names = [
        "runner-status.json",
        "canonical.json",
        "canonical.sha256",
        "reference.json",
        "taxonomy.json",
        "canary-manifest.json",
    ]
    paths = []
    for name in names:
        path = tmp_path / "sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}: version A\n", encoding="utf-8")
        paths.append(path)
    captured = [
        module._capture_file(path, name) for path, name in zip(paths, names)
    ]
    target = tmp_path / "sources" / mutated_name

    def mutate_before_recheck() -> None:
        target.write_text(f"{mutated_name}: version B\n", encoding="utf-8")

    monkeypatch.setattr(module, "_before_report_source_recheck", mutate_before_recheck)
    output = tmp_path / "not-published"

    with pytest.raises(ValueError, match="mutated during run"):
        module._publish_captured_report(
            output,
            report,
            inputs={
                name: module._captured_binding(source)
                for name, source in zip(names, captured)
            },
            captured_sources=captured,
            tree_sources=(),
        )

    assert output.exists() is False


def test_report_publication_aborts_if_captured_glob_inventory_mutates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_evaluator_module()
    fixture = _fixture(tmp_path)
    report = module.evaluate_verified_cases(
        canonical_cases=fixture.cases,
        taxonomy=fixture.taxonomy,
        selected=fixture.selected,
        runner_status=fixture.runner_status,
        runs_root=fixture.runs,
        verifier=_verified,
    )
    summaries = tmp_path / "summaries"
    first = summaries / "first" / "summary.json"
    first.parent.mkdir(parents=True)
    first.write_text("{}\n", encoding="utf-8")
    captured_glob = module._CapturedGlob(
        summaries,
        "*/summary.json",
        (first.absolute(),),
    )

    def mutate_before_recheck() -> None:
        second = summaries / "second" / "summary.json"
        second.parent.mkdir(parents=True)
        second.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(module, "_before_report_source_recheck", mutate_before_recheck)
    output = tmp_path / "not-published"

    with pytest.raises(ValueError, match="glob inventory mutated"):
        module._publish_captured_report(
            output,
            report,
            inputs={},
            captured_sources=(),
            tree_sources=(),
            glob_sources=(captured_glob,),
        )

    assert output.exists() is False


def test_captured_registry_discovery_rejects_unexpected_summary(
    tmp_path: Path,
) -> None:
    module = _load_evaluator_module()
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            [
                {
                    "episode_slug": "1-selected",
                    "evaluation_eligible": True,
                    "pitch_window_decision": "In",
                }
            ]
        ),
        encoding="utf-8",
    )
    summaries = tmp_path / "summaries"
    for episode in ("1-selected", "2-unexpected"):
        path = summaries / episode / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"episode_slug": episode}), encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "canonical-vc-evaluation-registry-v1",
                "investors": {
                    "vc": {
                        "display_name": "VC",
                        "label_file": "labels.json",
                        "eligible_count": 1,
                        "sources": [
                            {
                                "kind": "summary_glob",
                                "path": "summaries/*/summary.json",
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    references = tmp_path / "references"
    record = references / "records" / "vc" / "1-selected.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps({"vc_slug": "vc", "episode_slug": "1-selected"}),
        encoding="utf-8",
    )
    snapshot = module._snapshot_nonusable_run(tmp_path, references)

    with pytest.raises(ValueError, match="unexpected canonical artifacts"):
        module._discover_cases_from_captured_registry(
            module._capture_file(registry, "registry"),
            selected={
                ("vc", "1-selected"): {
                    "actual_decision": "In",
                    "role": "in",
                }
            },
            reference_root=references,
            reference_snapshot=snapshot,
        )
