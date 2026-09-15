from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_phase1_v44_canary as runner


EXPECTED_CASES = (
    ("charles-hudson", "39-this-pitch-is-damn-near-perfect"),
    ("charles-hudson", "152-oma-health"),
    ("charles-hudson", "135-thoras-ai-the-twin-effect"),
    ("cyan-banister", "169-mappa-can-voice-ai-find-you-a-job-or-a-date"),
    ("cyan-banister", "146-recraft-beer"),
    ("cyan-banister", "143-vital-audio"),
    ("elizabeth-yin", "136-kredfeed-the-next-mexican-unicorn"),
    ("elizabeth-yin", "120-bevz-a-tech-bro-walks-into-a-corner-store"),
    ("elizabeth-yin", "83-can-small-bras-be-a-big-market"),
    ("jesse-middleton", "149-dopl"),
    ("jesse-middleton", "159-aura-finance-chasing-the-ghost-of-mint"),
    ("jesse-middleton", "163-curiedx-ai-pocket-doctor"),
    ("jillian-manus", "8-hykso"),
    ("jillian-manus", "104-dressd-red-carpet-or-red-ocean"),
    ("jillian-manus", "3-industrial-organic"),
    ("phil-nadel", "51-this-bot-can-fight-your-atm-fees"),
    ("phil-nadel", "46-never-get-lost-again"),
    ("phil-nadel", "43-get-this-party-startup-in-here"),
)


def _case(position: int, *, vc_slug: str = "charles-hudson") -> dict[str, object]:
    return {
        "vc_slug": vc_slug,
        "episode_slug": f"{position}-case-{position}",
        "actual_decision": "In" if position % 2 else "Out",
        "role": "coverage",
    }


def _base_config(tmp_path: Path, vc_slug: str = "charles-hudson") -> Path:
    path = tmp_path / "configs" / f"{vc_slug}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'''[run]
vc_slug = "charles-hudson-precursor-ventures"
episode_slug = "1-case-1"
input_root = "inputs"
output_root = "outputs/phase1-v44-canary-2026-08-17/investors/{vc_slug}"
checkpoint_path = "outputs/phase1-v44-canary-2026-08-17/checkpoints/1-case-1.sqlite"
contract_version = "v4.4"
mode = "phase1_only"

[provider]
kind = "openrouter"
model = "openai/gpt-5.6-luna"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
require_parameters = true
data_collection = "deny"
max_output_tokens = 16384
context_window = 131072
request_timeout_seconds = 600

[embedding]
kind = "sentence_transformers"
model = "nomic-ai/nomic-embed-text-v1.5"
revision = "e9b6763023c676ca8431644204f50c2b100d9aab"
device = "auto"
batch_size = 16
normalize = true
document_prefix = "search_document: "
query_prefix = "search_query: "
require_complete_index = true

[phase1]
model = "openai/gpt-5.6-luna"
reasoning_effort = "high"
min_iterations = 1
max_iterations = 4
max_output_tokens = 16384
planning_max_output_tokens = 8192
planning_reasoning_effort = "low"
max_precedent_searches = 4
max_precedent_reads = 8

[phase2]
model = "openai/gpt-5.6-luna"
reasoning_effort = "high"
min_iterations = 1
max_iterations = 4
max_output_tokens = 16384
planning_max_output_tokens = 8192
planning_reasoning_effort = "low"
max_precedent_searches = 4
max_precedent_reads = 8

[retrieval]
top_k = 6
max_exact_reads = 12

[precedents]
enabled = true
corpus_path = "data/investors/charles-hudson-precursor-ventures/precedents"
allow_full_transcript = true
selection_policy = "semantic"
candidate_pool_k = 30
in_slots = 2
out_slots = 2

[portfolio_memory]
enabled = true
corpus_path = "data/investors/<vc_slug>/portfolio-memory"
retrieval_top_k = 5
candidate_pool_k = 15
require_complete_embeddings = true

[phase1_v44]
taxonomy_top_k = 5
claim_retrieval_top_k = 3
max_wiki_reads_per_claim = 2
max_precedent_reads_per_claim = 2
max_revisits = 1
''',
        encoding="utf-8",
    )
    return path


def _args(
    tmp_path: Path,
    manifest: Path,
    *,
    ceiling: float | None = 1.0,
    reserve: float = 0.10,
    dry_run: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=manifest,
        config_dir=tmp_path / "configs",
        output=tmp_path / "reports",
        cost_ceiling=ceiling,
        per_case_reserve=reserve,
        max_input_usd_per_million=0.20,
        max_output_usd_per_million=0.20,
        limit=None,
        dry_run=dry_run,
    )


def _ready_preflight(
    command: list[str], **_kwargs
) -> subprocess.CompletedProcess[str]:
    from vc_clone_graph.config import load_config

    config = load_config(Path(command[-1]))
    embedding = {
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "revision": "e9b6763023c676ca8431644204f50c2b100d9aab",
        "coverage": 1.0,
    }
    payload = {
        "status": "ready",
        "target": config.run.episode_slug,
        "target_accessible": False,
        "phase2_model": None,
        "wiki_embedding_index": embedding,
        "precedent_embedding_index": embedding,
        "taxonomy_embedding_preflight": {
            **embedding,
            "dimension": 768,
        },
        "portfolio_memory": {"target_accessible": False},
    }
    return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")


def _run_canary(
    args: argparse.Namespace,
    *,
    execute=subprocess.run,
    preflight_execute=_ready_preflight,
) -> int:
    original_root = runner.PROJECT_ROOT
    manifest = Path(args.manifest).resolve()
    active_root = Path(original_root).resolve()
    if not manifest.is_relative_to(active_root):
        runner.PROJECT_ROOT = manifest.parent
    try:
        return runner.run_canary(
            args,
            execute=execute,
            preflight_execute=preflight_execute,
        )
    finally:
        runner.PROJECT_ROOT = original_root


def _manifest(tmp_path: Path, cases: list[dict[str, object]]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    return path


def _write_state(
    root: Path,
    runtime_config: Path,
    *,
    phase1_status: str = "accepted",
    cost: float = 0.10,
    phases: tuple[str, ...] = ("phase1_claim_extraction", "phase1_adjudication"),
) -> None:
    from vc_clone_graph.config import load_config

    config = load_config(runtime_config)
    state_path = (
        root / config.run.output_root / config.run.episode_slug / "state.json"
    )
    path_by_phase = {
        "phase1_claim_extraction": "phase1/claim-extraction/call-01.json",
        "phase1_claim_extraction_repair": "phase1/claim-extraction/call-02.json",
        "phase1_adjudication": "phase1/adjudication/turn-01/call-01.json",
        "phase1_adjudication_repair": "phase1/adjudication/turn-01/call-02.json",
    }
    records: list[dict[str, object]] = []
    for i, phase in enumerate(phases, start=1):
        usage = {
            "input_tokens": 10,
            "cached_input_tokens": 0,
            "output_tokens": 5,
            "cost_usd": cost / len(phases),
        }
        payload = {
            "phase": phase,
            "prompt": "public pitch",
            "schema": {},
            "raw": "{}",
            "parsed": {},
            "provider_metadata": {},
            "usage": usage,
            "cost_usd": usage["cost_usd"],
            "elapsed_seconds": 0.1,
            "max_output_tokens": 100,
            "reasoning_effort": "high",
        }
        relative_path = path_by_phase[phase]
        call_path = state_path.parent / relative_path
        call_path.parent.mkdir(parents=True, exist_ok=True)
        call_path.write_bytes(_canonical_bytes(payload))
        records.append(
            {
                "phase": phase,
                "relative_path": relative_path,
                "sha256": sha256(_canonical_bytes(payload)).hexdigest(),
                "payload": payload,
                "wal_id": f"{i:064x}",
                "request_sha256": f"{i + 10:064x}",
            }
        )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "phase1_status": phase1_status,
                "phase2_status": "not_run",
                "usage": {
                    "input_tokens": 10 * len(phases),
                    "cached_input_tokens": 0,
                    "output_tokens": 5 * len(phases),
                    "cost_usd": cost,
                },
                "call_records": records,
                "events": [],
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = root / config.run.checkpoint_path
    if checkpoint_path.is_file():
        from langgraph.checkpoint.base import empty_checkpoint
        from langgraph.checkpoint.sqlite import SqliteSaver

        thread_id = f"{config.run.vc_slug}:{config.run.episode_slug}"
        value = json.loads(state_path.read_text(encoding="utf-8"))
        with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = value
            saver.put(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": "",
                    }
                },
                checkpoint,
                {"source": "loop", "step": 1, "parents": {}},
                {},
            )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def test_account_state_accepts_exact_workflow_call_artifact_bytes(
    tmp_path: Path,
) -> None:
    from vc_clone_graph.workflow_v44 import _canonical_json_bytes

    usage = {
        "input_tokens": 20,
        "cached_input_tokens": 0,
        "output_tokens": 5,
        "cost_usd": 0.01,
    }
    payload = {
        "phase": "phase1_claim_extraction",
        "prompt": "public café pitch",
        "schema": {},
        "raw": "{}",
        "parsed": {},
        "provider_metadata": {},
        "usage": usage,
        "cost_usd": usage["cost_usd"],
        "elapsed_seconds": 0.1,
        "max_output_tokens": 100,
        "reasoning_effort": "high",
    }
    raw = _canonical_json_bytes(payload)
    relative_path = "phase1/claim-extraction/call-01.json"
    call_path = tmp_path / relative_path
    call_path.parent.mkdir(parents=True)
    call_path.write_bytes(raw)
    state = {
        "phase1_status": "running",
        "phase2_status": "not_run",
        "usage": usage,
        "call_records": [
            {
                "relative_path": relative_path,
                "sha256": sha256(raw).hexdigest(),
                "payload": payload,
                "wal_id": "a" * 64,
                "request_sha256": "b" * 64,
            }
        ],
        "pending_call": None,
        "pending_response": None,
    }

    accounting = runner._account_state(
        state,
        run_root=tmp_path,
        source="checkpoint",
    )

    assert accounting.calls == 1
    assert accounting.cost_usd == pytest.approx(0.01)


def _write_checkpoint(
    root: Path,
    runtime_config: Path,
    *,
    cost: float,
    reconciled: bool = True,
    phase1_status: str = "running",
) -> Path:
    from langgraph.checkpoint.base import empty_checkpoint
    from langgraph.checkpoint.sqlite import SqliteSaver
    from vc_clone_graph.config import load_config

    config = load_config(runtime_config)
    checkpoint = root / config.run.checkpoint_path
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    usage = {
        "input_tokens": 20,
        "cached_input_tokens": 0,
        "output_tokens": 5,
        "cost_usd": cost,
    }
    payload = {
        "phase": "phase1_claim_extraction",
        "prompt": "public pitch",
        "schema": {},
        "raw": "{}",
        "parsed": {},
        "provider_metadata": {},
        "usage": usage,
        "cost_usd": cost,
        "elapsed_seconds": 0.1,
        "max_output_tokens": 100,
        "reasoning_effort": "high",
    }
    wal_id = "a" * 64
    record = {
        "relative_path": "phase1/claim-extraction/call-01.json",
        "sha256": sha256(_canonical_bytes(payload)).hexdigest(),
        "payload": payload,
        "wal_id": wal_id,
        "request_sha256": "b" * 64,
    }
    state = {
        "phase1_status": phase1_status,
        "phase2_status": "not_run",
        "usage": usage,
        "call_records": [record] if reconciled else [],
        "pending_call": None,
        "pending_response": None,
    }
    run_root = root / config.run.output_root / config.run.episode_slug
    if reconciled:
        call_path = run_root / str(record["relative_path"])
        call_path.parent.mkdir(parents=True, exist_ok=True)
        call_path.write_bytes(_canonical_bytes(payload))
    thread_id = f"{config.run.vc_slug}:{config.run.episode_slug}"
    with SqliteSaver.from_conn_string(str(checkpoint)) as saver:
        checkpoint_value = empty_checkpoint()
        checkpoint_value["channel_values"] = state
        saver.put(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
            checkpoint_value,
            {"source": "input", "step": 0, "parents": {}},
            {},
        )
    return checkpoint


def test_frozen_manifest_has_exact_18_case_order() -> None:
    cases = runner.load_cases(ROOT / runner.DEFAULT_MANIFEST)
    assert tuple((row["vc_slug"], row["episode_slug"]) for row in cases) == EXPECTED_CASES


def test_six_v44_configs_preserve_investor_sources_and_pin_local_embeddings() -> None:
    from vc_clone_graph.config import load_config

    expected_roots = {
        name: f"outputs/phase1-v44-canary-2026-08-17/investors/{name}"
        for name in runner.CONFIG_NAMES
    }
    for name, filename in runner.CONFIG_NAMES.items():
        v43 = load_config(ROOT / "configs/v43" / filename)
        path = ROOT / "configs/v44" / filename
        v44 = load_config(path)
        assert v44.run.contract_version == "v4.4"
        assert v44.run.mode == "phase1_only"
        assert v44.run.output_root == expected_roots[name]
        assert v44.run.vc_slug == v43.run.vc_slug
        assert v44.run.input_root == v43.run.input_root
        assert v44.run.taxonomy_path == v43.run.taxonomy_path
        assert v44.precedents.corpus_path == v43.precedents.corpus_path
        assert v44.portfolio_memory.corpus_path == v43.portfolio_memory.corpus_path
        assert v44.embedding is not None and v43.embedding is not None
        assert v44.embedding.model == "nomic-ai/nomic-embed-text-v1.5"
        assert v44.embedding.revision == v43.embedding.revision
        assert v44.embedding.require_complete_index is True
        assert v44.provider.kind == "openrouter"
        assert v44.provider.model == "openai/gpt-5.6-luna"
        assert v44.provider.api_key_env == "OPENROUTER_API_KEY"
        assert "sk-or-" not in path.read_text(encoding="utf-8")
        assert v44.phase1.reasoning_effort == "high"
        assert v44.phase1_v44 is not None
        assert v44.phase1_v44.model_dump() == {
            "taxonomy_top_k": 5,
            "claim_retrieval_top_k": 3,
            "max_wiki_reads_per_claim": 2,
            "max_precedent_reads_per_claim": 2,
            "max_revisits": 1,
        }


def test_explicit_dry_run_prints_exactly_18_and_never_executes(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    manifest = _manifest(
        tmp_path,
        runner.load_cases(ROOT / runner.DEFAULT_MANIFEST),
    )
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    for filename in runner.CONFIG_NAMES.values():
        (config_dir / filename).write_text(
            (ROOT / "configs/v44" / filename).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    result = _run_canary(
        argparse.Namespace(
            manifest=manifest,
            config_dir=config_dir,
            output=tmp_path / "dry-run",
            cost_ceiling=None,
            per_case_reserve=0.20,
            limit=None,
            dry_run=True,
        ),
        execute=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run executed a model command")
        ),
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert result == 0
    assert len(lines) == 18
    assert lines[0].endswith("charles-hudson 39-this-pitch-is-damn-near-perfect In")
    assert lines[-1].endswith("phil-nadel 43-get-this-party-startup-in-here Out")


def test_paid_mode_requires_explicit_cost_ceiling(tmp_path: Path) -> None:
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1)])
    with pytest.raises(ValueError, match="explicit --cost-ceiling"):
        _run_canary(_args(tmp_path, manifest, ceiling=None))


@pytest.mark.parametrize(
    ("ceiling", "reserve"),
    [
        (math.nan, 0.1),
        (math.inf, 0.1),
        (-math.inf, 0.1),
        (1.0, math.nan),
        (1.0, math.inf),
        (1.0, -0.1),
    ],
)
def test_nonfinite_or_negative_cost_controls_fail_before_execution(
    tmp_path: Path,
    ceiling: float,
    reserve: float,
) -> None:
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1)])
    called = False

    def execute(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid controls reached execution")

    with pytest.raises(ValueError, match="finite"):
        _run_canary(
            _args(tmp_path, manifest, ceiling=ceiling, reserve=reserve),
            execute=execute,
        )
    assert called is False


def test_render_case_config_uses_v44_checkpoint_and_requested_episode() -> None:
    base = (ROOT / "configs/v43/charles-hudson.toml").read_text(encoding="utf-8")
    rendered = runner.render_case_config(base, episode_slug="152-oma-health")
    assert 'episode_slug = "152-oma-health"' in rendered
    assert (
        'checkpoint_path = "outputs/phase1-v44-canary-2026-08-17/'
        'checkpoints/152-oma-health.sqlite"'
    ) in rendered


@pytest.mark.parametrize("phase1_status", ["accepted", "provisional"])
def test_runner_skips_usable_phase1_only_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase1_status: str
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1)])
    runtime = tmp_path / "reports/configs/charles-hudson/1-case-1.toml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(
        runner.render_case_config(
            (tmp_path / "configs/charles-hudson.toml").read_text(encoding="utf-8"),
            episode_slug="1-case-1",
        ),
        encoding="utf-8",
    )
    _write_state(tmp_path, runtime, phase1_status=phase1_status, cost=0.07)

    _run_canary(
        _args(tmp_path, manifest),
        execute=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("usable state was rerun")
        ),
    )
    status = json.loads((tmp_path / "reports/status.json").read_text(encoding="utf-8"))
    assert status["cases"][0]["status"] == "completed_existing"
    assert status["cases"][0]["cost_usd"] == pytest.approx(0.07)


def test_runner_resumes_checkpoint_and_counts_all_v44_call_kinds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    base = _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1)])
    checkpoint = (
        tmp_path / "outputs/phase1-v44-canary-2026-08-17/checkpoints/1-case-1.sqlite"
    )
    runtime = tmp_path / "reports/configs/charles-hudson/1-case-1.toml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(
        runner.render_case_config(
            (tmp_path / "configs/charles-hudson.toml").read_text(encoding="utf-8"),
            episode_slug="1-case-1",
        ),
        encoding="utf-8",
    )
    _write_checkpoint(tmp_path, runtime, cost=0.02)
    commands: list[str] = []

    def execute(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        commands.append(command[3])
        runtime = Path(command[-1])
        _write_state(
            tmp_path,
            runtime,
            phases=(
                "phase1_claim_extraction",
                "phase1_claim_extraction_repair",
                "phase1_adjudication",
                "phase1_adjudication_repair",
            ),
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    _run_canary(_args(tmp_path, manifest), execute=execute)
    rows = list(csv.DictReader((tmp_path / "reports/status.csv").open(encoding="utf-8")))
    assert commands == ["resume"]
    assert rows[0]["calls"] == "4"
    assert not list((tmp_path / "reports").glob("*.tmp"))
    assert base.is_file()


def test_runner_stops_before_remaining_cap_exceeds_ceiling_and_reloads_saved_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _base_config(tmp_path)
    cases = [_case(1), _case(2), _case(3)]
    manifest = _manifest(tmp_path, cases)
    executions: list[str] = []

    def execute(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        runtime = Path(command[-1])
        executions.append(runtime.stem)
        _write_state(tmp_path, runtime, cost=0.08)
        return subprocess.CompletedProcess(command, 0, "", "")

    args = _args(tmp_path, manifest, ceiling=0.25, reserve=0.10)
    args.max_input_usd_per_million = 0.11
    args.max_output_usd_per_million = 0.11
    assert _run_canary(args, execute=execute) == runner.EXIT_BUDGET_EXHAUSTED
    first = json.loads((tmp_path / "reports/status.json").read_text(encoding="utf-8"))
    assert executions == ["1-case-1", "2-case-2"]
    assert first["cases"][-1]["status"] == "not_started_hard_ceiling"
    assert first["cumulative_cost_usd"] == pytest.approx(0.16)

    executions.clear()
    assert _run_canary(args, execute=execute) == runner.EXIT_BUDGET_EXHAUSTED
    second = json.loads((tmp_path / "reports/status.json").read_text(encoding="utf-8"))
    assert executions == []
    assert second["cumulative_cost_usd"] == pytest.approx(0.16)
    assert second["cases"][-1]["status"] == "not_started_hard_ceiling"


def test_exact_ceiling_marks_every_remaining_selected_case_and_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1), _case(2)])
    executions: list[str] = []

    def execute(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        runtime = Path(command[-1])
        executions.append(runtime.stem)
        _write_state(tmp_path, runtime, cost=0.884736)
        return subprocess.CompletedProcess(command, 0, "", "")

    args = _args(tmp_path, manifest, ceiling=0.884736)
    args.max_input_usd_per_million = 1.0
    args.max_output_usd_per_million = 1.0
    assert _run_canary(args, execute=execute) == runner.EXIT_BUDGET_EXHAUSTED
    status = json.loads((tmp_path / "reports/status.json").read_text(encoding="utf-8"))
    assert executions == ["1-case-1"]
    assert [row["status"] for row in status["cases"]] == [
        "completed",
        "not_started_hard_ceiling",
    ]


def test_exact_ceiling_after_final_selected_case_is_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1)])

    def execute(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        _write_state(tmp_path, Path(command[-1]), cost=0.884736)
        return subprocess.CompletedProcess(command, 0, "", "")

    args = _args(tmp_path, manifest, ceiling=0.884736)
    args.max_input_usd_per_million = 1.0
    args.max_output_usd_per_million = 1.0
    assert _run_canary(args, execute=execute) == runner.EXIT_SUCCESS
    status = json.loads((tmp_path / "reports/status.json").read_text(encoding="utf-8"))
    assert [row["status"] for row in status["cases"]] == ["completed"]


def test_saved_partial_cost_is_charged_once_when_checkpoint_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1)])
    output = tmp_path / "reports"
    output.mkdir()
    saved = {
        "schema": "phase1-v44-canary-status-v1",
        "cost_ceiling_usd": 1.0,
        "per_case_reserve_usd": 0.1,
        "cumulative_cost_usd": 0.04,
        "cases": [
            {
                "position": 1,
                "vc_slug": "charles-hudson",
                "episode_slug": "1-case-1",
                "actual_decision": "In",
                "role": "coverage",
                "status": "failed_with_state",
                "phase1_status": "running",
                "calls": 1,
                "input_tokens": 10,
                "output_tokens": 2,
                "cost_usd": 0.04,
                "cumulative_cost_usd": 0.04,
                "elapsed_seconds": 1.0,
                "state_path": "state.json",
                "error": "interrupted",
            }
        ],
    }
    (output / "status.json").write_text(json.dumps(saved), encoding="utf-8")
    runtime = output / "configs/charles-hudson/1-case-1.toml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(
        runner.render_case_config(
            (tmp_path / "configs/charles-hudson.toml").read_text(encoding="utf-8"),
            episode_slug="1-case-1",
        ),
        encoding="utf-8",
    )
    _write_checkpoint(tmp_path, runtime, cost=0.04)

    def execute(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        _write_state(tmp_path, Path(command[-1]), cost=0.09)
        return subprocess.CompletedProcess(command, 0, "", "")

    _run_canary(_args(tmp_path, manifest), execute=execute)
    result = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert result["cumulative_cost_usd"] == pytest.approx(0.09)


def test_interrupted_checkpoint_cost_is_recovered_before_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1), _case(2)])
    runtime = tmp_path / "reports/configs/charles-hudson/1-case-1.toml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(
        runner.render_case_config(
            (tmp_path / "configs/charles-hudson.toml").read_text(encoding="utf-8"),
            episode_slug="1-case-1",
        ),
        encoding="utf-8",
    )
    _write_checkpoint(tmp_path, runtime, cost=0.18)
    commands: list[str] = []

    def execute(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        commands.append(command[3])
        _write_state(tmp_path, Path(command[-1]), cost=0.20)
        return subprocess.CompletedProcess(command, 0, "", "")

    args = _args(tmp_path, manifest, ceiling=0.25, reserve=0.10)
    args.max_input_usd_per_million = 0.25
    args.max_output_usd_per_million = 0.25
    assert _run_canary(
        args,
        execute=execute,
    ) == runner.EXIT_BUDGET_EXHAUSTED
    status = json.loads((tmp_path / "reports/status.json").read_text(encoding="utf-8"))
    assert commands == []
    assert status["cumulative_cost_usd"] == pytest.approx(0.18)
    assert status["cases"][0]["status"] == "not_started_hard_ceiling"


def test_unreconciled_checkpoint_usage_fails_closed_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1)])
    runtime = tmp_path / "reports/configs/charles-hudson/1-case-1.toml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(
        runner.render_case_config(
            (tmp_path / "configs/charles-hudson.toml").read_text(encoding="utf-8"),
            episode_slug="1-case-1",
        ),
        encoding="utf-8",
    )
    _write_checkpoint(tmp_path, runtime, cost=0.08, reconciled=False)
    called = False

    def execute(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unreconciled cost reached execution")

    with pytest.raises(ValueError, match="reconcile"):
        _run_canary(_args(tmp_path, manifest), execute=execute)
    assert called is False


def test_responded_wal_without_checkpointed_call_never_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langgraph.checkpoint.base import empty_checkpoint
    from langgraph.checkpoint.sqlite import SqliteSaver
    from vc_clone_graph.config import load_config

    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1)])
    runtime = tmp_path / "reports/configs/charles-hudson/1-case-1.toml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(
        runner.render_case_config(
            (tmp_path / "configs/charles-hudson.toml").read_text(encoding="utf-8"),
            episode_slug="1-case-1",
        ),
        encoding="utf-8",
    )
    config = load_config(runtime)
    checkpoint_path = tmp_path / config.run.checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    thread_id = f"{config.run.vc_slug}:{config.run.episode_slug}"
    checkpoint_state = {
        "phase1_status": "running",
        "phase2_status": "not_run",
        "usage": {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        },
        "call_records": [],
        "pending_call": {"intent_sha256": "c" * 64},
        "pending_response": None,
    }
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        value = empty_checkpoint()
        value["channel_values"] = checkpoint_state
        saver.put(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
            value,
            {"source": "input", "step": 0, "parents": {}},
            {},
        )
    wal = (
        tmp_path
        / config.run.output_root
        / config.run.episode_slug
        / ".phase1-v44-wal"
        / f"{'d' * 64}.json"
    )
    wal.parent.mkdir(parents=True, exist_ok=True)
    wal.write_text(
        json.dumps(
            {
                "schema_version": "phase1-provider-call-wal-v1",
                "wal_id": "d" * 64,
                "status": "responded_uncheckpointed",
                "uncertain_usage": {
                    "input_tokens": 20,
                    "cached_input_tokens": 0,
                    "output_tokens": 5,
                    "cost_usd": 0.08,
                },
            }
        ),
        encoding="utf-8",
    )
    called = False

    def execute(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("uncertain paid response was retried")

    with pytest.raises(ValueError, match="cannot be reconciled"):
        _run_canary(_args(tmp_path, manifest), execute=execute)
    assert called is False


def test_returncode_zero_with_failed_phase1_is_not_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1)])

    def execute(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        _write_state(
            tmp_path,
            Path(command[-1]),
            phase1_status="failed",
            cost=0.0,
            phases=(),
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    _run_canary(_args(tmp_path, manifest), execute=execute)
    status = json.loads((tmp_path / "reports/status.json").read_text(encoding="utf-8"))
    assert status["cases"][0]["status"] == "failed_with_state"


def test_runner_allows_only_one_persisted_resume_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1)])
    runtime = tmp_path / "reports/configs/charles-hudson/1-case-1.toml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(
        runner.render_case_config(
            (tmp_path / "configs/charles-hudson.toml").read_text(encoding="utf-8"),
            episode_slug="1-case-1",
        ),
        encoding="utf-8",
    )
    _write_checkpoint(tmp_path, runtime, cost=0.02)
    calls = 0

    def execute(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 1, "", "interrupted")

    _run_canary(_args(tmp_path, manifest), execute=execute)
    _run_canary(_args(tmp_path, manifest), execute=execute)
    status = json.loads((tmp_path / "reports/status.json").read_text(encoding="utf-8"))
    assert calls == 1
    assert status["cases"][0]["resume_attempts"] == 1
    assert status["cases"][0]["status"] == "resume_exhausted"


def test_limited_resume_preserves_unvisited_canonical_rows_and_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1), _case(2), _case(3)])
    executions: list[str] = []

    def execute(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        runtime = Path(command[-1])
        executions.append(runtime.stem)
        _write_state(tmp_path, runtime, cost=0.03)
        return subprocess.CompletedProcess(command, 0, "", "")

    first_args = _args(tmp_path, manifest)
    first_args.limit = 3
    _run_canary(first_args, execute=execute)
    second_args = _args(tmp_path, manifest)
    second_args.limit = 1
    _run_canary(second_args, execute=execute)
    status = json.loads((tmp_path / "reports/status.json").read_text(encoding="utf-8"))
    assert len(status["cases"]) == 3
    assert [row["episode_slug"] for row in status["cases"]] == [
        "1-case-1",
        "2-case-2",
        "3-case-3",
    ]
    assert status["cumulative_cost_usd"] == pytest.approx(0.09)
    assert executions == ["1-case-1", "2-case-2", "3-case-3"]


def test_authoritative_json_attempt_is_persisted_before_csv_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = [_case(1)]
    rows = {
        ("charles-hudson", "1-case-1"): {
            **runner._base_row(1, cases[0]),
            "status": "resuming",
            "attempts": 2,
            "resume_attempts": 1,
        }
    }

    def fail_csv(*_args, **_kwargs) -> None:
        raise OSError("simulated CSV replace failure")

    monkeypatch.setattr(runner, "_write_csv", fail_csv)
    with pytest.raises(OSError, match="CSV replace"):
        runner._persist_status(
            tmp_path,
            rows,
            cases,
            ceiling=1.0,
            reserve=0.1,
            cumulative=0.0,
        )
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["cases"][0]["resume_attempts"] == 1


@pytest.mark.parametrize("bad_cost", [math.nan, math.inf, -0.01])
def test_nonfinite_or_negative_authoritative_state_cost_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_cost: float
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1)])
    runtime = tmp_path / "reports/configs/charles-hudson/1-case-1.toml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(
        runner.render_case_config(
            (tmp_path / "configs/charles-hudson.toml").read_text(encoding="utf-8"),
            episode_slug="1-case-1",
        ),
        encoding="utf-8",
    )
    _write_checkpoint(tmp_path, runtime, cost=bad_cost)
    with pytest.raises(ValueError, match="finite nonnegative"):
        _run_canary(_args(tmp_path, manifest))


@pytest.mark.parametrize("bad_cost", [math.nan, math.inf, -0.01])
def test_invalid_saved_case_or_aggregate_cost_fails_before_execution(
    tmp_path: Path, bad_cost: float
) -> None:
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1)])
    output = tmp_path / "reports"
    output.mkdir()
    row = {
        **runner._base_row(1, _case(1)),
        "cost_usd": bad_cost,
        "cumulative_cost_usd": bad_cost,
    }
    (output / "status.json").write_text(
        json.dumps(
            {
                "schema": "phase1-v44-canary-status-v1",
                "cumulative_cost_usd": bad_cost,
                "cases": [row],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="saved canary status is invalid"):
        _run_canary(_args(tmp_path, manifest))


def test_v44_runner_help_exposes_explicit_dry_run_and_cost_controls() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_phase1_v44_canary.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    for option in (
        "--manifest",
        "--config-dir",
        "--output",
        "--cost-ceiling",
        "--per-case-reserve",
        "--max-input-usd-per-million",
        "--max-output-usd-per-million",
        "--limit",
        "--dry-run",
    ):
        assert option in result.stdout
    for documented_exit in (
        "0  all selected cases completed",
        "1  structural or workflow failure",
        "2  hard cost ceiling exhausted",
        "3  another canary runner is active",
    ):
        assert documented_exit in result.stdout


def test_hard_ceiling_blocks_before_synthetic_over_reserve_case_can_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1)])
    args = _args(tmp_path, manifest, ceiling=0.25, reserve=0.01)
    args.max_input_usd_per_million = 1.0
    args.max_output_usd_per_million = 4.0
    called = False

    def execute(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("case exceeding guaranteed headroom was started")

    assert _run_canary(args, execute=execute) == runner.EXIT_BUDGET_EXHAUSTED
    assert called is False
    status = json.loads((tmp_path / "reports/status.json").read_text(encoding="utf-8"))
    row = status["cases"][0]
    assert row["status"] == "not_started_hard_ceiling"
    assert row["remaining_guaranteed_cap_usd"] > 0.25
    assert status["rate_ceiling_assertion"]["conditional_guarantee"] is True


@pytest.mark.parametrize(
    ("input_rate", "output_rate"),
    [(None, 1.0), (1.0, None), (0.0, 1.0), (math.inf, 1.0)],
)
def test_paid_mode_requires_finite_positive_asserted_rate_ceilings(
    tmp_path: Path, input_rate: float | None, output_rate: float | None
) -> None:
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1)])
    args = _args(tmp_path, manifest)
    args.max_input_usd_per_million = input_rate
    args.max_output_usd_per_million = output_rate
    with pytest.raises(ValueError, match="true upper bound"):
        _run_canary(args)


@pytest.mark.parametrize(
    "episode_slug",
    ["../../escape", '1-good\"\ncheckpoint_path = \"escape', "Bad-Slug"],
)
def test_manifest_episode_slug_rejects_path_and_toml_injection(
    tmp_path: Path, episode_slug: str
) -> None:
    case = _case(1)
    case["episode_slug"] = episode_slug
    manifest = _manifest(tmp_path, [case])
    with pytest.raises(ValueError, match="episode_slug"):
        runner.load_cases(manifest)


def test_output_and_config_paths_must_resolve_beneath_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(runner, "PROJECT_ROOT", project)
    config = _base_config(project)
    manifest = _manifest(project, [_case(1)])
    args = _args(project, manifest)
    args.output = tmp_path / "outside"
    with pytest.raises(ValueError, match="beneath project root"):
        _run_canary(args)
    assert config.is_file()
    assert not args.output.exists()


def test_all_selected_preflights_finish_before_any_model_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1), _case(2)])
    preflighted: list[str] = []
    model_called = False

    def preflight(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        target = Path(command[-1]).stem
        preflighted.append(target)
        if target == "2-case-2":
            return subprocess.CompletedProcess(command, 1, "", "not ready")
        return _ready_preflight(command, **kwargs)

    def execute(*_args, **_kwargs):
        nonlocal model_called
        model_called = True
        raise AssertionError("model started before batch preflight completed")

    with pytest.raises(ValueError, match="preflight"):
        _run_canary(
            _args(tmp_path, manifest),
            execute=execute,
            preflight_execute=preflight,
        )
    assert preflighted == ["1-case-1", "2-case-2"]
    assert model_called is False


def test_preflight_rejects_wrong_taxonomy_embedding_dimension() -> None:
    result = _ready_preflight(
        [
            sys.executable,
            "-m",
            "vc_clone_graph.cli",
            "preflight",
            "--config",
            str(ROOT / "configs/v44/charles-hudson.toml"),
        ]
    )
    payload = json.loads(result.stdout)
    payload["taxonomy_embedding_preflight"]["dimension"] = 384
    tampered = subprocess.CompletedProcess(
        result.args,
        0,
        json.dumps(payload),
        "",
    )

    with pytest.raises(ValueError, match="preflight contract"):
        runner._parse_preflight(
            tampered,
            episode_slug="39-this-pitch-is-damn-near-perfect",
        )


def test_rendered_config_tamper_fails_before_preflight_or_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    config = _base_config(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            'model = "openai/gpt-5.6-luna"',
            'model = "unapproved/model"',
            1,
        ),
        encoding="utf-8",
    )
    manifest = _manifest(tmp_path, [_case(1)])
    preflighted = False
    model_called = False

    def preflight(*_args, **_kwargs):
        nonlocal preflighted
        preflighted = True
        raise AssertionError

    def execute(*_args, **_kwargs):
        nonlocal model_called
        model_called = True
        raise AssertionError

    with pytest.raises(ValueError, match="canonical v4.4 config projection"):
        _run_canary(
            _args(tmp_path, manifest),
            execute=execute,
            preflight_execute=preflight,
        )
    assert preflighted is model_called is False


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        (
            '[phase2]\nmodel = "openai/gpt-5.6-luna"',
            '[phase2]\nmodel = "unapproved/phase2-model"',
        ),
        ("taxonomy_top_k = 5", "taxonomy_top_k = 6"),
    ],
)
def test_complete_canonical_config_projection_rejects_any_field_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    needle: str,
    replacement: str,
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    config = _base_config(tmp_path)
    mutated = config.read_text(encoding="utf-8").replace(needle, replacement)
    assert mutated != config.read_text(encoding="utf-8")
    config.write_text(mutated, encoding="utf-8")
    manifest = _manifest(tmp_path, [_case(1)])
    preflighted = False
    model_called = False

    def preflight(*_args, **_kwargs):
        nonlocal preflighted
        preflighted = True
        raise AssertionError

    def execute(*_args, **_kwargs):
        nonlocal model_called
        model_called = True
        raise AssertionError

    with pytest.raises(ValueError, match="canonical v4.4 config projection"):
        _run_canary(
            _args(tmp_path, manifest),
            execute=execute,
            preflight_execute=preflight,
        )
    assert preflighted is model_called is False


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        ("context_window = 131072", "context_window = 65536"),
        ("max_output_tokens = 16384", "max_output_tokens = 8192"),
    ],
)
def test_rate_cap_token_limits_are_exact_and_cannot_be_lowered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    needle: str,
    replacement: str,
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    config = _base_config(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(needle, replacement),
        encoding="utf-8",
    )
    manifest = _manifest(tmp_path, [_case(1)])

    with pytest.raises(ValueError, match="canonical v4.4 config projection"):
        _run_canary(_args(tmp_path, manifest))


def test_concurrent_runner_fails_without_config_status_or_model_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1)])
    called = False

    def execute(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    with runner._batch_lock():
        assert _run_canary(_args(tmp_path, manifest), execute=execute) == (
            runner.EXIT_RUNNER_BUSY
        )
    assert called is False
    assert not (tmp_path / "reports").exists()


def test_structural_workflow_failure_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _base_config(tmp_path)
    manifest = _manifest(tmp_path, [_case(1)])

    def execute(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "structural failure")

    assert _run_canary(_args(tmp_path, manifest), execute=execute) == (
        runner.EXIT_FAILURE
    )
