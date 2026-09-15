#!/usr/bin/env python3
"""Run the frozen Phase 1 v4.4 canary sequentially under an explicit ceiling."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from time import perf_counter
import tomllib
from typing import Callable, Sequence

import fcntl

from langgraph.checkpoint.sqlite import SqliteSaver

from vc_clone_graph.config import RunConfig, load_config


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_BUDGET_EXHAUSTED = 2
EXIT_RUNNER_BUSY = 3
DEFAULT_MANIFEST = Path(
    "reports/evaluation/phase1-v43-diagnostic-2026-08-17/canary-manifest.json"
)
DEFAULT_OUTPUT = Path("reports/evaluation/phase1-v44-canary-2026-08-17")
CONFIG_NAMES = {
    "charles-hudson": "charles-hudson.toml",
    "cyan-banister": "cyan-banister.toml",
    "elizabeth-yin": "elizabeth-yin.toml",
    "jesse-middleton": "jesse-middleton.toml",
    "jillian-manus": "jillian-manus.toml",
    "phil-nadel": "phil-nadel.toml",
}
CALL_PHASES = frozenset(
    {
        "phase1_claim_extraction",
        "phase1_claim_extraction_repair",
        "phase1_adjudication",
        "phase1_adjudication_repair",
    }
)
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
STATUS_FIELDS = (
    "position",
    "vc_slug",
    "episode_slug",
    "actual_decision",
    "role",
    "status",
    "phase1_status",
    "attempts",
    "resume_attempts",
    "calls",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "per_case_estimate_usd",
    "guaranteed_case_cap_usd",
    "remaining_guaranteed_cap_usd",
    "max_input_usd_per_million",
    "max_output_usd_per_million",
    "cumulative_cost_usd",
    "elapsed_seconds",
    "state_path",
    "error",
)


@dataclass(frozen=True)
class CaseAccounting:
    cost_usd: float = 0.0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    phase1_status: str = ""
    phase2_status: str = ""
    source: str = "none"
    call_phases: tuple[str, ...] = ()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Exit codes:
  0  all selected cases completed (or dry-run succeeded)
  1  structural or workflow failure
  2  hard cost ceiling exhausted
  3  another canary runner is active""",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config-dir", type=Path, default=Path("configs/v44"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cost-ceiling", type=float)
    parser.add_argument(
        "--per-case-estimate",
        "--per-case-reserve",
        dest="per_case_estimate",
        type=float,
        default=0.20,
        help="non-binding expected cost per case; the hard cap uses asserted rates",
    )
    parser.add_argument("--max-input-usd-per-million", type=float)
    parser.add_argument("--max-output-usd-per-million", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _path(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def _confined(path: Path, root: Path, *, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} must resolve beneath project root")
    return resolved


class _RunnerBusyError(RuntimeError):
    pass


@contextmanager
def _batch_lock():
    root = PROJECT_ROOT.resolve()
    lock_path = root / ".phase1-v44-canary.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ValueError("cannot securely open the v4.4 batch lock") from exc
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("v4.4 batch lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as exc:
            raise _RunnerBusyError("another v4.4 canary runner is active") from exc
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=STATUS_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_cases(path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = payload["cases"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot load canary manifest: {path}") from exc
    if not isinstance(cases, list) or not cases:
        raise ValueError("canary manifest cases must be a nonempty list")
    result: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for raw in cases:
        if not isinstance(raw, dict):
            raise ValueError("canary manifest case must be an object")
        required = ("vc_slug", "episode_slug", "actual_decision", "role")
        if any(not isinstance(raw.get(key), str) or not raw[key] for key in required):
            raise ValueError("canary manifest case is incomplete")
        vc_slug = str(raw["vc_slug"])
        episode_slug = str(raw["episode_slug"])
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", episode_slug) is None:
            raise ValueError("canary manifest episode_slug is invalid")
        if vc_slug not in CONFIG_NAMES:
            raise ValueError(f"no v4.4 config for investor: {vc_slug}")
        identity = (vc_slug, episode_slug)
        if identity in seen:
            raise ValueError(f"duplicate canary case: {vc_slug}/{episode_slug}")
        seen.add(identity)
        result.append(dict(raw))
    return result


def render_case_config(base: str, *, episode_slug: str) -> str:
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", episode_slug) is None:
        raise ValueError("episode_slug is invalid")
    checkpoint = (
        "outputs/phase1-v44-canary-2026-08-17/checkpoints/"
        f"{episode_slug}.sqlite"
    )
    rendered, episode_count = re.subn(
        r'(?m)^episode_slug\s*=\s*"[^"]+"$',
        f"episode_slug = {json.dumps(episode_slug)}",
        base,
        count=1,
    )
    rendered, checkpoint_count = re.subn(
        r'(?m)^checkpoint_path\s*=\s*"[^"]+"$',
        f"checkpoint_path = {json.dumps(checkpoint)}",
        rendered,
        count=1,
    )
    if episode_count != 1 or checkpoint_count != 1:
        raise ValueError("base config lacks unique episode or checkpoint setting")
    return rendered


def existing_case_mode(state_path: Path, checkpoint_path: Path) -> str:
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        if (
            state.get("phase1_status") in {"accepted", "provisional"}
            and state.get("phase2_status") == "not_run"
        ):
            return "skip"
    if checkpoint_path.is_file():
        return "resume"
    return "run"


def _finite_nonnegative(value: object, *, field: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{field} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return number


def _token_count(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _usage(value: object, *, context: str) -> dict[str, int | float]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} usage cannot be reconciled")
    try:
        return {
            "input_tokens": _token_count(
                value["input_tokens"], field=f"{context} input_tokens"
            ),
            "cached_input_tokens": _token_count(
                value.get("cached_input_tokens", 0),
                field=f"{context} cached_input_tokens",
            ),
            "output_tokens": _token_count(
                value["output_tokens"], field=f"{context} output_tokens"
            ),
            "cost_usd": _finite_nonnegative(
                value["cost_usd"], field=f"{context} cost_usd"
            ),
        }
    except KeyError as exc:
        raise ValueError(f"{context} usage cannot be reconciled") from exc


def count_v44_calls(state: dict[str, object]) -> int:
    records = state.get("call_records", [])
    if not isinstance(records, list):
        return 0
    return sum(
        1
        for row in records
        if isinstance(row, dict)
        and isinstance(row.get("payload"), dict)
        and row["payload"].get("phase") in CALL_PHASES
    )


def _account_state(
    state: dict[str, object],
    *,
    run_root: Path,
    source: str,
) -> CaseAccounting:
    records = state.get("call_records")
    if not isinstance(records, list):
        raise ValueError(f"{source} provider call ledger cannot be reconciled")
    totals: dict[str, int | float] = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "per_case_estimate_usd": 0.0,
        "guaranteed_case_cap_usd": 0.0,
        "remaining_guaranteed_cap_usd": 0.0,
        "max_input_usd_per_million": 0.0,
        "max_output_usd_per_million": 0.0,
    }
    records_by_wal: dict[str, dict[str, object]] = {}
    seen_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{source} provider call ledger cannot be reconciled")
        payload = record.get("payload")
        relative = record.get("relative_path")
        digest = record.get("sha256")
        wal_id = record.get("wal_id")
        request_digest = record.get("request_sha256")
        if (
            not isinstance(payload, dict)
            or not isinstance(relative, str)
            or not isinstance(digest, str)
            or not isinstance(wal_id, str)
            or not re.fullmatch(r"[0-9a-f]{64}", wal_id)
            or not isinstance(request_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", request_digest)
            or not re.fullmatch(
                r"phase1/(?:claim-extraction|adjudication/turn-[0-9]{2})/"
                r"call-[0-9]{2}\.json",
                relative,
            )
            or relative in seen_paths
            or wal_id in records_by_wal
        ):
            raise ValueError(f"{source} provider call ledger cannot be reconciled")
        seen_paths.add(relative)
        records_by_wal[wal_id] = record
        call_path = run_root / relative
        try:
            if not call_path.is_file():
                raise ValueError
            raw = call_path.read_bytes()
            persisted_payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{source} provider call artifact cannot be reconciled"
            ) from exc
        call_usage = _usage(payload.get("usage"), context=f"{source} call")
        if _finite_nonnegative(
            payload.get("cost_usd"), field=f"{source} call cost_usd"
        ) != call_usage["cost_usd"]:
            raise ValueError(f"{source} provider call cost cannot be reconciled")
        if (
            sha256(raw).hexdigest() != digest
            or persisted_payload != payload
        ):
            raise ValueError(f"{source} provider call ledger cannot be reconciled")
        phase = payload.get("phase")
        if phase not in CALL_PHASES:
            raise ValueError(f"{source} provider call phase cannot be reconciled")
        for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
            totals[key] = int(totals[key]) + int(call_usage[key])
        totals["cost_usd"] = float(totals["cost_usd"]) + float(
            call_usage["cost_usd"]
        )

    reported = _usage(state.get("usage"), context=source)
    for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
        if reported[key] != totals[key]:
            raise ValueError(f"{source} provider usage cannot be reconciled")
    if float(reported["cost_usd"]) != float(totals["cost_usd"]):
        raise ValueError(f"{source} provider cost cannot be reconciled")

    wal_root = run_root / ".phase1-v44-wal"
    wal_paths = sorted(wal_root.glob("*.json")) if wal_root.is_dir() else []
    for wal_path in wal_paths:
        try:
            wal = json.loads(wal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{source} provider WAL cannot be reconciled") from exc
        if not isinstance(wal, dict) or wal.get("schema_version") != (
            "phase1-provider-call-wal-v1"
        ):
            raise ValueError(f"{source} provider WAL cannot be reconciled")
        wal_id = wal.get("wal_id")
        record = records_by_wal.get(str(wal_id))
        if (
            wal_path.stem != wal_id
            or wal.get("status") != "responded_uncheckpointed"
            or record is None
        ):
            raise ValueError(
                f"{source} provider outcome/cost cannot be reconciled; "
                "manual review required before retry"
            )
        wal_usage = _usage(wal.get("uncertain_usage"), context=f"{source} WAL")
        record_usage = _usage(record["payload"].get("usage"), context=f"{source} call")
        if wal_usage != record_usage:
            raise ValueError(f"{source} provider WAL cost cannot be reconciled")

    pending_call = state.get("pending_call")
    pending_response = state.get("pending_response")
    if pending_call is not None and pending_response is None:
        raise ValueError(
            f"{source} provider outcome/cost cannot be reconciled; "
            "manual review required before retry"
        )
    if pending_response is not None and pending_call is None:
        raise ValueError(f"{source} pending provider response cannot be reconciled")

    return CaseAccounting(
        cost_usd=float(reported["cost_usd"]),
        input_tokens=int(reported["input_tokens"]),
        cached_input_tokens=int(reported["cached_input_tokens"]),
        output_tokens=int(reported["output_tokens"]),
        calls=len(records),
        phase1_status=str(state.get("phase1_status", "")),
        phase2_status=str(state.get("phase2_status", "")),
        source=source,
        call_phases=tuple(
            str(record["payload"]["phase"])
            for record in records
            if isinstance(record, dict) and isinstance(record.get("payload"), dict)
        ),
    )


def _checkpoint_state(checkpoint_path: Path, *, thread_id: str) -> dict[str, object]:
    try:
        with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            item = saver.get_tuple(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": "",
                    }
                }
            )
    except Exception as exc:
        raise ValueError("checkpoint provider usage cannot be reconciled") from exc
    if item is None:
        raise ValueError("checkpoint provider usage cannot be reconciled")
    state = item.checkpoint.get("channel_values")
    if not isinstance(state, dict):
        raise ValueError("checkpoint provider usage cannot be reconciled")
    return state


def authoritative_accounting(config_path: Path) -> CaseAccounting:
    config = load_config(config_path)
    run_root = (
        PROJECT_ROOT / config.run.output_root / config.run.episode_slug
    )
    state_path = run_root / "state.json"
    checkpoint_path = PROJECT_ROOT / config.run.checkpoint_path
    terminal: CaseAccounting | None = None
    checkpoint: CaseAccounting | None = None
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("terminal provider usage cannot be reconciled") from exc
        if not isinstance(state, dict):
            raise ValueError("terminal provider usage cannot be reconciled")
        terminal = _account_state(state, run_root=run_root, source="terminal")
    if checkpoint_path.is_file():
        thread_id = f"{config.run.vc_slug}:{config.run.episode_slug}"
        checkpoint_state = _checkpoint_state(checkpoint_path, thread_id=thread_id)
        checkpoint = _account_state(
            checkpoint_state,
            run_root=run_root,
            source="checkpoint",
        )
    if terminal is not None and checkpoint is not None and terminal != CaseAccounting(
        **{**checkpoint.__dict__, "source": "terminal"}
    ):
        raise ValueError("terminal and checkpoint provider costs cannot be reconciled")
    return terminal or checkpoint or CaseAccounting()


def _validate_runtime_config(
    config_path: Path,
    *,
    case: dict[str, object],
    output: Path,
) -> object:
    runtime_root = (output / "configs").resolve()
    resolved_config = config_path.resolve()
    if not resolved_config.is_relative_to(runtime_root):
        raise ValueError("runtime config must resolve beneath the intended config root")
    config = load_config(config_path)
    alias = str(case["vc_slug"])
    episode_slug = str(case["episode_slug"])
    canonical_path = SOURCE_ROOT / "configs/v44" / CONFIG_NAMES[alias]
    try:
        canonical_rendered = render_case_config(
            canonical_path.read_text(encoding="utf-8"),
            episode_slug=episode_slug,
        )
        canonical = RunConfig.model_validate(tomllib.loads(canonical_rendered))
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        raise ValueError(
            f"cannot construct canonical v4.4 config projection for {alias}"
        ) from exc
    if config.model_dump(mode="json") != canonical.model_dump(mode="json"):
        raise ValueError(
            f"runtime config differs from canonical v4.4 config projection for {alias}"
        )
    intended_output = (
        PROJECT_ROOT / "outputs/phase1-v44-canary-2026-08-17"
    ).resolve()
    resolved_run_output = (PROJECT_ROOT / config.run.output_root).resolve()
    resolved_checkpoint = (PROJECT_ROOT / config.run.checkpoint_path).resolve()
    if not resolved_run_output.is_relative_to(intended_output) or not (
        resolved_checkpoint.is_relative_to(intended_output)
    ):
        raise ValueError("runtime output paths escape the intended v4.4 output root")
    return config


def _parse_preflight(
    result: subprocess.CompletedProcess[str],
    *,
    episode_slug: str,
) -> dict[str, object]:
    if result.returncode != 0:
        raise ValueError(
            f"local preflight failed for {episode_slug}: {result.stderr[-1000:]}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    try:
        report = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ValueError(f"local preflight output is invalid for {episode_slug}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"local preflight output is invalid for {episode_slug}")
    indexes = (
        report.get("wiki_embedding_index"),
        report.get("precedent_embedding_index"),
    )
    taxonomy = report.get("taxonomy_embedding_preflight")
    if (
        report.get("status") != "ready"
        or report.get("target") != episode_slug
        or report.get("target_accessible") is not False
        or report.get("phase2_model") is not None
        or not all(
            isinstance(index, dict)
            and index.get("model") == EMBEDDING_MODEL
            and index.get("revision") == EMBEDDING_REVISION
            and index.get("coverage") == 1.0
            for index in indexes
        )
        or not isinstance(taxonomy, dict)
        or taxonomy.get("model") != EMBEDDING_MODEL
        or taxonomy.get("revision") != EMBEDDING_REVISION
        or taxonomy.get("dimension") != 768
        or not isinstance(report.get("portfolio_memory"), dict)
        or report["portfolio_memory"].get("target_accessible") is not False
    ):
        raise ValueError(f"local preflight contract failed for {episode_slug}")
    return report


def _run_preflight(
    runtime_path: Path,
    *,
    episode_slug: str,
    execute: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, object]:
    result = execute(
        [
            sys.executable,
            "-m",
            "vc_clone_graph.cli",
            "preflight",
            "--config",
            str(runtime_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return _parse_preflight(result, episode_slug=episode_slug)


def _call_limits(config: object) -> dict[str, int]:
    revisits = config.phase1_v44.max_revisits
    return {
        "phase1_claim_extraction": 1,
        "phase1_claim_extraction_repair": 1,
        "phase1_adjudication": 1 + revisits,
        "phase1_adjudication_repair": 1 + revisits,
    }


def _cap_for_call_count(
    config: object,
    count: int,
    *,
    input_rate: float,
    output_rate: float,
) -> float:
    maximum_output = max(
        config.provider.max_output_tokens,
        config.phase1.max_output_tokens or 0,
    )
    per_call = (
        config.provider.context_window * input_rate
        + maximum_output * output_rate
    ) / 1_000_000
    return count * per_call


def guaranteed_cost_caps(
    config: object,
    account: CaseAccounting,
    *,
    input_rate: float,
    output_rate: float,
) -> tuple[float, float]:
    limits = _call_limits(config)
    completed = Counter(account.call_phases)
    if any(phase not in limits for phase in completed) or any(
        completed[phase] > limit for phase, limit in limits.items()
    ):
        raise ValueError("authoritative completed calls exceed the v4.4 call graph")
    maximum_calls = sum(limits.values())
    if (
        account.phase1_status in {"accepted", "provisional"}
        and account.phase2_status == "not_run"
    ):
        remaining_calls = 0
    else:
        remaining_calls = sum(
            limit - completed.get(phase, 0) for phase, limit in limits.items()
        )
    return (
        _cap_for_call_count(
            config,
            maximum_calls,
            input_rate=input_rate,
            output_rate=output_rate,
        ),
        _cap_for_call_count(
            config,
            remaining_calls,
            input_rate=input_rate,
            output_rate=output_rate,
        ),
    )


def _read_state(config_path: Path) -> tuple[Path, dict[str, object]]:
    config = load_config(config_path)
    state_path = (
        PROJECT_ROOT / config.run.output_root / config.run.episode_slug / "state.json"
    )
    value = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("v4.4 state must be a JSON object")
    return state_path, value


def _read_saved_status(path: Path) -> tuple[float, dict[tuple[str, str], dict]]:
    if not path.is_file():
        return 0.0, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cumulative = _finite_nonnegative(
            payload["cumulative_cost_usd"], field="saved cumulative_cost_usd"
        )
        cases = payload["cases"]
        if not isinstance(cases, list):
            raise ValueError
        indexed: dict[tuple[str, str], dict] = {}
        summed = 0.0
        for raw in cases:
            if not isinstance(raw, dict):
                raise ValueError
            key = (str(raw["vc_slug"]), str(raw["episode_slug"]))
            if key in indexed:
                raise ValueError
            row = dict(raw)
            row["cost_usd"] = _finite_nonnegative(
                row.get("cost_usd", 0.0), field="saved case cost_usd"
            )
            _finite_nonnegative(
                row.get("cumulative_cost_usd", 0.0),
                field="saved case cumulative_cost_usd",
            )
            row["attempts"] = _token_count(
                row.get("attempts", 0), field="saved attempts"
            )
            row["resume_attempts"] = _token_count(
                row.get("resume_attempts", 0), field="saved resume_attempts"
            )
            if row["resume_attempts"] > row["attempts"]:
                raise ValueError
            summed += float(row["cost_usd"])
            indexed[key] = row
        if not math.isclose(cumulative, summed, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"saved canary status is invalid: {path}") from exc
    return cumulative, indexed


def _base_row(position: int, case: dict[str, object]) -> dict[str, object]:
    return {
        "position": position,
        "vc_slug": case["vc_slug"],
        "episode_slug": case["episode_slug"],
        "actual_decision": case["actual_decision"],
        "role": case["role"],
        "status": "not_started",
        "phase1_status": "",
        "attempts": 0,
        "resume_attempts": 0,
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "cumulative_cost_usd": 0.0,
        "elapsed_seconds": 0.0,
        "state_path": "",
        "error": "",
    }


def _ordered_rows(
    rows: dict[tuple[str, str], dict[str, object]],
    cases: list[dict[str, object]],
) -> list[dict[str, object]]:
    ordered: list[dict[str, object]] = []
    cumulative = 0.0
    for position, case in enumerate(cases, start=1):
        key = (str(case["vc_slug"]), str(case["episode_slug"]))
        row = rows.get(key)
        if row is None:
            continue
        row = {**_base_row(position, case), **row, "position": position}
        cost = _finite_nonnegative(row["cost_usd"], field="reported case cost_usd")
        cumulative += cost
        row["cost_usd"] = cost
        row["cumulative_cost_usd"] = cumulative
        ordered.append(row)
    return ordered


def _total_cost(
    rows: dict[tuple[str, str], dict[str, object]],
    cases: list[dict[str, object]],
) -> float:
    return sum(float(row["cost_usd"]) for row in _ordered_rows(rows, cases))


def _mark_remaining_hard_ceiling(
    rows_by_key: dict[tuple[str, str], dict[str, object]],
    selected_cases: list[dict[str, object]],
    *,
    start_index: int,
) -> None:
    """Mark every remaining runnable selected case without starting a subprocess."""
    for case in selected_cases[start_index:]:
        key = (str(case["vc_slug"]), str(case["episode_slug"]))
        row = rows_by_key[key]
        if str(row.get("status", "")).startswith("completed"):
            continue
        row.update(
            {
                "status": "not_started_hard_ceiling",
                "error": (
                    "remaining guaranteed call cap would exceed the hard ceiling"
                ),
            }
        )
        rows_by_key[key] = row


def _persist_status(
    output: Path,
    rows_by_key: dict[tuple[str, str], dict[str, object]],
    cases: list[dict[str, object]],
    *,
    ceiling: float,
    reserve: float,
    cumulative: float,
    input_rate: float | None = None,
    output_rate: float | None = None,
) -> None:
    rows = _ordered_rows(rows_by_key, cases)
    derived = sum(float(row["cost_usd"]) for row in rows)
    if not math.isclose(derived, cumulative, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("reported cumulative cost cannot be reconciled")
    _write_json(
        output / "status.json",
        {
            "schema": "phase1-v44-canary-status-v1",
            "cost_ceiling_usd": ceiling,
            "per_case_estimate_usd": reserve,
            "cumulative_cost_usd": cumulative,
            "rate_ceiling_assertion": {
                "conditional_guarantee": True,
                "max_input_usd_per_million": input_rate,
                "max_output_usd_per_million": output_rate,
                "statement": (
                    "The hard ceiling is guaranteed only if these user-supplied "
                    "rates are true upper bounds for the provider/model."
                ),
            },
            "cases": rows,
        },
    )
    _write_csv(output / "status.csv", rows)


def _run_canary_locked(
    args: argparse.Namespace,
    *,
    execute: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    preflight_execute: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    root = PROJECT_ROOT.resolve()
    manifest_path = _confined(_path(args.manifest), root, label="manifest")
    output = _confined(_path(args.output), root, label="output")
    config_dir = _confined(_path(args.config_dir), root, label="config directory")
    all_cases = load_cases(manifest_path)
    cases = all_cases
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        cases = cases[: args.limit]
    estimate = _finite_nonnegative(
        getattr(args, "per_case_estimate", getattr(args, "per_case_reserve", 0.20)),
        field="--per-case-estimate",
    )
    if not args.dry_run and args.cost_ceiling is None:
        raise ValueError("paid execution requires an explicit --cost-ceiling")
    ceiling = None
    if args.cost_ceiling is not None:
        ceiling = _finite_nonnegative(
            args.cost_ceiling, field="--cost-ceiling"
        )
        if ceiling == 0:
            raise ValueError("--cost-ceiling must be finite and positive")
    input_rate: float | None = None
    output_rate: float | None = None
    if not args.dry_run:
        try:
            input_rate = _finite_nonnegative(
                getattr(args, "max_input_usd_per_million"),
                field="--max-input-usd-per-million true upper bound",
            )
            output_rate = _finite_nonnegative(
                getattr(args, "max_output_usd_per_million"),
                field="--max-output-usd-per-million true upper bound",
            )
        except (AttributeError, ValueError) as exc:
            raise ValueError(
                "paid execution requires finite positive asserted rates that are "
                "true upper bounds for the provider/model"
            ) from exc
        if input_rate == 0 or output_rate == 0:
            raise ValueError(
                "paid execution requires finite positive asserted rates that are "
                "true upper bounds for the provider/model"
            )

    runtime_configs = output / "configs"
    runtime_paths: dict[tuple[str, str], Path] = {}
    runtime_values: dict[tuple[str, str], object] = {}
    for position, case in enumerate(all_cases, start=1):
        vc_slug = str(case["vc_slug"])
        episode_slug = str(case["episode_slug"])
        base_path = config_dir / CONFIG_NAMES[vc_slug]
        runtime_path = runtime_configs / vc_slug / f"{episode_slug}.toml"
        rendered = render_case_config(
            base_path.read_text(encoding="utf-8"),
            episode_slug=episode_slug,
        )
        _atomic_write_text(runtime_path, rendered)
        key = (vc_slug, episode_slug)
        runtime_paths[key] = runtime_path
        runtime_values[key] = _validate_runtime_config(
            runtime_path,
            case=case,
            output=output,
        )
        if args.dry_run and position <= len(cases):
            print(
                f"{position:02d} {vc_slug} {episode_slug} "
                f"{case['actual_decision']}"
            )
    if args.dry_run:
        return EXIT_SUCCESS

    for case in cases:
        key = (str(case["vc_slug"]), str(case["episode_slug"]))
        _run_preflight(
            runtime_paths[key],
            episode_slug=key[1],
            execute=preflight_execute,
        )

    _, saved_rows = _read_saved_status(output / "status.json")
    manifest_keys = {
        (str(case["vc_slug"]), str(case["episode_slug"])) for case in all_cases
    }
    if not set(saved_rows) <= manifest_keys:
        raise ValueError("saved status contains a case outside the canary manifest")
    rows_by_key: dict[tuple[str, str], dict[str, object]] = {
        key: dict(row) for key, row in saved_rows.items()
    }
    accounts: dict[tuple[str, str], CaseAccounting] = {}
    assert input_rate is not None and output_rate is not None
    for position, case in enumerate(all_cases, start=1):
        key = (str(case["vc_slug"]), str(case["episode_slug"]))
        account = authoritative_accounting(runtime_paths[key])
        accounts[key] = account
        case_cap, remaining_cap = guaranteed_cost_caps(
            runtime_values[key],
            account,
            input_rate=input_rate,
            output_rate=output_rate,
        )
        if account.cost_usd > case_cap + 1e-12:
            raise ValueError(
                f"authoritative cost for {key[1]} exceeds the asserted rate ceiling"
            )
        saved = rows_by_key.get(key, _base_row(position, case))
        saved_cost = _finite_nonnegative(
            saved.get("cost_usd", 0.0), field="saved case cost_usd"
        )
        if account.source == "none" and saved_cost > 0:
            raise ValueError(
                f"saved cost for {key[1]} cannot be reconciled to workflow state"
            )
        if account.source != "none" and saved_cost > account.cost_usd + 1e-12:
            raise ValueError(
                f"saved cost for {key[1]} exceeds authoritative workflow cost"
            )
        if account.source != "none":
            saved.update(
                {
                    "phase1_status": account.phase1_status,
                    "calls": account.calls,
                    "input_tokens": account.input_tokens,
                    "output_tokens": account.output_tokens,
                    "cost_usd": account.cost_usd,
                }
            )
            if key not in rows_by_key:
                saved["status"] = (
                    "completed_existing"
                    if account.phase1_status in {"accepted", "provisional"}
                    and account.phase2_status == "not_run"
                    and account.source == "terminal"
                    else "interrupted_checkpoint"
                )
        saved.update(
            {
                "per_case_estimate_usd": estimate,
                "guaranteed_case_cap_usd": case_cap,
                "remaining_guaranteed_cap_usd": remaining_cap,
                "max_input_usd_per_million": input_rate,
                "max_output_usd_per_million": output_rate,
            }
        )
        rows_by_key[key] = saved
    cumulative_cost = _total_cost(rows_by_key, all_cases)

    assert ceiling is not None
    exit_code = EXIT_SUCCESS
    for position, case in enumerate(cases, start=1):
        vc_slug = str(case["vc_slug"])
        episode_slug = str(case["episode_slug"])
        key = (vc_slug, episode_slug)
        runtime_path = runtime_paths[key]

        config = load_config(runtime_path)
        state_path = (
            PROJECT_ROOT
            / config.run.output_root
            / config.run.episode_slug
            / "state.json"
        )
        checkpoint_path = PROJECT_ROOT / config.run.checkpoint_path
        mode = existing_case_mode(state_path, checkpoint_path)
        row = {**_base_row(position, case), **rows_by_key[key]}
        account = accounts[key]
        if mode == "skip":
            if not (
                account.phase1_status in {"accepted", "provisional"}
                and account.phase2_status == "not_run"
                and account.source == "terminal"
            ):
                raise ValueError("skippable Phase1 state is not authoritative")
            row.update(
                {
                    "status": "completed_existing",
                    "phase1_status": account.phase1_status,
                    "state_path": str(state_path),
                    "error": "",
                }
            )
            rows_by_key[key] = row
            _persist_status(
                output,
                rows_by_key,
                all_cases,
                ceiling=ceiling,
                reserve=estimate,
                cumulative=cumulative_cost,
                input_rate=input_rate,
                output_rate=output_rate,
            )
            continue

        if mode == "resume" and int(row.get("resume_attempts", 0)) >= 1:
            row.update(
                {
                    "status": "resume_exhausted",
                    "phase1_status": account.phase1_status,
                    "error": "one permitted resume attempt was already used",
                }
            )
            rows_by_key[key] = row
            _persist_status(
                output,
                rows_by_key,
                all_cases,
                ceiling=ceiling,
                reserve=estimate,
                cumulative=cumulative_cost,
                input_rate=input_rate,
                output_rate=output_rate,
            )
            exit_code = EXIT_FAILURE
            break

        remaining_cap = float(row["remaining_guaranteed_cap_usd"])
        if cumulative_cost + remaining_cap > ceiling + 1e-12:
            _mark_remaining_hard_ceiling(
                rows_by_key,
                cases,
                start_index=position - 1,
            )
            _persist_status(
                output,
                rows_by_key,
                all_cases,
                ceiling=ceiling,
                reserve=estimate,
                cumulative=cumulative_cost,
                input_rate=input_rate,
                output_rate=output_rate,
            )
            exit_code = EXIT_BUDGET_EXHAUSTED
            break

        started = perf_counter()
        row["attempts"] = int(row.get("attempts", 0)) + 1
        if mode == "resume":
            row["resume_attempts"] = int(row.get("resume_attempts", 0)) + 1
        row.update(
            {
                "status": "resuming" if mode == "resume" else "running",
                "error": "",
            }
        )
        rows_by_key[key] = row
        _persist_status(
            output,
            rows_by_key,
            all_cases,
            ceiling=ceiling,
            reserve=estimate,
            cumulative=cumulative_cost,
            input_rate=input_rate,
            output_rate=output_rate,
        )
        result = execute(
            [
                sys.executable,
                "-m",
                "vc_clone_graph.cli",
                mode,
                "--config",
                str(runtime_path),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        post = authoritative_accounting(runtime_path)
        if post.cost_usd + 1e-12 < account.cost_usd:
            raise ValueError("post-run provider cost cannot be reconciled")
        cumulative_cost += post.cost_usd - account.cost_usd
        accounts[key] = post
        case_cap, remaining_cap = guaranteed_cost_caps(
            runtime_values[key],
            post,
            input_rate=input_rate,
            output_rate=output_rate,
        )
        terminal = (
            post.source == "terminal"
            and post.phase1_status in {"accepted", "provisional"}
            and post.phase2_status == "not_run"
        )
        if result.returncode == 0 and terminal:
            status = "completed"
            error = ""
        else:
            status = "failed_with_state" if post.source != "none" else "failed"
            error = result.stderr[-1000:] or (
                "workflow did not produce accepted/provisional Phase1-only state"
            )
        if post.cost_usd > case_cap + 1e-12:
            status = "failed_rate_ceiling_breach"
            error = "observed cost exceeds the asserted provider/model rate ceiling"
        elif cumulative_cost > ceiling + 1e-12:
            status = "failed_cost_ceiling_exceeded"
            error = "actual provider cost exceeded the conditional hard ceiling"
        row.update(
            {
                "status": status,
                "phase1_status": post.phase1_status,
                "calls": post.calls,
                "input_tokens": post.input_tokens,
                "output_tokens": post.output_tokens,
                "cost_usd": post.cost_usd,
                "guaranteed_case_cap_usd": case_cap,
                "remaining_guaranteed_cap_usd": remaining_cap,
                "elapsed_seconds": round(perf_counter() - started, 3),
                "state_path": str(state_path) if state_path.is_file() else "",
                "error": error,
            }
        )
        rows_by_key[key] = row
        _persist_status(
            output,
            rows_by_key,
            all_cases,
            ceiling=ceiling,
            reserve=estimate,
            cumulative=cumulative_cost,
            input_rate=input_rate,
            output_rate=output_rate,
        )
        print(
            f"[{position}/{len(cases)}] {vc_slug} {episode_slug}: {status}; "
            f"phase1={post.phase1_status} cost=${post.cost_usd:.4f} "
            f"total=${cumulative_cost:.4f}",
            flush=True,
        )
        if status.startswith("failed"):
            exit_code = EXIT_FAILURE
            break

    ordered = _ordered_rows(rows_by_key, all_cases)
    completed = sum(
        str(row["status"]).startswith("completed") for row in ordered
    )
    print(
        f"completed={completed}/{len(cases)} "
        f"cumulative_cost_usd={cumulative_cost:.6f}"
    )
    return exit_code


def run_canary(
    args: argparse.Namespace,
    *,
    execute: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    preflight_execute: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    try:
        with _batch_lock():
            return _run_canary_locked(
                args,
                execute=execute,
                preflight_execute=preflight_execute,
            )
    except _RunnerBusyError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_RUNNER_BUSY


def main(argv: Sequence[str] | None = None) -> int:
    return run_canary(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
