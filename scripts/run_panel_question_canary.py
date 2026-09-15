#!/usr/bin/env python3
"""Run the staged panel-question Phase-1 canaries sequentially and resumably."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import traceback
from typing import Any

from vc_clone_graph.artifacts import verify_frozen
from vc_clone_graph.cli import command_run, command_verify
from vc_clone_graph.config import RunConfig
from vc_clone_graph.schemas_v4 import InvestigationV4, InvestigationV41


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _attempt_config(base: RunConfig, experiment_root: Path, attempt: int) -> RunConfig:
    episode = base.run.episode_slug
    vc = base.run.vc_slug
    output_root = experiment_root / "runs" / f"attempt-{attempt}" / vc
    checkpoint = experiment_root / "checkpoints" / f"attempt-{attempt}" / vc / f"{episode}.sqlite"
    run = base.run.model_copy(update={
        "output_root": output_root.as_posix(),
        "checkpoint_path": checkpoint.as_posix(),
    })
    return base.model_copy(update={"run": run})


def _artifact_root(config: RunConfig) -> Path:
    return Path(config.run.output_root) / config.run.episode_slug


def _is_complete(config: RunConfig) -> bool:
    root = _artifact_root(config)
    return (
        (root / "phase1/investigation.json").is_file()
        and (root / "phase1/investigation.sha256").is_file()
        and (root / "summary.json").is_file()
    )


def _verify(config: RunConfig) -> None:
    """Verify the artifact contract actually emitted by each canonical version."""
    if config.run.contract_version in {"v4", "v4.1"}:
        root = _artifact_root(config)
        verify_frozen(
            root / "phase1/investigation.json",
            root / "phase1/investigation.sha256",
        )
        payload = json.loads((root / "phase1/investigation.json").read_text(encoding="utf-8"))
        model = InvestigationV41 if config.run.contract_version == "v4.1" else InvestigationV4
        investigation = model.model_validate(payload)
        if investigation.episode_slug != config.run.episode_slug:
            raise ValueError("Phase 1 episode slug does not match config")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        if summary.get("phase1_status") not in {"accepted", "provisional"}:
            raise ValueError("Phase 1 summary is not accepted or provisional")
        if summary.get("phase2_status") != "not_run":
            raise ValueError("Phase 2 unexpectedly ran in the Phase-1-only experiment")
        return
    command_verify(config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("outputs/panel-question-phase1-canary-2026-08-20"),
    )
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")

    experiment_root = args.experiment_root
    selection = json.loads((experiment_root / "selection.json").read_text(encoding="utf-8"))
    rows = selection["cases"][: args.limit]
    status_path = experiment_root / "batch-status.json"
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.is_file()
        else {"schema": "panel-question-phase1-batch-v1", "cases": {}}
    )

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    for position, row in enumerate(rows, 1):
        key = f"{row['vc_slug']}::{row['episode_slug']}"
        current = status["cases"].get(key, {})
        if current.get("status") == "completed":
            print(f"[{position:02d}/{len(rows):02d}] SKIP completed {key}", flush=True)
            continue

        base = RunConfig.model_validate(json.loads(Path(row["config_path"]).read_text(encoding="utf-8")))
        attempts: list[dict[str, Any]] = list(current.get("attempts", []))
        completed = False
        for attempt in range(1, args.max_attempts + 1):
            config = _attempt_config(base, experiment_root, attempt)
            artifact_root = _artifact_root(config)
            if _is_complete(config):
                try:
                    _verify(config)
                    summary = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))
                    attempts.append({"attempt": attempt, "status": "completed", "artifact_root": artifact_root.as_posix(), "summary": summary})
                    completed = True
                    break
                except Exception:
                    pass

            print(f"[{position:02d}/{len(rows):02d}] RUN attempt={attempt} {key}", flush=True)
            log_path = experiment_root / "logs" / row["vc_slug"] / f"{row['episode_slug']}-attempt-{attempt}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with log_path.open("w", encoding="utf-8") as handle, redirect_stdout(handle):
                    command_run(config, resume=False)
                    _verify(config)
                summary = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))
                attempts.append({"attempt": attempt, "status": "completed", "artifact_root": artifact_root.as_posix(), "log_path": log_path.as_posix(), "summary": summary})
                completed = True
                print(
                    f"[{position:02d}/{len(rows):02d}] OK {key} iterations={summary.get('phase1_iterations')} cost_usd={summary.get('usage', {}).get('cost_usd')}",
                    flush=True,
                )
                break
            except Exception as exc:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write("\n" + traceback.format_exc())
                attempts.append({"attempt": attempt, "status": "failed", "artifact_root": artifact_root.as_posix(), "log_path": log_path.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{position:02d}/{len(rows):02d}] FAIL attempt={attempt} {key}: {exc}", flush=True)

        status["cases"][key] = {
            "vc_slug": row["vc_slug"],
            "episode_slug": row["episode_slug"],
            "actual_decision": row["actual_decision"],
            "insertion_count": row["insertion_count"],
            "status": "completed" if completed else "failed",
            "attempts": attempts,
        }
        _write_json(status_path, status)

    completed_count = sum(row.get("status") == "completed" for row in status["cases"].values())
    failed_count = sum(row.get("status") == "failed" for row in status["cases"].values())
    print(f"DONE completed={completed_count} failed={failed_count}", flush=True)
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
