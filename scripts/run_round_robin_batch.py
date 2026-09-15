#!/usr/bin/env python3
"""Run a resumable, sequential In/Out round-robin VC evaluation batch."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from vc_clone_graph.batch import (
    discover_observed_pitch_rows,
    round_robin_order,
    run_with_retries,
    scoped_config,
)
from vc_clone_graph.cli import command_preflight, command_run, command_verify
from vc_clone_graph.config import RunConfig, load_config


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _summary_path(config: RunConfig) -> Path:
    return (
        Path(config.run.output_root)
        / config.run.episode_slug
        / "summary.json"
    )


def _artifact_root(config: RunConfig) -> str:
    return str(Path(config.run.output_root) / config.run.episode_slug)


def _cost(summary: dict[str, Any]) -> float:
    usage = summary.get("usage")
    if type(usage) is not dict:
        return 0.0
    value = usage.get("cost_usd")
    return float(value) if isinstance(value, int | float) else 0.0


def _batch_cost(status: dict[str, Any]) -> float:
    records = list(status["completed_records"]) + list(status["failed_records"])
    return sum(
        _cost(summary)
        for record in records
        for summary in record.get("attempt_summaries", [])
    )


def _initial_status(
    *,
    config_path: Path,
    output_root: str,
    rows: tuple[tuple[str, str], ...],
    order: tuple[str, ...],
) -> dict[str, Any]:
    labels = dict(rows)
    return {
        "schema": "vc-clone-round-robin-batch-v1",
        "config_path": str(config_path),
        "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "output_root": output_root,
        "order_policy": "alternate_actual_in_then_actual_out_then_remaining_outs",
        "episode_count": len(order),
        "in_count": sum(value == "In" for value in labels.values()),
        "out_count": sum(value == "Out" for value in labels.values()),
        "episode_order": list(order),
        "completed_records": [],
        "failed_records": [],
        "cumulative_cost_usd": 0.0,
    }


def _load_or_initialize_status(
    status_path: Path,
    *,
    config_path: Path,
    output_root: str,
    rows: tuple[tuple[str, str], ...],
    order: tuple[str, ...],
) -> dict[str, Any]:
    expected = _initial_status(
        config_path=config_path,
        output_root=output_root,
        rows=rows,
        order=order,
    )
    if not status_path.is_file():
        _write_json_atomic(status_path, expected)
        return expected
    status = json.loads(status_path.read_text(encoding="utf-8"))
    for key in (
        "schema",
        "config_sha256",
        "output_root",
        "episode_order",
        "in_count",
        "out_count",
    ):
        if status.get(key) != expected[key]:
            raise ValueError(f"existing batch status changed at {key}")
    return status


def _attempt(
    base: RunConfig,
    slug: str,
    output_root: str,
    attempt_number: int,
) -> tuple[dict[str, Any], RunConfig]:
    config = scoped_config(
        base,
        slug,
        output_root,
        attempt=attempt_number,
    )
    try:
        command_run(config)
        summary = json.loads(_summary_path(config).read_text(encoding="utf-8"))
        try:
            command_verify(config)
        except ValueError as exc:
            if (
                str(exc) == "v4 investigation cites inaccessible evidence"
                and type(summary.get("decision")) is dict
                and summary["decision"].get("decision") in {"In", "Out"}
            ):
                summary["artifact_verification"] = {
                    "status": "warning",
                    "message": str(exc),
                }
            else:
                raise
        return summary, config
    except Exception as exc:
        summary_path = _summary_path(config)
        artifact_summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else None
        )
        return {
            "episode_slug": slug,
            "error": f"{type(exc).__name__}: {exc}",
            "artifact_summary": artifact_summary,
            "usage": (
                artifact_summary.get("usage", {})
                if type(artifact_summary) is dict
                else {}
            ),
        }, config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        dest="selected_episodes",
        help="Run only this episode now while retaining the full resumable cohort status.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base = load_config(args.config)
    investor_root = (
        Path(base.run.input_root)
        / "data"
        / "investors"
        / base.run.vc_slug
    )
    rows = discover_observed_pitch_rows(
        investor_root / "pitches",
        investor_root / "precedents" / "records",
    )
    order = round_robin_order(rows)
    labels = dict(rows)
    selected_episodes = set(args.selected_episodes or order)
    unknown = sorted(selected_episodes - set(order))
    if unknown:
        raise ValueError(f"selected episodes are outside the cohort: {unknown}")
    status_path = Path(args.output_root) / "batch-status.json"
    status = _load_or_initialize_status(
        status_path,
        config_path=args.config,
        output_root=args.output_root,
        rows=rows,
        order=order,
    )
    print(
        json.dumps(
            {
                "status": "planned",
                "episodes": len(order),
                "ins": status["in_count"],
                "outs": status["out_count"],
                "first_ten": list(order[:10]),
                "output_root": args.output_root,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    if args.preflight_only:
        for position, slug in enumerate(order, start=1):
            config = scoped_config(base, slug, args.output_root, attempt=1)
            command_preflight(config, args.config)
            print(
                json.dumps(
                    {"preflight": position, "episode_slug": slug, "status": "ready"},
                    sort_keys=True,
                ),
                flush=True,
            )
        return 0

    finished = {
        record["episode_slug"]
        for key in ("completed_records", "failed_records")
        for record in status[key]
    }
    for position, slug in enumerate(order, start=1):
        if slug in finished or slug not in selected_episodes:
            continue
        command_preflight(
            scoped_config(base, slug, args.output_root, attempt=1),
            args.config,
        )
        configs: dict[int, RunConfig] = {}

        def run_attempt(number: int) -> dict[str, Any]:
            summary, config = _attempt(base, slug, args.output_root, number)
            configs[number] = config
            return summary

        result = run_with_retries(run_attempt, max_attempts=args.max_attempts)
        selected_attempt = result["selected_attempt"]
        record = {
            "episode_slug": slug,
            "actual_decision": labels[slug],
            "batch_position": position,
            "status": result["status"],
            "selected_attempt": selected_attempt,
            "artifact_root": (
                _artifact_root(configs[selected_attempt])
                if isinstance(selected_attempt, int)
                else None
            ),
            "summary": result["summary"],
            "attempt_summaries": result["attempt_summaries"],
        }
        destination = (
            "completed_records" if result["status"] == "completed" else "failed_records"
        )
        status[destination].append(record)
        status["cumulative_cost_usd"] = _batch_cost(status)
        _write_json_atomic(status_path, status)
        decision = result["summary"].get("decision")
        print(
            json.dumps(
                {
                    "position": position,
                    "episode_slug": slug,
                    "actual": labels[slug],
                    "predicted": (
                        decision.get("decision") if type(decision) is dict else None
                    ),
                    "attempts": len(result["attempt_summaries"]),
                    "status": result["status"],
                    "episode_cost_usd": sum(
                        _cost(row) for row in result["attempt_summaries"]
                    ),
                    "cumulative_cost_usd": status["cumulative_cost_usd"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
