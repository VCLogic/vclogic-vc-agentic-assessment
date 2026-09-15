#!/usr/bin/env python3
"""Run the frozen Phase 1 v4.3 canary sequentially under a cost ceiling."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter

from vc_clone_graph.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(
    "reports/evaluation/phase1-v43-diagnostic-2026-08-17/canary-manifest.json"
)
DEFAULT_OUTPUT = Path("reports/evaluation/phase1-v43-canary-2026-08-17")
CONFIG_NAMES = {
    "charles-hudson": "charles-hudson.toml",
    "cyan-banister": "cyan-banister.toml",
    "elizabeth-yin": "elizabeth-yin.toml",
    "jesse-middleton": "jesse-middleton.toml",
    "jillian-manus": "jillian-manus.toml",
    "phil-nadel": "phil-nadel.toml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config-dir", type=Path, default=Path("configs/v43"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cost-ceiling", type=float, default=3.0)
    parser.add_argument("--per-case-reserve", type=float, default=0.20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _path(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "vc_slug", "episode_slug", "actual_decision", "role", "status", "cost_usd"
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def render_case_config(base: str, *, episode_slug: str) -> str:
    checkpoint = (
        f"outputs/phase1-v43-canary-2026-08-17/checkpoints/{episode_slug}.sqlite"
    )
    rendered, episode_count = re.subn(
        r'(?m)^episode_slug\s*=\s*"[^"]+"$',
        f'episode_slug = "{episode_slug}"',
        base,
        count=1,
    )
    rendered, checkpoint_count = re.subn(
        r'(?m)^checkpoint_path\s*=\s*"[^"]+"$',
        f'checkpoint_path = "{checkpoint}"',
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


def _state_for(config_path: Path) -> tuple[Path, dict]:
    config = load_config(config_path)
    state_path = (
        PROJECT_ROOT / config.run.output_root / config.run.episode_slug / "state.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return state_path, state


def main() -> int:
    args = parse_args()
    manifest_path = _path(args.manifest)
    output = _path(args.output)
    config_dir = _path(args.config_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = list(manifest["cases"])
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.cost_ceiling <= 0 or args.per_case_reserve < 0:
        raise ValueError("cost controls must be nonnegative and ceiling must be positive")

    runtime_configs = output / "configs"
    rows: list[dict[str, object]] = []
    cumulative_cost = 0.0
    for position, case in enumerate(cases, start=1):
        vc_slug = case["vc_slug"]
        episode_slug = case["episode_slug"]
        base_path = config_dir / CONFIG_NAMES[vc_slug]
        runtime_path = runtime_configs / vc_slug / f"{episode_slug}.toml"
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(
            render_case_config(base_path.read_text(encoding="utf-8"), episode_slug=episode_slug),
            encoding="utf-8",
        )
        if args.dry_run:
            print(f"{position:02d} {vc_slug} {episode_slug} {case['actual_decision']}")
            continue
        runtime_config = load_config(runtime_path)
        expected_state_path = (
            PROJECT_ROOT
            / runtime_config.run.output_root
            / runtime_config.run.episode_slug
            / "state.json"
        )
        checkpoint_path = PROJECT_ROOT / runtime_config.run.checkpoint_path
        mode = existing_case_mode(expected_state_path, checkpoint_path)
        if mode != "skip" and cumulative_cost + args.per_case_reserve > args.cost_ceiling:
            rows.append({
                "position": position,
                "vc_slug": vc_slug,
                "episode_slug": episode_slug,
                "actual_decision": case["actual_decision"],
                "role": case["role"],
                "status": "not_started_cost_ceiling",
                "phase1_status": "",
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "cumulative_cost_usd": round(cumulative_cost, 8),
                "elapsed_seconds": 0.0,
                "state_path": "",
                "error": "reserved per-case allowance would exceed ceiling",
            })
            break
        started = perf_counter()
        status = "completed_existing" if mode == "skip" else "completed"
        error = ""
        result = None
        if mode != "skip":
            result = subprocess.run(
                [
                    sys.executable, "-m", "vc_clone_graph.cli", mode,
                    "--config", str(runtime_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        try:
            state_path, state = _state_for(runtime_path)
        except Exception as exc:
            state_path, state = Path(), {}
            status = "failed"
            stderr = result.stderr[-1000:] if result is not None else ""
            error = f"{exc}; stderr={stderr}"
        if result is not None and result.returncode != 0:
            status = "failed_with_state" if state else "failed"
            error = error or result.stderr[-1000:]
        usage = state.get("usage", {})
        case_cost = float(usage.get("cost_usd", 0.0))
        cumulative_cost += case_cost
        calls = sum(
            1
            for event in state.get("events", [])
            if event.get("name") in {
                "phase1_plan", "phase1_investigation", "phase1_rationale_mapping"
            }
        )
        row = {
            "position": position,
            "vc_slug": vc_slug,
            "episode_slug": episode_slug,
            "actual_decision": case["actual_decision"],
            "role": case["role"],
            "status": status,
            "phase1_status": state.get("phase1_status", ""),
            "calls": calls,
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "cost_usd": round(case_cost, 8),
            "cumulative_cost_usd": round(cumulative_cost, 8),
            "elapsed_seconds": round(perf_counter() - started, 3),
            "state_path": str(state_path),
            "error": error,
        }
        rows.append(row)
        _write_csv(output / "status.csv", rows)
        _write_json(output / "status.json", {
            "schema": "phase1-v43-canary-status-v1",
            "cost_ceiling_usd": args.cost_ceiling,
            "per_case_reserve_usd": args.per_case_reserve,
            "cumulative_cost_usd": cumulative_cost,
            "cases": rows,
        })
        print(
            f"[{position}/{len(cases)}] {vc_slug} {episode_slug}: {status}; "
            f"phase1={state.get('phase1_status')} cost=${case_cost:.4f} total=${cumulative_cost:.4f}",
            flush=True,
        )
        if cumulative_cost >= args.cost_ceiling:
            break
    if args.dry_run:
        return 0
    completed = sum(str(row["status"]).startswith("completed") for row in rows)
    print(f"completed={completed}/{len(cases)} cumulative_cost_usd={cumulative_cost:.6f}")
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
