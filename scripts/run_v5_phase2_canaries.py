#!/usr/bin/env python3
"""Generate and run the fixed 1-In/1-Out v5 Phase 2 canary panel."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv


@dataclass(frozen=True)
class Canary:
    vc_slug: str
    episode_slug: str
    actual_decision: str


CANARIES = (
    Canary("charles-hudson", "127-nectir-the-classroom-of-the-future", "In"),
    Canary("charles-hudson", "20-harper-wilde", "Out"),
    Canary("elizabeth-yin", "78-got-goals-grab-a-cru", "In"),
    Canary("elizabeth-yin", "72-how-niche-is-too-niche", "Out"),
    Canary("jillian-manus", "12-i-dont-need-your-money-teamable", "In"),
    Canary("jillian-manus", "11-tesloop", "Out"),
    Canary("phil-nadel", "1-babyscripts", "In"),
    Canary("phil-nadel", "2-imirror", "Out"),
    Canary("jesse-middleton", "148-esai", "In"),
    Canary("jesse-middleton", "149-dopl", "Out"),
    Canary("cyan-banister", "135-thoras-ai-the-twin-effect", "In"),
    Canary("cyan-banister", "142-kinometrix", "Out"),
)


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {value!r}")


def _write_toml(path: Path, payload: dict[str, object]) -> None:
    lines: list[str] = []
    for section, raw_values in payload.items():
        if raw_values is None:
            continue
        if not isinstance(raw_values, dict):
            raise TypeError(f"top-level config value must be a table: {section}")
        lines.append(f"[{section}]")
        for key, value in raw_values.items():
            if value is not None:
                lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_config(root: Path, canary: Canary, output_root: Path) -> Path:
    canonical = (
        root / "outputs/canonical-v4-v41-portfolio-2026-08-15/investors"
        / canary.vc_slug / canary.episode_slug / "run-config.json"
    )
    config = json.loads(canonical.read_text(encoding="utf-8"))
    run_output = output_root / "investors" / canary.vc_slug
    config["run"].update({
        "contract_version": "v5",
        "output_root": run_output.as_posix(),
        "checkpoint_path": (
            output_root / "checkpoints" / canary.vc_slug
            / f"{canary.episode_slug}.sqlite"
        ).as_posix(),
    })
    config["provider"].update({
        "kind": "openrouter",
        "model": "openai/gpt-5.6-luna",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "context_window": 131072,
        "max_output_tokens": 16384,
        "request_timeout_seconds": 600,
    })
    config["phase1"]["model"] = "openai/gpt-5.6-luna"
    config["phase2"].update({
        "model": "openai/gpt-5.6-luna",
        "min_iterations": 1,
        "max_iterations": 2,
    })
    target = root / output_root / "configs" / f"{canary.vc_slug}__{canary.episode_slug}.toml"
    _write_toml(target, config)
    return target


def _usage(run_root: Path) -> dict[str, object]:
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    phase2 = summary.get("usage_by_phase", {}).get("phase2", {})
    decision = summary.get("decision", {})
    return {
        "predicted_decision": decision.get("decision"),
        "investment_likelihood": decision.get("investment_likelihood"),
        "decision_confidence": decision.get("decision_confidence"),
        "review_priority_score": decision.get("review_priority_score"),
        "phase2_status": summary.get("phase2_status"),
        "phase2_iterations": summary.get("phase2_iterations"),
        "input_tokens": phase2.get("input_tokens", 0),
        "cached_input_tokens": phase2.get("cached_input_tokens", 0),
        "output_tokens": phase2.get("output_tokens", 0),
        "cost_usd": float(phase2.get("cost_usd", 0.0)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(
        "outputs/v5-associated-rationales/phase2-canaries-2026-08-20"
    ))
    parser.add_argument("--cost-cap-usd", type=float, default=3.0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd()
    for env_path in (
        root / ".env",
        root.parent / ".env",
        root.parent / "agentic-vc-clone-framework/.env",
    ):
        if env_path.is_file():
            load_dotenv(env_path, override=False)
        if os.environ.get("OPENROUTER_API_KEY"):
            break
    rows: list[dict[str, object]] = []
    spent = 0.0
    for canary in CANARIES:
        config = prepare_config(root, canary, args.output_root)
        if args.prepare_only:
            continue
        if spent >= args.cost_cap_usd:
            break
        phase1 = (
            root / "outputs/v5-associated-rationales/phase1/investors"
            / canary.vc_slug / canary.episode_slug / "phase1"
        )
        command = [
            sys.executable, "-m", "vc_clone_graph.cli", "decide",
            "--config", str(config), "--phase1-from", str(phase1),
        ]
        completed = subprocess.run(command, cwd=root, env=os.environ.copy())
        run_root = args.output_root / "investors" / canary.vc_slug / canary.episode_slug
        row: dict[str, object] = {
            "vc_slug": canary.vc_slug,
            "episode_slug": canary.episode_slug,
            "actual_decision": canary.actual_decision,
            "exit_code": completed.returncode,
        }
        if completed.returncode == 0:
            row.update(_usage(run_root))
            spent += float(row["cost_usd"])
        else:
            row["error"] = "Phase 2 command failed; inspect run artifacts and stderr."
        rows.append(row)
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "results.json").write_text(
            json.dumps({"cost_cap_usd": args.cost_cap_usd, "spent_usd": spent, "rows": rows}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            break
    print(json.dumps({"spent_usd": spent, "completed": len(rows), "rows": rows}, indent=2))
    return 0 if all(row["exit_code"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
