#!/usr/bin/env python3
"""Reconcile pre-fix v5 canary retrieval artifacts into their persisted state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"retrieval artifact is not a list of objects: {path}")
    return value


def reconcile(run_root: Path) -> None:
    state_path = run_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    registry = dict(state.get("evidence_registry", {}))
    for phase in ("phase1", "phase2"):
        historical_paths = sorted((run_root / phase).glob("turn-*/historical-evidence.json"))
        historical_by_id: dict[str, dict[str, object]] = {}
        for path in historical_paths:
            for row in _rows(path):
                evidence_id = str(row["evidence_id"])
                historical_by_id.setdefault(evidence_id, row)
                registry.setdefault(evidence_id, row)
        historical = list(historical_by_id.values())
        state[f"{phase}_historical_evidence"] = historical
        if historical_paths:
            historical_paths[-1].write_text(
                json.dumps(historical, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        for name, key in (
            ("precedent-searches.json", f"{phase}_precedent_searches"),
            ("precedent-reads.json", f"{phase}_precedent_reads"),
        ):
            state[key] = [
                row
                for path in sorted((run_root / phase).glob(f"turn-*/{name}"))
                for row in _rows(path)
            ]
    state["evidence_registry"] = registry
    state_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = sorted(args.root.glob("investors/*/*/state.json"))
    for state_path in runs:
        reconcile(state_path.parent)
    print(f"reconciled {len(runs)} v5 canary runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
