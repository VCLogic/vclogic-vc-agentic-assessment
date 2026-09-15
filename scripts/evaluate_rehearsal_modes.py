#!/usr/bin/env python3
"""Evaluate one or more historical founder-rehearsal modes."""

from __future__ import annotations

import argparse
from pathlib import Path

from vc_clone_graph.rehearsal_mode_evaluation import (
    evaluate_replay_files,
    write_evaluation_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="MODE=FILE_OR_DIRECTORY; directories are searched recursively",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    grouped: dict[str, list[Path]] = {}
    for value in args.input:
        if "=" not in value:
            raise ValueError("--input must use MODE=PATH")
        mode, raw_path = value.split("=", 1)
        path = Path(raw_path)
        paths = (
            sorted(path.rglob("replay-evaluation.json"))
            if path.is_dir()
            else [path]
        )
        grouped.setdefault(mode, []).extend(paths)
    result = evaluate_replay_files(grouped)
    outputs = write_evaluation_outputs(args.output_dir, result)
    print(outputs["report_md"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
