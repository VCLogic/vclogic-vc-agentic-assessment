#!/usr/bin/env python3
"""Build production and leave-one-episode-out rehearsal classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

from vc_clone_graph.phase2_calibration_evaluation import load_phase2_records
from vc_clone_graph.rehearsal_classifier_training import (
    TrainedClassifierArtifact,
    train_classifier_artifact,
    write_classifier_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    workspace = Path.cwd().resolve()
    with args.config.open("rb") as stream:
        settings = tomllib.load(stream)["build"]
    source = (workspace / settings["source_registry"]).resolve()
    output = (workspace / settings["output_root"]).resolve()
    if not source.is_relative_to(workspace) or not output.is_relative_to(workspace):
        raise ValueError("build paths must remain inside the workspace")
    records = load_phase2_records(source)
    trained: list[TrainedClassifierArtifact] = []
    unavailable: list[dict[str, str]] = []
    seed = int(settings.get("seed", 20260823))
    for vc_index, vc_slug in enumerate(
        sorted({record.vc_slug for record in records})
    ):
        vc_records = [record for record in records if record.vc_slug == vc_slug]
        common = {
            "vc_slug": vc_slug,
            "model_version": str(settings["model_version"]),
            "source_registry": str(settings["source_registry"]),
            "label_version": str(settings["label_version"]),
        }
        try:
            trained.append(
                train_classifier_artifact(
                    vc_records, seed=seed + vc_index * 10_000, **common
                )
            )
        except ValueError as exc:
            unavailable.append(
                {"vc_slug": vc_slug, "context": "production", "reason": str(exc)}
            )
        for episode_index, record in enumerate(vc_records, start=1):
            try:
                trained.append(
                    train_classifier_artifact(
                        vc_records,
                        seed=seed + vc_index * 10_000 + episode_index,
                        excluded_episode_slug=record.episode_slug,
                        **common,
                    )
                )
            except ValueError as exc:
                unavailable.append(
                    {
                        "vc_slug": vc_slug,
                        "context": f"held:{record.episode_slug}",
                        "reason": str(exc),
                    }
                )
    registry = write_classifier_registry(output, trained)
    (output / "build-report.json").write_text(
        json.dumps(
            {
                "source_registry": str(source.relative_to(workspace)),
                "artifact_count": len(trained),
                "unavailable": unavailable,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(registry)
    print(f"artifacts={len(trained)} unavailable={len(unavailable)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
