from __future__ import annotations

import json
from pathlib import Path

from scripts import run_round_robin_batch
from vc_clone_graph.config import RunConfig


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig.model_validate(
        {
            "run": {
                "vc_slug": "elizabeth-yin-hustle-fund",
                "episode_slug": "102-tether-bodyguard-of-the-grid",
                "input_root": "inputs",
                "output_root": "outputs/base",
                "checkpoint_path": "outputs/base/checkpoint.sqlite",
                "contract_version": "v4",
            },
            "provider": {"kind": "fake", "model": "fake"},
            "phase1": {"min_iterations": 1, "max_iterations": 1},
            "phase2": {"min_iterations": 1, "max_iterations": 1},
            "retrieval": {"top_k": 1, "max_exact_reads": 1},
        }
    )


def test_attempt_keeps_usable_decision_when_only_citation_verification_warns(
    tmp_path: Path, monkeypatch
) -> None:
    base = _config(tmp_path)

    def write_run(config: RunConfig) -> None:
        summary_path = run_round_robin_batch._summary_path(config)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "episode_slug": config.run.episode_slug,
                    "decision": {"decision": "Out"},
                    "usage": {"cost_usd": 0.01},
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(run_round_robin_batch, "command_run", write_run)
    monkeypatch.setattr(
        run_round_robin_batch,
        "command_verify",
        lambda config: (_ for _ in ()).throw(
            ValueError("v4 investigation cites inaccessible evidence")
        ),
    )

    summary, _ = run_round_robin_batch._attempt(
        base,
        "102-tether-bodyguard-of-the-grid",
        str(tmp_path / "batch"),
        1,
    )

    assert summary["decision"]["decision"] == "Out"
    assert summary["artifact_verification"] == {
        "status": "warning",
        "message": "v4 investigation cites inaccessible evidence",
    }
