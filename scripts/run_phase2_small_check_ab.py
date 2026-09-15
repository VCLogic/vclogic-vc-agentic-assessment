#!/usr/bin/env python3
"""Controlled Phase 2-only A/B for the exploratory small-check decision bar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vc_clone_graph.providers.base import GenerationRequest
from vc_clone_graph.providers.openrouter import OpenRouterProvider
from vc_clone_graph.schemas_v4 import DecisionV4


CALIBRATION_MARKER = "CONTROLLED SMALL-CHECK CALIBRATION RULE"

CALIBRATION_BLOCK = f"""{CALIBRATION_MARKER}
For this diagnostic, apply the any-check endpoint consistently. In means that the
investor would make any genuine commitment of fund capital now, including a bounded
$25k-$100k exploratory check. Unsupported or nonfatal diligence gaps should reduce
check size, investment likelihood, and confidence rather than independently force Out.
Choose Out only when supported evidence identifies a mandate, integrity, conflict,
economics, scale, defensibility, execution, or opportunity-cost blocker strong enough
to make even the smallest real check unattractive. Founder enthusiasm alone remains
insufficient: an exploratory In still requires a credible company-specific wedge and
positive expected learning or ownership value. State the controlling blocker when Out,
or the bounded-check justification when In. Do not change review_priority_score merely
to make it agree with the capital-commitment decision.
END CALIBRATION RULE"""


def revised_prompt(original: str) -> str:
    """Append one episode-neutral calibration block to an exact saved prompt."""
    return f"{original}\n\n{CALIBRATION_BLOCK}"


CASES = (
    {
        "episode_slug": "39-this-pitch-is-damn-near-perfect",
        "actual_decision": "In",
        "artifact": Path(
            "outputs/openrouter-luna-charles-v41b-clean-canaries-2026-08-07-retry/"
            "attempt-1/39-this-pitch-is-damn-near-perfect/phase2/turn-01/"
            "decision-model-response.json"
        ),
    },
    {
        "episode_slug": "18-rowvigor",
        "actual_decision": "Out",
        "artifact": Path(
            "outputs/openrouter-luna-charles-v41b-clean-canaries-2026-08-07-retry/"
            "attempt-2/18-rowvigor/phase2/turn-04/decision-model-response.json"
        ),
    },
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()

    provider = OpenRouterProvider(
        "openai/gpt-5.6-luna",
        env_file=args.env_file,
        max_output_tokens=16384,
        request_timeout_seconds=300,
    )
    comparisons: list[dict[str, object]] = []
    for case in CASES:
        source_path = Path(case["artifact"])
        source = json.loads(source_path.read_text(encoding="utf-8"))
        original_prompt = source["prompt"]
        prompt = revised_prompt(original_prompt)
        episode_slug = str(case["episode_slug"])
        run_root = args.output_root / episode_slug
        request_record = {
            "schema": "phase2-small-check-ab-request-v1",
            "episode_slug": episode_slug,
            "source_artifact": source_path.as_posix(),
            "source_prompt_sha256": hashlib.sha256(
                original_prompt.encode("utf-8")
            ).hexdigest(),
            "revised_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "model": "openai/gpt-5.6-luna",
            "reasoning_effort": "high",
            "max_output_tokens": 16384,
            "prompt": prompt,
            "schema_contract": source["schema"],
        }
        _write_json(run_root / "request.json", request_record)
        result = provider.generate(
            GenerationRequest(
                phase="phase2_small_check_ab",
                prompt=prompt,
                schema=source["schema"],
                max_output_tokens=16384,
                reasoning_effort="high",
            )
        )
        validation_error = None
        validated = None
        try:
            validated = DecisionV4.model_validate(result.parsed).model_dump(mode="json")
        except Exception as exc:  # Preserve any paid malformed response for diagnosis.
            validation_error = f"{type(exc).__name__}: {exc}"
        _write_json(
            run_root / "response.json",
            {
                "schema": "phase2-small-check-ab-response-v1",
                "content": result.content,
                "parsed": result.parsed,
                "validated": validated,
                "validation_error": validation_error,
                "usage": result.usage.model_dump(mode="json"),
                "elapsed_seconds": result.elapsed_seconds,
                "raw_metadata": result.raw_metadata,
            },
        )
        original = source.get("parsed") or {}
        revised = validated or result.parsed or {}
        comparison = {
            "episode_slug": episode_slug,
            "actual_decision": case["actual_decision"],
            "original": {
                key: original.get(key)
                for key in (
                    "decision",
                    "investment_likelihood",
                    "decision_confidence",
                    "review_priority_score",
                    "recommended_check_tier",
                )
            },
            "revised": {
                key: revised.get(key)
                for key in (
                    "decision",
                    "investment_likelihood",
                    "decision_confidence",
                    "review_priority_score",
                    "recommended_check_tier",
                )
            },
            "validation_error": validation_error,
            "usage": result.usage.model_dump(mode="json"),
        }
        comparisons.append(comparison)
        _write_json(run_root / "comparison.json", comparison)
        print(json.dumps(comparison, sort_keys=True), flush=True)

    summary = {
        "schema": "phase2-small-check-ab-summary-v1",
        "calibration_rule": CALIBRATION_BLOCK,
        "comparisons": comparisons,
        "total_cost_usd": sum(
            float(row["usage"]["cost_usd"]) for row in comparisons
        ),
    }
    _write_json(args.output_root / "summary.json", summary)


if __name__ == "__main__":
    main()
