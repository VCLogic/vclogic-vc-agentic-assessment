#!/usr/bin/env python3
"""CLI for preflighting and running the one-call baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from batch import pilot_slugs, run_episode, write_json_atomic
from config import load_baseline_config
from input import assemble_context
from prompt import build_prompt
from schema import baseline_json_schema
from vc_clone_graph.batch import discover_observed_pitch_rows, round_robin_order
from vc_clone_graph.providers.base import GenerationRequest
from vc_clone_graph.providers.openrouter import OpenRouterProvider


def rows(config):
    root = Path(config.input_root) / "data" / "investors" / config.vc_slug
    return discover_observed_pitch_rows(root / "pitches", root / "precedents" / "records")


def provider(config):
    row = config.provider
    return OpenRouterProvider(
        row.model, base_url=row.base_url, api_key_env=row.api_key_env,
        max_output_tokens=row.max_output_tokens,
        require_parameters=row.require_parameters, data_collection=row.data_collection,
    )


def preflight(config, slugs):
    labels = dict(rows(config))
    output = []
    for slug in slugs:
        context, manifest = assemble_context(config, slug)
        prompt = build_prompt(context)
        taxonomy = {row["label"] for row in context.taxonomy}
        GenerationRequest(
            phase="one_shot_baseline", prompt=prompt,
            schema=baseline_json_schema(slug, taxonomy),
            max_output_tokens=config.provider.max_output_tokens,
            reasoning_effort=config.provider.reasoning_effort,
        )
        selected = [row.episode_slug for row in context.precedents]
        if slug in selected or len(selected) != 5:
            raise ValueError("target exclusion or precedent count failed")
        output.append({"episode_slug": slug, "actual_label": labels[slug], "prompt_characters": len(prompt), "selected_precedents": selected, "status": "ready"})
    return output


def run_many(config, slugs):
    observed = dict(rows(config))
    status_path = Path(config.output_root) / "batch-status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {
        "schema": "one-shot-baseline-batch-v1", "episode_order": list(slugs), "records": []
    }
    if status.get("episode_order") != list(slugs):
        raise ValueError("existing batch episode order differs")
    finished = {row["episode_slug"] for row in status["records"]}
    generator = provider(config)
    for slug in slugs:
        if slug in finished:
            continue
        summary = run_episode(config, slug, observed[slug], generator)
        status["records"].append(summary)
        status["cumulative_cost_usd"] = sum(float(row.get("usage", {}).get("cost_usd", 0)) for row in status["records"])
        write_json_atomic(status_path, status)
        print(json.dumps({"episode_slug": slug, "status": summary["status"], "actual": observed[slug], "prediction": summary.get("prediction", {}).get("decision"), "cost_usd": summary.get("usage", {}).get("cost_usd", 0)}, sort_keys=True), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "episode", "pilot", "batch"))
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    parser.add_argument("--episode")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    config = load_baseline_config(args.config)
    observed = rows(config)
    if args.command == "episode":
        if not args.episode:
            parser.error("episode requires --episode")
        result = run_many(config, (args.episode,))
    else:
        slugs = pilot_slugs(observed) if args.command == "pilot" or args.pilot else round_robin_order(observed)
        result = preflight(config, slugs) if args.command == "preflight" else run_many(config, slugs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
