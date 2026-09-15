#!/usr/bin/env python3
"""Verify probabilistic rationale-completion hypotheses on held-out canaries."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import tomllib

from vc_clone_graph.cli import (
    _embedding_provider,
    _filtered_precedents_for_target,
    _generation_provider,
    _index_path,
    _precedent_index_path,
)
from vc_clone_graph.config import load_config
from vc_clone_graph.firewall import verify_package
from vc_clone_graph.precedents import PrecedentCorpus
from vc_clone_graph.prompts_v4 import pitch_evidence_index
from vc_clone_graph.rationale_completion_canary import (
    normalize_minor_contract_fields,
    run_verification_call,
)
from vc_clone_graph.rationale_completion_verification import (
    augment_rationales,
    build_completion_verification_prompt,
    validate_completion_verification,
)
from vc_clone_graph.retrieval import HybridWikiIndex
from vc_clone_graph.retrieval_v4 import retrieve_v4


DEFAULT_EPISODES = (
    "39-this-pitch-is-damn-near-perfect",
    "20-harper-wilde",
)


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _hypotheses(path: Path, episode_slug: str) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["episode_slug"] == episode_slug
            and row["method"] == "ensemble"
            and int(row["rank"]) <= 3
        ]
    rows.sort(key=lambda row: int(row["rank"]))
    if len(rows) != 3:
        raise ValueError(f"expected three ensemble hypotheses for {episode_slug}")
    return tuple(rows)


def _taxonomy(path: Path) -> dict[str, str]:
    rows = _json(path)
    if not isinstance(rows, list):
        raise ValueError("taxonomy must be a list")
    return {str(row["label"]): str(row["definition"]) for row in rows}


def _reference_labels(predictions_path: Path, episode_slug: str) -> set[str]:
    with predictions_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["episode_slug"] == episode_slug:
                return {value for value in row["reference_labels"].split(";") if value}
    raise ValueError(f"missing reference labels for {episode_slug}")


def _scores(predicted: set[str], reference: set[str]) -> dict[str, float | int]:
    tp = len(predicted & reference)
    fp = len(predicted - reference)
    fn = len(reference - predicted)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "jaccard": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
    }


def _invalid_result(output: Path, episode_slug: str, reason: str) -> dict[str, object]:
    usage = _json(output / "usage.json")
    result = {
        "episode_slug": episode_slug,
        "status": "invalid",
        "reason": reason,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cost_usd": float(usage["cost_usd"]),
        "promoted_labels": [],
    }
    (output / "invalid-artifact.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/openrouter-luna-charles-v4-bluffworks-canary.toml"),
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("outputs/canonical-v4-v41-portfolio-2026-08-15/investors/charles-hudson"),
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("reports/evaluation/rationale-completion-charles-2026-08-19"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/rationale-completion-canaries-2026-08-19"),
    )
    parser.add_argument("--max-cost-usd", type=float, default=0.50)
    parser.add_argument("--input-price-per-million", type=float, default=0.22)
    parser.add_argument("--output-price-per-million", type=float, default=1.32)
    args = parser.parse_args()

    config = load_config(args.config)
    required_model = "openai/gpt-5.6-luna"
    if (
        config.provider.kind != "openrouter"
        or config.provider.model != required_model
        or config.phase1.model != required_model
    ):
        raise ValueError("completion canaries require OpenRouter openai/gpt-5.6-luna")
    if config.embedding is None:
        raise ValueError("completion canaries require configured local embeddings")
    embedder = _embedding_provider(config, config.embedding)
    provider = _generation_provider(config, "phase1")
    spent = 0.0
    results: list[dict[str, object]] = []

    first_package = verify_package(
        Path(config.run.input_root), config.run.vc_slug, DEFAULT_EPISODES[0]
    )
    wiki_canonical = HybridWikiIndex.load(
        _index_path(config),
        first_package.wiki,
        embedder,
        require_complete_embeddings=config.embedding.require_complete_index,
    )
    if first_package.precedents is None:
        raise ValueError("completion canaries require precedent corpus")
    precedents_canonical = PrecedentCorpus.load(
        _precedent_index_path(config),
        first_package.precedents,
        embedder,
        require_complete_embeddings=config.embedding.require_complete_index,
    )

    for episode_slug in DEFAULT_EPISODES:
        package = verify_package(
            Path(config.run.input_root), config.run.vc_slug, episode_slug
        )
        hypotheses = _hypotheses(args.analysis_root / "hypotheses.csv", episode_slug)
        definitions = _taxonomy(package.taxonomy)
        candidate_definitions = {
            row["label"]: definitions[row["label"]] for row in hypotheses
        }
        wiki_index, sanitization = wiki_canonical.for_pitch(package.target_company_aliases)
        precedents = _filtered_precedents_for_target(
            precedents_canonical,
            target_slug=episode_slug,
            target_company_aliases=package.target_company_aliases,
        )
        queries = tuple(
            f"{label}: {definition}"
            for label, definition in candidate_definitions.items()
        )
        retrieval = retrieve_v4(
            wiki_index=wiki_index,
            precedent_corpus=precedents,
            wiki_queries=queries,
            precedent_queries=queries,
            top_k=5,
            max_wiki_reads=6,
            max_precedent_reads=6,
            phase="phase1",
            turn=1,
            selection_policy="semantic",
        )
        investor_evidence = list(retrieval.wiki_evidence) + list(
            retrieval.historical_evidence
        )
        if not investor_evidence:
            raise ValueError(f"retrieval returned no investor evidence for {episode_slug}")
        investigation_path = args.canonical_root / episode_slug / "phase1/investigation.json"
        investigation = _json(investigation_path)
        if not isinstance(investigation, dict) or not isinstance(
            investigation.get("rationales"), list
        ):
            raise ValueError(f"invalid canonical investigation for {episode_slug}")
        canonical_rationales = investigation["rationales"]
        pitch_rows = pitch_evidence_index(package.pitch.read_text(encoding="utf-8"))
        with package.registry.open("rb") as stream:
            investor_name = str(tomllib.load(stream)["display_name"])
        episode_output = args.output_root / episode_slug
        episode_output.mkdir(parents=True, exist_ok=True)
        (episode_output / "hypotheses.json").write_text(
            json.dumps(hypotheses, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        retrieval_payload = {
                    "wiki_searches": retrieval.wiki_searches,
                    "precedent_searches": retrieval.precedent_searches,
                    "wiki_evidence": retrieval.wiki_evidence,
                    "historical_evidence": retrieval.historical_evidence,
                    "opened_episode_slugs": retrieval.opened_episode_slugs,
                    "warnings": retrieval.warnings,
                    "wiki_sanitization": sanitization.__dict__,
                }
        retrieval_text = json.dumps(
                retrieval_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            ) + "\n"
        (episode_output / "retrieval.json").write_text(retrieval_text, encoding="utf-8")
        expected_packet = {
            "episode_slug": episode_slug,
            "pitch_evidence": list(pitch_rows),
            "canonical_rationales": canonical_rationales,
            "candidate_definitions": dict(candidate_definitions),
            "investor_evidence": investor_evidence,
        }
        expected_prompt = build_completion_verification_prompt(
            investor_name=investor_name,
            episode_slug=episode_slug,
            pitch_evidence=pitch_rows,
            canonical_rationales=canonical_rationales,
            candidate_definitions=candidate_definitions,
            investor_evidence=investor_evidence,
        )
        reference_record_path = (
            Path("evaluation/phase1_ground_truth_rationales/records/charles-hudson")
            / f"{episode_slug}.json"
        )
        provenance = {
            "episode_slug": episode_slug,
            "canonical_investigation_sha256": sha256(investigation_path.read_bytes()).hexdigest(),
            "pitch_sha256": sha256(package.pitch.read_bytes()).hexdigest(),
            "taxonomy_sha256": sha256(package.taxonomy.read_bytes()).hexdigest(),
            "hypotheses_sha256": sha256(json.dumps(hypotheses, sort_keys=True).encode()).hexdigest(),
            "retrieval_sha256": sha256(retrieval_text.encode()).hexdigest(),
            "input_packet_sha256": sha256(
                json.dumps(expected_packet, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
            "reference_record_sha256": sha256(reference_record_path.read_bytes()).hexdigest(),
            "prompt_sha256": sha256(expected_prompt.encode("utf-8")).hexdigest(),
            "config_sha256": sha256(args.config.read_bytes()).hexdigest(),
            "provider_kind": config.provider.kind,
            "requested_model": required_model,
            "input_price_per_million_usd": args.input_price_per_million,
            "output_price_per_million_usd": args.output_price_per_million,
        }
        provenance_path = episode_output / "source-provenance.json"
        if provenance_path.is_file() and _json(provenance_path) != provenance:
            raise ValueError(f"saved provenance does not match current inputs: {episode_slug}")
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verification_path = episode_output / "verification.json"
        usage_path = episode_output / "usage.json"
        if verification_path.is_file() and usage_path.is_file():
            saved_packet = _json(episode_output / "input-packet.json")
            if saved_packet != expected_packet:
                raise ValueError(f"saved verification input packet is stale: {episode_slug}")
            if (episode_output / "prompt.txt").read_text(encoding="utf-8") != expected_prompt:
                raise ValueError(f"saved verification prompt is stale: {episode_slug}")
            raw_response = _json(episode_output / "raw-response.json")
            if raw_response["raw_metadata"].get("requested_model") != required_model:
                raise ValueError(f"saved verification model mismatch: {episode_slug}")
            pitch_ids = {row["evidence_id"] for row in pitch_rows}
            investor_ids = {str(row["evidence_id"]) for row in investor_evidence}
            normalized_raw = normalize_minor_contract_fields(
                raw_response["parsed"],
                pitch_evidence_ids=pitch_ids,
                investor_evidence_ids=investor_ids,
            )
            raw_verification = validate_completion_verification(
                normalized_raw,
                episode_slug=episode_slug,
                selected_labels=tuple(candidate_definitions),
                pitch_evidence_ids=pitch_ids,
                investor_evidence_ids=investor_ids,
            )
            verification = validate_completion_verification(
                _json(verification_path),
                episode_slug=episode_slug,
                selected_labels=tuple(candidate_definitions),
                pitch_evidence_ids=pitch_ids,
                investor_evidence_ids=investor_ids,
            )
            if raw_verification.model_dump(mode="json") != verification.model_dump(mode="json"):
                raise ValueError(f"saved verification diverges from raw response: {episode_slug}")
            usage = _json(usage_path)
            if not isinstance(usage, dict):
                raise ValueError(f"invalid saved usage for {episode_slug}")
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            cost_usd = float(usage["cost_usd"])
        else:
            if (episode_output / "raw-response.json").is_file():
                raw = _json(episode_output / "raw-response.json")
                normalized = normalize_minor_contract_fields(
                    raw["parsed"],
                    pitch_evidence_ids={row["evidence_id"] for row in pitch_rows},
                    investor_evidence_ids={
                        str(row["evidence_id"]) for row in investor_evidence
                    },
                )
                try:
                    validate_completion_verification(
                        normalized,
                        episode_slug=episode_slug,
                        selected_labels=tuple(candidate_definitions),
                        pitch_evidence_ids={row["evidence_id"] for row in pitch_rows},
                        investor_evidence_ids={
                            str(row["evidence_id"]) for row in investor_evidence
                        },
                    )
                except ValueError as exc:
                    invalid = _invalid_result(episode_output, episode_slug, str(exc))
                    spent += float(invalid["cost_usd"])
                    if spent > args.max_cost_usd:
                        raise ValueError("provider usage exceeds experiment cost cap")
                    results.append(invalid)
                    print(json.dumps(invalid, sort_keys=True), flush=True)
                    break
                raise ValueError(
                    f"valid raw response lacks verification artifact: {episode_slug}"
                )
            try:
                call = run_verification_call(
                    provider=provider,
                    investor_name=investor_name,
                    episode_slug=episode_slug,
                    pitch_evidence=pitch_rows,
                    canonical_rationales=canonical_rationales,
                    candidate_definitions=candidate_definitions,
                    investor_evidence=investor_evidence,
                    output_dir=episode_output,
                    max_cost_usd=args.max_cost_usd,
                    spent_cost_usd=spent,
                    input_price_per_million=args.input_price_per_million,
                    output_price_per_million=args.output_price_per_million,
                    max_output_tokens=8192,
                )
            except ValueError as exc:
                if (episode_output / "raw-response.json").is_file():
                    invalid = _invalid_result(episode_output, episode_slug, str(exc))
                    spent += float(invalid["cost_usd"])
                    if spent > args.max_cost_usd:
                        raise ValueError("provider usage exceeds experiment cost cap")
                    results.append(invalid)
                    print(json.dumps(invalid, sort_keys=True), flush=True)
                    break
                raise
            verification = call.verification
            input_tokens = call.input_tokens
            output_tokens = call.output_tokens
            cost_usd = call.cost_usd
        spent += cost_usd
        if spent > args.max_cost_usd:
            raise ValueError("saved and current usage exceeds cost cap")
        augmented = augment_rationales(canonical_rationales, verification)
        reference = _reference_labels(args.analysis_root / "predictions.csv", episode_slug)
        reference_record = _json(reference_record_path)
        reference_rows = reference_record["rationales"]
        primary_reference = {
            row["rationale_label"] for row in reference_rows if row["salience"] == "primary"
        }
        decision_reference = {
            row["rationale_label"] for row in reference_rows if row["decision_link"] == "explicit"
        }
        canonical_labels = {str(row["taxonomy_label"]) for row in canonical_rationales}
        augmented_labels = {str(row["taxonomy_label"]) for row in augmented.augmented}
        promoted_labels = {row.taxonomy_label for row in augmented.promoted}
        selected_labels = {row["label"] for row in hypotheses}
        missing_reference = reference - canonical_labels
        promoted_matches = [
            (promotion, row)
            for promotion in augmented.promoted
            for row in reference_rows
            if row["rationale_label"] == promotion.taxonomy_label
        ]
        result = {
            "episode_slug": episode_slug,
            "hypotheses": [row["label"] for row in hypotheses],
            "dispositions": [
                row.model_dump(mode="json") for row in verification.dispositions
            ],
            "promoted_labels": [row.taxonomy_label for row in augmented.promoted],
            "canonical_metrics": _scores(canonical_labels, reference),
            "augmented_metrics": _scores(augmented_labels, reference),
            "primary_reference_recall_canonical": len(canonical_labels & primary_reference)
            / len(primary_reference) if primary_reference else 1.0,
            "primary_reference_recall_augmented": len(augmented_labels & primary_reference)
            / len(primary_reference) if primary_reference else 1.0,
            "decision_linked_recall_canonical": len(canonical_labels & decision_reference)
            / len(decision_reference) if decision_reference else 1.0,
            "decision_linked_recall_augmented": len(augmented_labels & decision_reference)
            / len(decision_reference) if decision_reference else 1.0,
            "promotion_precision": len(promoted_labels & reference) / len(promoted_labels)
            if promoted_labels else 1.0,
            "rejected_count": sum(row.disposition == "rejected" for row in verification.dispositions),
            "question_only_count": sum(row.disposition == "question_only" for row in verification.dispositions),
            "candidate_generation_misses": sorted(missing_reference - selected_labels),
            "verification_false_rejections": sorted(
                (missing_reference & selected_labels) - promoted_labels
            ),
            "false_promotions": sorted(promoted_labels - reference),
            "promoted_match_direction_agreement": sum(
                promotion.direction == row["direction"] for promotion, row in promoted_matches
            ) / len(promoted_matches) if promoted_matches else None,
            "promoted_match_salience_agreement": sum(
                promotion.salience == row["salience"] for promotion, row in promoted_matches
            ) / len(promoted_matches) if promoted_matches else None,
            "canonical_reference_overlap": len(canonical_labels & reference),
            "augmented_reference_overlap": len(augmented_labels & reference),
            "reference_label_count": len(reference),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        }
        canonical_payload = list(augmented.canonical)
        promoted_payload = [row.model_dump(mode="json") for row in augmented.promoted]
        augmented_payload = list(augmented.augmented)
        (episode_output / "canonical-rationales.json").write_text(
            json.dumps(canonical_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (episode_output / "promoted-rationales.json").write_text(
            json.dumps(promoted_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (episode_output / "augmented-rationales.json").write_text(
            json.dumps(augmented_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (episode_output / "comparison.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    summary = {"max_cost_usd": args.max_cost_usd, "total_cost_usd": spent, "cases": results}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
