#!/usr/bin/env python3
"""Run reversible offline calibration of canonical Phase 1 rationales."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import joblib
import numpy as np

from vc_clone_graph.phase1_calibration_cases import build_calibration_cases
from vc_clone_graph.phase1_calibration_downstream import run_downstream_models
from vc_clone_graph.phase1_calibration_evaluation import evaluate_calibration
from vc_clone_graph.phase1_calibration_models import CONDITIONS, run_nested_calibration
from vc_clone_graph.phase1_evaluation import load_phase1_cases
from vc_clone_graph.phase2_calibration_evaluation import load_phase2_records
from vc_clone_graph.providers.sentence_transformers import SentenceTransformerEmbeddingProvider
from vc_clone_graph.sequential_evaluation import (
    evaluate_task_endpoints,
    evaluate_uncertainty_and_significance,
)


DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
DEFAULT_REGISTRY = Path("evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json")
DEFAULT_REFERENCES = Path("evaluation/phase1_ground_truth_rationales")
DEFAULT_TAXONOMY = Path("../agentic-vc-clone-framework/taxonomy/codebook_v_final.json")
DEFAULT_BENCHMARK = Path("evaluation/benchmarks/phase1-calibration-primary-2026-08-17.json")


def _budgets(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("review budgets must be positive integers")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    result.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    result.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--benchmark-output", type=Path, default=DEFAULT_BENCHMARK)
    result.add_argument("--embedding-model", default=DEFAULT_MODEL)
    result.add_argument("--embedding-revision", default=DEFAULT_REVISION)
    result.add_argument("--embedding-device", default="auto")
    result.add_argument("--embedding-batch-size", type=int, default=4)
    result.add_argument("--jobs", type=int, default=1)
    result.add_argument("--inner-splits", type=int, default=3)
    result.add_argument("--pca-components", type=int, default=32)
    result.add_argument("--bootstrap-iterations", type=int, default=2_000)
    result.add_argument("--permutation-iterations", type=int, default=5_000)
    result.add_argument("--review-budgets", type=_budgets, default=(1, 3, 5, 10, 20))
    result.add_argument("--seed", type=int, default=20260817)
    result.add_argument("--resume-analysis", action="store_true")
    return result


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _csv_value(value: object) -> object:
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
        temporary.replace(path)
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(field for field in row if field not in fields)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})
    temporary.replace(path)


def _downstream_rows(records, learned) -> list[dict[str, object]]:
    by_key = {(record.vc_slug, record.episode_slug): record for record in records}
    rows: list[dict[str, object]] = [
        {
            "method": "raw_phase2",
            "vc_slug": record.vc_slug,
            "vc_name": record.vc_name,
            "episode_slug": record.episode_slug,
            "actual_decision": "In" if record.target else "Out",
            "score": record.raw_likelihood,
            "balanced_decision": "In" if record.raw_decision else "Out",
            "precision_decision": "In" if record.raw_decision else "Out",
            "balanced_threshold": "fixed_agent_decision",
            "precision_threshold": "fixed_agent_decision",
            "raw_phase2_decision": "In" if record.raw_decision else "Out",
            "raw_phase2_likelihood": record.raw_likelihood,
        }
        for record in records
    ]
    for method, predictions in learned.items():
        for prediction in predictions:
            record = by_key[(prediction.vc_slug, prediction.episode_slug)]
            rows.append({
                "method": method,
                "vc_slug": prediction.vc_slug,
                "vc_name": prediction.vc_name,
                "episode_slug": prediction.episode_slug,
                "actual_decision": "In" if prediction.target else "Out",
                "score": prediction.score,
                "balanced_decision": "In" if prediction.balanced_decision else "Out",
                "precision_decision": "In" if prediction.precision_decision else "Out",
                "balanced_threshold": prediction.balanced_threshold,
                "precision_threshold": prediction.precision_threshold,
                "raw_phase2_decision": "In" if record.raw_decision else "Out",
                "raw_phase2_likelihood": record.raw_likelihood,
            })
    return rows


def _render_report(analysis: Mapping[str, object]) -> str:
    cases = analysis["cases"]
    rationale = analysis["rationale"]
    downstream = analysis["downstream"]
    rich = [
        row for row in rationale["metric_rows"]
        if row["scope"] == "rich" and row["scope_value"] == "rich"
    ]
    macro_class = [row for row in downstream["classification"] if row["scope"] == "macro"]
    macro_rank = [
        row for row in downstream["ranking"]
        if row["scope"] == "macro" and int(row["review_budget"]) == 5
    ]
    per_vc_rationale = [
        row for row in rationale["metric_rows"]
        if row["scope"] == "vc" and row["method"] in {"raw_phase1", "filtering_only"}
    ]
    lines = [
        "# Phase 1 Rationale Calibration",
        "",
        "> Development evaluation. Reference rationales are automated transcript-derived candidates, not final human-validated ground truth.",
        "",
        f"Population: **{len(cases)} investor-pitch cases**, **{len(set(case.vc_slug for case in cases))} VCs**, "
        f"**{sum(case.actual_decision == 'In' for case in cases)} Ins**, and "
        f"**{sum(case.actual_decision == 'Out' for case in cases)} Outs**.",
        "",
        "Canonical v4/v4.1 outputs were read only. This experiment used local embeddings and cost **$0.00 in API calls**.",
        "",
        "## Rationale recovery on rich references",
        "",
        "| Method | Precision | Recall | F1 | Family F1 | Avg labels | AP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rich:
        lines.append(
            f"| {row['method']} | {row['micro_precision']:.3f} | {row['micro_recall']:.3f} | "
            f"{row['micro_f1']:.3f} | {row['family_f1']:.3f} | "
            f"{row['average_predicted_size']:.2f} | {row['average_precision']:.3f} |"
        )
    lines.extend([
        "",
        "### Exact-label recovery by investor",
        "",
        "| Investor | Method | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|",
    ])
    vc_names = {case.vc_slug: case.vc_name for case in cases}
    for row in per_vc_rationale:
        lines.append(
            f"| {vc_names[row['scope_value']]} | {row['method']} | "
            f"{row['micro_precision']:.3f} | {row['micro_recall']:.3f} | "
            f"{row['micro_f1']:.3f} |"
        )
    lines.extend([
        "",
        "## Downstream investment-decision classification",
        "",
        "| Method | Balanced accuracy | In precision | In recall | In F1 |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in macro_class:
        lines.append(
            f"| {row['method']} | {row['balanced_accuracy']:.3f} | "
            f"{row['in_precision']:.3f} | {row['in_recall']:.3f} | {row['in_f1']:.3f} |"
        )
    lines.extend([
        "",
        "## Downstream opportunity prioritization",
        "",
        "| Method | AP | ROC AUC | Precision@5 | Recall@5 |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in macro_rank:
        lines.append(
            f"| {row['method']} | {row['average_precision']:.3f} | {row['roc_auc']:.3f} | "
            f"{row['precision_at_k']:.3f} | {row['recall_at_k']:.3f} |"
        )
    raw_rich = next(row for row in rich if row["method"] == "raw_phase1")
    filtered_rich = next(row for row in rich if row["method"] == "filtering_only")
    filtered_action = next(
        row for row in rationale["action_rows"] if row["method"] == "filtering_only"
    )
    raw_rank = next(row for row in macro_rank if row["method"] == "raw_phase1")
    filtered_rank = next(row for row in macro_rank if row["method"] == "filtering_only")
    lines.extend([
        "",
        "## Promotion-gate result",
        "",
        f"Filtering-only increased rich-reference F1 from **{raw_rich['micro_f1']:.3f}** to "
        f"**{filtered_rich['micro_f1']:.3f}**, with precision **{filtered_rich['micro_precision']:.3f}** "
        f"and recall **{filtered_rich['micro_recall']:.3f}**. It removed "
        f"{filtered_action['correct_removals']} false labels while also removing "
        f"{filtered_action['harmful_removals']} true labels.",
        "",
        f"It did **not** clear the promotion gates: precision remained below 0.35, and downstream "
        f"ranking AP changed from **{raw_rank['average_precision']:.3f}** for raw Phase 1 to "
        f"**{filtered_rank['average_precision']:.3f}**. No downstream comparison was significant "
        "after Holm correction. Canonical v4/v4.1 therefore remains the primary Phase 1 method.",
    ])
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The semantic control measures what pitch text and historical rationale prototypes can recover without Phase 1 agent signals. The selector-expansion condition measures whether those signals add value and whether absent labels can be rescued. Filtering-only can remove noisy candidates but cannot restore omissions.",
        "",
        "No challenger replaces canonical Phase 1 automatically. Promotion requires improved exact-label F1, useful precision/recall, no material loss in family and salient-rationale coverage, downstream ranking improvement, and evidence across multiple investors.",
        "",
    ])
    return "\n".join(lines)


def _write_outputs(analysis: Mapping[str, object], output: Path, inputs: Mapping[str, Path]) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    rationale = analysis["rationale"]
    downstream = analysis["downstream"]
    run_result = analysis["run_result"]
    paths = {
        "evaluation": output / "evaluation.md",
        "rationale_predictions": output / "rationale_predictions.csv",
        "rationale_metrics": output / "rationale_metrics.csv",
        "rationale_subsets": output / "rationale_subsets.csv",
        "rationale_actions": output / "rationale_actions.csv",
        "rationale_uncertainty": output / "rationale_uncertainty.csv",
        "downstream_predictions": output / "downstream_predictions.csv",
        "downstream_classification": output / "downstream_classification.csv",
        "downstream_ranking": output / "downstream_ranking.csv",
        "downstream_tradeoffs": output / "downstream_threshold_tradeoffs.csv",
        "downstream_uncertainty": output / "downstream_uncertainty.csv",
        "downstream_significance": output / "downstream_significance.csv",
        "fold_provenance": output / "fold_provenance.json",
        "embedding_metadata": output / "embedding-metadata.json",
        "benchmark_policy": output / "benchmark_policy.json",
        "manifest": output / "manifest.json",
    }
    _atomic_text(paths["evaluation"], _render_report(analysis))
    _write_csv(paths["rationale_predictions"], [asdict(row) for row in run_result.predictions])
    _write_csv(paths["rationale_metrics"], rationale["metric_rows"])
    _write_csv(paths["rationale_subsets"], rationale["subset_rows"])
    _write_csv(paths["rationale_actions"], rationale["action_rows"])
    _write_csv(paths["rationale_uncertainty"], rationale["uncertainty_rows"])
    _write_csv(paths["downstream_predictions"], analysis["downstream_prediction_rows"])
    _write_csv(paths["downstream_classification"], downstream["classification"])
    _write_csv(paths["downstream_ranking"], downstream["ranking"])
    _write_csv(paths["downstream_tradeoffs"], downstream["threshold_tradeoffs"])
    _write_csv(paths["downstream_uncertainty"], analysis["downstream_statistics"]["uncertainty"])
    _write_csv(paths["downstream_significance"], analysis["downstream_statistics"]["significance"])
    _write_json(paths["fold_provenance"], {
        "schema": "phase1-calibration-fold-provenance-v1",
        "rationale_folds": [asdict(row) for row in run_result.provenance],
        "downstream_folds": [
            {
                "method": method,
                "vc_slug": row.vc_slug,
                "held_episode_slug": row.episode_slug,
                "training_episode_slugs": row.training_episode_slugs,
                "training_groups": row.training_groups,
            }
            for method, predictions in analysis["learned"].items()
            for row in predictions
        ],
    })
    _write_json(paths["embedding_metadata"], analysis["embedding_metadata"])
    rich_metrics = [
        row for row in rationale["metric_rows"]
        if row["scope"] == "rich" and row["scope_value"] == "rich"
    ]
    policy = {
        "schema": "phase1-calibration-benchmark-policy-v1",
        "scientific_status": "development_not_untouched_holdout",
        "reference_status": "automated_candidate_not_human_validated",
        "canonical_phase1_replaced": False,
        "prespecified_primary": "selector_expansion",
        "descriptive_best_rich_f1": max(rich_metrics, key=lambda row: row["micro_f1"])["method"],
        "promotion_gates": {
            "exact_label_f1_above": 0.316,
            "precision_at_least": 0.35,
            "recall_at_least": 0.50,
            "requires_downstream_ranking_gain": True,
            "requires_multi_vc_support": True,
        },
    }
    _write_json(paths["benchmark_policy"], policy)
    manifest_inputs = {
        name: {"path": str(path.resolve()), "sha256": _hash(path.resolve())}
        for name, path in inputs.items()
    }
    cases = analysis["cases"]
    _write_json(paths["manifest"], {
        "schema": "phase1-calibration-analysis-v1",
        "scientific_status": "development_not_untouched_holdout",
        "population": {
            "cases": len(cases),
            "investors": len(set(case.vc_slug for case in cases)),
            "ins": sum(case.actual_decision == "In" for case in cases),
            "outs": sum(case.actual_decision == "Out" for case in cases),
            "labels": len(cases[0].labels),
        },
        "api_cost_usd": 0.0,
        "inputs": manifest_inputs,
        "outputs": sorted(str(path.relative_to(output)) for path in paths.values()),
        "canonical_phase1_modified": False,
    })
    return paths


def _validate_output_boundary(output: Path, canonical_inputs: Sequence[Path]) -> None:
    resolved = output.resolve()
    for canonical in canonical_inputs:
        source = canonical.resolve()
        if resolved == source or resolved in source.parents or source in resolved.parents:
            raise ValueError(f"output overlaps canonical input: {output} and {canonical}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    project_root = Path.cwd().resolve()
    input_paths = {
        "registry": args.registry,
        "reference_manifest": args.references / "manifest.json",
        "taxonomy": args.taxonomy,
    }
    _validate_output_boundary(args.output, tuple(input_paths.values()))
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / ".analysis_checkpoint.joblib"
    if args.resume_analysis:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"analysis checkpoint does not exist: {checkpoint}")
        analysis = joblib.load(checkpoint)
    else:
        phase2_records = load_phase2_records(args.registry)
        phase1_cases, taxonomy = load_phase1_cases(
            project_root, args.registry, args.references, args.taxonomy
        )
        calibration_by_key = {
            case.key: case
            for case in build_calibration_cases(project_root, phase1_cases, taxonomy)
        }
        cases = [calibration_by_key[(row.vc_slug, row.episode_slug)] for row in phase2_records]
        embedder = SentenceTransformerEmbeddingProvider(
            args.embedding_model,
            revision=args.embedding_revision,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
        )
        embeddings = np.asarray(embedder.embed_documents([case.pitch_text for case in cases]))
        joblib.dump(embeddings, args.output / ".embedding_cache.joblib")
        run_result = run_nested_calibration(
            cases,
            embeddings,
            inner_splits=args.inner_splits,
            pca_components=args.pca_components,
            jobs=args.jobs,
        )
        rationale = evaluate_calibration(
            cases,
            run_result.predictions,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed + 1,
        )
        learned = run_downstream_models(
            cases,
            run_result.predictions,
            taxonomy,
            conditions=CONDITIONS,
            inner_splits=args.inner_splits,
            seed=args.seed + 2,
            n_jobs=args.jobs,
        )
        downstream = evaluate_task_endpoints(
            phase2_records, learned, review_budgets=args.review_budgets
        )
        downstream_statistics = evaluate_uncertainty_and_significance(
            downstream,
            bootstrap_iterations=args.bootstrap_iterations,
            permutation_iterations=args.permutation_iterations,
            seed=args.seed + 3,
        )
        analysis = {
            "schema": "phase1-calibration-analysis-v1",
            "cases": cases,
            "records": phase2_records,
            "run_result": run_result,
            "rationale": rationale,
            "learned": learned,
            "downstream": downstream,
            "downstream_statistics": downstream_statistics,
            "downstream_prediction_rows": _downstream_rows(phase2_records, learned),
            "embedding_metadata": {
                **embedder.metadata,
                "dimensions": int(embeddings.shape[1]),
                "cases": len(cases),
                "cache": ".embedding_cache.joblib",
            },
            "api_cost_usd": 0.0,
        }
        joblib.dump(analysis, checkpoint)
    analysis["downstream_prediction_rows"] = _downstream_rows(
        analysis["records"], analysis["learned"]
    )
    paths = _write_outputs(analysis, args.output, input_paths)
    args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths["benchmark_policy"], args.benchmark_output)
    print(
        f"Phase 1 calibration complete: cases={len(analysis['cases'])} "
        f"vcs={len(set(case.vc_slug for case in analysis['cases']))} "
        f"api_cost_usd=0.00 report={paths['evaluation']} checkpoint={checkpoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
