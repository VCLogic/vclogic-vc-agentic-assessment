#!/usr/bin/env python3
"""Evaluate all frozen Phase 1, Phase 2, and audited-pitch signals."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

import numpy as np

from vc_clone_graph.full_inference_fusion import (
    EARLY_FUSION_METHODS,
    FullFusionPrediction,
    build_full_inference_population,
    fixed_score_fusions,
    native_phase2_score_fusions,
    nested_error_correction_predictions,
    nested_full_fusion_predictions,
    write_full_fusion_analysis,
)
from vc_clone_graph.posthoc_semantic_models import load_semantic_population
from vc_clone_graph.providers.sentence_transformers import (
    SentenceTransformerEmbeddingProvider,
)
from vc_clone_graph.standalone_pitch_models import load_verified_pitch_population


SEMANTIC_COMPONENTS = {
    "embedding_logistic": "full_embedding",
    "structured_boost": "structured",
    "tfidf_full": "full_tfidf",
}
PITCH_COMPONENTS = {
    "pitch_embedding_only": "pitch_embedding",
    "pitch_tfidf_only": "pitch_tfidf",
}


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def configure_offline_environment() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--vc-slug", required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--semantic-predictions", type=Path, required=True)
    parser.add_argument("--pitch-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--methods",
        default=",".join(EARLY_FUSION_METHODS),
        help="Comma-separated early-fusion methods; fixed fusion and correction always run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse verified method checkpoints already present in the output directory.",
    )
    return parser.parse_args()


def _load_components(
    path: Path, method_mapping: dict[str, str]
) -> dict[str, list[FullFusionPrediction]]:
    result = {name: [] for name in method_mapping.values()}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            source_method = str(raw["method"])
            if source_method not in method_mapping:
                continue
            result[method_mapping[source_method]].append(
                FullFusionPrediction(
                    method=method_mapping[source_method],
                    episode_slug=str(raw["episode_slug"]),
                    target=int(raw["target"]),
                    score=float(raw["score"]),
                    predicted=int(raw["predicted"]),
                    threshold=float(raw["threshold"]),
                    selected_config=json.loads(raw["selected_config"]),
                    training_slugs=tuple(json.loads(raw["training_slugs"])),
                )
            )
    missing = [name for name, rows in result.items() if not rows]
    if missing:
        raise ValueError(f"missing component methods in {path}: {missing}")
    return result


def _load_resume_predictions(
    output: Path, source_manifest: dict[str, str]
) -> dict[str, list[FullFusionPrediction]]:
    manifest_path = output / "manifest.json"
    predictions_path = output / "predictions.csv"
    if not (manifest_path.is_file() and predictions_path.is_file()):
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sources") != source_manifest:
        raise ValueError("resume checkpoint source manifest mismatch")
    result: dict[str, list[FullFusionPrediction]] = {}
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            method = str(raw["method"])
            result.setdefault(method, []).append(
                FullFusionPrediction(
                    method=method,
                    episode_slug=str(raw["episode_slug"]),
                    target=int(raw["target"]),
                    score=float(raw["score"]),
                    predicted=int(raw["predicted"]),
                    threshold=float(raw["threshold"]),
                    selected_config=json.loads(raw["selected_config"]),
                    training_slugs=tuple(json.loads(raw["training_slugs"])),
                )
            )
    return result


def main() -> None:
    args = parse_args()
    configure_offline_environment()
    requested_methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())

    print("verifying and aligning frozen inference and pitch artifacts...", flush=True)
    semantic = load_semantic_population(args.status, args.labels)
    pitches = load_verified_pitch_population(
        args.input_root, args.status, args.labels, args.vc_slug
    )
    records = build_full_inference_population(semantic, pitches)

    provider = SentenceTransformerEmbeddingProvider(
        args.embedding_model,
        revision=args.embedding_revision,
        device=args.device,
        batch_size=args.batch_size,
        normalize=True,
        document_prefix="search_document: ",
        query_prefix="search_query: ",
    )
    print("embedding audited pitches and complete Phase 1/2 reasoning locally...", flush=True)
    embeddings = {
        "pitch": np.asarray(
            provider.embed_documents([row.pitch_text for row in records]), dtype=float
        ),
        "full_reasoning": np.asarray(
            provider.embed_documents([row.full_reasoning for row in records]), dtype=float
        ),
    }
    source_manifest = {
        "input_root": str(args.input_root),
        "vc_slug": args.vc_slug,
        "status_path": str(args.status),
        "status_sha256": _digest(args.status),
        "labels_path": str(args.labels),
        "labels_sha256": _digest(args.labels),
        "semantic_predictions_path": str(args.semantic_predictions),
        "semantic_predictions_sha256": _digest(args.semantic_predictions),
        "pitch_predictions_path": str(args.pitch_predictions),
        "pitch_predictions_sha256": _digest(args.pitch_predictions),
    }

    predictions: dict[str, list[FullFusionPrediction]] = (
        _load_resume_predictions(args.output, source_manifest) if args.resume else {}
    )
    for method in requested_methods:
        if method in predictions:
            print(f"resuming completed {method}", flush=True)
            continue
        print(f"running {method}...", flush=True)
        predictions.update(
            nested_full_fusion_predictions(records, embeddings, methods=(method,))
        )
        write_full_fusion_analysis(
            records,
            predictions,
            args.output,
            embedding_metadata=provider.metadata,
            source_manifest=source_manifest,
        )
        print(f"completed {method}; checkpoint written", flush=True)

    print("combining existing held-out Phase 1/2 and pitch component scores...", flush=True)
    predictions.update(native_phase2_score_fusions(records))
    components = _load_components(args.semantic_predictions, SEMANTIC_COMPONENTS)
    components.update(_load_components(args.pitch_predictions, PITCH_COMPONENTS))
    predictions.update(fixed_score_fusions(records, components))

    print("running two-head false-In suppression and missed-In rescue...", flush=True)
    predictions["rich_error_correction"] = nested_error_correction_predictions(
        records, embeddings
    )
    result = write_full_fusion_analysis(
        records,
        predictions,
        args.output,
        embedding_metadata=provider.metadata,
        source_manifest=source_manifest,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.output / "report.md", args.report)

    population = result["population"]
    print(
        f"evaluated {population['episodes']} frozen episodes: "
        f"{population['ins']} Ins, {population['outs']} Outs",
        flush=True,
    )
    for method, metrics in result["methods"].items():
        matrix = metrics["confusion_matrix"]
        passed = matrix["tp"] >= 7 and matrix["fp"] <= 12
        print(
            f"{method}: TP={matrix['tp']} FP={matrix['fp']} TN={matrix['tn']} "
            f"FN={matrix['fn']} BA={metrics['balanced_accuracy']:.3f} "
            f"AP={metrics['average_precision']:.3f} PASS={passed}",
            flush=True,
        )


if __name__ == "__main__":
    main()
