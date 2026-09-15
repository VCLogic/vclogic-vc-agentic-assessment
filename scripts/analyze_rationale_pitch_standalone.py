#!/usr/bin/env python3
"""Compare standalone rationale and audited-pitch classifiers."""

from __future__ import annotations

import argparse
from hashlib import sha256
import os
from pathlib import Path
import shutil

import numpy as np

from vc_clone_graph.providers.sentence_transformers import (
    SentenceTransformerEmbeddingProvider,
)
from vc_clone_graph.standalone_pitch_models import (
    load_verified_pitch_population,
    nested_standalone_predictions,
    write_standalone_analysis,
)


METHODS = (
    "rationale_only",
    "pitch_tfidf_only",
    "rationale_plus_pitch_tfidf",
    "pitch_embedding_only",
    "rationale_plus_pitch_embedding",
)


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_offline_environment()
    print("verifying frozen rationales and pitch-only packages...", flush=True)
    records = load_verified_pitch_population(
        args.input_root, args.status, args.labels, args.vc_slug
    )
    provider = SentenceTransformerEmbeddingProvider(
        args.embedding_model,
        revision=args.embedding_revision,
        device=args.device,
        batch_size=args.batch_size,
        normalize=True,
        document_prefix="search_document: ",
        query_prefix="search_query: ",
    )
    print("embedding verified pitch-only texts locally...", flush=True)
    embeddings = np.asarray(
        provider.embed_documents([record.pitch_text for record in records]), dtype=float
    )
    predictions = {}
    for method in METHODS:
        print(f"running {method}...", flush=True)
        predictions.update(
            nested_standalone_predictions(records, embeddings, methods=(method,))
        )
        print(f"completed {method}", flush=True)
    result = write_standalone_analysis(
        records,
        predictions,
        args.output,
        embedding_metadata=provider.metadata,
        source_manifest={
            "input_root": str(args.input_root),
            "vc_slug": args.vc_slug,
            "status_path": str(args.status),
            "status_sha256": _digest(args.status),
            "labels_path": str(args.labels),
            "labels_sha256": _digest(args.labels),
        },
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.output / "report.md", args.report)
    population = result["population"]
    print(
        f"evaluated {population['episodes']} frozen episodes: "
        f"{population['ins']} Ins, {population['outs']} Outs"
    )
    for method, metrics in result["methods"].items():
        matrix = metrics["confusion_matrix"]
        passed = matrix["tp"] >= 7 and matrix["fp"] <= 12
        print(
            f"{method}: TP={matrix['tp']} FP={matrix['fp']} TN={matrix['tn']} "
            f"FN={matrix['fn']} BA={metrics['balanced_accuracy']:.3f} "
            f"AP={metrics['average_precision']:.3f} PASS={passed}"
        )


if __name__ == "__main__":
    main()
