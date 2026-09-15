#!/usr/bin/env python3
"""Compare local post-hoc semantic models over frozen v4 artifacts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import os
from pathlib import Path
import shutil

import numpy as np

from vc_clone_graph.posthoc_semantic_models import (
    load_semantic_population,
    nested_posthoc_predictions,
    write_posthoc_analysis,
)
from vc_clone_graph.providers.sentence_transformers import (
    SentenceTransformerEmbeddingProvider,
)


def _digest(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    records = load_semantic_population(args.status, args.labels)
    provider = SentenceTransformerEmbeddingProvider(
        args.embedding_model,
        revision=args.embedding_revision,
        device=args.device,
        batch_size=args.batch_size,
        normalize=True,
        document_prefix="search_document: ",
        query_prefix="search_query: ",
    )
    embeddings = {
        "phase1_rationales": np.asarray(
            provider.embed_documents([record.phase1_text for record in records]),
            dtype=float,
        ),
        "full_reasoning": np.asarray(
            provider.embed_documents([record.full_text for record in records]),
            dtype=float,
        ),
    }
    methods = (
        "tfidf_phase1",
        "tfidf_full",
        "embedding_prototype",
        "embedding_logistic",
        "structured_elastic",
        "structured_boost",
        "late_fusion",
    )
    predictions = {}
    for method in methods:
        print(f"running {method}...", flush=True)
        predictions.update(
            nested_posthoc_predictions(
                records,
                embeddings,
                methods=(method,),
                false_positive_rate_cap=0.20,
            )
        )
        print(f"completed {method}", flush=True)
    result = write_posthoc_analysis(
        records,
        predictions,
        args.output,
        embedding_metadata=provider.metadata,
        source_manifest={
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
