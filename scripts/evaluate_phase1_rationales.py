#!/usr/bin/env python3
"""Evaluate canonical Phase 1 rationale outputs against transcript references."""

from __future__ import annotations

import argparse
from pathlib import Path

from vc_clone_graph.phase1_evaluation import (
    add_semantic_diagnostics,
    evaluate_cases,
    load_phase1_cases,
    sha256_file,
    write_evaluation_outputs,
)
from vc_clone_graph.providers.sentence_transformers import SentenceTransformerEmbeddingProvider


DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
DEFAULT_REGISTRY = Path(
    "evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--references", type=Path, default=Path("evaluation/phase1_ground_truth_rationales"))
    parser.add_argument("--taxonomy", type=Path, default=Path("../agentic-vc-clone-framework/taxonomy/codebook_v_final.json"))
    parser.add_argument("--vc", action="append", dest="vcs", help="VC slug; repeat to select several")
    parser.add_argument("--top-k", action="append", type=int, dest="top_ks")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-revision", default=DEFAULT_REVISION)
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--artifact", choices=("final", "candidate"), default="final")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    cases, taxonomy = load_phase1_cases(
        project_root, args.registry, args.references, args.taxonomy, args.vcs,
        artifact=args.artifact,
    )
    top_ks = tuple(args.top_ks or (1, 3, 5, 10))
    result = evaluate_cases(cases, taxonomy, top_ks=top_ks)
    if not args.skip_semantic:
        embedder = SentenceTransformerEmbeddingProvider(
            args.embedding_model,
            revision=args.embedding_revision,
            device=args.embedding_device,
            batch_size=32,
        )
        add_semantic_diagnostics(result, cases, taxonomy, embedder)
    else:
        result["semantic_rows"] = []
        result["confusion_rows"] = []
        result["semantic_summary"] = None
    input_manifest = {
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry.resolve()),
        "reference_manifest": str((args.references / "manifest.json").resolve()),
        "reference_manifest_sha256": sha256_file((args.references / "manifest.json").resolve()),
        "taxonomy": str(args.taxonomy.resolve()),
        "taxonomy_sha256": sha256_file(args.taxonomy.resolve()),
        "phase1_artifact": args.artifact,
    }
    write_evaluation_outputs(result, cases, taxonomy, args.output, input_manifest=input_manifest)
    overall = result["overall"]
    print(
        f"Phase 1 rationale evaluation: {result['case_count']} cases, "
        f"P={overall['micro_precision']:.3f} R={overall['micro_recall']:.3f} "
        f"F1={overall['micro_f1']:.3f} Jaccard={overall['macro_jaccard']:.3f}"
    )
    for row in result["scope_rows"]:
        if row["dimension"] == "vc":
            print(
                f"  {row['value']}: n={row['n']} P={row['micro_precision']:.3f} "
                f"R={row['micro_recall']:.3f} F1={row['micro_f1']:.3f}"
            )
    print(f"Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
