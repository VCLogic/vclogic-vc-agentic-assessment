#!/usr/bin/env python3
"""Prepare or execute personalized multimodal model stages."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from typing import Sequence


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument(
        "--stage",
        choices=("prepare", "tabpfn", "setfit", "ensemble", "graph"),
        required=True,
    )
    result.add_argument("--resume", action="store_true")
    result.add_argument("--max-folds", type=int)
    result.add_argument("--fold-start", type=int, default=0)
    result.add_argument(
        "--conditions", nargs="+",
        help="Optional subset of prespecified ablation conditions.",
    )
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from vc_clone_graph.personalized_cases import load_personalized_cases
    from vc_clone_graph.personalized_config import load_personalized_config
    from vc_clone_graph.personalized_folds import make_episode_folds, write_fold_manifest
    from vc_clone_graph.personalized_matrices import ABLATIONS
    from vc_clone_graph.personalized_runner import (
        build_fold_views,
        implementation_digest,
        prepare_embeddings,
        run_checkpointed_stage,
    )
    from vc_clone_graph.personalized_tabpfn import run_tabpfn_fold
    from vc_clone_graph.phase1_evaluation import load_taxonomy

    config = load_personalized_config(args.config)
    requested_conditions = tuple(args.conditions) if args.conditions else ABLATIONS
    unknown_conditions = sorted(set(requested_conditions) - set(ABLATIONS))
    if unknown_conditions:
        raise ValueError(f"unknown ablation conditions: {', '.join(unknown_conditions)}")
    project = Path.cwd().resolve()
    registry = (project / config.data.source_registry).resolve()
    input_root = (project / config.data.input_root).resolve()
    references = (project / config.data.rationale_references).resolve()
    taxonomy_path = (project / config.data.taxonomy).resolve()
    output = (project / config.execution.output).resolve()
    cache = (project / config.execution.cache).resolve()
    cases = load_personalized_cases(
        project, registry, input_root, references, taxonomy_path
    )
    folds = make_episode_folds(
        cases,
        inner_splits=config.evaluation.inner_splits,
        seed=config.evaluation.seed,
    )
    if args.fold_start < 0 or args.fold_start >= len(folds):
        raise ValueError("fold-start must select an existing fold")
    selected_folds = folds[args.fold_start:]
    print(
        f"cases={len(cases)} ins={sum(row.target for row in cases)} "
        f"outs={len(cases)-sum(row.target for row in cases)} "
        f"episodes={len(folds)} vcs={len({row.vc_slug for row in cases})}",
        flush=True,
    )
    if args.dry_run:
        planned = len(
            selected_folds
            if args.max_folds is None
            else selected_folds[:args.max_folds]
        )
        print(f"stage={args.stage} planned_folds={planned} model_calls=0", flush=True)
        return 0

    source_registry_sha256 = sha256(registry.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    write_fold_manifest(
        output / "fold_manifest.json",
        cases,
        folds,
        source_registry_sha256=source_registry_sha256,
    )
    if args.stage == "prepare":
        print(f"prepared {len(folds)} episode folds; model_calls=0", flush=True)
        return 0
    if args.stage not in {"tabpfn", "setfit", "ensemble", "graph"}:
        raise ValueError(f"stage {args.stage} is not implemented yet")
    if args.stage == "tabpfn" and importlib.util.find_spec("tabpfn") is None:
        raise RuntimeError("TabPFN stage requires: uv sync --extra personalized")

    taxonomy = load_taxonomy(taxonomy_path)
    labels = tuple(sorted(taxonomy))
    embeddings = prepare_embeddings(
        cases, config, input_root=input_root, cache_root=cache / "embeddings"
    )
    views_by_episode = {}

    def fold_views(fold):
        if fold.held_episode not in views_by_episode:
            fold_index = next(
                index for index, candidate in enumerate(folds)
                if candidate.held_episode == fold.held_episode
            )
            views_by_episode[fold.held_episode] = build_fold_views(
                cases,
                embeddings,
                fold,
                labels,
                seed=config.evaluation.seed + fold_index * 1000,
            )
        return views_by_episode[fold.held_episode]

    if args.stage == "tabpfn":
        def execute_fold(condition, fold):
            return run_tabpfn_fold(
                cases,
                fold_views(fold),
                fold,
                condition=condition,
                n_estimators=config.tabpfn.n_estimators,
                pca_components=config.tabpfn.pca_components,
                device=config.tabpfn.device,
                seed=config.evaluation.seed,
            )
        conditions = requested_conditions
    elif args.stage == "setfit":
        from vc_clone_graph.personalized_setfit import (
            SentenceTransformerSetFitEncoder,
            run_setfit_fold,
        )

        def encoder_factory():
            return SentenceTransformerSetFitEncoder(
                config.setfit.model,
                config.setfit.revision,
                config.setfit.device,
            )

        def execute_fold(condition, fold):
            result = run_setfit_fold(
                cases,
                fold_views(fold),
                fold,
                condition=condition,
                epochs=config.setfit.epochs,
                lambda_rank=config.setfit.lambda_rank,
                lambda_aux=config.setfit.lambda_aux,
                max_pairs=config.setfit.max_pairs,
                taxonomy_labels=labels,
                seed=config.evaluation.seed,
                encoder_factory=encoder_factory,
                device=config.setfit.device,
            )
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            return result

        conditions = requested_conditions
    elif args.stage == "ensemble":
        from vc_clone_graph.personalized_ensemble import (
            component_scores_from_checkpoint,
            conditional_ensemble_fold,
        )
        from vc_clone_graph.personalized_runner import FoldRunResult

        if not config.ensemble.enabled:
            raise ValueError("ensemble stage is disabled in the configuration")
        raw_phase2 = {
            (case.vc_slug, case.episode_slug): float(
                case.phase2_features.get("investment_likelihood", 0.0)
            )
            for case in cases
        }

        def execute_fold(condition, fold):
            components = []
            for family in ("tabpfn", "setfit"):
                checkpoint = output / family / condition / f"{fold.held_episode}.json"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"missing {family} component: {checkpoint}")
                components.append(component_scores_from_checkpoint(
                    json.loads(checkpoint.read_text(encoding="utf-8")), family=family
                ))
            result = conditional_ensemble_fold(
                cases,
                fold,
                components,
                raw_phase2,
                minimum_disagreement_rate=config.ensemble.minimum_disagreement_rate,
                require_inner_improvement=config.ensemble.require_inner_improvement,
                seed=config.evaluation.seed,
            )
            return FoldRunResult(
                status=result.status,
                predictions=result.predictions,
                reason=result.reason,
            )

        if args.conditions and requested_conditions != ("all",):
            raise ValueError("ensemble stage supports only the all condition")
        conditions = ("all",)
    else:
        from vc_clone_graph.personalized_graph import (
            graph_execution_gate,
            run_graph_fold,
        )
        from vc_clone_graph.personalized_runner import FoldRunResult

        eligible, gate_reason = graph_execution_gate(
            enabled=config.graph.enabled,
            run_only_after_primary=config.graph.run_only_after_primary,
            output=output,
        )

        def execute_fold(condition, fold):
            if not eligible:
                return FoldRunResult(
                    status="skipped", predictions=(), reason=gate_reason
                )
            return run_graph_fold(
                cases, fold_views(fold), fold, condition=condition,
                taxonomy_labels=labels,
                hidden_dimensions=config.graph.hidden_dimensions,
                layers=config.graph.layers, dropout=config.graph.dropout,
                weight_decay=config.graph.weight_decay, epochs=config.graph.epochs,
                seed=config.evaluation.seed, device="cuda" if config.embedding.device == "cuda" else "cpu",
            )

        if args.conditions and requested_conditions != ("all",):
            raise ValueError("graph stage supports only the all condition")
        conditions = ("all",)

    results = run_checkpointed_stage(
        family=args.stage,
        conditions=conditions,
        folds=selected_folds,
        output=output,
        source_manifest={
            "registry_sha256": source_registry_sha256,
            "config_sha256": sha256(args.config.read_bytes()).hexdigest(),
            "embedding": dict(embeddings.embedding_metadata),
            "implementation_sha256": implementation_digest(
                (
                    *project.glob("src/vc_clone_graph/personalized_*.py"),
                    Path(__file__).resolve(),
                    project / "uv.lock",
                ),
                root=project,
            ),
        },
        run_fold=execute_fold,
        max_retries=config.execution.max_retries,
        resume=args.resume,
        max_folds=args.max_folds,
    )
    complete = sum(row["status"] == "complete" for row in results)
    skipped = sum(row["status"] == "skipped" for row in results)
    failed = sum(row["status"] == "failed" for row in results)
    print(
        f"stage={args.stage} complete={complete} skipped={skipped} failed={failed}",
        flush=True,
    )
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
